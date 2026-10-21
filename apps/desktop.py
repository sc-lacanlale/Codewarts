"""Tk webcam app: landmarks, 15-class LSTM, Tagalog text, optional TTS."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from fsl.labels import to_tagalog
from fsl.landmarks import KEYPOINT_DIM, HolisticTracker
from fsl.model import FSLBundle, load_bundle, predict_label
from fsl.semantic_layer import SemanticInterpreter
from fsl.sequence import (
    PREDICTION_CONFIDENCE_THRESHOLD,
    SEQUENCE_LENGTH,
    StableSignDetector,
    append_sign,
)
from fsl.tts import TagalogTTS

ASPECT_RATIO = 16 / 9
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
TRACKER_PROCESS_WIDTH = 480
DISPLAY_INTERVAL_MS = 16
INFERENCE_EVERY_N_FRAMES = 2
UI_LABEL_MIN_INTERVAL_MS = 80
DROP_STALE_CAMERA_FRAMES = 2
GEMINI_DEBOUNCE_SEC = 0.8

BG = "#0f0f0f"
BAR_BG = "#1a1a1a"
TEXT = "#e8e8e8"
MUTED = "#888888"
ACCENT = "#4a9eff"
TRANSCRIPT_FG = "#7dcea0"


def crop_to_aspect_ratio(frame: np.ndarray, aspect_ratio: float = ASPECT_RATIO) -> np.ndarray:
    """Center-crop a frame to the target aspect ratio."""
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame
    current_ratio = width / height
    if abs(current_ratio - aspect_ratio) < 0.01:
        return frame
    if current_ratio > aspect_ratio:
        new_width = int(height * aspect_ratio)
        x0 = (width - new_width) // 2
        return frame[:, x0 : x0 + new_width]
    new_height = int(width / aspect_ratio)
    y0 = (height - new_height) // 2
    return frame[y0 : y0 + new_height, :]


def fit_aspect_size(
    container_w: int, container_h: int, aspect_ratio: float = ASPECT_RATIO
) -> tuple[int, int]:
    """Largest width and height that fit in the container at the given aspect ratio."""
    if container_w <= 0 or container_h <= 0:
        return 640, 360
    if container_w / container_h > aspect_ratio:
        height = container_h
        width = int(height * aspect_ratio)
    else:
        width = container_w
        height = int(width / aspect_ratio)
    return max(width, 1), max(height, 1)


def letterbox_frame(frame: np.ndarray, container_w: int, container_h: int) -> np.ndarray:
    """Scale a frame to 16:9 inside the container and pad the rest with black."""
    render_w, render_h = fit_aspect_size(container_w, container_h)
    scaled = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
    if render_w == container_w and render_h == container_h:
        return scaled
    canvas = np.zeros((container_h, container_w, 3), dtype=np.uint8)
    x0 = (container_w - render_w) // 2
    y0 = (container_h - render_h) // 2
    canvas[y0 : y0 + render_h, x0 : x0 + render_w] = scaled
    return canvas


class FSLTranslatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FSL translator")
        self.root.geometry("1100x780")
        self.root.configure(bg=BG)
        self.root.minsize(900, 640)

        self.is_translating = False
        self.show_bones = True
        self.is_recording_video = False
        self.video_writer: cv2.VideoWriter | None = None
        self.record_path: str | None = None

        self.cap: cv2.VideoCapture | None = None
        self.tracker: HolisticTracker | None = None
        self.bundle: FSLBundle | None = None
        self.semantic = self._make_semantic()

        self.detected_signs: list[str] = []
        self.transcript = ""
        self.last_spoken = ""
        self.confidence_threshold = PREDICTION_CONFIDENCE_THRESHOLD
        self.sign_detector = StableSignDetector()
        self.sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.video_running = False
        self.current_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()
        self.frame_version = 0
        self._last_shown_version = -1
        self._container_size = (640, 360)
        self._frame_counter = 0
        self._last_live_sign = ""
        self._last_live_conf = 0.0
        self._last_label_update_ms = 0
        self.state_lock = threading.Lock()
        self.gemini_timer: str | None = None
        self.live_sign = ""
        self.live_confidence = 0.0

        self.tts = TagalogTTS()
        self.tts_available = self.tts.available

        self.setup_ui()
        self.load_model()
        self.root.after(100, self.start_camera)
        self.root.after(500, self._refresh_display_size)
        self.update_frame()

    def _make_semantic(self) -> SemanticInterpreter:
        return SemanticInterpreter()

    def setup_ui(self) -> None:
        self.camera_frame = tk.Frame(self.root, bg="#000000")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(16, 8))
        self.video_label = tk.Label(self.camera_frame, bg="#000000", bd=0)
        self.video_label.pack(expand=True)

        self.bottom_bar = tk.Frame(self.root, bg=BAR_BG, height=88)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.bottom_bar.pack_propagate(False)
        self.bottom_bar.grid_columnconfigure(0, weight=1, minsize=160)
        self.bottom_bar.grid_columnconfigure(1, weight=4, minsize=280)
        self.bottom_bar.grid_columnconfigure(2, weight=1, minsize=220)

        label_panel = tk.Frame(self.bottom_bar, bg=BAR_BG)
        label_panel.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=10)
        tk.Label(label_panel, text="Sign", font=("Segoe UI", 9), bg=BAR_BG, fg=MUTED).pack(
            anchor="w"
        )
        self.label_display = tk.Label(
            label_panel,
            text="—",
            font=("Segoe UI", 16, "bold"),
            bg=BAR_BG,
            fg=ACCENT,
            wraplength=140,
            justify=tk.LEFT,
        )
        self.label_display.pack(anchor="w", fill=tk.X)

        transcript_panel = tk.Frame(self.bottom_bar, bg=BAR_BG)
        transcript_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=10)
        tk.Label(
            transcript_panel, text="Translation", font=("Segoe UI", 9), bg=BAR_BG, fg=MUTED
        ).pack(anchor="w")
        self.transcript_display = tk.Label(
            transcript_panel,
            text="Start translating to begin",
            font=("Segoe UI", 14),
            bg=BAR_BG,
            fg=TRANSCRIPT_FG,
            wraplength=520,
            justify=tk.LEFT,
            anchor="w",
        )
        self.transcript_display.pack(anchor="w", fill=tk.BOTH, expand=True)

        btn_panel = tk.Frame(self.bottom_bar, bg=BAR_BG)
        btn_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 12), pady=10)
        btn_style = dict(
            font=("Segoe UI", 10), bd=0, padx=10, pady=8, cursor="hand2", relief=tk.FLAT
        )
        self.translate_btn = tk.Button(
            btn_panel,
            text="Start translating",
            command=self.toggle_translating,
            bg="#2d6a4f",
            fg=TEXT,
            activebackground="#40916c",
            activeforeground=TEXT,
            **btn_style,
        )
        self.translate_btn.pack(fill=tk.X, pady=(0, 4))
        self.record_btn = tk.Button(
            btn_panel,
            text="Record video",
            command=self.toggle_video_recording,
            bg="#3d3d3d",
            fg=TEXT,
            activebackground="#555555",
            activeforeground=TEXT,
            **btn_style,
        )
        self.record_btn.pack(fill=tk.X, pady=4)
        self.bones_btn = tk.Button(
            btn_panel,
            text="Turn off bones",
            command=self.toggle_bones,
            bg="#3d3d3d",
            fg=TEXT,
            activebackground="#555555",
            activeforeground=TEXT,
            **btn_style,
        )
        self.bones_btn.pack(fill=tk.X, pady=(4, 0))

    def load_model(self) -> None:
        try:
            self.bundle = load_bundle("fsl15")
            print(f"Loaded {len(self.bundle.labels)}-sign LSTM from {self.bundle.path.name}")
            self._warmup_model()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load model:\n{exc}")

    def _warmup_model(self) -> None:
        if self.bundle is None:
            return
        dummy = np.zeros((SEQUENCE_LENGTH, KEYPOINT_DIM), dtype=np.float32)
        predict_label(self.bundle, dummy)
        print("Model warmup complete")

    def _refresh_display_size(self) -> None:
        try:
            width = max(self.camera_frame.winfo_width(), 640)
            height = max(self.camera_frame.winfo_height(), 360)
            self._container_size = (width, height)
        except tk.TclError:
            pass
        self.root.after(500, self._refresh_display_size)

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _read_latest_frame(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None or not self.cap.isOpened():
            return False, None
        for _ in range(DROP_STALE_CAMERA_FRAMES):
            if not self.cap.grab():
                break
        return self.cap.retrieve()

    def start_camera(self) -> None:
        if self.video_running:
            return
        try:
            self.cap = self._open_camera()
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            ok, _frame = self._read_latest_frame()
            if not ok:
                raise RuntimeError("Could not read from camera")
            self.tracker = HolisticTracker(process_width=TRACKER_PROCESS_WIDTH)
            self.video_running = True
            threading.Thread(target=self.process_video, daemon=True).start()
        except Exception as exc:
            messagebox.showerror("Camera error", str(exc))

    def toggle_translating(self) -> None:
        if not self.is_translating:
            self.is_translating = True
            self.detected_signs = []
            self.transcript = ""
            self.last_spoken = ""
            self.sign_detector.reset()
            self.sequence.clear()
            self.live_sign = ""
            self.live_confidence = 0.0
            self.translate_btn.config(text="Stop translating", bg="#9b2226")
            self._update_transcript_display("Sign now. The translation updates as signs hold.")
            self.label_display.config(text="—")
            return
        self.is_translating = False
        self.translate_btn.config(text="Start translating", bg="#2d6a4f")

    def toggle_bones(self) -> None:
        self.show_bones = not self.show_bones
        self.bones_btn.config(text="Turn on bones" if not self.show_bones else "Turn off bones")

    def toggle_video_recording(self) -> None:
        if not self.is_recording_video:
            path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")],
                initialfile=f"fsl_recording_{datetime.now():%Y%m%d_%H%M%S}.mp4",
            )
            if not path:
                return
            self.record_path = path
            self.is_recording_video = True
            self.record_btn.config(text="Stop recording", bg="#9b2226")
            return
        self.is_recording_video = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.record_btn.config(text="Record video", bg="#3d3d3d")
        if self.record_path:
            messagebox.showinfo("Saved", f"Video saved to:\n{self.record_path}")
            self.record_path = None

    def _classify_sequence(self, sequence: deque[np.ndarray]) -> tuple[str | None, float]:
        if self.bundle is None or len(sequence) < SEQUENCE_LENGTH:
            return None, 0.0
        label, confidence = predict_label(self.bundle, sequence)
        return label, confidence

    def _on_sign_recorded(self, sign: str, confidence: float) -> None:
        with self.state_lock:
            if not append_sign(self.detected_signs, sign):
                return
            signs_copy = list(self.detected_signs)
        display = to_tagalog(sign)
        interim = self.semantic.fallback(signs_copy)
        self.root.after(0, lambda: self.label_display.config(text=display))
        self.root.after(0, lambda text=interim: self._update_transcript_display(text))
        print(f"Recorded: {display} ({confidence:.0%}) -> {interim}")
        self._schedule_semantic_update()

    def _schedule_semantic_update(self) -> None:
        if self.gemini_timer:
            self.root.after_cancel(self.gemini_timer)
        self.gemini_timer = self.root.after(
            int(GEMINI_DEBOUNCE_SEC * 1000), self._run_semantic_update
        )

    def _run_semantic_update(self) -> None:
        with self.state_lock:
            signs = list(self.detected_signs)
        if not signs:
            return

        def worker() -> None:
            text = self.semantic.interpret(signs)
            self.root.after(0, lambda: self._apply_transcript(text))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_transcript(self, text: str) -> None:
        if not text:
            return
        self.transcript = text
        self._update_transcript_display(text)
        if text != self.last_spoken and self.tts_available:
            self.last_spoken = text
            self.speak_text(text)

    def _update_transcript_display(self, text: str) -> None:
        self.transcript_display.config(text=text or "…")

    def process_video(self) -> None:
        while self.video_running and self.cap is not None and self.cap.isOpened():
            ok, frame = self._read_latest_frame()
            if not ok or frame is None or self.tracker is None:
                break
            frame = crop_to_aspect_ratio(frame)
            output = self.tracker.process(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.show_bones:
                self.tracker.draw_landmarks(rgb, output, inplace=True)
            if output.keypoints is not None:
                self.sequence.append(output.keypoints)

            self._frame_counter += 1
            window_ready = len(self.sequence) == SEQUENCE_LENGTH
            if window_ready and self._frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
                sign, confidence = self._classify_sequence(self.sequence)
                self.live_sign = sign or ""
                self.live_confidence = confidence
            else:
                sign, confidence = self.live_sign, self.live_confidence

            if self.is_translating and len(self.sequence) == SEQUENCE_LENGTH:
                now_ms = int(time.monotonic() * 1000)
                sign_changed = sign != self._last_live_sign
                conf_changed = abs(confidence - self._last_live_conf) >= 0.05
                label_due = (now_ms - self._last_label_update_ms) >= UI_LABEL_MIN_INTERVAL_MS
                if sign_changed or conf_changed or label_due:
                    self._last_live_sign = sign or ""
                    self._last_live_conf = confidence
                    self._last_label_update_ms = now_ms
                    live_text = to_tagalog(sign) if sign else "—"
                    self.root.after(0, lambda text=live_text: self.label_display.config(text=text))
                recorded = self.sign_detector.update(sign, confidence, self.confidence_threshold)
                if recorded:
                    self._on_sign_recorded(recorded, confidence)

            if self.is_recording_video:
                self._write_video_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            container_w, container_h = self._container_size
            display_rgb = letterbox_frame(rgb, container_w, container_h)
            with self.frame_lock:
                self.current_frame = display_rgb
                self.frame_version += 1

    def _write_video_frame(self, bgr_frame: np.ndarray) -> None:
        if self.record_path is None:
            return
        height, width = bgr_frame.shape[:2]
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.record_path, fourcc, 20.0, (width, height))
        self.video_writer.write(bgr_frame)

    def update_frame(self) -> None:
        try:
            with self.frame_lock:
                if self.current_frame is None or self.frame_version == self._last_shown_version:
                    self.root.after(DISPLAY_INTERVAL_MS, self.update_frame)
                    return
                frame = self.current_frame
                self._last_shown_version = self.frame_version
            image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.imgtk = image
            self.video_label.configure(image=image)
        except Exception as exc:
            print(f"Frame update error: {exc}")
        self.root.after(DISPLAY_INTERVAL_MS, self.update_frame)

    def speak_text(self, text: str) -> None:
        if not self.tts_available or not text.strip():
            return

        def worker() -> None:
            self.tts.speak(text)

        threading.Thread(target=worker, daemon=True).start()

    def on_closing(self) -> None:
        self.video_running = False
        if self.video_writer is not None:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
        if self.tracker is not None:
            self.tracker.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = FSLTranslatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
