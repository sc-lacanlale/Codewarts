"""Scripted demo UI. Each Start click plays the next canned sign sequence."""

from __future__ import annotations

import tkinter as tk

import cv2

from apps.desktop import FSLTranslatorApp, crop_to_aspect_ratio, letterbox_frame
from fsl.labels import to_tagalog
from fsl.semantic_layer import SemanticInterpreter

DEMO_SCENARIOS: list[list[str]] = [
    ["HELLO"],
    ["GOOD_AFTERNOON", "NICE_TO_MEET_YOU"],
    ["YES"],
    ["NO"],
    ["ONE", "TWO", "THREE"],
    ["SIX", "SEVEN"],
]

DEMO_TAGALOG_OVERRIDES = {
    "NICE_TO_MEET_YOU": "Ikinagagalak ko kayo makilala",
}

DEMO_LABEL_DELAY_MS = 5000


def demo_signs_to_tagalog(keys: list[str]) -> list[str]:
    return [DEMO_TAGALOG_OVERRIDES.get(key, to_tagalog(key)) for key in keys if key]


class FSLTranslatorDemoApp(FSLTranslatorApp):
    def __init__(self, root: tk.Tk) -> None:
        self._demo_index = 0
        self._demo_timers: list[str] = []
        super().__init__(root)
        self.root.title("FSL translator (demo)")
        self.translate_btn.config(text="Start translating")

    def _make_semantic(self) -> SemanticInterpreter:
        return SemanticInterpreter(label_fn=demo_signs_to_tagalog)

    def load_model(self) -> None:
        print("Demo mode: LSTM weights are not loaded")

    def _cancel_demo_timers(self) -> None:
        for timer_id in self._demo_timers:
            try:
                self.root.after_cancel(timer_id)
            except tk.TclError:
                pass
        self._demo_timers.clear()
        if self.gemini_timer:
            self.root.after_cancel(self.gemini_timer)
            self.gemini_timer = None

    def _schedule_demo_timer(self, delay_ms: int, callback) -> str:
        timer_id = self.root.after(delay_ms, callback)
        self._demo_timers.append(timer_id)
        return timer_id

    def _play_demo_scenario(self, signs: list[str]) -> None:
        """Reveal each sign 5 seconds apart. The sentence is spoken after the last one."""
        self._cancel_demo_timers()
        self.detected_signs = []
        self.label_display.config(text="—")
        self._update_transcript_display("Nakikinig")

        def reveal_sign(index: int) -> None:
            if not self.is_translating or index >= len(signs):
                return
            sign = signs[index]
            with self.state_lock:
                self.detected_signs.append(sign)
            tagalog = demo_signs_to_tagalog([sign])[0]
            self.label_display.config(text=tagalog)
            print(f"Demo recognized ({index + 1}/{len(signs)}): {tagalog}")
            if index + 1 < len(signs):
                self._schedule_demo_timer(
                    DEMO_LABEL_DELAY_MS, lambda next_index=index + 1: reveal_sign(next_index)
                )
                return
            signs_copy = list(self.detected_signs)
            interim = self.semantic.fallback(signs_copy)
            self._update_transcript_display(interim)
            print(f"Demo state complete: {signs_copy} -> {interim}")
            self._speak_demo_transcript(interim)
            self._schedule_semantic_update()

        self._schedule_demo_timer(DEMO_LABEL_DELAY_MS, lambda: reveal_sign(0))

    def _speak_demo_transcript(self, text: str) -> None:
        if not text or not self.tts_available:
            return
        self.last_spoken = text
        self.speak_text(text)

    def _schedule_semantic_update(self) -> None:
        if self.gemini_timer:
            self.root.after_cancel(self.gemini_timer)
            self.gemini_timer = None
        self._run_semantic_update()

    def _apply_transcript(self, text: str) -> None:
        """Gemini may replace the line. Speech already used the joined glosses."""
        if not text:
            return
        self.transcript = text
        self._update_transcript_display(text)

    def toggle_translating(self) -> None:
        if not self.is_translating:
            signs = list(DEMO_SCENARIOS[self._demo_index])
            self._demo_index = (self._demo_index + 1) % len(DEMO_SCENARIOS)
            self.is_translating = True
            self.transcript = ""
            self.last_spoken = ""
            self.translate_btn.config(text="Stop translating", bg="#9b2226")
            self._play_demo_scenario(signs)
            return
        self._cancel_demo_timers()
        self.is_translating = False
        self.translate_btn.config(text="Start translating", bg="#2d6a4f")

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
            if self.is_recording_video:
                self._write_video_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            container_w, container_h = self._container_size
            with self.frame_lock:
                self.current_frame = letterbox_frame(rgb, container_w, container_h)
                self.frame_version += 1

    def on_closing(self) -> None:
        self._cancel_demo_timers()
        super().on_closing()


def main() -> None:
    root = tk.Tk()
    app = FSLTranslatorDemoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
