"""Speak a Tagalog line. edge-tts is preferred; gTTS and pyttsx3 are fallbacks."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal

try:
    import edge_tts
except ImportError:  # pragma: no cover - optional app dependency
    edge_tts = None

try:
    from gtts import gTTS
except ImportError:  # pragma: no cover - optional app dependency
    gTTS = None

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional app dependency
    pyttsx3 = None

try:
    import pythoncom
except ImportError:  # pragma: no cover - Windows-only COM helper
    pythoncom = None

try:
    from playsound import playsound
except ImportError:  # pragma: no cover - optional playback
    playsound = None

try:
    import pygame
except ImportError:  # pragma: no cover - optional playback
    pygame = None

from fsl.checks import assert_never

DEFAULT_VOICE = "fil-PH-BlessicaNeural"
FALLBACK_VOICE = "fil-PH-AngeloNeural"
TTSBackend = Literal["edge", "gtts", "pyttsx3"]


def prepare_text_for_tts(text: str) -> str:
    """Pad a one-word gloss such as ``Oo`` so the neural voice does not drop it."""
    stripped = text.strip()
    core = stripped.rstrip(".!?")
    if len(core) <= 3 and " " not in core:
        return f"{core} po."
    return stripped


class TagalogTTS:
    """Speak Tagalog with edge-tts, then gTTS, then pyttsx3."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._voice = os.getenv("TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
        self._backend: TTSBackend | None = None
        self._available = False
        self._init_backend()

    def _init_backend(self) -> None:
        if edge_tts is not None:
            self._backend = "edge"
            self._available = True
            print(f"TTS ready (edge-tts, voice={self._voice})")
            return
        if gTTS is not None:
            self._backend = "gtts"
            self._available = True
            print("TTS ready (gTTS fallback)")
            return
        if pyttsx3 is not None:
            self._backend = "pyttsx3"
            self._available = True
            print("TTS ready (pyttsx3 fallback)")
            return
        print("TTS packages are not installed. Text will not be spoken.")

    @property
    def available(self) -> bool:
        return self._available

    def speak(self, text: str) -> None:
        if not self._available or self._backend is None or not text.strip():
            return
        spoken = prepare_text_for_tts(text)
        with self._lock:
            try:
                self._speak_with(self._backend, spoken)
            except Exception as exc:
                print(f"TTS error ({self._backend}): {exc}")
                if self._backend == "edge":
                    self._try_fallback_chain(spoken, skip="edge")

    def _speak_with(self, backend: TTSBackend, text: str) -> None:
        match backend:
            case "edge":
                self._speak_edge(text)
            case "gtts":
                self._speak_gtts(text)
            case "pyttsx3":
                self._speak_pyttsx3(text)
            case unreachable:
                assert_never(unreachable)

    def _try_fallback_chain(self, text: str, skip: TTSBackend) -> None:
        for backend in ("gtts", "pyttsx3"):
            if backend == skip:
                continue
            try:
                self._speak_with(backend, text)
                print(f"TTS recovered via {backend}")
                return
            except Exception as exc:
                print(f"TTS fallback {backend} failed: {exc}")

    def _speak_edge(self, text: str) -> None:
        path = self._run_async(self._edge_to_file(text))
        try:
            self._play_mp3(path)
        finally:
            Path(path).unlink(missing_ok=True)

    async def _edge_to_file(self, text: str) -> str:
        if edge_tts is None:
            raise ImportError("edge-tts is not installed")
        descriptor, path = tempfile.mkstemp(suffix=".mp3")
        os.close(descriptor)
        for voice in (self._voice, FALLBACK_VOICE):
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(path)
                if Path(path).stat().st_size > 800:
                    return path
            except Exception:
                if voice == FALLBACK_VOICE:
                    raise
        return path

    def _speak_gtts(self, text: str) -> None:
        if gTTS is None:
            raise ImportError("gTTS is not installed")
        descriptor, path = tempfile.mkstemp(suffix=".mp3")
        os.close(descriptor)
        try:
            gTTS(text=text, lang="tl").save(path)
            self._play_mp3(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def _speak_pyttsx3(self, text: str) -> None:
        if pyttsx3 is None:
            raise ImportError("pyttsx3 is not installed")
        com_init = False
        if pythoncom is not None:
            pythoncom.CoInitialize()
            com_init = True
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.say(text)
            engine.runAndWait()
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if com_init and pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    @staticmethod
    def _run_async(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @staticmethod
    def _play_mp3(path: str) -> None:
        if playsound is not None:
            try:
                playsound(path, block=True)
                return
            except Exception:
                pass
        if pygame is None:
            raise RuntimeError("No audio player is installed. pip install pygame")
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
