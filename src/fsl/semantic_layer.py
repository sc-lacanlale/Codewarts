"""Turn a sequence of glosses into one Tagalog line. Gemini is optional."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional app dependency
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional app dependency
    genai = None

from fsl.labels import to_tagalog_list
from fsl.paths import repo_root

SYSTEM_PROMPT = """Ikaw ay isang katulong sa pagsasalin ng Filipino Sign Language (FSL).
Tatanggap ka ng sunod-sunod na mga senyas mula sa live recognition system.
Pag-isahin mo ang mga ito sa isang maikli at natural na pangungusap o parirala sa TAGALOG.

Mga alituntunin:
- Gamitin lamang ang ibinigay na mga senyas; huwag mag-imbento.
- Ayusin ang gramatika at ayos ng salita sa natural na Tagalog.
- Panatilihing maikli ang sagot (isa o dalawang pangungusap lang).
- Ibabalik mo LAMANG ang isinaling teksto sa Tagalog — walang quotes, paliwanag, o Ingles.
- Kung hindi magkaugnay ang mga senyas, pagdugtungin nang malinaw gamit ang tamang bantas.
- Ang mga senyas ay ibinibigay sa Tagalog (hal. Salamat, Kumusta ka?, Okay lang ako)."""

LabelFn = Callable[[list[str]], list[str]]


def join_tagalog(words: list[str]) -> str:
    """Join display strings into one line and close it with punctuation."""
    if not words:
        return ""
    sentence = " ".join(words)
    if sentence[-1] not in ".?!":
        sentence += "."
    return sentence[0].upper() + sentence[1:]


class SemanticInterpreter:
    """Call Gemini when ``GEMINI_API_KEY`` is set. Otherwise join the Tagalog glosses."""

    def __init__(self, label_fn: LabelFn | None = None) -> None:
        self._label_fn = label_fn or to_tagalog_list
        self._lock = threading.Lock()
        self._model = None
        self._available = False
        self._init_client()

    def _init_client(self) -> None:
        if load_dotenv is not None:
            env_path = repo_root() / ".env"
            if env_path.is_file():
                load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = os.getenv("TANAW_GEMINI_MODEL", "gemini-2.5-flash").strip()
        if not api_key:
            print("GEMINI_API_KEY is unset. Gloss labels are joined without Gemini.")
            return
        if genai is None:
            print("google-generativeai is not installed. Gloss labels are joined without Gemini.")
            return
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name=model_name, system_instruction=SYSTEM_PROMPT
            )
            self._available = True
            print(f"Semantic layer ready ({model_name})")
        except Exception as exc:
            print(f"Semantic layer init failed: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def fallback(self, signs: list[str]) -> str:
        """Join the display labels. Does not call Gemini."""
        return join_tagalog(self._label_fn(signs))

    def interpret(self, signs: list[str]) -> str:
        if not signs:
            return ""
        if not self._available or self._model is None:
            return self.fallback(signs)

        labels_text = ", ".join(self._label_fn(signs))
        prompt = f"Natukoy na mga senyas (sunod-sunod): {labels_text}"
        with self._lock:
            try:
                response = self._model.generate_content(prompt)
                text = (response.text or "").strip()
                return text if text else self.fallback(signs)
            except Exception as exc:
                print(f"Gemini error: {exc}")
                return self.fallback(signs)
