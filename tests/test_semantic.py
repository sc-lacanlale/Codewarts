"""Tagalog joining when Gemini is not used."""

from __future__ import annotations

from fsl.semantic_layer import SemanticInterpreter, join_tagalog
from fsl.tts import prepare_text_for_tts


def test_join_tagalog_adds_terminal_punctuation() -> None:
    assert join_tagalog(["Salamat", "Oo"]) == "Salamat Oo."
    assert join_tagalog(["Kumusta ka?"]) == "Kumusta ka?"
    assert join_tagalog([]) == ""


def test_interpret_falls_back_without_calling_gemini(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    interpreter = SemanticInterpreter()
    interpreter._available = False
    interpreter._model = None
    assert interpreter.interpret(["YES", "THANK_YOU"]) == "Oo Salamat."
    assert interpreter.fallback(["HOW_ARE_YOU"]) == "Kumusta ka?"
    assert interpreter.interpret([]) == ""


def test_short_tagalog_is_padded_for_speech() -> None:
    assert prepare_text_for_tts("Oo.") == "Oo po."
    assert prepare_text_for_tts("  Hi! ") == "Hi po."
    assert prepare_text_for_tts("Salamat.") == "Salamat."
    assert prepare_text_for_tts("A B") == "A B"
