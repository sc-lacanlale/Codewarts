"""Gloss keys for the shipped models, and Tagalog display strings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Order matches models/fsl15/labels.npy.
ENGLISH_KEYS: tuple[str, ...] = (
    "GOOD_AFTERNOON",
    "NICE_TO_MEET_YOU",
    "YES",
    "NO",
    "THANK_YOU",
    "HOW_ARE_YOU",
    "ONE",
    "TWO",
    "THREE",
    "SIX",
    "SEVEN",
    "CORRECT",
    "WRONG",
    "IM_FINE",
    "UNDERSTAND",
)

# Order matches models/fsl11/labels.npy. The desktop app does not load this set.
LEGACY_11_KEYS: tuple[str, ...] = (
    "GOOD_EVENING",
    "NICE_TO_MEET_YOU",
    "HELLO",
    "HOW_ARE_YOU",
    "THANK_YOU",
    "YES",
    "LONGANISA",
    "TODAY",
    "SUNDAY",
    "YOURE_WELCOME",
    "IM_FINE",
)

TAGALOG_BY_KEY: dict[str, str] = {
    "HELLO": "Hello",
    "GOOD_AFTERNOON": "Magandang hapon",
    "NICE_TO_MEET_YOU": "Ikinagagalak kitang makilala",
    "YES": "Oo",
    "NO": "Hindi",
    "THANK_YOU": "Salamat",
    "HOW_ARE_YOU": "Kumusta ka?",
    "ONE": "Isa",
    "TWO": "Dalawa",
    "THREE": "Tatlo",
    "SIX": "Anim",
    "SEVEN": "Pito",
    "CORRECT": "Tama",
    "WRONG": "Mali",
    "IM_FINE": "Okay lang ako",
    "UNDERSTAND": "Naiintindihan ko",
}


def load_labels(path: Path) -> tuple[str, ...]:
    """Load a ``.npy`` vector of gloss strings, in model-output order."""
    array = np.load(path, allow_pickle=True)
    return tuple(str(item) for item in array.tolist())


def label_to_index(labels: tuple[str, ...] | list[str]) -> dict[str, int]:
    """Map each gloss to its softmax index. Later duplicates overwrite earlier ones."""
    return {str(label): index for index, label in enumerate(labels)}


def to_tagalog(key: str) -> str:
    """Map an internal gloss to the Tagalog string shown in the UI."""
    if not key:
        return "—"
    normalized = key.strip().upper()
    return TAGALOG_BY_KEY.get(normalized, key.replace("_", " "))


def to_tagalog_list(keys: list[str]) -> list[str]:
    return [to_tagalog(key) for key in keys if key]
