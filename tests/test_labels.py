"""Gloss order and Tagalog display strings."""

from __future__ import annotations

from fsl.labels import (
    ENGLISH_KEYS,
    LEGACY_11_KEYS,
    TAGALOG_BY_KEY,
    label_to_index,
    load_labels,
    to_tagalog,
    to_tagalog_list,
)
from fsl.paths import models_dir


def test_shipped_label_file_matches_the_15_class_order() -> None:
    loaded = load_labels(models_dir() / "fsl15" / "labels.npy")
    assert loaded == ENGLISH_KEYS
    assert len(loaded) == 15


def test_legacy_label_file_matches_the_11_class_order() -> None:
    loaded = load_labels(models_dir() / "fsl11" / "labels.npy")
    assert loaded == LEGACY_11_KEYS
    assert len(loaded) == 11


def test_label_index_round_trip() -> None:
    mapping = label_to_index(ENGLISH_KEYS)
    assert mapping["THANK_YOU"] == ENGLISH_KEYS.index("THANK_YOU")
    assert [ENGLISH_KEYS[mapping[key]] for key in ENGLISH_KEYS] == list(ENGLISH_KEYS)


def test_tagalog_display_strings() -> None:
    assert to_tagalog("yes") == "Oo"
    assert to_tagalog("  HOW_ARE_YOU ") == "Kumusta ka?"
    assert to_tagalog("foo_bar") == "foo bar"
    assert to_tagalog("") == "—"
    assert to_tagalog_list(["YES", "", "NO"]) == ["Oo", "Hindi"]


def test_every_shipped_gloss_has_a_tagalog_string() -> None:
    missing = [key for key in ENGLISH_KEYS if key not in TAGALOG_BY_KEY]
    assert missing == []
