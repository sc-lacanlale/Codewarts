"""Small helpers for exhaustive matching."""

from __future__ import annotations

from typing import Never


def assert_never(value: Never) -> Never:
    raise AssertionError(f"Unhandled value: {value!r}")
