"""Canonical endpoint-pair extraction for Path Tracing."""

from __future__ import annotations

import re


def strip_thinking(text: str) -> str:
    idx = text.rfind("</think>")
    if idx == -1:
        return text
    return text[idx + len("</think>") :].lstrip()


def extract_boxed_pairs(text: str) -> tuple[list[tuple[str, str]] | None, str]:
    boxed_matches = re.findall("\\\\?boxed\\s*\\{([^{}]*)\\}", text)
    if not boxed_matches:
        return (None, "no_boxed")
    boxed_content = boxed_matches[-1].strip()
    if not boxed_content:
        return (None, "malformed")
    pair_pattern = re.compile("\\(\\s*([^,()]+?)\\s*,\\s*([^,()]+?)\\s*\\)")
    pairs = []
    for m in pair_pattern.finditer(boxed_content):
        a = m.group(1).strip()
        b = m.group(2).strip()
        pairs.append((a, b))
    if not pairs:
        return (None, "malformed")
    return (pairs, "success")


def _normalize_label(label: str) -> str | int:
    label = label.strip()
    label = label.replace("\\", "")
    try:
        return int(label)
    except ValueError:
        pass
    return label.lower()


def normalize_pairs(pairs: list[tuple[str, str]]) -> frozenset[frozenset]:
    return frozenset((frozenset({_normalize_label(a), _normalize_label(b)}) for a, b in pairs))
