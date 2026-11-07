"""Lossless token/gap targets for a learned document-edit locator.

This is data construction, not a trained locator. It replaces generation of
numeric character offsets with two supervised decisions: which source tokens
to keep and which source-token gaps need infilling. Gold labels are derived
from old/new text only during training; inference must predict both decisions.
"""

import difflib
import re


_TOKEN = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def source_tokens(text: str) -> list[str]:
    """Split text without normalizing or dropping punctuation/whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens = _TOKEN.findall(text)
    if "".join(tokens) != text:
        raise AssertionError("tokenizer did not preserve source text")
    return tokens


def gold_token_edit(source: str, target: str) -> dict:
    """Return source-token KEEP labels and target strings for token gaps."""
    left, right = source_tokens(source), source_tokens(target)
    keep = [True] * len(left)
    insert = [""] * (len(left) + 1)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=left, b=right, autojunk=False
    ).get_opcodes():
        if op == "equal":
            continue
        if op in {"replace", "delete"}:
            keep[i1:i2] = [False] * (i2 - i1)
        if op in {"replace", "insert"}:
            insert[i1] += "".join(right[j1:j2])
    result = {"tokens": left, "keep": keep, "insert": insert}
    if apply_token_edit(**result) != target:
        raise AssertionError("alignment did not reconstruct target")
    return result


def apply_token_edit(tokens: list[str], keep: list[bool], insert: list[str]) -> str:
    """Copy each kept source token exactly and infill only selected gaps."""
    if len(tokens) != len(keep) or len(insert) != len(tokens) + 1:
        raise ValueError("token/edit lengths disagree")
    if any(type(x) is not bool for x in keep) or any(
        not isinstance(x, str) for x in insert
    ):
        raise TypeError("invalid edit labels")
    parts = []
    for i, token in enumerate(tokens):
        parts.append(insert[i])
        if keep[i]:
            parts.append(token)
    parts.append(insert[-1])
    return "".join(parts)
