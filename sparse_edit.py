"""One-window, exact-copy edit scripts for versioned documentation.

Gold scripts are derived only for supervised targets and diagnostics. At
inference the model must predict both the old anchor and its replacement.
"""
import json


def apply_script(old_span, script):
    """Apply a predicted edit only when its nonempty old anchor is unique."""
    if not isinstance(script, dict) or set(script) != {"old", "new"}:
        raise ValueError("script must contain exactly the old and new string fields")
    before, after = script["old"], script["new"]
    if not isinstance(before, str) or not isinstance(after, str) or not before:
        raise ValueError("old must be nonempty and both fields must be strings")
    if before == after:
        raise ValueError("script makes no change")
    if old_span.count(before) != 1:
        raise ValueError("old anchor must occur exactly once in the marked span")
    return old_span.replace(before, after, 1)


def parse_script(text):
    """Strict JSON decoding exposes format failures instead of repairing them."""
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid JSON edit script") from exc
    if not isinstance(value, dict):
        raise ValueError("edit script must be a JSON object")
    return value


def gold_script(old, new):
    """Derive the smallest single contiguous word-aligned replacement.

    Multiple distant changes make a larger window. This is intentionally a
    one-window pilot, not a claim that every document edit is sparse.
    """
    if not old or not new or old == new:
        raise ValueError("gold edit requires different, nonempty spans")
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    suffix = 0
    while (suffix < min(len(old) - p, len(new) - p)
           and old[len(old) - suffix - 1] == new[len(new) - suffix - 1]):
        suffix += 1
    start, end = p, len(old) - suffix
    target_start, target_end = p, len(new) - suffix

    # Avoid anchors that start or end in the middle of a word.
    while start > 0 and not old[start - 1].isspace():
        start -= 1
        target_start -= 1
    while end < len(old) and not old[end].isspace():
        end += 1
        target_end += 1

    # A pure insertion needs a nonempty anchor; prefer the preceding word.
    if start == end:
        if start:
            start -= 1
            target_start -= 1
            while start > 0 and not old[start - 1].isspace():
                start -= 1
                target_start -= 1
        else:
            end += 1
            target_end += 1
            while end < len(old) and not old[end].isspace():
                end += 1
                target_end += 1

    # Make the source anchor unique. Expansions are copied into the target.
    while old.count(old[start:end]) != 1:
        if start:
            start -= 1
            target_start -= 1
        elif end < len(old):
            end += 1
            target_end += 1
        else:
            raise AssertionError("entire old span cannot be ambiguous")

    script = {"old": old[start:end], "new": new[target_start:target_end]}
    if apply_script(old, script) != new:
        raise AssertionError("gold script must reconstruct target exactly")
    return script


def script_text(script):
    return json.dumps(script, ensure_ascii=False, separators=(",", ":"))


def edit_prompt(row):
    if row.get("prompt_version") != "v2_fact_card" or not row.get("change_spec"):
        raise ValueError("sparse editing requires a complete v2 fact card")
    facts = "\n".join("- " + x for x in row["change_spec"])
    protected = "\n".join("- " + x for x in row["protected_facts"])
    return (
        "Revise the marked technical-document paragraph for the target version. "
        "Return one JSON object with exactly two string fields, old and new. "
        "The old field must be an exact substring occurring once in the marked "
        "old paragraph. Replacing it with new must produce the revised paragraph. "
        "Copy all unchanged text exactly; make the old substring as short as "
        "possible while covering every required change. No code fence or explanation.\n"
        f"Question: {row['question']}\n"
        f"Version: {row['version_before']} -> {row['version_after']}\n"
        f"Scope: {row['scope']}\n"
        f"Required new facts:\n{facts}\n"
        f"Protected facts:\n{protected or '- Preserve still-correct details.'}\n"
        f"Stale value: {row['old_value']}\n"
        f"Text before marked paragraph: {row['prefix']}\n"
        f"Marked old paragraph: {row['old_span']}\n"
        f"Text after marked paragraph: {row['suffix']}"
    )
