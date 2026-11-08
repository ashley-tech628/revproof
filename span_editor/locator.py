"""Evidence-conditioned sentence and character-span localization helpers."""

import re


def question(row):
    # The target/replacement and gold offsets must never enter this text.
    return ("Find the outdated phrase in this sentence using the new evidence. "
            "If this sentence needs no change, answer no span. "
            f"New evidence: {row['evidence']}")


def lexical_overlap(sentence, evidence):
    a = set(re.findall(r"\w+", sentence.casefold()))
    b = set(re.findall(r"\w+", evidence.casefold()))
    return len(a & b) / max(1, len(a))


def candidate_sentences(row, negatives=3):
    gold = row["edit_index"]
    others = sorted((i for i in range(len(row["source_sentences"])) if i != gold),
                    key=lambda i: (-lexical_overlap(row["source_sentences"][i],
                                                     row["evidence"]), i))
    return [gold] + others[:negatives]


def encode_pair(tokenizer, row, sentence_index, max_length=384, training=False):
    context = row["source_sentences"][sentence_index]
    if not context:
        return None
    try:
        encoded = tokenizer(question(row), context, truncation="only_first",
                            max_length=max_length, return_offsets_mapping=True)
    except Exception as exc:
        # The fast tokenizer raises a plain Exception for an overlong second
        # sequence under `only_first`, and a ValueError in other versions.
        # Skip that pair; do not hide unrelated tokenizer failures.
        if isinstance(exc, ValueError) or "sequence to truncate too short" in str(exc).lower():
            return None
        raise
    sequence_ids = encoded.sequence_ids()
    offsets = encoded.pop("offset_mapping")
    context_tokens = [i for i, kind in enumerate(sequence_ids)
                      if kind == 1 and offsets[i][1] > offsets[i][0]]
    if not context_tokens:
        return None
    # If the context itself was truncated, do not silently train on a partial sentence.
    if offsets[context_tokens[-1]][1] < len(context.rstrip()):
        return None
    if training:
        null = encoded["input_ids"].index(tokenizer.cls_token_id)
        start = end = null
        if sentence_index == row["edit_index"]:
            a, b = row["span_start"], row["span_end"]
            starts = [i for i in context_tokens if offsets[i][0] <= a < offsets[i][1]]
            ends = [i for i in context_tokens if offsets[i][0] < b <= offsets[i][1]]
            if not starts or not ends:
                return None
            start, end = starts[0], ends[-1]
            if start > end:
                return None
        encoded["start_positions"] = start
        encoded["end_positions"] = end
    return encoded, offsets, context_tokens


def best_span(start_logits, end_logits, offsets, context_tokens, max_tokens=32):
    """Return (score above null, char_start, char_end) for a single sentence."""
    null = float(start_logits[0] + end_logits[0])
    best = None
    starts = sorted(context_tokens, key=lambda i: float(start_logits[i]), reverse=True)[:20]
    ends = sorted(context_tokens, key=lambda i: float(end_logits[i]), reverse=True)[:20]
    for i in starts:
        for j in ends:
            if i <= j and j - i + 1 <= max_tokens:
                a, b = offsets[i][0], offsets[j][1]
                value = (float(start_logits[i] + end_logits[j] - null), a, b)
                if a < b and (best is None or value[0] > best[0]):
                    best = value
    return best


def top_spans(start_logits, end_logits, offsets, context_tokens, top_k=3,
              max_tokens=32):
    """Distinct character boundaries ranked by null-relative QA score."""
    null = float(start_logits[0] + end_logits[0])
    starts = sorted(context_tokens, key=lambda i: float(start_logits[i]), reverse=True)[:30]
    ends = sorted(context_tokens, key=lambda i: float(end_logits[i]), reverse=True)[:30]
    by_boundary = {}
    for i in starts:
        for j in ends:
            if not i <= j or j - i + 1 > max_tokens:
                continue
            a, b = offsets[i][0], offsets[j][1]
            if a >= b:
                continue
            score = float(start_logits[i] + end_logits[j] - null)
            by_boundary[(a, b)] = max(score, by_boundary.get((a, b), float("-inf")))
    return sorted(((score, a, b) for (a, b), score in by_boundary.items()),
                  key=lambda x: (-x[0], x[1], x[2]))[:top_k]


def locate_sentence_candidates(model, tokenizer, row, sentence_index, torch,
                               max_length=384, top_k=3, max_span_tokens=32):
    """Locate spans in one supplied source sentence and evidence item."""
    encoded = encode_pair(tokenizer, row, sentence_index, max_length)
    if encoded is None:
        return []
    fields, offsets, context_tokens = encoded
    device = next(model.parameters()).device
    inputs = {key: torch.tensor([fields[key]], device=device)
              for key in ("input_ids", "attention_mask") if key in fields}
    model.eval()
    with torch.inference_mode():
        output = model(**inputs)
    spans = top_spans(output.start_logits[0], output.end_logits[0],
                      offsets, context_tokens, top_k, max_span_tokens)
    return [{"sentence_index": sentence_index, "start": a, "end": b,
             "old_span": row["source_sentences"][sentence_index][a:b],
             "score_above_null": score} for score, a, b in spans]


def locate(model, tokenizer, row, torch, max_length=384, max_span_tokens=32):
    model.eval()
    device = next(model.parameters()).device
    best = None
    for sentence_index in range(len(row["source_sentences"])):
        encoded = encode_pair(tokenizer, row, sentence_index, max_length)
        if encoded is None:
            continue
        fields, offsets, context_tokens = encoded
        inputs = {key: torch.tensor([fields[key]], device=device)
                  for key in ("input_ids", "attention_mask") if key in fields}
        with torch.inference_mode():
            output = model(**inputs)
        span = best_span(output.start_logits[0], output.end_logits[0],
                         offsets, context_tokens, max_span_tokens)
        if span is None:
            continue
        score, a, b = span
        item = {"sentence_index": sentence_index, "start": a, "end": b,
                "old_span": row["source_sentences"][sentence_index][a:b],
                "score_above_null": score}
        if best is None or score > best["score_above_null"]:
            best = item
    return best
