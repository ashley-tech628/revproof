"""Evidence-conditioned ranking of complete character-span boundaries.

The candidate generator and scorer accept only a source sentence, evidence,
and QA proposals at inference. Gold offsets are training labels only.
"""

import math
import re

from span_editor.locator import locate_sentence_candidates
from span_editor.train_selector import rank_pairs


def boundaries(sentence):
    points = {0, len(sentence)}
    for match in re.finditer(r"\S+", sentence):
        points.update((match.start(), match.end()))
    for match in re.finditer(r"\b", sentence):
        points.add(match.start())
    return sorted(points)


def nearby_spans(sentence, start, end, radius=3, max_chars=80,
                 max_variants=36):
    """Return valid whole-boundary variants, nearest the QA proposal first."""
    if not 0 <= start < end <= len(sentence):
        return []
    points = boundaries(sentence)

    def nearest(position):
        return sorted(points, key=lambda value: (abs(value - position), value))[:
            2 * radius + 3]

    options = {(start, end)}
    for a in nearest(start):
        for b in nearest(end):
            if 0 <= a < b <= len(sentence) and b - a <= max_chars:
                options.add((a, b))
    valid = [(a, b) for a, b in options if b - a <= max_chars and
             sentence[a:b].strip() == sentence[a:b]]
    valid.sort(key=lambda pair: (abs(pair[0] - start) + abs(pair[1] - end),
                                 abs(pair[0] - start), abs(pair[1] - end),
                                 pair[0], pair[1]))
    return valid[:max_variants]


def marked_sentence(sentence, start, end, context_chars=200):
    """Keep the proposed edit visible even when the article sentence is long."""
    if not 0 <= start < end <= len(sentence):
        raise ValueError("invalid candidate boundaries")
    left = max(0, start - context_chars)
    right = min(len(sentence), end + context_chars)
    return (sentence[left:start] + " [EDIT] " + sentence[start:end] +
            " [/EDIT] " + sentence[end:right])


def encode_candidate(tokenizer, sentence, evidence, start, end,
                     max_length=384):
    return tokenizer(marked_sentence(sentence, start, end), evidence,
                     truncation=True, max_length=max_length)


def build_examples(tokenizer, rows, max_length=384, negatives=6,
                   radius=3, max_span_chars=80):
    """Training negatives are perturbed gold spans, never dev labels."""
    examples = []
    positives = skipped = 0
    for n, row in enumerate(rows, 1):
        refs = [row["evidence_items"][i] for i in row["reference_ids"]
                if row["evidence_items"][i]]
        if not refs:
            skipped += 1
            continue
        sentence = row["source_sentences"][row["edit_index"]]
        start, end = row["span_start"], row["span_end"]
        variants = nearby_spans(sentence, start, end, radius,
                                max_span_chars, 100)
        if (start, end) not in variants:
            skipped += 1
            continue
        wrong = [pair for pair in variants if pair != (start, end)]
        if not wrong:
            skipped += 1
            continue
        # Alternate reference items without multiplying an already large dataset.
        evidence = refs[n % len(refs)]
        for (a, b), label in [((start, end), 1)] + [
                (pair, 0) for pair in wrong[:negatives]]:
            encoded = encode_candidate(tokenizer, sentence, evidence, a, b,
                                       max_length)
            encoded["labels"] = label
            examples.append(encoded)
        positives += 1
        if n == 1 or n % 1000 == 0 or n == len(rows):
            print(f"ENCODE BOUNDARY V3: {n}/{len(rows)}, "
                  f"positive={positives}, examples={len(examples)}", flush=True)
    return examples, positives, skipped


def score_candidates(model, tokenizer, candidates, torch, max_length=384,
                     batch_size=24):
    """Attach a learned log odds score to candidate dictionaries."""
    if not candidates:
        return []
    device = next(model.parameters()).device
    model.eval()
    scored = []
    for offset in range(0, len(candidates), batch_size):
        chunk = candidates[offset:offset + batch_size]
        fields = tokenizer([marked_sentence(x["sentence"], x["start"],
                                            x["end"]) for x in chunk],
                           [x["evidence"] for x in chunk], padding=True,
                           truncation=True, max_length=max_length,
                           return_tensors="pt")
        fields = {key: value.to(device) for key, value in fields.items()}
        with torch.inference_mode():
            logits = model(**fields).logits.float()
        margins = (logits[:, 1] - logits[:, 0]).tolist()
        scored.extend({**candidate, "boundary_logit": float(margin)}
                      for candidate, margin in zip(chunk, margins))
    return scored


def rank_boundary_candidates(candidates, max_candidates=8,
                             max_per_pair=2, selector_weight=0.5,
                             locator_weight=0.05):
    """Use a pair cap to avoid spending every generation on one evidence item."""
    unique = {}
    for x in candidates:
        probability = min(max(x["selector_probability"], 1e-6), 1 - 1e-6)
        score = (x["boundary_logit"] +
                 selector_weight * math.log(probability / (1 - probability)) +
                 locator_weight * x["score_above_null"])
        item = {**x, "proposal_score": score}
        key = (x["sentence_index"], x["evidence_id"], x["start"], x["end"])
        if key not in unique or score > unique[key]["proposal_score"]:
            unique[key] = item
    ranked = sorted(unique.values(),
                    key=lambda x: (-x["proposal_score"], x["pair_rank"],
                                   x["start"], x["end"]))
    selected, counts = [], {}
    for candidate in ranked:
        pair = candidate["pair_rank"]
        if counts.get(pair, 0) >= max_per_pair:
            continue
        selected.append(candidate)
        counts[pair] = counts.get(pair, 0) + 1
        if len(selected) == max_candidates:
            break
    return selected


def merge_candidates(primary, secondary, primary_limit=4, total_limit=8):
    """A fixed-budget hybrid for diagnosis; deduplicate article locations."""
    selected, seen = [], set()
    for group in (primary[:primary_limit], secondary,
                  primary[primary_limit:]):
        for candidate in group:
            key = (candidate["sentence_index"], candidate["start"],
                   candidate["end"])
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) == total_limit:
                return selected
    return selected


def propose(model, tokenizer, selector, selector_tokenizer, locator,
            locator_tokenizer, source, torch, selector_length, locator_length,
            boundary_length, top_sentences=4, top_pairs=6, top_spans=3,
            radius=3, max_variants=36, max_candidates=8,
            max_span_chars=80, max_per_pair=2,
            return_diagnostics=False):
    """All proposal inputs are inference-visible fields of `source`."""
    pairs = rank_pairs(selector, selector_tokenizer, source, torch,
                       top_sentences, top_pairs, selector_length)
    raw = []
    v2_candidates = []
    for pair_rank, pair in enumerate(pairs):
        evidence_id = pair["evidence_id"]
        sentence_index = pair["sentence_index"]
        sentence = source["source_sentences"][sentence_index]
        evidence = source["evidence_items"][evidence_id]
        local = {**source, "evidence": evidence}
        locations = locate_sentence_candidates(locator, locator_tokenizer,
                                               local, sentence_index, torch,
                                               locator_length, top_spans)
        for span_rank, location in enumerate(locations):
            if len(v2_candidates) < max_candidates:
                v2_candidates.append({**location, "evidence_id": evidence_id})
            for a, b in nearby_spans(sentence, location["start"],
                                     location["end"], radius,
                                     max_span_chars, max_variants):
                raw.append({**location, "start": a, "end": b,
                            "old_span": sentence[a:b], "sentence": sentence,
                            "evidence": evidence, "evidence_id": evidence_id,
                            "selector_probability": pair["selector_probability"],
                            "pair_rank": pair_rank, "span_rank": span_rank})
    unique = {}
    for candidate in raw:
        key = (candidate["sentence_index"], candidate["evidence_id"],
               candidate["start"], candidate["end"])
        if key not in unique or (candidate["score_above_null"],
                                 -candidate["span_rank"]) > (
                                     unique[key]["score_above_null"],
                                     -unique[key]["span_rank"]):
            unique[key] = candidate
    scored = score_candidates(model, tokenizer, list(unique.values()), torch,
                              boundary_length)
    chosen = rank_boundary_candidates(scored, max_candidates, max_per_pair)
    # `sentence` and `evidence` are duplicated for scoring, not result storage.
    compact = lambda items: [{k: v for k, v in x.items()
                              if k not in ("sentence", "evidence")}
                             for x in items]
    if return_diagnostics:
        return pairs, compact(chosen), {
            "v2_candidates": v2_candidates,
            "pool_locations": [{"sentence_index": x["sentence_index"],
                                "start": x["start"], "end": x["end"]}
                               for x in unique.values()]}
    return pairs, compact(chosen)
