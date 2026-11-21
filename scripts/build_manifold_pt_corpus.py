#!/usr/bin/env python3
"""Build a Manifold-Aligned PT corpus with In-Context Contrastive Neighbor Packing.

Why:
  1. When compressing residual width 3072 -> 1536, distant coarse domains stay
     separated, while nearby concepts in the same sub-discipline (MMLU distractors)
     collapse onto the same 1536-d subspace.
  2. Standard PT packs random unrelated documents into 2048-token windows, so
     easily confused neighbor concepts rarely co-occur in the same attention
     context and receive no local tangent-space separation gradient.
  3. This builder clusters knowledge documents by fine-grained rare-term /
     domain signatures and packs 3-4 semantically adjacent documents into the
     same ~2048-token (~7500-char) window, interleaved with high-score
     FineWeb-Edu discourse/reasoning carrier documents (preserving the C1
     super-additive synergy).

Usage:
  python scripts/build_manifold_pt_corpus.py \
      --knowledge data/pt_knowledge.jsonl \
      --web data/fw_score4_1B.jsonl \
      --out data/pt_manifold_200M.jsonl \
      --target-mb 800
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import random
import re
import time

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "all", "when", "can", "more",
    "which", "their", "there", "been", "one", "would", "will", "who", "about",
    "after", "into", "other", "than", "then", "them", "these", "two", "may",
    "first", "also", "new", "some", "could", "time", "such", "only", "its",
    "over", "between", "used", "where", "most", "many", "made", "through",
    "before", "years", "his", "her", "they", "she", "you", "your", "out",
    "during", "while", "under", "both", "each", "those", "however", "because",
    "known", "including", "called", "part", "state", "states", "system",
    "university", "century", "history", "world", "united", "american", "national",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,24}")


def _extract_signature(text: str, df: Counter, n_docs: int) -> tuple[str, str]:
    """Return (coarse_bucket, fine_key) from mid-rarity technical terms in the doc."""
    head = text[:2000].lower()
    words = [w for w in _WORD_RE.findall(head) if w not in _STOP]
    if not words:
        h = hashlib.md5(head.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return h[:3], h
    tf = Counter(words)
    scored = []
    max_df = max(5, int(n_docs * 0.03))
    for w, c in tf.items():
        d = df.get(w, 1)
        if 2 <= d <= max_df:
            # Favour specific technical terms that appear multiple times in the doc
            score = (c ** 1.2) / (d ** 0.5)
            scored.append((score, w))
    if not scored:
        scored = [(c, w) for w, c in tf.most_common(4)]
    scored.sort(reverse=True)
    top_terms = [w for _, w in scored[:3]]
    coarse = top_terms[0]
    fine = "|".join(sorted(top_terms[:2]))
    return coarse, fine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", default="data/pt_knowledge.jsonl")
    ap.add_argument("--extra-knowledge", default="",
                    help="optional second knowledge JSONL to merge before clustering")
    ap.add_argument("--extra-max-mb", type=float, default=300.0,
                    help="max MB to read from --extra-knowledge")
    ap.add_argument("--know-skip-mb", type=float, default=0.0,
                    help="skip initial MB of --knowledge to construct non-overlapping v2 shards")
    ap.add_argument("--web", default="data/fw_score4_1B.jsonl")
    ap.add_argument("--web-skip-mb", type=float, default=0.0,
                    help="skip initial MB of --web to construct non-overlapping v2 shards")
    ap.add_argument("--out", default="data/pt_manifold_200M.jsonl")
    ap.add_argument("--target-mb", type=float, default=800.0,
                    help="total output size in MB (~4 MB per 1M tokens -> 800 MB ~= 200M tokens)")
    ap.add_argument("--know-ratio", type=float, default=0.60,
                    help="share of neighbor-packed knowledge windows (default 0.60)")
    ap.add_argument("--window-chars", type=int, default=7600,
                    help="target chars per packed 2048-token window")
    ap.add_argument("--doc-cap-chars", type=int, default=2600,
                    help="max chars per individual neighbor doc inside a packed window")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    target_bytes = int(args.target_mb * 1e6)
    know_budget = int(target_bytes * args.know_ratio)
    web_budget = target_bytes - know_budget

    print(f"[manifold] reading knowledge pool from {args.knowledge} (skip={args.know_skip_mb:.0f} MB) ...", flush=True)
    know_docs: list[str] = []
    df: Counter = Counter()
    max_read_bytes = int(know_budget * 1.6)
    skip_know_bytes = int(args.know_skip_mb * 1e6)
    skipped_b = 0
    read_b = 0
    with open(args.knowledge, "r", encoding="utf-8") as f:
        for line in f:
            lb = len(line)
            if skipped_b < skip_know_bytes:
                skipped_b += lb
                continue
            try:
                t = (json.loads(line).get("text") or "").strip()
            except Exception:
                continue
            if len(t) < 300:
                continue
            know_docs.append(t)
            seen = set(_WORD_RE.findall(t[:1500].lower())) - _STOP
            for w in seen:
                df[w] += 1
            read_b += lb
            if read_b >= max_read_bytes and len(know_docs) >= 20000:
                break

    if args.extra_knowledge and os.path.isfile(args.extra_knowledge) and args.extra_max_mb > 0:
        extra_cap = int(args.extra_max_mb * 1e6)
        extra_b = 0
        with open(args.extra_knowledge, "r", encoding="utf-8") as ef:
            for line in ef:
                try:
                    t = (json.loads(line).get("text") or "").strip()
                except Exception:
                    continue
                if len(t) < 300:
                    continue
                know_docs.append(t)
                seen = set(_WORD_RE.findall(t[:1500].lower())) - _STOP
                for w in seen:
                    df[w] += 1
                lb = len(line)
                extra_b += lb
                read_b += lb
                if extra_b >= extra_cap:
                    break
        print(f"[manifold] merged extra knowledge from {args.extra_knowledge} ({extra_b/1e6:.1f} MB)", flush=True)

    rng.shuffle(know_docs)

    print(f"[manifold] loaded {len(know_docs)} knowledge docs ({read_b/1e6:.1f} MB) in {time.time()-t0:.1f}s",
          flush=True)

    # Group by coarse anchor term, sorted by fine signature so adjacent items
    # are hard semantic neighbors (same technical anchor, contrasting specifics).
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for d in know_docs:
        coarse, fine = _extract_signature(d, df, len(know_docs))
        buckets[coarse].append((fine, d[: args.doc_cap_chars]))

    packed_windows: list[str] = []
    know_written = 0
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)

    leftover: list[str] = []
    for k in bucket_keys:
        items = buckets[k]
        if len(items) < 2:
            leftover.extend(doc for _, doc in items)
            continue
        items.sort(key=lambda x: x[0])
        cur_parts: list[str] = []
        cur_len = 0
        for _, doc in items:
            sep = "\n\n---\n\n" if cur_parts else ""
            cur_parts.append(sep + doc)
            cur_len += len(sep) + len(doc)
            if cur_len >= args.window_chars:
                win = "".join(cur_parts)
                packed_windows.append(win)
                know_written += len(win.encode("utf-8"))
                cur_parts, cur_len = [], 0
                if know_written >= know_budget:
                    break
        if cur_parts:
            leftover.append("".join(cur_parts))
        if know_written >= know_budget:
            break

    # Pack any remaining singletons
    cur_parts, cur_len = [], 0
    for doc in leftover:
        if know_written >= know_budget:
            break
        sep = "\n\n---\n\n" if cur_parts else ""
        cur_parts.append(sep + doc)
        cur_len += len(sep) + len(doc)
        if cur_len >= args.window_chars:
            win = "".join(cur_parts)
            packed_windows.append(win)
            know_written += len(win.encode("utf-8"))
            cur_parts, cur_len = [], 0

    rng.shuffle(packed_windows)
    print(f"[manifold] built {len(packed_windows)} neighbor-packed knowledge windows "
          f"({know_written/1e6:.1f} MB)", flush=True)

    # Interleave with FineWeb-Edu reasoning carrier docs
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    web_written = 0
    ki = 0
    n_out = 0
    with open(args.web, "r", encoding="utf-8") as wsrc, \
            open(args.out, "w", encoding="utf-8") as out:
        skip_web_bytes = int(args.web_skip_mb * 1e6)
        skipped_web = 0
        while skipped_web < skip_web_bytes:
            wline = wsrc.readline()
            if not wline:
                break
            skipped_web += len(wline.encode("utf-8"))
        while (know_written > 0 and ki < len(packed_windows)) or web_written < web_budget:
            # Write 1-2 neighbor-packed knowledge windows
            for _ in range(2):
                if ki < len(packed_windows):
                    line = json.dumps({"text": packed_windows[ki]}, ensure_ascii=False) + "\n"
                    out.write(line)
                    ki += 1
                    n_out += 1
            # Write web carrier docs to maintain target ratio
            target_web_now = int((ki / max(1, len(packed_windows))) * web_budget)
            while web_written < target_web_now or (ki >= len(packed_windows) and web_written < web_budget):
                wline = wsrc.readline()
                if not wline:
                    web_budget = web_written
                    break
                out.write(wline)
                web_written += len(wline.encode("utf-8"))
                n_out += 1

    total_mb = os.path.getsize(args.out) / 1e6
    print(f"[manifold] DONE: wrote {n_out} records ({total_mb:.1f} MB, "
          f"~{total_mb/4.0:.0f}M tokens) -> {args.out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
