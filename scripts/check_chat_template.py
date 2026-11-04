#!/usr/bin/env python
"""Fail loudly when a checkpoint ships an unusable chat template.

Why this exists: `LRC-1.5B-Base` shipped `chat_template.jinja` containing the literal
string "Entry not found" (15 bytes).  transformers prefers that file over the correct
`chat_template` field in `tokenizer_config.json`, so `AutoTokenizer.chat_template`
silently returned the stub and every `apply_chat_template` call rendered 15 characters
instead of the conversation.  It surfaced only when the PT corpus builder produced
15-byte instruction documents.  Anything that ships weights should run this first.

  python scripts/check_chat_template.py /home/chenwh/models/LRC-1.5B-Base
  python scripts/check_chat_template.py --all /home/chenwh/models

Exit code 0 = usable, 1 = at least one checkpoint is broken.
"""
import argparse
import glob
import json
import os
import sys

STUBS = ("entry not found", "none", "null", "")
MIN_LEN = 200
# Role delimiters across the families we ship: Llama-3, Qwen/ChatML, Mistral/Mixtral.
ROLE_MARKERS = ("<|start_header_id|>", "<|im_start|>", "[INST]", "<|user|>",
                "<|assistant|>", "User:", "user\n")


def check(path, quiet=False):
    from transformers import AutoTokenizer

    problems = []
    jinja = os.path.join(path, "chat_template.jinja")
    if os.path.isfile(jinja):
        raw = open(jinja, "r", encoding="utf-8", errors="ignore").read().strip()
        if raw.lower() in STUBS or len(raw) < MIN_LEN:
            problems.append(f"chat_template.jinja is a stub ({len(raw)} B: {raw[:40]!r})")

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tpl = tok.chat_template or ""
    if tpl.strip().lower() in STUBS or len(tpl) < MIN_LEN:
        problems.append(f"AutoTokenizer.chat_template is unusable ({len(tpl)} chars)")

    rendered = ""
    if not problems:
        msgs = [{"role": "user", "content": "ping"},
                {"role": "assistant", "content": "pong"},
                {"role": "user", "content": "ping again"}]
        try:
            rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception as e:                      # noqa: BLE001
            problems.append(f"apply_chat_template raised: {e}")
        if rendered:
            if not any(m in rendered for m in ROLE_MARKERS):
                problems.append("rendered text has no role delimiter from any known family")
            if "ping" not in rendered or "pong" not in rendered:
                problems.append("rendered text lost the message contents")
            try:
                ids = tok(rendered, add_special_tokens=False)["input_ids"]
                if len(ids) < len(rendered) / 20:
                    problems.append(f"tokenisation collapsed ({len(ids)} ids for {len(rendered)} chars)")
                if tok.decode(ids).count("ping") < 2:
                    problems.append("decode() lost the turns")
            except Exception as e:                  # noqa: BLE001
                problems.append(f"tokenisation raised: {e}")

    if not quiet:
        status = "OK  " if not problems else "FAIL"
        print(f"[{status}] {path}  template={len(tpl)} chars  rendered={len(rendered)} chars")
        for p in problems:
            print(f"       - {p}")
        if rendered and not problems:
            print(f"       {rendered[:100]!r}")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", metavar="DIR", help="check every sub-dir that has a config.json")
    args = ap.parse_args()

    targets = list(args.paths)
    if args.all:
        targets += sorted(d for d in glob.glob(os.path.join(args.all, "*"))
                          if os.path.isfile(os.path.join(d, "config.json")))
    if not targets:
        ap.error("give one or more checkpoint dirs, or --all <dir>")

    ok = True
    for t in targets:
        try:
            ok = check(t) and ok
        except Exception as e:                      # noqa: BLE001
            print(f"[FAIL] {t}  load raised: {e}")
            ok = False
    print("all usable" if ok else "BROKEN chat template(s) found")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
