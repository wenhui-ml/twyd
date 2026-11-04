#!/usr/bin/env python
"""Check OpenReview author-response lengths.

AAAI-27 allows one response per review, 1,500 characters each, no links, no uploads.
OpenReview counts the text as typed, so LaTeX markup counts character by character.

    python scripts/check_rebuttal_len.py [path/to/REBUTTAL_openreview.md]
"""
import re
import sys

DEFAULT = "/home/chenwh/Papers/paper/TWYD-LRC/AAAI2027/REBUTTAL_openreview.md"
LIMIT = 1500
BLOCK = re.compile(r"BEGIN\n(.*?)\nEND", re.S)
HEAD = re.compile(r"^##\s+(.+)$", re.M)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    text = open(path, encoding="utf-8").read()
    heads = [(m.start(), m.group(1)) for m in HEAD.finditer(text)]
    blocks = list(BLOCK.finditer(text))
    print(f"{path}\n{len(blocks)} response block(s), limit {LIMIT} characters\n")
    ok = True
    for i, b in enumerate(blocks):
        title = "<untitled>"
        for pos, t in heads:
            if pos < b.start():
                title = t
        body = b.group(1)
        n = len(body)
        flag = "OK " if n <= LIMIT else "OVER"
        if n > LIMIT:
            ok = False
        print(f"[{flag}] {n:5d} chars  ({n - LIMIT:+d})  {title}")
        # link check: the PC rejects papers with external links in responses
        if re.search(r"https?://|www\.", body):
            print("        !! contains a URL -- not allowed")
            ok = False
    print("\nall within limit" if ok else "\ntrim the over-limit blocks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
