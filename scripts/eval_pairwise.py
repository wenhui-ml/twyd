import itertools
import json
import math
import sys


def main():
    d = json.load(open(sys.argv[1]))
    rows = d["rows"]
    lab = []
    for r in rows:
        parts = r["model"].rstrip("/").split("/")
        if len(parts) >= 4 and parts[-1].startswith("checkpoint-"):
            lab.append(f"{parts[-4]}/{parts[-3]}/{parts[-1].replace('checkpoint-', 'ckpt-')}")
        elif parts[-1].startswith("checkpoint-"):
            lab.append(f"{parts[-3]}/{parts[-1]}")
        else:
            lab.append(parts[-1])
    n = len(rows[0]["per_chunk_ce"])
    print(f"{n} chunks, paired differences (row - column), mean [95% CI]")
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = rows[i]["per_chunk_ce"], rows[j]["per_chunk_ce"]
        diff = [x - y for x, y in zip(a, b)]
        m = sum(diff) / n
        var = sum((x - m) ** 2 for x in diff) / (n - 1)
        se = math.sqrt(var / n)
        print(f"{lab[i][:34]:<35s} vs {lab[j][:26]:<27s} {m:+.4f} "
              f"[{m - 1.96 * se:+.4f},{m + 1.96 * se:+.4f}]  z={m / se:+.1f}")


if __name__ == "__main__":
    main()
