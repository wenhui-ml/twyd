import glob
import json
import statistics as st
import sys


def stats(path):
    rows = [json.loads(l) for l in open(path)]
    g = [r["grad_norm"] for r in rows if "grad_norm" in r]
    return {
        "n": len(g),
        "median": st.median(g) if g else float("nan"),
        "p90": sorted(g)[int(0.9 * len(g))] if g else float("nan"),
        "max": max(g) if g else float("nan"),
        "clip_frac": (sum(1 for x in g if x > 1.0) / len(g)) if g else float("nan"),
    }


def main():
    print(f"{'run':<40s} {'median':>8s} {'p90':>8s} {'max':>9s} {'frac>1.0':>9s}")
    for pat in sys.argv[1:]:
        for path in sorted(glob.glob(pat)):
            s = stats(path)
            name = path.split("/output/")[-1].split("/logging.jsonl")[0]
            print(f"{name:<40s} {s['median']:8.2f} {s['p90']:8.2f} {s['max']:9.2f} "
                  f"{100 * s['clip_frac']:8.0f}%")


if __name__ == "__main__":
    main()
