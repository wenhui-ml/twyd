import glob
import json
import sys


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return {int(x["global_step/max_steps"].split("/")[0]): x
            for x in rows if "loss" in x and "global_step/max_steps" in x}


def main():
    for arm in sys.argv[1:] or ["22-27"]:
        runs = sorted(glob.glob(f"output/v15-w1/{arm}/v*/logging.jsonl"))
        if len(runs) < 2:
            print(f"{arm}: {len(runs)} run(s), nothing to compare")
            continue
        a, b = load(runs[0]), load(runs[1])
        common = sorted(set(a) & set(b))
        dl = [abs(a[s]["loss"] - b[s]["loss"]) for s in common]
        da = [abs(a[s]["token_acc"] - b[s]["token_acc"]) for s in common]
        ml = [sum(d[s]["loss"] for s in common) / len(common) for d in (a, b)]
        ma = [sum(d[s]["token_acc"] for s in common) / len(common) for d in (a, b)]
        print(f"{arm}: {len(common)} common steps")
        print(f"  per-step |dloss|: mean {sum(dl)/len(dl):.3f} max {max(dl):.2f}")
        print(f"  per-step |dacc| : mean {100*sum(da)/len(da):.2f}pp max {100*max(da):.2f}pp")
        print(f"  mean loss over steps: {ml[0]:.2f} vs {ml[1]:.2f} (d={ml[0]-ml[1]:+.2f})")
        print(f"  mean acc  over steps: {ma[0]:.4f} vs {ma[1]:.4f} (d={ma[0]-ma[1]:+.4f})")


if __name__ == "__main__":
    main()
