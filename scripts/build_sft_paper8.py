#!/usr/bin/env python
"""Build Paper 1's 8-dataset SFT mixture as ONE local jsonl (App. A.6).

ms-swift can mix hub datasets itself, but concatenating them trips over schema
mismatches (`idx` large_string vs string) and the subset+count parser. A single local
file sidesteps both, pins the mixture proportions, and is the format our SFT arms
already use ({"messages": [...]}).
"""
import argparse, glob, json, os, sys

SPEC = [  # (modelscope id, target samples)
    ('swift/ultrachat_200k', 60000),
    ('swift/OpenHermes-2.5', 50000),
    ('swift/SlimOrca', 50000),
    ('AI-ModelScope/WizardLM_evol_instruct_V2_196k', 40000),
    ('AI-ModelScope/Open-Platypus', 30000),
    ('AI-ModelScope/CodeAlpaca-20k', 15000),
    ('AI-ModelScope/MathInstruct', 40000),
]

def to_messages(row):
    if isinstance(row.get('messages'), list) and row['messages']:
        out=[]
        for m in row['messages']:
            r = m.get('role') or m.get('from') or ''
            c = m.get('content') or m.get('value') or ''
            r = {'human':'user','gpt':'assistant'}.get(r, r)
            if r in ('user','assistant','system') and c: out.append({'role':r,'content':str(c)})
        return out or None
    if isinstance(row.get('conversations'), list) and row['conversations']:
        out=[]
        for m in row['conversations']:
            r = {'human':'user','gpt':'assistant'}.get(m.get('from',''), m.get('from',''))
            if r in ('user','assistant','system') and m.get('value'):
                out.append({'role':r,'content':str(m['value'])})
        return out or None
    q = row.get('instruction') or row.get('question') or row.get('query')
    a = row.get('output') or row.get('response') or row.get('answer')
    if q and a:
        extra = row.get('input')
        u = str(q) + (('\n' + str(extra)) if extra else '')
        return [{'role':'user','content':u},{'role':'assistant','content':str(a)}]
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='data/sft_paper8.jsonl')
    ap.add_argument('--scale', type=float, default=1.0)
    a=ap.parse_args()
    from modelscope import snapshot_download
    from datasets import load_dataset
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    total=0
    with open(a.out,'w',encoding='utf-8') as fo:
        for mid, want in SPEC:
            want=int(want*a.scale)
            try:
                p=snapshot_download(mid, repo_type='dataset')
            except Exception as e:
                print(f'[skip] {mid}: {e}'); continue
            files=[f for f in glob.glob(os.path.join(p,'**','*'), recursive=True)
                   if f.endswith(('.jsonl','.parquet'))]
            if not files:
                print(f'[skip] {mid}: no jsonl/parquet'); continue
            n=0
            for f in files:
                if n>=want: break
                try:
                    ds=load_dataset('parquet' if f.endswith('.parquet') else 'json',
                                    data_files=f, split='train')
                except Exception as e:
                    print(f'  [warn] {os.path.basename(f)}: {str(e)[:80]}'); continue
                for row in ds:
                    msgs=to_messages(row)
                    if not msgs: continue
                    fo.write(json.dumps({'messages':msgs}, ensure_ascii=False)+'\n')
                    n+=1; total+=1
                    if n>=want: break
            print(f'[ok] {mid}: {n} samples')
    print(f'[done] {total} samples -> {a.out} ({os.path.getsize(a.out)/2**20:.1f} MiB)')

if __name__=='__main__': main()
