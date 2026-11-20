"""วัดผลหลายโมเดลกับชุดทดสอบแบบ tsv (เช่น data/siang_lao/test.tsv ที่มีหลายสำเนียง)

  .venv\\Scripts\\python.exe scripts\\eval_tsv.py data\\siang_lao\\test.tsv --n 120
"""
import argparse
import collections
import random
import sys
import time
from pathlib import Path

import jiwer
from common import RESULTS, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")


def read_rows(tsv: Path, n: int):
    lines = tsv.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]
    for r in rows:
        r["file"] = tsv.parent / r["file"]
    random.Random(0).shuffle(rows)  # สุ่มแบบคงที่ ทุกโมเดลได้ชุดเดียวกัน
    return rows[:n]


def run_whisper(rows, adapter, base="openai/whisper-small"):
    import evaluate as ev

    processor, model = ev.load_model(base, adapter)
    out = ev.transcribe(processor, model, [r["file"] for r in rows])
    import torch

    del model
    torch.cuda.empty_cache()
    return out


def run_buzz(rows):
    from buzz_fast import Buzz, load_audio

    b = Buzz()
    out = b.transcribe([load_audio(r["file"]) for r in rows])
    del b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    rows = read_rows(Path(a.tsv), a.n)
    refs = [norm_for_cer(r["text"]) for r in rows]
    print(f"ชุดทดสอบ {len(rows)} คลิป | สำเนียง: {dict(collections.Counter(r.get('accent', '?') for r in rows))}\n")

    models = {
        "ของเรา v1": lambda: run_whisper(rows, "models/lora-small-lo/best"),
        "ของเรา v2": lambda: run_whisper(rows, "models/lora-small-lo-v2/best"),
        "ของเรา v3": lambda: run_whisper(rows, None, "models/full-lao-v3"),
        "BuzzASR": lambda: run_buzz(rows),
    }
    table = {}
    for name, fn in models.items():
        t = time.time()
        hyps = fn()
        hn = [norm_for_cer(h) for h in hyps]
        by = collections.defaultdict(lambda: ([], []))
        for r, ref, h in zip(rows, refs, hn):
            by[r.get("accent", "?")][0].append(ref)
            by[r.get("accent", "?")][1].append(h)
        table[name] = {"รวม": jiwer.cer(refs, hn), **{k: jiwer.cer(v[0], v[1]) for k, v in by.items()}}
        print(f"{name}: CER {table[name]['รวม']*100:.1f}%  ({time.time()-t:.0f} วิ)", flush=True)

    cols = list(next(iter(table.values())).keys())
    lines = ["| โมเดล | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, d in table.items():
        lines.append(f"| {name} | " + " | ".join(f"{d[c]*100:.1f}%" for c in cols) + " |")
    md = "\n".join(lines)
    print("\n" + md)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_{Path(a.tsv).parent.name}_{a.n}.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
