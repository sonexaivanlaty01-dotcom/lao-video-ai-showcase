"""เทียบโมเดลตระกูล v3 (v3, v4, ...) กับข้อสอบหลายชุด: FLEURS test + คลิปจริงของทีม

  .venv\\Scripts\\python.exe scripts\\compare_models.py models\\full-lao-v3 models\\full-lao-v4

ข้อสอบ:
  - FLEURS test 100 คลิป (เสียงอ่านข่าว)
  - คลิปทีม: data/my_train/*/metadata.tsv (ต้องไม่ได้เอาไปเทรน!)
ผลลัพธ์บันทึกที่ results/compare_<ชื่อ>.txt
"""
import gc
import sys
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from common import RESULTS, ROOT, load_fleurs, norm_for_cer
from train import load_tsv
from v3_asr import V3

sys.stdout.reconfigure(encoding="utf-8")


def load_audio(f):
    import librosa

    return librosa.load(str(f), sr=16000)[0]


def tests():
    sets = {"FLEURS 100": load_fleurs("test", 100)}
    for tsv in sorted((ROOT / "data" / "my_train").glob("*/metadata.tsv")):
        sets[f"ทีม: {tsv.parent.name}"] = load_tsv(tsv)
    return sets


def cer(refs, hyps):
    r = [norm_for_cer(x) for x in refs]
    h = [norm_for_cer(x) or "-" for x in hyps]
    return jiwer.cer(r, h) * 100


def main():
    models = [Path(m) for m in sys.argv[1:]]
    sets = tests()
    audio = {k: [load_audio(r["file"]) for r in rows] for k, rows in sets.items()}
    table = {}
    for m in models:
        asr = V3(model_dir=m)
        for k, rows in sets.items():
            hyps = asr.transcribe(audio[k], max_new=200)
            table[(m.name, k)] = cer([r["text"] for r in rows], hyps)
            print(f"{m.name:14s} | {k:40s} | CER {table[(m.name, k)]:5.1f}%  ({len(rows)} คลิป)", flush=True)
        del asr
        gc.collect()
        torch.cuda.empty_cache()
    lines = ["ข้อสอบ".ljust(42) + "".join(m.name.rjust(14) for m in models)]
    for k, rows in sets.items():
        lines.append(f"{k} ({len(rows)})".ljust(42) + "".join(f"{table[(m.name, k)]:13.1f}%" for m in models))
    out = "\n".join(lines)
    print("\n" + out)
    (RESULTS / f"compare_{'_vs_'.join(m.name for m in models)}.txt").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
