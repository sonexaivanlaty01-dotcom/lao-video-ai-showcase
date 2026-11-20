"""วัดความเร็ว + ความแม่น: PyTorch (เดิม) เทียบ CTranslate2 หลายแบบ บนชุดสอบเดียวกับ compare_models
  .venv\\Scripts\\python.exe scripts\\bench_fast.py
ผล: results/bench_fast.txt
"""
import gc
import sys
import time

import jiwer
import torch
from common import RESULTS, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")


def cer(refs, hyps):
    return 100 * jiwer.cer([norm_for_cer(x) for x in refs], [norm_for_cer(x) or "-" for x in hyps])


def main():
    from compare_models import load_audio, tests
    from ct2_asr import V3Fast
    from v3_asr import V3

    sets = tests()
    audio = {k: [load_audio(r["file"]) for r in rows] for k, rows in sets.items()}
    refs = {k: [r["text"] for r in rows] for k, rows in sets.items()}
    total_sec = sum(len(a) for v in audio.values() for a in v) / 16000
    configs = [
        ("PyTorch เดิม (Ultra3)", lambda: V3(model_dir="models/full-lao-ultra3"), {}),
        ("CT2 int8_float32 b16", lambda: V3Fast("models/full-lao-ultra3-ct2-int8", compute_type="int8_float32", batch=16), {}),
        ("CT2 int8 b16", lambda: V3Fast("models/full-lao-ultra3-ct2-int8", compute_type="int8", batch=16), {}),
        ("CT2 int8_float32 b24", lambda: V3Fast("models/full-lao-ultra3-ct2-int8", compute_type="int8_float32", batch=24), {}),
        ("CT2 f16-weights float32", lambda: V3Fast("models/full-lao-ultra3-ct2-f16", compute_type="float32", batch=16), {}),
    ]
    lines = [f"เสียงทดสอบรวม {total_sec / 60:.1f} นาที ({sum(len(v) for v in audio.values())} คลิป)"]
    print(lines[0], flush=True)
    for name, make, _ in configs:
        try:
            m = make()
            m.transcribe([next(iter(audio.values()))[0]])  # อุ่นเครื่อง
            t = time.time()
            res = {k: m.transcribe_detailed(v, max_alts=0) for k, v in audio.items()}
            dt = time.time() - t
            parts = [f"{k[:16]} {cer(refs[k], [d['text'] for d in res[k]]):5.1f}%" for k in sets]
            msg = f"{name:26s} | {dt:6.1f}s ({total_sec / dt:5.1f}x realtime) | " + " | ".join(parts)
        except Exception as e:
            msg = f"{name:26s} | ERROR {e}"
        print(msg, flush=True)
        lines.append(msg)
        m = None
        gc.collect()
        torch.cuda.empty_cache()
    (RESULTS / "bench_fast.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
