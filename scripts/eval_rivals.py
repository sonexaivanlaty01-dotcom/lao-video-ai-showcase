"""จัดอันดับแบบยุติธรรม: วัด Ultra3 ⚡ กับ AI ลาวของเจ้าอื่น บนชุดสอบเดียวกันทุกชุด
ชุดสอบ: FLEURS 100 + คลิปทีม (เหมือน compare_models) + SiangLao test (1,200 คลิป ชุดที่ SiangLao ใช้ประกาศผล)
คิด CER ด้วย norm_for_cer เดียวกันทุกตัว

  .venv\\Scripts\\python.exe scripts\\eval_rivals.py
"""
import gc
import os
import sys
import time

import jiwer
import torch
from common import MODELS, RESULTS, ROOT, norm_for_cer
from compare_models import load_audio, tests
from train import load_tsv

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")

CTC_RIVALS = ["SiangLao/xls-r-lao-asr", "SiangLao/xlsr-53-lao-asr"]


def cer(refs, hyps):
    return 100 * jiwer.cer([norm_for_cer(r) for r in refs], [norm_for_cer(h) or "-" for h in hyps])


def ctc_runner(repo):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(repo)
    model = Wav2Vec2ForCTC.from_pretrained(repo).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def run(clips):
        out = []
        with torch.no_grad():
            for x in clips:  # ทีละคลิป: ไม่ต้อง pad (บางรุ่นไม่มี attention_mask)
                v = proc(x, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
                ids = model(v).logits.argmax(-1).cpu()
                out.append(proc.batch_decode(ids)[0].replace("<unk>", " ").strip())
        return out

    return run, model


def main():
    sets = tests()
    siang = load_tsv(ROOT / "data" / "siang_lao" / "test.tsv")
    sets["SiangLao test"] = siang
    print({k: len(v) for k, v in sets.items()}, flush=True)
    audio = {k: [load_audio(r["file"]) for r in rows] for k, rows in sets.items()}
    total = sum(len(a) for v in audio.values() for a in v) / 16000
    results = {}

    from ct2_asr import V3Fast

    asr = V3Fast(MODELS / "full-lao-ultra3-ct2-int8")
    t = time.time()
    for k, rows in sets.items():
        hyps = asr.transcribe(audio[k], max_new=200)
        results[("Ultra3 ⚡ (เรา)", k)] = cer([r["text"] for r in rows], hyps)
        print(f"Ultra3⚡ | {k}: {results[('Ultra3 ⚡ (เรา)', k)]:.1f}%", flush=True)
    print(f"  เวลา {time.time() - t:.0f}s / เสียง {total:.0f}s", flush=True)
    del asr
    gc.collect()
    torch.cuda.empty_cache()

    for repo in CTC_RIVALS:
        try:
            run, model = ctc_runner(repo)
        except Exception as e:
            print(f"โหลด {repo} ไม่ได้: {str(e)[:150]}", flush=True)
            continue
        name = repo.split("/")[-1]
        t = time.time()
        for k, rows in sets.items():
            hyps = run(audio[k])
            results[(name, k)] = cer([r["text"] for r in rows], hyps)
            print(f"{name} | {k}: {results[(name, k)]:.1f}% | ตัวอย่าง: {hyps[0][:50]}", flush=True)
        print(f"  เวลา {time.time() - t:.0f}s", flush=True)
        del model, run
        gc.collect()
        torch.cuda.empty_cache()

    names = list(dict.fromkeys(m for m, _ in results))
    lines = ["\n=== จัดอันดับ (CER %, ต่ำ = ดี) ===", "ชุดสอบ".ljust(42) + "".join(n[:18].rjust(20) for n in names)]
    for k in sets:
        lines.append(k[:40].ljust(42) + "".join(f"{results.get((n, k), float('nan')):20.1f}" for n in names))
    msg = "\n".join(lines)
    print(msg)
    (RESULTS / "eval_rivals.txt").write_text(msg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
