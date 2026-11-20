"""วัดโมเดล CTC ที่เทรนเอง (เช่น models/ctc-xlsr-v1/best) บนชุดสอบเดียวกับ eval_rivals.py
+ วัดความเร็ว GPU (ทุกชุด) และ CPU (FLEURS 100) เทียบ Ultra3 ⚡ บนเครื่องเดียวกัน

  .venv\\Scripts\\python.exe scripts\\eval_ctc_local.py models\\ctc-xlsr-v1\\best [--cpu-ultra]
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch
from common import MODELS, RESULTS, ROOT
from compare_models import load_audio, tests
from eval_rivals import cer
from train import load_tsv
from train_ctc import decode

sys.stdout.reconfigure(encoding="utf-8")


def ctc_local(path, dev: str):
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    if not Path(path).exists():  # repo บน Hugging Face (เช่น OmniASR ลาว) → ใช้ processor ของเขา
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(path)
        model = AutoModelForCTC.from_pretrained(path).eval().to(dev)

        @torch.no_grad()
        def run_hf(clips):
            out = []
            for x in clips:
                b = proc(x, sampling_rate=16000, return_tensors="pt")
                ids = model(b.input_values.to(dev)).logits.argmax(-1).cpu()
                out.append(proc.batch_decode(ids)[0].replace("<unk>", " ").strip())
            return out

        return run_hf, model

    path = Path(path)
    fe = AutoFeatureExtractor.from_pretrained(path)
    model = AutoModelForCTC.from_pretrained(path).eval().to(dev)
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in vocab.items()}

    @torch.no_grad()
    def run(clips):
        out = []
        for x in clips:  # ทีละคลิป เหมือนที่วัด SiangLao
            b = fe(x, sampling_rate=16000, return_tensors="pt")
            key = "input_features" if "input_features" in b else "input_values"
            ids = model(b[key].to(dev)).logits.argmax(-1)[0].cpu().numpy()
            out.append(decode(ids, inv))
        return out

    return run, model


def main():
    import os

    os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
    path = sys.argv[1]
    p = Path(path)
    name = p.parent.name if p.name == "best" else p.name
    sets = tests()
    sets["SiangLao test"] = load_tsv(ROOT / "data" / "siang_lao" / "test.tsv")
    audio = {k: [load_audio(r["file"]) for r in rows] for k, rows in sets.items()}
    secs = {k: sum(len(a) for a in v) / 16000 for k, v in audio.items()}
    lines = [f"=== {name} (CER %, ต่ำ = ดี) ==="]

    run, model = ctc_local(path, "cuda")
    t = time.time()
    for k, rows in sets.items():
        hyps = run(audio[k])
        c = cer([r["text"] for r in rows], hyps)
        lines.append(f"{k[:40].ljust(42)}{c:8.1f}")
        print(f"{name} | {k}: {c:.1f}% | ตัวอย่าง: {hyps[0][:50]}", flush=True)
    gpu = time.time() - t
    total = sum(secs.values())
    lines.append(f"GPU: {gpu:.0f}s / เสียง {total:.0f}s = เร็วกว่าเวลาจริง {total / gpu:.1f} เท่า")
    del run, model
    gc.collect()
    torch.cuda.empty_cache()

    # CPU: FLEURS 100 เท่านั้น (ทั้งชุดช้าเกิน)
    k = "FLEURS 100"
    torch.set_num_threads(max(1, torch.get_num_threads()))
    run, model = ctc_local(path, "cpu")
    t = time.time()
    run(audio[k])
    cpu = time.time() - t
    lines.append(f"CPU ({k}): {cpu:.0f}s / เสียง {secs[k]:.0f}s = {secs[k] / cpu:.1f} เท่า")
    del run, model
    gc.collect()

    if "--cpu-ultra" in sys.argv:
        from ct2_asr import V3Fast

        asr = V3Fast(MODELS / "full-lao-ultra3-ct2-int8", device="cpu")
        t = time.time()
        asr.transcribe(audio[k], max_new=200)
        u = time.time() - t
        lines.append(f"Ultra3 ⚡ CPU ({k}): {u:.0f}s / เสียง {secs[k]:.0f}s = {secs[k] / u:.1f} เท่า")

    msg = "\n".join(lines)
    print(msg)
    (RESULTS / f"eval_{name}.txt").write_text(msg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
