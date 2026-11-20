"""วัดผล Whisper (ตัวเดิม หรือ ตัวที่เทรนแล้ว) กับเสียงลาว FLEURS ชุด test

ตัวอย่าง:
  # ตัวเดิมก่อนเทรน
  .venv\\Scripts\\python.exe scripts\\evaluate.py --n 100
  # ตัวที่เทรนแล้ว (LoRA adapter)
  .venv\\Scripts\\python.exe scripts\\evaluate.py --n 100 --adapter models\\lora-small-lo\\best

CER = % ตัวอักษรที่ผิด (นับสระและวรรณยุกต์ด้วย) ยิ่งต่ำยิ่งดี
"""
import argparse
import csv
import time
from pathlib import Path

import common  # noqa: F401  (ตั้งค่า CUDA path)
import jiwer
import librosa
import torch
from common import RESULTS, load_fleurs, norm_for_cer
from thai2lao import thai_to_lao
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# GTX 16xx (เช่น GTX 1650) คำนวณ fp16 แล้วได้ค่าเพี้ยน (ออกมาเป็น "!!!!") จึงใช้ fp32 เป็นค่าเริ่มต้น
DTYPE = torch.float32


DEVICE = "cuda"
# BuzzASR/lao ใส่คำสั่งภาษาไว้ในตัวโมเดลแล้ว ไม่ต้องบอก language เอง
BAKED = {"BuzzASR/lao": dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2)}


def load_model(base: str, adapter: str | None):
    if base in BAKED or Path(base, "tokenizer.json").exists():  # โมเดลที่มีตารางคำของตัวเอง
        processor = WhisperProcessor.from_pretrained(base)
    else:
        processor = WhisperProcessor.from_pretrained(base, language="lao", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=DTYPE)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.base_name = base
    return processor, model.to(DEVICE).eval()


BEAMS = 1  # ตั้งจาก --beams: 1 = เลือกคำที่ดีที่สุดทีละคำ, 5 = คิด 5 ทางแล้วเลือกประโยคที่ดีที่สุด


@torch.no_grad()
def transcribe(processor, model, files, batch=8):
    out = []
    gen = dict(BAKED.get(model.base_name, dict(language="lo", task="transcribe")))
    if BEAMS > 1:
        gen.update(num_beams=BEAMS)
        batch = max(1, batch // 2)  # beam ใช้หน่วยความจำมากขึ้น
    for i in range(0, len(files), batch):
        audios = [librosa.load(str(f), sr=16000)[0] for f in files[i : i + batch]]
        feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        ids = model.generate(feats.to(DEVICE, DTYPE), max_new_tokens=440, **gen)
        out += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        print(f"  ... {min(i + batch, len(files))}/{len(files)} คลิป", flush=True)
    return [t.strip() for t in out]


def score(rows, hyps):
    refs = [norm_for_cer(r["text"]) for r in rows]
    raw = jiwer.cer(refs, [norm_for_cer(h) for h in hyps])
    conv = jiwer.cer(refs, [norm_for_cer(thai_to_lao(h)) for h in hyps])
    thai_share = sum(any("฀" <= c <= "๿" for c in h) for h in hyps) / max(len(hyps), 1)
    return raw, conv, thai_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--name", default=None, help="ชื่อไฟล์ผลลัพธ์")
    ap.add_argument("--fp16", action="store_true", help="ใช้ fp16 (เฉพาะการ์ดจอที่รองรับดี เช่น T4/RTX)")
    ap.add_argument("--device", default="cuda", help="cuda หรือ cpu")
    ap.add_argument("--beams", type=int, default=1, help="จำนวนทางที่ให้ AI คิด (beam search) เช่น 5")
    args = ap.parse_args()
    global DTYPE, DEVICE, BEAMS
    BEAMS = args.beams
    if args.fp16:
        DTYPE = torch.float16
    DEVICE = args.device

    name = args.name or ("finetuned" if args.adapter else "base")
    rows = load_fleurs(args.split, args.n)
    processor, model = load_model(args.base, args.adapter)
    print(f"วัดผล [{name}] กับ {len(rows)} คลิป ({args.split})")

    t0 = time.time()
    hyps = transcribe(processor, model, [r["file"] for r in rows])
    raw, conv, thai_share = score(rows, hyps)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"eval_{name}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "ข้อความถูก", "AI ฟังได้", "CER"])
        for r, h in zip(rows, hyps):
            w.writerow([r["file"].name, r["text"], h, f"{jiwer.cer(norm_for_cer(r['text']), norm_for_cer(h)):.2f}"])

    print("\n================ ผลรวม ================")
    print(f"CER (ตรงตัว)               : {raw*100:.1f}%")
    print(f"CER (ถ้าแปลงตัวไทย→ลาวก่อน): {conv*100:.1f}%")
    print(f"ประโยคที่ยังมีตัวอักษรไทยปน : {thai_share*100:.0f}%")
    print(f"ใช้เวลา {time.time()-t0:.0f} วินาที → {path}")
    for r, h in list(zip(rows, hyps))[:3]:
        print("\n  ถูก:", r["text"][:80])
        print("  AI :", h[:80])


if __name__ == "__main__":
    main()
