"""วัดผลว่า Whisper ฟังภาษาลาวได้แม่นแค่ไหน (Baseline)

ใช้ชุดเสียง FLEURS ภาษาลาว (test) ที่โหลดไว้ใน data/fleurs_lo
ผลลัพธ์: results/baseline_<model>.csv และคะแนน CER/WER

CER = อัตราตัวอักษรผิด  (ภาษาลาวไม่มีเว้นวรรคระหว่างคำ จึงดู CER เป็นหลัก)
WER = อัตราคำผิด

วิธีรัน:
  .venv\\Scripts\\python.exe scripts\\baseline.py --model small --n 30
"""
import argparse
import csv
import os
import time
from pathlib import Path

# ให้ faster-whisper ใช้ไฟล์ CUDA (cuBLAS/cuDNN) ที่มากับ PyTorch
import torch

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
os.environ["PATH"] = str(Path(torch.__file__).parent / "lib") + os.pathsep + os.environ["PATH"]

import jiwer
import librosa
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fleurs_lo"
OUT = ROOT / "results"


def load_fleurs(n):
    """อ่าน test.tsv: id, file, raw text, normalized text, ..."""
    rows = []
    with open(DATA / "test.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            wav = next(DATA.rglob(parts[1]), None)
            if wav:
                rows.append({"file": wav, "ref": parts[2]})
            if len(rows) >= n:
                break
    return rows


def norm(s):
    # ตัดเว้นวรรคและเครื่องหมายวรรคตอน เพื่อเทียบเฉพาะตัวอักษรลาว
    return "".join(ch for ch in s if ch.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="tiny / base / small / medium")
    ap.add_argument("--n", type=int, default=30, help="จำนวนคลิปที่ใช้ทดสอบ")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = load_fleurs(args.n)
    print(f"ทดสอบ {len(rows)} คลิป ด้วย Whisper {args.model} บน {args.device}")

    compute = "int8_float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute)

    OUT.mkdir(exist_ok=True)
    refs, hyps = [], []
    t0 = time.time()
    with open(OUT / f"baseline_{args.model}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "reference (ข้อความถูก)", "whisper (AI ฟังได้)", "CER"])
        for i, r in enumerate(rows, 1):
            audio, _ = librosa.load(str(r["file"]), sr=16000, mono=True)
            segs, _ = model.transcribe(audio, language="lo", beam_size=5)
            hyp = "".join(s.text for s in segs).strip()
            cer = jiwer.cer(norm(r["ref"]), norm(hyp) or " ")
            refs.append(norm(r["ref"]))
            hyps.append(norm(hyp) or " ")
            w.writerow([r["file"].name, r["ref"], hyp, f"{cer:.2f}"])
            print(f"[{i}/{len(rows)}] CER={cer:.2f}  AI: {hyp[:60]}")

    total_cer = jiwer.cer(refs, hyps)
    print("\n================ ผลรวม ================")
    print(f"Whisper {args.model}: CER = {total_cer*100:.1f}%  (ยิ่งต่ำยิ่งดี)")
    print(f"ใช้เวลา {time.time()-t0:.0f} วินาที")
    print(f"ไฟล์รายละเอียด: {OUT / f'baseline_{args.model}.csv'}")


if __name__ == "__main__":
    main()
