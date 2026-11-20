"""ตรวจคุณภาพเฉลยของชุดข้อมูลเพิ่ม: ให้ AI v3 ฟังทุกคลิป แล้วเทียบกับเฉลย (CER ต่อคลิป)

  .venv\\Scripts\\python.exe scripts\\check_extra.py data\\extra\\kittzk [--max-cer 0.5]

ความคิด: v3 ผิดเฉลี่ยประมาณ 10–20% ถ้าคลิปไหน v3 กับเฉลยต่างกันมากผิดปกติ (เช่น >50%)
มักแปลว่า "เฉลยผิด / เสียงไม่ตรงข้อความ / เป็นเพลง / ไม่ใช่ภาษาลาว" → ตัดทิ้ง

ผลลัพธ์ (ไม่แก้ไฟล์เดิม all.tsv เพื่อย้อนกลับได้เสมอ):
  checked.tsv  ทุกคลิป + v3_text + cer
  keep.tsv     เฉพาะคลิปที่ผ่าน (ใช้เทรน: train.py --extra data\\extra\\kittzk\\keep.tsv)
  sample.txt   สุ่มตัวอย่างให้คนฟังตรวจ
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import jiwer
import soundfile as sf
from common import norm_for_cer
from v3_asr import V3

sys.stdout.reconfigure(encoding="utf-8")


def cer(ref: str, hyp: str) -> float:
    r, h = norm_for_cer(ref), norm_for_cer(hyp)
    if not r:
        return 1.0
    return min(jiwer.cer(r, h) if h else 1.0, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--max-cer", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=None, help="โฟลเดอร์โมเดลที่ใช้ตรวจ (ไม่ใส่ = v3) เช่น models/full-lao-ultra")
    a = ap.parse_args()
    d = Path(a.folder)
    with open(d / "all.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    print(f"{d.name}: {len(rows)} คลิป → ให้ v3 ฟัง...", flush=True)
    m = V3(model_dir=a.model)
    order = sorted(rows, key=lambda r: len(r["text"]))  # จัดคลิปความยาวใกล้กันไว้ชุดเดียวกัน → เร็วขึ้น
    for i in range(0, len(order), 64):
        chunk = order[i : i + 64]
        audios = [sf.read(d / r["file"], dtype="float32")[0] for r in chunk]
        # ตารางคำลาวของ v3 ใช้ ~35 ชิ้นต่อประโยค 30 วิ → จำกัดตามความยาวเฉลย กันการเขียนวนไม่จบ (ช้ามาก)
        longest = max(len(r["text"]) for r in chunk)
        for r, t in zip(chunk, m.transcribe(audios, batch=a.batch, max_new=min(160, 24 + longest // 2))):
            r["v3_text"], r["cer"] = t, f"{cer(r['text'], t):.3f}"
        print(f"  {min(i + 64, len(rows))}/{len(rows)}", flush=True)

    cols = list(rows[0].keys())
    with open(d / "checked.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    keep = [r for r in rows if float(r["cer"]) <= a.max_cer]
    with open(d / "keep.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["file", "text", "duration"])
        w.writerows([[r["file"], r["text"], r["duration"]] for r in keep])

    cs = sorted(float(r["cer"]) for r in rows)
    hrs = lambda rs: sum(float(r["duration"]) for r in rs) / 3600
    print(f"\nv3 เทียบเฉลย: ค่ากลาง CER {cs[len(cs)//2]*100:.1f}% | ≤20%: {sum(c<=0.2 for c in cs)} | 20–50%: "
          f"{sum(0.2<c<=0.5 for c in cs)} | >50%: {sum(c>0.5 for c in cs)}")
    print(f"✅ ผ่าน {len(keep)}/{len(rows)} คลิป = {hrs(keep):.2f} จาก {hrs(rows):.2f} ชม. → {d/'keep.tsv'}")

    random.seed(0)
    good = random.sample(keep, min(8, len(keep)))
    bad = sorted([r for r in rows if r not in keep], key=lambda r: -float(r["cer"]))[:8]
    with open(d / "sample.txt", "w", encoding="utf-8") as fh:
        for title, rs in (("ผ่าน (สุ่ม)", good), ("ไม่ผ่าน (ต่างกันมากสุด)", bad)):
            fh.write(f"===== {title} =====\n")
            for r in rs:
                fh.write(f"{r['file']}  CER {float(r['cer'])*100:.0f}%\n  เฉลย: {r['text']}\n  v3  : {r['v3_text']}\n\n")
    print(f"ตัวอย่างให้ฟัง: {d/'sample.txt'}")


if __name__ == "__main__":
    main()
