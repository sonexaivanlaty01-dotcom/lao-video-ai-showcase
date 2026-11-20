"""คัดข่าว RFA รอบสอง หลัง check_extra.py (ใช้ checked.tsv ที่มีคำตอบของโมเดลตรวจ)

1. ตัวสะกดเก่าที่ไม่มี ຣ แต่ยังเก่า (ກະສິກັມ, ຜແນກ, ຜລິຕ, ກິຈ, ວັທ) → ตัดท่อนทิ้ง
2. ขอบท่อนเพี้ยน: คำแรก/คำสุดท้ายของเฉลยไม่อยู่ในหัว/ท้ายของสิ่งที่โมเดลได้ยิน
   (เช่น "ອນ ເຄີຍມີ..." / "ດ ແລະ ຊອກຫາ...") → ตัดคำนั้นออก แล้วคิด CER ใหม่
3. เก็บเฉพาะ CER ใหม่ ≤ --max-cer (ค่าเริ่ม 0.4)

ไม่ลบอะไร: keep.tsv เดิมถูกย้ายไป keep_v1.tsv (ย้อนกลับได้)
  .venv\\Scripts\\python.exe scripts\\refine_rfa.py data\\extra\\rfa
"""
import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

from check_extra import cer
from common import norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
OLD = re.compile(r"(?<!ເຕ)ັມ|ຜລ|ຜແ|ຕພ|ກິຈ|ວັທ|ປະເທສ|ເສຖ|໌")  # v6: + ປະເທສ (ປະເທດ), ເສຖະກິດ (ເສດຖະກິດ), ໌


def edge_trim(text: str, hyp: str) -> str:
    toks, h = text.split(" "), norm_for_cer(hyp)
    if len(toks) > 2:
        f = norm_for_cer(toks[0])
        if f and len(f) <= 3 and f not in h[: len(f) + 4]:  # เฉพาะเศษคำสั้น ๆ ไม่ตัดคำจริงที่สะกดต่างกัน
            toks = toks[1:]
    if len(toks) > 2:
        last = norm_for_cer(toks[-1])
        if last and len(last) <= 3 and last not in h[-(len(last) + 4) :]:
            toks = toks[:-1]
    return " ".join(toks).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--max-cer", type=float, default=0.5)
    a = ap.parse_args()
    d = Path(a.folder)
    rows = list(csv.DictReader(open(d / "checked.tsv", encoding="utf-8"), delimiter="\t"))
    if not (d / "keep_v1.tsv").exists() and (d / "keep.tsv").exists():
        shutil.copy(d / "keep.tsv", d / "keep_v1.tsv")
    keep, old, trimmed = [], 0, 0
    for r in rows:
        if OLD.search(r["text"]):
            old += 1
            continue
        t = edge_trim(r["text"], r["v3_text"])
        trimmed += t != r["text"]
        if t and cer(t, r["v3_text"]) <= a.max_cer:
            keep.append([r["file"], t, r["duration"]])
    with open(d / "keep.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["file", "text", "duration"])
        w.writerows(keep)
    hrs = sum(float(k[2]) for k in keep) / 3600
    print(f"{d.name}: {len(rows)} → ผ่าน {len(keep)} ท่อน = {hrs:.2f} ชม. | ตัวสะกดเก่า {old} | ตัดคำขอบ {trimmed}")


if __name__ == "__main__":
    main()
