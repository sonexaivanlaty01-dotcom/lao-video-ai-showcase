"""ทำความสะอาดเฉลยข่าว RFA ก่อนตรวจคุณภาพ (เก็บของเดิมไว้ที่ all_raw.tsv เพื่อย้อนกลับได้)

ปัญหาที่เจอในเฉลย WorldSpeech/RFA:
1. ท่อนเริ่มกลางคำ เช่น "ນ. ທ່ານເວົ້າວ່າ..." หรือ "າດທີ່ວ່ານັ້ນ..." (เศษคำจากท่อนก่อน) → ตัดคำแรกทิ้ง
   และ/หรือท่อนจบกลางคำ → ไม่แก้ (ตรวจด้วย AI แล้วตัดทิ้งถ้าต่างกันมาก)
2. RFA ใช้ตัวสะกดแบบเก่า (มี ຣ เช่น ຣັຖບານ, ສາທາຣະນະ) ต่างจากที่ทีมใช้ (ລັດຖະບານ, ສາທາລະນະ)
   → ตัดท่อนที่มีตัว ຣ ทิ้ง กัน AI เรียนตัวสะกดเก่า (นับไว้ในรายงาน)

  .venv\\Scripts\\python.exe scripts\\clean_rfa.py
"""
import csv
import re
import shutil
import sys

from common import ROOT

sys.stdout.reconfigure(encoding="utf-8")
D = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/extra/rfa")  # เช่น data/extra/rfa2
DEP = re.compile(r"^[ະ-ຼ່-ໍ]")  # สระ/วรรณยุกต์ที่ขึ้นต้นคำไม่ได้


def fix(text: str) -> str | None:
    t = text.strip()
    if DEP.match(t):  # เริ่มด้วยเศษสระ → ตัดถึงช่องว่างแรก
        t = t.split(" ", 1)[1] if " " in t else ""
    first = t.split(" ", 1)[0]
    if len(first) <= 3 and first.endswith(".") and " " in t:  # เช่น "ນ. ທ່ານ..." เศษท้ายประโยคก่อน
        t = t.split(" ", 1)[1]
    return t.strip() or None


def main():
    raw = D / "all_raw.tsv"
    if not raw.exists():
        shutil.copy(D / "all.tsv", raw)
    with open(raw, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out, old, empty, trimmed = [], 0, 0, 0
    for r in rows:
        if "ຣ" in r["text"]:
            old += 1
            continue
        t = fix(r["text"])
        if not t:
            empty += 1
            continue
        trimmed += t != r["text"].strip()
        r["text"] = t
        out.append(r)
    with open(D / "all.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    hrs = lambda rs: sum(float(x["duration"]) for x in rs) / 3600
    print(f"ทั้งหมด {len(rows)} ท่อน ({hrs(rows):.2f} ชม.) → เหลือ {len(out)} ({hrs(out):.2f} ชม.)")
    print(f"  ตัดเพราะตัวสะกดเก่า (ຣ): {old} | ตัดเศษคำต้นท่อน: {trimmed} | เหลือว่าง: {empty}")


if __name__ == "__main__":
    main()
