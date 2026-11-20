"""แปลงตัวอักษรไทย → ตัวอักษรลาว (แบบหยาบ ตัวต่อตัว)

ทำไมต้องมี: Whisper ฟังเสียงลาวแล้วมักเขียนออกมาเป็น "ตัวอักษรไทย"
ตัวไทยและตัวลาวมาจากรากเดียวกัน ใน Unicode ตำแหน่งห่างกัน 0x80 พอดีหลายตัว
จึงแปลงคร่าวๆ ได้ เพื่อดูว่า "เสียง" ที่ AI ฟังได้ ถูกจริงแค่ไหน

วิธีรัน (คิดคะแนนใหม่จากผล baseline):
  .venv\\Scripts\\python.exe scripts\\thai2lao.py results\\baseline_small.csv
"""
import csv
import sys
import unicodedata

import jiwer

# ตัวไทยที่ลาวไม่มี → ตัวลาวที่ออกเสียงใกล้สุด
SPECIAL = {
    "ฃ": "ຂ", "ฅ": "ຄ", "ฆ": "ຄ", "ฉ": "ສ", "ฌ": "ຊ", "ซ": "ຊ", "ญ": "ຍ",
    "ฎ": "ດ", "ฏ": "ຕ", "ฐ": "ຖ", "ฑ": "ທ", "ฒ": "ທ", "ณ": "ນ", "ธ": "ທ",
    "ภ": "ພ", "ศ": "ສ", "ษ": "ສ", "ร": "ລ", "ฬ": "ລ", "ฤ": "ລຶ", "ฦ": "ລຶ",
    "ย": "ຍ", "็": "",
}


def thai_to_lao(text: str) -> str:
    out = []
    for ch in text:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif "฀" <= ch <= "๿":
            lao = chr(ord(ch) + 0x80)
            try:
                unicodedata.name(lao)
                out.append(lao)
            except ValueError:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def norm(s):
    return "".join(ch for ch in s if ch.isalnum())


def main(path):
    refs, raw, conv = [], [], []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ref = norm(row["reference (ข้อความถูก)"])
            hyp = row["whisper (AI ฟังได้)"]
            refs.append(ref)
            raw.append(norm(hyp) or " ")
            conv.append(norm(thai_to_lao(hyp)) or " ")
    print(f"CER ก่อนแปลงตัวอักษร: {jiwer.cer(refs, raw)*100:.1f}%")
    print(f"CER หลังแปลงไทย→ลาว: {jiwer.cer(refs, conv)*100:.1f}%")
    print("\nตัวอย่าง 3 ประโยคแรก:")
    for r, c in list(zip(refs, conv))[:3]:
        print("  ถูก :", r[:70])
        print("  AI  :", c[:70])
        print()


if __name__ == "__main__":
    main(sys.argv[1])
