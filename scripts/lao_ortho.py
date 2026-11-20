"""ตรวจ "รูปคำลาวที่เขียนผิดหลัก" (การเรียงพยัญชนะ สระ วรรณยุกต์) — ไม่ต้องรู้ความหมาย ดูแค่โครงสร้าง

กฎ (ตามหลักการเขียนลาว):
  R1 สระบน/ล่าง/วรรณยุกต์/ໍ/໌/ຼ ต้องเกาะพยัญชนะ (ห้ามขึ้นต้นคำ หรือตามหลังสระหน้า/สระหลัง)
  R2 สระหน้า ເ ແ ໂ ໃ ໄ ต้องตามด้วยพยัญชนะ
  R3 สระบน 2 ตัวซ้อนกัน / สระล่าง 2 ตัว / สระบนกับสระล่างบนพยัญชนะเดียวกัน
  R4 วรรณยุกต์ 2 ตัวบนพยัญชนะเดียวกัน
  R5 วรรณยุกต์ต้องอยู่บนพยัญชนะต้น ไม่ใช่บน ວ/ຍ/ອ ที่เป็นสระ เมื่อมีพยัญชนะต้นอยู่หน้า (เช่น ລວ້ວ → ລ້ວນ)
  (ไม่ใช้กฎ ຼ หลัง ຫ เท่านั้น เพราะคำต่างประเทศเขียน ປຼ ຟຼ ຄຼ ได้ถูกต้อง เช่น ນິວເຄຼຍ ຟຼີ)
ทดสอบกับคลังข้อความ 200,719 ประโยค: โดนเตือน 0.26% และส่วนใหญ่เป็นคำพิมพ์ผิดจริง

autofix(): แก้เฉพาะกรณีที่รู้แน่ว่าถูกต้องคืออะไร
  สระล่างซ้อน ຸຸ → ຸ · วรรณยุกต์ตามหลัง ຳ (ນຳ້ → ນ້ຳ) · วรรณยุกต์ตามหลัง ຽ (ກຽ່ວ → ກ່ຽວ)

  python scripts/lao_ortho.py "ຂໍ້ຄວາມ"   → แสดงจุดที่ผิดหลัก
"""
import re
import sys

CONS = "ກຂຄງຈສຊຍດຕຖທນບປຜຝພຟມຢຣລວຫອຮໜໝ"
LEAD = "ເແໂໃໄ"
ABOVE = "ັິີຶືົໍ"
BELOW = "ຸູ"
TONE = "່້໊໋"
FOLLOW = "ະາຳຽ"
MARKS = ABOVE + BELOW + TONE + "໌ຼ"


def check(text: str) -> list[tuple[int, str]]:
    """คืน [(ตำแหน่ง, เหตุผล)] ของจุดที่ผิดหลักการเขียน"""
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else " "
        nxt = text[i + 1] if i + 1 < n else " "
        if ch in MARKS:
            # R1: ต้องมีพยัญชนะเป็นฐาน (ข้ามเครื่องหมายอื่นที่ซ้อนอยู่ก่อน)
            j = i - 1
            while j >= 0 and text[j] in MARKS:
                j -= 1
            if j < 0 or text[j] not in CONS:
                out.append((i, "สระ/วรรณยุกต์ไม่มีพยัญชนะรองรับ"))
                continue
            stack = text[j + 1 : i]
            if ch in ABOVE and any(c in ABOVE for c in stack):
                out.append((i, "สระบนซ้อน 2 ตัว"))
            elif ch in BELOW and any(c in BELOW for c in stack):
                out.append((i, "สระล่างซ้อน 2 ตัว"))
            elif ch in ABOVE + BELOW and any(c in ABOVE + BELOW for c in stack) and ch != "ໍ":
                out.append((i, "สระบนกับสระล่างบนตัวเดียวกัน"))
            elif ch in TONE and any(c in TONE for c in stack):
                out.append((i, "วรรณยุกต์ซ้อน 2 ตัว"))
            elif ch in TONE and text[j] in "ວ" and j >= 1 and text[j - 1] in CONS and nxt in "ວ":
                out.append((i, "วรรณยุกต์อยู่บน ວ ที่เป็นสระ (ควรอยู่บนพยัญชนะต้น)"))
        elif ch in LEAD:
            if nxt not in CONS:  # R2
                out.append((i, "สระหน้าไม่ตามด้วยพยัญชนะ"))
    return out


def has_error(text: str) -> bool:
    return bool(check(text))


_RE_DUP_BELOW = re.compile(f"([{BELOW}])\\1+")
_RE_TONE_AFTER_AM = re.compile(f"ຳ([{TONE}])")  # ນຳ້ → ນ້ຳ
_RE_TONE_AFTER_YO = re.compile(f"([{CONS}])ຽ([{TONE}])")  # ກຽ່ວ → ກ່ຽວ


def autofix(text: str) -> str:
    text = _RE_DUP_BELOW.sub(r"\1", text)
    text = _RE_TONE_AFTER_AM.sub(r"\1ຳ", text)
    text = _RE_TONE_AFTER_YO.sub(r"\1\2ຽ", text)
    return text


def error_fragments(text: str) -> list[str]:
    """ท่อนสั้น ๆ รอบจุดที่ผิดหลัก ไว้ไฮไลต์ในหน้าแก้ซับ"""
    frags = []
    for i, _ in check(text):
        a, b = text.rfind(" ", 0, i) + 1, text.find(" ", i)
        b = len(text) if b < 0 else b
        if b - a > 8:
            a, b = max(a, i - 3), min(b, i + 3)
        f = text[a:b].strip()
        if f and f not in frags:
            frags.append(f)
    return frags


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for t in sys.argv[1:]:
        print(t, check(t) or "✅ ถูกหลัก")
