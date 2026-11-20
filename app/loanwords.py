"""แก้คำต่างประเทศ (ชื่อแบรนด์/ชื่อย่อ) ที่ AI เขียนเป็นอักษรลาวไม่เหมือนเดิมทุกครั้ง
เช่น ແອັບປຼ / ແອັບປໍ / ແອັບພິ → Apple, ອາເມນສອນ / ອາເມນຊອລ໌ → Amazon, s&p 500 / smp 500 → S&P 500

ไม่ใช้การจับคู่ตรงตัว (กฎสมุดคำ) แต่เทียบ "เสียง": พับพยัญชนะเสียงเดียวกัน (ສ/ຊ, ຂ/ຄ, ທ/ຖ, ລ/ຼ/ຣ …)
ตัดวรรณยุกต์ สระสั้น/ยาวเป็นตัวเดียวกัน แล้ววัดความคล้าย ≥ 0.78 กับคำที่รู้จัก
แก้รายการคำได้ที่ data/loanwords.json  ({"Apple": ["ແອັບເປິ້ນ", ...], ...})
"""
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "loanwords.json"
THRESH = 0.84

DEFAULT = {
    "Apple": ["ແອັບເປິ້ນ", "ແອັບປຼ", "ແອັບປໍ", "ແອັບພິ", "ແອັບເປິລ"],
    "Amazon": ["ອາເມຊອນ", "ອາເມນສອນ", "ອາມັນສອນ", "ອາເມນຊອລ໌", "ອາມາຊອນ"],
    "Microsoft": ["ໄມໂຄຣຊອບ", "ໄມໂຄຊັອບ", "ໄມໂຄຊອບ", "ໄມໂຄຣຊອຟ"],
    "Nvidia": ["ເອັນວິເດຍ", "ເອນວີດເດີດ", "ເອັນເວທີດ", "ເອນວີດເຍດ", "ເອັນວີເດຍ"],
    "Tesla": ["ເທສລາ", "ເທສ໌ລາ", "ເທສະລາ"],
    "SpaceX": ["ສະເປັສເອັກ", "ເຊບເອກສ໌", "ເຊປດເອກສ໌", "ເຊບເອດ", "ສະເປດເອັກ"],
    "Google": ["ກູໂກ", "ກູເກິ້ລ", "ກູເກີ້ນ"],
    "Facebook": ["ເຟສບຸກ", "ເຟດບຸກ", "ເຟສບຸ໊ກ"],
    "YouTube": ["ຢູທູບ", "ຢູທູ້ບ"],
    "TikTok": ["ຕິກຕອກ", "ຕິກຕັອກ"],
}

# S&P 500 เขียนได้หลายแบบมาก → ใช้ regex
PATTERNS = [
    (re.compile(r"(?i)\bs\s*(?:&\s*p|n\s*p|m\s*p|m)\s*500\b"), "S&P 500"),
    (re.compile(r"(?:ເອັດ\s*)?(?:ແສນ|ຊັ້ນ|ສ)\s*[ພທ]ີ\s*(?:ຫ້າຮ້ອຍ|500)"), "S&P 500"),
]

FOLD = {**dict.fromkeys("ຂຄ", "k"), "ກ": "g", **dict.fromkeys("ສຊ", "s"), **dict.fromkeys("ທຖ", "t"), "ຕ": "T",
        **dict.fromkeys("ພຜ", "p"), "ປ": "P", **dict.fromkeys("ຟຝ", "f"), **dict.fromkeys("ມໝ", "m"),
        **dict.fromkeys("ນໜ", "n"), **dict.fromkeys("ລຼຣ", "l"), "ວ": "w", "ຫ": "", "ຮ": "h", "ອ": "",
        **dict.fromkeys("ຍຢ", "y"), "ດ": "d", "ບ": "b", "ງ": "N", "ຈ": "j",
        **dict.fromkeys("ະາັ", "a"), **dict.fromkeys("ິີ", "i"), **dict.fromkeys("ຶື", "U"), **dict.fromkeys("ຸູ", "u"),
        "ເ": "e", "ແ": "E", "ໂ": "o", **dict.fromkeys("ໄໃ", "I"), "ົ": "o", "ຳ": "am", "ໍ": "o", "ຽ": "ia",
        **dict.fromkeys("່້໊໋໌ໆ", "")}


def fold(t: str) -> str:
    return "".join(FOLD.get(c, c) for c in t)


_cache = None
_common = None


def common_words() -> set:
    """คำลาวที่พบบ่อยในคลังข้อความ (≥10 ครั้ง) → เป็นคำจริง ห้ามเปลี่ยน (กัน ປະມານ → Amazon)"""
    global _common
    if _common is None:
        from collections import Counter

        f = ROOT / "data" / "lm" / "corpus.txt"
        c = Counter(w for line in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []) for w in line.split(" "))
        known = {v for vs in DEFAULT.values() for v in vs}
        _common = {w for w, k in c.items() if k >= 10 and w not in known}
    return _common


def load() -> list[tuple[str, list[str]]]:
    global _cache
    if _cache is None:
        d = DEFAULT
        if PATH.exists():
            try:
                d = json.loads(PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        _cache = [(canon, [fold(v) for v in variants]) for canon, variants in d.items()]
    return _cache


def fix(text: str) -> tuple[str, int]:
    """คืน (ข้อความใหม่, จำนวนจุดที่แก้)"""
    original, n = text, 0
    for pat, rep in PATTERNS:
        text, k = pat.subn(rep, text)
        n += k
    words = text.split(" ")
    table = load()
    common = common_words()
    for i, w in enumerate(words):
        if not re.search(r"[຀-໿]", w) or w in common:
            continue
        fw = fold(w)
        if len(fw) < 3:
            continue
        if any(w.startswith(c) and len(c) >= 4 for c in (w[:k] for k in range(4, len(w))) if c in common):
            continue  # ขึ้นต้นด้วยคำลาวที่มีอยู่จริง เช่น ປະມານຫ້າແສນ
        best, score = None, 0.0
        for canon, fvars in table:
            for fv in fvars:
                if not 0.75 <= len(fw) / max(1, len(fv)) <= 1.34:
                    continue
                s = difflib.SequenceMatcher(None, fw, fv).ratio()
                if s > score:
                    best, score = canon, s
        if best and score >= THRESH:
            words[i] = best
            n += 1
    text = " ".join(words)
    return text, (n if text != original else 0)


def ensure_file():
    if not PATH.exists():
        PATH.write_text(json.dumps(DEFAULT, ensure_ascii=False, indent=1), encoding="utf-8")
