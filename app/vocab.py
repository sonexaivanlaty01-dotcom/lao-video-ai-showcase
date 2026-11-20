"""สมุดคำเฉพาะของทีม: แก้คำที่ AI ฟังผิดซ้ำๆ ให้อัตโนมัติ + เรียนรู้จากซับที่ทีมแก้

ไฟล์: data/vocab.json
  rules:       กฎที่ใช้จริง  {wrong, right, enabled, count, source: manual|auto, added}
  suggestions: คำที่ระบบเห็นว่าทีมแก้บ่อย รอให้คนกดยืนยัน {wrong, right, count, examples}

ทำไมไม่ "บอกใบ้ AI ก่อนฟัง" (prompt): ทดลองแล้ว แก้ ເລຂາໄຊ ได้ แต่ทำให้คำอื่นเพี้ยน (AI→ເອອາຍ)
และ FLEURS แย่ลง 18.5→18.7% → ใช้การแทนคำหลังฟังแทน ซึ่งคุมได้แน่นอน
"""
import difflib
import json
import re
import threading
import time
from pathlib import Path

_lock = threading.Lock()
MIN_WRONG = 3  # คำผิดสั้นกว่านี้อันตราย (เช่น ໃສ່→ໄຊ จะไปแก้คำอื่นที่ถูกอยู่แล้ว)


class Vocab:
    def __init__(self, path: Path):
        self.path = Path(path)

    # ---------- ไฟล์ ----------
    def load(self) -> dict:
        if self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            d = {}
        d.setdefault("rules", [])
        d.setdefault("suggestions", [])
        return d

    def save(self, d: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # ---------- ใช้กฎ ----------
    def apply(self, text: str, d: dict | None = None) -> tuple[str, int]:
        """แทนคำผิดด้วยคำถูก (คำผิดที่ยาวกว่าแทนก่อน) คืน (ข้อความใหม่, จำนวนที่แก้)"""
        d = d or self.load()
        n = 0
        for r in sorted((r for r in d["rules"] if r.get("enabled", True)), key=lambda r: -len(r["wrong"])):
            if r["wrong"] and r["wrong"] in text:
                c = text.count(r["wrong"])
                text = text.replace(r["wrong"], r["right"])
                n += c
        return text, n

    def apply_segments(self, segs: list[dict]) -> int:
        """แก้ทุกบรรทัด + นับว่ากฎไหนถูกใช้กี่ครั้ง (ไว้ดูว่ากฎไหนมีประโยชน์)"""
        with _lock:
            d = self.load()
            total = 0
            for s in segs:
                for r in sorted((r for r in d["rules"] if r.get("enabled", True)), key=lambda r: -len(r["wrong"])):
                    c = s["text"].count(r["wrong"]) if r["wrong"] else 0
                    if c:
                        s["text"] = s["text"].replace(r["wrong"], r["right"])
                        r["count"] = r.get("count", 0) + c
                        total += c
            if total:
                self.save(d)
        # คำต่างประเทศ (ชื่อแบรนด์ / S&P 500) ที่ AI สะกดไม่เหมือนเดิมทุกครั้ง → เทียบเสียงแล้วแก้เป็นรูปเดียว
        try:
            from loanwords import fix

            for s in segs:
                s["text"], c = fix(s["text"])
                total += c
        except Exception as e:
            print("⚠️ ข้ามแก้คำต่างประเทศ:", e)
        return total

    def add_rule(self, wrong: str, right: str, source: str = "manual") -> dict:
        wrong, right = wrong.strip(), right.strip()
        if len(wrong) < 2 or not right or wrong == right:
            raise ValueError("คำผิดต้องยาวอย่างน้อย 2 ตัวอักษร และต่างจากคำถูก")
        with _lock:
            d = self.load()
            d["rules"] = [r for r in d["rules"] if r["wrong"] != wrong]
            d["rules"].append({"wrong": wrong, "right": right, "enabled": True, "count": 0, "source": source, "added": time.strftime("%Y-%m-%d")})
            d["suggestions"] = [s for s in d["suggestions"] if s["wrong"] != wrong]
            self.save(d)
        return d

    # ---------- เรียนรู้จากซับที่แก้แล้ว ----------
    def learn(self, segs: list[dict], tokenize, is_common_word=None, clean=None) -> int:
        """เทียบข้อความ AI เดิม กับที่คนแก้ → จุดที่ถูกแทนที่ เก็บเป็น "คำแนะนำ" (ยังไม่ใช้จนกว่าคนกดยืนยัน)

        tokenize:       หั่นข้อความเป็นชิ้นคำ (ตารางคำลาวของ v3 → ระดับคำ/พยางค์ ไม่ตัดกลางพยางค์)
        is_common_word: คืน True ถ้าคำนั้นเป็นคำปกติที่ใช้บ่อย (เช่น ແລະ, ຫຼາຍ) → ต้องขยายบริบท
                        ไม่งั้นกฎ "ແລະ → ແອັດ" จะไปแก้ ແລະ ทุกที่ในซับ
        clean:          ล้างวรรคตอน/สะกดก่อนเทียบ"""
        d0 = self.load()
        common_ = is_common_word or (lambda w: False)
        clean = clean or (lambda t: t)

        def risky(w: str) -> bool:
            return len(w) < MIN_WRONG or common_(w)

        pairs = []
        for s in segs:
            # เทียบกับข้อความ "หลังใช้กฎเดิมแล้ว" → จุดที่กฎเดิมแก้ได้อยู่แล้ว ไม่ต้องแนะนำซ้ำ
            ai = clean(self.apply((s.get("ai_text") or "").strip(), d0)[0])
            fixed = clean((s.get("text") or "").strip())
            if not ai or not fixed or ai == fixed:
                continue
            a, b = tokenize(ai), tokenize(fixed)
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != "replace":
                    continue
                # ขยายบริบทซ้าย-ขวา (ส่วนที่ตรงกัน) จนคำผิดไม่ใช่คำทั่วไป/ยาวพอ — สูงสุด 3 รอบ
                for _ in range(3):
                    if not risky("".join(a[i1:i2]).strip()):
                        break
                    grown = False
                    if i1 > 0 and j1 > 0 and a[i1 - 1] == b[j1 - 1]:
                        i1, j1, grown = i1 - 1, j1 - 1, True
                    if i2 < len(a) and j2 < len(b) and a[i2] == b[j2]:
                        i2, j2, grown = i2 + 1, j2 + 1, True
                    if not grown:
                        break
                w, r = "".join(a[i1:i2]).strip(), "".join(b[j1:j2]).strip()
                if not r or w == r or risky(w):
                    continue
                if w.replace(" ", "") == r.replace(" ", ""):  # ต่างกันแค่เว้นวรรค
                    continue
                if re.fullmatch(r"[\s\d.,]+", w):
                    continue
                pairs.append((w, r, fixed))
        if not pairs:
            return 0
        with _lock:
            d = self.load()
            ruled = {r["wrong"] for r in d["rules"]}
            sug = {(s["wrong"], s["right"]): s for s in d["suggestions"]}
            for w, r, ex in pairs:
                if w in ruled:
                    continue
                s = sug.setdefault((w, r), {"wrong": w, "right": r, "count": 0, "examples": []})
                s["count"] += 1
                if len(s["examples"]) < 3 and ex not in s["examples"]:
                    s["examples"].append(ex[:80])
            d["suggestions"] = sorted(sug.values(), key=lambda s: -s["count"])[:200]
            self.save(d)
        return len(pairs)
