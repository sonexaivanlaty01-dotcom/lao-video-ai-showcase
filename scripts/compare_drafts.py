"""เทียบว่า AI รุ่นไหนถอดคลิปนี้แม่นกว่า: เอาร่างของแต่ละรุ่น (ที่เคยกดสร้างซับ) มาเทียบกับซับที่คนแก้แล้ว

  .venv\\Scripts\\python.exe scripts\\compare_drafts.py Screen_Recording_20261115_035812_Messenger
  (ใส่ชื่อคลิปบางส่วนก็ได้ — รวมทุกครั้งที่อัปโหลดคลิปเดียวกัน เช่น ..._1794737053, ..._1794772305)

เฉลย = ซับล่าสุดของคลิปที่ "กด ✅ ตรวจเสร็จ" (ถ้ามีหลายอัน ใช้อันที่แก้มากที่สุด)
ร่าง  = data/subs/history/*.lo.json + ร่าง AI ของซับปัจจุบัน (ai_full) → ใช้ร่างล่าสุดของแต่ละรุ่น
⚠️ ร่างที่ใช้เป็นจุดเริ่มแก้จะได้เปรียบ (เฉลยติดคำของร่างนั้น) → สลับรุ่นที่ใช้เริ่มแก้ในแต่ละคลิปจะแฟร์ที่สุด
"""
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import jiwer
from common import ROOT, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
SUBS = ROOT / "data" / "subs"


def base(stem: str) -> str:
    return re.sub(r"_\d{9,}$", "", stem)


def full(doc) -> str:
    return " ".join(s.get("text", "").strip() for s in doc["segments"])


def raw(doc) -> str:
    """ร่างดิบของ AI (ai_text ก่อนสมุดคำ/คนแก้) — ai_full เก็บหลังสมุดคำแล้ว จึงไม่ใช้"""
    t = " ".join((s.get("ai_text") or "").strip() for s in doc["segments"]).strip()
    return t or doc.get("ai_full", "")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    docs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in SUBS.glob("*.lo.json") if key in p.stem]
    if not docs:
        sys.exit(f"ไม่เจอซับที่มีชื่อ '{key}'")
    groups = {}
    for p, d in docs:
        groups.setdefault(base(p.stem.removesuffix(".lo")), []).append((p, d))
    for name, items in groups.items():
        corrected = [(p, d) for p, d in items if d.get("status") == "corrected"]
        pool = corrected or items
        ref_p, ref_d = max(pool, key=lambda x: sum(s.get("text") != s.get("ai_text") for s in x[1]["segments"]))
        ref = full(ref_d)
        n_edit = sum(s.get("text") != s.get("ai_text") for s in ref_d["segments"])
        print(f"\n=== {name} ===")
        print(f"เฉลย: {ref_p.name} | สถานะ {ref_d.get('status')} | คนแก้ {n_edit} บรรทัด | เริ่มแก้จากร่าง {ref_d.get('engine')}")
        if ref_d.get("status") != "corrected":
            print("⚠️ ยังไม่ได้กด ✅ ตรวจเสร็จ — ตัวเลขข้างล่างเชื่อถือได้น้อย")
        drafts = {}
        stems = {p.stem.removesuffix(".lo") for p, _ in items}
        for p, d in items:
            if raw(d):
                drafts.setdefault(d.get("engine"), []).append((p.stat().st_mtime, raw(d), p.name))
        for h in (SUBS / "history").glob("*.lo.json"):
            stem = h.name.split(".")[0]
            if stem in stems or base(stem) == name:
                d = json.loads(h.read_text(encoding="utf-8"))
                if raw(d):
                    drafts.setdefault(d.get("engine"), []).append((h.stat().st_mtime, raw(d), h.name))
        rows = []
        for eng, lst in drafts.items():
            _, hyp, src = max(lst)
            r, h = norm_for_cer(ref), norm_for_cer(hyp) or "-"
            cer = 100 * jiwer.cer(r, h)
            rows.append((cer, eng, src, hyp))
        for cer, eng, src, _ in sorted(rows):
            star = " ← ร่างตั้งต้น (ได้เปรียบ)" if eng == ref_d.get("engine") else ""
            print(f"  {eng:22s} ผิด {cer:5.1f}%{star}")
        if rows:
            best = min(rows)
            print(f"\n  ตัวอย่างจุดที่ {best[1]} ยังต่างจากเฉลย:")
            sm = SequenceMatcher(None, ref, best[3], autojunk=False)
            k = 0
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != "equal" and k < 10:
                    print(f"    เฉลย '{ref[i1:i2]}'  →  AI '{best[3][j1:j2]}'")
                    k += 1


if __name__ == "__main__":
    main()

