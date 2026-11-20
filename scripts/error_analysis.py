"""วิเคราะห์ว่า AI ผิดแบบไหนบ่อยที่สุด (ระดับตัวอักษร) บน FLEURS test + คลิปทีม

  .venv\\Scripts\\python.exe scripts\\error_analysis.py models\\full-lao-ultra2
ผล: results/error_analysis_<model>.txt
"""
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from common import RESULTS, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")

TONE = set("່້໊໋")
VOW = set("ະັາຳິີຶືຸູົຼຽເແໂໃໄໍ")
CONS = set("ກຂຄງຈສຊຍດຕຖທນບປຜຝພຟມຢຣລວຫອຮໜໝຼ")
DIG = set("0123456789໐໑໒໓໔໕໖໗໘໙")


def cls(c: str) -> str:
    if c in TONE:
        return "วรรณยุกต์"
    if c in VOW:
        return "สระ"
    if c in CONS:
        return "พยัญชนะ"
    if c in DIG:
        return "ตัวเลข"
    if c.isascii() and c.isalpha():
        return "อักษรอังกฤษ"
    return "อื่น ๆ"


def main():
    from compare_models import load_audio, tests
    from v3_asr import V3

    model = Path(sys.argv[1] if len(sys.argv) > 1 else "models/full-lao-ultra2")
    asr = V3(model_dir=model)
    sets = tests()
    kinds, pairs, dels, ins = Counter(), Counter(), Counter(), Counter()
    by_set, by_len, pos = {}, {"<2 วิ": [], "2–4 วิ": [], "4–8 วิ": [], ">8 วิ": []}, Counter()
    tot_ref = 0
    for name, rows in sets.items():
        audio = [load_audio(r["file"]) for r in rows]
        hyps = asr.transcribe(audio, max_new=200)
        e_set = n_set = 0
        for r, h, a in zip(rows, hyps, audio):
            ref, hyp = norm_for_cer(r["text"]), norm_for_cer(h)
            n = max(1, len(ref))
            sm = SequenceMatcher(None, ref, hyp, autojunk=False)
            errs = 0
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal":
                    continue
                rseg, hseg = ref[i1:i2], hyp[j1:j2]
                errs += max(len(rseg), len(hseg))
                where = "ต้นประโยค" if i1 < 0.15 * n else "ท้ายประโยค" if i2 > 0.85 * n else "กลางประโยค"
                pos[where] += max(len(rseg), len(hseg))
                if op == "replace":
                    pairs[(rseg, hseg)] += 1
                    for c in rseg:
                        kinds[cls(c)] += 1
                elif op == "delete":
                    dels[rseg] += 1
                    kinds["ตกหล่น (AI ไม่เขียน)"] += len(rseg)
                else:
                    ins[hseg] += 1
                    kinds["เกินมา (AI เขียนเพิ่ม)"] += len(hseg)
            e_set += errs
            n_set += n
            dur = len(a) / 16000
            b = "<2 วิ" if dur < 2 else "2–4 วิ" if dur < 4 else "4–8 วิ" if dur < 8 else ">8 วิ"
            by_len[b].append(errs / n)
        by_set[name] = 100 * e_set / max(1, n_set)
        tot_ref += n_set
    tot = sum(kinds.values())
    out = [f"วิเคราะห์ข้อผิดพลาด {model.name} (เทียบเฉลยทุกตัวอักษร {tot_ref:,} ตัว)", ""]
    out.append("== ผิดตามชุดสอบ (CER %) ==")
    out += [f"  {k}: {v:.1f}%" for k, v in by_set.items()]
    out.append("\n== ผิดแบบไหน (สัดส่วนของตัวอักษรที่ผิด) ==")
    out += [f"  {k:24s} {v:5d}  ({100 * v / tot:.0f}%)" for k, v in kinds.most_common()]
    out.append("\n== ผิดตรงไหนของประโยค ==")
    pt = sum(pos.values())
    out += [f"  {k}: {100 * v / pt:.0f}%" for k, v in pos.most_common()]
    out.append("\n== ผิดตามความยาวคลิป (CER เฉลี่ยต่อคลิป) ==")
    out += [f"  {k}: {100 * np.mean(v):.1f}% ({len(v)} คลิป)" for k, v in by_len.items() if v]
    out.append("\n== คู่ที่สับสนบ่อยสุด (เฉลย → AI) ==")
    out += [f"  {a!r} → {b!r}: {c}" for (a, b), c in pairs.most_common(30)]
    out.append("\n== ตกหล่นบ่อยสุด ==")
    out += [f"  {a!r}: {c}" for a, c in dels.most_common(12)]
    out.append("\n== เขียนเกินบ่อยสุด ==")
    out += [f"  {a!r}: {c}" for a, c in ins.most_common(12)]
    txt = "\n".join(out)
    print(txt)
    (RESULTS / f"error_analysis_{model.name}.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
