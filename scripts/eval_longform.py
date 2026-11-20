"""วัดความแม่นแบบทั้งคลิป (ไม่ขึ้นกับวิธีตัดท่อน): รวมข้อความทั้งคลิปแล้วคิด CER เทียบเฉลย

  .venv\\Scripts\\python.exe scripts\\eval_longform.py --model models\\full-lao-ultra2 --max-s 8 15 20

ชุดทดสอบ: data/my_train/*/metadata.tsv (คลิปทีม) + data/eval/sp500_ref.tsv (ข่าว S&P 500, เฉลยจาก Claude)
ตัดเสียงเฉพาะช่วงที่มีเฉลย (จากเริ่มท่อนแรกถึงจบท่อนสุดท้าย)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jiwer
from common import RESULTS, ROOT, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")


def load_sets():
    sets = []
    for tsv in sorted(list((ROOT / "data" / "my_train").glob("*/metadata.tsv")) + [ROOT / "data" / "eval" / "sp500_ref.tsv"]):
        rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
        video = ROOT / "data" / "my_clips" / rows[0]["source"]
        if video.exists():
            sets.append({"name": tsv.parent.name if tsv.name == "metadata.tsv" else "sp500", "video": video, "rows": rows})
    return sets


def cer_text(ref: str, hyp: str) -> float:
    return 100 * jiwer.cer(norm_for_cer(ref), norm_for_cer(hyp) or "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/full-lao-ultra2")
    ap.add_argument("--max-s", type=float, nargs="+", default=[8.0])
    ap.add_argument("--prev-ctx", action="store_true")
    ap.add_argument("--merge-gap", type=float, default=0.0, help="รวมท่อนที่เว้นช่วงไม่เกินนี้ (วินาที)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lm", type=float, nargs="+", default=[0.0], help="น้ำหนักคลังข้อความ (0 = ปิด) ลองหลายค่าได้")
    ap.add_argument("--lm-bonus", type=float, default=2.0)
    ap.add_argument("--lm-unsure", type=float, default=0.8, help="ช่วยเฉพาะขั้นที่ AI มั่นใจต่ำกว่านี้")
    ap.add_argument("--rescore", type=float, nargs="*", default=None,
                    help="เลือกประโยคทางเลือกแทน ถ้าคลังข้อความให้คะแนนต่อตัวอักษรดีกว่าเกินค่านี้ (ลองหลายค่า)")
    a = ap.parse_args()
    from buzz_fast import load_audio
    from longform import transcribe_audio
    from v3_asr import SR, V3

    asr = V3(model_dir=ROOT / a.model)
    sets = load_sets()
    audio = {}
    for s in sets:
        full = load_audio(s["video"])
        st, en = float(s["rows"][0]["start"]), float(s["rows"][-1]["end"])
        audio[s["name"]] = full[int(max(0, st - 0.3) * SR) : int((en + 0.3) * SR)]
    report = []
    if a.rescore is not None:  # สร้างทางเลือกครั้งเดียว แล้วลองหลาย margin (ใช้ CPU อย่างเดียว)
        from char_lm import load_lm

        lm = load_lm()
        dets = {s["name"]: transcribe_audio(asr, audio[s["name"]], max_s=a.max_s[0], max_alts=3)[1] for s in sets}

        def per_char(t):
            return lm.score(t, eos=True) / max(1, len(t))

        for m in [None] + list(a.rescore):
            tot_e = tot_n = 0
            line = []
            for s in sets:
                ref = " ".join(r["text"] for r in s["rows"])
                hyp = []
                for d in dets[s["name"]]:
                    best = d["text"]
                    if m is not None and d["alts"]:
                        base = per_char(best)
                        cand = max(d["alts"], key=lambda x: per_char(x["text"]))["text"]
                        if per_char(cand) - base > m:
                            best = cand
                    hyp.append(best)
                c = cer_text(ref, " ".join(hyp))
                n = len(norm_for_cer(ref))
                tot_e += c * n
                tot_n += n
                line.append(f"{s['name'][:14]} {c:5.1f}%")
            msg = f"{Path(a.model).name} rescore margin={m} | รวม {tot_e / tot_n:5.1f}% | " + " | ".join(line)
            print(msg, flush=True)
            report.append(msg)
        a.max_s = []
    asr.lm_bonus = a.lm_bonus
    asr.lm_unsure = a.lm_unsure
    for ms, lw in [(m, w) for m in a.max_s for w in a.lm]:
        asr.lm_weight = lw
        tot_e = tot_n = 0
        line = []
        t0 = time.time()
        for s in sets:
            ref = " ".join(r["text"] for r in s["rows"])
            spans, det = transcribe_audio(asr, audio[s["name"]], max_s=ms, prev_ctx=a.prev_ctx, prompt=a.prompt, merge_gap=a.merge_gap)
            hyp = " ".join(d["text"] for d in det)
            c = cer_text(ref, hyp)
            n = len(norm_for_cer(ref))
            tot_e += c * n
            tot_n += n
            line.append(f"{s['name'][:14]} {c:5.1f}%")
        dt = time.time() - t0
        msg = f"{Path(a.model).name} max_s={ms:<4} gap={a.merge_gap} prev={int(a.prev_ctx)} prompt={'Y' if a.prompt else '-'} lm={lw}/{a.lm_bonus}/{a.lm_unsure} | รวม {tot_e / tot_n:5.1f}% | " + " | ".join(line) + f" | {dt:.0f}s"
        print(msg, flush=True)
        report.append(msg)
    with open(RESULTS / "eval_longform.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(f"{time.strftime('%Y-%m-%d %H:%M')} {a.tag} {m}" for m in report) + "\n")


if __name__ == "__main__":
    main()
