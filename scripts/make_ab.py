"""เตรียมแบบทดสอบ "ฟังเทียบแบบปิดชื่อ" ระหว่าง 2 โมเดล (ค่าเริ่มต้น v3 กับ v4)

ให้ทั้งสองโมเดลฟังทุกบรรทัดของคลิปทีม (data/subs/*.lo.json) → เก็บเฉพาะบรรทัดที่สองตัวตอบต่างกัน
สุ่มว่าตัวไหนเป็น A/B (คนฟังไม่รู้ว่าเป็นโมเดลไหน) → เปิดหน้า http://localhost:8766/ab.html ให้คนเลือก

  .venv\\Scripts\\python.exe scripts\\make_ab.py [--n 25]

ผลลัพธ์: data/ab/pairs.json + data/ab/wav/NNN.wav (คะแนนที่โหวตเก็บใน data/ab/votes.json)
"""
import argparse
import gc
import json
import random
import sys

import soundfile as sf
import torch
from buzz_fast import load_audio
from common import ROOT, norm_for_cer
from v3_asr import V3

sys.stdout.reconfigure(encoding="utf-8")
SR = 16000
OUT = ROOT / "data" / "ab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, default=["v3", "v4"])
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()

    segs = []
    for doc_path in sorted((ROOT / "data" / "subs").glob("*.lo.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        clip = ROOT / "data" / "my_clips" / doc.get("source", "")
        if not clip.exists():
            continue
        audio = load_audio(clip)
        for s in doc.get("segments", []):
            x = audio[int(s["start"] * SR) : int(s["end"] * SR)]
            if 0.5 * SR <= len(x) <= 30 * SR:
                segs.append({"clip": clip.name, "start": s["start"], "end": s["end"], "audio": x})
    print(f"ทั้งหมด {len(segs)} บรรทัด จาก data/subs", flush=True)

    outs = {}
    for m in a.models:
        asr = V3(model_dir=ROOT / "models" / f"full-lao-{m}")
        outs[m] = asr.transcribe([s["audio"] for s in segs], max_new=200)
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  {m} ฟังเสร็จ", flush=True)

    m1, m2 = a.models
    diff = [i for i in range(len(segs)) if norm_for_cer(outs[m1][i]) != norm_for_cer(outs[m2][i])]
    print(f"ตอบต่างกัน {len(diff)} บรรทัด → สุ่ม {min(a.n, len(diff))} บรรทัด", flush=True)
    random.seed(42)
    pick = sorted(random.sample(diff, min(a.n, len(diff))), key=lambda i: (segs[i]["clip"], segs[i]["start"]))

    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    pairs = []
    for k, i in enumerate(pick):
        s = segs[i]
        sf.write(OUT / "wav" / f"{k:03}.wav", s["audio"], SR)
        a_model = random.choice(a.models)
        b_model = m2 if a_model == m1 else m1
        pairs.append({"id": k, "clip": s["clip"], "start": s["start"], "end": s["end"],
                      "A": outs[a_model][i], "B": outs[b_model][i], "A_is": a_model, "B_is": b_model})
    (OUT / "pairs.json").write_text(json.dumps({"models": a.models, "pairs": pairs}, ensure_ascii=False, indent=1), encoding="utf-8")
    votes = OUT / "votes.json"
    if votes.exists():  # แบบทดสอบใหม่ → เก็บคะแนนเก่าไว้ ไม่ลบ
        votes.rename(OUT / f"votes_old_{int(votes.stat().st_mtime)}.json")
    print(f"✅ เตรียม {len(pairs)} คู่ → เปิด http://localhost:8766/ab.html")


if __name__ == "__main__":
    main()
