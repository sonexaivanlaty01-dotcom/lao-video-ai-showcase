"""ตรวจว่าเสียงคลิป YouTube เป็น "ภาษาลาว" จริงไหม (กันภาษาไทย/ม้ง/อังกฤษหลุดเข้าข้อมูลเทรน)
ใช้ speechbrain/lang-id-voxlingua107-ecapa (Apache-2.0, 107 ภาษา รวม lo/th) — สุ่ม 6 ท่อนต่อคลิป แล้วโหวต

  .venv\\Scripts\\python.exe scripts\\yt_lid.py   → data/yt_cc/lid.tsv (video, lang, lao_votes, n, top_langs, hours)
"""
import collections
import csv
import os
import sys

import torch
from common import ROOT

sys.stdout.reconfigure(encoding="utf-8")
D = ROOT / "data" / "yt_cc"
K = 6


def main():
    os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
    import soundfile as sf
    import torchaudio
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    lid = EncoderClassifier.from_hparams("speechbrain/lang-id-voxlingua107-ecapa",
                                         savedir=str(ROOT / "models" / "lid-voxlingua107"),
                                         local_strategy=LocalStrategy.COPY,  # Windows สร้าง symlink ไม่ได้
                                         run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    by = collections.defaultdict(list)
    for s in csv.DictReader(open(D / "segments.tsv", encoding="utf-8"), delimiter="\t"):
        by[s["video"]].append(s)
    out = D / "lid.tsv"
    done = {r["video"] for r in csv.DictReader(open(out, encoding="utf-8"), delimiter="\t")} if out.exists() else set()
    if not out.exists():
        out.write_text("video\tlang\tlao_votes\tn\ttop_langs\thours\n", encoding="utf-8")
    for i, (v, segs) in enumerate(by.items()):
        if v in done:
            continue
        hours = sum(float(s["duration"]) for s in segs) / 3600
        long = sorted([s for s in segs if float(s["duration"]) >= 3], key=lambda s: s["start"] and float(s["start"]))
        pick = [long[int(j * len(long) / K)] for j in range(min(K, len(long)))] or segs[:K]
        votes = collections.Counter()
        for s in pick:
            p = D / s["file"]
            x, sr = sf.read(str(p if p.exists() else p.with_suffix(".flac")), dtype="float32")
            wav = torch.from_numpy(x if x.ndim == 1 else x[:, 0]).unsqueeze(0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            lab = lid.classify_batch(wav)[3][0]  # เช่น "lo: Lao"
            votes[lab.split(":")[0].strip()] += 1
        lang = votes.most_common(1)[0][0]
        top = ",".join(f"{k}{c}" for k, c in votes.most_common(3))
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"{v}\t{lang}\t{votes.get('lo', 0)}\t{len(pick)}\t{top}\t{hours:.3f}\n")
        if i % 25 == 0:
            print(f"{i}/{len(by)} {v}: {top}", flush=True)
    rows = list(csv.DictReader(open(out, encoding="utf-8"), delimiter="\t"))
    agg = collections.Counter()
    for r in rows:
        agg[r["lang"]] += float(r["hours"])
    print("✅ ชั่วโมงตามภาษาที่ตรวจได้:", {k: round(h, 1) for k, h in agg.most_common()})


if __name__ == "__main__":
    main()
