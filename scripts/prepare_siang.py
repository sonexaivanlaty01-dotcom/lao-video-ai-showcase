"""แปลงชุดข้อมูล SiangLao (parquet) → wav 16kHz + train.tsv / test.tsv

ที่มา: huggingface.co/datasets/SiangLao/lao-asr-thesis-dataset  (CC-BY-NC-4.0: ห้ามใช้เชิงการค้า)
- train+validation+dev → train.tsv (ใช้เทรน)
- test → test.tsv (เก็บไว้สอบสำเนียงต่างๆ)
- ตัดประโยคที่ซ้ำกับข้อสอบ FLEURS (dev/test) ออก กัน "เห็นข้อสอบก่อนสอบ"
- ตัดคลิปยาวเกิน 30 วินาที (Whisper รับได้แค่ 30 วิ)

  .venv\\Scripts\\python.exe scripts\\prepare_siang.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import polars as pl  # pyarrow.parquet ถูก Windows App Control บล็อก → ใช้ polars แทน
import soundfile as sf
from common import ROOT, load_fleurs, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
SRC = ROOT / "data" / "siang_lao" / "raw" / "data"
OUT = ROOT / "data" / "siang_lao"
SR = 16000


def main():
    banned = {norm_for_cer(r["text"]) for s in ("dev", "test") for r in load_fleurs(s)}
    groups = {"train": ["train", "validation", "dev"], "test": ["test"]}
    for name, splits in groups.items():
        wav_dir = OUT / name
        wav_dir.mkdir(parents=True, exist_ok=True)
        rows, skipped, secs = [], {"ซ้ำข้อสอบ": 0, "ยาวเกิน": 0, "ไม่มีข้อความ": 0}, 0.0
        for sp in splits:
            for f in sorted(SRC.glob(f"{sp}-*.parquet")):
                t = pl.read_parquet(f)
                cols = t.columns
                for i, rec in enumerate(t.iter_rows(named=True)):
                    text = (rec.get("text") or "").strip()
                    if not text:
                        skipped["ไม่มีข้อความ"] += 1
                        continue
                    if norm_for_cer(text) in banned:
                        skipped["ซ้ำข้อสอบ"] += 1
                        continue
                    audio, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != SR:
                        import librosa

                        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
                    dur = len(audio) / SR
                    if dur > 30 or dur < 0.3:
                        skipped["ยาวเกิน"] += 1
                        continue
                    fn = f"{sp}_{f.stem.split('-')[1]}_{i:05}.wav"
                    sf.write(wav_dir / fn, np.asarray(audio), SR)
                    accent = rec.get("accent", "") if "accent" in cols else ""
                    rows.append(f"{name}/{fn}\t{text}\t{dur:.2f}\t{accent}\t{rec.get('speaker_id', '')}")
                    secs += dur
                print(f"  {f.name}: สะสม {len(rows)} คลิป", flush=True)
        (OUT / f"{name}.tsv").write_text(
            "file\ttext\tduration\taccent\tspeaker\n" + "\n".join(rows) + "\n", encoding="utf-8"
        )
        print(f"✅ {name}: {len(rows)} คลิป = {secs/3600:.1f} ชั่วโมง | ตัดออก {skipped}")


if __name__ == "__main__":
    main()
