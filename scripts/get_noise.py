"""ดึงเสียงรบกวนจริง + ดนตรี (MUSAN, CC-BY 4.0) ไว้ผสมตอนเทรน → data/noise/*.wav (16kHz mono)

  python scripts/get_noise.py --music-shards 3 --max-sec 60

แหล่ง:
  - FluidInference/musan  : noise/ (free-sound + sound-bible ~780 ไฟล์ เสียงรอบข้าง รถ ลม เครื่องจักร ฯลฯ)
  - corypaik/musan        : music/ (parquet 10 ก้อน ~0.5 GB ต่อก้อน; ดนตรีหลายแนว fma/jamendo/rfm/classical)
ตัดแต่ละไฟล์ให้ยาวไม่เกิน --max-sec วินาที (สุ่มช่วง) ประหยัดที่และโหลดเร็ว
"""
import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
sys.stdout.reconfigure(encoding="utf-8")


def to16k(a: np.ndarray, sr: int) -> np.ndarray:
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        import librosa

        a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=SR)
    return a.astype(np.float32)


def clip(a: np.ndarray, max_sec: float, rng, sr: int = SR) -> np.ndarray:
    n = int(max_sec * sr)
    if len(a) <= n:
        return a
    s = rng.randint(0, len(a) - n)
    return a[s : s + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "noise"))
    ap.add_argument("--music-shards", type=int, default=3, help="จำนวนก้อนดนตรี (0–10)")
    ap.add_argument("--max-sec", type=float, default=60)
    ap.add_argument("--limit", type=int, default=None, help="จำกัดจำนวนไฟล์ต่อแหล่ง (ไว้ทดสอบ)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    from huggingface_hub import snapshot_download

    d = Path(snapshot_download("FluidInference/musan", repo_type="dataset", allow_patterns=["noise/**/*.wav"]))
    files = sorted(d.glob("noise/**/*.wav"))[: a.limit]
    n_noise = 0
    for f in files:
        try:
            x, sr = sf.read(f, dtype="float32")
            x = clip(to16k(x, sr), a.max_sec, rng)
            if len(x) >= SR:
                sf.write(out / f"noise_{n_noise:04d}.wav", x, SR)
                n_noise += 1
        except Exception as e:
            print("ข้าม", f.name, e)
    print(f"เสียงรบกวน: {n_noise} ไฟล์", flush=True)

    import polars as pl

    n_music = 0
    from huggingface_hub import hf_hub_download

    for k in range(a.music_shards):
        # โหลดไฟล์มาก่อน (polars hf:// บน Kaggle ใช้ไม่ได้: "Invalid repo name") แล้วค่อยอ่าน
        p = hf_hub_download("corypaik/musan", f"music/train-{k:05d}-of-00010.parquet", repo_type="dataset")
        lf = pl.scan_parquet(p)
        df = (lf.head(a.limit) if a.limit else lf).select("audio").collect()
        for r in df["audio"].to_list():
            try:
                x, sr = sf.read(io.BytesIO(r["bytes"]), dtype="float32")
                x = to16k(clip(x, a.max_sec, rng, sr), sr)  # ตัดก่อนแล้วค่อยแปลงความถี่ (เร็วกว่ามาก)
                if len(x) >= SR:
                    sf.write(out / f"music_{n_music:04d}.wav", x, SR)
                    n_music += 1
            except Exception as e:
                print("ข้ามดนตรี", e)
        print(f"  ดนตรีก้อน {k + 1}/{a.music_shards}: สะสม {n_music} ไฟล์", flush=True)
    tot = sum(p.stat().st_size for p in out.glob("*.wav")) / 1e9
    print(f"✅ data/noise: เสียงรบกวน {n_noise} + ดนตรี {n_music} ไฟล์ ({tot:.2f} GB)")


if __name__ == "__main__":
    main()
