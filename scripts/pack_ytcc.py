"""แพ็กท่อนเสียง YouTube CC-BY เป็น FLAC (เล็กลง ~ครึ่ง) สำหรับอัปโหลดเป็น Kaggle dataset ส่วนตัว `lao-ytcc-seg`

  .venv\\Scripts\\python.exe scripts\\pack_ytcc.py   → kaggle/lao-ytcc-seg.zip
  (ข้างใน: yt_cc/segments.tsv, yt_cc/credits.tsv, yt_cc/seg/*.flac — ชื่อใน segments.tsv ยังเป็น .wav
   pseudo_label.py / train อ่าน .flac แทนอัตโนมัติถ้าไม่มี .wav)
"""
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from common import ROOT

sys.stdout.reconfigure(encoding="utf-8")
D = ROOT / "data" / "yt_cc"
TMP = ROOT / "kaggle" / "_ytcc_pack" / "yt_cc"


def conv(w):
    out = TMP / "seg" / (w.stem + ".flac")
    if not out.exists():
        x, sr = sf.read(w, dtype="int16")
        sf.write(out, x, sr, format="FLAC")


def main():
    (TMP / "seg").mkdir(parents=True, exist_ok=True)
    wavs = sorted((D / "seg").glob("*.wav"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(conv, wavs))
    for f in ("segments.tsv", "credits.tsv"):
        shutil.copy(D / f, TMP / f)
    z = ROOT / "kaggle" / "lao-ytcc-seg.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:  # FLAC บีบแล้ว ไม่ต้องบีบซ้ำ
        for p in sorted(TMP.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(TMP.parent))
    print(f"✅ {len(wavs)} ท่อน → {z} ({z.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
