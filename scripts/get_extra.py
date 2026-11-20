"""โหลดชุดเสียงลาวเพิ่มจาก Hugging Face → wav 16kHz + all.tsv (file, text, duration, source, license)

  .venv\\Scripts\\python.exe scripts\\get_extra.py            (ทุกชุด)
  .venv\\Scripts\\python.exe scripts\\get_extra.py kittzk     (ชุดเดียว)

ชุดที่มี:
  kittzk  KitTzk/lao_stt_training_data   ~3 ชม. ภาษาพูดจาก YouTube  CC-BY-4.0 (ใช้เชิงการค้าได้ ต้องให้เครดิต)
          (ชุด validation ของเขาขนาดเท่า train เป๊ะ = ข้อมูลซ้ำ → โหลดแค่ train)
  cv      Mozilla Common Voice 22 (lo)    ~10 นาที                   CC0 (ฟรี 100%)

ตัดประโยคที่ซ้ำข้อสอบ FLEURS ออก และคลิปยาวเกิน 30 วิ / สั้นกว่า 0.5 วิ
ผลลัพธ์: data/extra/<ชุด>/wav/*.wav + data/extra/<ชุด>/all.tsv
ขั้นต่อไป: scripts/check_extra.py ให้ v3 ฟังเทียบเฉลย คัดประโยคที่เฉลยน่าสงสัยออก
"""
import csv
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import polars as pl  # pyarrow.parquet ถูก Windows App Control บล็อก
import soundfile as sf
from common import ROOT, clean_text, ffmpeg_exe, load_fleurs, norm_for_cer
from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "data" / "extra"
SR = 16000


def to16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def decode_ffmpeg(data: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(data)
    try:
        raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", f.name, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
    finally:
        Path(f.name).unlink(missing_ok=True)
    return np.frombuffer(raw, dtype=np.float32)


class Writer:
    def __init__(self, name: str, source: str, license: str):
        self.dir = OUT / name
        (self.dir / "wav").mkdir(parents=True, exist_ok=True)
        self.source, self.license = source, license
        self.rows, self.secs = [], 0.0
        self.skip = {"ซ้ำข้อสอบ": 0, "ความยาวไม่เหมาะ": 0, "ไม่มีข้อความ": 0, "ซ้ำกันเอง": 0, "เปิดไฟล์ไม่ได้": 0}
        self.banned = {norm_for_cer(r["text"]) for s in ("dev", "test") for r in load_fleurs(s)}
        self.seen = set()

    def add(self, key: str, text: str, audio: np.ndarray | None):
        text = clean_text(text or "", strip_punct=False).strip()
        if not text:
            self.skip["ไม่มีข้อความ"] += 1
            return
        n = norm_for_cer(text)
        if n in self.banned:
            self.skip["ซ้ำข้อสอบ"] += 1
            return
        if audio is None:
            self.skip["เปิดไฟล์ไม่ได้"] += 1
            return
        dur = len(audio) / SR
        if not 0.5 <= dur <= 30:
            self.skip["ความยาวไม่เหมาะ"] += 1
            return
        sig = (n, round(dur, 1))
        if sig in self.seen:
            self.skip["ซ้ำกันเอง"] += 1
            return
        self.seen.add(sig)
        fn = f"wav/{key}.wav"
        sf.write(self.dir / fn, audio, SR)
        self.rows.append([fn, text, f"{dur:.2f}", self.source, self.license])
        self.secs += dur

    def save(self):
        with open(self.dir / "all.tsv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["file", "text", "duration", "source", "license"])
            w.writerows(self.rows)
        print(f"✅ {self.dir.name}: {len(self.rows)} คลิป = {self.secs/3600:.2f} ชม. | ตัดออก {self.skip}", flush=True)


def get_kittzk():
    w = Writer("kittzk", "KitTzk/lao_stt_training_data", "CC-BY-4.0")
    p = hf_hub_download("KitTzk/lao_stt_training_data", "data/train-00000-of-00001.parquet", repo_type="dataset")
    t = pl.read_parquet(p)
    print(f"kittzk: {t.height} แถว คอลัมน์ {t.columns}", flush=True)
    for i, rec in enumerate(t.iter_rows(named=True)):
        a = rec["audio"]
        audio = None
        if a and a.get("bytes"):
            try:
                audio = to16k(*sf.read(io.BytesIO(a["bytes"]), dtype="float32"))
            except Exception:
                audio = decode_ffmpeg(a["bytes"])
        w.add(f"k{i:05}", rec.get("sentence", ""), audio)
        if i % 300 == 0:
            print(f"  {i}/{t.height}", flush=True)
    w.save()


def get_cv():
    repo = "fsicoli/common_voice_22_0"
    w = Writer("cv", "Mozilla Common Voice 22 (lo)", "CC0")
    tsv = hf_hub_download(repo, "transcript/lo/validated.tsv", repo_type="dataset")
    text_of = {}
    with open(tsv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            text_of[r["path"]] = r["sentence"]
    for split in ("train", "dev", "test", "other"):
        try:
            tar = hf_hub_download(repo, f"audio/lo/{split}/lo_{split}_0.tar", repo_type="dataset")
        except Exception:
            continue
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                name = Path(m.name).name
                if not m.isfile() or name not in text_of:
                    continue
                data = tf.extractfile(m).read()
                w.add(f"cv_{Path(name).stem}", text_of.pop(name), decode_ffmpeg(data))
    w.save()


SETS = {"kittzk": get_kittzk, "cv": get_cv}

if __name__ == "__main__":
    for name in sys.argv[1:] or SETS:
        SETS[name]()
