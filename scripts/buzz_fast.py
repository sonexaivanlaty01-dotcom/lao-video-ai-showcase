"""BuzzASR แบบเร็ว (CTranslate2 int8 บนการ์ดจอ)

ต้องแปลงโมเดลก่อน 1 ครั้ง:  scripts\\convert_buzzasr.py
(เรียก CTranslate2 ตรงๆ ไม่ผ่าน faster-whisper เพราะ tokenizer ของ BuzzASR ไม่ตรงผังมาตรฐาน)

วัดผลกับ FLEURS (เทียบกับ evaluate.py ได้ตรงๆ):
  .venv\\Scripts\\python.exe scripts\\buzz_fast.py --eval 30
ทำซับจากคลิปของเรา (ได้ .srt):
  .venv\\Scripts\\python.exe scripts\\buzz_fast.py data\\my_clips\\คลิป.mp4
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import common  # noqa: F401  (ตั้งค่า CUDA DLL path)
import jiwer
import numpy as np
from common import RESULTS, ROOT, load_fleurs, norm_for_cer
from faster_whisper.feature_extractor import FeatureExtractor
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")
MODEL_DIR = ROOT / "models" / "buzzasr-ct2-int8"
SR = 16000
# ผังมาตรฐาน Whisper large-v3: <|startoftranscript|> <|lo|> <|transcribe|> <|notimestamps|>
PROMPT = [51866, 51944, 51968, 51972]
TEXT_LIMIT = 51865  # id ต่ำกว่านี้คือข้อความ
GEN = dict(beam_size=1, no_repeat_ngram_size=3, repetition_penalty=1.2, max_length=448)


class Buzz:
    # int8_float16 ใช้ได้ดีบน GTX 1650 (ต่างจาก PyTorch fp16) ส่วน int8_float32 กิน VRAM เกิน 4GB
    def __init__(self, compute_type="int8_float16", device="cuda"):
        global ctranslate2
        import ctranslate2  # โหลดเฉพาะตอนใช้ BuzzASR (Windows อาจบล็อกไฟล์นี้ → ส่วนอื่นยังใช้ได้)

        self.model = ctranslate2.models.Whisper(str(MODEL_DIR), device=device, compute_type=compute_type)
        self.fe = FeatureExtractor(feature_size=128)
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    def _features(self, audio: np.ndarray) -> np.ndarray:
        audio = audio[: 30 * SR]
        audio = np.pad(audio, (0, 30 * SR - len(audio)))
        return self.fe(audio, padding=0)[:, :3000]

    def transcribe(self, audios: list[np.ndarray], batch: int = 4) -> list[str]:
        """แต่ละท่อนต้องยาวไม่เกิน 30 วินาที"""
        out = []
        for i in range(0, len(audios), batch):
            feats = np.stack([self._features(a) for a in audios[i : i + batch]]).astype(np.float32)
            res = self.model.generate(ctranslate2.StorageView.from_array(feats), [PROMPT] * len(feats), **GEN)
            for r in res:
                ids = [t for t in r.sequences_ids[0] if t < TEXT_LIMIT]
                out.append(common.fix_am(self.tok.decode(ids)).strip())
        return out


def load_audio(path) -> np.ndarray:
    p = str(path)
    if p.lower().endswith(".wav"):
        import librosa

        return librosa.load(p, sr=SR)[0]
    # วิดีโอ/mp3/m4a ฯลฯ → ใช้ ffmpeg (PyAV ของ faster-whisper พังกับ PyAV 19)
    import subprocess

    raw = subprocess.run(
        [common.ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", p, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def evaluate(buzz: Buzz, n: int, name: str):
    rows = load_fleurs("test", n)
    audios = [load_audio(r["file"]) for r in rows]
    t0 = time.time()
    hyps = []
    for i in range(0, len(audios), 10):
        hyps += buzz.transcribe(audios[i : i + 10])
        print(f"  ... {len(hyps)}/{len(rows)} คลิป ({time.time()-t0:.0f} วิ)", flush=True)
    secs = time.time() - t0
    cer = jiwer.cer([norm_for_cer(r["text"]) for r in rows], [norm_for_cer(h) for h in hyps])

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"eval_{name}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "ข้อความถูก", "AI ฟังได้", "CER"])
        for r, h in zip(rows, hyps):
            w.writerow([r["file"].name, r["text"], h, f"{jiwer.cer(norm_for_cer(r['text']), norm_for_cer(h)):.2f}"])
    print("\n================ ผลรวม ================")
    print(f"CER: {cer*100:.1f}%   เวลา {secs:.0f} วิ ({secs/len(rows):.1f} วิ/คลิป) → {path}")
    for r, h in list(zip(rows, hyps))[:2]:
        print("\n  ถูก:", r["text"][:80])
        print("  AI :", h[:80])


def srt_time(t: float) -> str:
    h, rem = divmod(int(max(0.0, t) * 1000), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(buzz: Buzz, media: Path, out_dir: Path, max_sec: float = 8.0):
    """VAD หาช่วงที่มีเสียงพูด → ตัดเป็นท่อนสั้นๆ (เหมาะกับซับ) → ถอดเสียงทีละหลายท่อน"""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    t0 = time.time()
    audio = load_audio(media)
    spans = get_speech_timestamps(
        audio, VadOptions(max_speech_duration_s=max_sec, min_silence_duration_ms=300, speech_pad_ms=150)
    )
    texts = buzz.transcribe([audio[s["start"] : s["end"]] for s in spans])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{media.stem}.lo.buzz.srt"
    lines, n = [], 0
    for s, t in zip(spans, texts):
        if t:
            n += 1
            lines += [str(n), f"{srt_time(s['start']/SR)} --> {srt_time(s['end']/SR)}", t, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    dur = len(audio) / SR
    print(f"✅ คลิปยาว {dur:.0f} วิ → {n} บรรทัด ใน {time.time()-t0:.0f} วิ → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="ไฟล์วิดีโอ/เสียง")
    ap.add_argument("--eval", type=int, default=0, help="วัดผลกับ FLEURS test N คลิป")
    ap.add_argument("--compute", default="int8_float16", help="int8_float16 (แนะนำ) / int8 / int8_float32 (VRAM ไม่พอบน 4GB)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "subs"))
    args = ap.parse_args()

    buzz = Buzz(args.compute, args.device)
    if args.eval:
        evaluate(buzz, args.eval, args.name or f"buzzfast_{args.compute}_{args.eval}")
    elif args.path:
        to_srt(buzz, Path(args.path), Path(args.out))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
