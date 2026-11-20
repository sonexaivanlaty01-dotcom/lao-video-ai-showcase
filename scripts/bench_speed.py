"""วัดความเร็วทำซับ (ไม่ใช้การ์ดจอ เพื่อไม่ไปแย่งกับการเทรน)

  .venv\\Scripts\\python.exe scripts\\bench_speed.py [--threads 4]

- เสียงทดสอบ: ประโยคจาก FLEURS test ต่อกันให้ยาว ~60 วินาที
- BuzzASR (large-v3, int8) บน CPU  → แทน "เซิร์ฟเวอร์/Cloud ที่ไม่มีการ์ดจอ"
- Whisper small (int8) บน CPU      → ขนาดเท่าตัวที่เราเทรน = แทน "ทำในมือถือ" (มือถือจริงอาจช้ากว่านี้)
ผล: วินาทีที่ใช้ต่อเสียง 1 นาที (ยิ่งน้อยยิ่งเร็ว)
"""
import argparse
import sys
import time

import common  # noqa: F401
import numpy as np
from common import load_fleurs

sys.stdout.reconfigure(encoding="utf-8")
SR = 16000


def make_audio(target_sec=60):
    import librosa

    parts, total = [], 0
    for r in load_fleurs("test", 40):
        a = librosa.load(str(r["file"]), sr=SR)[0]
        parts += [a, np.zeros(int(0.6 * SR), np.float32)]
        total += len(a) / SR + 0.6
        if total >= target_sec:
            break
    return np.concatenate(parts)


def vad_spans(audio):
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return get_speech_timestamps(audio, VadOptions(max_speech_duration_s=8, min_silence_duration_ms=300, speech_pad_ms=150))


def bench_buzz(audio, threads):
    import ctranslate2
    from buzz_fast import Buzz

    t = time.time()
    b = Buzz.__new__(Buzz)
    b.model = ctranslate2.models.Whisper(str(common.ROOT / "models/buzzasr-ct2-int8"), device="cpu", compute_type="int8", intra_threads=threads)
    from faster_whisper.feature_extractor import FeatureExtractor
    from tokenizers import Tokenizer

    b.fe = FeatureExtractor(feature_size=128)
    b.tok = Tokenizer.from_file(str(common.ROOT / "models/buzzasr-ct2-int8/tokenizer.json"))
    load = time.time() - t
    spans = vad_spans(audio)
    t = time.time()
    texts = b.transcribe([audio[s["start"] : s["end"]] for s in spans], batch=1)
    return load, time.time() - t, texts


def bench_small(audio, threads):
    from faster_whisper import WhisperModel

    t = time.time()
    m = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=threads)
    load = time.time() - t
    t = time.time()
    segs, _ = m.transcribe(audio, language="lo", beam_size=1, vad_filter=True, condition_on_previous_text=False)
    texts = [s.text for s in segs]
    return load, time.time() - t, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    audio = make_audio()
    dur = len(audio) / SR
    print(f"เสียงทดสอบยาว {dur:.0f} วินาที | ใช้ CPU {a.threads} เธรด\n", flush=True)
    for name, fn in [("Whisper small int8 (ขนาดเท่าตัวเรา)", bench_small), ("BuzzASR int8", bench_buzz)]:
        load, sec, texts = fn(audio, a.threads)
        print(f"{name}: โหลด {load:.1f} วิ | ทำซับ {sec:.1f} วิ = {sec/dur*60:.0f} วิ ต่อเสียง 1 นาที ({dur/sec:.1f}x เร็วกว่าเวลาจริง)")
        print("   ตัวอย่าง:", (texts[0] if texts else "")[:70], "\n", flush=True)


if __name__ == "__main__":
    main()
