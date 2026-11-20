"""ให้ Gemini (Google) ร่างซับภาษาลาวจากวิดีโอ/เสียง → ได้ไฟล์ .srt + .json

ขั้นตอน:
  1. FFmpeg แยกเสียงจากวิดีโอ (mp3 16kHz mono เล็กมาก อัปโหลดเร็ว)
  2. ส่งเสียงให้ Gemini ฟัง ขอผลเป็นประโยคพร้อมเวลาเริ่ม/จบ
  3. บันทึก .srt (ใช้กับโปรแกรมตัดต่อ/YouTube) และ .json (ไว้แก้ในแอป + เป็นข้อมูลเทรน)

ต้องมีไฟล์ .env ที่มี GEMINI_API_KEY=... (ดู .env.example)

วิธีรัน:
  .venv\\Scripts\\python.exe scripts\\gemini_subs.py data\\my_clips\\คลิป.mp4
  .venv\\Scripts\\python.exe scripts\\gemini_subs.py data\\my_clips      (ทำทุกไฟล์ในโฟลเดอร์)

⚠️ Gemini แบบฟรี: Google อาจนำข้อมูลที่ส่งไปใช้พัฒนาระบบ อย่าส่งคลิปที่เป็นความลับ
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.stdout.reconfigure(encoding="utf-8")

MEDIA = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".3gp"}

PROMPT = """You are a professional Lao subtitler.
Transcribe the speech in this audio exactly as spoken, in LAO SCRIPT (ພາສາລາວ) only.
- Never write Lao words in Thai script.
- Keep English words / brand names as they are spoken (Latin letters are OK).
- Split into short subtitle lines: max ~42 characters or ~6 seconds each, cut at natural pauses.
- start/end are in seconds from the beginning of the audio, as accurate as possible.
- Do not translate, summarize or add anything that was not said.
- If a part is music/noise with no speech, skip it."""

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "text": {"type": "string"},
        },
        "required": ["start", "end", "text"],
    },
}


def extract_audio(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(dst)],
        check=True,
    )


def srt_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(t * 1000), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segs, path: Path):
    lines = []
    for i, s in enumerate(segs, 1):
        lines += [str(i), f"{srt_time(s['start'])} --> {srt_time(s['end'])}", s["text"].strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def transcribe(client, model: str, media: Path, out_dir: Path):
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.mp3"
        extract_audio(media, audio)
        print(f"  🎧 แยกเสียงแล้ว ({audio.stat().st_size/1e6:.1f} MB) → ส่งให้ Gemini ({model})...")
        uploaded = client.files.upload(file=audio)
        while uploaded.state and uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        t0 = time.time()
        resp = client.models.generate_content(
            model=model,
            contents=[uploaded, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=SCHEMA, temperature=0
            ),
        )
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    segs = sorted(json.loads(resp.text), key=lambda s: s["start"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / media.stem
    write_srt(segs, base.with_suffix(".lo.srt"))
    base.with_suffix(".lo.json").write_text(
        json.dumps(
            {"source": media.name, "engine": f"gemini:{model}", "status": "draft", "segments": segs},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"  ✅ {len(segs)} บรรทัด ใน {time.time()-t0:.0f} วิ → {base.with_suffix('.lo.srt').name}")
    for s in segs[:3]:
        print(f"     [{srt_time(s['start'])}] {s['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="ไฟล์วิดีโอ/เสียง หรือโฟลเดอร์")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
    ap.add_argument("--out", default=str(ROOT / "data" / "subs"))
    args = ap.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("❌ ยังไม่มี GEMINI_API_KEY ในไฟล์ .env (ดูวิธีใน .env.example)")

    p = Path(args.path)
    files = [f for f in sorted(p.iterdir()) if f.suffix.lower() in MEDIA] if p.is_dir() else [p]
    if not files:
        sys.exit("❌ ไม่พบไฟล์วิดีโอ/เสียง")

    client = genai.Client()
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        try:
            transcribe(client, args.model, f, Path(args.out))
        except Exception as e:
            print(f"  ❌ ผิดพลาด: {e}")


if __name__ == "__main__":
    main()
