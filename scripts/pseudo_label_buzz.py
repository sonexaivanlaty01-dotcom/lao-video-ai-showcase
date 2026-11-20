"""ข้อ 6: ครูสอนลูกศิษย์ — ให้ BuzzASR (ครู) ทำ "เฉลย" ให้เสียงลาวที่ยังไม่มีข้อความ

แล้วให้ AI ของเรา (v2) ฟังซ้ำเป็น "ผู้ตรวจ": ถ้าสองตัวตอบใกล้กัน → เชื่อได้ → เก็บเป็นข้อมูลเทรน
ถ้าตอบต่างกันมาก → ไม่เก็บ (ส่งให้คนตรวจในหน้าแก้ซับแทน)

  .venv\\Scripts\\python.exe scripts\\pseudo_label_buzz.py <โฟลเดอร์คลิป/เสียงลาว> --out data\\pseudo
  แล้วเทรน: train.py ... --extra data\\pseudo\\train.tsv

⚠️ ใช้เฉพาะเสียงที่มีสิทธิ์ใช้ (คลิปของทีมเอง / ได้รับอนุญาต / สัญญาอนุญาตเปิด)
⚠️ เฉลยจาก AI ไม่ถูก 100% → ควรผสมกับข้อมูลที่คนตรวจแล้วเสมอ อย่าใช้อย่างเดียว
"""
import argparse
import sys
from pathlib import Path

import common  # noqa: F401
import jiwer
import soundfile as sf
from common import clean_text, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
MEDIA = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".3gp", ".flac"}
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="โฟลเดอร์ไฟล์เสียง/วิดีโอภาษาลาว")
    ap.add_argument("--out", default=str(common.ROOT / "data" / "pseudo"))
    ap.add_argument("--max-diff", type=float, default=0.15, help="ยอมให้ครูกับผู้ตรวจต่างกันได้ไม่เกินนี้ (0.15 = 15%)")
    ap.add_argument("--checker", default="models/lora-small-lo-v2/best", help="LoRA ของ AI ผู้ตรวจ")
    a = ap.parse_args()

    from buzz_fast import Buzz, load_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    files = sorted(p for p in Path(a.src).rglob("*") if p.suffix.lower() in MEDIA)
    print(f"พบ {len(files)} ไฟล์", flush=True)
    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)

    # 1) ตัดเป็นท่อนตามช่วงที่มีเสียงพูด
    segs = []
    for f in files:
        audio = load_audio(f)
        spans = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=20, min_silence_duration_ms=400, speech_pad_ms=200))
        for i, s in enumerate(spans):
            clip = audio[s["start"] : s["end"]]
            if len(clip) >= SR:  # สั้นกว่า 1 วินาทีไม่เอา
                segs.append({"src": f, "i": i, "audio": clip, "start": s["start"] / SR})
    print(f"ได้ {len(segs)} ท่อน รวม {sum(len(s['audio']) for s in segs)/SR/60:.1f} นาที", flush=True)
    if not segs:
        return

    # 2) ครู (BuzzASR) ทำเฉลย
    buzz = Buzz()
    teacher = buzz.transcribe([s["audio"] for s in segs])
    del buzz
    import torch

    torch.cuda.empty_cache()

    # 3) ผู้ตรวจ (AI ของเรา) ฟังซ้ำ
    import evaluate as ev

    processor, model = ev.load_model("openai/whisper-small", a.checker)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    paths = []
    for k, s in enumerate(segs):
        p = tmp / f"{k}.wav"
        sf.write(p, s["audio"], SR)
        paths.append(p)
    checker = ev.transcribe(processor, model, paths)

    # 4) เก็บเฉพาะท่อนที่สองตัวเห็นตรงกัน
    rows, kept_sec, review = [], 0.0, []
    for s, t, c, p in zip(segs, teacher, checker, paths):
        t_clean = clean_text(t)
        diff = jiwer.cer(norm_for_cer(t_clean), norm_for_cer(c)) if t_clean else 1.0
        name = f"{s['src'].stem}_{s['i']:04}.wav"
        if t_clean and diff <= a.max_diff:
            p.replace(out / "wav" / name)
            rows.append(f"wav/{name}\t{t_clean}\t{diff:.3f}\t{s['src'].name}\t{s['start']:.2f}")
            kept_sec += len(s["audio"]) / SR
        else:
            review.append(f"{s['src'].name}\t{s['start']:.2f}\t{diff:.3f}\t{t}\t{c}")
            p.unlink()
    tmp.rmdir()
    (out / "train.tsv").write_text("file\ttext\tdiff\tsource\tstart\n" + "\n".join(rows) + "\n", encoding="utf-8")
    (out / "need_review.tsv").write_text("source\tstart\tdiff\tครู(BuzzASR)\tผู้ตรวจ(ของเรา)\n" + "\n".join(review) + "\n", encoding="utf-8")
    print(f"✅ เก็บ {len(rows)}/{len(segs)} ท่อน ({kept_sec/60:.1f} นาที) → {out/'train.tsv'}")
    print(f"   ไม่เก็บ {len(review)} ท่อน (สองตัวเห็นต่างกันเกิน {a.max_diff:.0%}) → {out/'need_review.tsv'}")


if __name__ == "__main__":
    main()
