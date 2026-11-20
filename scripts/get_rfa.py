"""ทดลอง: ประกอบชุดเสียงข่าว RFA ภาษาลาว จากเฉลยของ WorldSpeech (disco-eth/WorldSpeech, config la_lo)

WorldSpeech แจกแค่ "ข้อความเฉลย (คนพิมพ์) + เวลาเริ่ม/จบในไฟล์เสียงต้นฉบับ" ไม่แจกเสียง
→ สคริปต์นี้หาหน้าข่าวบน rfa.org จากรหัสข่าว → เอาลิงก์ mp3 → ตัดท่อนตามเวลา → wav 16kHz

  .venv\\Scripts\\python.exe scripts\\get_rfa.py --hours 10

⚠️ เสียงข่าว RFA มีลิขสิทธิ์ และเฉลย WorldSpeech เป็น CC-BY-NC → ใช้ในทีม/วิจัยเท่านั้น ห้ามใช้เชิงการค้า
ผลลัพธ์: data/extra/rfa/wav/*.wav + all.tsv (file, text, duration, source, license, article)
แล้วตรวจคุณภาพ: scripts/check_extra.py data/extra/rfa
"""
import argparse
import csv
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
from common import ROOT, clean_text, ffmpeg_exe, load_fleurs, norm_for_cer
from huggingface_hub import hf_hub_download, list_repo_files

sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "data" / "extra" / "rfa"
SR = 16000
CATS = ["economy", "politics", "social", "society", "human-rights", "culture", "health", "environment", "education",
        "labor", "labour", "agriculture", "corruption", "land", "refugees", "drug", "drugs", "tourism", "news",
        "ecognomy", "daily", "international", "religion", "women", "trafficking", "dam", "mining", "legal"]
UA = {"User-Agent": "Mozilla/5.0 (research; lao-asr)"}


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find_mp3(slug: str) -> str | None:
    """ลองหมวดข่าวทีละหมวดบนเว็บจริง (ผิดหมวด = 404 เร็ว) ไม่เจอค่อยถาม Wayback"""
    for cat in CATS:
        try:
            html = get(f"https://www.rfa.org/lao/daily/{cat}/{slug}.html").decode("utf-8", "ignore")
        except Exception:
            continue
        m = re.search(r"https://audio\.rfa\.org/[^\"'\s<]+\.mp3", html)
        return m.group(0) if m else None
    return None


def load_segments():
    fs = sorted(f for f in list_repo_files("disco-eth/WorldSpeech", repo_type="dataset") if f.startswith("data/la_lo/") and "train" in f)
    cols = ["segment_id", "human_transcript", "source_start_s", "source_end_s", "duration", "cer"]
    frames = [pl.read_parquet(hf_hub_download("disco-eth/WorldSpeech", f, repo_type="dataset"), columns=cols) for f in fs]
    t = pl.concat(frames)
    arts = defaultdict(list)
    for r in t.iter_rows(named=True):
        arts[r["segment_id"].rsplit("_", 1)[0].removeprefix("rfa_lao_")].append(r)
    return arts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10)
    ap.add_argument("--max-cer", type=float, default=0.5, help="ตัดท่อนที่ ASR ของ WorldSpeech ต่างจากเฉลยเกินนี้ (มักจัดเวลาเพี้ยน)")
    ap.add_argument("--out", default=None, help="โฟลเดอร์ผลลัพธ์ (ไม่ใส่ = data/extra/rfa) เช่น data/extra/rfa2 สำหรับชุดเพิ่ม")
    ap.add_argument("--skip", type=int, default=0, help="ข้ามข่าว N อันแรกในลำดับสุ่ม (ชุดก่อนลองไปแล้ว)")
    ap.add_argument("--skip-old", action="store_true", help="ไม่เก็บท่อนที่มีตัวสะกดเก่า ຣ ตั้งแต่ตอนโหลด (นับชั่วโมงเฉพาะท่อนใช้ได้)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = ROOT / a.out
    banned = {norm_for_cer(r["text"]) for s in ("dev", "test") for r in load_fleurs(s)}
    arts = load_segments()
    print(f"WorldSpeech la_lo: {len(arts)} ข่าว", flush=True)
    names = [k for k in arts if re.search(r"\d{14}$", k)]  # ข่าวที่มีรหัสวันที่ (ลิงก์แบบเดิม)
    random.seed(1)
    random.shuffle(names)
    names = names[a.skip :]

    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    rows, secs, tried, miss = [], 0.0, 0, 0
    for slug in names:
        if secs >= a.hours * 3600:
            break
        tried += 1
        mp3 = find_mp3(slug)
        if not mp3:
            miss += 1
            continue
        try:
            raw = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", mp3, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                 capture_output=True, check=True, timeout=300).stdout
        except Exception:
            miss += 1
            continue
        audio = np.frombuffer(raw, dtype=np.float32)
        for r in arts[slug]:
            text = clean_text(r["human_transcript"] or "", strip_punct=False).strip()
            s, e = float(r["source_start_s"]), float(r["source_end_s"])
            if not text or norm_for_cer(text) in banned or not (0.5 <= e - s <= 30) or e * SR > len(audio):
                continue
            if r["cer"] is not None and float(r["cer"]) > a.max_cer:
                continue
            if a.skip_old and "ຣ" in text:
                continue
            fn ="wav/" + re.sub(r"[^A-Za-z0-9_\-]", "_", r["segment_id"][8:])[-80:] + ".wav"
            sf.write(OUT / fn, audio[int(s * SR) : int(e * SR)], SR)
            rows.append([fn, text, f"{e - s:.2f}", "RFA Lao (WorldSpeech la_lo)", "CC-BY-NC / RFA copyright", slug, mp3, f"{s:.3f}", f"{e:.3f}"])
            secs += e - s
        print(f"  ข่าว {tried} (ไม่เจอ {miss}) | สะสม {secs/3600:.2f} ชม. {len(rows)} ท่อน", flush=True)
        time.sleep(0.5)  # ไม่ยิงเว็บเขาถี่เกินไป
    with open(OUT / "all.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["file", "text", "duration", "source", "license", "article", "mp3", "start", "end"])
        w.writerows(rows)
    print(f"✅ RFA: {len(rows)} ท่อน = {secs/3600:.2f} ชม. จาก {tried - miss} ข่าว (หาไม่เจอ {miss}) → {OUT}")


def rebuild(manifest: str, keep: str):
    """บน Kaggle: สร้างเสียงใหม่จาก all.tsv (มีลิงก์ mp3 + เวลา) เฉพาะท่อนที่อยู่ใน keep.tsv (ผ่านการตรวจในเครื่องเราแล้ว)"""
    with open(keep, encoding="utf-8") as fh:
        ok = {r["file"] for r in csv.DictReader(fh, delimiter="\t")}
    with open(manifest, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["file"] in ok]
    by_mp3 = defaultdict(list)
    for r in rows:
        by_mp3[r["mp3"]].append(r)
    out = Path(manifest).resolve().parent  # wav อยู่ข้าง all.tsv (rfa หรือ rfa2)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    done = 0
    for k, (mp3, rs) in enumerate(by_mp3.items(), 1):
        try:
            raw = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", mp3, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                 capture_output=True, check=True, timeout=300).stdout
        except Exception as ex:
            print("  โหลดไม่ได้:", mp3, ex)
            continue
        audio = np.frombuffer(raw, dtype=np.float32)
        for r in rs:
            sf.write(out / r["file"], audio[int(float(r["start"]) * SR) : int(float(r["end"]) * SR)], SR)
            done += 1
        if k % 25 == 0:
            print(f"  {k}/{len(by_mp3)} ไฟล์เสียง | {done} ท่อน", flush=True)
    print(f"✅ สร้างใหม่ {done}/{len(rows)} ท่อน")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        rebuild(sys.argv[2], sys.argv[3])
    else:
        main()
