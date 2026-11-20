"""ดาวน์โหลดเฉพาะเสียงของคลิป YouTube CC-BY ที่ผ่าน yt_cc_search.py แล้วตัดเป็นท่อน 2–15 วิ ตามช่วงเงียบ
(ยังไม่มีเฉลย — ขั้นต่อไปให้ AI ถอดเสียงเป็นเฉลย: scripts/pseudo_label.py)

  .venv\\Scripts\\python.exe scripts\\yt_cc_download.py [--max-hours 60]
→ data/yt_cc/seg/<id>_NNNN.wav + data/yt_cc/segments.tsv (file, video, start, end, duration)
→ data/yt_cc/credits.tsv (id, title, channel, license, url) — ต้องใส่เครดิตตามสัญญา CC-BY
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yt_dlp
from common import ROOT, ffmpeg_exe

sys.stdout.reconfigure(encoding="utf-8")
D = ROOT / "data" / "yt_cc"
SR = 16000


def segments(x: np.ndarray, lo: float = 2.0, hi: float = 15.0) -> list[tuple[int, int]]:
    """ตัดตามช่วงเงียบ แล้วรวมช่วงพูดติดกันให้ยาว lo–hi วินาที"""
    iv = librosa.effects.split(x, top_db=32, frame_length=1024, hop_length=256)
    out, cur = [], None
    for s, e in iv:
        if cur and (e - cur[0]) / SR <= hi and (s - cur[1]) / SR < 0.6:
            cur[1] = e
            continue
        if cur:
            out.append(tuple(cur))
        cur = [s, e]
    if cur:
        out.append(tuple(cur))
    res = []
    for s, e in out:
        while (e - s) / SR > hi:  # ท่อนยาวเกิน (ไม่มีช่วงเงียบ) → ผ่าเป็นชิ้นเท่าๆ กัน
            res.append((s, s + int(hi * SR)))
            s += int(hi * SR)
        if (e - s) / SR >= lo:
            res.append((s, e))
    pad = int(0.15 * SR)
    return [(max(0, s - pad), min(len(x), e + pad)) for s, e in res]


def main():
    max_h = float(sys.argv[sys.argv.index("--max-hours") + 1]) if "--max-hours" in sys.argv else 60
    per_ch = float(sys.argv[sys.argv.index("--max-per-channel") + 1]) if "--max-per-channel" in sys.argv else 3.0
    raw = list(csv.DictReader(open(D / "candidates.tsv", encoding="utf-8"), delimiter="\t"))
    # จำกัดชั่วโมงต่อช่อง (กันช่องเดียวครองข้อมูล เช่น เทศน์ยาวๆ) + ข้ามช่องเพลง
    cands, used = [], {}
    for c in sorted(raw, key=lambda c: float(c["duration"])):
        ch = c["channel"]
        if "music" in ch.lower():
            continue
        if used.get(ch, 0) + float(c["duration"]) / 3600 > per_ch:
            continue
        used[ch] = used.get(ch, 0) + float(c["duration"]) / 3600
        cands.append(c)
    print(f"จะโหลด {len(cands)} คลิป ≈ {sum(used.values()):.1f} ชม. จาก {len(used)} ช่อง", flush=True)
    (D / "seg").mkdir(parents=True, exist_ok=True)
    seg_f, cred_f = D / "segments.tsv", D / "credits.tsv"
    done = set()
    if seg_f.exists():
        done = {r["video"] for r in csv.DictReader(open(seg_f, encoding="utf-8"), delimiter="\t")}
    else:
        seg_f.write_text("file\tvideo\tstart\tend\tduration\n", encoding="utf-8")
    if not cred_f.exists():
        cred_f.write_text("id\ttitle\tchannel\tlicense\turl\n", encoding="utf-8")
    total = sum(float(r["duration"]) for r in csv.DictReader(open(seg_f, encoding="utf-8"), delimiter="\t"))
    for i, c in enumerate(cands):
        if total / 3600 >= max_h:
            break
        vid = c["id"]
        if vid in done:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            opts = {"quiet": True, "no_warnings": True, "format": "bestaudio/best", "outtmpl": f"{tmp}/a.%(ext)s"}
            try:
                with yt_dlp.YoutubeDL(opts) as y:
                    info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
                if "creative commons" not in (info.get("license") or "").lower():  # ตรวจซ้ำตอนโหลดจริง
                    print(f"  ข้าม {vid}: สัญญาไม่ใช่ CC", flush=True)
                    continue
                src = next(Path(tmp).glob("a.*"))
                raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(src), "-f", "f32le", "-ac", "1",
                                      "-ar", str(SR), "-"], capture_output=True, check=True).stdout
            except Exception as e:
                print(f"  โหลด {vid} ไม่ได้: {str(e)[:120]}", flush=True)
                continue
        x = np.frombuffer(raw, dtype=np.float32)
        rows = []
        for k, (s, e) in enumerate(segments(x)):
            fn = f"seg/{vid}_{k:04}.wav"
            sf.write(D / fn, x[s:e], SR)
            rows.append(f"{fn}\t{vid}\t{s / SR:.2f}\t{e / SR:.2f}\t{(e - s) / SR:.2f}")
        with open(seg_f, "a", encoding="utf-8") as fh:
            fh.write("".join(r + "\n" for r in rows))
        with open(cred_f, "a", encoding="utf-8") as fh:
            fh.write(f"{vid}\t{c['title']}\t{c['channel']}\t{info.get('license')}\thttps://www.youtube.com/watch?v={vid}\n")
        total += sum(float(r.split("\t")[-1]) for r in rows)
        print(f"[{i + 1}/{len(cands)}] {vid} {len(rows)} ท่อน | รวม {total / 3600:.2f} ชม. | {c['title'][:50]}", flush=True)
    print(f"✅ เสร็จ: {total / 3600:.2f} ชม. → {seg_f}")


if __name__ == "__main__":
    main()
