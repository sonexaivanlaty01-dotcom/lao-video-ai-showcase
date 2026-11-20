"""ส่งออกวิดีโอ MP4 ที่ฝังซับแอนิเมชัน

วิธีทำ (ภาพตรงกับหน้าเว็บ 100% เพราะใช้ตัววาดเดียวกัน fonts/caprender.js):
  1. เปิด render.html ใน Edge แบบไม่มีหน้าต่าง (Playwright) ขนาดเท่าวิดีโอ พื้นหลังโปร่งใส
  2. ไล่เวลาทีละเฟรมเฉพาะช่วงที่มีซับ → ถามตัววาดว่า "ภาพนี้ key อะไร"
     key ซ้ำ = ภาพเดิม → ไม่ต้องถ่ายใหม่ (ซับนิ่งถ่ายแค่บรรทัดละ 1 รูป)
  3. ทำรายการ "รูปไหน แสดงกี่วินาที" (ffconcat) แล้วให้ ffmpeg วางทับวิดีโอ
ผลลัพธ์: data/exports/<ชื่อคลิป>_lo_<เวลา>.mp4 (ไม่เขียนทับไฟล์เก่า)
"""
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "data" / "exports"


def probe(path: Path, ffmpeg: str) -> dict:
    """ความยาว + ขนาดภาพที่แสดงจริง (หมุนตาม metadata ของมือถือแล้ว)"""
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if ffmpeg.endswith(".exe") else "ffprobe"))
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, encoding="utf-8").stdout
    d = json.loads(out or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    info = {"duration": float(d.get("format", {}).get("duration") or 0), "w": 0, "h": 0, "has_video": v is not None}
    if v:
        w, h = int(v["width"]), int(v["height"])
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        rot = rot or int((v.get("tags") or {}).get("rotate", 0) or 0)
        if abs(rot) % 180 == 90:
            w, h = h, w
        info.update(w=w, h=h)
    return info


def export(clip: Path, segments: list, style: dict, base_url: str, ffmpeg: str, fps: int = 30, progress=None) -> Path:
    from playwright.sync_api import sync_playwright

    def prog(**kw):
        if progress:
            progress(**kw)

    info = probe(clip, ffmpeg)
    if not info["has_video"]:
        raise RuntimeError("ไฟล์นี้ไม่มีภาพวิดีโอ (เป็นไฟล์เสียง) — ส่งออกเป็น MP4 ฝังซับไม่ได้")
    W, H = info["w"] - info["w"] % 2, info["h"] - info["h"] % 2
    dur = info["duration"] or max((s["end"] for s in segments), default=0)
    segs = sorted([s for s in segments if (s.get("text") or "").strip() and s["end"] > s["start"]], key=lambda s: s["start"])
    EXPORTS.mkdir(parents=True, exist_ok=True)
    out = EXPORTS / f"{clip.stem}_lo_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    tmp = Path(tempfile.mkdtemp(prefix="laoexp_"))
    try:
        # ---------- 1–2: วาดซับทีละเฟรม ----------
        events = []  # (เวลาเริ่ม, ไฟล์รูป)
        cache = {}
        total_frames = sum(max(1, int((s["end"] - s["start"]) * fps)) for s in segs) or 1
        done = 0
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
            page.goto(base_url.rstrip("/") + "/render.html")
            page.wait_for_function("window.renderReady === true", timeout=30000)
            page.evaluate("cfg => window.setup(cfg)", {"segments": segs, "style": style, "w": W, "h": H})

            def shot(key):
                if key not in cache:
                    f = tmp / f"c{len(cache):05}.png"
                    page.screenshot(path=str(f), omit_background=True, type="png")
                    cache[key] = f
                return cache[key]

            blank = shot(page.evaluate("t => window.frameAt(t)", -1.0))
            events.append((0.0, blank))
            for s in segs:
                n = max(1, int((s["end"] - s["start"]) * fps))
                last_key = None
                for i in range(n):
                    t = s["start"] + i / fps
                    key = page.evaluate("t => window.frameAt(t)", t + 1e-4)
                    if key != last_key:
                        events.append((t, shot(key)))
                        last_key = key
                    done += 1
                    if done % 15 == 0:
                        prog(phase="render", pct=round(80 * done / total_frames), frames=len(cache))
                events.append((s["end"], blank))
            browser.close()
        # รวมช่วงเวลาที่ติดกันและรูปเดียวกัน, ตัดช่วงยาว 0
        tl = []
        for t, f in sorted(events, key=lambda e: e[0]):
            if tl and abs(tl[-1][0] - t) < 1e-6:
                tl[-1] = (t, f)
            elif not tl or tl[-1][1] != f:
                tl.append((t, f))
        lines = ["ffconcat version 1.0"]
        for i, (t, f) in enumerate(tl):
            nxt = tl[i + 1][0] if i + 1 < len(tl) else max(dur, t + 0.1)
            lines += [f"file '{f.name}'", f"duration {max(0.001, nxt - t):.4f}"]
        lines.append(f"file '{tl[-1][1].name}'")
        (tmp / "list.ffconcat").write_text("\n".join(lines), encoding="utf-8")

        # ---------- 3: ffmpeg วางซับทับวิดีโอ ----------
        prog(phase="encode", pct=80, frames=len(cache))
        cmd = [ffmpeg, "-y", "-v", "error", "-progress", "pipe:1", "-i", str(clip),
               "-f", "concat", "-safe", "0", "-i", str(tmp / "list.ffconcat"),
               "-filter_complex", f"[0:v]scale={W}:{H}[b];[1:v]format=rgba[c];[b][c]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[v]",
               "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        for line in proc.stdout:
            m = re.match(r"out_time_us=(\d+)", line.strip())
            if m and dur:
                prog(phase="encode", pct=80 + min(19, round(19 * int(m.group(1)) / 1e6 / dur)), frames=len(cache))
        err = proc.stderr.read()
        if proc.wait() != 0 or not out.exists():
            raise RuntimeError("ffmpeg ล้มเหลว: " + err[-600:])
        prog(phase="done", pct=100, frames=len(cache))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
