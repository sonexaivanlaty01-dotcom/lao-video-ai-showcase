"""หน้าเว็บแก้ซับ (ใช้ในคอม/วงแลนของทีม)

รัน:   .venv\\Scripts\\python.exe app\\server.py
เปิด:  http://localhost:8765

ทำอะไรได้:
  - อัปโหลดคลิป / เลือกคลิปจาก data\\my_clips
  - กด "สร้างซับ" → BuzzASR (การ์ดจอ) ร่างซับให้
  - ดูวิดีโอไปพร้อมแก้ข้อความ/เวลา แยก/รวม/ลบบรรทัด
  - กด "ตรวจเสร็จ" → บันทึก + ตัดเสียงเป็นท่อนๆ เก็บเป็นข้อมูลเทรนใน data\\my_train
  - ดาวน์โหลด .srt
"""
import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import jiwer  # noqa: E402
import soundfile as sf  # noqa: E402
from common import norm_for_cer  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

CLIPS = ROOT / "data" / "my_clips"
SUBS = ROOT / "data" / "subs"
TRAIN = ROOT / "data" / "my_train"
MEDIA = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".3gp"}
SR = 16000

app = FastAPI(title="Lao Subtitle Editor")
_gpu_lock = threading.Lock()
sys.path.insert(0, str(Path(__file__).parent))
from vocab import Vocab  # noqa: E402

_tok = None


def vocab() -> Vocab:
    return Vocab(SUBS.parent / "vocab.json")  # data/vocab.json (ตามโฟลเดอร์ --data)


_corpus = None


def is_common_word(w: str) -> bool:
    """คำนี้เป็นคำปกติไหม: เจอในข้อความเทรนภาษาลาว (FLEURS + SiangLao) ≥ 20 ครั้ง
    → ถ้าใช่ ห้ามเป็นกฎเดี่ยวๆ (เช่น ແລະ, ຫຼາຍ, ສຽງ) ต้องมีคำข้างเคียงประกอบ"""
    global _corpus
    if _corpus is None:
        parts = []
        for p, col in [(ROOT / "data/fleurs_lo/train.tsv", 2), (ROOT / "data/siang_lao/train.tsv", 1)]:
            if p.exists():
                parts += [ln.split("\t")[col] for ln in p.read_text(encoding="utf-8").splitlines() if ln.count("\t") >= col]
        _corpus = "\n".join(parts)
    return _corpus.count(w) >= 20


def tokenize(text: str) -> list[str]:
    """หั่นข้อความเป็นชิ้นคำด้วยตารางคำลาวของ v3 (ไว้เทียบว่าทีมแก้คำไหน)"""
    global _tok
    if _tok is None:
        from transformers import WhisperTokenizerFast

        _tok = WhisperTokenizerFast.from_pretrained(ROOT / "models" / "full-lao-v3")
    return [_tok.decode([i]) for i in _tok(text, add_special_tokens=False).input_ids]


# ---------- ไฟล์ ----------
def clip_path(name: str) -> Path:
    """กันชื่อไฟล์แปลกๆ (เช่น ..\\..\\) ให้เปิดได้แค่ไฟล์ในโฟลเดอร์คลิป"""
    p = (CLIPS / name).resolve()
    if p.parent != CLIPS.resolve() or not p.is_file() or p.suffix.lower() not in MEDIA:
        raise HTTPException(404, "ไม่พบคลิป")
    return p


def subs_path(name: str) -> Path:
    return SUBS / f"{Path(name).stem}.lo.json"


def read_doc(name: str) -> dict | None:
    p = subs_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_doc(name: str, doc: dict):
    SUBS.mkdir(parents=True, exist_ok=True)
    doc["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = subs_path(name).with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(subs_path(name))


def backup_doc(name: str) -> Path | None:
    """เก็บซับเดิมไว้ก่อนถูกทับ (เช่น กดสร้างซับใหม่) → data/subs/history/<ชื่อ>.<เวลา>.lo.json"""
    p = subs_path(name)
    if not p.exists():
        return None
    hist = SUBS / "history"
    hist.mkdir(parents=True, exist_ok=True)
    dst = hist / f"{Path(name).stem}.{time.strftime('%Y%m%d-%H%M%S')}.lo.json"
    dst.write_bytes(p.read_bytes())
    return dst


def full_text(segs) -> str:
    return " ".join(s.get("text", "").strip() for s in segs)


def ai_cer(doc: dict) -> float | None:
    """AI ผิดไปกี่ % เมื่อเทียบกับที่คนแก้แล้ว (ทั้งคลิป)"""
    ai = doc.get("ai_full")
    if not ai or doc.get("status") != "corrected":
        return None
    ref = norm_for_cer(full_text(doc["segments"]))
    return round(jiwer.cer(ref, norm_for_cer(ai)) * 100, 1)


# ---------- AI ----------
# AI ที่เลือกได้: v3 (ของเรา แม่นสุด เร็วสุด) / buzz (BuzzASR)
ENGINES = {"v3": "AI ของเรา v3", "v4": "AI ของเรา v4", "v5": "AI ของเรา v5", "ultra": "AI ของเรา Ultra (v3+v4)", "ultra2": "AI ของเรา Ultra2 (v3+v5)", "v6": "AI ของเรา v6", "ultra3": "AI ของเรา Ultra3 (v3+v6)", "buzz": "BuzzASR"}
V_MODELS = {  # ตระกูลเดียวกัน ใช้โค้ด V3 ร่วมกัน
    "v3": ROOT / "models" / "full-lao-v3",
    "v4": ROOT / "models" / "full-lao-v4",
    "v5": ROOT / "models" / "full-lao-v5",  # + ข่าว RFA 8.53 ชม. (ใช้ในทีมเท่านั้น)
    "ultra": ROOT / "models" / "full-lao-ultra",  # ผสมสมอง v3+v4 (scripts/merge_models.py)
    "ultra2": ROOT / "models" / "full-lao-ultra2",  # ผสมสมอง v3+v5 (ใช้ในทีมเท่านั้น)
    "v6": ROOT / "models" / "full-lao-v6",  # v5 + เสียงรอบข้างจริง MUSAN (ใช้ในทีมเท่านั้น)
    "ultra3": ROOT / "models" / "full-lao-ultra3",  # ผสมสมอง v3+v6 — ทั้งคลิปดีที่สุด 10.6% (ใช้ในทีมเท่านั้น)
    "ultra3fast": ROOT / "models" / "full-lao-ultra3-ct2-int8",  # Ultra3 ตัวเดิม รันด้วย CTranslate2 (เร็วกว่า)
}
FAST = {"ultra3fast": {"compute_type": "int8_float32", "batch": 16}}  # รันด้วย CTranslate2 (scripts/ct2_asr.py)
ENGINES["ultra3fast"] = "AI ของเรา Ultra3 ⚡ เร็ว"
# OmniASR (CTC) ย่อเป็น ONNX int8 รันบน CPU — ตัวเดียวกับที่จะใส่แอปมือถือ, ใช้เชิงการค้าได้ (scripts/onnx_ctc.py)
ONNX_MODELS = {"omni1int8": ROOT / "models" / "ctc-omni-v1", "omni2int8": ROOT / "models" / "ctc-omni-v2"}
ENGINES["omni1int8"] = "OmniASR v1 ⚡ ย่อ 355MB (ขายได้)"
ENGINES["omni2int8"] = "OmniASR v2 ⚡ ย่อ + LM (ขายได้)"  # beam + char 6-gram ขายได้: ทีม 13.8→11.9
_engines: dict = {}


def get_engine(name: str):
    if name not in ENGINES:
        raise HTTPException(400, f"ไม่รู้จัก AI: {name}")
    if name not in _engines:
        _engines.clear()  # การ์ดจอ 4GB: เปิดทีละตัว ปิดตัวเก่าก่อน
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        if name in V_MODELS:
            from v3_asr import V3

            if not V_MODELS[name].exists():
                raise HTTPException(503, f"ยังไม่มีโมเดล {name} ในเครื่อง")
            if name in FAST:
                from ct2_asr import V3Fast

                _engines[name] = V3Fast(V_MODELS[name], **FAST[name])
            else:
                _engines[name] = V3(model_dir=V_MODELS[name])
        elif name in ONNX_MODELS:
            from onnx_ctc import OnnxCTC

            try:
                _engines[name] = OnnxCTC(ONNX_MODELS[name])
            except FileNotFoundError as e:
                raise HTTPException(503, str(e)) from e
        else:
            from buzz_fast import Buzz

            try:
                _engines[name] = Buzz()
            except ImportError as e:
                raise HTTPException(503, "BuzzASR ใช้ไม่ได้ตอนนี้: Windows บล็อกไฟล์ของมัน ให้เลือก AI ของเรา v3 แทน") from e
    return _engines[name]


import re as _re  # noqa: E402

MARK_ONLY = _re.compile("[ັິ-ຼ່-ໍ\\s]+")


def make_chips(base: str, alt: str, src: str, p: float | None = None) -> list[dict]:
    """เทียบประโยคหลักกับประโยคทางเลือก → จุดที่ต่าง เป็นปุ่ม "แตะเปลี่ยนคำ" {wrong, right, src, p}
    รวมจุดที่ต่างซึ่งอยู่ติดกัน (ห่างกันไม่เกิน 1 ชิ้น) ให้เป็นคำเดียว ไม่หั่นกลางคำ"""
    import difflib

    if not alt or alt == base:
        return []
    a, b = tokenize(base), tokenize(alt)
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    groups = []
    for op in ops:
        if groups and op[1] - groups[-1][2] <= 1:
            groups[-1] = (groups[-1][0], groups[-1][1], op[2], groups[-1][3], op[4])
        else:
            groups.append((op[0], op[1], op[2], op[3], op[4]))
    chips = []
    for _, i1, i2, j1, j2 in groups:
        if i1 == i2 and i1 > 0:  # แทรกคำ: ผูกกับคำข้างหน้า จะได้มีที่ให้แทน
            i1, j1 = i1 - 1, j1 - 1
        w, r = "".join(a[i1:i2]).strip(), "".join(b[j1:j2]).strip()
        if not w or w.replace(" ", "") == r.replace(" ", ""):
            continue
        # กรองปุ่มที่ไม่มีประโยชน์/อันตราย:
        if len(w) > 14 or len(r) > 16:  # เปลี่ยนเกือบทั้งประโยค (มักเป็นตัวที่ผิดกว่า)
            continue
        if len(w.replace(" ", "")) < 2 or MARK_ONLY.fullmatch(w):  # ตัวอักษรเดียว/สระลอย → แทนผิดที่ได้
            continue
        if base.count(w) != 1:  # คำนี้มีหลายที่ในบรรทัด → ไม่รู้จะแทนตรงไหน
            continue
        if any(c["wrong"] == w and c["right"] == r for c in chips):
            continue
        chips.append({"wrong": w, "right": r, "src": src, "p": p})
    return chips[:4]


LINE_MAX = 6.0  # บรรทัดซับยาวสุด (วินาที) เกินนี้แบ่งที่ช่องว่างตามเวลาจริงของคำ


CHUNK = 16  # ถอดทีละ 16 ท่อน แล้วส่งความคืบหน้า (ซับขึ้นทีละช่วง ไม่ต้องรอทั้งคลิป)


def transcribe_file(path: Path, engine: str = "v3", assist: bool = True, progress=None) -> list[dict]:
    """assist=True: หาคำทางเลือก (AI ของเราลองคำอันดับ 2 ตรงจุดลังเล) → ปุ่มทางเลือก
    assist=False: ไม่มีปุ่มทางเลือกเลย
    progress(done_sec, total_sec, segs_so_far): เรียกหลังถอดเสร็จแต่ละช่วง"""
    from buzz_fast import SR as BSR
    from buzz_fast import load_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    from longform import split_by_times

    audio = load_audio(path)
    total = len(audio) / BSR
    spans = get_speech_timestamps(
        audio, VadOptions(max_speech_duration_s=8, min_silence_duration_ms=300, speech_pad_ms=150)
    )
    clips = [audio[s["start"] : s["end"]] for s in spans]
    segs = []
    with _gpu_lock:
        eng = get_engine(engine)
        for c0 in range(0, len(clips), CHUNK):
            part_spans, part_clips = spans[c0 : c0 + CHUNK], clips[c0 : c0 + CHUNK]
            if engine in V_MODELS:
                # ไม่ติ๊ก "ขอคำทางเลือก" = ไม่หาคำทางเลือกเลย → เร็วขึ้น แต่ยังได้ความมั่นใจไว้เตือน ⚠️
                det = eng.transcribe_detailed(part_clips, max_alts=2 if assist else 0)
            else:
                det = [{"text": t, "conf": None, "alts": []} for t in eng.transcribe(part_clips)]
            new = []
            for s, c, d in zip(part_spans, part_clips, det):
                if not d["text"]:
                    continue
                chips, seen = [], set()
                for a in d["alts"]:
                    for ch in make_chips(d["text"], a["text"], "v3", a["p"]):
                        if (ch["wrong"], ch["right"]) not in seen:
                            seen.add((ch["wrong"], ch["right"]))
                            chips.append(ch)
                st, en = s["start"] / BSR, s["end"] / BSR
                wt = []
                # บรรทัดยาวเกิน LINE_MAX: หาเวลาจริงของแต่ละคำ ไว้แบ่งบรรทัดให้ซับขึ้นตรงจังหวะ
                if engine in V_MODELS and " " in d["text"] and en - st > LINE_MAX:
                    try:
                        wt = eng.word_times(c, d["text"])
                    except Exception as e:
                        print("⚠️ หาเวลาคำไม่ได้:", e)
                for a, b, part in split_by_times(d["text"], st, en, wt, LINE_MAX):
                    new.append({
                        "start": round(a, 3), "end": round(b, 3),
                        "text": part, "ai_text": part, "conf": d["conf"],
                        "chips": [ch for ch in chips if ch["wrong"] in part][:6],
                    })
            # ai_text = ที่ AI ฟังได้จริง (ไว้วัดผล/เรียนรู้) · text = หลังแก้ด้วยสมุดคำ
            vocab().apply_segments(new)
            segs += new
            if progress:
                done = part_spans[-1]["end"] / BSR if part_spans else total
                progress(min(done, total), total, segs)
    return segs


ALIGN_PREF = ["ultra3fast", "v3"]  # โมเดลที่ใช้จับเวลารายคำ (ใช้ตัวที่เปิดอยู่ก่อน ถ้าเป็นตระกูล v3)


def _align_engine():
    for n in list(_engines):
        if n in V_MODELS:
            return _engines[n]
    for n in ALIGN_PREF:
        if V_MODELS[n].exists():
            return get_engine(n)
    return None


def align_segments(path: Path, segs: list[dict], only_stale: bool = True) -> int:
    """เติมเวลาจริงของแต่ละคำ: seg["words"] = [{start,end}] + seg["wtext"] = ข้อความตอนที่จับเวลา
    (แก้ข้อความทีหลัง → wtext ไม่ตรง → ตัววาดแบ่งเวลาเอง จนกว่าจะจับเวลาใหม่) คืนจำนวนบรรทัดที่จับเวลา"""
    from buzz_fast import SR as BSR
    from buzz_fast import load_audio
    from word_align import word_spans

    todo = [i for i, s in enumerate(segs) if (s.get("text") or "").strip() and (not only_stale or s.get("wtext") != s["text"])]
    if not todo:
        return 0
    audio = load_audio(path)
    total = len(audio) / BSR
    n = 0
    with _gpu_lock:
        eng = _align_engine()
        if eng is None:
            return 0
        for i in todo:
            s = segs[i]
            a0, a1 = max(0.0, s["start"] - 0.05), min(total, s["end"] + 0.05)
            if a1 - a0 < 0.2 or a1 - a0 > 29:
                continue
            try:
                w = word_spans(eng, audio[int(a0 * BSR) : int(a1 * BSR)], s["text"], a0, s["start"], s["end"])
            except Exception as e:  # จับเวลาไม่ได้ไม่ใช่เรื่องร้ายแรง: ตัววาดแบ่งเวลาเอง
                print("⚠️ จับเวลาคำไม่ได้:", e)
                w = None
            s["words"], s["wtext"] = w, s["text"]
            n += 1
    return n


@app.post("/api/clips/{name}/align")
def align_clip(name: str):
    """จับเวลารายคำให้บรรทัดที่ยังไม่มี/ข้อความเปลี่ยน แล้วบันทึก → คืนเวลาของทุกบรรทัด"""
    path = clip_path(name)
    doc = read_doc(name)
    if not doc:
        raise HTTPException(404, "ยังไม่มีซับ")
    n = align_segments(path, doc["segments"])
    if n:
        write_doc(name, doc)
    return {"aligned": n, "words": [{"text": s["text"], "start": s["start"], "words": s.get("words"), "wtext": s.get("wtext")} for s in doc["segments"]]}


_jobs: dict = {}


def _run_job(jid: str, name: str, engine: str, assist: bool):
    job = _jobs[jid]
    try:
        t0 = time.time()

        def prog(done, total, segs):
            job.update(done=round(done, 1), total=round(total, 1), segments=list(segs))

        segs = transcribe_file(clip_path(name), engine, assist, prog)
        job.update(phase="align")
        try:
            align_segments(clip_path(name), segs, only_stale=False)  # เวลารายคำ → คาราโอเกะตรงเสียง
        except Exception as e:
            print("⚠️ จับเวลาคำไม่สำเร็จ:", e)
        old = read_doc(name) or {}
        backup_doc(name)
        doc = {"source": name, "engine": engine, "status": "draft", "ai_full": full_text(segs), "segments": segs}
        if old.get("style"):
            doc["style"] = old["style"]  # สร้างซับใหม่ แต่สไตล์ที่แต่งไว้ยังอยู่
        write_doc(name, doc)
        doc["seconds"] = round(time.time() - t0, 1)
        doc["audio_seconds"] = job.get("total") or 0  # ความยาวคลิป ไว้คิดว่าเร็วกว่าเวลาจริงกี่เท่า
        job.update(status="done", doc=annotate_odd(doc), segments=None)
    except Exception as e:
        job.update(status="error", error=getattr(e, "detail", None) or str(e))


@app.post("/api/clips/{name}/transcribe_start")
def transcribe_start(name: str, engine: str = "v3", assist: bool = True):
    """เริ่มถอดแบบเบื้องหลัง → ดูความคืบหน้า/ซับที่ได้แล้วที่ GET /api/jobs/{id}"""
    import threading
    import uuid

    clip_path(name)
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {"status": "running", "done": 0, "total": 0, "segments": [], "name": name}
    threading.Thread(target=_run_job, args=(jid, name, engine, assist), daemon=True).start()
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str, since: int = 0):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "ไม่พบงานนี้")
    out = {k: v for k, v in job.items() if k != "segments"}
    if job.get("segments") is not None:
        out["segments"] = job["segments"][since:]
        out["count"] = len(job["segments"])
    return out


def transcribe_voice(path: Path) -> str:
    """🎤 พูดแก้: ถอดเสียงที่ผู้ใช้พูดใส่ไมค์ (ปกติสั้น ไม่เกิน 30 วิ) ด้วย v3"""
    from buzz_fast import load_audio

    audio = load_audio(path)
    if len(audio) < SR // 4:
        raise HTTPException(400, "เสียงสั้นเกินไป ลองกดค้างแล้วพูดใหม่")
    audio = audio[: 30 * SR]
    with _gpu_lock:
        text = get_engine("v3").transcribe([audio])[0]
    return vocab().apply(text)[0]


# ---------- ข้อมูลเทรน ----------
def export_training(name: str, doc: dict) -> int:
    """ตัดเสียงตามบรรทัดที่แก้แล้ว → data/my_train/<ชื่อคลิป>/NNN.wav + metadata.tsv"""
    from buzz_fast import load_audio

    stem = Path(name).stem
    out = TRAIN / stem
    if out.exists():
        shutil.rmtree(out)  # สร้างใหม่ทุกครั้งที่กดตรวจเสร็จ (โฟลเดอร์นี้ระบบสร้างเอง)
    out.mkdir(parents=True)
    audio = load_audio(clip_path(name))
    rows = []
    for i, s in enumerate(doc["segments"], 1):
        text = s.get("text", "").strip()
        dur = s["end"] - s["start"]
        if not text or not (0.5 <= dur <= 30):
            continue
        f = out / f"{i:04}.wav"
        sf.write(f, audio[int(s["start"] * SR) : int(s["end"] * SR)], SR)
        rows.append(f"{stem}/{f.name}\t{text}\t{s['start']}\t{s['end']}\t{name}")
    (out / "metadata.tsv").write_text("file\ttext\tstart\tend\tsource\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def train_stats() -> dict:
    n, secs = 0, 0.0
    for m in TRAIN.glob("*/metadata.tsv"):
        for line in m.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 4:
                n += 1
                secs += float(parts[3]) - float(parts[2])
    return {"lines": n, "minutes": round(secs / 60, 1)}


# ---------- API ----------
@app.get("/api/clips")
def list_clips():
    CLIPS.mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted(CLIPS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.suffix.lower() not in MEDIA:
            continue
        doc = read_doc(f.name)
        items.append(
            {
                "name": f.name,
                "mb": round(f.stat().st_size / 1e6, 1),
                "status": doc.get("status", "draft") if doc else "none",
                "lines": len(doc["segments"]) if doc else 0,
                "ai_cer": ai_cer(doc) if doc else None,
            }
        )
    return {"clips": items, "train": train_stats()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in MEDIA:
        raise HTTPException(400, "รองรับเฉพาะไฟล์วิดีโอ/เสียง")
    CLIPS.mkdir(parents=True, exist_ok=True)
    dest = CLIPS / name
    if dest.exists():
        dest = CLIPS / f"{dest.stem}_{int(time.time())}{dest.suffix}"
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return {"name": dest.name}


@app.get("/media/{name}")
def media(name: str):
    return FileResponse(clip_path(name))  # รองรับการเลื่อนดู (Range) ในตัว


@app.post("/api/clips/{name}/transcribe")
def transcribe(name: str, engine: str = "v3", assist: bool = True):
    path = clip_path(name)
    t0 = time.time()
    segs = transcribe_file(path, engine, assist)
    try:
        align_segments(path, segs, only_stale=False)
    except Exception as e:
        print("⚠️ จับเวลาคำไม่สำเร็จ:", e)
    old_style = (read_doc(name) or {}).get("style")
    backup_doc(name)  # ซับเก่า (รวมที่คนแก้ไว้) ไม่หาย
    doc = {
        **({"style": old_style} if old_style else {}),
        "source": name,
        "engine": {"v3": "lao-v3", "v4": "lao-v4", "ultra": "lao-ultra", "buzz": "buzzasr-ct2-int8"}.get(engine, engine),
        "status": "draft",
        "ai_full": full_text(segs),
        "segments": segs,
    }
    write_doc(name, doc)
    doc["seconds"] = round(time.time() - t0, 1)
    return annotate_odd(doc)


def annotate_odd(doc: dict) -> dict:
    """เติม odd = คำ/ท่อนที่ไม่เหมือนภาษาลาวที่เคยเห็น (จากคลังข้อความ 2 แสนประโยค) ไว้ไฮไลต์ — ไม่บันทึกลงไฟล์"""
    try:
        from char_lm import load_lm, odd_fragments

        from lao_ortho import error_fragments

        lm = load_lm()
        for s in doc.get("segments", []):
            t = s.get("text", "")
            frags = error_fragments(t)  # ผิดหลักการเรียงสระ/วรรณยุกต์ (แน่นอนกว่า)
            if lm:
                frags = merge_frags(frags, odd_fragments(t, lm))
            s["odd"], s["odd_text"] = frags, t
    except Exception as e:  # ไม่มีไฟล์คลังข้อความ → ข้ามไป ใช้งานได้ปกติ
        print("⚠️ ข้ามไฮไลต์คำแปลก:", e)
    return doc


class OddBody(BaseModel):
    text: str


@app.post("/api/odd")
def odd(body: OddBody):
    from char_lm import odd_fragments
    from lao_ortho import error_fragments

    return {"odd": merge_frags(error_fragments(body.text), odd_fragments(body.text))}


def merge_frags(a: list[str], b: list[str]) -> list[str]:
    """รวมท่อนไฮไลต์ 2 แหล่ง ไม่ให้ซ้ำ/ทับกัน (เก็บท่อนที่ยาวกว่า)"""
    out = list(a)
    for f in b:
        hit = next((i for i, g in enumerate(out) if f in g or g in f), None)
        if hit is None:
            out.append(f)
        elif len(f) > len(out[hit]):
            out[hit] = f
    return out


@app.get("/api/clips/{name}/subs")
def get_subs(name: str):
    clip_path(name)
    doc = read_doc(name)
    if not doc:
        raise HTTPException(404, "ยังไม่มีซับ")
    return annotate_odd(doc)


class Seg(BaseModel):
    start: float
    end: float
    text: str
    ai_text: str | None = None
    conf: float | None = None  # ความมั่นใจของ AI (ต่ำ = ไฮไลต์)
    chips: list[dict] | None = None  # ปุ่มแตะเปลี่ยนคำ
    words: list[dict] | None = None  # เวลาจริงของแต่ละคำ [{start,end}] (ซับคาราโอเกะ)
    wtext: str | None = None  # ข้อความตอนที่จับเวลา (ไม่ตรง text = ต้องจับใหม่)


@app.post("/api/asr")
async def asr_voice(file: UploadFile = File(...)):
    """🎤 พูดแก้: รับเสียงจากไมค์ (webm/ogg/wav) → ข้อความลาว"""
    import tempfile

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "ไฟล์เสียงใหญ่เกินไป")
    suffix = Path(file.filename or "voice.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"voice{suffix}"
        p.write_bytes(data)
        from starlette.concurrency import run_in_threadpool

        t0 = time.time()
        text = await run_in_threadpool(transcribe_voice, p)  # ไม่ให้เว็บค้างระหว่าง AI ฟัง
    return {"text": text, "seconds": round(time.time() - t0, 1)}


class SaveBody(BaseModel):
    segments: list[Seg]
    status: str = "draft"  # draft / corrected


@app.put("/api/clips/{name}/subs")
def save_subs(name: str, body: SaveBody):
    clip_path(name)
    doc = read_doc(name) or {"source": name, "engine": "manual"}
    segs = sorted((s.model_dump() for s in body.segments), key=lambda s: s["start"])
    for s in segs:
        s["start"], s["end"] = round(max(0, s["start"]), 3), round(max(s["start"], s["end"]), 3)
    doc["segments"] = segs
    doc["status"] = "corrected" if body.status == "corrected" else "draft"
    exported = export_training(name, doc) if doc["status"] == "corrected" else 0
    learned = 0
    if doc["status"] == "corrected":  # เรียนรู้คำที่ทีมแก้ → เป็นคำแนะนำในสมุดคำ
        try:
            from common import clean_text

            learned = vocab().learn(segs, tokenize, is_common_word, clean_text)
        except Exception as e:  # ไม่ให้การเรียนรู้ทำให้บันทึกพัง
            print("เรียนรู้สมุดคำไม่สำเร็จ:", e)
    write_doc(name, doc)
    return {
        "ok": True,
        "status": doc["status"],
        "exported": exported,
        "learned": learned,
        "ai_cer": ai_cer(doc),
        "train": train_stats(),
    }


# ---------- สมุดคำ ----------
class Rule(BaseModel):
    wrong: str
    right: str


@app.get("/api/vocab")
def get_vocab():
    return vocab().load()


@app.post("/api/vocab/rules")
def add_rule(r: Rule):
    try:
        return vocab().add_rule(r.wrong, r.right)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/vocab/rules/delete")
def delete_rule(r: Rule):
    v = vocab()
    d = v.load()
    d["rules"] = [x for x in d["rules"] if x["wrong"] != r.wrong]
    v.save(d)
    return d


@app.post("/api/vocab/rules/toggle")
def toggle_rule(r: Rule):
    v = vocab()
    d = v.load()
    for x in d["rules"]:
        if x["wrong"] == r.wrong:
            x["enabled"] = not x.get("enabled", True)
    v.save(d)
    return d


@app.post("/api/vocab/suggestions/accept")
def accept_suggestion(r: Rule):
    try:
        return vocab().add_rule(r.wrong, r.right, source="auto")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/vocab/suggestions/dismiss")
def dismiss_suggestion(r: Rule):
    v = vocab()
    d = v.load()
    d["suggestions"] = [s for s in d["suggestions"] if not (s["wrong"] == r.wrong and s["right"] == r.right)]
    v.save(d)
    return d


@app.post("/api/clips/{name}/apply-vocab")
def apply_vocab(name: str):
    """ใช้สมุดคำกับซับที่มีอยู่แล้ว (เช่น เพิ่งเพิ่มคำใหม่)"""
    doc = get_subs(name)
    n = vocab().apply_segments(doc["segments"])
    write_doc(name, doc)
    doc["fixed"] = n
    return doc


def srt_time(t: float) -> str:
    h, rem = divmod(int(max(0.0, t) * 1000), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


@app.get("/api/clips/{name}/srt")
def get_srt(name: str):
    doc = get_subs(name)
    lines = []
    for i, s in enumerate((s for s in doc["segments"] if s["text"].strip()), 1):
        lines += [str(i), f"{srt_time(s['start'])} --> {srt_time(s['end'])}", s["text"].strip(), ""]
    fname = f"{Path(name).stem}.lo.srt"
    return PlainTextResponse(
        "\n".join(lines),
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"},
    )


# ---------- สไตล์ซับแอนิเมชัน + ส่งออก MP4 ----------
STYLE_DEFAULT = ROOT / "data" / "caption_style.json"  # สไตล์ล่าสุดที่ทีมใช้ → คลิปใหม่เริ่มจากอันนี้


class StyleBody(BaseModel):
    style: dict


@app.get("/api/style/default")
def get_default_style():
    try:
        return {"style": json.loads(STYLE_DEFAULT.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"style": None}


@app.put("/api/clips/{name}/style")
def save_style(name: str, body: StyleBody):
    """เก็บสไตล์ไว้กับซับของคลิปนี้ (+ เป็นค่าเริ่มต้นของคลิปถัดไป)"""
    clip_path(name)
    doc = read_doc(name)
    if doc:  # คลิปที่ยังไม่มีซับ: เก็บเป็นค่าเริ่มต้นอย่างเดียว (ไม่สร้างไฟล์ซับเปล่า)
        doc["style"] = body.style
        write_doc(name, doc)
    STYLE_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    STYLE_DEFAULT.write_text(json.dumps(body.style, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


_export_lock = threading.Lock()


def _run_export(jid: str, name: str, base_url: str, fps: int):
    job = _jobs[jid]
    try:
        from common import ffmpeg_exe
        from export_video import export

        doc = read_doc(name) or {}
        if not doc.get("segments"):
            raise RuntimeError("คลิปนี้ยังไม่มีซับ")
        with _export_lock:  # ส่งออกทีละงาน (Edge + ffmpeg กินเครื่อง)
            t0 = time.time()
            job.update(phase="align", pct=0)
            try:  # บรรทัดที่แก้ข้อความหลังจับเวลา → จับเวลารายคำใหม่ก่อนวาด
                if align_segments(clip_path(name), doc["segments"]):
                    write_doc(name, doc)
            except Exception as e:
                print("⚠️ จับเวลาคำไม่สำเร็จ:", e)
            out = export(clip_path(name), doc["segments"], doc.get("style") or {}, base_url, ffmpeg_exe(), fps,
                         progress=lambda **kw: job.update(**kw))
        job.update(status="done", file=out.name, mb=round(out.stat().st_size / 1e6, 1), seconds=round(time.time() - t0, 1))
    except Exception as e:
        job.update(status="error", error=str(e)[:800])


@app.post("/api/clips/{name}/export")
def export_start(name: str, request: Request, fps: int = 30):
    """เริ่มส่งออก MP4 ฝังซับ (เบื้องหลัง) → ดูความคืบหน้าที่ GET /api/jobs/{id}"""
    import uuid

    clip_path(name)
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {"status": "running", "kind": "export", "phase": "start", "pct": 0, "name": name}
    base = str(request.base_url)
    threading.Thread(target=_run_export, args=(jid, name, base, max(10, min(60, fps))), daemon=True).start()
    return {"job": jid}


@app.get("/api/exports/{fname}")
def get_export(fname: str):
    p = (ROOT / "data" / "exports" / Path(fname).name)
    if not p.is_file():
        raise HTTPException(404, "ไม่พบไฟล์")
    return FileResponse(p, media_type="video/mp4", filename=p.name)


# ---------- ฟังเทียบแบบปิดชื่อ (scripts/make_ab.py เตรียมคู่ไว้) ----------
def ab_dir() -> Path:
    return SUBS.parent / "ab"


def ab_votes() -> dict:
    f = ab_dir() / "votes.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


@app.get("/api/ab")
def ab_pairs():
    f = ab_dir() / "pairs.json"
    if not f.exists():
        raise HTTPException(404, "ยังไม่มีแบบทดสอบ: รัน scripts/make_ab.py ก่อน")
    data = json.loads(f.read_text(encoding="utf-8"))
    votes = ab_votes()
    # ไม่ส่งชื่อโมเดลไปหน้าเว็บ → คนฟังไม่รู้ว่าตัวไหนเป็นตัวไหน
    pairs = [{"id": p["id"], "A": p["A"], "B": p["B"], "vote": votes.get(str(p["id"]))} for p in data["pairs"]]
    return {"pairs": pairs}


@app.get("/api/ab/audio/{pid}")
def ab_audio(pid: int):
    f = ab_dir() / "wav" / f"{pid:03}.wav"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/wav")


class AbVote(BaseModel):
    id: int
    choice: str  # A / B / same / none


@app.post("/api/ab/vote")
def ab_vote(v: AbVote):
    if v.choice not in {"A", "B", "same", "none"}:
        raise HTTPException(400, "ตัวเลือกไม่ถูกต้อง")
    votes = ab_votes()
    votes[str(v.id)] = v.choice
    (ab_dir() / "votes.json").write_text(json.dumps(votes, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True, "done": len(votes)}


@app.get("/api/ab/result")
def ab_result():
    data = json.loads((ab_dir() / "pairs.json").read_text(encoding="utf-8"))
    votes = ab_votes()
    score = {m: 0 for m in data["models"]} | {"เท่ากัน": 0, "ผิดทั้งคู่": 0}
    for p in data["pairs"]:
        c = votes.get(str(p["id"]))
        if c in ("A", "B"):
            score[p[f"{c}_is"]] += 1
        elif c == "same":
            score["เท่ากัน"] += 1
        elif c == "none":
            score["ผิดทั้งคู่"] += 1
    return {"voted": len(votes), "total": len(data["pairs"]), "score": score}


# ---------- 🎙️ อัดเสียงอ่านประโยค (record.html) → ข้อมูลเทรนของเราเอง ขายได้ ----------
# ประโยคมาจากชุดที่ใช้เชิงการค้าได้: KitTzk (CC-BY-4.0, ภาษาพูด) + Common Voice (CC0) + FLEURS train (CC-BY-4.0)
# เสียงเก็บใน data/my_record/<ผู้พูด>/NNNNN.wav + metadata.tsv, ความยินยอมใน consent.json
CONSENT_TEXT = ("ข้าพเจ้ายินยอมให้นำเสียงที่อัดนี้ไปใช้สอน AI ถอดเสียง/สังเคราะห์เสียง รวมถึงใช้ในผลิตภัณฑ์ที่ขายได้ "
                "(ไม่เผยแพร่เสียงต้นฉบับ และไม่สร้างเสียงปลอมเลียนแบบข้าพเจ้าโดยไม่ขออนุญาตก่อน)\n"
                "ຂ້ອຍຍິນຍອມໃຫ້ນຳສຽງທີ່ອັດນີ້ໄປໃຊ້ສອນ AI ຖອດສຽງ/ສັງເຄາະສຽງ ລວມທັງໃຊ້ໃນຜະລິດຕະພັນທີ່ຂາຍໄດ້ "
                "(ບໍ່ເຜີຍແຜ່ສຽງຕົ້ນສະບັບ ແລະ ບໍ່ສ້າງສຽງປອມເລຽນແບບຂ້ອຍໂດຍບໍ່ໄດ້ຂໍອະນຸຍາດກ່ອນ)")
_sentences: list[str] | None = None


def record_dir() -> Path:
    return SUBS.parent / "my_record"


def sentence_pool() -> list[str]:
    global _sentences
    if _sentences is None:
        import csv as _csv
        import re

        from common import clean_text, load_fleurs

        banned = {norm_for_cer(r["text"]) for s in ("dev", "test") for r in load_fleurs(s)}
        seen, out = set(), []
        srcs = [ROOT / "data/extra/kittzk/keep.tsv", ROOT / "data/extra/cv/keep.tsv"]
        rows = []
        for p in srcs:
            if p.exists():
                with open(p, encoding="utf-8") as fh:
                    rows += [r["text"] for r in _csv.DictReader(fh, delimiter="\t")]
        rows += [r["text"] for r in load_fleurs("train")]
        for t in rows:
            t = clean_text(t, strip_punct=False).strip()
            n = norm_for_cer(t)
            if 8 <= len(t) <= 90 and n not in banned and n not in seen and not re.search(r"[A-Za-z()\[\]/]", t):
                seen.add(n)
                out.append(t)
        _sentences = out
    return _sentences


def speaker_slug(name: str) -> str:
    import re

    s = re.sub(r"[^\w຀-໿-]+", "_", name.strip())[:40].strip("_")
    if not s:
        raise HTTPException(400, "ใส่ชื่อก่อน")
    return s


def speaker_done(slug: str) -> set[str]:
    m = record_dir() / slug / "metadata.tsv"
    if not m.exists():
        return set()
    return {ln.split("\t")[1] for ln in m.read_text(encoding="utf-8").splitlines()[1:] if "\t" in ln}


@app.get("/api/record/next")
def record_next(speaker: str, skip: str = ""):
    import random

    slug = speaker_slug(speaker)
    done = speaker_done(slug)
    pool = sentence_pool()
    left = [s for s in pool if s not in done and s != skip]
    if not left:
        raise HTTPException(404, "อ่านครบทุกประโยคแล้ว ขอบคุณมาก")
    cons = (record_dir() / slug / "consent.json").exists()
    total = sum(len(speaker_done(d.name)) for d in record_dir().glob("*") if d.is_dir()) if record_dir().exists() else 0
    return {"text": random.choice(left), "done": len(done), "consent": cons, "total_all": total,
            "consent_text": CONSENT_TEXT}


class Consent(BaseModel):
    speaker: str
    agree: bool


@app.post("/api/record/consent")
def record_consent(c: Consent):
    if not c.agree:
        raise HTTPException(400, "ต้องยินยอมก่อนจึงอัดได้")
    d = record_dir() / speaker_slug(c.speaker)
    d.mkdir(parents=True, exist_ok=True)
    (d / "consent.json").write_text(json.dumps({"name": c.speaker, "text": CONSENT_TEXT, "agreed_at": time.strftime(
        "%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True}


@app.post("/api/record")
async def record_save(speaker: str = Form(...), text: str = Form(...), file: UploadFile = File(...)):
    import subprocess
    import tempfile

    import numpy as np
    from common import ffmpeg_exe

    slug = speaker_slug(speaker)
    d = record_dir() / slug
    if not (d / "consent.json").exists():
        raise HTTPException(400, "ต้องยินยอมก่อนจึงอัดได้")
    if text not in set(sentence_pool()):
        raise HTTPException(400, "ประโยคไม่อยู่ในรายการ")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "ไฟล์ใหญ่เกินไป")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / ("in" + (Path(file.filename or "a.webm").suffix or ".webm"))
        src.write_bytes(data)
        raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(src), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    dur = len(x) / SR
    if not 0.8 <= dur <= 30:
        raise HTTPException(400, f"เสียงยาว {dur:.1f} วิ — ลองอัดใหม่")
    if float(np.sqrt(np.mean(x ** 2))) < 0.003:
        raise HTTPException(400, "เสียงเบาเกินไป (ไมค์ปิดอยู่หรือเปล่า?) — ลองอัดใหม่")
    meta = d / "metadata.tsv"
    if not meta.exists():
        meta.write_text("file\ttext\tduration\tspeaker\ttime\n", encoding="utf-8")
    n = len(meta.read_text(encoding="utf-8").splitlines())
    fn = f"{n:05}.wav"
    sf.write(d / fn, x, SR)
    with open(meta, "a", encoding="utf-8") as fh:
        fh.write(f"{fn}\t{text}\t{dur:.2f}\t{slug}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return {"ok": True, "done": len(speaker_done(slug)), "seconds": round(dur, 1)}


app.mount("/fonts", StaticFiles(directory=ROOT / "fonts"), name="fonts")  # ฟอนต์ลาว + ตัววาดซับ (ใช้ร่วมกัน)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    import os

    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = ให้มือถือในวงแลนเดียวกันเข้าได้")
    ap.add_argument("--clips", default=None, help="โฟลเดอร์คลิป (ค่าเริ่มต้น data\\my_clips)")
    ap.add_argument("--data", default=None, help="โฟลเดอร์เก็บ subs/ และ my_train/ (ไว้ทดสอบ ไม่ปนข้อมูลจริง)")
    a = ap.parse_args()
    if a.clips:
        CLIPS = Path(a.clips).resolve()
    if a.data:
        SUBS, TRAIN = Path(a.data) / "subs", Path(a.data) / "my_train"
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"📂 คลิป: {CLIPS}\n🌐 เปิด http://localhost:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port)
