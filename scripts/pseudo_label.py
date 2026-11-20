"""ให้ AI ใส่เฉลยให้ท่อนเสียงที่ยังไม่มีเฉลย (data/yt_cc/segments.tsv) แล้วคัดเฉพาะท่อนที่ AI 2 ตัวถอดตรงกัน

ครู (ใช้เชิงการค้าได้ทั้งคู่ — ไม่ใช้ Ultra/v3–v6 ที่เทรนจากข้อมูลห้ามขาย):
  qwen  = LuoYiSULIXAY/qwen3-asr-lao-H (Apache-2.0) — แม่นสุดบนคลิปทีมในกลุ่มที่ขายได้, ช้า → รันบน Kaggle T4
  omni  = models/ctc-omni-v1 (ของเราเอง) — เร็ว 38 เท่า รันในเครื่องได้

  python scripts/pseudo_label.py qwen [--shard 0/2] [--batch 16]   → data/yt_cc/labels_qwen[.0].tsv
  python scripts/pseudo_label.py omni                               → data/yt_cc/labels_omni.tsv
  python scripts/pseudo_label.py filter [--max-cer 25]              → data/extra/ytcc/keep.tsv (พร้อมเทรน)
"""
import csv
import os
import shutil
import sys
import time

import jiwer
import librosa
import torch
from common import MODELS, ROOT, clean_text, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
D = ROOT / "data" / "yt_cc"
SR = 16000


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def audio_path(f: str):
    p = D / f
    return p if p.exists() else p.with_suffix(".flac")  # บน Kaggle แพ็กเป็น FLAC


def load_segs():
    return list(csv.DictReader(open(D / "segments.tsv", encoding="utf-8"), delimiter="\t"))


def write(path, done: dict):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("file\ttext\n")
        for f, t in done.items():
            fh.write(f"{f}\t{t.replace(chr(9), ' ').replace(chr(10), ' ')}\n")


def read(path) -> dict:
    if not path.exists():
        return {}
    return {r["file"]: r["text"] for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}


def run_qwen():
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import SUPPORTED_LANGUAGES

    if "Lao" not in SUPPORTED_LANGUAGES:
        SUPPORTED_LANGUAGES.append("Lao")
    k, n = (int(v) for v in arg("--shard", "0/1").split("/"))
    bs = int(arg("--batch", "16"))
    out = D / (f"labels_qwen.{k}.tsv" if n > 1 else "labels_qwen.tsv")
    dev = f"cuda:{k % max(1, torch.cuda.device_count())}"
    asr = Qwen3ASRModel.from_pretrained(arg("--repo", "LuoYiSULIXAY/qwen3-asr-lao-H"), dtype=torch.float16,
                                        device_map=dev, max_inference_batch_size=bs, max_new_tokens=160)
    segs = [s for i, s in enumerate(load_segs()) if i % n == k]
    done = read(out)
    todo = [s for s in segs if s["file"] not in done]
    t0, audio = time.time(), 0.0
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        xs = [librosa.load(audio_path(s["file"]), sr=SR)[0] for s in chunk]
        audio += sum(len(x) for x in xs) / SR
        res = asr.transcribe(audio=[(x, SR) for x in xs], language=["Lao"] * len(xs))
        for s, r in zip(chunk, res):
            done[s["file"]] = r.text
        if (i // bs) % 20 == 0:
            write(out, done)
            el = time.time() - t0
            print(f"[{dev}] {len(done)}/{len(segs)} | {audio / el:.1f} เท่าของเวลาจริง | {chunk[0]['file']}: {res[0].text[:40]}",
                  flush=True)
    write(out, done)
    print(f"✅ qwen shard {k}/{n}: {len(done)} ท่อน → {out}")


def run_omni():
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    from train_ctc import decode

    m = MODELS / arg("--model", "ctc-omni-v1")
    fe = AutoFeatureExtractor.from_pretrained(m)
    model = AutoModelForCTC.from_pretrained(m).to("cuda").eval()
    import json

    inv = {v: k for k, v in json.loads((m / "vocab.json").read_text(encoding="utf-8")).items()}
    out = D / "labels_omni.tsv"
    done = read(out)
    todo = [s for s in load_segs() if s["file"] not in done]
    for i, s in enumerate(todo):
        x = librosa.load(audio_path(s["file"]), sr=SR)[0]
        with torch.no_grad():
            lg = model(fe(x, sampling_rate=SR, return_tensors="pt").input_values.to("cuda")).logits
        done[s["file"]] = decode(lg.argmax(-1)[0].cpu().numpy(), inv)
        if i % 500 == 0:
            write(out, done)
            print(f"omni {len(done)} | {s['file']}: {done[s['file']][:40]}", flush=True)
    write(out, done)
    print(f"✅ omni: {len(done)} ท่อน → {out}")


def run_filter():
    # 35: ช่วง 25–35% ส่วนใหญ่ Qwen ถูก แต่ omni v1 ผิด (เราใช้ข้อความของ Qwen) — 25% เหลือแค่ 2.85 ชม., 35% = 5.23 ชม.
    max_cer = float(arg("--max-cer", "35"))
    q = {}
    for p in sorted(D.glob("labels_qwen*.tsv")):
        q |= read(p)
    o = read(D / "labels_omni.tsv")
    segs = {s["file"]: s for s in load_segs()}
    # ตรวจภาษาจากเสียง (scripts/yt_lid.py): ทิ้งคลิปที่ลาวได้ไม่ถึงครึ่งของท่อนที่สุ่มฟัง (ไทย/ม้ง/อังกฤษ/เพลงสวด)
    lid_f = D / "lid.tsv"
    lao_vid = None
    if lid_f.exists():
        lao_vid = {r["video"] for r in csv.DictReader(open(lid_f, encoding="utf-8"), delimiter="\t")
                   if int(r["lao_votes"]) * 2 >= int(r["n"])}
    dst = ROOT / "data" / "extra" / "ytcc"
    (dst / "wav").mkdir(parents=True, exist_ok=True)
    keep, stat, secs = [], {"ไม่ครบ": 0, "เสียงไม่ใช่ลาว": 0, "ไม่ใช่ลาว": 0, "ไม่ตรงกัน": 0, "ซ้ำวน": 0}, 0.0
    for f, tq in q.items():
        to = o.get(f)
        if to is None or f not in segs:
            stat["ไม่ครบ"] += 1
            continue
        if lao_vid is not None and segs[f]["video"] not in lao_vid:
            stat["เสียงไม่ใช่ลาว"] += 1
            continue
        tq = clean_text(tq, strip_punct=False).strip()
        nq, no = norm_for_cer(tq), norm_for_cer(to)
        lao = sum(0x0E80 <= ord(c) <= 0x0EFF for c in nq)
        if len(nq) < 4 or lao < 0.8 * len(nq.replace(" ", "")):
            stat["ไม่ใช่ลาว"] += 1
            continue
        if any(nq.count(nq[i:i + 6]) >= 4 for i in range(0, max(1, len(nq) - 6), 6)):
            stat["ซ้ำวน"] += 1  # AI หลอน พูดวนซ้ำ
            continue
        cer = 100 * jiwer.cer(nq, no or "-")
        if cer > max_cer:
            stat["ไม่ตรงกัน"] += 1
            continue
        name = f.split("/")[-1]
        shutil.copyfile(audio_path(f), dst / "wav" / name)
        dur = float(segs[f]["duration"])
        keep.append(f"wav/{name}\t{tq}\t{dur:.2f}\t{cer:.1f}")
        secs += dur
    (dst / "keep.tsv").write_text("file\ttext\tduration\tagree_cer\n" + "\n".join(keep) + "\n", encoding="utf-8")
    print(f"✅ ผ่าน {len(keep)}/{len(q)} ท่อน = {secs / 3600:.2f} ชม. (CER ระหว่าง 2 AI ≤ {max_cer}%) | ตัดออก {stat}")
    print(f"   → {dst / 'keep.tsv'}")


if __name__ == "__main__":
    if os.name == "nt":
        os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
    {"qwen": run_qwen, "omni": run_omni, "filter": run_filter}[sys.argv[1]]()
