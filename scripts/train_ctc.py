"""เทรน AI ถอดเสียงแบบ CTC (ฟังรอบเดียว ออกข้อความทั้งประโยคพร้อมกัน — เร็ว เหมาะกับมือถือ)

โมเดลฐาน (ใช้เชิงการค้าได้): facebook/wav2vec2-xls-r-300m (Apache 2.0) — ค่าเริ่มต้น
                              facebook/w2v-bert-2.0 (MIT) — แม่นกว่า ใหญ่กว่า (ใส่ --base)
ตารางตัวอักษร: สร้างจากข้อความเทรน (ตัวอักษรลาว + ตัวเลข + อังกฤษที่เจอ) ช่องว่าง = "|"

  python scripts/train_ctc.py --extra data/siang_lao/train.tsv data/extra/kittzk/keep.tsv ... \
      --out models/ctc-xlsr-v1 --epochs 15 --batch 16 --fp16

ทดสอบโค้ดในเครื่อง (ไม่กี่ก้าว):  --limit 32 --max-steps 4 --batch 2
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import jiwer
import librosa
import numpy as np
import torch
from augment import augment_audio, list_noise
from common import MODELS, ROOT, clean_text, load_fleurs, norm_for_cer
from train import load_tsv

sys.stdout.reconfigure(encoding="utf-8")
SR = 16000


def build_vocab(texts, out: Path) -> dict:
    chars = sorted({c for t in texts for c in t if c != " "})
    vocab = {"<pad>": 0, "<unk>": 1, "|": 2}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    return vocab


LAO_EXTRA = {"ໜ": "ห", "ໝ": "ห", "ຽ": "ย", "ົ": "ั"}  # อักษรลาวที่ไม่มีคู่ไทยตรงตำแหน่ง → ใกล้เคียงสุด


def init_head_from_omni(model, vocab: dict, base: Path, log) -> None:
    """ตั้งค่าเริ่มหัว CTC ลาวจากแถวอักษรไทยของหัวเดิม omniASR (ฐานได้ยินภาษาลาวถูกแต่เขียนเป็นอักษรไทย)
    อักษรลาว U+0E80–0EFF ตรงกับอักษรไทย U+0E00–0E7F (ลบ 0x80) เกือบทุกตัว"""
    h = torch.load(base / "omni_head.pt", map_location="cpu")
    piece = {p: i for i, p in enumerate(h["pieces"])}
    w, b = model.lm_head.weight.data, model.lm_head.bias.data
    b.fill_(-5.0)  # แถวที่ไม่ได้ตั้งค่า (อังกฤษ ตัวเลข เครื่องหมาย) เริ่มแบบ "ไม่ค่อยออก" จะได้ไม่แย่งอักษรลาว
    hit = 0
    for ch, idx in vocab.items():
        if ch == "<pad>":
            cand = [h["pieces"][0]]  # blank ของ CTC (omniASR: <s> = id 0 ขั้นระหว่างทุกตัวอักษร)
        elif ch == "|":
            cand = [" ", "▁"]
        elif len(ch) == 1 and 0x0E80 <= ord(ch) <= 0x0EFF:
            # ฐานเขียนลาวได้ดี (FLEURS) แต่บางครั้งเขียนเป็นไทย (ภาษาพูด) → ใช้แถวลาวก่อน ไม่มีค่อยใช้แถวไทยคู่กัน
            cand = [ch, LAO_EXTRA.get(ch) or chr(ord(ch) - 0x80)]
        elif len(ch) == 1:
            cand = [ch]  # อังกฤษ/ตัวเลข/เครื่องหมาย ที่ฐานมี
        else:
            continue
        j = next((piece[c] for c in cand if c in piece), None)
        if j is not None:
            w[idx] = h["weight"][j].to(w.dtype)
            b[idx] = h["bias"][j].to(b.dtype)
            hit += 1
    log(f"ตั้งหัว CTC จาก omniASR: {hit}/{len(vocab)} ตัวอักษร")


def encode(text: str, vocab: dict) -> list[int]:
    return [vocab.get("|" if c == " " else c, 1) for c in text]


def decode(ids, inv: dict) -> str:
    out, prev = [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(inv.get(int(i), ""))
        prev = i
    return "".join(out).replace("|", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default=str(MODELS / "ctc-xlsr-v1"))
    ap.add_argument("--epochs", type=float, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-sec", type=float, default=16.0, help="ตัดคลิปยาวเกินนี้ทิ้ง (หน่วยความจำ)")
    ap.add_argument("--aug", type=float, default=0.5)
    ap.add_argument("--noise-dir", default=str(ROOT / "data" / "noise"))
    ap.add_argument("--log", default=None)
    ap.add_argument("--head-init", choices=["none", "omni"], default="none",
                    help="omni = ตั้งหัวลาวจากแถวอักษรไทยของ omniASR (ต้องมี omni_head.pt ในโฟลเดอร์ --base)")
    a = ap.parse_args()
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)
    torch.manual_seed(0)
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    logf = open(a.log, "a", encoding="utf-8") if a.log else None

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        if logf:
            logf.write(s + "\n")
            logf.flush()

    rows = load_fleurs("train", a.limit)
    for tsv in a.extra:
        ex = load_tsv(tsv)
        rows += ex[: a.limit] if a.limit else ex
        log(f"  + {tsv}: {len(ex)} คลิป")
    for r in rows:
        r["text"] = clean_text(r["text"])
    rows = [r for r in rows if r["text"]]
    out = Path(a.out)
    vocab = build_vocab([r["text"] for r in rows], out)
    inv = {v: k for k, v in vocab.items()}
    log(f"ข้อมูล {len(rows)} คลิป | ตารางตัวอักษร {len(vocab)} ตัว")
    dev_rows = load_fleurs("dev", a.eval_n)

    fe = AutoFeatureExtractor.from_pretrained(a.base)
    model = AutoModelForCTC.from_pretrained(
        a.base, vocab_size=len(vocab), pad_token_id=0, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        mask_time_prob=0.05, layerdrop=0.05, ignore_mismatched_sizes=True)
    if a.head_init == "omni":
        init_head_from_omni(model, vocab, Path(a.base), log)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.to(dev).train()
    noise = list_noise(a.noise_dir) if a.aug > 0 else []
    log(f"เสียงรบกวน {len(noise)} ไฟล์ | อุปกรณ์ {dev}")

    def load(r, aug):
        x = librosa.load(str(r["file"]), sr=SR)[0]
        if aug and a.aug > 0:
            x = augment_audio(x, noise, a.aug)
        return x

    def feats(xs):
        b = fe(xs, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
        key = "input_features" if "input_features" in b else "input_values"
        return b[key].to(dev), b["attention_mask"].to(dev)

    @torch.no_grad()
    def evaluate():
        model.eval()
        hyps, refs = [], []
        for i in range(0, len(dev_rows), 8):
            chunk = dev_rows[i : i + 8]
            x, m = feats([load(r, False) for r in chunk])
            with torch.autocast(dev, enabled=a.fp16 and dev == "cuda"):
                lg = model(x, attention_mask=m).logits
            hyps += [decode(p, inv) for p in lg.argmax(-1).cpu().numpy()]
            refs += [clean_text(r["text"]) for r in chunk]
        model.train()
        return 100 * jiwer.cer([norm_for_cer(t) for t in refs], [norm_for_cer(h) or "-" for h in hyps]), hyps[:2], refs[:2]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.005)
    steps_per_epoch = max(1, len(rows) // a.batch)
    total = a.max_steps or int(steps_per_epoch * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler(enabled=a.fp16 and dev == "cuda")
    best, step, t0 = 1e9, 0, time.time()
    log(f"เริ่มเทรน {total} ก้าว ({a.epochs} รอบ)")
    while step < total:
        random.shuffle(rows)
        for i in range(0, len(rows) - a.batch + 1, a.batch):
            batch, xs = [], []
            for r in rows[i : i + a.batch]:
                x = load(r, True)
                if len(x) / SR <= a.max_sec:
                    batch.append(r)
                    xs.append(x)
            if not batch:
                continue
            x, m = feats(xs)
            labels = [torch.tensor(encode(r["text"], vocab)) for r in batch]
            lab = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100).to(dev)
            with torch.autocast(dev, enabled=a.fp16 and dev == "cuda"):
                loss = model(x, attention_mask=m, labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 20 == 0:
                el = (time.time() - t0) / 60
                log(f"step {step}/{total} loss {loss.item():.3f} ใช้ไป {el:.1f} นาที เหลือ ~{el / step * (total - step):.0f} นาที")
            if step % a.eval_every == 0 or step == total:
                cer, h, r = evaluate()
                log(f">>> step {step}: CER {cer:.1f}%\n    ถูก: {r[0]}\n    AI : {h[0]}")
                if cer < best:
                    best = cer
                    model.save_pretrained(out / "best")
                    fe.save_pretrained(out / "best")
                    (out / "best" / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
                    log(f"    💾 บันทึกตัวที่ดีที่สุด (CER {cer:.1f}%)")
            if step >= total:
                break
    log(f"เสร็จ! ใช้เวลา {(time.time() - t0) / 60:.0f} นาที | CER ดีที่สุด {best:.1f}% | {out / 'best'}")


if __name__ == "__main__":
    main()
