"""เทรน Whisper ให้ฟังภาษาลาว (LoRA = สอนเพิ่มแค่ส่วนเล็กๆ ของโมเดล ประหยัดหน่วยความจำ)

ออกแบบให้รันได้บนการ์ดจอ 4GB (GTX 1650):
  - LoRA: เทรนแค่ ~3% ของพารามิเตอร์
  - fp16 + gradient checkpointing: ลดหน่วยความจำ
  - batch เล็ก + gradient accumulation: เหมือน batch ใหญ่แต่ใช้แรมน้อย

ตัวอย่าง:
  # ทดสอบสั้นๆ ว่าไม่พัง
  .venv\\Scripts\\python.exe scripts\\train.py --max-steps 10 --limit 80
  # เทรนจริง
  .venv\\Scripts\\python.exe scripts\\train.py --epochs 3
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import common  # noqa: F401
import jiwer
import librosa
import torch
from augment import augment_audio, list_noise, spec_augment
from common import MODELS, ROOT, clean_text, load_fleurs, norm_for_cer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

import sys
import warnings

import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
sys.stdout.reconfigure(encoding="utf-8")


class Tee:
    """พิมพ์ขึ้นจอ + บันทึกลงไฟล์ log ไปพร้อมกัน"""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()


MAX_LABEL = 440
# GTX 16xx คำนวณ fp16 แล้วเพี้ยน → ค่าเริ่มต้นใช้ fp32 (เปิด --fp16 บน T4/RTX ได้)
USE_FP16 = False
DEV = "cuda"


def save_last(model, out: Path, state: dict):
    """เซฟระหว่างทาง: เขียนลงโฟลเดอร์ใหม่ก่อนแล้วค่อยสลับชื่อ → ไฟดับกลางคันก็ยังมีตัวเก่าที่ใช้ได้"""
    import shutil

    new, last = out / "last_new", out / "last"
    if new.exists():
        shutil.rmtree(new)
    model.save_pretrained(new)
    torch.save(state, new / "state.pt")
    if last.exists():
        shutil.rmtree(last)
    new.rename(last)


def find_last(out: Path):
    for name in ("last", "last_new"):  # last_new = ไฟดับตอนกำลังสลับชื่อพอดี
        if (out / name / "state.pt").exists():
            return out / name
    return None


def load_tsv(path):
    """อ่าน tsv ที่มีหัวคอลัมน์ file, text (เช่น data/siang_lao/train.tsv, data/my_train/*/metadata.tsv)"""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    fi, ti = head.index("file"), head.index("text")
    rows = []
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) > max(fi, ti) and p[ti].strip():
            f = path.parent / p[fi]
            if not f.exists():
                f = path.parent.parent / p[fi]  # metadata.tsv ของ my_train เก็บ path แบบ <คลิป>/NNNN.wav
            if f.exists():
                rows.append({"file": f, "text": p[ti].strip()})
    return rows


class LaoSpeech(Dataset):
    def __init__(self, rows, processor, aug_p=0.0, noise_files=()):
        self.rows, self.p = rows, processor
        self.aug_p, self.noise = aug_p, list(noise_files)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio, _ = librosa.load(str(r["file"]), sr=16000)
        if self.aug_p > 0:
            audio = augment_audio(audio, self.noise, self.aug_p)
        feats = self.p.feature_extractor(audio, sampling_rate=16000).input_features[0]
        if self.aug_p > 0:
            feats = spec_augment(feats, len(audio) // 160, self.aug_p)
        labels = self.p.tokenizer(r["text"]).input_ids
        return {"feats": torch.tensor(feats), "labels": labels}


class Collate:
    def __init__(self, sot_id):
        self.sot = sot_id

    def __call__(self, batch):
        feats = torch.stack([b["feats"] for b in batch])
        labels = [b["labels"][1:] if b["labels"][0] == self.sot else b["labels"] for b in batch]
        n = max(len(l) for l in labels)
        lab = torch.full((len(labels), n), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, : len(l)] = torch.tensor(l)
        return feats, lab


@torch.no_grad()
def quick_eval(model, processor, rows):
    model.eval()
    hyps = []
    for i in range(0, len(rows), 8):
        audios = [librosa.load(str(r["file"]), sr=16000)[0] for r in rows[i : i + 8]]
        feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_FP16):
            ids = model.generate(input_features=feats.to(DEV), language="lo", task="transcribe", max_new_tokens=MAX_LABEL)
        hyps += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
    model.train()
    return jiwer.cer([norm_for_cer(r["text"]) for r in rows], [norm_for_cer(h) for h in hyps]), hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--out", default=str(MODELS / "lora-small-lo"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="ใช้แค่ N คลิป (ไว้ทดสอบ)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--log", default=None, help="บันทึกข้อความลงไฟล์นี้ด้วย")
    ap.add_argument("--extra", nargs="*", default=[], help="ไฟล์ tsv เพิ่มเติม (คอลัมน์ file, text; file อิงจากโฟลเดอร์ของ tsv)")
    ap.add_argument("--init-adapter", default=None, help="เริ่มจาก LoRA ที่เคยเทรนแล้ว (เช่น models\\lora-small-lo\\best)")
    ap.add_argument("--save-every", type=int, default=25, help="เซฟระหว่างทางทุก N ขั้น (0 = ปิด)")
    ap.add_argument("--resume", action="store_true", help="เทรนต่อจากจุดที่เซฟล่าสุดใน --out (ใช้ตอนคอมดับ/หลับกลางคัน)")
    ap.add_argument("--aug", type=float, default=0.5, help="โอกาสผสมเสียงรบกวนต่อคลิป 0–1 (0 = ปิด)")
    ap.add_argument("--noise-dir", default=str(ROOT / "data" / "noise"), help="โฟลเดอร์เพลง/เสียงรอบข้างไว้ผสม (ไม่มีก็ได้)")
    ap.add_argument("--keep-punct", action="store_true", help="ไม่ตัดเครื่องหมายวรรคตอนออกจากข้อความเทรน")
    ap.add_argument("--device", default="cuda", help="cuda หรือ cpu (cpu ไว้ทดสอบโค้ดเท่านั้น ช้ามาก)")
    ap.add_argument("--full", action="store_true", help="เทรนทั้งตัวแทน LoRA (ใช้กับ --base models\\whisper-small-lao-vocab-v2 บน Kaggle)")
    ap.add_argument("--embed-lr", type=float, default=None, help="lr ของตารางคำ (ค่าเริ่มต้น = lr x10)")
    ap.add_argument("--freeze-encoder", action="store_true", help="--full แบบประหยัด: ไม่เทรนส่วนฟังเสียง")
    args = ap.parse_args()
    global DEV
    DEV = args.device
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(args.log)
        print(f"=== เริ่มเทรน {time.strftime('%Y-%m-%d %H:%M')} ===", flush=True)
    global USE_FP16
    USE_FP16 = args.fp16

    random.seed(0)
    torch.manual_seed(0)
    processor = WhisperProcessor.from_pretrained(args.base, language="lao", task="transcribe")
    tok = processor.tokenizer

    train_rows = load_fleurs("train", args.limit)
    for tsv in args.extra:
        extra = load_tsv(tsv)
        print(f"  + {tsv}: {len(extra)} คลิป")
        train_rows += extra[: args.limit] if args.limit else extra
    n_all = len(train_rows)
    n_fixed = 0
    for r in train_rows:  # สะกดแบบเดียวกันทุกชุดข้อมูล (ສໍາ→ສຳ, ເເ→ແ ...) + ตัดวรรคตอน
        new = clean_text(r["text"], strip_punct=not args.keep_punct)
        n_fixed += new != r["text"]
        r["text"] = new
    print(f"ล้างข้อความ: แก้ {n_fixed} จาก {n_all} ประโยค")
    train_rows = [r for r in train_rows if r["text"] and len(tok(r["text"]).input_ids) <= MAX_LABEL]
    random.shuffle(train_rows)
    dev_rows = load_fleurs("dev", args.eval_n)
    print(f"ข้อมูลเทรน {len(train_rows)} คลิป (ตัดข้อความยาวเกิน {n_all-len(train_rows)}) | ตรวจระหว่างทาง {len(dev_rows)} คลิป")

    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.gradient_checkpointing_enable()
    model.model.encoder.conv1.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = find_last(out)
    if last and not args.resume:
        print(f"⚠️ มีงานที่เซฟค้างไว้ใน {last} — ถ้าจะเทรนต่อจากเดิมให้ใส่ --resume (รอบนี้จะเริ่มใหม่และเขียนทับ)")
    if args.resume and not last:
        print("ไม่พบจุดเซฟใน --out → เริ่มเทรนใหม่ตั้งแต่ต้น")
    resume_from = last if args.resume else None
    if args.full:
        # ข้อ 4: เทรนทั้งตัว (ทุกพารามิเตอร์) — ต้องใช้การ์ดจอ ≥12GB เช่น Kaggle T4/P100
        if resume_from:
            model = WhisperForConditionalGeneration.from_pretrained(resume_from)
            model.config.use_cache = False
            model.generation_config.forced_decoder_ids = None
            model.gradient_checkpointing_enable()
            print(f"เริ่มจากจุดเซฟ: {resume_from}")
        if args.freeze_encoder:  # ประหยัดหน่วยความจำ: เทรนเฉพาะตัวถอดข้อความ (decoder)
            for p in model.model.encoder.parameters():
                p.requires_grad_(False)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all_p = sum(p.numel() for p in model.parameters())
        print(f"เทรนทั้งตัว: {n_train:,} จาก {n_all_p:,} พารามิเตอร์ ({n_train/n_all_p:.0%})")
    elif resume_from or args.init_adapter:
        from peft import PeftModel

        src = resume_from or args.init_adapter
        model = PeftModel.from_pretrained(model, src, is_trainable=True)
        print(f"เริ่มจาก LoRA เดิม: {src}")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.rank * 2,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            ),
        )
    if not args.full:
        model.print_trainable_parameters()
    model.to(DEV).train()

    noise_files = list_noise(args.noise_dir) if args.aug > 0 else []
    if args.aug > 0:
        print(f"ผสมเสียงรบกวน: โอกาส {args.aug:.0%} ต่อคลิป | ไฟล์เพลง/เสียงรอบข้าง {len(noise_files)} ไฟล์")
    dl = DataLoader(
        LaoSpeech(train_rows, processor, args.aug, noise_files),
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        collate_fn=Collate(tok.convert_tokens_to_ids("<|startoftranscript|>")),
    )
    steps_per_epoch = math.ceil(len(dl) / args.accum)
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    if args.full:
        # ชิ้นคำลาวใหม่ (embedding) เพิ่งเริ่มจากค่าเฉลี่ย → ให้เรียนเร็วกว่าส่วนอื่น
        emb = model.get_input_embeddings().weight
        rest = [p for p in model.parameters() if p.requires_grad and p is not emb]
        groups = [{"params": rest, "lr": args.lr}, {"params": [emb], "lr": args.embed_lr or args.lr * 10}]
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, max(1, min(30, total // 8)), total)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)

    log = []
    step, best, t0 = 0, 9.9, time.time()
    if resume_from:
        st = torch.load(resume_from / "state.pt", map_location=DEV, weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step, best, log = st["step"], st["best"], st["log"]
        torch.manual_seed(step)  # สุ่มลำดับคลิปใหม่ ไม่วนซ้ำชุดเดิมทุกครั้งที่เทรนต่อ
        print(f"▶️ เทรนต่อจาก step {step}/{total} (CER ดีที่สุดก่อนหน้า {best*100:.1f}%)", flush=True)
    step0 = step
    run_loss = 0.0
    while step < total:
        for i, (feats, labels) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_FP16):
                loss = model(input_features=feats.to(DEV), labels=labels.to(DEV)).loss / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
            if (i + 1) % args.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % 10 == 0:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"step {step}/{total}  loss {run_loss/10:.3f}  "
                    f"ใช้ไป {el/60:.1f} นาที  เหลือ ~{el/(step-step0)*(total-step)/60:.0f} นาที  VRAM {mem:.1f}GB",
                    flush=True,
                )
                log.append({"step": step, "loss": run_loss / 10})
                run_loss = 0.0

            if step % steps_per_epoch == 0 or step == total:
                cer, hyps = quick_eval(model, processor, dev_rows)
                print(f"\n>>> ตรวจที่ step {step}: CER = {cer*100:.1f}%")
                print("    ถูก:", dev_rows[0]["text"][:70])
                print("    AI :", hyps[0][:70], "\n", flush=True)
                log.append({"step": step, "dev_cer": cer})
                model.save_pretrained(out / f"step{step}")
                if args.full:  # โมเดลเต็มมีตารางคำของตัวเอง → เก็บไว้คู่กัน
                    processor.save_pretrained(out / f"step{step}")
                if cer < best:
                    best = cer
                    model.save_pretrained(out / "best")
                    if args.full:
                        processor.save_pretrained(out / "best")
                    print(f"    💾 บันทึกตัวที่ดีที่สุด (CER {cer*100:.1f}%)", flush=True)

            if args.save_every and (step % args.save_every == 0 or step % steps_per_epoch == 0) and step < total:
                save_last(
                    model,
                    out,
                    {
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "best": best,
                        "log": log,
                    },
                )
            if step >= total:
                break

    (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nเสร็จ! ใช้เวลา {(time.time()-t0)/60:.0f} นาที | CER ดีที่สุด {best*100:.1f}% | {out/'best'}")


if __name__ == "__main__":
    main()
