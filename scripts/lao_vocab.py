"""ข้อ 3: เพิ่ม "ตารางคำภาษาลาว" ให้ Whisper small (แบบเดียวกับที่ BuzzASR ทำให้ตัวใหญ่)

ปัญหา: Whisper เดิมไม่รู้จักตัวอักษรลาว หั่นประโยคลาวเป็นชิ้นเล็กมาก (~345 ชิ้น/ประโยค)
วิธี:  เรียนรู้ "ชิ้นคำลาวที่เจอบ่อย" จากข้อความเทรน (BPE) แล้วเพิ่มเข้าไปในตารางคำของ Whisper
       ชิ้นใหม่แต่ละชิ้นเริ่มจากค่าเฉลี่ยของชิ้นเล็กเดิมที่ประกอบกันเป็นมัน (AI จึงไม่เริ่มจากศูนย์)
ผล:    models\\whisper-small-lao-vocab\\  (โมเดล + ตารางคำใหม่ พร้อมเทรนทั้งตัว ข้อ 4)

  .venv\\Scripts\\python.exe scripts\\lao_vocab.py --size 6000
ใช้เฉพาะข้อความ "train" เท่านั้น (ไม่แตะข้อสอบ dev/test)
"""
import argparse
import re
import sys
from pathlib import Path

import common  # noqa: F401
from common import MODELS, ROOT, clean_text, load_fleurs

sys.stdout.reconfigure(encoding="utf-8")
BASE = "openai/whisper-small"
LAO = range(0x0E80, 0x0F00)
MARK = re.compile("[ັິ-ຼ່-ໍ]")  # สระ/วรรณยุกต์ที่ต้องเกาะพยัญชนะ


def train_texts(extra_text_dir=None):
    texts = [clean_text(r["text"]) for r in load_fleurs("train")]
    siang = ROOT / "data" / "siang_lao" / "train.tsv"
    if siang.exists():
        for ln in siang.read_text(encoding="utf-8").splitlines()[1:]:
            p = ln.split("\t")
            if len(p) > 1:
                texts.append(clean_text(p[1]))
    if extra_text_dir:  # ข้อความลาวเพิ่ม (.txt) เช่นข่าว/บทความที่มีสิทธิ์ใช้ → ตารางคำครอบคลุมขึ้น
        for f in Path(extra_text_dir).glob("*.txt"):
            texts += [clean_text(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [t for t in texts if t]


def learn_lao_pieces(texts, size, max_len=8):
    """BPE ระดับตัวอักษรบนข้อความลาว → คืนชิ้นที่ยาว 2..max_len ตัวอักษรและเป็นอักษรลาวล้วน

    จำกัดความยาว: SiangLao ให้หลายคนอ่านประโยคเดียวกัน BPE จึงเรียน "ทั้งประโยค" เป็นชิ้นเดียว
    → AI จะท่องจำประโยคแทนที่จะฟัง  ชิ้นยาวไม่เกิน ~8 ตัวอักษร ≈ คำ/1–3 พยางค์"""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=size, min_frequency=3, special_tokens=["[UNK]"], show_progress=False)
    tok.train_from_iterator(texts, trainer)
    pieces = [
        p
        for p in tok.get_vocab()
        if 2 <= len(p) <= max_len and all(ord(c) in LAO for c in p) and not MARK.match(p)
    ]
    # ตัดชิ้นที่มักไป "ตัดกลางพยางค์": จบด้วยพยัญชนะที่ในข้อความจริงมีสระ/วรรณยุกต์ตามมา
    # เช่น ບັນຍ (จาก ບັນຍາ) → เจอ ບັນຍຸຕິທຳ จะเหลือ ຸ หลุดเป็นเศษไบต์
    corpus = "\n".join(texts)
    keep = []
    for p in pieces:
        total = corpus.count(p)
        broken = len(re.findall(re.escape(p) + MARK.pattern, corpus))
        if total and broken / total <= 0.05:
            keep.append(p)
    # ตัวอักษรลาวทุกตัว (เดี่ยวๆ) — Whisper เดิมไม่มีเลย จึงแตกเป็นเศษไบต์ 2–3 ชิ้นต่อตัวอักษร
    singles = sorted({c for c in corpus if ord(c) in LAO} | {chr(c) for c in range(0x0E81, 0x0EDF) if chr(c).isprintable()})
    return sorted(set(keep) | set(singles), key=lambda p: (-len(p), p))


def broken_rate(tok, texts):
    """สัดส่วนชิ้นที่เป็น "เศษไบต์" (ตัวอักษรลาวถูกหั่นจนอ่านไม่ออก) ยิ่งน้อยยิ่งดี"""
    n = bad = 0
    for t in texts:
        toks = tok.convert_ids_to_tokens(tok(t, add_special_tokens=False).input_ids)
        n += len(toks)
        bad += sum(1 for x in toks if not any(ord(c) in LAO for c in x) and x != "Ġ" and not x.isascii())
    return bad / max(n, 1)


def count_pieces(tok, texts):
    return sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts) / len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=6000, help="ขนาดตาราง BPE ภาษาลาวที่จะเรียนรู้")
    ap.add_argument("--text-dir", default=None, help="โฟลเดอร์ .txt ข้อความลาวเพิ่ม (ไม่บังคับ)")
    ap.add_argument("--max-len", type=int, default=8, help="ความยาวสูงสุดของชิ้นคำ (ตัวอักษร)")
    ap.add_argument("--out", default=str(MODELS / "whisper-small-lao-vocab"))
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--init-adapter", default=None, help="รวม LoRA ที่เทรนแล้วก่อนขยายตารางคำ เช่น models\\lora-small-lo-v2\\best")
    a = ap.parse_args()

    from transformers import WhisperTokenizerFast

    texts = train_texts(a.text_dir)
    print(f"ข้อความเทรน {len(texts)} ประโยค", flush=True)
    pieces = learn_lao_pieces(texts, a.size, a.max_len)
    print(f"ได้ชิ้นคำลาวใหม่ {len(pieces)} ชิ้น เช่น {pieces[:3]} ... {pieces[-3:]}")

    old = WhisperTokenizerFast.from_pretrained(BASE, language="lao", task="transcribe")
    new = WhisperTokenizerFast.from_pretrained(BASE, language="lao", task="transcribe")
    n_added = new.add_tokens(pieces)
    test = [clean_text(r["text"]) for r in load_fleurs("test", 200)]  # วัดกับข้อสอบที่ไม่ได้ใช้สร้างตาราง
    try:
        buzz = WhisperTokenizerFast.from_pretrained("BuzzASR/lao")
        b = f" | BuzzASR {count_pieces(buzz, test):.0f}"
    except Exception:
        b = ""
    print(f"ชิ้นต่อประโยค (ข้อสอบ 200 ประโยค): Whisper เดิม {count_pieces(old, test):.0f} | ตารางใหม่ {count_pieces(new, test):.0f}{b}")
    print(f"เศษไบต์ (หั่นกลางตัวอักษร): ตารางใหม่ {broken_rate(new, test)*100:.1f}% ของชิ้นทั้งหมด")
    sample = test[1][:40]
    ids = new(sample, add_special_tokens=False).input_ids
    print("ตัวอย่างการหั่น:", " | ".join(new.convert_ids_to_tokens(ids)))
    rt = new.decode(ids)
    assert rt == sample, f"ถอดกลับไม่ตรง: {rt!r} != {sample!r}"
    if a.measure_only:
        return

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained(BASE)
    if a.init_adapter:  # เริ่มจาก AI ที่เทรนแล้ว (หูฟังเสียงลาวเป็นแล้ว) ไม่ใช่ Whisper เปล่า
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.init_adapter).merge_and_unload()
        print(f"รวม LoRA เข้าตัวหลักแล้ว: {a.init_adapter}")
    emb = model.get_input_embeddings().weight.data
    n_old = emb.shape[0]
    model.resize_token_embeddings(len(new), mean_resizing=False)
    emb = model.get_input_embeddings().weight.data
    with torch.no_grad():
        for p in pieces:
            nid = new.convert_tokens_to_ids(p)
            if nid < n_old:
                continue
            sub = old(p, add_special_tokens=False).input_ids
            emb[nid] = emb[sub].mean(0)
    out = Path(a.out)
    model.save_pretrained(out)
    processor = WhisperProcessor.from_pretrained(BASE, language="lao", task="transcribe")
    processor.tokenizer = new
    processor.save_pretrained(out)
    print(f"✅ เพิ่ม {n_added} ชิ้น ตารางคำ {n_old} → {len(new)} | บันทึกที่ {out}")


if __name__ == "__main__":
    main()
