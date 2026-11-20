"""แปลง BuzzASR/lao (HF) → CTranslate2 int8 ให้รันเร็วบนการ์ดจอ 4GB

รัน:  .venv\\Scripts\\python.exe scripts\\convert_buzzasr.py
      .venv\\Scripts\\python.exe scripts\\convert_buzzasr.py --fix-only   (แก้ vocabulary อย่างเดียว)
ผล:   models\\buzzasr-ct2-int8\\  (~1.6 GB)
"""
import json
import sys

import common  # noqa: F401
from common import ROOT

SRC = "BuzzASR/lao"
OUT = ROOT / "models" / "buzzasr-ct2-int8"
BASE = 51865  # token ข้อความธรรมดา 0..51864 ถูกต้องอยู่แล้ว


def fix_special_tokens(out):
    """tokenizer ของ BuzzASR ขาด <|endoftext|> กับ <|startoftranscript|> ทำให้ token พิเศษเลื่อนไป 2 ช่อง
    แต่ตัวโมเดลใช้ผังมาตรฐานของ Whisper large-v3 (generation_config: eos=51865, sot=51866, lo=51944)
    → สร้างส่วน token พิเศษของ vocabulary.json ใหม่ตามผังมาตรฐาน"""
    vpath = out / "vocabulary.json"
    vocab = json.loads(vpath.read_text(encoding="utf-8"))
    if vocab[BASE] == "<|endoftext|>":  # แก้รอบแรกไปแล้ว → เอาชื่อภาษาจากตำแหน่งใหม่
        langs = vocab[BASE + 2 : BASE + 102]
    else:
        langs = vocab[BASE : BASE + 100]  # ภาษา 100 ตัว (เดิมเริ่มที่ 51865 ผิดไป 2 ช่อง)
    assert langs[0] == "<|en|>" and langs[-1] == "<|yue|>", langs[-1]
    specials = (
        ["<|endoftext|>", "<|startoftranscript|>"]
        + langs
        + ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]
        + [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
    )
    # tokenizer ของ BuzzASR มี token พิเศษซ้ำอยู่ที่ id ต่ำๆ ด้วย (เช่น <|startoftranscript|> = 1)
    # ต้องเปลี่ยนชื่อตัวซ้ำ ไม่งั้น CTranslate2 จะหยิบ id ผิดตัว
    special_set = set(specials)
    low = [f"<|buzz_unused_{i}|>" if t in special_set else t for i, t in enumerate(vocab[:BASE])]
    new = low + specials
    assert len(new) == len(vocab) and new[51944] == "<|lo|>" and new[51968] == "<|transcribe|>"
    vpath.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")

    cpath = out / "config.json"
    cfg = json.loads(cpath.read_text(encoding="utf-8"))
    cfg["lang_ids"] = list(range(BASE + 2, BASE + 102))
    cpath.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("✅ แก้ token พิเศษแล้ว")


if __name__ == "__main__":
    if "--fix-only" not in sys.argv:
        from ctranslate2.converters import TransformersConverter
        from transformers import WhisperTokenizerFast

        print("กำลังแปลงโมเดล (ใช้ RAM ~7GB, ประมาณ 3-10 นาที)...", flush=True)
        TransformersConverter(SRC, copy_files=["preprocessor_config.json"]).convert(
            str(OUT), quantization="int8", force=True
        )
        WhisperTokenizerFast.from_pretrained(SRC).backend_tokenizer.save(str(OUT / "tokenizer.json"))
    fix_special_tokens(OUT)
    print("เสร็จ →", OUT)
