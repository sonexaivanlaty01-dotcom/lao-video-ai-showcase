"""ผสมสมองโมเดล (model soup): เฉลี่ยค่าน้ำหนักของโมเดลที่สูตร/ตารางคำเดียวกัน → ได้ข้อดีของทั้งคู่

  .venv\\Scripts\\python.exe scripts\\merge_models.py models\\full-lao-v3 models\\full-lao-v4 --out models\\full-lao-ultra [--w 0.5]

--w = น้ำหนักของโมเดลตัวที่สอง (0.5 = ครึ่งต่อครึ่ง, 0.7 = เอียงไปทางตัวที่สอง)
ใช้ได้เฉพาะโมเดลที่ไฟล์ tokenizer.json / config.json เหมือนกันทุกตัว (เช่น v3, v4 ที่เทรนสูตรเดียวกัน)
ไม่แก้ไขโมเดลต้นทาง → ย้อนกลับได้เสมอ
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=0.5, help="น้ำหนักของโมเดลตัวที่สอง")
    args = ap.parse_args()
    a, b, out = Path(args.a), Path(args.b), Path(args.out)
    if out.exists():
        sys.exit(f"มีโฟลเดอร์ {out} อยู่แล้ว ไม่เขียนทับ (ตั้งชื่ออื่น หรือลบเองก่อน)")
    for f in ("tokenizer.json", "config.json"):
        if not filecmp.cmp(a / f, b / f, shallow=False):
            sys.exit(f"❌ {f} ไม่เหมือนกัน → ผสมไม่ได้ (คนละสูตร/คนละตารางคำ)")

    sa, sb = load_file(a / "model.safetensors"), load_file(b / "model.safetensors")
    assert sa.keys() == sb.keys(), "โครงสร้างโมเดลไม่ตรงกัน"
    merged = {}
    for k in sa:
        if sa[k].is_floating_point():
            merged[k] = ((1 - args.w) * sa[k].float() + args.w * sb[k].float()).to(sa[k].dtype).contiguous()
        else:
            merged[k] = sa[k]
    out.mkdir(parents=True)
    for f in a.iterdir():
        if f.name != "model.safetensors":
            shutil.copy(f, out / f.name)
    save_file(merged, out / "model.safetensors", metadata={"format": "pt"})
    (out / "MERGE.txt").write_text(f"ผสมจาก {a.name} x{1 - args.w:.2f} + {b.name} x{args.w:.2f}\n", encoding="utf-8")
    print(f"✅ ผสม {len(merged)} ชั้น: {a.name} x{1 - args.w:.2f} + {b.name} x{args.w:.2f} → {out}")


if __name__ == "__main__":
    main()
