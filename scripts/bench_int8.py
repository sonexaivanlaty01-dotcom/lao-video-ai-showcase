"""ย่อโมเดล CTC เป็น ONNX int8 (แบบที่ใช้บนมือถือ) แล้ววัด ขนาด / ความเร็ว CPU / ความแม่น (FLEURS 100) เทียบตัวเต็ม fp32
ไม่แก้โมเดลเดิม — ไฟล์ใหม่อยู่ที่ models/<ชื่อ>-onnx/ (model_fp32.onnx, model_int8.onnx)

  .venv\\Scripts\\python.exe scripts\\bench_int8.py models\\ctc-omni-v1
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from common import RESULTS
from compare_models import load_audio, tests
from eval_rivals import cer
from train_ctc import decode

sys.stdout.reconfigure(encoding="utf-8")


def main():
    os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    src = Path(sys.argv[1])
    out = src.parent / f"{src.name}-onnx"
    out.mkdir(exist_ok=True)
    # ย่อเฉพาะ MatMul (transformer) — ConvInteger ใน onnxruntime ช้ามาก ถ้าย่อ conv ด้วยจะช้ากว่าตัวเต็ม
    f32, i8 = out / "model_fp32.onnx", out / "model_int8_matmul.onnx"
    fe = AutoFeatureExtractor.from_pretrained(src)
    inv = {v: k for k, v in json.loads((src / "vocab.json").read_text(encoding="utf-8")).items()}
    if not f32.exists():
        model = AutoModelForCTC.from_pretrained(src).eval()
        torch.onnx.export(model, (torch.zeros(1, 16000 * 5),), str(f32), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {1: "t"}, "logits": {1: "f"}}, opset_version=17, dynamo=False)
        del model
    if not i8.exists():
        quantize_dynamic(str(f32), str(i8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])

    rows = tests()["FLEURS 100"]
    audio = [load_audio(r["file"]) for r in rows]
    secs = sum(len(a) for a in audio) / 16000
    refs = [r["text"] for r in rows]
    lines = [f"=== {src.name}: ONNX fp32 vs int8 บน CPU ({os.cpu_count()} threads), FLEURS 100 = เสียง {secs:.0f} วิ ==="]

    def mb(p):
        return sum(f.stat().st_size for f in p.parent.glob(p.name + "*")) / 1e6  # รวมไฟล์ .data ถ้ามี

    res = {}
    for name, p in (("fp32", f32), ("int8", i8)) if "--int8-only" not in sys.argv else (("int8", i8),):
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        run = lambda x: decode(sess.run(None, {"x": fe(x, sampling_rate=16000, return_tensors="np").input_values.astype(np.float32)})[0].argmax(-1)[0], inv)
        [run(a) for a in audio[:2]]
        t = time.time()
        hyp = [run(a) for a in audio]
        res[name] = time.time() - t
        lines.append(f"{name} : ขนาด {mb(p):6.0f} MB | {res[name]:5.0f} วิ = เร็ว {secs / res[name]:4.1f} เท่าของเวลาจริง | CER {cer(refs, hyp):.1f}%")
        print(lines[-1], flush=True)
    if "fp32" in res:
        lines.append(f"→ int8 เล็กลง {mb(f32) / mb(i8):.1f} เท่า, เร็วขึ้น {res['fp32'] / res['int8']:.1f} เท่า")
    msg = "\n".join(lines)
    print(msg)
    with open(RESULTS / f"bench_int8_{src.name}.txt", "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


if __name__ == "__main__":
    main()
