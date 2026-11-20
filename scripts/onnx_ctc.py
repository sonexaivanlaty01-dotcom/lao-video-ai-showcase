"""รันโมเดล CTC ที่ย่อเป็น ONNX int8 (scripts/bench_int8.py) บน CPU — ตัวเดียวกับที่จะใส่ในแอปมือถือ

  from onnx_ctc import OnnxCTC
  asr = OnnxCTC(Path("models/ctc-omni-v1"), "model_int8_matmul.onnx")
  asr.transcribe([np_audio_16k, ...]) → ["ข้อความ", ...]
"""
import json
import os
from pathlib import Path

import numpy as np
from common import fix_spelling


class OnnxCTC:
    def __init__(self, model_dir: Path, onnx_name: str = "model_int8_matmul.onnx", threads: int | None = None,
                 use_lm: bool = True):
        import onnxruntime as ort
        from transformers import AutoFeatureExtractor

        model_dir = Path(model_dir)
        onnx = model_dir.parent / f"{model_dir.name}-onnx" / onnx_name
        if not onnx.exists():
            raise FileNotFoundError(f"ไม่มีไฟล์ {onnx} (รัน scripts/bench_int8.py ก่อน)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 4))
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.fe = AutoFeatureExtractor.from_pretrained(model_dir)
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        self.inv = {v: k for k, v in vocab.items()}
        # ภาษาลาวช่วยเดา (beam search + char 6-gram ขายได้, scripts/ctc_lm.py) — ไม่มีไฟล์ LM = ถอดแบบธรรมดา
        self.beam = None
        if use_lm:
            from ctc_lm import LM_SELL, BeamDecoder, load_sell

            if LM_SELL.exists():
                cfg_f = model_dir / "lm_decode.json"
                cfg = json.loads(cfg_f.read_text(encoding="utf-8")) if cfg_f.exists() else {"alpha": 0.5, "beta": 1.0, "beam": 8}
                self.beam = BeamDecoder(self.inv, load_sell(), cfg["alpha"], cfg["beta"], cfg.get("beam", 8))

    def _decode(self, ids) -> str:
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                out.append(self.inv.get(int(i), ""))
            prev = i
        return "".join(out).replace("|", " ").strip()

    def transcribe(self, clips) -> list[str]:
        res = []
        for x in clips:
            v = self.fe(x, sampling_rate=16000, return_tensors="np").input_values.astype(np.float32)
            lg = self.sess.run(None, {"x": v})[0][0]
            if self.beam:
                lp = lg - lg.max(-1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
                text = self.beam.decode(lp)
            else:
                text = self._decode(lg.argmax(-1))
            res.append(" ".join(fix_spelling(text).split()))
        return res
