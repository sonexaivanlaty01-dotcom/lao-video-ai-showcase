"""วัด Qwen3-ASR ลาว (เช่น LuoYiSULIXAY/qwen3-asr-lao-H) บนชุดสอบเดียวกับ eval_rivals.py
qwen-asr ต้องใช้ transformers 4.x → แยก venv (.venv-qwen) แล้วแบ่งเป็น 3 ขั้น:

  1) .venv\\Scripts\\python.exe scripts\\eval_qwen3asr.py export          → results/qwen_eval/sets.json
  2) .venv-qwen\\Scripts\\python.exe scripts\\eval_qwen3asr.py run REPO     → results/qwen_eval/<name>.json
  3) .venv\\Scripts\\python.exe scripts\\eval_qwen3asr.py score REPO        → results/eval_<name>.txt

การ์ดจอ GTX 1650 มี 4 GB → โหลดแบบ 8-bit (bitsandbytes)
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "qwen_eval"
sys.stdout.reconfigure(encoding="utf-8")


def export():
    from compare_models import tests

    sets = tests()  # FLEURS 100 + คลิปทีม (ข้าม SiangLao test ยาว 3 ชม. เพราะโมเดลนี้ช้า)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: [{"file": str(r["file"]), "text": r["text"]} for r in rows] for k, rows in sets.items()}
    (OUT / "sets.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in data.items()})


def run(repo):
    import os

    os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
    import librosa
    import torch
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
    from transformers import BitsAndBytesConfig

    if "Lao" not in SUPPORTED_LANGUAGES:
        SUPPORTED_LANGUAGES.append("Lao")
    asr = Qwen3ASRModel.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda:0", max_inference_batch_size=1, max_new_tokens=256,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    data = json.loads((OUT / "sets.json").read_text(encoding="utf-8"))
    out, total_audio, t0 = {}, 0.0, time.time()
    for k, rows in data.items():
        hyps = []
        for r in rows:
            x = librosa.load(r["file"], sr=16000)[0]
            total_audio += len(x) / 16000
            hyps.append(asr.transcribe(audio=(x, 16000), language="Lao")[0].text)
        out[k] = hyps
        print(f"{k}: {len(hyps)} คลิป | ตัวอย่าง: {hyps[0][:50]}", flush=True)
    secs = time.time() - t0
    name = repo.split("/")[-1]
    (OUT / f"{name}.json").write_text(json.dumps({"hyps": out, "seconds": secs, "audio": total_audio}, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"GPU 8-bit: {secs:.0f}s / เสียง {total_audio:.0f}s = {total_audio / secs:.1f} เท่า")


def run_hf(repo):
    """รุ่นแบบ -hf (เช่น Phonepadith/qwen3-asr-laos) ใช้ transformers 5 ใน .venv หลักได้เลย ไม่ต้อง qwen-asr"""
    import os

    os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
    import librosa
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    proc = AutoProcessor.from_pretrained(repo)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(repo, dtype=torch.float16).to("cuda").eval()
    data = json.loads((OUT / "sets.json").read_text(encoding="utf-8"))
    out, total_audio, t0 = {}, 0.0, time.time()
    for k, rows in data.items():
        hyps = []
        for r in rows:
            x = librosa.load(r["file"], sr=16000)[0]
            total_audio += len(x) / 16000
            try:
                inp = proc.apply_transcription_request(audio=x, language="Lao")
            except Exception:
                inp = proc.apply_transcription_request(audio=x)
            inp = inp.to("cuda", dtype=torch.float16) if hasattr(inp, "to") else inp
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=256, do_sample=False)
            txt = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            hyps.append(proc.extract_transcription(txt))
        out[k] = hyps
        print(f"{k}: {len(hyps)} คลิป | ตัวอย่าง: {hyps[0][:50]}", flush=True)
    secs = time.time() - t0
    name = repo.split("/")[-1]
    (OUT / f"{name}.json").write_text(json.dumps({"hyps": out, "seconds": secs, "audio": total_audio}, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"GPU fp16: {secs:.0f}s / เสียง {total_audio:.0f}s = {total_audio / secs:.1f} เท่า")


def score(repo):
    from eval_rivals import cer

    name = repo.split("/")[-1]
    data = json.loads((OUT / "sets.json").read_text(encoding="utf-8"))
    res = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
    lines = [f"=== {name} (CER %, ต่ำ = ดี) ==="]
    for k, rows in data.items():
        lines.append(f"{k[:40].ljust(42)}{cer([r['text'] for r in rows], res['hyps'][k]):8.1f}")
    lines.append(f"GPU (GTX 1650): {res['seconds']:.0f}s / เสียง {res['audio']:.0f}s = {res['audio'] / res['seconds']:.1f} เท่า")
    msg = "\n".join(lines)
    print(msg)
    (HERE.parent / "results" / f"eval_{name}.txt").write_text(msg + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"export": lambda: export(), "run": lambda: run(sys.argv[2]), "run_hf": lambda: run_hf(sys.argv[2]),
     "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
