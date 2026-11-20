"""ทดลอง Meta MMS (CTC, รองรับภาษาลาวสำเร็จรูป) เทียบกับ Ultra3 บนชุดสอบเดียวกับ compare_models
⚠️ MMS = CC-BY-NC (ทดลองเท่านั้น) — ใช้ดูว่า "แนวทาง CTC" เร็ว/แม่นแค่ไหนก่อนลงแรงเทรนของเราเอง

  .venv\\Scripts\\python.exe scripts\\eval_mms.py D:\\lao-video-ai-archive\\hf\\mms-1b-all
"""
import sys
import time

import jiwer
import torch
from common import RESULTS, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")


def main():
    from compare_models import load_audio, tests
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    path = sys.argv[1]
    proc = AutoProcessor.from_pretrained(path, target_lang="lao")
    model = Wav2Vec2ForCTC.from_pretrained(path, target_lang="lao", ignore_mismatched_sizes=True)
    sets = tests()
    audio = {k: [load_audio(r["file"]) for r in rows] for k, rows in sets.items()}
    total = sum(len(a) for v in audio.values() for a in v) / 16000

    def run(dev, dtype, clips, bs=4):
        m = model.to(dev, dtype=dtype).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(clips), bs):
                b = proc(clips[i : i + bs], sampling_rate=16000, return_tensors="pt", padding=True)
                lg = m(b.input_values.to(dev, dtype=dtype), attention_mask=b.get("attention_mask", None).to(dev) if b.get("attention_mask") is not None else None).logits
                out += proc.batch_decode(lg.float().argmax(-1).cpu())
        return out

    first = next(iter(audio.values()))[:4]
    cpu_ref = run("cpu", torch.float32, first)
    mode = ("cpu", torch.float32)
    if torch.cuda.is_available():
        try:
            g = run("cuda", torch.float16, first)
            same = sum(a == b for a, b in zip(g, cpu_ref))
            print(f"GPU fp16 ตรงกับ CPU {same}/4", flush=True)
            if same >= 3:
                mode = ("cuda", torch.float16)
        except Exception as e:
            print("GPU ใช้ไม่ได้:", str(e)[:120], flush=True)
            model.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
    print("ใช้", mode, flush=True)
    t = time.time()
    lines = []
    for k, rows in sets.items():
        hyps = run(*mode, audio[k])
        c = 100 * jiwer.cer([norm_for_cer(r["text"]) for r in rows], [norm_for_cer(h) or "-" for h in hyps])
        lines.append(f"{k[:20]} {c:5.1f}%")
        print(f"  {k}: CER {c:.1f}% | ตัวอย่าง: {hyps[0][:60]}", flush=True)
    dt = time.time() - t
    msg = f"MMS-1b-all ({mode[0]} {str(mode[1]).split('.')[-1]}) | {dt:.0f}s ({total / dt:.1f}x realtime) | " + " | ".join(lines)
    print(msg)
    with open(RESULTS / "bench_fast.txt", "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


if __name__ == "__main__":
    main()
