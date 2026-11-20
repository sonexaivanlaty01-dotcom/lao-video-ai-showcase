"""วัดผล BuzzASR กับไฟล์เสียงที่รู้คำตอบอยู่แล้ว (เช่นเสียง TTS ที่สร้างจาก lines.json)

  .venv\\Scripts\\python.exe scripts\\check_lines.py <โฟลเดอร์ NN.wav> <lines.json>
"""
import json
import sys
from pathlib import Path

import jiwer
from buzz_fast import Buzz, load_audio
from common import RESULTS, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
wav_dir, lines_path = Path(sys.argv[1]), Path(sys.argv[2])
lines = json.loads(lines_path.read_text(encoding="utf-8"))
wavs = [wav_dir / f"{i:02}.wav" for i in range(len(lines))]
refs = [ln["say"] for ln in lines]

buzz = Buzz()
hyps = buzz.transcribe([load_audio(w) for w in wavs])
rows = []
for i, (r, h) in enumerate(zip(refs, hyps)):
    c = jiwer.cer(norm_for_cer(r), norm_for_cer(h))
    rows.append({"no": i, "ref": r, "ai": h, "cer": round(c * 100, 1)})
    print(f"{i:02} [{c*100:5.1f}%]\n   ถูก: {r}\n   AI : {h}")
total = jiwer.cer([norm_for_cer(r) for r in refs], [norm_for_cer(h) for h in hyps])
print(f"\nรวม CER = {total*100:.1f}%")
RESULTS.mkdir(exist_ok=True)
(RESULTS / "check_lekha_lines.json").write_text(
    json.dumps({"cer": round(total * 100, 1), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
)
