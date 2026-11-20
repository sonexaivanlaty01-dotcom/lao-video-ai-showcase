"""แยกเสียงพูดออกจากเพลง (Demucs ของ Meta, สัญญาอนุญาต MIT) ก่อนส่งให้ AI ทำซับ

  from separate_vocals import Vocals
  v = Vocals(); clean = v.extract(audio_16k)   # numpy 16kHz → เสียงพูดอย่างเดียว 16kHz

ทดสอบ:  .venv\\Scripts\\python.exe scripts\\separate_vocals.py <ไฟล์วิดีโอ/เสียง> [ไฟล์ผล.wav]
"""
import sys

import common  # noqa: F401
import numpy as np
import torch

SR = 16000


class Vocals:
    def __init__(self, model_name: str = "htdemucs", device: str | None = None):
        from demucs.pretrained import get_model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(model_name).to(self.device).eval()
        self.sr = self.model.samplerate  # 44100
        self.vocal_idx = self.model.sources.index("vocals")

    @torch.no_grad()
    def extract(self, audio16k: np.ndarray) -> np.ndarray:
        import librosa
        from demucs.apply import apply_model

        wav = librosa.resample(audio16k.astype(np.float32), orig_sr=SR, target_sr=self.sr)
        x = torch.from_numpy(np.stack([wav, wav]))[None]  # (1, 2ch, T) — Demucs ต้องการสเตอริโอ
        ref = x.mean(1)
        mean, std = ref.mean(), ref.std() + 1e-8
        out = apply_model(self.model, ((x - mean) / std).to(self.device), split=True, overlap=0.25, progress=False)
        voc = (out[0, self.vocal_idx] * std + mean).mean(0).cpu().numpy()
        return librosa.resample(voc, orig_sr=self.sr, target_sr=SR).astype(np.float32)


if __name__ == "__main__":
    import time

    import soundfile as sf
    from buzz_fast import load_audio

    a = load_audio(sys.argv[1])
    t = time.time()
    v = Vocals().extract(a)
    print(f"แยกเสียง {len(a)/SR:.0f} วิ ใช้เวลา {time.time()-t:.1f} วิ")
    if len(sys.argv) > 2:
        sf.write(sys.argv[2], v, SR)
