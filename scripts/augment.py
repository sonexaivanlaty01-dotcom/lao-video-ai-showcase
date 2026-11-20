"""ผสมเสียงรบกวนตอนเทรน: ทำให้ AI ชินกับคลิปจริง (มีเพลง เสียงรอบข้าง พูดเร็ว/ช้า ดัง/เบา)

เสียงเดิม 1 คลิป → ทุกรอบที่เทรนจะได้ยินไม่เหมือนเดิม = เหมือนมีข้อมูลมากขึ้นโดยไม่ต้องอัดใหม่
  - ความเร็ว 0.9–1.1 เท่า
  - เสียงซ่า (ขาว/ชมพู/น้ำตาล) หรือ เพลง/เสียงรอบข้างจากโฟลเดอร์ data\\noise (ถ้ามีไฟล์)
  - ความดัง
  - ปิดบางช่วงของภาพเสียง (SpecAugment)

ลองฟังผล:  .venv\\Scripts\\python.exe scripts\\augment.py ไฟล์.wav   → ได้ไฟล์ _aug1.wav ... ไว้ฟัง
"""
import random
from pathlib import Path

import numpy as np

SR = 16000
NOISE_EXT = {".wav", ".flac", ".mp3", ".ogg"}
_noise_cache: dict = {}


def list_noise(folder) -> list:
    folder = Path(folder)
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in NOISE_EXT) if folder.is_dir() else []


def _load_noise(path) -> np.ndarray:
    if path not in _noise_cache:
        import librosa

        _noise_cache[path] = librosa.load(str(path), sr=SR, duration=180)[0]
    return _noise_cache[path]


def _hiss(n: int, rng) -> np.ndarray:
    """เสียงซ่า 3 แบบ: ขาว (ซ่าแหลม) / ชมพู / น้ำตาล (หึ่งทุ้ม คล้ายแอร์ รถ)"""
    white = rng.standard_normal(n).astype(np.float32)
    kind = int(rng.integers(3))
    if kind == 0:
        return white
    spec = np.fft.rfft(white)
    spec /= np.arange(1, len(spec) + 1) ** (0.5 if kind == 1 else 1.0)
    return np.fft.irfft(spec, n).astype(np.float32)


def _mix(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    pa, pn = float(np.mean(audio**2)) + 1e-10, float(np.mean(noise**2)) + 1e-10
    return audio + noise * np.sqrt(pa / (pn * 10 ** (snr_db / 10)))


def augment_audio(audio: np.ndarray, noise_files=(), p: float = 0.5) -> np.ndarray:
    """โอกาส p ที่จะแต่งเสียง (ที่เหลือปล่อยเสียงเดิม ให้ AI ยังจำเสียงสะอาดได้)"""
    if random.random() >= p or len(audio) < SR // 2:
        return audio
    rng = np.random.default_rng(random.getrandbits(32))
    a = audio.astype(np.float32)
    if rng.random() < 0.5:  # ความเร็ว
        n = int(len(a) / rng.uniform(0.9, 1.1))
        if n < 29.5 * SR:
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    if rng.random() < 0.7:  # เสียงรบกวน
        if len(noise_files) and rng.random() < 0.6:
            nz = _load_noise(noise_files[int(rng.integers(len(noise_files)))])
            if len(nz) < len(a):
                nz = np.tile(nz, len(a) // max(len(nz), 1) + 1)
            start = int(rng.integers(0, len(nz) - len(a) + 1))
            a = _mix(a, nz[start : start + len(a)], rng.uniform(5, 20))
        else:
            a = _mix(a, _hiss(len(a), rng), rng.uniform(10, 30))
    if rng.random() < 0.5:  # ความดัง
        a = a * (rng.uniform(0.2, 0.95) / (float(np.abs(a).max()) + 1e-6))
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def spec_augment(feats: np.ndarray, n_frames: int, p: float = 0.5) -> np.ndarray:
    """ปิดแถบความถี่ 2 แถบ + ช่วงเวลา 2 ช่วงสั้นๆ (เฉพาะส่วนที่มีเสียงจริง) ด้วยค่า "เงียบ" """
    if random.random() >= p:
        return feats
    rng = np.random.default_rng(random.getrandbits(32))
    f = feats.copy()
    quiet = f.min()
    n_mel, n_frames = f.shape[0], max(1, min(n_frames, f.shape[1]))
    for _ in range(2):
        w = int(rng.integers(0, n_mel // 7 + 1))
        s = int(rng.integers(0, n_mel - w + 1))
        f[s : s + w, :n_frames] = quiet
    for _ in range(2):
        w = int(rng.integers(0, min(40, n_frames // 8) + 1))  # ไม่เกิน 0.4 วิ และไม่เกิน 1/8 ของคลิป
        s = int(rng.integers(0, n_frames - w + 1))
        f[:, s : s + w] = quiet
    return f


if __name__ == "__main__":
    import sys

    import librosa
    import soundfile as sf

    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent
    wav = librosa.load(str(src), sr=SR)[0]
    noise = list_noise(Path(__file__).resolve().parent.parent / "data" / "noise")
    for i in range(1, 5):
        sf.write(out_dir / f"{src.stem}_aug{i}.wav", augment_audio(wav, noise, p=1.0), SR)
    print("saved 4 files to", out_dir)
