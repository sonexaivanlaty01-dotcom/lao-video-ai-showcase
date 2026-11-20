"""ของที่ใช้ร่วมกันระหว่างสคริปต์เทรนและสคริปต์วัดผล"""
import os
import re
import unicodedata
from pathlib import Path

import torch

# ให้ไลบรารีที่ใช้ CUDA หาไฟล์ของการ์ดจอที่มากับ PyTorch เจอ
_TORCH_LIB = str(Path(torch.__file__).parent / "lib")
if os.name == "nt":  # เฉพาะ Windows (Kaggle/Linux ไม่ต้อง)
    os.add_dll_directory(_TORCH_LIB)
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ["PATH"]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

def _faster_whisper_lite():
    """ใช้ส่วนตัดเสียงพูด (VAD) / FeatureExtractor ของ faster-whisper โดยไม่โหลด ctranslate2
    (Windows App Control บล็อกไฟล์ DLL ของ ctranslate2 ได้ → ถ้า import faster_whisper ตรงๆ จะพังทั้งระบบ)
    วิธี: ลงทะเบียนแพ็กเกจเปล่าไว้ก่อน แล้ว import เฉพาะไฟล์ย่อยที่ไม่ต้องใช้ ctranslate2"""
    import importlib.util
    import sys
    import types

    if "faster_whisper" in sys.modules:
        return
    try:
        import ctranslate2  # noqa: F401  ใช้ได้ปกติ → ใช้ faster_whisper ตัวเต็ม
        return
    except ImportError:
        pass
    spec = importlib.util.find_spec("faster_whisper")
    if spec is None:
        return
    pkg = types.ModuleType("faster_whisper")
    pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules["faster_whisper"] = pkg


_faster_whisper_lite()

ROOT = Path(__file__).resolve().parent.parent
FLEURS = ROOT / "data" / "fleurs_lo"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def ffmpeg_exe() -> str:
    """หา ffmpeg: ใน PATH ก่อน ไม่เจอค่อยหาในที่ที่ winget ติดตั้งไว้ (บางโปรแกรมไม่ได้รับ PATH ใหม่)"""
    import shutil

    if exe := shutil.which("ffmpeg"):
        return exe
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
    for cand in [winget / "Links" / "ffmpeg.exe", *sorted(winget.glob("Packages/*FFmpeg*/**/bin/ffmpeg.exe"))]:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError("ไม่พบ ffmpeg — ติดตั้งด้วย: winget install Gyan.FFmpeg")


def load_fleurs(split: str, limit: int | None = None):
    """อ่าน FLEURS ลาว คืน list ของ {"file": Path, "text": ข้อความลาว}

    คอลัมน์ใน tsv: id, ชื่อไฟล์, ข้อความ (มีวรรคตอน), ข้อความ (ตัวเล็ก), ...
    """
    folder = FLEURS / split
    have = {p.name: p for p in folder.glob("*.wav")}
    rows = []
    with open(FLEURS / f"{split}.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[1] not in have:
                continue
            rows.append({"file": have[parts[1]], "text": parts[2].strip()})
            if limit and len(rows) >= limit:
                break
    return rows


_TONE = "\u0ec8-\u0ecb"  # ່ ້ ໊ ໋
_ABOVE = "\u0eb1\u0eb4-\u0eb7\u0ebb\u0ecd"  # ັ ິ ີ ຶ ື ົ ໍ
_RE_INVISIBLE = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")
_RE_TONE_FIRST = re.compile(f"([{_TONE}]+)([{_ABOVE}]+)")
_RE_DUP_ABOVE = re.compile(f"([{_ABOVE}])[{_ABOVE}]+")
_RE_DUP_TONE = re.compile(f"([{_TONE}])[{_TONE}]+")
_RE_AM = re.compile(f"\u0ecd([{_TONE}]?)\u0eb2")
# ตัวเลขติดตัวอักษรลาว → เว้นวรรคตามหลักการเขียน: "20ປີ" → "20 ປີ", "ປີ2024" → "ປີ 2024" (ทั้งเลขอารบิกและเลขลาว ໐-໙)
_DIGIT = "0-9໐-໙"
_LAO_CH = "ກ-໏ໜ-ໟ"
_RE_NUM_LAO = re.compile(f"(?<=[{_DIGIT}])(?=[{_LAO_CH}])|(?<=[{_LAO_CH}])(?=[{_DIGIT}])")
_RE_PUNCT = re.compile("[\u201c\u201d\"\u2018\u2019()\\[\\]!?;:\u2026]|(?<![0-9])[.,]|[.,](?![0-9])")


def fix_spelling(s: str) -> str:
    """ทำให้คำที่ "หน้าตาเหมือนกันแต่พิมพ์ต่างกัน" เป็นแบบเดียว (ไม่เปลี่ยนความหมาย/เสียงอ่าน)

    ที่เจอจริงในข้อมูล (FLEURS + SiangLao):
      ສໍາ → ສຳ, ນໍ້າ → ນ້ຳ      (ໍ+າ 2 ตัว → ຳ ตัวเดียว)
      ເເ → ແ                    (สระ ເ 2 ตัว → ແ)
      ຫນ → ໜ, ຫມ → ໝ, ຫລ → ຫຼ
      วรรณยุกต์พิมพ์ก่อนสระบน → สลับให้สระมาก่อน, สระ/วรรณยุกต์ซ้อน 2 ตัว → เหลือตัวเดียว
      ตัวอักษรที่มองไม่เห็น (zero-width) → ลบ
    """
    s = unicodedata.normalize("NFC", s)
    s = _RE_INVISIBLE.sub("", s)
    s = s.replace("\u0ec0\u0ec0", "\u0ec1")
    s = _RE_TONE_FIRST.sub(r"\2\1", s)
    s = _RE_DUP_ABOVE.sub(r"\1", s)
    s = _RE_DUP_TONE.sub(r"\1", s)
    s = _RE_AM.sub("\\1\u0eb3", s)
    from lao_ortho import autofix  # สระล่างซ้อน, วรรณยุกต์ตามหลัง ຳ/ຽ → เรียงให้ถูกหลัก

    s = autofix(s)
    from old2new import modernize  # ตัวสะกดเก่า (ข่าว RFA) → แบบปัจจุบัน: ຜົລ→ຜົນ, ສເນີ→ສະເໜີ, ກະສິກັມ→ກະສິກຳ …

    s = modernize(s)
    s = s.replace("\u0eab\u0e99", "\u0edc").replace("\u0eab\u0ea1", "\u0edd").replace("\u0eab\u0ea5", "\u0eab\u0ebc")
    s = _RE_NUM_LAO.sub(" ", s)
    return s.replace("\u0e4d\u0e32", "\u0e33")  # ไทย: ํา → ำ


fix_am = fix_spelling  # ชื่อเดิม (ยังมีโค้ดเรียกอยู่)


def clean_text(s: str, strip_punct: bool = True) -> str:
    """ข้อความสำหรับใช้เทรน: สะกดแบบเดียว + (ค่าเริ่มต้น) ตัดเครื่องหมายวรรคตอน เพราะซับไม่ใช้
    และข้อมูลบางชุดมี บางชุดไม่มี (. , ระหว่างตัวเลขเช่น 2.5 ยังเก็บไว้)"""
    s = fix_spelling(s)
    if strip_punct:
        s = _RE_PUNCT.sub(" ", s)
    return " ".join(s.split())


def norm_for_cer(s: str) -> str:
    """เก็บเฉพาะตัวอักษร สระ วรรณยุกต์ ตัวเลข (ตัดเว้นวรรค/วรรคตอน) เพื่อคิด CER"""
    s = fix_spelling(s.lower())
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LMN") or " "
