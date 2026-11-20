"""คลังข้อความภาษาลาว + แบบจำลองภาษาระดับตัวอักษร (char n-gram) ใช้ 2 อย่าง:
  1) ไฮไลต์จุดที่ "ไม่เหมือนภาษาลาวที่เคยเห็น" (คำไร้ความหมาย/สะกดแปลก) ในหน้าแก้ซับ
  2) ช่วย AI เลือกคำตอนลังเล (shallow fusion) — ไม่แตะตัวโมเดลฟังเสียงเลย ปิด/ถอดออกได้

สร้าง:  .venv\\Scripts\\python.exe scripts\\char_lm.py build
ผล:     models/lm/lao_char6.pkl  (+ data/lm/corpus.txt ข้อความที่ใช้)

แหล่งข้อความ (ไม่รวมชุดสอบ FLEURS dev/test, SiangLao test, คลิปทีมที่ใช้วัดผล, ข่าว S&P 500):
  SiangLao train, FLEURS train, KitTzk, Common Voice, RFA ที่คัดแล้ว, WorldSpeech la_lo ทั้งหมด (ตัดตัวสะกดเก่า),
  ซับที่ทีมกด "ตรวจเสร็จ" (data/subs ที่ status=corrected) และ data/lm/extra_*.txt (ประโยคที่ทีมเพิ่มเอง)
"""
import csv
import glob
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict

from common import ROOT, clean_text, load_fleurs, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
N = 6
LM_PATH = ROOT / "models" / "lm" / "lao_char6.pkl"
OLD = re.compile(r"ຣ|(?<!ເຕ)ັມ|ຜລ|ຜແ|ຕພ|ກິຈ|ວັທ|ປະເທສ|ເສຖ|໌")
LAO = re.compile(r"[຀-໿]")
BOS, EOS = "\x02", "\x03"


def norm(t: str) -> str:
    t = clean_text(t, strip_punct=False)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tsv_texts(path, col="text"):
    try:
        with open(path, encoding="utf-8") as fh:
            return [r[col] for r in csv.DictReader(fh, delimiter="\t") if r.get(col)]
    except FileNotFoundError:
        return []


def collect() -> list[str]:
    banned = set()
    for s in ("dev", "test"):
        banned |= {norm_for_cer(r["text"]) for r in load_fleurs(s)}
    for p in list((ROOT / "data" / "my_train").glob("*/metadata.tsv")) + [ROOT / "data" / "eval" / "sp500_ref.tsv", ROOT / "data" / "siang_lao" / "test.tsv"]:
        banned |= {norm_for_cer(t) for t in tsv_texts(p)}
    texts = []
    texts += tsv_texts(ROOT / "data" / "siang_lao" / "train.tsv")
    texts += [r["text"] for r in load_fleurs("train")]
    for s in ("kittzk", "cv", "rfa", "rfa2"):
        texts += tsv_texts(ROOT / "data" / "extra" / s / "keep.tsv")
    try:  # WorldSpeech la_lo (เฉลยข่าว RFA ทั้งหมด หลายหมื่นท่อน)
        import polars as pl
        for f in glob.glob(str(ROOT.home() / ".cache/huggingface/hub/datasets--disco-eth--WorldSpeech/snapshots/*/data/la_lo/*.parquet")):
            texts += [t for t in pl.read_parquet(f, columns=["human_transcript"])["human_transcript"].to_list() if t]
    except Exception as e:
        print("ข้าม WorldSpeech:", e)
    for f in glob.glob(str(ROOT / "data" / "subs" / "*.lo.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("status") == "corrected":
            texts += [s["text"] for s in d["segments"]]
    for f in glob.glob(str(ROOT / "data" / "lm" / "extra_*.txt")):
        texts += open(f, encoding="utf-8").read().splitlines()
    out, seen = [], set()
    for t in texts:
        t = norm(t)
        k = norm_for_cer(t)
        if not t or not LAO.search(t) or OLD.search(t) or k in banned or k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


class CharLM:
    """char n-gram + stupid backoff (เร็ว ไม่ต้องติดตั้งอะไรเพิ่ม)"""

    def __init__(self, counts, total, n=N):
        self.c, self.total, self.n = counts, total, n

    @classmethod
    def train(cls, texts, n=N):
        counts = defaultdict(Counter)  # context(str) → Counter(next char)
        total = 0
        for t in texts:
            s = BOS * (n - 1) + t + EOS
            for i in range(n - 1, len(s)):
                for k in range(0, n):
                    counts[s[i - k : i]][s[i]] += 1
            total += len(t) + 1
        counts = {ctx: (sum(c.values()), dict(c)) for ctx, c in counts.items()}
        return cls(counts, total, n)

    def logp(self, ctx: str, ch: str) -> float:
        """log10 P(ch | ctx) แบบ stupid backoff (0.4 ต่อขั้น)"""
        pen = 0.0
        ctx = (BOS * (self.n - 1) + ctx)[-(self.n - 1) :]
        for k in range(len(ctx), -1, -1):
            c = self.c.get(ctx[len(ctx) - k :])
            if c and ch in c[1]:
                return pen + math.log10(c[1][ch] / c[0])
            pen += math.log10(0.4)
        return pen + math.log10(1 / (self.total + 1000))

    def score(self, text: str, prefix: str = "", eos: bool = False) -> float:
        lp, ctx = 0.0, prefix
        for ch in text:
            lp += self.logp(ctx, ch)
            ctx += ch
        if eos:
            lp += self.logp(ctx, EOS)
        return lp

    def surprise(self, text: str) -> list[float]:
        """ค่าความแปลก (−log10 P) ของแต่ละตัวอักษร — สูง = ไม่เหมือนภาษาลาวที่เคยเห็น"""
        out, ctx = [], ""
        for ch in text:
            out.append(-self.logp(ctx, ch))
            ctx += ch
        return out


ODD_TH = 2.6  # ทดสอบกับข่าว S&P 500: จับบรรทัดที่ AI ผิด 9/21 เตือนผิดในบรรทัดถูก 2/35


def odd_fragments(text: str, lm=None, th: float = ODD_TH) -> list[str]:
    """ส่วนของข้อความที่ "ไม่เหมือนภาษาลาวที่เคยเห็น" (เฉลี่ยความแปลก 3 ตัวอักษร ≥ th) → คืนคำ/ท่อนนั้น ๆ ไว้ไฮไลต์"""
    lm = lm or load_lm()
    if lm is None or not text:
        return []
    sp = lm.surprise(text)
    hits = []
    for i, ch in enumerate(text):
        if not LAO.match(ch):
            continue
        w = [sp[j] for j in range(max(0, i - 1), min(len(text), i + 2)) if LAO.match(text[j])]
        if w and sum(w) / len(w) >= th:
            hits.append(i)
    ranges = []
    for i in hits:  # ขยายเป็นคำ (คั่นด้วยช่องว่าง) ถ้าสั้น ไม่งั้นเอา ±3 ตัวอักษร
        a, b = text.rfind(" ", 0, i) + 1, text.find(" ", i)
        b = len(text) if b < 0 else b
        if b - a > 8:
            a, b = max(a, i - 3), min(b, i + 4)
        if ranges and a <= ranges[-1][1]:  # ทับ/ติดกัน → รวมเป็นช่วงเดียว
            ranges[-1][1] = max(ranges[-1][1], b)
        else:
            ranges.append([a, b])
    frags = []
    for a, b in ranges:
        f = text[a:b].strip()
        if f and f not in frags:
            frags.append(f)
    return frags


_lm = None


def load_lm():
    global _lm
    if _lm is None and LM_PATH.exists():
        with open(LM_PATH, "rb") as fh:
            _lm = pickle.load(fh)
    return _lm


def build():
    texts = collect()
    (ROOT / "data" / "lm").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "lm" / "corpus.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    lm = CharLM.train(texts)
    LM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LM_PATH, "wb") as fh:
        pickle.dump(lm, fh, protocol=pickle.HIGHEST_PROTOCOL)
    chars = sum(len(t) for t in texts)
    print(f"✅ คลังข้อความ {len(texts):,} ประโยค {chars:,} ตัวอักษร → {LM_PATH} ({LM_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        import char_lm  # ให้ pickle อ้างคลาสเป็น char_lm.CharLM (ไม่ใช่ __main__) เซิร์ฟเวอร์จะโหลดได้

        char_lm.build()
