"""ให้ OmniASR (CTC) ใช้ "ความรู้ภาษาลาว" ช่วยเลือกตัวอักษร: prefix beam search + char 6-gram (shallow fusion)
ไม่ต้องเทรนโมเดลใหม่ ไม่ใช้ Kaggle — ใช้ได้เชิงการค้า: คลังข้อความเฉพาะแหล่งที่ขายได้

  .venv\\Scripts\\python.exe scripts\\ctc_lm.py build              → models/lm/lao_char6_sell.pkl
  .venv\\Scripts\\python.exe scripts\\ctc_lm.py tune models\\ctc-omni-v2   → หา alpha/beta บน FLEURS dev, วัดผลชุดสอบทั้งหมด

แหล่งข้อความ (ขายได้): FLEURS train (CC-BY-4.0), KitTzk (CC-BY-4.0), Common Voice (CC0), เฉลย YouTube CC-BY ที่คัดแล้ว
❌ ไม่ใช้ SiangLao / RFA / WorldSpeech · ❌ ไม่ใช้ซับทีม (มีคลิปที่ใช้วัดผลอยู่ — กันรั่ว) · ตัดประโยคที่ตรงกับชุดสอบ
"""
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from char_lm import LAO, OLD, CharLM, norm, tsv_texts
from common import ROOT, load_fleurs, norm_for_cer

sys.stdout.reconfigure(encoding="utf-8")
LM_SELL = ROOT / "models" / "lm" / "lao_char6_sell.pkl"
LN10 = math.log(10)


def collect_sell() -> list[str]:
    from compare_models import tests

    banned = set()
    for s in ("dev", "test"):
        banned |= {norm_for_cer(r["text"]) for r in load_fleurs(s)}
    for rows in tests().values():
        banned |= {norm_for_cer(r["text"]) for r in rows}
    texts = [r["text"] for r in load_fleurs("train")]
    for s in ("kittzk", "cv", "ytcc"):
        texts += tsv_texts(ROOT / "data" / "extra" / s / "keep.tsv")
    for f in sorted((ROOT / "data" / "lm").glob("sell_*.txt")):  # ข้อความขายได้ที่เพิ่มทีหลัง (เช่น วิกิพีเดียลาว)
        texts += f.read_text(encoding="utf-8").splitlines()
    out, seen = [], set()
    for t in texts:
        t = norm(t)
        k = norm_for_cer(t)
        if not t or not LAO.search(t) or OLD.search(t) or k in banned or k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def build():
    texts = collect_sell()
    # ข้อความจากเสียงจริง (ภาษาพูด) ให้น้ำหนักมากกว่าวิกิพีเดีย (ภาษาเขียน): ซ้ำ IN_W เท่า ตอนนับ
    in_w = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--in-w=")), "1"))
    if in_w > 1:
        wiki = set(norm(t) for t in (ROOT / "data" / "lm" / "sell_wiki_lo.txt").read_text(encoding="utf-8").splitlines()) \
            if (ROOT / "data" / "lm" / "sell_wiki_lo.txt").exists() else set()
        texts = texts + [t for t in texts if t not in wiki] * (in_w - 1)
    (ROOT / "data" / "lm").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "lm" / "corpus_sell.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    import char_lm  # pickle อ้างเป็น char_lm.CharLM

    lm = char_lm.CharLM.train(texts)
    with open(LM_SELL, "wb") as fh:
        pickle.dump(lm, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ คลังขายได้ {len(texts):,} ประโยค {sum(map(len, texts)):,} ตัวอักษร → {LM_SELL} ({LM_SELL.stat().st_size / 1e6:.0f} MB)")


def load_sell():
    with open(LM_SELL, "rb") as fh:
        return pickle.load(fh)


def _lse(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class BeamDecoder:
    """prefix beam search สำหรับ CTC + char LM (shallow fusion)
    คะแนน = log P_ctc + alpha·log P_lm + beta·จำนวนตัวอักษร"""

    def __init__(self, inv: dict, lm: CharLM, alpha=0.5, beta=1.0, beam=8, prune=-7.0):
        self.inv, self.lm, self.alpha, self.beta, self.beam, self.prune = inv, lm, alpha, beta, beam, prune
        self.chars = {i: (" " if c == "|" else c) for i, c in inv.items()}
        self._cache = {}

    def _lmlp(self, prefix: str, ch: str) -> float:
        key = (prefix[-5:], ch)
        v = self._cache.get(key)
        if v is None:
            v = self.lm.logp(prefix, ch) * LN10
            self._cache[key] = v
        return v

    def decode(self, logp: np.ndarray) -> str:
        """logp: (T, V) log-softmax ของเฟรม — index 0 = blank"""
        # beams: prefix(str) → [p_blank, p_nonblank, lm_score, last_id]
        beams = {"": [0.0, -np.inf, 0.0, None]}
        for t in range(logp.shape[0]):
            row = logp[t]
            cand = np.where(row > self.prune)[0]
            nb = {}

            def get(p, lm_s, last):
                if p not in nb:
                    nb[p] = [-np.inf, -np.inf, lm_s, last]
                return nb[p]

            for prefix, (pb, pnb, lms, last) in beams.items():
                ptot = _lse(pb, pnb)
                for i in cand:
                    pr = float(row[i])
                    if i == 0:  # blank
                        e = get(prefix, lms, last)
                        e[0] = _lse(e[0], ptot + pr)
                        continue
                    ch = self.chars.get(int(i), "")
                    if not ch:
                        continue
                    if i == last:  # ตัวซ้ำ: อยู่ที่เดิม (ต่อจาก non-blank) หรือขึ้นตัวใหม่ (หลัง blank)
                        e = get(prefix, lms, last)
                        e[1] = _lse(e[1], pnb + pr)
                        if pb == -np.inf:
                            continue
                        src = pb
                    else:
                        src = ptot
                    if ch == " " and (not prefix or prefix.endswith(" ")):
                        continue
                    np_ = prefix + ch
                    e = get(np_, lms + self.alpha * self._lmlp(prefix, ch) + self.beta, int(i))
                    e[1] = _lse(e[1], src + pr)
            beams = dict(sorted(nb.items(), key=lambda kv: -(_lse(kv[1][0], kv[1][1]) + kv[1][2]))[: self.beam])
        best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][2]
                   + self.alpha * self.lm.logp(kv[0], "\x03") * LN10)
        return " ".join(best[0].split())


def tune(model_dir: Path):
    import os

    import torch
    from compare_models import load_audio, tests
    from eval_rivals import cer
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    from common import fix_spelling
    from train_ctc import decode as greedy

    os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(model_dir).to("cuda").eval()
    inv = {v: k for k, v in json.loads((model_dir / "vocab.json").read_text(encoding="utf-8")).items()}
    lm = load_sell()

    sets = {"FLEURS dev 100 (จูน)": load_fleurs("dev", 100)} | tests()
    logits = {}
    for k, rows in sets.items():
        out = []
        for r in rows:
            x = load_audio(r["file"])
            with torch.no_grad():
                lg = model(fe(x, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")).logits[0]
            out.append(torch.log_softmax(lg.float(), -1).cpu().numpy())
        logits[k] = out
        print(f"logits {k}: {len(out)}", flush=True)
    del model
    torch.cuda.empty_cache()

    def run(k, dec):
        hyp = [fix_spelling(dec(lp)) for lp in logits[k]]
        return cer([r["text"] for r in sets[k]], hyp)

    gd = lambda lp: greedy(lp.argmax(-1), inv)
    lines = [f"=== {model_dir.name}: greedy vs beam+LM (char 6-gram ขายได้) ==="]
    base = {k: run(k, gd) for k in sets}
    tk = "FLEURS dev 100 (จูน)"
    best = (base[tk], None)
    for a in (0.3, 0.5, 0.7, 1.0):
        for b in (0.0, 0.5, 1.0, 2.0):
            d = BeamDecoder(inv, lm, a, b)
            c = run(tk, d.decode)
            print(f"alpha {a} beta {b}: dev {c:.2f} (greedy {base[tk]:.2f})", flush=True)
            if c < best[0]:
                best = (c, (a, b))
    lines.append(f"จูนบน FLEURS dev: ดีสุด alpha/beta = {best[1]} → {best[0]:.1f} (greedy {base[tk]:.1f})")
    if best[1]:
        d = BeamDecoder(inv, lm, *best[1])
        t0, secs = time.time(), 0
        for k in sets:
            c = run(k, d.decode)
            lines.append(f"{k[:40].ljust(42)} greedy {base[k]:5.1f} → LM {c:5.1f}")
            print(lines[-1], flush=True)
        n = sum(lp.shape[0] for v in logits.values() for lp in v)
        lines.append(f"เวลาถอดแบบ beam (CPU Python): {time.time() - t0:.0f} วิ สำหรับ {n / 50 / 60:.0f} นาทีเสียง")
        (model_dir / "lm_decode.json").write_text(json.dumps({"alpha": best[1][0], "beta": best[1][1], "beam": 8}), encoding="utf-8")
    msg = "\n".join(lines)
    print(msg)
    from common import RESULTS

    (RESULTS / f"ctc_lm_{model_dir.name}.txt").write_text(msg + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"build": lambda: build(), "tune": lambda: tune(Path(sys.argv[2]))}[sys.argv[1]]()
