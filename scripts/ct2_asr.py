"""AI ของเรา (ตระกูล V3) แบบเร็ว: รันด้วย CTranslate2 แทน PyTorch — โมเดลตัวเดิม แค่เครื่องยนต์เร็วกว่า

แปลง:  ctranslate2.converters.TransformersConverter("models/full-lao-ultra3", copy_files=[...]).convert(
          "models/full-lao-ultra3-ct2-int8", quantization="int8_float32")
ใช้แบบเดียวกับ V3:  V3Fast(model_dir).transcribe([...]) / transcribe_detailed([...]) / word_times(audio, text)
GTX 16xx: fp16 เพี้ยน → ค่าเริ่มต้น compute_type="int8_float32" (น้ำหนัก int8 คำนวณ fp32)
"""
import common  # noqa: F401  (ตั้งค่า CUDA DLL path)
import numpy as np
from common import fix_spelling
from v3_asr import SR, collapse_loops


class V3Fast:
    def __init__(self, model_dir, device: str | None = None, compute_type: str = "int8_float32", batch: int = 16):
        import ctranslate2
        from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

        self.device = device or ("cuda" if ctranslate2.get_cuda_device_count() else "cpu")
        self.model = ctranslate2.models.Whisper(str(model_dir), device=self.device, compute_type=compute_type)
        self.fe = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.tok = WhisperTokenizerFast.from_pretrained(model_dir)
        self.prompt = self.tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|lo|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eos = self.tok.eos_token_id
        self.batch = batch

    def _feats(self, chunk):
        import ctranslate2

        f = self.fe([a[: 30 * SR] for a in chunk], sampling_rate=SR, return_tensors="np").input_features.astype(np.float32)
        return ctranslate2.StorageView.from_array(np.ascontiguousarray(f))

    def _text(self, ids) -> str:
        return collapse_loops(fix_spelling(self.tok.decode(ids, skip_special_tokens=True)).strip())[0]

    def transcribe(self, audios, batch: int | None = None, max_new: int = 440) -> list[str]:
        out = []
        b = batch or self.batch
        for i in range(0, len(audios), b):
            res = self.model.generate(self._feats(audios[i : i + b]), [self.prompt] * len(audios[i : i + b]),
                                      beam_size=1, max_length=max_new)
            out += [self._text(r.sequences_ids[0]) for r in res]
        return out

    def transcribe_detailed(self, audios, batch: int | None = None, max_alts: int = 2, unsure: float = 0.6,
                            min_alt: float = 0.12, prompt=None) -> list[dict]:
        """เหมือน V3.transcribe_detailed: {text, conf, alts}"""
        import ctranslate2

        cpu = ctranslate2.Device.cpu
        results = []
        b = batch or self.batch
        for i in range(0, len(audios), b):
            chunk = audios[i : i + b]
            feats = self._feats(chunk)
            res = self.model.generate(feats, [self.prompt] * len(chunk), beam_size=1, max_length=440,
                                      return_logits_vocab=True)
            enc = None
            for j, r in enumerate(res):
                ids = r.sequences_ids[0]
                logits = r.logits[0]  # ต่อขั้น: log-prob ทั้งตาราง (ต่อท้ายด้วยขั้น eos)
                p1s, cands = [], []
                for k, t in enumerate(ids):
                    lp = np.asarray(logits[k].to_device(cpu)).astype(np.float32)
                    pr = np.exp(lp - lp.max())
                    pr /= pr.sum()
                    p1 = float(pr[t])
                    p1s.append(p1)
                    top2 = np.argpartition(-pr, 2)[:2]
                    alt = int(top2[0]) if int(top2[0]) != t else int(top2[1])
                    if p1 < unsure and pr[alt] >= min_alt and alt != self.eos:
                        cands.append((p1, k, alt, float(pr[alt])))
                text = self._text(ids)
                alts = []
                if cands and max_alts > 0:
                    if enc is None:
                        enc = np.asarray(self.model.encode(feats, to_cpu=True))
                    for _, k, alt, ap in sorted(cands)[:max_alts]:
                        pref = self.prompt + list(ids[:k]) + [alt]
                        if len(pref) >= 440:
                            continue
                        # ต่อจาก prefix (greedy) — ใช้ผลฟังเสียงที่คำนวณไว้แล้ว ไม่ต้องฟังใหม่
                        sub = self.model.generate(self._slice(enc, j), [pref], beam_size=1, max_length=440)
                        got = list(sub[0].sequences_ids[0])
                        head = list(ids[:k]) + [alt]
                        if got[: len(head)] != head:  # (CTranslate2 คืนรวม prefix มาด้วย — กันไว้เผื่อบางรุ่นไม่รวม)
                            got = head + got
                        at = self._text(got)
                        if at and at != text and all(a["text"] != at for a in alts):
                            alts.append({"text": at, "p": round(ap, 3)})
                conf = round(min(p1s), 3) if p1s else 1.0
                if collapse_loops(fix_spelling(self.tok.decode(ids, skip_special_tokens=True)).strip())[1]:
                    conf = min(conf, 0.05)
                results.append({"text": text, "conf": conf, "alts": alts})
        return results

    @staticmethod
    def _slice(enc: np.ndarray, j: int):
        import ctranslate2

        return ctranslate2.StorageView.from_array(np.ascontiguousarray(enc[j : j + 1]))

    def word_times(self, audio, text: str) -> list[tuple[str, float]]:
        """เวลาเริ่มของแต่ละชิ้นคำ (วินาทีจากต้นท่อน) ด้วย CTranslate2 align (cross-attention + DTW ในตัว)"""
        ids = self.tok.encode(text, add_special_tokens=False)
        if not ids:
            return []
        n_frames = int(np.ceil(min(len(audio), 30 * SR) / SR / 0.01))  # mel 10 ms/เฟรม
        res = self.model.align(self._feats([audio]), self.prompt[:3], [ids], [n_frames])[0]
        first = {}
        for ti, tf in res.alignments:
            first.setdefault(ti, tf)
        return [(self.tok.decode([t]), round(first.get(k, 0) * 0.02, 2)) for k, t in enumerate(ids)]
