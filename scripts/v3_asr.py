"""AI ของเรา v3 (Whisper small + ตารางคำลาว + เทรนทั้งตัวบน Kaggle)

FLEURS 30 คลิป: CER 18.5% (BuzzASR 21.0%) และเร็วกว่า BuzzASR ~4 เท่าบน GTX 1650
ใช้แบบเดียวกับ buzz_fast.Buzz:  V3().transcribe([audio_np, ...]) → [ข้อความ, ...]
(แต่ละท่อนยาวไม่เกิน 30 วินาที, เสียง 16kHz)

⚠️ เทรนด้วย SiangLao (CC-BY-NC) → ห้ามใช้เชิงการค้า
"""
import re

import common  # noqa: F401  (ตั้งค่า CUDA DLL path)
import numpy as np
import torch
from common import ROOT, fix_spelling

MODEL_DIR = ROOT / "models" / "full-lao-v3"
SR = 16000


def collapse_loops(text: str, max_rep: int = 2) -> tuple[str, bool]:
    """AI เขียนวน เช่น "ຫຼື 100,000 ກີບ ຫຼື 100,000 ກີບ ..." ซ้ำหลายสิบรอบ → เหลือไม่เกิน max_rep รอบ
    คืน (ข้อความใหม่, เจอการวนไหม)"""
    w = text.split()
    found = False
    for n in range(1, 26):  # วลียาว 1–25 คำ (เจอวนทั้งประโยค ~13 คำ เช่น "ຕິດພັນກັບກົດໝາຍ ... ມາດຕາ 25 ຫາ 26")
        i, out = 0, []
        while i < len(w):
            chunk = w[i : i + n]
            reps = 1
            while len(chunk) == n and w[i + reps * n : i + (reps + 1) * n] == chunk:
                reps += 1
            long_loop = n >= 4 and reps >= 2  # วลียาว 4+ คำซ้ำติดกัน = AI วน (คนไม่พูดประโยคยาวซ้ำทันที)
            if (reps >= 4 or long_loop) and len(chunk) == n:  # คำสั้นซ้ำ 2–3 ครั้งคนพูดจริงได้ (ບໍ່ ບໍ່ ບໍ່)
                out += chunk * (1 if long_loop else max_rep)
                i += reps * n
                found = True
            else:
                out.append(w[i])
                i += 1
        w = out
    text2 = " ".join(w)
    # ระดับตัวอักษร: ท่อนยาว ≥20 ตัวอักษรที่ซ้ำติดกัน (จับกรณีที่จุดตัดคำไม่ตรงกันในแต่ละรอบ)
    t3 = re.sub(r"(.{20,}?)(?:\s*\1)+", r"\1", text2)
    if t3 != text2:
        found, text2 = True, t3
    return text2, found


class LMFusion:
    """ให้คลังข้อความภาษาลาว (char n-gram) ช่วยเลือกคำ "ตอนที่ AI ลังเล" เท่านั้น (shallow fusion)
    คะแนนใหม่ = log P(คำ | เสียง) + weight × (log P_ภาษา(ตัวอักษรของคำ | ข้อความก่อนหน้า) + bonus × จำนวนตัวอักษร)
    ไม่แตะตัวโมเดลฟังเสียง ปิดได้ด้วย weight=0"""

    def __init__(self, tok, lm, weight: float, bonus: float = 2.0, topk: int = 5, unsure: float = 0.8):
        self.tok, self.lm, self.w, self.bonus, self.topk, self.unsure = tok, lm, weight, bonus, topk, unsure
        self.eos = tok.eos_token_id
        self.start = None
        self.cache = {}

    def _piece(self, t: int) -> str | None:
        if t not in self.cache:
            s = self.tok.decode([t], skip_special_tokens=False)
            self.cache[t] = None if ("�" in s or s.startswith("<|")) else s
        return self.cache[t]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        import math

        if self.start is None:
            self.start = input_ids.shape[1]
        logp = torch.log_softmax(scores.float(), -1)
        top = torch.topk(logp, self.topk, dim=-1)
        out = scores.clone()
        ln10 = math.log(10)
        for b in range(scores.shape[0]):
            if math.exp(float(top.values[b, 0])) >= self.unsure:
                continue  # AI มั่นใจพอแล้ว ไม่ต้องยุ่ง
            ctx = self.tok.decode(input_ids[b, self.start :].tolist(), skip_special_tokens=True)
            new = []
            for v, t in zip(top.values[b].tolist(), top.indices[b].tolist()):
                if t == self.eos:
                    lm = self.lm.logp(ctx, "\x03") * ln10
                    n = 1
                else:
                    p = self._piece(t)
                    if p is None:
                        new.append((t, v))
                        continue
                    lm = self.lm.score(p, prefix=ctx) * ln10
                    n = len(p)
                new.append((t, v + self.w * (lm + self.bonus * n)))
            out[b, :] = float("-inf")
            for t, s in new:
                out[b, t] = s
        return out


class V3:
    def __init__(self, device: str | None = None, model_dir=None):
        """model_dir: โฟลเดอร์โมเดลอื่นที่สูตรเดียวกัน เช่น models/full-lao-v4 (ไม่ใส่ = v3)"""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_dir = model_dir or MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        # GTX 16xx คำนวณ fp16 แล้วเพี้ยน → ใช้ fp32 (โมเดลเล็ก ใช้การ์ดจอแค่ ~1.5GB)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _greedy_from(self, enc: torch.Tensor, prefix: list[int], max_new: int = 120) -> list[int]:
        """เขียนต่อจาก prefix ทีละชิ้น (เลือกชิ้นที่โอกาสสูงสุด) จนจบประโยค"""
        ids = list(prefix)
        eos = self.processor.tokenizer.eos_token_id
        # ห้ามยาวเกินตำแหน่งที่ Whisper รองรับ (448) ไม่งั้น CUDA พัง (index out of bounds) แล้วเซิร์ฟเวอร์ใช้การ์ดจอต่อไม่ได้
        limit = self.model.config.max_target_positions
        max_new = min(max_new, limit - len(ids))
        # ใช้ cache (past_key_values): ส่งทั้ง prefix ครั้งแรก แล้วส่งทีละ 1 ชิ้น ไม่ต้องคำนวณประโยคใหม่ทั้งหมดทุกครั้ง
        past, x = None, torch.tensor([ids], device=self.device)
        for _ in range(max(0, max_new)):
            out = self.model(encoder_outputs=(enc,), decoder_input_ids=x, past_key_values=past, use_cache=True)
            past = out.past_key_values
            t = int(out.logits[0, -1].argmax())
            if t == eos:
                break
            ids.append(t)
            x = torch.tensor([[t]], device=self.device)
        return ids[len(prefix) :]

    @torch.no_grad()
    def transcribe_detailed(self, audios: list[np.ndarray], batch: int = 8, max_alts: int = 2,
                            unsure: float = 0.6, min_alt: float = 0.12, prompt: str | None = None) -> list[dict]:
        """ทำซับ + บอกความมั่นใจ + คำทางเลือก

        คืน [{text, conf, alts: [{text, p}]}]
          conf = โอกาสต่ำสุดของชิ้นที่เลือกในบรรทัดนั้น (ต่ำ = AI ลังเล)
          alts = ตรงจุดที่ลังเล (อันดับ 1 < 60% และอันดับ 2 ≥ 12%) ให้ลองเลือกอันดับ 2 แล้วเขียนต่อจนจบประโยค
        (beam search ใช้ไม่ได้: ได้คำตอบซ้ำกัน ช้า 11 เท่า และกินการ์ดจอ 6.4GB)"""
        tok = self.processor.tokenizer
        tok.set_prefix_tokens(language="lao", task="transcribe", predict_timestamps=False)
        # prompt = คำใบ้/ประโยคก่อนหน้า (Whisper <|startofprev|>) ช่วยให้เลือกคำตามบริบท
        p_ids = None
        if prompt:  # get_prompt_ids ใช้ไม่ได้ (คำลาวที่เพิ่มในตารางคำถูกมองเป็น special token) → ประกอบเอง
            ids = [tok.convert_tokens_to_ids("<|startofprev|>")] + tok.encode(" " + prompt.strip(), add_special_tokens=False)
            p_ids = torch.tensor(ids[-200:] if len(ids) > 200 else ids, device=self.device)
        gen_kw = {"prompt_ids": p_ids} if p_ids is not None else {}
        fuse_w = getattr(self, "lm_weight", 0.0)
        if fuse_w > 0:
            from char_lm import load_lm

            lm = load_lm()
            if lm is not None:
                gen_kw["output_logits"] = True  # ความมั่นใจคิดจากที่ AI ได้ยินจริง (ก่อนคลังข้อความช่วย)
        prompt = (p_ids.tolist() if p_ids is not None else []) + tok.prefix_tokens  # ต้นประโยคสำหรับหาคำทางเลือก
        eos = tok.eos_token_id
        results = []
        max_new = 440 - (len(p_ids) if p_ids is not None else 0)
        for i in range(0, len(audios), batch):
            chunk = [a[: 30 * SR] for a in audios[i : i + batch]]
            feats = self.processor.feature_extractor(chunk, sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
            kw = dict(gen_kw)
            if "output_logits" in kw:
                from transformers import LogitsProcessorList

                kw["logits_processor"] = LogitsProcessorList([LMFusion(tok, lm, fuse_w, getattr(self, "lm_bonus", 2.0),
                                                                       unsure=getattr(self, "lm_unsure", 0.8))])
            out = self.model.generate(feats, language="lo", task="transcribe", max_new_tokens=max_new,
                                      output_scores=True, return_dict_in_generate=True, **kw)
            raw = out.logits if "output_logits" in kw else out.scores
            steps = len(raw)
            probs = [torch.softmax(s.float(), -1) for s in raw]  # แต่ละขั้น: (batch, คำทั้งหมด)
            enc_all = None
            for b in range(feats.shape[0]):
                seq = out.sequences[b, -steps:].tolist()
                picked, p1s, cands = [], [], []
                for k, t in enumerate(seq):
                    if t == eos or t == tok.pad_token_id:
                        break
                    pk = probs[k][b]
                    top = torch.topk(pk, 2)
                    picked.append(t)
                    p1 = float(pk[t])
                    p1s.append(p1)
                    alt_t, alt_p = int(top.indices[1]), float(top.values[1])
                    if int(top.indices[0]) != t:  # (กันกรณีแปลก)
                        alt_t, alt_p = int(top.indices[0]), float(top.values[0])
                    if p1 < unsure and alt_p >= min_alt and alt_t != eos:
                        cands.append((p1, k, alt_t, alt_p))
                text = fix_spelling(tok.decode(picked, skip_special_tokens=True)).strip()
                alts = []
                if cands and max_alts > 0:
                    if enc_all is None:
                        enc_all = self.model.model.encoder(feats).last_hidden_state
                    enc = enc_all[b : b + 1]
                    for _, k, alt_t, alt_p in sorted(cands)[:max_alts]:
                        if len(prompt) + k + 1 >= self.model.config.max_target_positions:
                            continue  # บรรทัดยาวจนเต็มแล้ว (มักเป็น AI เขียนวน) ไม่ต้องหาคำทางเลือก
                        rest = self._greedy_from(enc, prompt + picked[:k] + [alt_t])
                        alt_text = fix_spelling(tok.decode(picked[:k] + [alt_t] + rest, skip_special_tokens=True)).strip()
                        if alt_text and alt_text != text and all(a["text"] != alt_text for a in alts):
                            alts.append({"text": alt_text, "p": round(alt_p, 3)})
                conf = round(min(p1s), 3) if p1s else 1.0
                text, looped = collapse_loops(text)
                if looped:
                    conf = min(conf, 0.05)  # ให้หน้าเว็บขึ้นเตือน ⚠️ ว่าบรรทัดนี้ต้องฟังใหม่
                for a in alts:
                    a["text"] = collapse_loops(a["text"])[0]
                results.append({"text": text, "conf": conf, "alts": alts})
        return results

    @torch.no_grad()
    def word_times(self, audio: np.ndarray, text: str) -> list[tuple[str, float]]:
        """เวลาเริ่มของแต่ละชิ้นคำในท่อนเสียง (วินาทีจากต้นท่อน) — คำนวณครั้งเดียวจาก cross-attention + DTW
        (แบบเดียวกับ Whisper word timestamps แต่ไม่ต้องรันทีละขั้น จึงเร็วกว่ามาก)"""
        tok = self.processor.tokenizer
        tok.set_prefix_tokens(language="lao", task="transcribe", predict_timestamps=False)
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            return []
        prefix = tok.prefix_tokens
        feats = self.processor.feature_extractor([audio[: 30 * SR]], sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
        dec = torch.tensor([prefix + ids], device=self.device)
        prev_impl = self.model.config._attn_implementation
        if prev_impl != "eager":  # sdpa ไม่คืนค่า attention → สลับเป็น eager ชั่วคราว
            self.model.set_attn_implementation("eager")
        try:
            out = self.model(input_features=feats, decoder_input_ids=dec, output_attentions=True)
        finally:
            if prev_impl != "eager":
                self.model.set_attn_implementation(prev_impl)
        heads = self.model.generation_config.alignment_heads
        n_frames = max(1, int(np.ceil(len(audio[: 30 * SR]) / SR / 0.02)))  # 1 เฟรม encoder = 20 ms
        w = torch.stack([out.cross_attentions[l][0, h] for l, h in heads])  # (heads, tokens, 1500)
        w = w[:, len(prefix) :, :n_frames].float()
        w = (w - w.mean(-2, keepdim=True)) / (w.std(-2, keepdim=True) + 1e-6)
        m = w.mean(0).cpu().numpy()  # (tokens, frames)
        cost = -m
        T, F = cost.shape
        acc = np.full((T + 1, F + 1), np.inf)
        acc[0, 0] = 0
        for i in range(1, T + 1):
            for j in range(1, F + 1):
                acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
        i, j, path = T, F, []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
            i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        first = {}
        for ti, fj in reversed(path):
            first.setdefault(ti, fj)
        return [(tok.decode([t]), round(first.get(k, 0) * 0.02, 2)) for k, t in enumerate(ids)]

    @torch.no_grad()
    def transcribe(self, audios: list[np.ndarray], batch: int = 8, max_new: int = 440) -> list[str]:
        """max_new: จำกัดจำนวนชิ้นที่เขียนได้ (คลิปเพลง/เสียงแปลกๆ อาจทำให้เขียนวนไม่จบ → ช้ามาก)"""
        out = []
        for i in range(0, len(audios), batch):
            chunk = [a[: 30 * SR] for a in audios[i : i + batch]]
            feats = self.processor.feature_extractor(chunk, sampling_rate=SR, return_tensors="pt").input_features
            ids = self.model.generate(feats.to(self.device), language="lo", task="transcribe", max_new_tokens=max_new)
            out += [collapse_loops(fix_spelling(t).strip())[0] for t in self.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]
        return out
