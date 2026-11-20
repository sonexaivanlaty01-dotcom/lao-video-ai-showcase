"""ถอดเสียงทั้งคลิป: ตัดท่อนด้วย VAD → ถอดด้วย V3 (v3/v4/v5/Ultra…) → แบ่งบรรทัดซับ

ตัวเลือกที่ปรับได้ (ไว้ทดลองว่าอะไรช่วยให้แม่นขึ้น):
  max_s      ความยาวท่อนเสียงสูงสุด (วินาที) — ยาวขึ้น = AI เห็นบริบททั้งประโยค
  prev_ctx   ส่งข้อความท่อนก่อนหน้าเป็นคำใบ้ (ต้องถอดทีละท่อน ช้าลง)
  prompt     คำใบ้คงที่ทั้งคลิป เช่น หัวข้อ/คำสำคัญ
  line_s     ความยาวบรรทัดซับสูงสุดตอนแสดงผล (ท่อนยาวจะถูกแบ่งเป็นหลายบรรทัด)
"""
import numpy as np
from v3_asr import SR

DEFAULT = {"max_s": 8.0, "prev_ctx": False, "prompt": None, "line_s": 7.0}


def segment(audio: np.ndarray, max_s: float = 8.0, merge_gap: float = 0.0) -> list[dict]:
    """VAD ตัดทุกช่วงเงียบ ≥0.3 วิ → ท่อนสั้นมาก (ประโยคขาดกลาง)
    merge_gap > 0: รวมท่อนติดกันที่เว้นช่วงไม่เกิน merge_gap วินาที จนยาวไม่เกิน max_s → AI เห็นทั้งประโยค"""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    spans = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=min(max_s, 8.0) if merge_gap else max_s,
                                                    min_silence_duration_ms=300, speech_pad_ms=150))
    if not merge_gap or not spans:
        return spans
    out = [dict(spans[0])]
    for s in spans[1:]:
        cur = out[-1]
        if (s["start"] - cur["end"]) / SR <= merge_gap and (s["end"] - cur["start"]) / SR <= max_s:
            cur["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def split_line(text: str, start: float, end: float, line_s: float) -> list[tuple[float, float, str]]:
    """ท่อนยาวเกิน line_s → แบ่งที่ช่องว่าง เวลาแบ่งตามจำนวนตัวอักษร (ประมาณ; ขั้นจัดเวลาระดับคำจะแม่นกว่า)"""
    dur = end - start
    if dur <= line_s or " " not in text:
        return [(start, end, text)]
    n = int(np.ceil(dur / line_s))
    words = text.split(" ")
    total = sum(len(w) for w in words) or 1
    target = total / n
    parts, cur, acc = [], [], 0
    for w in words:
        cur.append(w)
        acc += len(w)
        if acc >= target * (len(parts) + 1) and len(parts) < n - 1:
            parts.append(" ".join(cur))
            cur = []
    if cur:
        parts.append(" ".join(cur))
    out, t = [], start
    for p in parts:
        d = dur * len(p.replace(" ", "")) / total
        out.append((round(t, 3), round(t + d, 3), p))
        t += d
    return out


def split_by_times(text: str, start: float, end: float, wt: list[tuple[str, float]], line_s: float = 6.0) -> list[tuple[float, float, str]]:
    """แบ่งบรรทัดยาวที่ "ช่องว่างระหว่างคำ" โดยใช้เวลาจริงของคำ (word_times) แทนการเดาจากจำนวนตัวอักษร"""
    dur = end - start
    if dur <= line_s or " " not in text or not wt:
        return [(start, end, text)]
    n = int(np.ceil(dur / line_s))
    pos, t_at = 0, []  # ตำแหน่งตัวอักษร → เวลา ของแต่ละช่องว่าง
    for piece, t in wt:
        for k, ch in enumerate(piece):
            if ch == " " and 0 < pos + k < len(text) - 1:
                t_at.append((pos + k, t))
        pos += len(piece)
    if not t_at:
        return split_line(text, start, end, line_s)
    cuts = []
    for j in range(1, n):
        target = dur * j / n
        c = min(t_at, key=lambda x: abs(x[1] - target))
        if c not in cuts and (not cuts or c[0] > cuts[-1][0]):
            cuts.append(c)
    out, a, ta = [], 0, 0.0
    for p, t in cuts:
        out.append((round(start + ta, 3), round(start + t, 3), text[a:p].strip()))
        a, ta = p + 1, t
    out.append((round(start + ta, 3), round(end, 3), text[a:].strip()))
    return [o for o in out if o[2]]


def transcribe_audio(asr, audio: np.ndarray, max_s=8.0, prev_ctx=False, prompt=None, max_alts=0, merge_gap=0.0) -> tuple[list[dict], list[dict]]:
    """คืน (spans, det) — det[i] = {text, conf, alts} ของท่อน spans[i]"""
    spans = segment(audio, max_s, merge_gap)
    clips = [audio[s["start"] : s["end"]] for s in spans]
    if not prev_ctx:
        return spans, asr.transcribe_detailed(clips, max_alts=max_alts, prompt=prompt)
    det, prev = [], ""
    for c in clips:
        hint = " ".join(x for x in (prompt, prev[-120:]) if x).strip() or None
        d = asr.transcribe_detailed([c], max_alts=max_alts, prompt=hint)[0]
        det.append(d)
        if d["text"]:
            prev = d["text"]
    return spans, det
