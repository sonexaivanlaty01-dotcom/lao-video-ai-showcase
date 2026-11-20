"""เวลาจริงของแต่ละคำในบรรทัดซับ (สำหรับซับคาราโอเกะ/เด้งทีละคำ)

ใช้ word_times() ของโมเดลตระกูล v3 (cross-attention + DTW) → เวลาเริ่มของแต่ละชิ้นคำ (token)
แล้วรวมชิ้นให้เป็น "คำ" ตามช่องว่างในข้อความ (คำที่แสดงบนจอ)

word_spans(eng, audio, text, audio_t0, seg_start, seg_end) → [{"start", "end"}, ...] หนึ่งรายการต่อคำ
หรือ None ถ้าเวลาที่ได้ดูไม่น่าเชื่อ (ให้ตัววาดแบ่งเวลาตามความยาวคำแทน)
"""
import re


def word_spans(eng, audio, text: str, audio_t0: float, seg_start: float, seg_end: float):
    clean = text.replace("*", "")  # *คำเน้น* ไม่ใช่ส่วนของเสียง
    words = list(re.finditer(r"\S+", clean))
    if not words:
        return None
    if len(words) == 1:
        return [{"start": round(seg_start, 3), "end": round(seg_end, 3)}]
    wt = eng.word_times(audio, clean)
    if not wt:
        return None
    # ตำแหน่งตัวอักษรที่แต่ละชิ้นจบ
    pieces = [p for p, _ in wt]
    if "".join(pieces) == clean:
        ends, acc = [], 0
        for p in pieces:
            acc += len(p)
            ends.append(acc)
    else:  # ชิ้นถอดไม่ครบตัวอักษร → ถอดแบบสะสมแทน
        tok = getattr(eng, "tok", None) or eng.processor.tokenizer
        ids = tok.encode(clean, add_special_tokens=False)
        if len(ids) != len(wt):
            return None
        ends = [len(tok.decode(ids[: k + 1])) for k in range(len(ids))]
    starts = []
    for m in words:
        # ชิ้นแรกที่กินตัวอักษรตัวแรกของคำ (ข้ามช่องว่างนำหน้า)
        k = next((i for i, e in enumerate(ends) if e > m.start()), len(ends) - 1)
        starts.append(audio_t0 + wt[k][1])
    # ให้เวลาเรียงต่อกันและอยู่ในบรรทัด
    out, prev = [], seg_start
    for i, s in enumerate(starts):
        s = min(max(s, prev, seg_start), seg_end - 0.05)
        out.append(s)
        prev = s
    out[0] = seg_start  # คำแรกขึ้นพร้อมบรรทัด
    span = out[-1] - out[0]
    if span < 0.08 * (len(out) - 1):  # เวลาเกาะกันหมด = จับเวลาไม่ได้
        return None
    res = []
    for i, s in enumerate(out):
        e = out[i + 1] if i + 1 < len(out) else seg_end
        res.append({"start": round(s, 3), "end": round(max(e, s + 0.05), 3)})
    return res
