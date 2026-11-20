"""วาดกราฟอัตราผิด (CER) ของ AI ถอดเสียงลาวแต่ละค่าย
ตัวเลขมาจาก results/eval_rivals.txt, results/bench_fast.txt และที่ค่ายอื่นประกาศ (ดูหมายเหตุในรูป)

  .venv\\Scripts\\python.exe scripts\\plot_ranking.py          → results/ranking_cer.png (ไทย)
  .venv\\Scripts\\python.exe scripts\\plot_ranking.py --lo     → results/ranking_cer_lo.png (ລາວ)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS

plt.rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "sans-serif"]  # มีทั้งอักษรไทยและลาว

T = {
    "th": dict(
        title="อัตราผิดของ AI ถอดเสียงภาษาลาว (CER ยิ่งต่ำยิ่งดี)", left="ชุดสอบมาตรฐาน FLEURS",
        right="คลิปจริงของทีม (วัดเองทั้งหมด)", ours="ทีมเรา", nuol="ม.แห่งชาติลาว", self="วัดเอง",
        vendor="ค่ายประกาศ", paper="งานวิจัย", fast="Ultra3 ⚡ เร็ว", over="~106–110% (เกินสเกล)",
        xlabel="CER (%) บน FLEURS ภาษาลาว", clip="คลิป", ours_legend="Ultra3 ⚡ (เรา)",
        omni="Phonepadith (ฐาน Meta)", sell="OmniASR v2 (เรา, ขายได้)",
        foot="ที่มา: วัดเอง = สคริปต์ eval_rivals.py / eval_ctc_local.py / eval_mms.py ชุดสอบเดียวกัน (FLEURS test 100 ประโยค) · BuzzASR = huggingface.co/BuzzASR/lao · "
             "SeamlessM4T, Whisper = งานวิจัย Fleurs-SLU (arXiv 2501.06117)\nOmniASR Lao = huggingface.co/Phonepadith/omniASR_CTC_300M_v2_Laos (ไม่เปิดเผยข้อมูลเทรน) · "
             "Google, Azure, ElevenLabs, AssemblyAI รองรับภาษาลาวแต่ไม่ประกาศ CER · อัปเดต 17 พ.ย. 2026",
        out="ranking_cer.png"),
    "lo": dict(
        title="ອັດຕາຜິດຂອງ AI ຖອດສຽງພາສາລາວ (CER ຍິ່ງຕ່ຳຍິ່ງດີ)", left="ຊຸດທົດສອບມາດຕະຖານ FLEURS",
        right="ຄລິບຕົວຈິງຂອງທີມ (ວັດແທກເອງທັງໝົດ)", ours="ທີມເຮົາ", nuol="ມ.ແຫ່ງຊາດລາວ", self="ວັດແທກເອງ",
        vendor="ຜູ້ພັດທະນາປະກາດ", paper="ບົດຄົ້ນຄວ້າ", fast="Ultra3 ⚡ ໄວ", over="~106–110% (ເກີນຂອບເຂດ)",
        xlabel="CER (%) ໃນ FLEURS ພາສາລາວ", clip="ຄລິບ", ours_legend="Ultra3 ⚡ (ທີມເຮົາ)",
        omni="Phonepadith (ພື້ນຖານ Meta)", sell="OmniASR v2 (ທີມເຮົາ, ຂາຍໄດ້)",
        foot="ທີ່ມາ: ວັດແທກເອງ = ສະຄຣິບ eval_rivals.py / eval_ctc_local.py / eval_mms.py ຊຸດທົດສອບດຽວກັນ (FLEURS test 100 ປະໂຫຍກ) · BuzzASR = huggingface.co/BuzzASR/lao · "
             "SeamlessM4T, Whisper = ບົດຄົ້ນຄວ້າ Fleurs-SLU (arXiv 2501.06117)\nOmniASR Lao = huggingface.co/Phonepadith/omniASR_CTC_300M_v2_Laos (ບໍ່ເປີດເຜີຍຂໍ້ມູນເທຣນ) · "
             "Google, Azure, ElevenLabs, AssemblyAI ຮອງຮັບພາສາລາວ ແຕ່ບໍ່ໄດ້ປະກາດ CER · ອັບເດດ 17 ພະຈິກ 2026",
        out="ranking_cer_lo.png"),
}
L = T["lo" if "--lo" in sys.argv else "th"]

OURS, RIVAL, DIM, INK, MUTED, BG = "#c8102e", "#2f4a8f", "#9aa3bd", "#141a2e", "#5b6380", "#ffffff"

fleurs = [  # (ชื่อ, ค่าย, CER, ที่มา)
    ("Ultra3", L["ours"], 15.7, L["self"]),
    ("OmniASR 300M Lao", L["omni"], 15.9, L["self"]),
    (L["fast"], L["ours"], 16.0, L["self"]),
    (L["sell"], L["ours"], 20.3, L["self"]),
    ("MMS-1b-all", "Meta", 20.4, L["self"]),
    ("CTC XLS-R v1", L["ours"], 20.6, L["self"]),
    ("BuzzASR Lao", "BuzzASR", 21.8, L["vendor"]),
    ("SiangLao XLSR-53", L["nuol"], 27.6, L["self"]),
    ("SiangLao XLS-R", L["nuol"], 28.0, L["self"]),
    ("SeamlessM4T v2", "Meta", 30.3, L["paper"]),
    ("Whisper large-v3", "OpenAI", 106.0, L["paper"]),
]
team = {  # คลิปจริงของทีม (ไม่รวม Download 6 เพราะซับต้นแบบแก้จากผล AI เรา)
    # (Ultra3 ⚡, OmniASR v2 เรา ขายได้, OmniASR Lao Phonepadith, SiangLao XLS-R)
    f"{L['clip']} Download (2)": (12.3, 21.8, 24.9, 34.8),
    f"{L['clip']} Download (3)": (7.8, 11.4, 11.7, 21.8),
    f"{L['clip']} lekha-chai": (8.7, 9.0, 5.3, 18.0),
}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.6), gridspec_kw={"width_ratios": [1.35, 1]}, facecolor=BG)
fig.suptitle(L["title"], fontsize=20, fontweight="bold", color=INK, x=0.02, ha="left", y=0.98)

# --- ซ้าย: FLEURS ---
MAX = 40
names = [f"{n}\n{c}" for n, c, _, _ in fleurs][::-1]
vals = [v for _, _, v, _ in fleurs][::-1]
cols = [OURS if c == L["ours"] else (DIM if v > MAX else RIVAL) for _, c, v, _ in fleurs][::-1]
srcs = [s for _, _, _, s in fleurs][::-1]
bars = a1.barh(names, [min(v, MAX) for v in vals], color=cols, height=0.62)
for b, v, s in zip(bars, vals, srcs):
    inside = v > MAX
    x = min(v, MAX)
    a1.text(x - 0.6 if inside else x + 0.6, b.get_y() + b.get_height() / 2, L["over"] if inside else f"{v:.1f}%",
            va="center", ha="right" if inside else "left", fontsize=12, fontweight="bold", color="white" if inside else INK)
    if not inside:
        a1.text(x + 0.6, b.get_y() - 0.02, s, va="top", ha="left", fontsize=9, color=MUTED)
a1.set_xlim(0, MAX + 6)
a1.set_xlabel(L["xlabel"], color=MUTED, fontsize=11)
a1.set_title(L["left"], loc="left", fontsize=14, color=INK, pad=10)

# --- ขวา: คลิปจริงของทีม ---
x = np.arange(len(team))
w = 0.2
series = [(L["ours_legend"], OURS), (L["sell"], "#f08a9a"), ("OmniASR Lao (Phonepadith)", "#e08a1e"), ("SiangLao XLS-R", RIVAL)]
for i, (name, col) in enumerate(series):
    ys = [v[i] for v in team.values()]
    bb = a2.bar(x + (i - 1.5) * w, ys, w, color=col, label=name)
    for b, y in zip(bb, ys):
        a2.text(b.get_x() + b.get_width() / 2, y + 0.6, f"{y:.1f}", ha="center", fontsize=9, color=INK)
a2.set_xticks(x, list(team.keys()), fontsize=11)
a2.set_ylabel("CER (%)", color=MUTED, fontsize=11)
a2.set_ylim(0, 42)
a2.legend(frameon=False, fontsize=11, loc="upper right")
a2.set_title(L["right"], loc="left", fontsize=14, color=INK, pad=10)

for ax in (a1, a2):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#e1e4ee")
    ax.tick_params(colors=INK)
    ax.set_axisbelow(True)
a1.grid(axis="x", color="#eef0f6")
a2.grid(axis="y", color="#eef0f6")

fig.text(0.02, 0.015, L["foot"], fontsize=9, color=MUTED)
plt.tight_layout(rect=(0, 0.06, 1, 0.94))
out = RESULTS / L["out"]
plt.savefig(out, dpi=160, facecolor=BG)
print(out)
