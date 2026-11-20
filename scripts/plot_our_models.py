"""กราฟจัดอันดับโมเดลของทีมเราเอง (CER ยิ่งต่ำยิ่งดี) — แยกสี: ขายได้ / ใช้ในทีมเท่านั้น

  .venv\\Scripts\\python.exe scripts\\plot_our_models.py        → results/our_models_cer.png (ไทย)
  .venv\\Scripts\\python.exe scripts\\plot_our_models.py --lo   → results/our_models_cer_lo.png (ລາວ)
ตัวเลข: results/eval_rivals.txt, results/ctc_lm_ctc-omni-v2.txt, memory v5/v6 (FLEURS 100 ชุดเดียวกัน)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

plt.rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "sans-serif"]
SELL, TEAM, INK, MUTED, GRID, BG = "#d6336c", "#5b6b8c", "#141a2e", "#5b6380", "#e9ecf2", "#ffffff"
LO = "--lo" in sys.argv
TR = {
    "Ultra3 ⚡ เร็ว": "Ultra3 ⚡ ໄວ",
    "OmniASR v2 + ภาษาลาวช่วยเดา": "OmniASR v2 + ພາສາລາວຊ່ວຍເດົາ",
    "อันดับโมเดลถอดเสียงลาวของทีมเรา (CER ยิ่งต่ำยิ่งดี)": "ອັນດັບໂມເດວຖອດສຽງລາວຂອງທີມເຮົາ (CER ຍິ່ງຕ່ຳຍິ່ງດີ)",
    "■ ขายได้ (ข้อมูลใช้เชิงการค้าได้)": "■ ຂາຍໄດ້ (ຂໍ້ມູນໃຊ້ທາງການຄ້າໄດ້)",
    "■ ใช้ในทีมเท่านั้น (เทรนจาก SiangLao/ข่าว RFA ห้ามขาย)": "■ ໃຊ້ພາຍໃນທີມເທົ່ານັ້ນ (ເທຣນຈາກ SiangLao/ຂ່າວ RFA ຫ້າມຂາຍ)",
    "ชุดสอบมาตรฐาน FLEURS (100 ประโยค)": "ຊຸດທົດສອບມາດຕະຖານ FLEURS (100 ປະໂຫຍກ)",
    "คลิปจริงของทีม (เฉลี่ย 3 คลิป)": "ຄລິບຕົວຈິງຂອງທີມ (ສະເລ່ຍ 3 ຄລິບ)",
    "FOOT": "ວັດແທກດ້ວຍສະຄຣິບຊຸດດຽວກັນ (eval_rivals.py / eval_ctc_local.py / ctc_lm.py) · ບໍ່ນັບຄລິບ Download (6) ເພາະຄຳບັນຍາຍຕົ້ນແບບແກ້ຈາກຜົນ AI ຂອງເຮົາ · "
            "ພາສາລາວຊ່ວຍເດົາ = beam search + char 6-gram ຈາກຂໍ້ຄວາມທີ່ຂາຍໄດ້ (FLEURS, KitTzk, CV, YouTube CC-BY, ວິກິພີເດຍລາວ CC BY-SA)\n"
            "OmniASR = ພື້ນຖານ Meta omniASR-CTC-300M (Apache-2.0) · Ultra/v3–v6 = ພື້ນຖານ Whisper small · ອັບເດດ 18 ພະຈິກ 2026",
}
FOOT_TH = ("วัดด้วยสคริปต์ชุดเดียวกัน (eval_rivals.py / eval_ctc_local.py / ctc_lm.py) · ไม่นับคลิป Download (6) เพราะซับต้นแบบแก้จากผล AI เรา · "
           "ภาษาลาวช่วยเดา = beam search + char 6-gram จากข้อความขายได้ (FLEURS, KitTzk, CV, YouTube CC-BY, วิกิพีเดียลาว CC BY-SA)\n"
           "OmniASR = ฐาน Meta omniASR-CTC-300M (Apache-2.0) · Ultra/v3–v6 = ฐาน Whisper small · อัปเดต 18 พ.ย. 2026")


def t(s: str) -> str:
    return TR.get(s, s) if LO else s

fleurs = [  # (ชื่อ, CER, ขายได้)
    ("Ultra3 (v3+v6)", 15.7, False), ("v6", 15.8, False), ("Ultra3 ⚡ เร็ว", 16.0, False), ("v5", 16.0, False),
    ("Ultra2 (v3+v5)", 16.3, False), ("v4", 16.8, False), ("Ultra (v3+v4)", 16.8, False),
    ("OmniASR v2 + ภาษาลาวช่วยเดา", 17.4, True), ("v3", 18.0, False), ("OmniASR v2", 20.3, True),
    ("CTC XLS-R v1", 20.6, False), ("OmniASR v1", 23.3, True),
]
team = [  # (ชื่อ, D2, D3, lekha, ขายได้) — ไม่นับ Download (6) เพราะซับต้นแบบแก้จากผล AI เรา
    ("Ultra3 ⚡", 12.3, 7.8, 8.7, False),
    ("OmniASR v2 + ภาษาลาวช่วยเดา", 18.5, 10.3, 9.0, True),
    ("OmniASR v2", 21.8, 11.4, 9.0, True),
    ("OmniASR v1", 25.7, 15.0, 8.1, True),
    ("CTC XLS-R v1", 28.2, 14.5, 7.1, False),
]
team = sorted(team, key=lambda r: sum(r[1:4]) / 3)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.15, 1]}, facecolor=BG)
fig.subplots_adjust(left=0.2, right=0.97, top=0.83, bottom=0.14, wspace=0.55)
fig.text(0.02, 0.95, t("อันดับโมเดลถอดเสียงลาวของทีมเรา (CER ยิ่งต่ำยิ่งดี)"), fontsize=20, weight="bold", color=INK)
fig.text(0.02, 0.905, t("■ ขายได้ (ข้อมูลใช้เชิงการค้าได้)"), color=SELL, fontsize=12, weight="bold")
fig.text(0.24 if LO else 0.21, 0.905, t("■ ใช้ในทีมเท่านั้น (เทรนจาก SiangLao/ข่าว RFA ห้ามขาย)"), color=TEAM, fontsize=12, weight="bold")

# ซ้าย: FLEURS
names = [t(r[0]) for r in fleurs][::-1]
vals = [r[1] for r in fleurs][::-1]
cols = [SELL if r[2] else TEAM for r in fleurs][::-1]
y = range(len(names))
a1.barh(y, vals, color=cols, height=0.68)
for i, v in enumerate(vals):
    a1.text(v + 0.25, i, f"{v:.1f}%", va="center", fontsize=11, color=INK, weight="bold")
    a1.text(-0.5, i, f"#{len(vals) - i}", va="center", ha="right", fontsize=10, color=MUTED, transform=a1.transData)
a1.set_yticks(list(y), names, fontsize=11)
for lab, c in zip(a1.get_yticklabels(), cols):
    lab.set_color(c)
    lab.set_weight("bold" if c == SELL else "normal")
a1.set_xlim(0, 27)
a1.set_title(t("ชุดสอบมาตรฐาน FLEURS (100 ประโยค)"), loc="left", fontsize=14, color=INK, pad=12)
a1.set_xlabel("CER (%)", color=MUTED)
a1.tick_params(axis="y", pad=26)

# ขวา: คลิปทีม เฉลี่ย 3 คลิป
tn = [t(r[0]) for r in team][::-1]
tv = [sum(r[1:4]) / 3 for r in team][::-1]
tc = [SELL if r[4] else TEAM for r in team][::-1]
ty = range(len(tn))
a2.barh(ty, tv, color=tc, height=0.6)
for i, (v, r) in enumerate(zip(tv, team[::-1])):
    a2.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=12, color=INK, weight="bold")
    a2.text(v + 0.3, i - 0.3, f"D2 {r[1]} · D3 {r[2]} · lekha {r[3]}", va="center", fontsize=8.5, color=MUTED)
a2.set_yticks(list(ty), tn, fontsize=11)
for lab, c in zip(a2.get_yticklabels(), tc):
    lab.set_color(c)
    lab.set_weight("bold" if c == SELL else "normal")
a2.set_xlim(0, 24)
a2.set_title(t("คลิปจริงของทีม (เฉลี่ย 3 คลิป)"), loc="left", fontsize=14, color=INK, pad=12)
a2.set_xlabel("CER (%)", color=MUTED)

for a in (a1, a2):
    a.set_facecolor(BG)
    a.grid(axis="x", color=GRID)
    a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(axis="x", colors=MUTED, length=0)
    a.tick_params(axis="y", length=0)

fig.text(0.02, 0.03, TR["FOOT"] if LO else FOOT_TH, fontsize=8.5, color=MUTED)
out = RESULTS / ("our_models_cer_lo.png" if LO else "our_models_cer.png")
plt.savefig(out, dpi=150, facecolor=BG)
print(out)
