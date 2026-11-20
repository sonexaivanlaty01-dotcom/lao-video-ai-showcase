"""กราฟอันดับโมเดลของทีม แบบมืออาชีพ (หลักแบบ FT / The Economist / Our World in Data)
- หัวข้อบอก "ข้อสรุป" ไม่ใช่แค่ชื่อกราฟ · หัวรองบอกว่าวัดอะไร
- เน้นสีเดียว (ขายได้ = น้ำเงิน) ที่เหลือเป็นเทา (บริบท) · ป้ายตัวเลขติดปลายแท่ง ไม่ต้องอ่านแกน
- เส้นตาราง/แกนจางสุด · ที่มา + หมายเหตุวิธีวัดด้านล่าง
สีจากชุดสีอ้างอิงที่ผ่านการตรวจตาบอดสี (dataviz skill, references/palette.md)

  .venv\\Scripts\\python.exe scripts\\plot_models_pro.py        → results/models_pro.png (ไทย)
  .venv\\Scripts\\python.exe scripts\\plot_models_pro.py --lo   → results/models_pro_lo.png (ລາວ)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

plt.rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "sans-serif"]
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
HL, CTX = "#2a78d6", "#b9b8b0"  # ขายได้ = น้ำเงิน (slot 1) · ใช้ในทีม = เทาบริบท
LO = "--lo" in sys.argv

S = {
    "th": dict(
        title="OmniASR v2 ไล่ทันโมเดลภายในทีมได้เร็ว — เป็นตัวที่ดีที่สุดในกลุ่ม \"ขายได้\"",
        sub="อัตราผิดระดับตัวอักษร (CER, %) ของโมเดลถอดเสียงลาวที่ทีมสร้าง · ยิ่งต่ำยิ่งดี",
        lg_sell="ขายได้ (ข้อมูลใช้เชิงการค้าได้)", lg_team="ใช้ในทีมเท่านั้น (ข้อมูลห้ามขาย)",
        p1="A  ชุดทดสอบมาตรฐาน FLEURS", p1s="100 ประโยค · ทุกโมเดลวัดชุดเดียวกัน",
        p2="B  คลิปจริงของทีม: OmniASR ดีขึ้นทุกรุ่น", p2s="เฉลี่ย 3 คลิป (Download 2, Download 3, lekha)",
        lm="OmniASR v2 + ภาษาลาวช่วยเดา", fast="Ultra3 ⚡ เร็ว",
        x1="v1", x2="v2", x3="v2 + ภาษาลาว\nช่วยเดา",
        ref="Ultra3 ⚡ (ภายในทีม) 9.6%", gap="ห่างอีก 3.0", gain="−3.7 จาก v1",
        note="ที่มา: วัดเองด้วยสคริปต์ชุดเดียวกัน (eval_rivals.py, eval_ctc_local.py, ctc_lm.py) · 18 พ.ย. 2026\n"
             "หมายเหตุ: ไม่นับคลิป Download (6) เพราะซับต้นแบบแก้จากผล AI ของทีม · \"ภาษาลาวช่วยเดา\" = beam search + char 6-gram จากข้อความที่ขายได้ "
             "(FLEURS, KitTzk, Common Voice, YouTube CC-BY, วิกิพีเดียลาว CC BY-SA)\n"
             "OmniASR = ฐาน Meta omniASR-CTC-300M (Apache-2.0) · Ultra / v3–v6 = ฐาน Whisper small เทรนด้วย SiangLao และข่าว RFA (ใช้ในทีมเท่านั้น)",
        out="models_pro.png"),
    "lo": dict(
        title="OmniASR v2 ໄລ່ທັນໂມເດວພາຍໃນທີມໄດ້ໄວ — ເປັນຕົວທີ່ດີທີ່ສຸດໃນກຸ່ມ \"ຂາຍໄດ້\"",
        sub="ອັດຕາຜິດລະດັບຕົວອັກສອນ (CER, %) ຂອງໂມເດວຖອດສຽງລາວທີ່ທີມສ້າງ · ຍິ່ງຕ່ຳຍິ່ງດີ",
        lg_sell="ຂາຍໄດ້ (ຂໍ້ມູນໃຊ້ທາງການຄ້າໄດ້)", lg_team="ໃຊ້ພາຍໃນທີມເທົ່ານັ້ນ (ຂໍ້ມູນຫ້າມຂາຍ)",
        p1="A  ຊຸດທົດສອບມາດຕະຖານ FLEURS", p1s="100 ປະໂຫຍກ · ທຸກໂມເດວວັດແທກຊຸດດຽວກັນ",
        p2="B  ຄລິບຕົວຈິງຂອງທີມ: OmniASR ດີຂຶ້ນທຸກລຸ້ນ", p2s="ສະເລ່ຍ 3 ຄລິບ (Download 2, Download 3, lekha)",
        lm="OmniASR v2 + ພາສາລາວຊ່ວຍເດົາ", fast="Ultra3 ⚡ ໄວ",
        x1="v1", x2="v2", x3="v2 + ພາສາລາວ\nຊ່ວຍເດົາ",
        ref="Ultra3 ⚡ (ພາຍໃນທີມ) 9.6%", gap="ຫ່າງອີກ 3.0", gain="−3.7 ຈາກ v1",
        note="ທີ່ມາ: ວັດແທກເອງດ້ວຍສະຄຣິບຊຸດດຽວກັນ (eval_rivals.py, eval_ctc_local.py, ctc_lm.py) · 18 ພະຈິກ 2026\n"
             "ໝາຍເຫດ: ບໍ່ນັບຄລິບ Download (6) ເພາະຄຳບັນຍາຍຕົ້ນແບບແກ້ຈາກຜົນ AI ຂອງທີມ · \"ພາສາລາວຊ່ວຍເດົາ\" = beam search + char 6-gram ຈາກຂໍ້ຄວາມທີ່ຂາຍໄດ້ "
             "(FLEURS, KitTzk, Common Voice, YouTube CC-BY, ວິກິພີເດຍລາວ CC BY-SA)\n"
             "OmniASR = ພື້ນຖານ Meta omniASR-CTC-300M (Apache-2.0) · Ultra / v3–v6 = ພື້ນຖານ Whisper small ເທຣນດ້ວຍ SiangLao ແລະ ຂ່າວ RFA (ໃຊ້ພາຍໃນທີມເທົ່ານັ້ນ)",
        out="models_pro_lo.png"),
}
L = S["lo" if LO else "th"]

fleurs = [  # (ชื่อ, CER, ขายได้)
    ("Ultra3 (v3+v6)", 15.7, False), ("v6", 15.8, False), (L["fast"], 16.0, False), ("v5", 16.0, False),
    ("Ultra2 (v3+v5)", 16.3, False), ("v4", 16.8, False), ("Ultra (v3+v4)", 16.8, False),
    (L["lm"], 17.4, True), ("v3", 18.0, False), ("OmniASR v2", 20.3, True),
    ("CTC XLS-R v1", 20.6, False), ("OmniASR v1", 23.3, True),
]
prog = [16.3, 14.1, 12.6]  # OmniASR บนคลิปทีม (เฉลี่ย 3 คลิป): v1, v2, v2+LM
ULTRA = 9.6

fig = plt.figure(figsize=(16, 9), facecolor=SURF)
fig.text(0.035, 0.945, L["title"], fontsize=21, weight="bold", color=INK)
fig.text(0.035, 0.905, L["sub"], fontsize=13, color=INK2)
# คำอธิบายสี (legend) แถวเดียว ใต้หัวข้อ
for i, (c, t) in enumerate([(HL, L["lg_sell"]), (CTX, L["lg_team"])]):
    x = 0.035 + i * 0.25
    fig.patches.append(plt.Rectangle((x, 0.858), 0.011, 0.019, color=c, transform=fig.transFigure, figure=fig))
    fig.text(x + 0.017, 0.8675, t, fontsize=11.5, color=INK2, va="center")

# ---------- A: FLEURS ----------
a1 = fig.add_axes([0.2, 0.17, 0.33, 0.6])
rows = fleurs[::-1]
y = range(len(rows))
cols = [HL if r[2] else CTX for r in rows]
a1.barh(y, [r[1] for r in rows], color=cols, height=0.62, zorder=2)
for i, (n, v, s) in enumerate(rows):
    a1.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=11, color=INK if s else INK2, weight="bold" if s else "normal")
a1.set_yticks(list(y), [r[0] for r in rows], fontsize=11.5)
for lab, (n, v, s) in zip(a1.get_yticklabels(), rows):
    lab.set_color(INK if s else INK2)
    lab.set_weight("bold" if s else "normal")
a1.set_xlim(0, 26)
a1.set_xticks([0, 5, 10, 15, 20, 25])
fig.text(0.035, 0.80, L["p1"], fontsize=14.5, weight="bold", color=INK)
fig.text(0.035, 0.775, L["p1s"], fontsize=11, color=MUTED)

# ---------- B: ความคืบหน้าบนคลิปทีม ----------
a2 = fig.add_axes([0.63, 0.17, 0.33, 0.6])
xs = [0, 1, 2]
a2.axhline(ULTRA, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
a2.text(2.25, ULTRA - 0.45, L["ref"], color=INK2, fontsize=11, ha="right", va="top")
a2.plot(xs, prog, color=HL, lw=2.5, zorder=3, solid_capstyle="round")
a2.scatter(xs, prog, s=110, color=HL, edgecolor=SURF, linewidth=2.5, zorder=4)
for x, v in zip(xs, prog):
    a2.text(x, v + 0.45, f"{v:.1f}", ha="center", va="bottom", fontsize=14, weight="bold", color=INK)
# ช่องว่างที่เหลือ
a2.annotate("", xy=(2.13, ULTRA + 0.1), xytext=(2.13, prog[-1] - 0.1),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
a2.text(2.2, (ULTRA + prog[-1]) / 2, L["gap"], fontsize=11, color=INK2, va="center")
a2.text(1.0, 16.2, L["gain"], fontsize=11.5, color=HL, weight="bold", ha="center")
a2.set_xticks(xs, [L["x1"], L["x2"], L["x3"]], fontsize=12)
a2.set_xlim(-0.35, 2.75)
a2.set_ylim(0, 18)
a2.set_yticks([0, 5, 10, 15])
fig.text(0.575, 0.80, L["p2"], fontsize=14.5, weight="bold", color=INK)
fig.text(0.575, 0.775, L["p2s"], fontsize=11, color=MUTED)

for a, ax_grid in ((a1, "x"), (a2, "y")):
    a.set_facecolor(SURF)
    a.grid(axis=ax_grid, color=GRID, lw=0.8, zorder=0)
    a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(AXIS)
    a.tick_params(axis="both", length=0)
    a.tick_params(axis=ax_grid, colors=MUTED, labelsize=10.5)
a2.tick_params(axis="x", colors=INK2)

fig.add_artist(plt.Line2D([0.035, 0.965], [0.085, 0.085], color=GRID, lw=0.8, transform=fig.transFigure))
fig.text(0.035, 0.018, L["note"], fontsize=9, color=MUTED, linespacing=1.6)
out = RESULTS / L["out"]
plt.savefig(out, dpi=200, facecolor=SURF)
print(out)
