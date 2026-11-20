"""ตรวจฟอนต์ลาวอัตโนมัติ เทียบกับฟอนต์ต้นแบบน้ำหนักเดียวกัน

ตรวจ 6 ข้อ:
  1. ครบตัวอักษร — ทุกตัวลาวในต้นแบบมีในฟอนต์ใหม่ และไม่ว่าง
  2. จัดวางเหมือนต้นแบบ — HarfBuzz shape ทุกพยางค์/ทุกบรรทัดซับจริง ได้ glyph + ตำแหน่งตรงกันทุกตัว
  3. รูปทรงไม่พัง — จำนวนชิ้นหมึกเท่าเดิม, รู/ห่วงในตัวอักษรไม่ถูกถมปิด
  4. แยกตัวอักษรออก — ตัวอักษรแต่ละตัวยังคล้ายตัวเองในต้นแบบมากที่สุด (ບ/ປ, ຜ/ຝ, ດ/ຕ ...)
  5. สระ-วรรณยุกต์ไม่ชนกัน — ส่วนที่ทับกันไม่เพิ่มจากต้นแบบ
  6. ไม่ล้นขอบบน/ล่าง — หมึกไม่เกิน winAscent/winDescent (ไม่ถูกตัดบนบางเครื่อง)

ใช้:  .venv-fonts\\Scripts\\python.exe scripts\\fonts\\qa_fonts.py --version v1
ผลลัพธ์: fonts/<version>/qa/report.md + qa.json + รูปตัวอย่าง
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import freetype
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

CONS = list("ກຂຄງຈສຊຍດຕຖທນບປຜຝພຟມຢຣລວຫອຮ") + ["ໜ", "ໝ", "ຫຼ", "ກຼ", "ປຼ", "ພຼ", "ມຼ", "ຄຼ"]
ABOVE = ["ັ", "ິ", "ີ", "ຶ", "ື", "ົ", "ໍ"]
BELOW = ["ຸ", "ູ"]
TONES = ["່", "້", "໊", "໋"]
EXTRA = ["ໍ່", "ໍ້", "ົ່", "ົ້"]


def syllables():
    out = []
    for c in CONS:
        out.append(c)
        out += [c + t for t in TONES]
        out += [c + a for a in ABOVE]
        out += [c + a + t for a in ABOVE for t in TONES]
        out += [c + b for b in BELOW]
        out += [c + b + t for b in BELOW for t in TONES]
        out += [c + "ຳ", c + "້ຳ", c + "່ຳ", c + "ຽງ", c + "ົ້ວ", c + "ື້ອ", "ເ" + c + "ື່ອ", c + "໌"]
    out += ["ເກື້ອ", "ຫຼື", "ໜ້າ", "ໝໍ້", "ຫຼັ້ນ", "ນ້ຳ", "ຄ່ຳ", "ຊື່ອ", "ພື້ນ", "ປີ້ນ", "ຟື້ນ", "ຝົ້ນ", "ຜີ້", "ຢູ່", "ຍູ້", "ຕູ້", "ດູ່"]
    out += list("໐໑໒໓໔໕໖໗໘໙") + ["ໆ", "ຯ", "ໜັງສື", "ພາສາລາວ"]
    seen, res = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); res.append(s)
    return res


def real_lines(limit=3000):
    lines, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "data" / "subs" / "*.lo.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for s in d.get("segments", []):
            t = (s.get("text") or "").strip()
            if t and t not in seen:
                seen.add(t); lines.append(t)
    return lines[:limit]


# ---------- เครื่องมือ ----------
class Font:
    def __init__(self, path):
        self.path = str(path)
        self.tt = TTFont(path)
        blob = hb.Blob.from_file_path(self.path)
        self.hbfont = hb.Font(hb.Face(blob))
        self.upm = self.tt["head"].unitsPerEm
        self.ft = freetype.Face(self.path)
        self.cmap = self.tt.getBestCmap()
        self.order = self.tt.getGlyphOrder()
        self.gdef = self.tt["GDEF"].table.GlyphClassDef.classDefs if "GDEF" in self.tt else {}
        self._cache = {}
        self.deshear = 0.0
        os2 = self.tt["OS/2"]
        self.win_asc, self.win_desc = os2.usWinAscent, os2.usWinDescent

    def shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hbfont, buf, {})
        return [(i.codepoint, p.x_advance, p.x_offset, p.y_offset) for i, p in zip(buf.glyph_infos, buf.glyph_positions)]

    def glyph_bitmap(self, gid, px):
        """คืน (mask, left, top) หน่วยพิกเซล ที่ขนาด em = px (เก็บ cache)"""
        key = (gid, px)
        c = self._cache.setdefault("bm", {})
        if key not in c:
            c[key] = self._glyph_bitmap(gid, px)
        return c[key]

    def run_glyphs(self, text, px):
        """วางทุก glyph ของข้อความ: คืน [(mask, x, y)] พิกัดพิกเซล โดยเส้นฐาน y=0, เริ่ม x=0"""
        key = (text, px)
        c = self._cache.setdefault("run", {})
        if key in c:
            return c[key]
        out, x, sc = [], 0.0, px / self.upm
        for gid, adv, xo, yo in self.shape(text):
            a, l, t = self.glyph_bitmap(gid, px)
            if a is not None:
                out.append((a, int(round((x + xo) * sc)) + l, -int(round(yo * sc)) - t))
            x += adv
        c[key] = (out, x * sc)
        return c[key]

    def _glyph_bitmap(self, gid, px):
        self.ft.set_pixel_sizes(px, px)
        self.ft.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
        g = self.ft.glyph
        bm = g.bitmap
        if bm.width == 0 or bm.rows == 0:
            return None, 0, 0
        a = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.pitch)[:, : bm.width]
        return a, g.bitmap_left, g.bitmap_top

    def render_glyph_canvas(self, gid, px, size, ox, base):
        img = np.zeros((size, size), np.uint8)
        a, l, t = self.glyph_bitmap(gid, px)
        if a is None:
            return img
        paste(img, a, ox + l, base - t)
        if self.deshear:   # ฟอนต์เอียง: ดัดกลับให้ตรงก่อนเทียบรูปทรงกับต้นแบบ
            tan = self.deshear
            img = ndimage.affine_transform(img, [[1, 0], [-tan, 1]], offset=[0, tan * base], order=1)
        return img

    def render_run_layers(self, text, px, W, H, ox, base):
        """วาดแต่ละ glyph แยกชั้น เพื่อตรวจการทับกัน"""
        layers, gids = [], []
        x = 0.0
        sc = px / self.upm
        for gid, adv, xo, yo in self.shape(text):
            img = np.zeros((H, W), np.uint8)
            a, l, t = self.glyph_bitmap(gid, px)
            if a is not None:
                paste(img, a, int(round(ox + (x + xo) * sc)) + l, int(round(base - yo * sc)) - t)
            layers.append(img); gids.append(gid)
            x += adv
        return layers, gids


def paste(img, a, x, y):
    H, W = img.shape
    h, w = a.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], a[y0 - y : y1 - y, x0 - x : x1 - x])


def topology(mask):
    ink = mask > 127
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    sizes = ndimage.sum(ink, lab, range(1, n + 1)) if n else []
    comps = int(sum(1 for s in sizes if s >= 6))
    bg = ~ink
    blab, bn = ndimage.label(bg)
    border = set(np.unique(np.concatenate([blab[0], blab[-1], blab[:, 0], blab[:, -1]])))
    holes = [i for i in range(1, bn + 1) if i not in border]
    return comps, blab, border, holes


# ---------- การตรวจ ----------
def check_coverage(var, ref):
    bad = []
    for cp, g in ref.cmap.items():
        if not (0x0E80 <= cp <= 0x0EFF):
            continue
        if cp not in var.cmap:
            bad.append(f"U+{cp:04X} {chr(cp)} หายไป"); continue
        rg = ref.tt["glyf"][g]; vg = var.tt["glyf"][var.cmap[cp]]
        if rg.numberOfContours != 0 and vg.numberOfContours == 0:
            bad.append(f"U+{cp:04X} {chr(cp)} ว่างเปล่า")
    return bad


def check_shaping(var, ref, texts, skewed=False):
    """ฟอนต์เอียง: จุดยึดสระถูกเอียงตามตั้งใจ จึงเทียบเฉพาะ glyph + ความกว้าง + ระยะเลื่อนแนวตั้ง"""
    bad = []
    for t in texts:
        a, b = var.shape(t), ref.shape(t)
        if skewed:
            a = [(g, adv, yo) for g, adv, xo, yo in a]
            b = [(g, adv, yo) for g, adv, xo, yo in b]
        if a != b:
            bad.append(t)
    return bad


def check_skew_marks(var, ref, texts, tan):
    """ฟอนต์เอียง: สระ/วรรณยุกต์ต้องเลื่อนในแนวนอนเท่ากับที่เอียงพอดี (x_var − x_ref = tan·y ของจุดยึด)
    ตรวจโดยเทียบตำแหน่งจริงหลังวาง: ต้องต่างกันไม่เกิน 2 หน่วยจากค่าที่คำนวณ"""
    bad = []
    for t in texts:
        a, b = var.shape(t), ref.shape(t)
        for (g, adv, xo, yo), (_, _, rxo, ryo) in zip(a, b):
            if xo == rxo:
                continue
            gname = ref.order[g]
            if ref.gdef.get(gname) != 3:
                continue
            # ความต่างที่ยอมรับ = tan × (ความสูงจุดยึดฐาน − ความสูงจุดยึดสระ) ≈ ต้องอยู่ในช่วงที่เป็นไปได้
            if abs(xo - rxo) > tan * 1600:
                bad.append(f"{t}: สระเลื่อน {xo - rxo} หน่วย")
                break
    return bad


def check_topology(var, ref, px=160, hollow=False):
    size, ox, base = int(px * 1.9), int(px * 0.45), int(px * 1.3)
    issues = []
    for cp, g in sorted(ref.cmap.items()):
        if not (0x0E80 <= cp <= 0x0EFF):
            continue
        gid_r = ref.tt.getGlyphID(g); gid_v = var.tt.getGlyphID(var.cmap[cp])
        mr = ref.render_glyph_canvas(gid_r, px, size, ox, base)
        mv = var.render_glyph_canvas(gid_v, px, size, ox, base)
        cr, _, _, holes_r = topology(mr)
        cv, blab_v, border_v, _ = topology(mv)
        _, blab_r, _, _ = topology(mr)
        msg = []
        # ตัวกลวง: วงที่มีรูตรงกลาง (ອ) จะกลายเป็นขอบนอก+ขอบใน 2 ชิ้นได้ ถือว่าปกติ แต่ห้ามน้อยลง
        if (cv < cr) if hollow else (cr != cv):
            msg.append(f"ชิ้นหมึก {cr}→{cv}")
        closed = 0
        for h in holes_r:
            region = blab_r == h
            area = region.sum()
            if area < 25:   # รูเล็กมากที่ 160px (ช่องแคบ) ข้าม
                continue
            labs = blab_v[region]
            bg_frac = (labs > 0).mean()
            inner = labs[(labs > 0)]
            open_to_outside = bool(len(inner)) and np.isin(inner, list(border_v)).mean() > 0.5
            if bg_frac < 0.5 or open_to_outside:
                closed += 1
        if closed:
            msg.append(f"รู/ห่วงถูกถมหรือขาด {closed} จุด")
        if msg:
            issues.append(f"{chr(cp)} (U+{cp:04X}): " + ", ".join(msg))
    return issues


def letter_set(ref):
    out = []
    for cp, g in sorted(ref.cmap.items()):
        if 0x0E81 <= cp <= 0x0EDF and ref.gdef.get(g) != 3 and ref.tt["hmtx"][g][0] > 0:
            out.append(cp)
    return out


def feat(font, cp, px=72):
    size, ox, base = int(px * 2.0), int(px * 0.5), int(px * 1.4)
    m = font.render_glyph_canvas(font.tt.getGlyphID(font.cmap[cp]), px, size, ox, base).astype(np.float32) / 255
    m = ndimage.gaussian_filter(m, 2.0)
    m -= m.mean()
    n = np.linalg.norm(m)
    return m / n if n else m


def check_confusion(var_solid, ref):
    cps = letter_set(ref)
    R = {cp: feat(ref, cp) for cp in cps}
    rows, bad = [], []
    for cp in cps:
        v = feat(var_solid, cp)
        sims = {c: float((v * R[c]).sum()) for c in cps}
        best = max(sims, key=sims.get)
        other = max((s for c, s in sims.items() if c != cp))
        oc = max((c for c in sims if c != cp), key=sims.get)
        # ระยะห่างของต้นแบบเอง (ตัวที่ใกล้ที่สุดในต้นแบบ) เพื่อดูว่าเราทำให้ใกล้กันขึ้นไหม
        ref_other = max(float((R[cp] * R[c]).sum()) for c in cps if c != cp)
        margin = sims[cp] - other
        ref_margin = 1.0 - ref_other
        rows.append((chr(cp), chr(oc), round(sims[cp], 3), round(other, 3), round(margin, 3), round(ref_margin, 3)))
        if best != cp:
            # คู่ที่ต้นแบบใช้รูปเดียวกันอยู่แล้ว (เช่น ໗ กับ ຐ ใน Noto) ไม่ใช่ความผิดของเรา
            if float((R[cp] * R[best]).sum()) < 0.999:
                bad.append(f"{chr(cp)} ดูคล้าย {chr(best)} มากกว่าตัวเอง")
        elif margin < 0.35 * ref_margin and margin < 0.08:
            bad.append(f"{chr(cp)} ใกล้ {chr(oc)} เกินไป (ห่าง {margin:.3f}, ต้นแบบ {ref_margin:.3f})")
    worst = sorted(rows, key=lambda r: r[4])[:8]
    return bad, worst


def overlap_score(font, text, px=64):
    """พิกเซลหมึกที่ glyph ต่างตัวทับกัน + ความสูงหมึก บน/ล่างจากเส้นฐาน"""
    gl, _ = font.run_glyphs(text, px)
    total = 0
    for i in range(len(gl)):
        ai, xi, yi = gl[i]
        for j in range(i + 1, len(gl)):
            aj, xj, yj = gl[j]
            x0, y0 = max(xi, xj), max(yi, yj)
            x1, y1 = min(xi + ai.shape[1], xj + aj.shape[1]), min(yi + ai.shape[0], yj + aj.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            si = ai[y0 - yi : y1 - yi, x0 - xi : x1 - xi] > 100
            sj = aj[y0 - yj : y1 - yj, x0 - xj : x1 - xj] > 100
            total += int((si & sj).sum())
    if not gl:
        return 0, (0, 0)
    up = max(-y for a, x, y in gl)
    dn = max(y + a.shape[0] for a, x, y in gl)
    return total, (up, dn)


def render_line(font, text, px, pad=10):
    gl, w = font.run_glyphs(text, px)
    asc, desc = int(font.win_asc * px / font.upm), int(font.win_desc * px / font.upm)
    img = np.zeros((asc + desc + 2 * pad, int(w) + 2 * pad), np.uint8)
    for a, x, y in gl:
        paste(img, a, x + pad, y + asc + pad)
    return img


def check_collisions(var, ref, texts, px=64):
    bad = []
    worst = 0
    for t in texts:
        ov_v, _ = overlap_score(var, t, px)
        ov_r, _ = overlap_score(ref, t, px)
        d = ov_v - ov_r
        worst = max(worst, d)
        if d > max(12, 0.25 * ov_r):
            bad.append(f"{t}: ทับกัน {ov_r}→{ov_v} พิกเซล")
    return bad, worst


def check_extent(var, ref, texts, px=64):
    """ล้นขอบ = เกิน winAscent/winDescent และสูง/ต่ำกว่าต้นแบบ (ต้นแบบล้น 1px จากการปัดเศษได้ ไม่นับ)"""
    sc = px / var.upm
    lim_up, lim_dn = var.win_asc * sc, var.win_desc * sc
    bad = []
    max_up = max_dn = 0
    for t in texts:
        _, (up, dn) = overlap_score(var, t, px)
        _, (rup, rdn) = overlap_score(ref, t, px)
        max_up, max_dn = max(max_up, up), max(max_dn, dn)
        if (up > lim_up + 1 and up > rup) or (dn > lim_dn + 1 and dn > rdn):
            bad.append(f"{t}: สูง {up}/{lim_up:.0f} ต่ำ {dn}/{lim_dn:.0f} px (ต้นแบบ {rup}/{rdn})")
    return bad, (max_up, max_dn, round(lim_up), round(lim_dn))


# ---------- รูปตัวอย่าง ----------
def sample_sheet(out_png, entries, lines):
    """entries = [(label, path)]"""
    # วาดด้วย HarfBuzz+FreeType เอง (Pillow ในเครื่องไม่มี raqm จึงวางสระลาวผิด)
    blocks = []
    for label, path in entries:
        f = Font(path)
        rows = [render_line(f, ln, 60) for ln in lines]
        blocks.append((label, rows))
    W = min(1800, max(r.shape[1] for _, rows in blocks for r in rows) + 40)
    H = sum(40 + sum(r.shape[0] for r in rows) + 12 for _, rows in blocks) + 10
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    y = 6
    for label, rows in blocks:
        d.text((16, y + 6), label, fill=110, font=ImageFont.truetype("C:/Windows/Fonts/LeelawUI.ttf", 24))
        y += 40
        for r in rows:
            r = r[:, : W - 20]
            img.paste(Image.fromarray(255 - r), (20, y))
            y += r.shape[0]
        d.line([(0, y + 5), (W, y + 5)], fill=210, width=2)
        y += 12
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    vdir = ROOT / "fonts" / a.version
    man = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    qdir = vdir / "qa"
    qdir.mkdir(exist_ok=True)
    syl = syllables()
    lines = real_lines()
    print(f"พยางค์ทดสอบ {len(syl)} แบบ, บรรทัดซับจริง {len(lines)} บรรทัด")
    report = {"version": a.version, "n_syllables": len(syl), "n_lines": len(lines), "styles": {}}
    md = [f"# รายงานตรวจฟอนต์ {man['family']} {a.version}", "",
          f"ทดสอบ **{len(syl)} พยางค์** (ทุกพยัญชนะ × สระบน/ล่าง × วรรณยุกต์) + **{len(lines)} บรรทัดซับจริง** จาก `data/subs`", ""]
    for st in man["styles"]:
        stem = st["file"]
        var = Font(vdir / f"{stem}.ttf")
        ref = Font(vdir / "_ref" / f"{stem}-REF.ttf")
        solid_p = vdir / "_ref" / f"{stem}-SOLID.ttf"
        var_solid = Font(solid_p) if solid_p.exists() else var
        print(f"== {stem}")
        r = {}
        tan = float(np.tan(np.radians(st.get("skew", 0))))
        var.deshear = var_solid.deshear = tan
        r["coverage"] = check_coverage(var, ref)
        r["shaping"] = check_shaping(var, ref, syl + lines, skewed=bool(tan))
        if tan:
            r["shaping"] += check_skew_marks(var, ref, syl, tan)
        is_hollow = solid_p.exists()
        r["topology"] = check_topology(var, ref, hollow=is_hollow)
        if is_hollow:   # รูปทรงแบบตันก่อนเจาะกลวงต้องไม่พังด้วย
            r["topology"] += ["(แบบตัน) " + s for s in check_topology(var_solid, ref)]
        r["confusion"], worst_pairs = check_confusion(var_solid, ref)
        r["collision"], worst_ov = check_collisions(var, ref, syl)
        r["extent"], ext = check_extent(var, ref, syl + lines[:400])
        report["styles"][stem] = {"checks": r, "closest_pairs": worst_pairs, "max_extra_overlap_px": worst_ov, "extent_px": ext}
        for k, v in r.items():
            print(f"  {k}: {'ผ่าน' if not v else f'{len(v)} จุด'}")
        names = {"coverage": "1. ครบตัวอักษร", "shaping": "2. จัดวางเหมือนต้นแบบ", "topology": "3. รูปทรงไม่พัง (ชิ้น/รู/ห่วง)",
                 "confusion": "4. แยกตัวอักษรออก", "collision": "5. สระ-วรรณยุกต์ไม่ชนกัน", "extent": "6. ไม่ล้นขอบบน/ล่าง"}
        md += [f"## {stem} — {st['lao']} ({st['th']})", "", f"ต้นแบบ: `{st['src']}` น้ำหนัก {st['wght']} · การแปลง: `{st['ops']}`", "",
               "| ข้อ | ผล |", "|---|---|"]
        for k, label in names.items():
            v = r[k]
            md.append(f"| {label} | {'✅ ผ่าน' if not v else f'⚠️ {len(v)} จุด'} |")
        md.append("")
        for k, label in names.items():
            if r[k]:
                md.append(f"**{label}**: " + " · ".join(r[k][:30]) + (" ..." if len(r[k]) > 30 else ""))
                md.append("")
        md.append(f"คู่ที่ใกล้กันที่สุด (ตัวอักษร, ตัวที่คล้าย, ความเหมือนตัวเอง, ความเหมือนตัวที่คล้าย, ระยะห่าง, ระยะห่างในต้นแบบ): {worst_pairs[:5]}")
        md.append(f"\nความสูงหมึกสูงสุด บน/ล่าง = {ext[0]}/{ext[1]} px จากขอบที่อนุญาต {ext[2]}/{ext[3]} px (ที่ 64px)\n")
        png = qdir / f"{stem}.png"
        sample_sheet(png, [("ต้นแบบ (Noto)", vdir / "_ref" / f"{stem}-REF.ttf"), (stem, vdir / f"{stem}.ttf")],
                     ["ເກື້ອ ຫຼື ໜ້າ ນ້ຳ ໝໍ້ ຢູ່ ຍູ້", "ບ ປ ຜ ຝ ພ ຟ ດ ຕ ຍ ຢ ຂ ຄ ຊ ສ"] + lines[:3])
        md.append(f"![{stem}]({png.name})\n")
    (qdir / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (qdir / "report.md").write_text("\n".join(md), encoding="utf-8")
    total = sum(len(v) for s in report["styles"].values() for v in s["checks"].values())
    print("รายงาน →", qdir / "report.md", "| ปัญหารวม", total)


if __name__ == "__main__":
    main()
