"""สร้างฟอนต์ลาวสไตล์ของเราจาก Noto Sans Lao (OFL 1.1) ด้วยการแปลงเส้นเวกเตอร์ (ไม่ใช้ AI วาดรูป)

หลักการ: ทุกการแปลงมีขนาดรวมใกล้เคียงของเดิม (มนมุม = หดแล้วขยายเท่ากัน, ตัวกลวง = ขอบนอกเดิม)
→ ตาราง GSUB/GPOS/GDEF (การวางสระ-วรรณยุกต์ซ้อน) ใช้ของต้นแบบได้ทั้งชุด ไม่ต้องแก้

ใช้:  .venv-fonts\\Scripts\\python.exe scripts\\fonts\\build_fonts.py --version v1
ผลลัพธ์: fonts/<version>/<ชื่อ>.ttf + .woff2  (เวอร์ชันใหม่ = โฟลเดอร์ใหม่ ไม่เขียนทับ)
"""
import argparse, json, shutil, sys
from pathlib import Path

import pathops
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.basePen import BasePen
from fontTools.pens.transformPen import TransformPen
import math

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "fonts" / "base"

# ---- สไตล์ (แก้ค่าตรงนี้แล้ว build เป็นเวอร์ชันใหม่) ----
STYLES = [
    {
        "id": "Mon", "lao": "ມົນ", "th": "หนาโค้งมน",
        "src": "NotoSansLaoLooped-VF.ttf", "wght": 800,
        "ops": [("open", 44)],              # มนมุมนูน
    },
    {
        "id": "Khop", "lao": "ຂອບ", "th": "ตัวกลวงมีขอบ (หัวข้อ/ป้าย)",
        "src": "NotoSansLaoLooped-VF.ttf", "wght": 900,
        "ops": [("open", 20), ("hollow", 46)],  # มนนิดหน่อย แล้วเจาะกลวงเหลือขอบ 46 หน่วย
    },
    {
        "id": "Num", "lao": "ນຸ່ມ", "th": "นุ่มน่ารัก (ไม่มีหัว)",
        "src": "NotoSansLao-VF.ttf", "wght": 760,
        "ops": [("open", 48), ("close", 22)],  # มนทั้งมุมนูนและมุมเว้า
    },
    # ---- v2: แนวคิดจากกลุ่มสไตล์ฟอนต์ไทย (ดู docs/ศึกษาสไตล์ฟอนต์ไทย.md) ----
    {
        "id": "Khaep", "lao": "ແຄບ", "th": "แคบสูง พาดหัว",
        "src": "NotoSansLaoLooped-VF.ttf", "wght": 850, "wdth": 75,
        "ops": [("open", 20)],
    },
    {
        "id": "Liam", "lao": "ຫຼ່ຽມ", "th": "เหลี่ยม เทค/เกม",
        "src": "NotoSansLao-VF.ttf", "wght": 720,
        "ops": [("angular", 16)],            # โค้งทุกเส้น → หลายเหลี่ยม (คลาดจากเส้นเดิมไม่เกิน 16/1000 em)
    },
    {
        "id": "Ieng", "lao": "ອຽງ", "th": "เอียงไดนามิก",
        "src": "NotoSansLaoLooped-VF.ttf", "wght": 780, "skew": 11,
        "ops": [("open", 30)],
    },
]
FAMILY = "LaoVideo"


def copy_path(p):
    q = pathops.Path()
    p.draw(q.getPen())
    return q


def stroked(p, width, join):
    s = copy_path(p)
    s.stroke(width, pathops.LineCap.ROUND_CAP, join, 4.0)
    s.convertConicsToQuads(0.25)   # วงกลมจาก stroke เป็น conic → แปลงเป็น quad (TrueType)
    return s


def offset(p, d, join=pathops.LineJoin.ROUND_JOIN):
    """d>0 ขยาย, d<0 หด (ด้วยการ union/difference กับเส้นขอบหนา 2|d|)"""
    if d == 0:
        return p
    kind = pathops.PathOp.UNION if d > 0 else pathops.PathOp.DIFFERENCE
    # pathops บางครั้งคำนวณไม่ผ่านเพราะเลขทศนิยม → ลองขยับรัศมีนิดเดียว (≤0.4%) แล้วทำใหม่
    for k in (1.0, 1.002, 0.998, 1.004):
        try:
            ring = stroked(p, 2 * abs(d) * k, join)
            return pathops.op(p, ring, kind, clockwise=True)
        except pathops.PathOpsError:
            p = pathops.simplify(p, clockwise=True)
    raise pathops.PathOpsError(f"offset {d} ไม่สำเร็จ")


def holes(p):
    return sum(1 for c in p.contours if not c.clockwise)


def round_convex(p, r):
    """มนมุมนูน: หด(มุมแหลม)→ขยาย(มน) แต่ส่วนที่บางกว่า 2r (เช่นห่วงหัวตัวอักษรลาว) จะหายไปด้วย
    → เอาชิ้นที่หายไปและใหญ่กว่าเศษมุมกลับคืนมา ห่วง/เส้นบางจึงอยู่ครบ"""
    o = offset(offset(p, -r, pathops.LineJoin.MITER_JOIN), r)
    # ส่วนบางที่อยู่ไกลจากเนื้อหลัก (ห่วง/หาง) → ขยายกลับเข้าไปในรูปเดิมให้ต่อกับเนื้อหลักเหมือนเดิม
    # เศษมุมอยู่ใกล้เนื้อหลักเสมอ (< r) จึงไม่ถูกเอากลับ = มุมยังมน
    far = pathops.op(p, offset(o, 1.05 * r), pathops.PathOp.DIFFERENCE, clockwise=True)
    if list(far.contours):
        back = pathops.op(offset(far, 1.15 * r), p, pathops.PathOp.INTERSECTION, clockwise=True)
        o = pathops.op(o, back, pathops.PathOp.UNION, clockwise=True)
    return o


def round_concave(p, r):
    """มนมุมเว้า: ขยาย→หด แต่ห้ามถมรู/ห่วง → เติมเฉพาะชิ้นเล็กที่ไม่ทำให้จำนวนรูเปลี่ยน"""
    c_all = offset(offset(p, r), -r)
    added = pathops.op(c_all, p, pathops.PathOp.DIFFERENCE, clockwise=True)
    limit = 3.0 * r * r
    h0 = holes(p)
    for c in added.contours:
        if c.clockwise and c.area < limit:
            cand = pathops.op(p, c, pathops.PathOp.UNION, clockwise=True)
            if holes(cand) == h0:
                p = cand
    return p


def shape_sig(p):
    """(จำนวนชิ้น, จำนวนรู) — ใช้ยืนยันว่าการมนมุมไม่ทำให้ชิ้นแยก/ติดกัน หรือรูหาย"""
    cs = list(p.contours)
    return sum(1 for c in cs if c.clockwise), sum(1 for c in cs if not c.clockwise)


def safe_round(fn, p, r):
    """ถ้าการมนทำให้รูปทรงเปลี่ยน (เช่น จุดของ ຶ ติดตัว, เลข ໘ ขาด) → ลดรัศมีลง ถ้ายังไม่ได้ก็คงรูปเดิม"""
    sig = shape_sig(p)
    for k in (1.0, 0.6, 0.35):
        q = fn(p, r * k)
        if shape_sig(q) == sig:
            return q
    return p


class AngularPen(BasePen):
    """แปลงเส้นโค้งเป็นเส้นตรง 3 ท่อน (ตัดมุม) แบบตัวเหลี่ยม: p0 → a → b → p1
    k = ระยะเข้าหาจุดควบคุม (0 = ตัดตรง, 1 = มุมแหลมเต็ม)"""

    def __init__(self, out, k):
        super().__init__(None)
        self.out, self.k = out, k

    def _moveTo(self, p):
        self.out.moveTo(p)

    def _lineTo(self, p):
        self.out.lineTo(p)

    def _corner(self, p0, c0, c1, p1):
        k = self.k
        a = (p0[0] + k * (c0[0] - p0[0]), p0[1] + k * (c0[1] - p0[1]))
        b = (p1[0] + k * (c1[0] - p1[0]), p1[1] + k * (c1[1] - p1[1]))
        self.out.lineTo(a); self.out.lineTo(b); self.out.lineTo(p1)

    def _qCurveToOne(self, c, p1):
        self._corner(self._getCurrentPoint(), c, c, p1)

    def _curveToOne(self, c1, c2, p1):
        self._corner(self._getCurrentPoint(), c1, c2, p1)

    def _closePath(self):
        self.out.closePath()

    def _endPath(self):
        self.out.endPath()


class FlattenPen(BasePen):
    """เก็บเส้นขอบเป็นจุดถี่ ๆ ต่อ contour (โค้ง → 10 จุด)"""

    def __init__(self):
        super().__init__(None)
        self.contours, self.cur = [], []

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _qCurveToOne(self, c, p1):
        p0 = self._getCurrentPoint()
        for i in range(1, 11):
            t = i / 10
            self.cur.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
                             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]))

    def _curveToOne(self, c1, c2, p1):
        p0 = self._getCurrentPoint()
        for i in range(1, 11):
            t = i / 10; u = 1 - t
            self.cur.append((u**3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t**3 * p1[0],
                             u**3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t**3 * p1[1]))

    def _closePath(self):
        if len(self.cur) > 2:
            self.contours.append(self.cur)
        self.cur = []

    _endPath = _closePath


def rdp(pts, eps):
    """Ramer–Douglas–Peucker: ลดจุดให้เหลือเฉพาะมุมสำคัญ → เส้นตรงเป็นเหลี่ยม"""
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy) or 1e-9
    dmax, idx = 0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * pts[i][0] - dx * pts[i][1] + x1 * y0 - y1 * x0) / n
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(pts[: idx + 1], eps)[:-1] + rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]


def angularize(p, eps):
    """ตัวเหลี่ยม: เส้นโค้งทั้งหมด → รูปหลายเหลี่ยมที่ลดจุดเหลือ eps หน่วย (ยิ่งมากยิ่งเหลี่ยม)"""
    fp = FlattenPen()
    p.draw(fp)
    q = pathops.Path()
    pen = q.getPen()
    for c in fp.contours:
        # เริ่มที่จุดที่อยู่ซ้ายสุด เพื่อให้ RDP แบ่งครึ่ง contour ได้สมดุล
        k = min(range(len(c)), key=lambda i: c[i])
        ring = c[k:] + c[:k] + [c[k]]
        far = max(range(len(ring)), key=lambda i: (ring[i][0] - ring[0][0]) ** 2 + (ring[i][1] - ring[0][1]) ** 2)
        poly = rdp(ring[: far + 1], eps)[:-1] + rdp(ring[far:], eps)[:-1]
        if len(poly) < 3:
            continue
        pen.moveTo(poly[0])
        for pt in poly[1:]:
            pen.lineTo(pt)
        pen.closePath()
    return pathops.simplify(q, clockwise=True)


def skew_anchors(font, tan):
    """เอียงตัวอักษรแล้วต้องเอียงจุดยึดสระ/วรรณยุกต์ด้วย (x += tan·y) ไม่งั้นสระจะลอยผิดที่"""
    seen = set()

    def fix(a):
        if a is not None and id(a) not in seen:
            seen.add(id(a))
            a.XCoordinate = round(a.XCoordinate + tan * a.YCoordinate)

    for lk in font["GPOS"].table.LookupList.Lookup:
        for st in lk.SubTable:
            if lk.LookupType == 9:
                st = st.ExtSubTable
            t = st.LookupType if hasattr(st, "LookupType") else lk.LookupType
            if t in (4, 5, 6):
                arr = st.Mark1Array if t == 6 else st.MarkArray
                for m in arr.MarkRecord:
                    fix(m.MarkAnchor)
            if t == 4:
                for b in st.BaseArray.BaseRecord:
                    for a in b.BaseAnchor:
                        fix(a)
            elif t == 5:
                for lig in st.LigatureArray.LigatureAttach:
                    for comp in lig.ComponentRecord:
                        for a in comp.LigatureAnchor:
                            fix(a)
            elif t == 6:
                for m2 in st.Mark2Array.Mark2Record:
                    for a in m2.Mark2Anchor:
                        fix(a)
            elif t == 3:
                for e in st.EntryExitRecord:
                    fix(e.EntryAnchor); fix(e.ExitAnchor)


def apply_ops(p, ops):
    p = pathops.simplify(p, clockwise=True)
    for name, v in ops:
        if name == "angular":
            p = angularize(p, v)
        elif name == "open":     # มนมุมนูน ขนาดเท่าเดิม
            p = safe_round(round_convex, p, v)
        elif name == "close":  # มนมุมเว้า
            p = safe_round(round_concave, p, v)
        elif name == "hollow":
            inner = offset(p, -v, pathops.LineJoin.ROUND_JOIN)
            p = pathops.op(p, inner, pathops.PathOp.DIFFERENCE, clockwise=True)
        else:
            raise ValueError(name)
    return p


def set_names(font, family, style_name, version):
    name = font["name"]
    full = f"{family} {style_name}"
    ps = f"{family}-{style_name}".replace(" ", "")
    # ลบชื่อเก่าทั้งหมดที่อ้างถึง Noto (OFL: ฟอนต์ดัดแปลงต้องเปลี่ยนชื่อ)
    keep = {0, 13, 14}  # copyright, license, license URL
    name.names = [n for n in name.names if n.nameID in keep]
    for nid, val in {
        1: full, 2: "Regular", 3: f"{version};{ps}", 4: full, 5: f"Version {version}",
        6: ps, 16: family, 17: style_name,
    }.items():
        name.setName(val, nid, 3, 1, 0x409)
        name.setName(val, nid, 1, 0, 0)
    cr = name.getDebugName(0) or ""
    name.setName(cr + " Modified 2026 by the LaoVideo project.", 0, 3, 1, 0x409)
    font["OS/2"].usWeightClass = 400
    font["OS/2"].fsSelection = (font["OS/2"].fsSelection & ~0b1100001) | 0b1000000  # REGULAR
    font["head"].macStyle = 0


def build_style(st, version, out_dir):
    src = TTFont(BASE / st["src"])
    wdth = st.get("wdth", 100)
    tan = math.tan(math.radians(st.get("skew", 0)))
    font = instancer.instantiateVariableFont(src, {"wght": st["wght"], "wdth": wdth}, updateFontNames=False)
    gs = font.getGlyphSet()
    glyf = font["glyf"]
    order = font.getGlyphOrder()
    stats = {"glyphs": 0, "changed": 0, "empty": 0}
    new = {}
    for gname in order:
        pen_path = pathops.Path()
        gs[gname].draw(pen_path.getPen(glyphSet=gs))     # แตก component ออกเป็นเส้นจริง
        if not list(pen_path.contours):
            stats["empty"] += 1
            new[gname] = None
            continue
        out = apply_ops(pen_path, st["ops"])
        tt = TTGlyphPen(None)
        pen = Cu2QuPen(tt, max_err=1.0, reverse_direction=False)
        out.draw(TransformPen(pen, (1, 0, tan, 1, 0, 0)) if tan else pen)   # เอียง: x' = x + tan·y
        new[gname] = tt.glyph()
        stats["changed"] += 1
    for gname, g in new.items():
        if g is None:
            continue
        glyf[gname] = g
    stats["glyphs"] = len(order)
    # อัปเดต lsb ให้ตรงกับรูปใหม่
    hmtx = font["hmtx"]
    for gname in order:
        g = glyf[gname]
        g.recalcBounds(glyf)
        adv, _ = hmtx[gname]
        hmtx[gname] = (adv, getattr(g, "xMin", 0) if g.numberOfContours else 0)
    for t in ("prep", "fpgm", "cvt ", "gasp", "hdmx", "LTSH", "VDMX", "DSIG"):
        if t in font:
            del font[t]
    font["maxp"].maxZones = 1
    if tan:
        skew_anchors(font, tan)
        font["post"].italicAngle = -st["skew"]
        font["hhea"].caretSlopeRise, font["hhea"].caretSlopeRun = 1000, round(1000 * tan)
    set_names(font, FAMILY, st["id"], version)
    stem = f"{FAMILY}-{st['id']}"
    ttf = out_dir / f"{stem}.ttf"
    font.save(ttf)
    font.flavor = "woff2"
    font.save(out_dir / f"{stem}.woff2")
    # ต้นแบบ static น้ำหนักเดียวกัน ไว้ให้สคริปต์ตรวจเทียบ
    ref = instancer.instantiateVariableFont(TTFont(BASE / st["src"]), {"wght": st["wght"], "wdth": wdth})
    ref.save(out_dir / "_ref" / f"{stem}-REF.ttf")
    # สไตล์กลวง: เก็บแบบตันไว้ตรวจว่าอ่านแยกตัวอักษรได้ (ตรวจรูปทรง ไม่ใช่ลายขอบ)
    if any(op == "hollow" for op, _ in st["ops"]):
        solid_st = {**st, "id": st["id"] + "SOLID", "ops": [o for o in st["ops"] if o[0] != "hollow"]}
        tmp = out_dir / "_ref" / "_solid"
        tmp.mkdir(exist_ok=True)
        s_stem, _ = build_style_solid(solid_st, version, tmp)
        shutil.move(str(tmp / f"{s_stem}.ttf"), out_dir / "_ref" / f"{stem}-SOLID.ttf")
        shutil.rmtree(tmp)
    return stem, stats


def build_style_solid(st, version, out_dir):
    (out_dir / "_ref").mkdir(exist_ok=True)
    stem, stats = build_style(st, version, out_dir)
    return stem, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="เช่น v1 → fonts/v1/")
    ap.add_argument("--force", action="store_true", help="ยอมเขียนทับโฟลเดอร์เวอร์ชันนี้ (ห้ามใช้กับเวอร์ชันที่ส่งออกแล้ว)")
    a = ap.parse_args()
    out_dir = ROOT / "fonts" / a.version
    if out_dir.exists() and not a.force:
        sys.exit(f"มี {out_dir} อยู่แล้ว → ใช้เลขเวอร์ชันใหม่")
    (out_dir / "_ref").mkdir(parents=True, exist_ok=True)
    shutil.copy(BASE / "OFL-NotoSansLao.txt", out_dir / "OFL.txt")
    manifest = {"version": a.version, "family": FAMILY, "license": "SIL OFL 1.1 (ดัดแปลงจาก Noto Sans Lao / Noto Sans Lao Looped)", "styles": []}
    for st in STYLES:
        stem, stats = build_style(st, a.version, out_dir)
        print(f"{stem}: {stats}")
        manifest["styles"].append({**{k: st[k] for k in ("id", "lao", "th", "src", "wght")}, "wdth": st.get("wdth", 100), "skew": st.get("skew", 0), "ops": st["ops"], "file": stem, **stats})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("เสร็จ →", out_dir)


if __name__ == "__main__":
    main()
