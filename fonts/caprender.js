/* ตัววาดซับลาว (ใช้ร่วมกัน: หน้าแต่งซับ, หน้าแก้ซับ, ตอนส่งออก MP4)
 *
 * หลักการ: วาดจาก "เวลา t" อย่างเดียว ไม่มีตัวจับเวลา/CSS animation
 *   → เวลาเดียวกันได้ภาพเดียวกันเสมอ ภาพที่เห็นในหน้าเว็บ = ภาพในไฟล์ MP4
 * ขนาดทุกอย่างอิงความสูงจอ (size = พิกเซลเมื่อจอสูง 640) จึงย่อ/ขยายได้ทุกความละเอียด
 *
 * ใช้:
 *   await CapRender.loadFonts("/fonts/v2/")
 *   const key = CapRender.render(capEl, style, seg, t, stageHeightPx)
 *   seg = {start, end, text}  (ใส่ *คำ* เพื่อเน้น)
 *   key = ข้อความที่บอกสถานะภาพ — key เดิม = ภาพเดิม (ตอนส่งออกใช้เก็บภาพซ้ำ)
 */
(function () {
  const CR = {};
  CR.FONTS = [];
  CR.TOPPINGS = {
    stroke: { name: "ขอบ", def: { color: "#000000", width: 14 } },
    shadow: { name: "เงา", def: { color: "#000000", x: 4, y: 6, blur: 0 } },
    glow: { name: "เรืองแสง", def: { color: "#ff2fa8", size: 14 } },
    box: { name: "กล่อง", def: { color: "#ffd400", opacity: 100, radius: 22, mode: "word" } },
    marker: { name: "แถบปากกาเน้น", def: { color: "#ff4d6d", opacity: 85 } },
    emph: { name: "เน้นคำ", def: { color: "#ffe100", scale: 120 } },
    emoji: { name: "อีโมจิ", def: { text: "🔥", where: "end" } },
  };
  CR.ANIMS = [["none", "ไม่มี"], ["karaoke", "คาราโอเกะ"], ["pop", "เด้งทีละคำ"], ["type", "พิมพ์ทีละคำ"], ["bounce", "ตัวเด้งดึ๋ง"], ["shake", "สั่น"]];
  CR.PRESETS = {
    "ขาว ขอบดำ": { font: "LaoVideo-Mon", fill: { type: "solid", c1: "#ffffff", c2: "#ffd400" }, tops: [{ type: "stroke", color: "#000000", width: 16 }], anim: "none" },
    "เหลือง ขอบดำ": { font: "LaoVideo-Mon", fill: { type: "solid", c1: "#ffe100", c2: "#ffffff" }, tops: [{ type: "stroke", color: "#000000", width: 16 }, { type: "shadow", color: "#000000", x: 3, y: 5, blur: 0 }], anim: "none" },
    "ขอบสองชั้น": { font: "LaoVideo-Khaep", fill: { type: "solid", c1: "#ffffff", c2: "#ffffff" }, tops: [{ type: "stroke", color: "#e8492f", width: 10 }, { type: "stroke", color: "#000000", width: 12 }], anim: "pop" },
    "กล่องข่าว": { font: "LaoVideo-Num", fill: { type: "solid", c1: "#111111", c2: "#111111" }, tops: [{ type: "box", color: "#ffd400", opacity: 100, radius: 18, mode: "word" }], anim: "none" },
    "นีออน": { font: "LaoVideo-Khop", fill: { type: "solid", c1: "#ffffff", c2: "#ffffff" }, tops: [{ type: "glow", color: "#21e6ff", size: 18 }, { type: "glow", color: "#ff2fa8", size: 8 }], anim: "none" },
    "ไล่สีพาดหัว": { font: "LaoVideo-Ieng", fill: { type: "gradient", c1: "#fff27a", c2: "#ff7a00" }, tops: [{ type: "stroke", color: "#2a0a00", width: 16 }, { type: "emph", color: "#ffffff", scale: 125 }], anim: "karaoke", hl: "#ffffff" },
    "เกม/เทค": { font: "LaoVideo-Liam", fill: { type: "gradient", c1: "#7CFFCB", c2: "#4aa3ff" }, tops: [{ type: "stroke", color: "#05162b", width: 14 }, { type: "glow", color: "#4aa3ff", size: 12 }], anim: "shake" },
  };
  CR.DEFAULT = Object.assign(JSON.parse(JSON.stringify(CR.PRESETS["ขาว ขอบดำ"])), { size: 42, pos: 16, hl: "#ffe100" });

  CR.loadFonts = async function (base) {
    try {
      const m = await (await fetch(base + "manifest.json")).json();
      CR.FONTS = m.styles.map((s) => ({ id: s.id, file: s.file, lao: s.lao, th: s.th }));
    } catch (e) {
      CR.FONTS = ["Mon", "Khop", "Num", "Khaep", "Liam", "Ieng"].map((id) => ({ id, file: "LaoVideo-" + id, lao: id, th: id }));
    }
    const css = CR.FONTS.map((f) => `@font-face{font-family:"${f.file}";src:url("${base}${f.file}.woff2") format("woff2");}`).join("");
    const st = document.createElement("style");
    st.textContent = css + `
      .cr-cap{position:absolute;left:5%;right:5%;text-align:center;line-height:1.6;}
      .cr-w{position:relative;display:inline-block;white-space:pre;}
      .cr-in{position:relative;display:inline-block;}
      .cr-ly{position:absolute;left:0;top:0;pointer-events:none;}
      .cr-fill{position:relative;}`;
    document.head.appendChild(st);
    await Promise.all(CR.FONTS.map((f) => document.fonts.load(`40px "${f.file}"`, "ກ").catch(() => null)));
    return CR.FONTS;
  };

  CR.normalize = function (s) {
    const d = JSON.parse(JSON.stringify(CR.DEFAULT));
    s = Object.assign(d, s || {});
    s.fill = Object.assign({ type: "solid", c1: "#ffffff", c2: "#ffd400" }, s.fill || {});
    s.tops = Array.isArray(s.tops) ? s.tops : [];
    return s;
  };

  const hexA = (hex, a) => { const n = parseInt(String(hex).slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a / 100})`; };
  const plain = (w) => w.replace(/^\*(.+)\*$/, "$1");
  const Q = 16; // แอนิเมชันวนซ้ำ (เด้งดึ๋ง/สั่น) แบ่งเป็น 16 จังหวะ → ภาพซ้ำได้ ส่งออกเร็วขึ้น

  /** เวลาของแต่ละคำในบรรทัด: ใช้เวลาจริงจาก AI (seg.words) ถ้าข้อความยังตรงกับตอนจับเวลา
   *  ไม่งั้นแบ่งตามความยาวตัวอักษร */
  CR.wordTimes = function (seg, words) {
    if (seg.words && seg.words.length === words.length && (seg.wtext == null || seg.wtext === seg.text))
      return seg.words.map((w) => [w.start, w.end]);
    const len = words.map((w) => Math.max(1, plain(w).length));
    const total = len.reduce((a, b) => a + b, 0);
    const dur = Math.max(0.2, seg.end - seg.start);
    let acc = seg.start;
    return len.map((l) => { const s = acc; acc += (dur * l) / total; return [s, acc]; });
  };

  function popEase(p) { // ขยายเกินแล้วหดกลับ (back-out)
    const c1 = 1.9, c3 = c1 + 1;
    return 1 + c3 * Math.pow(p - 1, 3) + c1 * Math.pow(p - 1, 2);
  }

  /** วาดซับของ seg ณ เวลา t ลงใน el (div ที่อยู่ในกรอบวิดีโอ) — คืน key ของภาพ */
  CR.render = function (el, style, seg, t, stageH) {
    const S = CR.normalize(style);
    if (!seg || !seg.text || !seg.text.trim() || t < seg.start || t >= seg.end) {
      if (el._key !== "") { el.innerHTML = ""; el.style.background = "none"; el._key = ""; }
      return "";
    }
    const k = stageH / 640;
    const tops = S.tops;
    const strokes = tops.filter((x) => x.type === "stroke");
    const emph = tops.find((x) => x.type === "emph");
    const box = tops.find((x) => x.type === "box");
    const marker = tops.find((x) => x.type === "marker");
    const emoji = tops.filter((x) => x.type === "emoji");
    const filters = tops.filter((x) => x.type === "shadow" || x.type === "glow").map((x) => x.type === "shadow"
      ? `drop-shadow(${x.x / 100}em ${x.y / 100}em ${x.blur / 100}em ${x.color})`
      : `drop-shadow(0 0 ${x.size / 200}em ${x.color}) drop-shadow(0 0 ${x.size / 100}em ${x.color})`).join(" ");

    const words = seg.text.trim().split(/\s+/).filter(Boolean);
    const wt = CR.wordTimes(seg, words);
    // สถานะของแต่ละคำ ณ เวลา t (ใช้ทั้งวาดและทำ key)
    const states = words.map((w, i) => {
      const [ws] = wt[i];
      if (S.anim === "karaoke") return t >= wt[i][0] && (t < wt[i][1] || i === words.length - 1) ? "on" : "off";
      if (S.anim === "type") return t >= ws ? "v" : "h";
      if (S.anim === "pop") { const p = Math.min(1, Math.max(0, (t - ws) / 0.32)); return "p" + Math.round(p * 10); }
      if (S.anim === "bounce" || S.anim === "shake") {
        const per = S.anim === "bounce" ? 0.9 : 0.5;
        const ph = (((t - seg.start) / per + i * 0.09) % 1 + 1) % 1;
        return "q" + Math.floor(ph * Q);
      }
      return "";
    });
    const key = JSON.stringify([seg.start, seg.text, Math.round(stageH), states, S]);
    if (el._key === key) return key;
    el._key = key;

    el.className = "cr-cap";
    el.style.fontFamily = `"${S.font}"`;
    el.style.fontSize = S.size * k + "px";
    el.style.bottom = S.pos + "%";
    if (box && box.mode === "line") { el.style.background = hexA(box.color, box.opacity); el.style.borderRadius = box.radius / 100 + "em"; el.style.padding = ".1em .4em"; }
    else { el.style.background = "none"; el.style.padding = "0"; }
    el.innerHTML = "";
    const pre = emoji.filter((e) => e.where === "start").map((e) => ({ t: e.text, emoji: 1 }));
    const post = emoji.filter((e) => e.where === "end").map((e) => ({ t: e.text, emoji: 1 }));
    const items = [...pre, ...words.map((w, i) => ({ t: w, i })), ...post];
    items.forEach((it, n) => {
      let txt = it.t, isEm = false;
      const m = txt.match(/^\*(.+)\*$/); if (m) { txt = m[1]; isEm = true; }
      const w = document.createElement("span"); w.className = "cr-w";
      const st = it.emoji ? "" : states[it.i];
      if (it.emoji) w.textContent = txt;
      else {
        const inner = document.createElement("span"); inner.className = "cr-in"; w.appendChild(inner);
        let acc = 0; const layers = [];
        strokes.forEach((s) => { acc += s.width; layers.push({ color: s.color, w: acc }); });
        layers.reverse().forEach((L) => {
          const ly = document.createElement("span"); ly.className = "cr-ly"; ly.textContent = txt;
          ly.style.webkitTextStroke = `${(L.w * 2) / 100}em ${L.color}`; ly.style.color = L.color; ly.style.paintOrder = "stroke fill";
          inner.appendChild(ly);
        });
        const f = document.createElement("span"); f.className = "cr-fill"; f.textContent = txt;
        if (st === "on") f.style.color = S.hl;
        else if (isEm && emph) f.style.color = emph.color;
        else if (S.fill.type === "gradient") {
          f.style.backgroundImage = `linear-gradient(180deg, ${S.fill.c1}, ${S.fill.c2})`;
          f.style.webkitBackgroundClip = "text"; f.style.backgroundClip = "text"; f.style.color = "transparent";
        } else f.style.color = S.fill.c1;
        inner.appendChild(f);
        if (isEm && emph) w.style.fontSize = emph.scale + "%";
      }
      if (filters) w.style.filter = filters;
      if (box && box.mode === "word" && !it.emoji) { w.style.background = hexA(box.color, box.opacity); w.style.borderRadius = box.radius / 100 + "em"; w.style.padding = ".06em .22em"; w.style.margin = "0 .05em .14em"; }
      if (marker && !it.emoji) { const c = hexA(marker.color, marker.opacity); w.style.backgroundImage = `linear-gradient(transparent 58%, ${c} 58%, ${c} 92%, transparent 92%)`; }
      // แอนิเมชันคำนวณจากเวลา (ไม่มี CSS animation)
      if (st === "h") w.style.visibility = "hidden";
      if (st && st[0] === "p") {
        const p = +st.slice(1) / 10;
        if (p <= 0) w.style.opacity = "0";
        else { w.style.opacity = String(Math.min(1, p * 2.5)); w.style.transform = `scale(${0.4 + 0.6 * popEase(p)}) translateY(${(1 - p) * 0.3}em)`; }
      }
      if (st && st[0] === "q") {
        const ph = +st.slice(1) / Q;
        w.style.transform = S.anim === "bounce" ? `translateY(${-0.18 * (1 - Math.cos(2 * Math.PI * ph)) / 2}em)` : `rotate(${4 * Math.sin(2 * Math.PI * ph)}deg)`;
      }
      el.appendChild(w);
      if (n < items.length - 1) el.appendChild(document.createTextNode(" "));
    });
    // ภาษาลาวไม่เว้นวรรค: วลียาวเป็น "คำเดียว" ตัดบรรทัดไม่ได้ → ย่อตัวอักษรจนพอดีกรอบ (ต่ำสุด 55%)
    const avail = el.clientWidth;
    if (avail > 0) {
      let widest = 0;
      el.querySelectorAll(".cr-w").forEach((w) => { widest = Math.max(widest, w.offsetWidth); });
      if (widest > avail) el.style.fontSize = Math.max(0.55, (avail / widest) * 0.98) * S.size * k + "px";
    }
    return key;
  };

  /** หาบรรทัดที่ต้องแสดง ณ เวลา t */
  CR.segAt = function (segs, t) { return segs.find((s) => t >= s.start && t < s.end) || null; };

  window.CapRender = CR;
})();
