/* แผงแต่งสไตล์ซับ (ฟอนต์ / สี / ท็อปปิ้ง / แอนิเมชัน / JSON) — ใช้ร่วมกันทุกหน้า
 * ต้องโหลด caprender.js และเรียก CapRender.loadFonts() ก่อน
 *
 *   const ed = CapEditor.mount(divElement, style, (newStyle) => { ...บันทึก/วาดใหม่... })
 *   ed.set(style)   // เปลี่ยนสไตล์จากภายนอก (เช่น เปิดคลิปใหม่)
 */
(function () {
  const CSS = `
  .ce{--ce-line:var(--line,#deded9);--ce-soft:var(--soft,var(--bg,#f0efeb));--ce-acc:var(--accent,#e8492f);--ce-muted:var(--muted,#6b6b70);font-size:14px}
  .ce h3{font-size:13px;margin:12px 0 6px;color:var(--ce-muted);font-weight:600}
  .ce .ce-chips{display:flex;flex-wrap:wrap;gap:6px}
  .ce .ce-chip{border:1px solid var(--ce-line);background:var(--ce-soft);border-radius:999px;padding:3px 10px;cursor:pointer;font:inherit;font-size:13px;color:inherit}
  .ce .ce-chip[aria-pressed=true]{background:var(--ce-acc);border-color:var(--ce-acc);color:#fff}
  .ce .ce-font{font-size:18px;padding:2px 11px}
  .ce .ce-row{display:flex;align-items:center;gap:8px;margin:6px 0}
  .ce .ce-row>label{min-width:70px;color:var(--ce-muted);font-size:13px}
  .ce .ce-row input[type=range]{flex:1;min-width:0}
  .ce input[type=color]{width:34px;height:26px;padding:0;border:1px solid var(--ce-line);border-radius:6px;background:none;cursor:pointer}
  .ce .ce-tp{border:1px solid var(--ce-line);border-radius:10px;padding:6px 10px;margin:6px 0;background:var(--ce-soft)}
  .ce .ce-tp header{display:flex;align-items:center;gap:6px;font-weight:600;padding:0;border:0;background:none}
  .ce .ce-tp header span{flex:1}
  .ce .ce-ib{border:1px solid var(--ce-line);background:transparent;border-radius:6px;width:26px;height:24px;cursor:pointer;line-height:1;padding:0;color:inherit}
  .ce .ce-tools{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
  .ce .ce-tool{border:1px dashed var(--ce-line);background:var(--ce-soft);border-radius:8px;padding:4px 9px;cursor:pointer;font:inherit;font-size:13px;color:inherit}
  .ce .ce-note{font-size:12px;color:var(--ce-muted);margin:4px 0 0}
  .ce pre{white-space:pre-wrap;word-break:break-all;font-size:11px;max-height:120px;overflow:auto;background:var(--ce-soft);border-radius:8px;padding:8px;margin:6px 0 0}
  .ce .ce-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:4px 18px}`;

  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k.startsWith("on")) e[k] = v; else e.setAttribute(k, v); }
    kids.forEach((c) => e.append(c));
    return e;
  }

  function mount(root, style, onChange) {
    if (!document.getElementById("ce-css")) document.head.append(el("style", { id: "ce-css" }, CSS));
    const CR = window.CapRender;
    let S = CR.normalize(style);
    root.classList.add("ce");
    root.innerHTML = "";
    const colA = el("div"), colB = el("div");
    root.append(el("div", { class: "ce-cols" }, colA, colB));

    const emit = (rebuild) => { if (rebuild) build(); else marks(); onChange(JSON.parse(JSON.stringify(S))); };
    const chip = (label, pressed, fn, extra = "") => el("button", { class: "ce-chip " + extra, "aria-pressed": String(!!pressed), type: "button", onclick: fn }, label);
    const range = (obj, key, min, max) => { const i = el("input", { type: "range", min, max }); i.value = obj[key]; i.oninput = () => { obj[key] = +i.value; emit(false); }; return i; };
    const color = (obj, key) => { const i = el("input", { type: "color" }); i.value = obj[key]; i.oninput = () => { obj[key] = i.value; emit(false); }; return i; };
    const row = (label, ...kids) => el("div", { class: "ce-row" }, el("label", {}, label), ...kids);

    let fontsBox, fillBox, animBox, c2wrap, hlRow, json;
    function marks() {
      fontsBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === S.font)));
      fillBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === S.fill.type)));
      animBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === S.anim)));
      c2wrap.style.display = S.fill.type === "gradient" ? "" : "none";
      hlRow.style.display = S.anim === "karaoke" ? "" : "none";
      json.textContent = JSON.stringify(S);
    }

    function topCard(t) {
      const i = S.tops.indexOf(t);
      const move = (d) => { const j = i + d; if (j < 0 || j >= S.tops.length) return; [S.tops[i], S.tops[j]] = [S.tops[j], S.tops[i]]; emit(true); };
      const card = el("div", { class: "ce-tp" },
        el("header", {}, el("span", {}, CR.TOPPINGS[t.type].name),
          el("button", { class: "ce-ib", type: "button", title: "ขึ้น (ใกล้ตัวอักษร)", onclick: () => move(1) }, "↑"),
          el("button", { class: "ce-ib", type: "button", title: "ลง", onclick: () => move(-1) }, "↓"),
          el("button", { class: "ce-ib", type: "button", title: "ลบ", onclick: () => { S.tops.splice(i, 1); emit(true); } }, "✕")));
      const opts = (key, list) => el("div", { class: "ce-chips" }, ...list.map(([v, l]) => chip(l, t[key] === v, () => { t[key] = v; emit(true); })));
      if (t.type === "stroke") card.append(row("สี", color(t, "color")), row("หนา", range(t, "width", 2, 30)));
      if (t.type === "shadow") card.append(row("สี", color(t, "color")), row("ซ้าย-ขวา", range(t, "x", -15, 15)), row("บน-ล่าง", range(t, "y", -15, 15)), row("เบลอ", range(t, "blur", 0, 20)));
      if (t.type === "glow") card.append(row("สี", color(t, "color")), row("ขนาด", range(t, "size", 2, 40)));
      if (t.type === "box") card.append(row("สี", color(t, "color")), row("ทึบ", range(t, "opacity", 10, 100)), row("มุมมน", range(t, "radius", 0, 50)), row("แบบ", opts("mode", [["word", "ทีละคำ"], ["line", "ทั้งบรรทัด"]])));
      if (t.type === "marker") card.append(row("สี", color(t, "color")), row("ทึบ", range(t, "opacity", 20, 100)));
      if (t.type === "emph") card.append(row("สี", color(t, "color")), row("ขยาย %", range(t, "scale", 100, 160)), el("p", { class: "ce-note" }, "ใส่ *คำ* ในซับ = คำเน้น"));
      if (t.type === "emoji") {
        const inp = el("input", { type: "text", size: "6" }); inp.value = t.text; inp.style.fontSize = "18px";
        inp.oninput = () => { t.text = inp.value; emit(false); };
        card.append(row("อีโมจิ", inp), row("ตำแหน่ง", opts("where", [["start", "หน้า"], ["end", "ท้าย"]])));
      }
      return card;
    }

    function build() {
      colA.innerHTML = ""; colB.innerHTML = "";
      colA.append(el("h3", {}, "สไตล์เริ่มต้น (กดแล้วแก้ต่อได้)"),
        el("div", { class: "ce-chips" }, ...Object.keys(CR.PRESETS).map((k) => chip(k, false, () => {
          const p = JSON.parse(JSON.stringify(CR.PRESETS[k])); S = CR.normalize(Object.assign({}, S, p, { hl: p.hl || S.hl })); emit(true);
        }))));
      fontsBox = el("div", { class: "ce-chips" }, ...CR.FONTS.map((f) => { const b = chip(f.lao, false, () => { S.font = f.file; emit(false); }, "ce-font"); b.dataset.v = f.file; b.title = f.th; b.style.fontFamily = `"${f.file}"`; return b; }));
      fillBox = el("div", { class: "ce-chips" }, ...[["solid", "สีเดียว"], ["gradient", "ไล่สี"]].map(([v, l]) => { const b = chip(l, false, () => { S.fill.type = v; emit(false); }); b.dataset.v = v; return b; }));
      c2wrap = el("span", {}, color(S.fill, "c2"), el("span", { class: "ce-note" }, " สีที่ 2"));
      colA.append(el("h3", {}, "ฟอนต์"), fontsBox,
        el("h3", {}, "สีตัวอักษร"), row("แบบ", fillBox), row("สี", color(S.fill, "c1"), c2wrap),
        row("ขนาด", range(S, "size", 20, 80)), row("ตำแหน่ง", range(S, "pos", 4, 80)));
      animBox = el("div", { class: "ce-chips" }, ...CR.ANIMS.map(([v, l]) => { const b = chip(l, false, () => { S.anim = v; emit(false); }); b.dataset.v = v; return b; }));
      hlRow = row("สีคำที่พูด", color(S, "hl"));
      json = el("pre");
      const copy = el("button", { class: "ce-tool", type: "button" }, "คัดลอก JSON");
      copy.onclick = async () => { try { await navigator.clipboard.writeText(JSON.stringify(S)); copy.textContent = "คัดลอกแล้ว ✓"; } catch { copy.textContent = "คัดลอกไม่ได้"; } setTimeout(() => (copy.textContent = "คัดลอก JSON"), 1500); };
      const paste = el("button", { class: "ce-tool", type: "button", onclick: () => { const v = prompt("วาง JSON ของสไตล์"); if (!v) return; try { S = CR.normalize(JSON.parse(v)); emit(true); } catch { alert("JSON ไม่ถูกต้อง"); } } }, "วาง JSON");
      colB.append(el("h3", {}, "ท็อปปิ้ง (ซ้อนได้ · ชั้นบน = ใกล้ตัวอักษร)"),
        ...(S.tops.length ? [...S.tops].reverse().map(topCard) : [el("p", { class: "ce-note" }, "ยังไม่มีท็อปปิ้ง — กด + ด้านล่าง")]),
        el("div", { class: "ce-tools" }, ...Object.entries(CR.TOPPINGS).map(([k, v]) => el("button", { class: "ce-tool", type: "button", onclick: () => { S.tops.push(Object.assign({ type: k }, JSON.parse(JSON.stringify(v.def)))); emit(true); } }, "+ " + v.name))),
        el("h3", {}, "แอนิเมชัน"), animBox, hlRow,
        el("h3", {}, "บันทึก / แชร์สไตล์"), el("div", { class: "ce-tools" }, copy, paste), json);
      marks();
    }
    build();
    return { set(s) { S = CR.normalize(s); build(); }, get: () => JSON.parse(JSON.stringify(S)) };
  }

  window.CapEditor = { mount };
})();
