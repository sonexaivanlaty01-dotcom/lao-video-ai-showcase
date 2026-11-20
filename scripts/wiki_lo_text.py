"""แยกข้อความภาษาลาวจาก dump วิกิพีเดียลาว (CC-BY-SA 4.0) → data/lm/sell_wiki_lo.txt ไว้สร้าง char LM (scripts/ctc_lm.py)
เครดิต: "Wikipedia ภาษาลาว (lo.wikipedia.org) ผู้เขียนวิกิพีเดีย, CC BY-SA 4.0"

  .venv\\Scripts\\python.exe scripts\\wiki_lo_text.py D:\\lao-video-ai-archive\\wiki\\lowiki-latest-pages-articles.xml.bz2
"""
import bz2
import re
import sys
import xml.etree.ElementTree as ET

from common import ROOT

sys.stdout.reconfigure(encoding="utf-8")
LAO = re.compile(r"[\u0e80-\u0eff]")
NON_LAO = re.compile(r"[^\u0e80-\u0eff0-9 ]")


def strip_wiki(t: str) -> str:
    for _ in range(3):
        t = re.sub(r"\{\{[^{}]*\}\}", " ", t)  # แม่แบบ
    t = re.sub(r"\{\|.*?\|\}", " ", t, flags=re.S)  # ตาราง
    t = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)  # [[ลิงก์|ข้อความ]]
    t = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}|={2,}", " ", t)
    return t


def sentences(t: str):
    for line in t.splitlines():
        line = line.strip()
        if not line or line[0] in "*#:;!|{" or line.lower().startswith(("file:", "category:", "ໝວດ")):
            continue
        for s in re.split(r"[.!?។\u0eaf]\s+|\s{2,}", line):
            s = " ".join(s.split())
            lao = len(LAO.findall(s))
            if 12 <= len(s) <= 300 and lao >= 0.8 * len(s.replace(" ", "")) and not re.search(r"[A-Za-z]{3,}", s):
                yield s


def main():
    src = sys.argv[1]
    out, n_pages = [], 0
    ns = None
    for ev, el in ET.iterparse(bz2.open(src), events=("end",)):
        tag = el.tag.split("}")[-1]
        if tag == "page":
            nsv = el.find("{*}ns")
            txt = el.find("{*}revision/{*}text")
            if nsv is not None and nsv.text == "0" and txt is not None and txt.text and not txt.text.lower().startswith("#redirect"):
                out += list(sentences(strip_wiki(txt.text)))
                n_pages += 1
            el.clear()
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    dst = ROOT / "data" / "lm" / "sell_wiki_lo.txt"
    dst.write_text("\n".join(uniq) + "\n", encoding="utf-8")
    print(f"✅ วิกิพีเดียลาว {n_pages:,} บทความ → {len(uniq):,} ประโยค {sum(map(len, uniq)):,} ตัวอักษร → {dst}")


if __name__ == "__main__":
    main()
