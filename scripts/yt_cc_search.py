"""ค้นคลิปภาษาลาวบน YouTube ที่ติดสัญญา Creative Commons (CC-BY = ใช้เชิงการค้าได้ ต้องให้เครดิต)
ยังไม่ดาวน์โหลด — แค่เก็บรายชื่อ + ตรวจสัญญาทีละคลิปจากข้อมูล YouTube เอง

  .venv\\Scripts\\python.exe scripts\\yt_cc_search.py            → data/yt_cc/candidates.tsv
  .venv\\Scripts\\python.exe scripts\\yt_cc_search.py --per 60   (ผลต่อคำค้น)

คัดเฉพาะ: license = Creative Commons, ยาว 1–60 นาที, ชื่อ/คำอธิบายมีอักษรลาว, ไม่ใช่หมวดเพลง (Music)
"""
import csv
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import yt_dlp
from common import ROOT

sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "data" / "yt_cc"
CC_FILTER = "EgIwAQ%3D%3D"  # ตัวกรอง "Creative Commons" ของหน้าค้นหา YouTube
LAO = re.compile(r"[຀-໿]")
MUSIC = re.compile(r"(?i)\b(mv|official music|karaoke|lyrics?|cover|remix|song)\b|ເພງ|ລຳ|ຄາລາໂອເກະ|ໝໍລຳ")
QUERIES = [
    "ຂ່າວ", "ຂ່າວລາວ", "ລາວ", "ພາສາລາວ", "ສອນ", "ຮຽນ", "ເລົ່າເລື່ອງ", "ນິທານ", "ສຳພາດ", "ສົນທະນາ",
    "ສຸຂະພາບ", "ກະສິກຳ", "ທ່ອງທ່ຽວລາວ", "ອາຫານລາວ", "ການສຶກສາ", "ວິທະຍາສາດ", "ປະຫວັດສາດ", "ເທດສະໜາ",
    "ທຳມະ", "ລາຍການ", "ເຕັກໂນໂລຊີ", "ທຸລະກິດ", "ກົດໝາຍ", "ແມ່ຍິງ", "ເດັກນ້ອຍ", "ສິ່ງແວດລ້ອມ", "ປ່າໄມ້",
    "ນ້ຳ", "ຊຸມຊົນ", "ບ້ານ", "ໂຮງຮຽນ", "ມະຫາວິທະຍາໄລ", "ວຽງຈັນ", "ຫຼວງພະບາງ", "ປາກເຊ", "ສະຫວັນນະເຂດ",
    "vlog ລາວ", "podcast ລາວ", "lao language", "lao interview", "laos documentary", "ແນະນຳ", "ວິທີ",
    # รอบ 2 (17 พ.ย.): หัวข้อพูดคุยทั่วไปเพิ่ม — ได้ผ่านคัดแค่ ~22% ของชั่วโมง ต้องหาเยอะ
    "ຂ່າວມື້ນີ້", "ຂ່າວເສດຖະກິດ", "ຂ່າວກິລາ", "ບານເຕະລາວ", "ຕະຫຼາດ", "ຄ້າຂາຍ", "ລົດ", "ໂທລະສັບ", "ຄອມພິວເຕີ",
    "ສອນພາສາອັງກິດ", "ສອນເຮັດອາຫານ", "ສູດອາຫານ", "ປູກຜັກ", "ລ້ຽງສັດ", "ລ້ຽງປາ", "ເຮັດນາ", "ກາເຟລາວ",
    "ປະເພນີລາວ", "ບຸນ", "ວັດ", "ພຣະ", "ຟັງທຳ", "ຄຳສອນ", "ສຸຂະພາບແມ່ ແລະ ເດັກ", "ໂຄວິດ", "ຢາ", "ທ່ານໝໍ",
    "ໄວລຸ້ນ", "ຄວາມຮັກ", "ຄອບຄົວ", "ຊີວິດ", "ແຮງບັນດານໃຈ", "ການເງິນ", "ທະນາຄານ", "ນັກສຶກສາ", "ທຶນການສຶກສາ",
    "ໄຟຟ້າ", "ເຂື່ອນ", "ຖະໜົນ", "ອຸບັດຕິເຫດ", "ໄພພິບັດ", "ນ້ຳຖ້ວມ", "ອາກາດ", "ແຂວງ", "ເມືອງ", "ອຸດົມໄຊ",
    "ຫົວພັນ", "ຈຳປາສັກ", "ຄຳມ່ວນ", "ບໍລິຄຳໄຊ", "ໄຊຍະບູລີ", "ຫຼວງນ້ຳທາ", "ຊຽງຂວາງ", "ອັດຕະປື", "ສາລະວັນ",
    "ລາຍການໂທລະທັດ", "ທອລ໌ກໂຊ", "ຕະຫຼົກ", "ເລື່ອງຜີ", "ເລື່ອງຈິງ", "ທົບທວນ", "review ລາວ", "unbox ລາວ",
    "lao news", "lao podcast", "lao vlog", "speak lao", "learn lao", "lao culture", "lao food", "laos travel",
]


def search(q: str, per: int) -> list[str]:
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}&sp={CC_FILTER}"
    opts = {"quiet": True, "extract_flat": True, "playlistend": per, "skip_download": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=False)
        return [e["id"] for e in info.get("entries") or [] if e.get("id") and len(e["id"]) == 11]
    except Exception as e:
        print(f"  ค้น '{q}' ไม่ได้: {e}", flush=True)
        return []


def meta(vid: str) -> dict | None:
    opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            i = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    except Exception:
        return None
    return {"id": vid, "title": (i.get("title") or "").replace("\t", " ").replace("\n", " "),
            "channel": i.get("channel") or i.get("uploader") or "", "license": i.get("license") or "",
            "duration": i.get("duration") or 0, "categories": ",".join(i.get("categories") or []),
            "lang": i.get("language") or "", "desc": (i.get("description") or "")[:400].replace("\t", " ").replace("\n", " ")}


def main():
    per = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 40
    OUT.mkdir(parents=True, exist_ok=True)
    ids = []
    for q in QUERIES:
        got = search(q, per)
        new = [v for v in got if v not in ids]
        ids += new
        print(f"'{q}': {len(got)} ผล ใหม่ {len(new)} | รวม {len(ids)}", flush=True)
    import json
    import time

    cache_f = OUT / "meta_cache.jsonl"  # จำข้อมูลคลิปที่ดึงแล้ว รันซ้ำไม่ต้องดึงใหม่ (YouTube บล็อกถ้าถี่เกิน)
    cache = {}
    if cache_f.exists():
        for ln in cache_f.read_text(encoding="utf-8").splitlines():
            m = json.loads(ln)
            cache[m["id"]] = m
    todo = [v for v in ids if v not in cache]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    print(f"ดึงข้อมูลคลิปใหม่ {len(todo)} (มีในแคช {len(cache)})", flush=True)

    def slow_meta(v):
        m = meta(v)
        time.sleep(1.5)
        return m

    with ThreadPoolExecutor(workers) as ex, open(cache_f, "a", encoding="utf-8") as cf:
        for m in ex.map(slow_meta, todo):
            if m:
                cache[m["id"]] = m
                cf.write(json.dumps(m, ensure_ascii=False) + "\n")
                cf.flush()
    metas = [cache[v] for v in ids if v in cache]
    keep, why = [], {}
    for m in metas:
        reason = None
        if "creative commons" not in m["license"].lower():
            reason = "ไม่ใช่ CC"
        elif not 60 <= m["duration"] <= 3600:
            reason = "ความยาว"
        elif not LAO.search(m["title"] + m["desc"]):
            reason = "ไม่มีอักษรลาว"
        elif "Music" in m["categories"] or MUSIC.search(m["title"]):
            reason = "เพลง"
        if reason:
            why[reason] = why.get(reason, 0) + 1
        else:
            keep.append(m)
    with open(OUT / "candidates.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metas[0]) if metas else ["id"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(keep)
    hrs = sum(m["duration"] for m in keep) / 3600
    print(f"✅ ผ่าน {len(keep)} คลิป = {hrs:.1f} ชม. | ตัดออก {why} | {OUT / 'candidates.tsv'}")


if __name__ == "__main__":
    main()
