"""ดาวน์โหลดข้อมูลเทรนทั้งหมดในที่เดียว (ใช้บน Kaggle หรือเครื่องใหม่)

  python scripts/get_data.py            → data/fleurs_lo + data/siang_lao (train.tsv/test.tsv)

- FLEURS ลาว (Google, CC-BY-4.0)
- SiangLao (CC-BY-NC-4.0: ห้ามใช้เชิงการค้า)
"""
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def fleurs():
    from huggingface_hub import hf_hub_download

    out = ROOT / "data" / "fleurs_lo"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        tsv = hf_hub_download("google/fleurs", f"data/lo_la/{split}.tsv", repo_type="dataset")
        shutil.copy(tsv, out / f"{split}.tsv")
        if (out / split).is_dir() and any((out / split).glob("*.wav")):
            continue
        tgz = hf_hub_download("google/fleurs", f"data/lo_la/audio/{split}.tar.gz", repo_type="dataset")
        with tarfile.open(tgz) as t:
            t.extractall(out, filter="data")
        print(f"FLEURS {split}: {len(list((out / split).glob('*.wav')))} ไฟล์", flush=True)


def siang():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "SiangLao/lao-asr-thesis-dataset",
        repo_type="dataset",
        local_dir=str(ROOT / "data" / "siang_lao" / "raw"),
        allow_patterns=["data/*.parquet", "README.md"],
    )
    import prepare_siang

    prepare_siang.main()


if __name__ == "__main__":
    fleurs()
    if "--no-siang" not in sys.argv:
        siang()
