"""แปลง facebook/omniASR-CTC-300M (fairseq2 .pt, Apache-2.0) → Hugging Face Wav2Vec2ForCTC
โครงสร้างเหมือน XLS-R 300M (24 ชั้น, 1024, pre-LN) จึงใช้ train_ctc.py เทรนต่อได้ทันที (--base <โฟลเดอร์ที่ได้>)

  .venv\\Scripts\\python.exe scripts\\convert_omniasr.py                 → models/omniasr-ctc-300m-hf (ไม่มีหัว CTC, ให้ train_ctc สร้างหัวลาวใหม่)
  .venv\\Scripts\\python.exe scripts\\convert_omniasr.py --check FILE.wav → ลองถอดเสียงด้วยหัวเดิมของ Meta เพื่อตรวจว่าแปลงถูก
"""
import os
import sys

import torch
from common import MODELS

if os.name == "nt":  # เครื่องเรา: เก็บแคชไว้ไดรฟ์ D (บน Kaggle ใช้ค่าเริ่มต้น)
    os.environ.setdefault("HF_HOME", r"D:\lao-video-ai-archive\hf")
sys.stdout.reconfigure(encoding="utf-8")
REPO = "facebook/omniASR-CTC-300M"


def convert(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        n = None
        if k.startswith("encoder_frontend.feature_extractor.layers."):
            n = k.replace("encoder_frontend.feature_extractor.layers.", "wav2vec2.feature_extractor.conv_layers.")
        elif k.startswith("encoder_frontend.post_extract_layer_norm."):
            n = k.replace("encoder_frontend.post_extract_layer_norm.", "wav2vec2.feature_projection.layer_norm.")
        elif k.startswith("encoder_frontend.model_dim_proj."):
            n = k.replace("encoder_frontend.model_dim_proj.", "wav2vec2.feature_projection.projection.")
        elif k == "encoder_frontend.pos_encoder.conv.bias":
            n = "wav2vec2.encoder.pos_conv_embed.conv.bias"
        elif k == "encoder_frontend.pos_encoder.conv.weight_g":
            n = "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        elif k == "encoder_frontend.pos_encoder.conv.weight_v":
            n = "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        elif k.startswith("encoder.layers."):
            n = ("wav2vec2." + k).replace(".self_attn_layer_norm.", ".layer_norm.") \
                .replace(".self_attn.output_proj.", ".attention.out_proj.").replace(".self_attn.", ".attention.") \
                .replace(".ffn_layer_norm.", ".final_layer_norm.") \
                .replace(".ffn.inner_proj.", ".feed_forward.intermediate_dense.") \
                .replace(".ffn.output_proj.", ".feed_forward.output_dense.")
        elif k.startswith("encoder.layer_norm."):
            n = "wav2vec2." + k
        elif k.startswith("final_proj."):
            n = k.replace("final_proj.", "lm_head.")
        if n is None:
            raise KeyError(f"ไม่รู้จัก key: {k}")
        out[n] = v
    return out


def build(vocab_size: int):
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    cfg = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
    cfg.vocab_size = vocab_size
    return Wav2Vec2ForCTC(cfg)


def main():
    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor

    ck = torch.load(hf_hub_download(REPO, "omniASR-CTC-300M.pt"), map_location="cpu", weights_only=False)["model"]
    sd = convert(ck)
    head = sd["lm_head.weight"].shape[0]
    model = build(head)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", missing, "| unexpected:", unexpected)
    assert not unexpected and set(missing) <= {"wav2vec2.masked_spec_embed"}, "แปลงไม่ครบ"

    if "--check" in sys.argv:
        import librosa
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=hf_hub_download(REPO, "omniASR_tokenizer.model"))
        fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        x = librosa.load(sys.argv[sys.argv.index("--check") + 1], sr=16000)[0]
        with torch.no_grad():
            ids = model.eval()(fe(x, sampling_rate=16000, return_tensors="pt").input_values).logits.argmax(-1)[0].tolist()
        dedup = [i for j, i in enumerate(ids) if (j == 0 or i != ids[j - 1]) and i != 0]
        print("ผลถอดด้วยหัวเดิมของ Meta:", sp.decode(dedup))
        return

    out = MODELS / "omniasr-ctc-300m-hf"
    if out.exists():
        print(f"มีอยู่แล้ว: {out} (ไม่เขียนทับ)")
        return
    # train_ctc.py จะสร้างหัว CTC ลาวใหม่ (ignore_mismatched_sizes) แล้วตั้งค่าเริ่มจากแถวอักษรไทยของหัวเดิม (--head-init omni)
    import sentencepiece as spm

    model.save_pretrained(out)
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    fe.save_pretrained(out)
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(REPO, "omniASR_tokenizer.model"))
    torch.save({"weight": sd["lm_head.weight"], "bias": sd["lm_head.bias"],
                "pieces": [sp.id_to_piece(i) for i in range(sp.get_piece_size())]}, out / "omni_head.pt")
    print("บันทึกแล้ว:", out)


if __name__ == "__main__":
    main()
