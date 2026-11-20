# Lao Video AI

**Speech-to-text and animated captions for Lao-language video.**
A portfolio project: I built the speech-recognition models, the Lao caption fonts, and a caption editor that exports video with the captions burned in.

> Lao is a low-resource language: there is very little public speech data and almost no ready-made tools. This project tries to close that gap end to end, from raw audio to a finished captioned video.

(สรุปภาษาไทยอยู่ท้ายหน้า)

---

## What it does

1. **Transcribes Lao speech** from a video or audio clip into timed subtitles.
2. **Lets you fix the subtitles** in a browser editor. Lines the AI is unsure about are highlighted, and a word list remembers corrections.
3. **Styles the captions** with custom Lao fonts, colours, outlines, shadows, glow, boxes and 6 animation styles.
4. **Exports an MP4** with the captions rendered into the video. The on-screen preview and the exported file come from the same renderer, so what you see is what you get.

## Highlights

### 1. Lao speech recognition (ASR)
- Fine-tuned several models for Lao: Whisper (LoRA and full fine-tune), and CTC models based on wav2vec2 / OmniASR.
- Built a repeatable pipeline: data collection, cleaning, training on Kaggle GPUs, evaluation, and model merging.
- Compared against public Lao models on the same test sets (character error rate, lower is better):

| Test set | Mine (Ultra3) | Public xls-r Lao | Public xlsr-53 Lao | Public OmniASR-300M Lao |
|---|---|---|---|---|
| FLEURS (100 clips) | **16.0** | 28.0 | 27.6 | 15.9 |
| Real video clip A | **12.3** | 34.8 | 35.6 | 24.9 |
| Real video clip B | **7.8** | 21.8 | 20.9 | 11.7 |
| Real video clip C | **3.6** | 19.2 | 19.3 | 11.3 |

  Full table and notes: [`results/eval_rivals.txt`](results/eval_rivals.txt).
- **Speed**: converted to CTranslate2 int8 for a 2x or better speed-up with the same accuracy. A separate CTC model exported to ONNX int8 shrinks from 1,263 MB to 355 MB and runs about 7x faster than real time on CPU ([`results/bench_int8_ctc-omni-v2.txt`](results/bench_int8_ctc-omni-v2.txt)).
- **Honest evaluation**: I found that one public test set overlapped with my training data, so I left it out of the table above and noted why in the results files. I also tested ideas that did *not* help (longer segments, prompts, language-model fusion) and kept the results.

### 2. Lao caption fonts (`fonts/`)
- 6 video-caption styles generated **by code** (vector transformations of the open-licence Noto Sans Lao), not hand-drawn and not AI-drawn.
- Every build is checked automatically against 6 rules (2,052 syllables and 1,207 real subtitle lines render correctly, including vowel and tone marks).
- Licence: SIL Open Font License. See [`fonts/README.md`](fonts/README.md).

### 3. Caption editor and renderer (`app/`, `fonts/caprender.js`)
- Browser-based editor on a small Python server.
- One shared renderer draws captions from timing data, used for both the live preview and the MP4 export (Playwright + ffmpeg).
- Word-level timing from the AI drives karaoke-style and word-by-word pop animations.

## Repository layout

| Folder | What is inside |
|---|---|
| `app/` | Web server and editor UI (Python, HTML, JavaScript) |
| `fonts/` | Font builder outputs, QA reports, caption renderer and style editor |
| `scripts/` | Training, evaluation, model conversion and data-processing scripts |
| `results/` | Benchmark outputs, comparison tables, charts and training logs |
| `docs/` | Notes on font-style research |

## Not included here

- **Model weights and datasets.** They are large, and some data sources have licences that do not allow redistribution. The scripts show how everything was built.
- **Private subtitle files and voice recordings** from real clips.

## Tech

Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, CTranslate2, ONNX Runtime, wav2vec2 / OmniASR, Whisper, fontTools, Playwright, ffmpeg, vanilla JavaScript, Kaggle GPU training.

## Licence

Fonts: SIL OFL 1.1 (derived from Noto Sans Lao). Code: published for portfolio review; no reuse licence is granted yet.

---

## สรุปภาษาไทย

โปรเจกต์ระบบทำซับวิดีโอภาษาลาวครบวงจร: **AI ถอดเสียงลาว** (ฝึกและปรับแต่งเอง เทียบกับโมเดลสาธารณะที่ชุดทดสอบเดียวกัน) + **ฟอนต์ลาว 6 สไตล์** สร้างด้วยโค้ดและตรวจอัตโนมัติ + **ตัวแก้ซับและแต่งซับแอนิเมชัน** ที่ส่งออกเป็นวิดีโอ MP4 โดยภาพที่เห็นบนจอตรงกับไฟล์ที่ได้

ไม่ได้ใส่โมเดลกับข้อมูลเสียงไว้ในที่นี้ เพราะไฟล์ใหญ่และบางแหล่งข้อมูลมีสัญญาอนุญาตที่ห้ามแจกต่อ แต่สคริปต์ทั้งหมดอยู่ครบเพื่อดูวิธีทำ
