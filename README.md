# VTS-Eval — Video-to-Speech / Visual Dubbing Evaluation Toolkit

End-to-end evaluation pipeline for video-grounded speech generation: WER, speaker similarity, MCD, UTMOS, lip-sync, and audio-visual sync. Originally built for the **[CineDub](https://cinedub2026.github.io/)** project, packaged here as a standalone benchmark.

> **See also**: the [`benchmark/`](../benchmark_fix/) sibling directory ships the **CineDub-Multi** and **CineDub-SA** benchmark JSONLs (and demo clips) that this toolkit is designed to evaluate against.

## Supported Metrics

| Metric | Output Key | Description | Conda Env |
|---|---|---|---|
| `wer_whisper` | `WER`, `CER` | Whisper-large-v3 ASR-based WER/CER | `vts_benchmark` |
| `wer_qwen3` | `WER`, `CER` | Qwen3-ASR-based WER/CER (robust to speech+audio mixtures) | `qwen3-asr` |
| `sim_wavlm` | `SIM_WAVLM` | WavLM-base-sv speaker similarity (vs GT) | `vts_benchmark` |
| `mcd` | `MCD_DTW`, `MCD_DTW_SL` | Mel cepstral distortion (DTW + DTW-SL) | `vts_benchmark` |
| `utmos` | `UTMOS` | UTMOS naturalness score | `vts_benchmark` |
| `syncnet` | `LSE_D`, `LSE_C` | SyncNet lip-sync error (LSE-D ↓ / LSE-C ↑) | `vts_benchmark` |
| `syncformer` | `AV_DESYNC` | Synchformer A/V de-sync score (uncropped video) | `vts_benchmark` |
| `cpwer_gemini` | `CPWER` | Multi-speaker cpWER via Gemini transcription | `vts_benchmark` |

All metrics use publicly downloadable models.

---

## 1. Installation

```bash
git clone <this-repo> vts-eval
cd vts-eval

# Option A — helper that creates conda envs
bash setup.sh main          # vts_benchmark only (7 metrics)
bash setup.sh full          # both envs (adds qwen3-asr for wer_qwen3)

# Option B — manual install into your own env
pip install -r requirements.txt
```

### Conda environment matrix (only install what you need)

| Env | Setup command | Metrics covered | GPU req. |
|---|---|---|---|
| `vts_benchmark` | `bash setup.sh main` | wer_whisper, sim_wavlm, mcd, utmos, syncnet, syncformer, cpwer_gemini (**7 metrics**) | any CUDA (`syncformer` needs Ampere/Hopper) |
| `qwen3-asr` | `bash setup.sh qwen3` | wer_qwen3 (kept separate: pins `transformers>=4.57`) | any CUDA |

#### Environment 1 / 2 — `vts_benchmark` (7 metrics)

```bash
conda create -y -n vts_benchmark python=3.10
conda activate vts_benchmark
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate librosa soundfile jiwer word2number \
            pymcd speechmos syncnet-python tqdm numpy colorlog scipy \
            einops omegaconf hydra-core decord opencv-python \
            google-genai google-generativeai
```

**Metrics**: `wer_whisper` `sim_wavlm` `mcd` `utmos` `syncnet` `syncformer` `cpwer_gemini`

**Additional setup for `syncformer`**:
```bash
bash download_models.sh --syncformer
# clones https://github.com/v-iashin/Synchformer to thirdparty/av-benchmark
# downloads synchformer_state_dict.pth (~950 MB)
```

**Additional setup for `cpwer_gemini`**:
- Fill in your OpenAI-compatible `key` and `location` in `extra_scripts/gemini_multispk_transcribe.py`.
- Set `CPWER_REF` env var to a JSONL containing per-speaker reference transcripts (`multispk_trans`).

**Quick test**:
```bash
conda activate vts_benchmark
bash eval_all.sh --gen_dir <gen> --gt_jsonl <gt> --datasets chem --metrics mcd --result /tmp/r.json
```

**Known issue**: `syncformer` fails on Blackwell sm_100 (B200) due to CUDA arch mismatch — run on A100/A800/H100 instead.

#### Environment 2 / 2 — `qwen3-asr` (Qwen3-ASR-based WER)

```bash
conda create -y -n qwen3-asr python=3.10
conda activate qwen3-asr
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.57" accelerate qwen-asr librosa soundfile jiwer word2number tqdm
```

**Required model**: Qwen3-ASR-1.7B (HF: `Qwen/Qwen3-ASR-1.7B`, ~3.5 GB). Auto-downloads on first run, or set `QWEN3_ASR_MODEL=/path/to/local/Qwen3-ASR-1.7B`.

**Metrics**: `wer_qwen3` (output `WER`, `CER`)

**Test**:
```bash
conda activate qwen3-asr
python evaluate_wer_qwen3asr.py --audio_dir <gen> --gt_jsonl <gt> --key trans --language English --mode both --trans_jsonl /tmp/qwen3_trans.jsonl
```

---

## 2. Download Model Weights

```bash
# Auto-download SyncNet + Syncformer into ckpts/ / thirdparty/
bash download_models.sh

# Selective:
bash download_models.sh --syncnet
bash download_models.sh --syncformer
```

| Model | Where | Size | Source |
|---|---|---|---|
| `sfd_face.pth` (s3fd-619a316812) | `ckpts/` | ~30 MB | adrianbulat.com |
| `syncnet_v2.model` | `ckpts/` | ~20 MB | Oxford VGG |
| `synchformer_state_dict.pth` | `thirdparty/av-benchmark/av_bench/weights/` | ~950 MB | `hkchengrex/MMAudio` HF |
| HuggingFace models (whisper, wavlm-base-sv, qwen3-asr) | HF cache | auto on first run | HF Hub |
| UTMOS | `cache/torch_hub/` | auto via torch.hub on first run | torch.hub |

Override paths via env vars (see [Customization](#5-customization)).

---

## 3. Input Format

### Generated audio directory
A flat directory of `{id}.wav` files:
```
gen_dir/
├── chem_clip_00001.wav
├── chem_clip_00002.wav
└── ...
```

### Ground-truth JSONL (`--gt_jsonl`)

A ready-to-use sample is shipped in this repo: **`gt_test_lrs3+grid+chem.jsonl`** (3990 rows covering chem 196 / lrs3 514 / grid 3280). Use it as a working example of the expected schema.

**Paths use the placeholder `<VTS_DATA_ROOT>`** — replace it with your local data root before running any metric:

```bash
sed -i "s|<VTS_DATA_ROOT>|/your/path/to/data|g" gt_test_lrs3+grid+chem.jsonl
```

Expected directory layout under your data root:

```
<VTS_DATA_ROOT>/
├── testset/
│   ├── LRS3_debug/{raw,autoavsr}/test/...        # LRS3 videos + audio
│   ├── chem/{new_intervals_correct,new_intervals_wavs_correct}/...
│   ├── GRID_zenodo/{video_mp4,audio_from_video}/s*/...   # GRID reference clips
│   └── dubbging_video_features/{clips,syncformers}/*.npy # CLIP / Syncformer features for LRS3 + chem
├── GRID/videos/s*/...                            # GRID target videos
└── features/{clip,sync}/*.npy                    # CLIP / Syncformer features for GRID
```

You are responsible for preparing the raw test data (LRS3 / GRID / your chem-lecture split) yourself under this structure; only the JSONL metadata + pre-computed feature paths are provided here.


One JSON object per line:
```json
{"id": "chem_clip_00001", "dataset": "chem", "wav": "/abs/path/to/gt.wav", "trans": "the transcript text"}
```

Required fields:
- `id` — must match `{id}.wav` in `gen_dir`
- `dataset` — used to filter; e.g. `chem`, `lrs3`, `grid`, `cinedub_multi`
- `wav` — GT waveform (used for sim_wavlm, mcd)
- `trans` — GT transcript (used for WER)

Optional fields:
- `video_path` — used by `syncnet` (face ROI) and `syncformer` (full-frame)
- `ref_wav`, `ref_trans` — used by reference-speaker / voice-cloning metric variants
- `feature.clip`, `feature.sync` — pre-extracted vision features; scripts re-extract if missing

For `cpwer_gemini`: also set `CPWER_REF` env var to a JSONL containing per-speaker reference transcripts under `multispk_trans`.

---

## 4. Usage

### One-line: full pipeline

```bash
bash eval_all.sh \
    --gen_dir /path/to/gen_wavs \
    --gt_jsonl gt_test_lrs3+grid+chem.jsonl \
    --datasets "chem lrs3 grid" \
    --metrics wer_whisper sim_wavlm mcd utmos syncnet \
    --ngpu 8 \
    --result /path/to/result.json
```

### Single metric, single dataset

```bash
bash eval_all.sh \
    --gen_dir /path/to/gen_wavs \
    --gt_jsonl /path/to/gt.jsonl \
    --datasets chem \
    --metrics utmos \
    --result /tmp/result.json
```

### All metrics

```bash
bash eval_all.sh \
    --gen_dir ... --gt_jsonl ... \
    --datasets "chem lrs3 grid" \
    --result /tmp/result.json
# default --metrics = full ALL_METRICS list (8 metrics)
```

### Direct Python invocation (per metric)

If you don't want the bash dispatcher / multi-env switching:
```bash
# WER via Whisper-large-v3
python evaluate_wer.py \
    --audio_dir /path/to/gen_wavs --gt_jsonl /path/to/gt.jsonl \
    --key trans --language en --mode both

# WavLM speaker similarity
python eval_sim.py \
    --gen_dir ... --gt_jsonl ... --cache_dir cache/sim_wavlm_chem

# MCD
python eval_mcd.py --gen_dir ... --gt_jsonl ...

# UTMOS
python eval_utmos.py --gen_dir ... --gt_jsonl ...

# SyncNet
python eval_syncnet.py --gen_dir ... --gt_jsonl ... --dataset chem --ngpu 1
```

### Output format

`result.json` has one entry per dataset, plus an aggregated `dataset1+dataset2+...` key:
```json
{
  "chem": {
    "WER": 0.0722, "CER": 0.0481,
    "SIM_WAVLM": 0.9481, "MCD_DTW": 6.06, "UTMOS": 3.71,
    "LSE_D": 7.46, "LSE_C": 7.40
  },
  "v2c":   { ... },
  "chem+v2c": { ... }   // mean of available metrics
}
```

---

## 5. Customization

### Environment variables (override hardcoded defaults)

| Variable | Default | Used by |
|---|---|---|
| `CONDA_BASE` | `$HOME/anaconda3` | `eval_all.sh` |
| `CKPTS_DIR` | `./ckpts` | `download_models.sh` |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | `evaluate_wer.py` |
| `QWEN3_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | `evaluate_wer_qwen3asr.py` |
| `WAVLM_BASE_SV` | `microsoft/wavlm-base-sv` | `eval_sim.py` |
| `S3FD_MODEL` | `./ckpts/sfd_face.pth` | `eval_syncnet.py` |
| `SYNCNET_MODEL` | `./ckpts/syncnet_v2.model` | `eval_syncnet.py` |
| `ROI_CACHE_ROOT` | `./cache/roi` | `eval_syncnet.py` |
| `TORCH_HUB_DIR` | `./cache/torch_hub` | `eval_utmos.py` |
| `AV_BENCHMARK_ROOT` | `./thirdparty/av-benchmark` | `eval_syncformer.py` |
| `CPWER_REF` | (required) | `cpwer_gemini` |
| `FFMPEG` | `ffmpeg` (PATH) | `eval_syncnet.py` |
| `PYSCRIPTS_DIR` | `./extra_scripts` | `eval_all.sh` |

### Caching behavior
- `--gt_cache_dir`: stores GT speaker embeddings / mel features (computed once, reused across runs)
- `--pred_cache_dir`: stores prediction-side intermediate features and per-metric logs
- `eval_all.sh` skips a metric if the corresponding `log_{dataset}_{metric}.txt` already exists. Delete the log to force rerun.

### Multi-GPU (WER only)
`accelerate launch` / `torchrun` is used for `wer_*` when `--ngpu > 1`. Other metrics run on a single GPU (parallelism inside the script).

---

## 6. Repository Layout

```
.
├── eval_all.sh               # main bash dispatcher (8 metrics)
├── eval_mcd.py               # MCD-DTW / MCD-DTW-SL
├── eval_sim.py               # WavLM-base-sv similarity (sim_wavlm)
├── eval_syncnet.py           # SyncNet (LSE-D, LSE-C)
├── eval_syncformer.py        # Synchformer A/V de-sync
├── eval_utmos.py             # UTMOS naturalness
├── evaluate_wer.py           # Whisper-large-v3 WER/CER
├── evaluate_wer_qwen3asr.py  # Qwen3-ASR WER/CER
├── utils.py
├── extra_scripts/            # cpWER multi-speaker helpers
│   ├── gemini_multispk_transcribe.py
│   └── evaluate_cpwer.py
├── SKILL.md                  # optional Claude Code /eval_vts skill
├── gt_test_lrs3+grid+chem.jsonl  # sample GT JSONL (3990 rows, chem+lrs3+grid)
├── ckpts/                    # downloaded weights (gitignored)
├── cache/                    # runtime cache (gitignored)
├── thirdparty/               # external repos like av-benchmark (gitignored)
├── requirements.txt
├── setup.sh
├── download_models.sh
├── README.md
└── .gitignore
```

---

## 7. Automation with Claude Code (optional)

If you use [Claude Code](https://github.com/anthropics/claude-code), this repo ships a **`SKILL.md`** that turns evaluation into a slash-command (`/eval_vts`):

```bash
# one-time: install the skill for Claude Code
mkdir -p ~/.claude/skills/eval_vts
cp SKILL.md ~/.claude/skills/eval_vts/SKILL.md

# then in Claude Code:
/eval_vts gen_dir=/path/to/gen_wavs
```

The skill knows the paper-column ↔ metric-key mapping, handles single-model and multi-model parallel runs, auto-splits GPUs across concurrent `gen_dir`s, and prints a compact summary table. See `SKILL.md` for details.

---

## 8. Troubleshooting

- **`syncformer torchhub.load fails offline`**: clone the av-benchmark repo to `thirdparty/av-benchmark` and set `AV_BENCHMARK_ROOT`.
- **`syncnet ffmpeg not found`**: install ffmpeg or set `FFMPEG=/path/to/ffmpeg`.
- **`SIM_WAVLM is NaN` on very short clips**: clips <0.1 s are shorter than WavLM conv kernels — they are skipped by design; valid samples remain in the 0.13–0.98 range.
- **HF download blocked**: use `huggingface-cli download <model>` while a proxy is active, then point env vars at the local cache path.

---

## 9. Citation

If this tooling helped your work, please cite our paper:

```bibtex
@article{dai2026cinedub,
  title   = {CineDub: Scaling End-to-End Video Dubbing to Multi-Speaker
             Dialogues with Coherent Sound Effects},
  author  = {Dai, Yusheng and Wang, Kangdi and Gao, Baolong and Jiang, Yuxuan and
             Wang, Weiqiang and Ke, Qiuhong and Cai, Jianfei},
  journal = {arXiv preprint},
  year    = {2026}
}
```

As well as the underlying methods:
- WavLM — Chen et al., 2022
- UTMOS — Saeki et al., 2022
- SyncNet — Chung & Zisserman, 2016
- Synchformer — Iashin et al., 2024
- Whisper — Radford et al., 2022
- Qwen3-ASR — Shi et al., 2026

---

## 10. Contact

If you have any comments or questions, feel free to contact: **yusheng.dai@monash.edu**

---

## 11. License

Code packaged here for research use. Underlying models retain their original licenses.
