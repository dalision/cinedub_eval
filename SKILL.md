---
name: eval_vts
description: Run video-to-speech / visual dubbing evaluation with the vts-eval toolkit (WER, SIM, UTMOS, MCD, SyncNet, Synchformer, cpWER). Triggers when the user says "evaluate", "score", "run metrics", "跑指标", "评估", "/eval_vts".
allowed-tools: Bash, Read, Grep, Glob
---

# /eval_vts — VTS-Eval automation skill

Automates evaluation of generated audio using this repo's `eval_all.sh` dispatcher (see `README.md` for full metric documentation). Handles single-model and multi-model (parallel) evaluation, GPU allocation, and result summarisation.

**Prerequisite**: the vts-eval repo (this directory) is checked out and one of the conda envs from `setup.sh` is activated. All paths in this skill are relative to the vts-eval repo root — set `VTS_EVAL_ROOT` if you invoke from elsewhere.

## Configuration (override via env vars)

```bash
VTS_EVAL_ROOT="${VTS_EVAL_ROOT:-$(pwd)}"         # repo root
EVAL_SCRIPT="${VTS_EVAL_ROOT}/eval_all.sh"
GT_JSONL="${GT_JSONL:-${VTS_EVAL_ROOT}/gt_test_lrs3+grid+chem.jsonl}"
GT_CACHE_DIR="${GT_CACHE_DIR:-${VTS_EVAL_ROOT}/cache/gt}"
DEFAULT_DATASETS="chem lrs3 grid"
DEFAULT_METRICS="wer_whisper sim_wavlm mcd utmos syncnet"
NGPU_DEFAULT=8
```

### Sample GT JSONL

A ready-to-use example shipped in this repo:

- **File**: `gt_test_lrs3+grid+chem.jsonl` (~4.6 MB, 3990 rows)
- **Datasets**: `chem` (196), `lrs3` (514), `grid` (3280)

Schema (one JSON object per line):

```json
{
  "id": "RiM5aSvaNkg_00001_00002",
  "dataset": "lrs3",
  "wav": "/abs/path/to/gt.wav",
  "trans": "and the soldier on the front tank said we have unconditional orders to destroy this",
  "dur": 5.632,
  "sr": 16000,
  "ref_wav": "/abs/path/to/ref_speaker.wav",
  "ref_trans": "i went to extreme lengths to try to be straight",
  "ref_dur": 2.624,
  "video_path": "/abs/path/to/gt_video.mp4",
  "ref_video_path": "/abs/path/to/ref_video.mp4",
  "trans_cap": "A mature man with an American accent ... <S>and the soldier on the front tank said ...<E>",
  "feature": {
    "clip":  "/abs/path/to/clip_feature.npy",
    "sync":  "/abs/path/to/syncformer_feature.npy"
  },
  "split": "test",
  "video_id": "RiM5aSvaNkg",
  "src_utt": "00001",
  "tgt_utt": "00002",
  "org_trans": "..."
}
```

Field usage:

| Field | Required by |
|---|---|
| `id` | all — must equal `<id>.wav` in `--gen_dir` |
| `dataset` | all — used by `--datasets` filter |
| `wav` | `sim_wavlm`, `mcd`, `wer_*` |
| `trans` | `wer_whisper`, `wer_qwen3` |
| `video_path` | `syncnet`, `syncformer` |
| `ref_wav`, `ref_trans` | reference-speaker / voice-cloning variants (optional) |
| `feature.clip`, `feature.sync` | pre-extracted vision features (optional; scripts re-extract if missing) |
| `trans_cap`, `org_trans`, `split`, `dur`, `sr`, ... | model-specific / diagnostic (unused by eval metrics) |

For multi-speaker dubbing on CineDub-Multi, set `DEFAULT_METRICS` to include `cpwer_gemini syncformer` and `DEFAULT_DATASETS="cinedub_multi"`; for V2SA on CineDub-SA, use `wer_qwen3` instead of `wer_whisper`.

## Metrics ↔ paper columns

| Paper column | Metric key | Env |
|---|---|---|
| WER ↓ (single-speaker) | `wer_whisper` | `vts_benchmark` |
| WER ↓ (V2SA) | `wer_qwen3` | `qwen3-asr` |
| cpWER ↓ (multi-speaker) | `cpwer_gemini` | `vts_benchmark` |
| SIM ↑ (SPK-SIM, WavLM-SV) | `sim_wavlm` | `vts_benchmark` |
| MCD-DTW ↓ / MCD-DTW-SL ↓ | `mcd` | `vts_benchmark` |
| UTMOS ↑ | `utmos` | `vts_benchmark` |
| LSE-D ↓ / LSE-C ↑ | `syncnet` | `vts_benchmark` |
| Desync ↓ | `syncformer` | `vts_benchmark` |

## Env activation

`eval_all.sh` switches envs automatically via its `run_in_env` helper — you do **not** need to `conda activate` manually when invoking the dispatcher.

Only if you call a metric script directly (e.g. `python eval_syncformer.py ...`) do you need to activate the right env yourself:

- **`vts_benchmark`** — 7 metrics: `wer_whisper` `sim_wavlm` `mcd` `utmos` `syncnet` `syncformer` `cpwer_gemini`
- **`qwen3-asr`** — 1 metric: `wer_qwen3`

Installed by `bash setup.sh main` and `bash setup.sh qwen3` (or `bash setup.sh full` for both).

## Execution flow

### Step 1 — parse user input

| Argument | Required | Default | Description |
|---|---|---|---|
| `gen_dir` | ✅ | — | flat directory of `{id}.wav` (can be given multiple times) |
| `datasets` | | `chem v2c lrs3 grid` | space-separated dataset names in `--gt_jsonl` |
| `metrics` | | 5 default core metrics | subset of the metric-key list above |
| `ngpu` | | 8 | number of GPUs for WER metrics |
| `gt_jsonl` | | env `GT_JSONL` | ground-truth JSONL |

If the user gives a model name + step instead of a full `gen_dir`, resolve it against your local convention (project-specific). Example: `<exp_root>/<model>/generation/<step>/<test_name>/`.

### Step 2 — build & run the evaluation

**Single `gen_dir`**:

```bash
cd "$VTS_EVAL_ROOT"
result_path="${gen_dir%/}_result.json"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$EVAL_SCRIPT" \
  --gen_dir "$gen_dir" \
  --gt_jsonl "$GT_JSONL" \
  --datasets "$DATASETS" \
  --metrics $METRICS \
  --ngpu $NGPU_DEFAULT \
  --gt_cache_dir "$GT_CACHE_DIR" \
  --pred_cache_dir "${gen_dir}/../metric_cache" \
  --result "$result_path"
```

**Multiple `gen_dir` in parallel** (share the same GPU pool):

```bash
# 2 dirs: split (0,1,2,3) + (4,5,6,7), ngpu=4 each
# 3 dirs: split (0,1,2) + (3,4,5) + (6,7), ngpu=3,3,2
# 4 dirs: split each pair of GPUs, ngpu=2

for i in "${!GEN_DIRS[@]}"; do
    devices="${DEVICE_SPLITS[$i]}"
    ngpu="${NGPU_SPLITS[$i]}"
    gen="${GEN_DIRS[$i]}"
    log="${gen}/../eval_${i}.log"
    result="${gen%/}_result.json"

    CUDA_VISIBLE_DEVICES=$devices nohup bash "$EVAL_SCRIPT" \
        --gen_dir "$gen" \
        --gt_jsonl "$GT_JSONL" \
        --datasets "$DATASETS" \
        --metrics $METRICS \
        --ngpu $ngpu \
        --gt_cache_dir "$GT_CACHE_DIR" \
        --pred_cache_dir "${gen}/../metric_cache" \
        --result "$result" \
        > "$log" 2>&1 &
done
wait
```

### Step 3 — read `result.json` and format

```bash
cat "$result_path"
```

Present results as a compact table, one row per dataset (and one aggregated row across datasets when `--datasets` has more than one):

```
| Dataset | WER↓ | SIM↑ | UTMOS↑ | MCD_DTW↓ | MCD_SL↓ | LSE-D↓ | LSE-C↑ |
```

For multi-speaker CineDub-Multi results the table becomes:

```
| Dataset       | cpWER↓ | WER↓ | UTMOS↑ | MCD_DTW↓ | Desync↓ |
```

### Progress monitoring (background runs)

```bash
# is the eval still running?
ps aux | grep eval_all.sh | grep -v grep

# tail the log
tail -20 "$log"

# check partial result
cat "$result_path" 2>/dev/null
```

## Example interactions

### Example 1 — full evaluation of a single model

User: `/eval_vts gen_dir=./exp/mymodel/step100k/wavs`

1. Verify `GT_JSONL` is set; run `eval_all.sh` with the 5 default metrics on 4 default datasets.
2. Wait for completion (or background if the user expects to check later).
3. Read `_result.json` and print the summary table.

### Example 2 — parallel evaluation of multiple checkpoints

User: `evaluate step100k step200k step300k of ./exp/mymodel`

1. Resolve the three `gen_dir` paths.
2. Apply the 3-way GPU split (0,1,2)+(3,4,5)+(6,7), ngpu=3,3,2.
3. Launch three background `eval_all.sh` invocations, wait, then aggregate.

### Example 3 — recompute a single metric

User: `just re-run utmos on this dir`

1. Delete the cached log: `rm -f <pred_cache>/log_<dataset>_utmos.txt` (`eval_all.sh` skips a metric if its log already exists).
2. Invoke `eval_all.sh` with `--metrics utmos`.

### Example 4 — multi-speaker dubbing evaluation

User: `evaluate ./gen/mymodel on CineDub-Multi`

1. Set `DATASETS="cinedub_multi"`, `METRICS="wer_whisper cpwer_gemini sim_wavlm mcd utmos syncformer"`.
2. Ensure `CPWER_REF` env var points to the multi-speaker reference JSONL (per-speaker `multispk_trans`).
3. Ensure the OpenAI-compatible key/endpoint is filled in `extra_scripts/gemini_multispk_transcribe.py`.
4. Run `eval_all.sh` and report cpWER + Desync + speech metrics.

## Notes on caching

- `eval_all.sh` skips a metric when `<pred_cache>/log_<dataset>_<metric>.txt` already exists — delete the log to force a rerun.
- `GT_CACHE_DIR` stores speaker embeddings / mel features computed once from the ground-truth JSONL. Reuse across gen_dirs by passing the same `--gt_cache_dir`.
- `syncnet` builds a `roi/` cache under `ROI_CACHE_ROOT` (default `./cache/roi`) — this can be shared across runs on the same test set.

## Troubleshooting

- **Wrong / missing conda env**: this skill assumes you've run `bash setup.sh main` (or `full`) at least once. See the repo README's env matrix.
- **`syncformer` needs Ampere/Hopper**: the skill silently skips it on Blackwell (sm_100); rerun on A100/A800/H100.
- **cpwer_gemini rate limits**: reduce `--num_processes` inside `gemini_multispk_transcribe.py` or set a smaller `--max_rounds`.
- **`GT_JSONL` schema**: each row needs `id`, `dataset`, `wav`, `trans`, optionally `video` (for sync metrics) and `multispk_trans` (for cpWER).
