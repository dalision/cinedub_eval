#!/bin/bash
# Full evaluation dispatcher — routes each metric to its conda env, aggregates results.
#
# Usage (single dataset):
#   bash eval_all.sh --gen_dir /path/to/gen_wavs --gt_jsonl /path/to/gt.jsonl \
#       --datasets chem --result /path/to/result.json
#
# Usage (multi dataset, auto-merged):
#   bash eval_all.sh --gen_dir ... --gt_jsonl ... \
#       --datasets "chem v2c" --result /path/to/result.json
#
# Available metrics: wer_whisper wer_qwen3 sim_wavlm mcd utmos syncnet syncformer cpwer_gemini
#
# Env mapping:
#   wer_whisper / sim_wavlm / mcd / utmos / syncnet / syncformer / cpwer_gemini → vts_benchmark
#   wer_qwen3                                                                    → qwen3-asr

# set -e  # 关闭:单个指标失败不应中断整个流程

# ─── 配置 ─────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSCRIPTS_DIR="${PYSCRIPTS_DIR:-${SCRIPT_DIR}/extra_scripts}"

ALL_METRICS="wer_whisper wer_qwen3 sim_wavlm mcd utmos syncnet syncformer cpwer_gemini"

# ─── 参数解析 ──────────────────────────────────────────────────────
GEN_DIR=""
GT_JSONL=""
DATASETS=""       # space-separated, e.g. "chem v2c"
METRICS=""
NGPU=1
GT_TEXT_KEY="trans"
LANGUAGE="en"
GT_CACHE_DIR=""
PRED_CACHE_DIR=""
RESULT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gen_dir)        GEN_DIR="$2";        shift 2 ;;
        --gt_jsonl)       GT_JSONL="$2";       shift 2 ;;
        --datasets)       DATASETS="$2";       shift 2 ;;
        --dataset)        DATASETS="$2";       shift 2 ;;  # legacy alias
        --metrics)        shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do METRICS="$METRICS $1"; shift; done ;;
        --ngpu)           NGPU="$2";           shift 2 ;;
        --key)            GT_TEXT_KEY="$2";    shift 2 ;;
        --language)       LANGUAGE="$2";       shift 2 ;;
        --gt_cache_dir)   GT_CACHE_DIR="$2";   shift 2 ;;
        --pred_cache_dir) PRED_CACHE_DIR="$2"; shift 2 ;;
        --result)         RESULT="$2";         shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

METRICS="${METRICS:-$ALL_METRICS}"

if [[ -z "$GEN_DIR" || -z "$GT_JSONL" || -z "$DATASETS" ]]; then
    echo "Usage: bash eval_all.sh --gen_dir DIR --gt_jsonl FILE --datasets 'chem v2c' --result FILE [--metrics ...]"
    exit 1
fi

PRED_CACHE_DIR="${PRED_CACHE_DIR:-${GEN_DIR}/../metric_cache}"
GT_CACHE_DIR="${GT_CACHE_DIR:-${PRED_CACHE_DIR}}"
mkdir -p "$PRED_CACHE_DIR" "$GT_CACHE_DIR"

echo "============================================================"
echo "  gen_dir        : $GEN_DIR"
echo "  gt_jsonl       : $GT_JSONL"
echo "  datasets       : $DATASETS"
echo "  metrics        : $METRICS"
echo "  ngpu           : $NGPU"
echo "  key            : $GT_TEXT_KEY"
echo "  gt_cache_dir   : $GT_CACHE_DIR"
echo "  pred_cache_dir : $PRED_CACHE_DIR"
echo "  result         : $RESULT"
echo "============================================================"

# ─── conda 环境执行封装 ───────────────────────────────────────────
run_in_env() {
    local env_name=$1; shift
    source "${CONDA_BASE}/bin/activate" "$env_name" \
        && "$@" \
        && source "${CONDA_BASE}/bin/deactivate" 2>/dev/null
}

# ─── Step 1: 按 dataset 过滤 GT jsonl ────────────────────────────
for DATASET in $DATASETS; do
    CACHE_DIR="${PRED_CACHE_DIR}"
    FILTERED_JSONL="${CACHE_DIR}/gt_${DATASET}.jsonl"
    ID_LIST="${CACHE_DIR}/ids_${DATASET}.txt"

    echo
    echo "==========[ dataset: $DATASET ]=========="
    python3 -c "
import json
kept, ids = [], []
for line in open('${GT_JSONL}'):
    d = json.loads(line)
    if d.get('dataset') == '${DATASET}':
        kept.append(line)
        ids.append(d['id'])
open('${FILTERED_JSONL}','w').writelines(kept)
open('${ID_LIST}','w').write('\n'.join(ids)+'\n')
print(f'  filtered {len(kept)} items → ${FILTERED_JSONL}')
"
    if [[ ! -s "$FILTERED_JSONL" ]]; then
        echo "  [SKIP] no items for dataset=$DATASET"
        continue
    fi

    DS_RESULT="${CACHE_DIR}/result_${DATASET}.json"

    # ─── Step 2: 逐指标运行 ────────────────────────────────────
    for metric in $METRICS; do
        METRIC_LOG="${CACHE_DIR}/log_${DATASET}_${metric}.txt"
        if [[ -s "$METRIC_LOG" ]]; then
            echo "  [cache] $metric log exists → $METRIC_LOG (delete to rerun)"
            continue
        fi
        echo
        echo "  ---- [$metric] on $DATASET ----"

        case $metric in
            wer_whisper)
                if [[ $NGPU -gt 1 ]]; then
                    run_in_env vts_benchmark \
                        accelerate launch --num_processes=$NGPU \
                        "${SCRIPT_DIR}/evaluate_wer.py" \
                        --audio_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" \
                        --key "$GT_TEXT_KEY" --language "$LANGUAGE" --id_list "$ID_LIST" --mode both \
                        | tee "$METRIC_LOG"
                else
                    run_in_env vts_benchmark \
                        python3 "${SCRIPT_DIR}/evaluate_wer.py" \
                        --audio_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" \
                        --key "$GT_TEXT_KEY" --language "$LANGUAGE" --id_list "$ID_LIST" --mode both \
                        | tee "$METRIC_LOG"
                fi
                ;;

            wer_qwen3)
                if [[ $NGPU -gt 1 ]]; then
                    run_in_env qwen3-asr \
                        torchrun --nproc_per_node=$NGPU \
                        "${SCRIPT_DIR}/evaluate_wer_qwen3asr.py" \
                        --audio_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" \
                        --key "$GT_TEXT_KEY" --language English --id_list "$ID_LIST" --mode both \
                        | tee "$METRIC_LOG"
                else
                    run_in_env qwen3-asr \
                        python3 "${SCRIPT_DIR}/evaluate_wer_qwen3asr.py" \
                        --audio_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" \
                        --key "$GT_TEXT_KEY" --language English --id_list "$ID_LIST" --mode both \
                        | tee "$METRIC_LOG"
                fi
                ;;

            sim_wavlm)
                run_in_env vts_benchmark \
                    python3 "${SCRIPT_DIR}/eval_sim.py" \
                    --gen_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" --id_list "$ID_LIST" \
                    --cache_dir "$GT_CACHE_DIR/sim_wavlm_${DATASET}" \
                    | tee "$METRIC_LOG"
                ;;

            mcd)
                run_in_env vts_benchmark \
                    python3 "${SCRIPT_DIR}/eval_mcd.py" \
                    --gen_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" --id_list "$ID_LIST" \
                    | tee "$METRIC_LOG"
                ;;

            utmos)
                run_in_env vts_benchmark \
                    python3 "${SCRIPT_DIR}/eval_utmos.py" \
                    --gen_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" --id_list "$ID_LIST" \
                    | tee "$METRIC_LOG"
                ;;

            syncnet)
                run_in_env vts_benchmark \
                    python3 "${SCRIPT_DIR}/eval_syncnet.py" \
                    --gen_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" --id_list "$ID_LIST" \
                    --dataset "$DATASET" \
                    --ngpu $NGPU --num_workers 2 \
                    | tee "$METRIC_LOG"
                ;;

            syncformer)
                run_in_env vts_benchmark \
                    PYTHONPATH=${AV_BENCHMARK_ROOT:-${SCRIPT_DIR}/thirdparty/av-benchmark}:\$PYTHONPATH \
                    python3 "${SCRIPT_DIR}/eval_syncformer.py" \
                    --gen_dir "$GEN_DIR" --gt_jsonl "$FILTERED_JSONL" --id_list "$ID_LIST" \
                    --cache_dir "$GT_CACHE_DIR/syncformer_${DATASET}" \
                    | tee "$METRIC_LOG"
                ;;

            cpwer_gemini)
                # Step 1: Gemini multi-speaker transcription
                CPWER_HYP="${CACHE_DIR}/multispk_trans_${DATASET}.jsonl"
                CPWER_REF="${CPWER_REF:?ERROR: set CPWER_REF env var to multispk reference jsonl}"
                if [[ ! -s "$CPWER_HYP" ]] || [[ $(wc -l < "$CPWER_HYP") -lt $(wc -l < "$ID_LIST") ]]; then
                    echo "  [cpwer_gemini] Running Gemini multispk transcription..."
                    run_in_env vts_benchmark \
                        python3 "${PYSCRIPTS_DIR}/gemini_multispk_transcribe.py" \
                        --input_dir "$GEN_DIR" \
                        --output_file "$CPWER_HYP" \
                        --id_list "$ID_LIST" \
                        --num_processes 256 --max_rounds 3
                fi
                echo "  [cpwer_gemini] Computing cpWER..."
                run_in_env vts_benchmark \
                    python3 "${PYSCRIPTS_DIR}/evaluate_cpwer.py" \
                    --ref "$CPWER_REF" --hyp "$CPWER_HYP" \
                    --ref_field multispk_trans --hyp_field multispk_trans \
                    | tee "$METRIC_LOG"
                ;;

            *)
                echo "  [SKIP] Unknown metric: $metric"
                ;;
        esac
    done

    # aggregate per-dataset metrics → result_${DATASET}.json
    python3 -c "
import json, re, os
cache_dir = '${CACHE_DIR}'
result_path = '${DS_RESULT}'
patterns = {
    'WER': (r'Corpus WER\s*:\s*([\d.]+)', ['wer_whisper', 'wer_qwen3']),
    'CER': (r'Corpus CER\s*:\s*([\d.]+)', ['wer_whisper', 'wer_qwen3']),
    'SIM_WAVLM': (r'Mean Speaker Similarity\s*:\s*([\d.]+)', ['sim_wavlm']),
    'MCD_DTW': (r'MCD \(DTW\)\s*:\s*([\d.]+)', None),
    'MCD_DTW_SL': (r'MCD \(DTW-SL\)\s*:\s*([\d.]+)', None),
    'UTMOS': (r'Mean UTMOS\s*:\s*([\d.]+)', None),
    'LSE_D': (r'LSE-D.*?:\s*([\d.]+)', None),
    'LSE_C': (r'LSE-C.*?:\s*([\d.]+)', None),
    'AV_DESYNC': (r'AV DeSync.*?:\s*([\d.]+)', None),
    'CPWER': (r'cpWER.*?:\s*([\d.]+)%', ['cpwer_gemini']),
}
results = {}
prefix = 'log_${DATASET}_'
for f in sorted(os.listdir(cache_dir)):
    if not f.startswith(prefix) or not f.endswith('.txt'):
        continue
    metric_name = f[len(prefix):-4]
    text = open(os.path.join(cache_dir, f)).read()
    for key, (pat, only_metrics) in patterns.items():
        if only_metrics is not None and metric_name not in only_metrics:
            continue
        m = re.search(pat, text)
        if m:
            results[key] = round(float(m.group(1)), 6)
with open(result_path, 'w') as out:
    json.dump(results, out, indent=2)
print(f'Result ({len(results)} metrics) → {result_path}')
for k, v in sorted(results.items()):
    print(f'  {k}: {v}')
"
done

echo ""
echo "============================================================"
echo "  All datasets done!"
echo "============================================================"

# ─── Step 3: merge all datasets → final RESULT JSON ──────────────
if [[ -n "$RESULT" ]]; then
    source ${CONDA_BASE}/bin/activate vts_benchmark
    python3 -c "
import json, os
import numpy as np
datasets = '${DATASETS}'.split()
combined_key = '+'.join(datasets)
pred_cache = '${PRED_CACHE_DIR}'
result_path = '${RESULT}'
all_results = {}
for ds in datasets:
    ds_file = os.path.join(pred_cache, f'result_{ds}.json')
    if os.path.exists(ds_file):
        all_results[ds] = json.load(open(ds_file))
if len(all_results) > 1:
    keys = set(k for v in all_results.values() for k in v)
    merged = {k: round(float(np.mean([all_results[ds][k] for ds in datasets if k in all_results.get(ds, {})])), 6)
              for k in sorted(keys)}
    all_results[combined_key] = merged
os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
with open(result_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Final result → {result_path}')
for k, v in all_results.items():
    print(f'  [{k}] {v}')
"
fi
