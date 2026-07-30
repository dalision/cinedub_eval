#!/bin/bash
# setup.sh — Create the conda environments needed by VTS-Eval.
#
# 8 metrics share these 2 environments (mapping defined in eval_all.sh):
#
#   vts_benchmark : wer_whisper, sim_wavlm, mcd, utmos, syncnet, syncformer, cpwer_gemini
#   qwen3-asr     : wer_qwen3   (kept separate because it pins transformers>=4.57)
#
# Usage:
#   bash setup.sh main          # vts_benchmark only (covers 7 metrics)
#   bash setup.sh full          # both envs
#   bash setup.sh qwen3         # qwen3-asr env only

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── 1. vts_benchmark — covers 7 metrics ──────────────────────────
create_vts_benchmark() {
    echo "==> Creating env: vts_benchmark (python=3.10, CUDA 12.1)"
    conda create -y -n vts_benchmark python=3.10
    conda run -n vts_benchmark pip install \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # core deps
    conda run -n vts_benchmark pip install \
        transformers \
        accelerate \
        librosa \
        soundfile \
        jiwer \
        word2number \
        pymcd \
        speechmos \
        syncnet-python \
        tqdm \
        numpy \
        colorlog \
        scipy
    # syncformer deps (av-benchmark stack)
    conda run -n vts_benchmark pip install \
        einops \
        omegaconf \
        hydra-core \
        decord \
        opencv-python
    # cpwer_gemini deps (Gemini / OpenAI-compatible client)
    conda run -n vts_benchmark pip install \
        google-genai \
        google-generativeai
    echo "    [OK] vts_benchmark created"
    echo "    NOTES:"
    echo "      - syncformer: clone Synchformer/av-benchmark to thirdparty/av-benchmark"
    echo "                    or set AV_BENCHMARK_ROOT env var. See download_models.sh --syncformer"
    echo "      - cpwer_gemini: fill in your OpenAI-compatible key & endpoint in"
    echo "                     extra_scripts/gemini_multispk_transcribe.py, and set CPWER_REF"
}

# ─── 2. qwen3-asr — for Qwen3-ASR-1.7B (kept separate) ────────────
create_qwen3_asr() {
    echo "==> Creating env: qwen3-asr (python=3.10)"
    conda create -y -n qwen3-asr python=3.10
    conda run -n qwen3-asr pip install \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    conda run -n qwen3-asr pip install \
        "transformers>=4.57" \
        accelerate \
        qwen-asr \
        librosa \
        soundfile \
        jiwer \
        word2number \
        tqdm
    echo "    [OK] qwen3-asr created"
}

case "${1:-}" in
    main|vts_benchmark) create_vts_benchmark ;;
    qwen3|qwen3-asr|qwen3_asr)
                        create_qwen3_asr ;;
    full)
        create_vts_benchmark
        create_qwen3_asr
        ;;
    *)
        cat <<EOF
Usage: bash setup.sh <env>

Environments:
  main / vts_benchmark   — 7 metrics (wer_whisper, sim_wavlm, mcd, utmos, syncnet, syncformer, cpwer_gemini)
  qwen3 / qwen3-asr      — wer_qwen3

Convenience:
  full                   — both envs

Examples:
  bash setup.sh main         # quick start, most users
  bash setup.sh full         # everything
EOF
        exit 1
        ;;
esac

echo
echo "Activate with: conda activate <env_name>"
