#!/bin/bash
# Download all model weights required by VTS evaluation metrics.
# Models are placed under ./ckpts/ unless overridden by environment variables.
#
# Usage:
#   bash download_models.sh                # download everything
#   bash download_models.sh --skip-hf      # skip HF auto-cached ones (whisper, wavlm, etc.)
#   bash download_models.sh --syncnet      # only syncnet weights
#   bash download_models.sh --syncformer   # only syncformer weights
#
# Environment overrides:
#   CKPTS_DIR=./ckpts        # destination root
#   HF_TOKEN=<token>         # for gated repos / faster downloads

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPTS_DIR="${CKPTS_DIR:-${SCRIPT_DIR}/ckpts}"
mkdir -p "$CKPTS_DIR"

cd "$CKPTS_DIR"

DOWNLOAD_SYNCNET=0
DOWNLOAD_SYNCFORMER=0
DOWNLOAD_HF=1
ALL=1

if [[ $# -gt 0 ]]; then
    ALL=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            --syncnet)        DOWNLOAD_SYNCNET=1; shift ;;
            --syncformer)     DOWNLOAD_SYNCFORMER=1; shift ;;
            --skip-hf)        DOWNLOAD_HF=0; ALL=1; shift ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
fi

echo "Destination: $CKPTS_DIR"
echo

# ─── 1. SyncNet (sfd_face + syncnet_v2) ─────────────────────────
if [[ $ALL == 1 || $DOWNLOAD_SYNCNET == 1 ]]; then
    echo "[1/3] SyncNet face detector + SyncNet v2"
    if [[ ! -f sfd_face.pth ]]; then
        # Source: face-alignment / python-fan (s3fd-619a316812.pth)
        wget -q --show-progress -O sfd_face.pth \
            "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
            || echo "  WARN: sfd_face.pth download failed"
    fi
    if [[ ! -f syncnet_v2.model ]]; then
        # Source: Oxford VGG official lipsync data
        wget -q --show-progress -O syncnet_v2.model \
            "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model" \
            || echo "  WARN: syncnet_v2.model download failed"
    fi
    # sfd_face_remapped.pth — the original sfd_face.pth uses legacy VGG-style
    # key names (conv1_1 / conv2_1 / … / fc7) that the current syncnet_python
    # S3FDNet does not accept (it expects vgg.0 / vgg.2 / extras / loc / conf /
    # L2Norm3_3). Perform a real state_dict key remap.
    if [[ -f sfd_face.pth && ! -f sfd_face_remapped.pth ]]; then
        python3 - <<'PYEOF'
import torch
sd = torch.load('sfd_face.pth', map_location='cpu', weights_only=False)
mapping = {
    'conv1_1':'vgg.0','conv1_2':'vgg.2','conv2_1':'vgg.5','conv2_2':'vgg.7',
    'conv3_1':'vgg.10','conv3_2':'vgg.12','conv3_3':'vgg.14',
    'conv4_1':'vgg.17','conv4_2':'vgg.19','conv4_3':'vgg.21',
    'conv5_1':'vgg.24','conv5_2':'vgg.26','conv5_3':'vgg.28',
    'fc6':'vgg.31','fc7':'vgg.33',
    'conv3_3_norm':'L2Norm3_3','conv4_3_norm':'L2Norm4_3','conv5_3_norm':'L2Norm5_3',
    'conv6_1':'extras.0','conv6_2':'extras.1','conv7_1':'extras.2','conv7_2':'extras.3',
    'conv3_3_norm_mbox_loc':'loc.0','conv4_3_norm_mbox_loc':'loc.1','conv5_3_norm_mbox_loc':'loc.2',
    'fc7_mbox_loc':'loc.3','conv6_2_mbox_loc':'loc.4','conv7_2_mbox_loc':'loc.5',
    'conv3_3_norm_mbox_conf':'conf.0','conv4_3_norm_mbox_conf':'conf.1','conv5_3_norm_mbox_conf':'conf.2',
    'fc7_mbox_conf':'conf.3','conv6_2_mbox_conf':'conf.4','conv7_2_mbox_conf':'conf.5',
}
out = {}
for k, v in sd.items():
    stem, _, suf = k.rpartition('.')
    if stem not in mapping:
        raise RuntimeError(f'sfd_face.pth: unmapped key {k!r} — remap table needs update')
    out[f'{mapping[stem]}.{suf}'] = v
torch.save(out, 'sfd_face_remapped.pth')
print(f'  remapped {len(out)} keys -> sfd_face_remapped.pth')
PYEOF
    fi
    echo "  done"
fi

# ─── 2. Syncformer (audio-visual sync transformer) ──────────────
if [[ $ALL == 1 || $DOWNLOAD_SYNCFORMER == 1 ]]; then
    echo "[2/3] Syncformer"
    mkdir -p "${SCRIPT_DIR}/thirdparty"
    if [[ ! -d "${SCRIPT_DIR}/thirdparty/av-benchmark" ]]; then
        echo "  Cloning v-iashin/Synchformer (provides av-benchmark evaluation code)..."
        git clone https://github.com/v-iashin/Synchformer.git "${SCRIPT_DIR}/thirdparty/av-benchmark" \
            || echo "  WARN: clone failed; place av-benchmark manually"
    fi
    # Syncformer state dict — from MMAudio HF (hkchengrex/MMAudio)
    AV_WEIGHTS="${SCRIPT_DIR}/thirdparty/av-benchmark/av_bench/weights"
    mkdir -p "$AV_WEIGHTS"
    if [[ ! -f "$AV_WEIGHTS/synchformer_state_dict.pth" ]]; then
        wget -q --show-progress -O "$AV_WEIGHTS/synchformer_state_dict.pth" \
            "https://huggingface.co/hkchengrex/MMAudio/resolve/main/ext_weights/synchformer_state_dict.pth" \
            || echo "  WARN: synchformer_state_dict.pth download failed"
    fi
    echo "  done → $AV_WEIGHTS/synchformer_state_dict.pth"
fi

# ─── 3. HF-cached models (auto-pulled at first run if HF accessible) ─
if [[ $ALL == 1 && $DOWNLOAD_HF == 1 ]]; then
    echo "[3/3] HF auto-cached models"
    echo "  microsoft/wavlm-base-sv      (eval_sim.py)               → auto from HF Hub on first run"
    echo "  openai/whisper-large-v3      (evaluate_wer.py)           → auto"
    echo "  Qwen/Qwen3-ASR-1.7B          (evaluate_wer_qwen3asr.py)  → auto"
    echo "  UTMOS                        (eval_utmos.py)             → torch.hub auto on first run"
    echo
    echo "  If your environment cannot reach HF, prefetch with:"
    echo "    huggingface-cli download microsoft/wavlm-base-sv"
    echo "    huggingface-cli download openai/whisper-large-v3"
    echo "    huggingface-cli download Qwen/Qwen3-ASR-1.7B"
fi

echo
echo "============================================================"
echo "Model download complete. Files under: $CKPTS_DIR"
echo "============================================================"
ls -lh "$CKPTS_DIR" 2>/dev/null
