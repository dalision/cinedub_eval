import os
#!/usr/bin/env python3
"""
UTMOS 评估脚本（预测 MOS 音质评分）
  - 使用 utmos22_strong 模型（从 torch.hub 加载 SpeechMOS）
  - 只需生成音频，无需 GT
  - 批处理加速

用法：
  python eval_utmos.py --gen_dir /path/to/gen_wavs
  python eval_utmos.py --gen_dir ... --batch_size 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import librosa
from tqdm import tqdm


TORCH_HUB_DIR = os.environ.get("TORCH_HUB_DIR", os.path.join(os.path.dirname(__file__), "cache", "torch_hub"))


def load_utmos(device):
    print("Loading UTMOS (utmos22_strong) from torch.hub ...")
    torch.hub.set_dir(TORCH_HUB_DIR)
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True
    ).to(device).eval()
    return predictor


@torch.no_grad()
def score_batch(predictor, wav_batch: list, device) -> list:
    """
    wav_batch: list of np.ndarray (各自长度可不同，逐条处理)
    返回分数列表
    """
    scores = []
    for audio in wav_batch:
        tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        score = predictor(tensor, sr=16000).item()
        scores.append(score)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音频目录")
    parser.add_argument("--gt_jsonl", type=str, default=None, help="GT JSONL 文件（用于 dur 裁剪）")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小（默认 8）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--id_list", type=str, default=None, help="ID 过滤文件（每行一个 ID）")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    gen_dir = Path(args.gen_dir)
    wav_files = sorted(gen_dir.glob("*.wav"))
    if args.id_list:
        with open(args.id_list) as f:
            id_set = {line.strip() for line in f if line.strip()}
        wav_files = [p for p in wav_files if p.stem in id_set]
        print(f"Filtered to {len(wav_files)} files by id_list")
    if not wav_files:
        print(f"ERROR: {gen_dir} 下没有 wav 文件")
        sys.exit(1)

    # 构建 dur 映射
    dur_map = {}
    if args.gt_jsonl:
        from utils import build_dur_map
        dur_map = build_dur_map(args.gt_jsonl)

    print(f"Found {len(wav_files)} wav files")
    predictor = load_utmos(device)

    all_scores = []
    errors = []

    for i in tqdm(range(0, len(wav_files), args.batch_size), desc="UTMOS"):
        batch_files = wav_files[i: i + args.batch_size]
        batch_audio = []
        batch_ids = []
        for p in batch_files:
            try:
                audio, _ = librosa.load(str(p), sr=16000, mono=True)
                dur = dur_map.get(p.stem)
                if dur is not None:
                    n = int(dur * 16000)
                    if n < len(audio):
                        audio = audio[:n]
                batch_audio.append(audio)
                batch_ids.append(p.stem)
            except Exception as e:
                errors.append((p.stem, str(e)))

        if batch_audio:
            try:
                scores = score_batch(predictor, batch_audio, device)
                all_scores.extend(scores)
            except Exception as e:
                for sid in batch_ids:
                    errors.append((sid, str(e)))

    if errors:
        print(f"\n  [WARNING] {len(errors)} 样本出错：")
        for sid, err in errors[:5]:
            print(f"    [{sid}] {err}")

    if not all_scores:
        print("ERROR: 所有样本计算失败")
        sys.exit(1)

    mean_utmos = float(np.mean(all_scores))
    print(f"\n[UTMOS] {len(all_scores)} samples")
    print(f"  Mean UTMOS : {mean_utmos:.4f}")
    print(f"  Min / Max  : {min(all_scores):.4f} / {max(all_scores):.4f}")
    return mean_utmos


if __name__ == "__main__":
    main()
