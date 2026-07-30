import os
#!/usr/bin/env python3
"""
Speaker Similarity (SIM) 评估脚本
  - 使用 WavLM-base-sv 提取说话人 embedding
  - 缓存：GT embedding 预提取并保存为 {cache_dir}/gt_spk_embs.pt
    下次运行只需提取生成音频的 embedding，与缓存的 GT embedding 计算余弦相似度

用法：
  python eval_sim.py --gen_dir /path/to/gen_wavs --gt_jsonl /path/to/gt.jsonl
  python eval_sim.py --gen_dir ... --gt_jsonl ... --model_path microsoft/wavlm-base-sv
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import build_dur_map, build_gt_map, get_gen_audio_map, load_audio_16k, load_cache, match_gen_gt, save_cache

DEFAULT_MODEL = os.environ.get("WAVLM_BASE_SV", "microsoft/wavlm-base-sv")


# ─── 模型加载 ─────────────────────────────────────────────────────────────────

def load_wavlm(model_name_or_path: str, device):
    from transformers import AutoFeatureExtractor, WavLMForXVector
    print(f"Loading WavLM from {model_name_or_path} ...")
    extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    model = WavLMForXVector.from_pretrained(model_name_or_path).to(device).eval()
    return extractor, model


# ─── 提取 embedding ───────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(audio: np.ndarray, extractor, model, device) -> torch.Tensor:
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = model(**inputs).embeddings  # (1, D)
    emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0).cpu()


def extract_embeddings_batch(
    items: list,      # [(id, wav_path), ...]
    extractor, model, device,
    dur_map: dict = None,
    desc="Extracting"
) -> dict:
    embs = {}
    for sid, wav_path in tqdm(items, desc=desc):
        try:
            dur = dur_map.get(sid) if dur_map else None
            audio = load_audio_16k(wav_path, dur=dur)
            embs[sid] = extract_embedding(audio, extractor, model, device)
        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")
    return embs


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音频目录")
    parser.add_argument("--gt_jsonl", type=str, required=True, help="GT JSONL 文件")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="缓存目录（默认 gen_dir/../metric_cache/sim）")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL,
                        help=f"WavLM 模型路径（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--id_list", type=str, default=None, help="ID 过滤文件")
    parser.add_argument("--ref_key", type=str, default="wav",
                        help="JSONL 中用作参考的 wav 字段（默认 'wav'，可改为 'ref_wav'）")
    parser.add_argument("--force_gt_cache", action="store_true",
                        help="强制重新提取 GT embedding（忽略已有缓存）")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 缓存路径
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.gen_dir).parent / "metric_cache" / "sim")
    os.makedirs(args.cache_dir, exist_ok=True)

    # 加载数据
    from utils import load_id_list
    id_set = load_id_list(args.id_list)
    gt_map = build_gt_map(args.gt_jsonl)
    gen_map = get_gen_audio_map(args.gen_dir, id_set)
    matched = match_gen_gt(gen_map, gt_map)
    if not matched:
        print("ERROR: 无匹配样本")
        sys.exit(1)

    # 加载模型
    extractor, model = load_wavlm(args.model_path, device)

    # ── 参考 embedding 缓存 ──
    ref_key = args.ref_key
    gt_emb_cache_name = f"gt_spk_embs_{ref_key}.pt"
    gt_embs = None if args.force_gt_cache else load_cache(args.cache_dir, gt_emb_cache_name)

    if gt_embs is None:
        print(f"\n[REF Cache] 提取参考 speaker embeddings (ref_key={ref_key}) ...")
        gt_items = []
        for sid, gen_path, gt_item in matched:
            ref_wav = gt_item.get(ref_key) or (gt_item.get("wav_path") if ref_key == "wav" else None)
            if ref_wav and Path(ref_wav).exists():
                gt_items.append((sid, ref_wav))
            else:
                print(f"  [SKIP REF] {sid}: 参考 wav 不存在 ({ref_wav})")
        gt_embs = extract_embeddings_batch(gt_items, extractor, model, device, desc="REF embeddings")
        save_cache(args.cache_dir, gt_emb_cache_name, gt_embs)
    else:
        print(f"[REF Cache] 已加载 {len(gt_embs)} 条参考 embeddings")

    # ── 生成音频 embedding ──
    print("\n[GEN] 提取生成音频 speaker embeddings ...")
    dur_map = build_dur_map(args.gt_jsonl)
    gen_items = [(sid, gen_path) for sid, gen_path, _ in matched if sid in gt_embs]
    gen_embs = extract_embeddings_batch(gen_items, extractor, model, device, dur_map=dur_map, desc="GEN embeddings")

    # ── 计算余弦相似度 ──
    sims = []
    for sid in sorted(gen_embs.keys()):
        if sid not in gt_embs:
            continue
        sim = F.cosine_similarity(gen_embs[sid].unsqueeze(0), gt_embs[sid].unsqueeze(0)).item()
        sims.append(sim)

    if not sims:
        print("ERROR: 无法计算相似度，检查 ID 匹配")
        sys.exit(1)

    mean_sim = float(np.mean(sims))
    print(f"\n[SIM] {len(sims)} samples")
    print(f"  Mean Speaker Similarity : {mean_sim:.4f}")
    print(f"  Min / Max               : {min(sims):.4f} / {max(sims):.4f}")
    return mean_sim


if __name__ == "__main__":
    main()
