#!/usr/bin/env python3
"""
公共工具模块：JSONL 加载、ID 匹配、缓存读写、音频加载。
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─── JSONL 工具 ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_gt_map(jsonl_path: str, id_field: str = "id") -> Dict[str, dict]:
    """
    读取 GT JSONL，返回 {id: item_dict}。
    支持 id 字段不存在时用 wav/video_path 的 stem 作为 id。
    """
    data = load_jsonl(jsonl_path)
    gt_map = {}
    for item in data:
        item_id = _extract_id(item, id_field)
        if item_id:
            gt_map[item_id] = item
    return gt_map


def _extract_id(item: dict, id_field: str = "id") -> str:
    if id_field in item and item[id_field]:
        return str(item[id_field])
    for key in ("wav", "wav_path", "video_path"):
        if key in item:
            return Path(item[key]).stem
    return ""


# ─── 生成音频目录扫描 ─────────────────────────────────────────────────────────

def load_id_list(path: str) -> Optional[set]:
    """从文件加载 ID 集合（每行一个 ID）。"""
    if not path:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def get_gen_audio_map(audio_dir: str, id_set: set = None) -> Dict[str, Path]:
    """
    扫描目录下所有 .wav 文件，返回 {stem_id: Path}。
    如果提供 id_set，只保留匹配的文件。
    """
    audio_dir = Path(audio_dir)
    result = {p.stem: p for p in sorted(audio_dir.glob("*.wav"))}
    if id_set:
        result = {k: v for k, v in result.items() if k in id_set}
    return result


def match_gen_gt(
    gen_map: Dict[str, Path],
    gt_map: Dict[str, dict],
) -> List[Tuple[str, Path, dict]]:
    """
    按 ID 匹配生成音频和 GT 元信息。
    返回 [(id, gen_wav_path, gt_item), ...]，只保留双方都有的 ID。
    """
    common_ids = sorted(set(gen_map.keys()) & set(gt_map.keys()))
    missing_gen = set(gt_map.keys()) - set(gen_map.keys())
    missing_gt = set(gen_map.keys()) - set(gt_map.keys())

    if missing_gen:
        print(f"  [WARNING] {len(missing_gen)} GT items have no generated audio")
    if missing_gt:
        print(f"  [WARNING] {len(missing_gt)} generated files have no GT entry")
    print(f"  Matched {len(common_ids)} samples")

    return [(sid, gen_map[sid], gt_map[sid]) for sid in common_ids]


# ─── 缓存工具 ─────────────────────────────────────────────────────────────────

def cache_path(cache_dir: str, name: str) -> Path:
    return Path(cache_dir) / name


def load_cache(cache_dir: str, name: str) -> Optional[object]:
    p = cache_path(cache_dir, name)
    if p.exists():
        if p.suffix == ".pt":
            return torch.load(p, weights_only=True, map_location="cpu")
        elif p.suffix == ".pkl":
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def save_cache(cache_dir: str, name: str, obj) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path(cache_dir, name)
    if p.suffix == ".pt":
        torch.save(obj, p)
    elif p.suffix == ".pkl":
        with open(p, "wb") as f:
            pickle.dump(obj, f)
    print(f"  [cache] Saved → {p}")


# ─── 音频加载 ─────────────────────────────────────────────────────────────────

def load_audio_16k(path, dur: float = None) -> np.ndarray:
    """加载音频，重采样到 16kHz，返回 float32 numpy array。若指定 dur 则截取前 dur 秒。"""
    import librosa
    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    if dur is not None:
        n = int(dur * 16000)
        if n < len(audio):
            audio = audio[:n]
    return audio


def build_dur_map(jsonl_path: str, id_set: set = None) -> Dict[str, float]:
    """从 JSONL 读取 dur 映射 {id: dur_seconds}，仅包含有 dur 字段的条目。"""
    dur_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sid = str(item.get("id", ""))
            if not sid:
                continue
            if id_set and sid not in id_set:
                continue
            if "dur" in item and item["dur"] is not None:
                dur_map[sid] = float(item["dur"])
    return dur_map


# ─── 打印汇总 ─────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """打印所有指标汇总表格。"""
    col_w = 18
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric:<{col_w}}: {value:.4f}")
        else:
            print(f"  {metric:<{col_w}}: {value}")
    print("=" * 60)
