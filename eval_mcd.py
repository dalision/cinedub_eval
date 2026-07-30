#!/usr/bin/env python3
"""
MCD（Mel-Cepstral Distortion）评估脚本
  - 使用 pymcd 库，DTW 和 DTW-SL 两种变体
  - ProcessPoolExecutor 并行计算
  - 无需缓存（计算快，且依赖 GT wav）

用法：
  python eval_mcd.py --gen_dir /path/to/gen_wavs --gt_jsonl /path/to/gt.jsonl
  python eval_mcd.py --gen_dir ... --gt_jsonl ... --num_workers 8
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import build_dur_map, build_gt_map, get_gen_audio_map, match_gen_gt


# ─── 单样本计算（在子进程中运行）────────────────────────────────────────────

def _compute_mcd_single(args_tuple):
    sid, gen_wav, gt_wav, dur = args_tuple
    tmp_path = None
    try:
        gen_wav_str = str(gen_wav)
        # 按 dur 裁剪 gen wav
        if dur is not None:
            import soundfile as sf
            import tempfile
            data, sr = sf.read(gen_wav_str)
            n_samples = int(dur * sr)
            if n_samples < len(data):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, data[:n_samples], sr)
                gen_wav_str = tmp_path

        from pymcd.mcd import Calculate_MCD
        mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
        mcd_dtw = mcd_toolbox.calculate_mcd(gen_wav_str, str(gt_wav))

        mcd_toolbox_sl = Calculate_MCD(MCD_mode="dtw_sl")
        mcd_dtw_sl = mcd_toolbox_sl.calculate_mcd(gen_wav_str, str(gt_wav))

        return sid, mcd_dtw, mcd_dtw_sl, None
    except Exception as e:
        return sid, None, None, str(e)
    finally:
        if tmp_path:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音频目录")
    parser.add_argument("--gt_jsonl", type=str, required=True, help="GT JSONL 文件")
    parser.add_argument("--num_workers", type=int, default=8, help="并行进程数（默认 8）")
    parser.add_argument("--id_list", type=str, default=None, help="ID 过滤文件")
    args = parser.parse_args()

    from utils import load_id_list
    id_set = load_id_list(args.id_list)
    gt_map = build_gt_map(args.gt_jsonl)
    gen_map = get_gen_audio_map(args.gen_dir, id_set)
    matched = match_gen_gt(gen_map, gt_map)
    if not matched:
        print("ERROR: 无匹配样本")
        sys.exit(1)

    dur_map = build_dur_map(args.gt_jsonl)

    # 过滤掉 GT wav 不存在的样本
    tasks = []
    for sid, gen_path, gt_item in matched:
        gt_wav = gt_item.get("wav") or gt_item.get("wav_path")
        if gt_wav and Path(gt_wav).exists():
            tasks.append((sid, gen_path, gt_wav, dur_map.get(sid)))
        else:
            print(f"  [SKIP] {sid}: GT wav 不存在 ({gt_wav})")

    if not tasks:
        print("ERROR: 没有可用样本")
        sys.exit(1)

    print(f"\n[MCD] 计算 {len(tasks)} 对样本（{args.num_workers} workers）...")

    mcd_dtw_list, mcd_sl_list, errors = [], [], []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_compute_mcd_single, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="MCD"):
            sid, mcd_dtw, mcd_sl, err = future.result()
            if err:
                errors.append((sid, err))
            else:
                mcd_dtw_list.append(mcd_dtw)
                mcd_sl_list.append(mcd_sl)

    if errors:
        print(f"\n  [WARNING] {len(errors)} 样本出错：")
        for sid, err in errors[:5]:
            print(f"    [{sid}] {err}")

    if not mcd_dtw_list:
        print("ERROR: 所有样本计算失败")
        sys.exit(1)

    mean_dtw = float(np.mean(mcd_dtw_list))
    mean_sl = float(np.mean(mcd_sl_list))

    print(f"\n[MCD] {len(mcd_dtw_list)} samples evaluated")
    print(f"  MCD (DTW)    : {mean_dtw:.4f} dB")
    print(f"  MCD (DTW-SL) : {mean_sl:.4f} dB")
    return mean_dtw, mean_sl


if __name__ == "__main__":
    main()
