#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 cpWER（Concatenated minimum-Permutation Word Error Rate）
对比 hypothesis 和 reference 的多说话人转写结果
"""

import json
import argparse
from meeteval.wer import cpwer
from meeteval.io import STM, STMLine


def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data[item['id']] = item
    return data


def multispk_dict_to_stm(multispk_dict, recording_id, channel="1"):
    """将 {"spk1": "text1", "spk2": "text2"} 转为 STM 格式"""
    lines = []
    for spk, text in multispk_dict.items():
        if text and text.strip():
            lines.append(STMLine(
                filename=recording_id,
                channel=channel,
                speaker_id=spk,
                begin_time=0,
                end_time=1,
                transcript=text.strip(),
            ))
    return lines


def main():
    parser = argparse.ArgumentParser(description="计算 cpWER")
    parser.add_argument('--hyp', type=str, required=True, help='Hypothesis JSONL（multispk_trans 字段）')
    parser.add_argument('--ref', type=str, required=True, help='Reference JSONL（multispk_transcription 字段）')
    parser.add_argument('--hyp_field', type=str, default='multispk_trans', help='Hypothesis 中的字段名')
    parser.add_argument('--ref_field', type=str, default='multispk_trans', help='Reference 中的字段名')
    parser.add_argument('--verbose', action='store_true', help='打印每条样本的详细结果')
    args = parser.parse_args()

    hyp_data = load_jsonl(args.hyp)
    ref_data = load_jsonl(args.ref)

    # 找到共同的 ID
    common_ids = sorted(set(hyp_data.keys()) & set(ref_data.keys()))
    print(f"Hypothesis 样本数: {len(hyp_data)}")
    print(f"Reference 样本数: {len(ref_data)}")
    print(f"共同 ID 数: {len(common_ids)}")

    if not common_ids:
        print("没有共同的样本 ID！")
        return

    # 逐条计算 cpWER
    total_errors = 0
    total_length = 0
    per_sample_results = []

    for uid in common_ids:
        hyp_item = hyp_data[uid]
        ref_item = ref_data[uid]

        hyp_multispk = hyp_item.get(args.hyp_field, {})
        ref_multispk = ref_item.get(args.ref_field, {})

        if not hyp_multispk or not ref_multispk:
            continue

        # 构建 STM
        ref_lines = multispk_dict_to_stm(ref_multispk, uid)
        hyp_lines = multispk_dict_to_stm(hyp_multispk, uid)

        if not ref_lines or not hyp_lines:
            continue

        ref_stm = STM(ref_lines)
        hyp_stm = STM(hyp_lines)

        try:
            result_dict = cpwer(ref_stm, hyp_stm)
            # cpwer returns dict keyed by filename, get the single result
            result = list(result_dict.values())[0]
            errors = result.errors
            ref_len = result.length
            sample_wer = result.error_rate

            total_errors += errors
            total_length += ref_len

            per_sample_results.append({
                'id': uid,
                'wer': sample_wer,
                'errors': errors,
                'length': ref_len,
                'insertions': result.insertions,
                'deletions': result.deletions,
                'substitutions': result.substitutions,
            })

            if args.verbose:
                print(f"  {uid}: cpWER={sample_wer:.4f} (errors={errors}, len={ref_len}, "
                      f"ins={result.insertions}, del={result.deletions}, sub={result.substitutions})")

        except Exception as e:
            print(f"  {uid}: 计算失败 - {e}")

    # 汇总结果
    if total_length > 0:
        overall_cpwer = total_errors / total_length
    else:
        overall_cpwer = float('inf')

    print(f"\n{'='*60}")
    print(f"总体 cpWER: {overall_cpwer:.4f} ({overall_cpwer*100:.2f}%)")
    print(f"总 errors: {total_errors}")
    print(f"总 ref words: {total_length}")
    print(f"有效样本数: {len(per_sample_results)}")

    total_ins = sum(r['insertions'] for r in per_sample_results)
    total_del = sum(r['deletions'] for r in per_sample_results)
    total_sub = sum(r['substitutions'] for r in per_sample_results)
    print(f"总 insertions: {total_ins}")
    print(f"总 deletions: {total_del}")
    print(f"总 substitutions: {total_sub}")
    print(f"{'='*60}")

    # 按 WER 排序，显示最差的 10 条
    if per_sample_results:
        per_sample_results.sort(key=lambda x: x['wer'], reverse=True)
        print(f"\n最差 10 条样本:")
        for r in per_sample_results[:10]:
            print(f"  {r['id']}: cpWER={r['wer']:.4f} (errors={r['errors']}, len={r['length']})")

        print(f"\n最好 10 条样本:")
        for r in per_sample_results[-10:]:
            print(f"  {r['id']}: cpWER={r['wer']:.4f} (errors={r['errors']}, len={r['length']})")


if __name__ == "__main__":
    main()
