#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Gemini API 对生成的 WAV 进行多说话人转写，输出 multispk_trans
支持多轮自动重试、断点续传、防重复写入、进程锁防并发
"""

import os
import requests
import argparse
import random
import base64
import subprocess
import shutil
from multiprocessing import Pool, Lock, Manager
import time
import json
import glob
import fcntl
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# API 配置 (fill in your own credentials)
# key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# location = "http://your.endpoint:port/v1/chat/completions"
key = None
location = None
if key is None or location is None:
    raise ValueError(
        "Please fill in `key` and `location` above with your own "
        "OpenAI-compatible API credentials before running this script."
    )

# 全局文件写入锁
file_lock = Lock()

PROMPT = """You are an expert speaker diarization and transcription system. Your primary job is to accurately identify WHO is speaking at every moment and transcribe their words.

## Task
1. First, identify all distinct speakers by carefully analyzing voice characteristics: pitch, timbre, accent, speaking pace, vocal quality, gender, and age.
2. Then, transcribe the entire audio chronologically, creating a new turn entry every time the speaker changes.
3. Output ONLY a valid JSON array.

## Critical Rules for Speaker Identification
- Pay extremely close attention to voice characteristics. Even subtle changes in pitch, timbre, or speaking style indicate a different speaker.
- Do NOT group speech from different voices into the same turn. If the voice changes, you MUST start a new turn with the correct speaker label.
- When in doubt, listen again — accuracy of speaker assignment is more important than merging turns.
- Common patterns to watch for: Speaker A says something, then Speaker B responds, then Speaker A comes back. Do not accidentally attribute Speaker A's return to Speaker B.

## Output Format
- Label speakers as "spk1", "spk2", ... in the order they first appear.
- Each turn: {"speaker": "spk1", "text": "what they said"}
- A speaker can appear multiple times across different turns.
- Preserve exact temporal order.
- Punctuation is fine. No timestamps. No markdown. No explanation.
- Output ONLY the JSON array.

## Examples

Example 1 — Two speakers alternating:

[{"speaker": "spk1", "text": "How are you doing today? I haven't seen you in a while."}, {"speaker": "spk2", "text": "I'm doing great, thanks for asking."}, {"speaker": "spk1", "text": "So what have you been working on?"}, {"speaker": "spk2", "text": "Mostly the new project launch."}]

Example 2 — Single speaker:

[{"speaker": "spk1", "text": "Welcome back everyone. Today we're going to talk about something really exciting."}]

Example 3 — Speaker A speaks, then B, then A returns:

[{"speaker": "spk1", "text": "Let me explain the background first."}, {"speaker": "spk2", "text": "Sure, go ahead."}, {"speaker": "spk1", "text": "So the issue started last week when we deployed the update."}]

## Now transcribe the provided audio. Focus on getting the speaker identity RIGHT for every segment:
"""


def read_audio_as_base64(audio_path: str, tmpdir: str, process_id: int):
    """读取音频文件并转换为 base64（16kHz mono WAV）"""
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    unique_wav_name = f"{audio_name}_{process_id}_{int(time.time() * 1000000)}.wav"
    wav_save_path = os.path.join(tmpdir, unique_wav_name)

    try:
        cmd = f'ffmpeg -v error -i "{audio_path}" -f wav -ac 1 -ar 16000 "{wav_save_path}"'
        subprocess.run(cmd, shell=True, check=True)
        with open(wav_save_path, "rb") as audio_file:
            base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        return base64_audio, wav_save_path
    except Exception as e:
        raise Exception(f"音频处理失败: {e}")


def call_gemini(base64_audio, model="gemini-2.5-pro", seed=0):
    """调用 Gemini API 进行多说话人转写"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    content_list = [
        {"type": "text", "text": PROMPT},
        {
            "type": "input_audio",
            "input_audio": {
                "data": f"data:audio/wav;base64,{base64_audio}",
                "format": "wav"
            }
        }
    ]

    params = {
        "model": model,
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 4096,
        "temperature": 0.3,
        "seed": seed
    }

    max_retries = 10
    for retry in range(max_retries):
        try:
            with requests.Session() as session:
                session.headers.update(headers)
                resp = session.post(location, json=params, timeout=120)
                data = resp.json()

                if "error" in data:
                    print(f"  API错误: {data['error']}, 重试 {retry+1}/{max_retries}")
                    time.sleep(1)
                    continue

                input_tokens = data['usage'].get('prompt_tokens', 0)
                output_tokens = data['usage'].get('completion_tokens', 0)
                print(f"  Input={input_tokens}, Output={output_tokens}")

                text = data["choices"][0]["message"]["content"].strip()
                if not text:
                    print(f"  空结果，重试...")
                    params["seed"] = random.randint(0, 99999)
                    continue

                # 去掉可能的 markdown 包装
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                    if text.startswith("json"):
                        text = text[4:].strip()

                result = json.loads(text)
                return result

        except json.JSONDecodeError as e:
            print(f"  JSON解析失败: {e}, 原文: {text[:200]}")
            params["seed"] = random.randint(0, 99999)
            continue
        except Exception as e:
            print(f"  请求失败: {e}, 重试 {retry+1}/{max_retries}")
            time.sleep(1)
            continue

    return None


def merge_turns_by_speaker(turns):
    """将对话轮次列表合并为按 speaker 拼接的 dict
    [{"speaker":"spk1","text":"a"},{"speaker":"spk2","text":"b"},{"speaker":"spk1","text":"c"}]
    -> {"spk1": "a c", "spk2": "b"}
    """
    merged = {}
    for turn in turns:
        spk = turn.get("speaker", "spk1")
        text = turn.get("text", "").strip()
        if not text:
            continue
        if spk in merged:
            merged[spk] = merged[spk] + " " + text
        else:
            merged[spk] = text
    return merged


def process_single_item(args):
    """处理单个 WAV 文件，使用共享 completed_ids 防止重复写入"""
    wav_path, process_id, tmpdir_base, output_file, gemini_model, completed_ids = args

    basename = os.path.splitext(os.path.basename(wav_path))[0]

    # 双重检查：写入前再确认没被其他进程完成
    if basename in completed_ids:
        print(f"[进程 {process_id}] 跳过（已完成）: {basename}")
        return

    try:
        os.makedirs(tmpdir_base, exist_ok=True)
        print(f"\n[进程 {process_id}] 处理: {basename}")

        wav_save_path = None
        try:
            base64_audio, wav_save_path = read_audio_as_base64(wav_path, tmpdir_base, process_id)
        except Exception as e:
            print(f"[进程 {process_id}] 音频读取失败: {e}")
            return
        finally:
            if wav_save_path and os.path.exists(wav_save_path):
                os.remove(wav_save_path)

        seed = random.randint(0, 99999)
        result = call_gemini(base64_audio, model=gemini_model, seed=seed)

        if result is None:
            print(f"[进程 {process_id}] 转写失败: {basename}")
            return

        # 后处理：将 list of turns 合并为 {"spk1": "all text", "spk2": "all text"}
        if isinstance(result, list):
            merged = merge_turns_by_speaker(result)
        else:
            merged = result

        entry = {
            "id": basename,
            "wav": wav_path,
            "multispk_trans": merged
        }

        # 加锁写入，防止重复
        with file_lock:
            # 写入前最终检查
            if basename in completed_ids:
                print(f"[进程 {process_id}] 跳过写入（已被其他进程完成）: {basename}")
                return
            completed_ids[basename] = True
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"[进程 {process_id}] 完成: {basename} -> {merged}")

    except Exception as e:
        print(f"[进程 {process_id}] 错误: {e}")


def dedup_jsonl(filepath):
    """对 JSONL 文件去重（按 id），保留每个 id 的第一次出现"""
    if not os.path.exists(filepath):
        return 0
    seen = set()
    unique_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item_id = item['id']
                if item_id not in seen:
                    seen.add(item_id)
                    unique_lines.append(line)
            except:
                pass
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    return len(seen)


def read_existing_ids(filepath):
    """读取已完成的 ID 集合"""
    ids = set()
    if not os.path.exists(filepath):
        return ids
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)['id'])
                except:
                    pass
    return ids


class ProcessLock:
    """文件锁，防止同一个输出文件被多个脚本实例并发写入"""

    def __init__(self, lockfile):
        self.lockfile = lockfile
        self.fd = None

    def acquire(self):
        self.fd = open(self.lockfile, 'w')
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            return True
        except IOError:
            self.fd.close()
            self.fd = None
            return False

    def release(self):
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
            try:
                os.remove(self.lockfile)
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Gemini 多说话人转写")
    parser.add_argument('--input_dir', type=str, required=True, help='输入 WAV 目录')
    parser.add_argument('--output_file', type=str, default=None, help='输出 JSONL 路径（默认：输入目录的父目录/multispk_trans.jsonl）')
    parser.add_argument('--num_processes', type=int, default=256, help='并行进程数')
    parser.add_argument('--gemini_model', type=str, default="gemini-2.5-pro", help='Gemini 模型')
    parser.add_argument('--max_samples', type=int, default=None, help='最大处理数（测试用）')
    parser.add_argument('--tmpdir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_multispk"), help='临时目录')
    parser.add_argument('--max_rounds', type=int, default=3, help='最大重试轮数（断点续传）')
    parser.add_argument('--id_list', type=str, default=None, help='只处理此文件中列出的 ID（每行一个，不含 .wav 后缀）')

    args = parser.parse_args()

    input_dir = args.input_dir
    if args.output_file:
        output_file = args.output_file
    else:
        parent_dir = os.path.dirname(input_dir.rstrip('/'))
        output_file = os.path.join(parent_dir, "multispk_trans.jsonl")

    tmpdir = args.tmpdir

    # === 进程锁：防止并发实例 ===
    lockfile = output_file + ".lock"
    proc_lock = ProcessLock(lockfile)
    if not proc_lock.acquire():
        print(f"错误：另一个实例正在处理 {output_file}，退出。")
        print(f"如果确认没有其他实例在运行，请删除锁文件: {lockfile}")
        sys.exit(1)

    try:
        _run(args, input_dir, output_file, tmpdir)
    finally:
        proc_lock.release()


def _run(args, input_dir, output_file, tmpdir):
    # 收集 WAV 文件（支持 --id_list 过滤）
    all_wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if args.id_list:
        with open(args.id_list) as f:
            valid_ids = {line.strip() for line in f if line.strip()}
        all_wav_files = [w for w in all_wav_files if os.path.splitext(os.path.basename(w))[0] in valid_ids]
        print(f"--id_list 过滤: {len(valid_ids)} IDs → {len(all_wav_files)} WAV 文件")
    total_count = len(all_wav_files)
    if args.max_samples:
        all_wav_files = all_wav_files[:args.max_samples]
        total_count = len(all_wav_files)
    print(f"找到 {total_count} 个 WAV 文件")
    print(f"输出文件: {output_file}")

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # === 启动前先去重（处理上次异常退出留下的重复） ===
    if os.path.exists(output_file):
        before = sum(1 for _ in open(output_file))
        after = dedup_jsonl(output_file)
        if before != after:
            print(f"去重: {before} 行 -> {after} 行")

    # 构建全部目标 ID 集合
    all_target_ids = set(os.path.splitext(os.path.basename(w))[0] for w in all_wav_files)

    # === 多轮重试 ===
    for round_idx in range(1, args.max_rounds + 1):
        existing_ids = read_existing_ids(output_file)
        pending = [w for w in all_wav_files
                   if os.path.splitext(os.path.basename(w))[0] not in existing_ids]

        print(f"\n{'='*60}")
        print(f"第 {round_idx}/{args.max_rounds} 轮: 已完成 {len(existing_ids)}/{total_count}, 待处理 {len(pending)}")
        print(f"{'='*60}")

        if not pending:
            print("全部完成！无需继续。")
            break

        # 清理临时目录
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)

        # 使用 Manager.dict 作为共享的已完成集合，防止同一轮内重复写入
        manager = Manager()
        completed_ids = manager.dict()
        for eid in existing_ids:
            completed_ids[eid] = True

        # 准备参数（多传一个 completed_ids）
        process_args = [
            (wav, i % args.num_processes, tmpdir, output_file, args.gemini_model, completed_ids)
            for i, wav in enumerate(pending)
        ]

        num_workers = min(args.num_processes, len(pending))
        print(f"启动 {num_workers} 个进程，模型: {args.gemini_model}")

        with Pool(processes=num_workers) as pool:
            pool.map(process_single_item, process_args)

        # 清理临时目录
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

        # 轮次结束后去重（防御性）
        done_count = dedup_jsonl(output_file)
        print(f"\n第 {round_idx} 轮结束: 已完成 {done_count}/{total_count}")

        if done_count >= total_count:
            print("全部完成！")
            break
    else:
        final_done = len(read_existing_ids(output_file))
        remaining = total_count - final_done
        print(f"\n{args.max_rounds} 轮结束，仍有 {remaining} 个未完成。")

    print(f"\n结果写入: {output_file}")


if __name__ == "__main__":
    main()
