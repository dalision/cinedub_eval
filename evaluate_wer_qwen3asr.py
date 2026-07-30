import os
#!/usr/bin/env python3
"""
Evaluate WER/CER for generated speech using Qwen3-ASR.
Supports multi-GPU inference via accelerate/torchrun.

Modes:
  transcribe  - Transcribe wav files, save results to jsonl
  evaluate    - Compute WER/CER from existing transcription jsonl
  both        - Transcribe then evaluate

Usage:
  # Single GPU
  python evaluate_wer_qwen3asr.py --audio_dir /path/to/wavs --gt_jsonl /path/to/gt.jsonl --mode both

  # Multi-GPU with torchrun
  torchrun --nproc_per_node=8 evaluate_wer_qwen3asr.py --audio_dir /path/to/wavs --gt_jsonl /path/to/gt.jsonl --mode both

  # Specify language
  torchrun --nproc_per_node=4 evaluate_wer_qwen3asr.py --audio_dir /path/to/wavs --gt_jsonl /path/to/gt.jsonl --mode both --language English
"""

import argparse
import json
import os
import re
import string
import sys
from pathlib import Path

import torch
import jiwer
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from word2number import w2n
    HAS_W2N = True
except ImportError:
    HAS_W2N = False

DEFAULT_MODEL_PATH = os.environ.get("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")


# ─── Dur map ───────────────────────────────────────────────────────

def _build_dur_map(jsonl_path: str) -> dict:
    """从 JSONL 读取 {id: dur_seconds}，仅包含有 dur 字段的条目。"""
    dur_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sid = str(item.get("id", ""))
            if sid and "dur" in item and item["dur"] is not None:
                dur_map[sid] = float(item["dur"])
    return dur_map


# ─── Text normalization ────────────────────────────────────────────

def normalize_numbers(text: str) -> str:
    if not HAS_W2N:
        return text
    words = text.lower().split()
    i = 0
    result = []
    while i < len(words):
        matched = False
        for j in range(min(7, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + j])
            try:
                number = w2n.word_to_num(phrase)
                result.append(str(number))
                i += j
                matched = True
                break
            except Exception:
                continue
        if not matched:
            result.append(words[i])
            i += 1
    return " ".join(result)


def normalize_text(text: str) -> str:
    for ch in string.punctuation:
        text = text.replace(ch, "")
    text = text.lower()
    text = normalize_numbers(text)
    return text


# ─── GRID-specific normalization ────────────────────────────────────

_DIGIT_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_GRID_CMD_MAP = {
    "been": "bin", "ben": "bin", "bing": "bin", "being": "bin",
    "bean": "bin", "band": "bin", "bend": "bin", "win": "bin",
    "plays": "place", "play": "place", "placed": "place", "played": "place",
    "placing": "place", "plains": "place",
}
_GRID_LETTER_MAP = {
    "you": "u", "ewe": "u", "hay": "h", "aitch": "h",
    "zay": "z", "zee": "z", "our": "r", "are": "r",
    "why": "y", "wye": "y", "ex": "x", "jay": "j", "kay": "k",
    "ay": "a", "sea": "c", "see": "c", "tee": "t", "tea": "t",
    "gee": "g", "pee": "p", "dee": "d", "ef": "f", "el": "l", "vee": "v",
}
_GRID_NUM_MAP = {
    "too": "two", "to": "two",
    "for": "four", "fore": "four",
    "won": "one",
}


def normalize_text_grid(text: str) -> str:
    """GRID-specific normalization: base + cmd/letter/number correction."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"([a-z])(\d)", lambda m: m.group(1) + " " + _DIGIT_MAP[m.group(2)], text)
    text = re.sub(r"(\d)([a-z])", lambda m: _DIGIT_MAP[m.group(1)] + " " + m.group(2), text)
    text = re.sub(r"\b(\d)\b", lambda m: _DIGIT_MAP[m.group(1)], text)
    words = text.split()
    return " ".join(words)


def _detect_dataset(gt_jsonl: str) -> str:
    try:
        with open(gt_jsonl, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        return first.get("dataset", "").lower()
    except Exception:
        return ""


def extract_speech_text(text: str) -> str:
    """Extract text between <S> and <E> tags, concatenate multiple segments."""
    segments = re.findall(r"<S>(.*?)<E>", text)
    if segments:
        return " ".join(seg.strip() for seg in segments)
    return text


def get_id_from_item(item: dict) -> str:
    """Extract ID from jsonl item. Use 'id' field if present, otherwise derive from wav_path."""
    if "id" in item:
        return item["id"]
    for key in ("wav_path", "video_path"):
        if key in item:
            return Path(item[key]).stem
    return ""


# ─── Distributed helpers ───────────────────────────────────────────

def get_rank():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


# ─── Core functions ────────────────────────────────────────────────

def load_qwen3_asr(model_path: str, device):
    from qwen_asr import Qwen3ASRModel

    print(f"[Rank {get_rank()}] Loading Qwen3-ASR from {model_path} on {device}")
    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        max_new_tokens=256,
    )
    return asr


def transcribe_folder(asr, audio_dir: str, language: str = None, batch_size: int = 8, id_set: set = None, dur_map: dict = None) -> list[dict]:
    """Transcribe wav files assigned to this rank. Returns list of {id, wav_path, transcription, language}."""
    import numpy as np
    import soundfile as sf

    audio_dir = Path(audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))
    if id_set:
        wav_files = [p for p in wav_files if p.stem in id_set]
    if not wav_files:
        print(f"No wav files found in {audio_dir}")
        sys.exit(1)

    rank = get_rank()
    world_size = get_world_size()
    wav_files_rank = wav_files[rank::world_size]
    print(f"[Rank {rank}/{world_size}] Transcribing {len(wav_files_rank)}/{len(wav_files)} files "
          f"(batch_size={batch_size}, language={language or 'auto'})")

    results = []
    for i in tqdm(range(0, len(wav_files_rank), batch_size),
                  desc=f"Transcribing [rank {rank}]", disable=(rank != 0)):
        batch_paths = wav_files_rank[i:i + batch_size]
        batch_audio = []
        for wav_path in batch_paths:
            wav, sr = sf.read(str(wav_path), dtype="float32")
            # dur 裁剪
            if dur_map is not None:
                dur = dur_map.get(wav_path.stem)
                if dur is not None:
                    wav = wav[:int(dur * sr)]
            batch_audio.append((np.asarray(wav, dtype=np.float32), int(sr)))

        lang_arg = [language] * len(batch_audio) if language else None
        transcriptions = asr.transcribe(
            audio=batch_audio,
            language=lang_arg,
            return_time_stamps=False,
        )

        for wav_path, trans in zip(batch_paths, transcriptions):
            results.append({
                "id": wav_path.stem,
                "wav_path": str(wav_path),
                "transcription": trans.text,
                "language": trans.language,
            })

    return results


def gather_results(local_results: list[dict]) -> list[dict]:
    """Gather results from all ranks to rank 0."""
    if not torch.distributed.is_initialized():
        return local_results
    from torch.distributed import all_gather_object
    gathered = [None for _ in range(get_world_size())]
    all_gather_object(gathered, local_results)
    if get_rank() == 0:
        all_results = [item for sublist in gathered for item in sublist]
        return all_results
    return []


def save_transcriptions(results: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} transcriptions to {output_path}")


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_wer_cer(gt_jsonl: str, trans_jsonl: str, key: str = "transcription"):
    """Compute WER and CER between ground truth and transcription jsonl."""
    gt_data = load_jsonl(gt_jsonl)
    trans_data = load_jsonl(trans_jsonl)

    dataset = _detect_dataset(gt_jsonl)
    if dataset == "grid":
        norm_fn = normalize_text_grid
        print(f"[Dataset=grid] Using GRID-specific normalization (base+letter+number)")
    else:
        norm_fn = normalize_text

    gt_map = {get_id_from_item(item): extract_speech_text(item[key]) for item in gt_data}
    trans_map = {get_id_from_item(item): item["transcription"] for item in trans_data}

    common_ids = sorted(set(gt_map.keys()) & set(trans_map.keys()))
    if not common_ids:
        print("ERROR: No matching IDs between gt and transcription jsonl.")
        sys.exit(1)

    missing_gt = set(trans_map.keys()) - set(gt_map.keys())
    missing_trans = set(gt_map.keys()) - set(trans_map.keys())
    if missing_gt:
        print(f"Warning: {len(missing_gt)} transcribed items have no GT")
    if missing_trans:
        print(f"Warning: {len(missing_trans)} GT items have no transcription")

    references = []
    hypotheses = []
    per_sample = []

    for sample_id in common_ids:
        ref = norm_fn(gt_map[sample_id].strip())
        hyp = norm_fn(trans_map[sample_id].strip())

        if not ref or not hyp:
            print(f"[SKIP] Empty ref/hyp for {sample_id}")
            continue

        references.append(ref)
        hypotheses.append(hyp)

        sample_wer = jiwer.wer(ref, hyp)
        sample_cer = jiwer.cer(ref, hyp)
        per_sample.append({
            "id": sample_id,
            "ref": ref,
            "hyp": hyp,
            "wer": sample_wer,
            "cer": sample_cer,
        })

    corpus_wer = jiwer.wer(references, hypotheses)
    corpus_cer = jiwer.cer(references, hypotheses)

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({len(references)} samples)")
    print(f"{'='*60}")
    print(f"  Corpus WER: {corpus_wer:.4f} ({corpus_wer*100:.2f}%)")
    print(f"  Corpus CER: {corpus_cer:.4f} ({corpus_cer*100:.2f}%)")
    print(f"{'='*60}")

    per_sample.sort(key=lambda x: x["wer"], reverse=True)
    print(f"\nTop-10 worst WER samples:")
    for s in per_sample[:10]:
        print(f"  [{s['id']}] WER={s['wer']:.4f} CER={s['cer']:.4f}")
        print(f"    REF: {s['ref']}")
        print(f"    HYP: {s['hyp']}")

    return corpus_wer, corpus_cer


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER/CER with Qwen3-ASR (multi-GPU supported)")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing wav files")
    parser.add_argument("--gt_jsonl", type=str, default=None, help="Ground truth jsonl file")
    parser.add_argument("--key", type=str, default="transcription", help="Key in gt jsonl for ground truth text")
    parser.add_argument("--mode", type=str, choices=["transcribe", "evaluate", "both"], default="both")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for transcription jsonl")
    parser.add_argument("--trans_jsonl", type=str, default=None, help="Transcription results jsonl path")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to Qwen3-ASR model")
    parser.add_argument("--language", type=str, default=None, help="Language hint (e.g. English, Chinese). Default: auto-detect")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for transcription per rank")
    parser.add_argument("--id_list", type=str, default=None, help="File with one ID per line to filter wav files")
    args = parser.parse_args()

    # Init distributed if launched with accelerate/torchrun
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine trans_jsonl path
    if args.trans_jsonl is None:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            args.trans_jsonl = os.path.join(args.output_dir, "transcriptions_qwen3asr.jsonl")
        elif args.audio_dir:
            parent_dir = str(Path(args.audio_dir).parent)
            args.trans_jsonl = os.path.join(parent_dir, "transcriptions_qwen3asr.jsonl")
        else:
            print("ERROR: --trans_jsonl or --output_dir or --audio_dir is required")
            sys.exit(1)

    need_transcribe = args.mode in ("transcribe", "both")
    need_evaluate = args.mode in ("evaluate", "both")

    if need_transcribe:
        if not args.audio_dir:
            print("ERROR: --audio_dir is required for transcribe mode")
            sys.exit(1)
        id_set = None
        if args.id_list:
            with open(args.id_list) as f:
                id_set = {line.strip() for line in f if line.strip()}
            print(f"Filtering to {len(id_set)} IDs from {args.id_list}")
        dur_map = _build_dur_map(args.gt_jsonl) if args.gt_jsonl else None
        asr = load_qwen3_asr(args.model_path, device)
        local_results = transcribe_folder(asr, args.audio_dir, args.language, args.batch_size, id_set, dur_map)
        all_results = gather_results(local_results)

        # Only rank 0 saves
        if get_rank() == 0:
            save_transcriptions(all_results, args.trans_jsonl)

        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Barrier so all ranks wait for save
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if need_evaluate and get_rank() == 0:
        if not args.gt_jsonl:
            print("ERROR: --gt_jsonl is required for evaluate/both mode")
            sys.exit(1)
        if not os.path.exists(args.trans_jsonl):
            print(f"ERROR: Transcription jsonl not found: {args.trans_jsonl}")
            sys.exit(1)
        evaluate_wer_cer(args.gt_jsonl, args.trans_jsonl, args.key)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
