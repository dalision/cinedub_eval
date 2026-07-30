import os
#!/usr/bin/env python3
"""
Synchformer 评估脚本（AV DeSync 分数）
  - GT 视频特征：按 dataset 缓存到 {cache_dir}/synchformer_video.pth
  - 生成音频特征：每次实时提取（支持 dur 裁剪）
  - DeSync 计算：Synchformer compare_v_a，越低越好

用法：
  python eval_syncformer.py --gen_dir /path/to/gen_wavs --gt_jsonl /path/to/gt.jsonl \
      --cache_dir /path/to/cache/syncformer_chem --id_list /path/to/ids.txt

依赖：
  av_bench (av-benchmark)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from tqdm import tqdm

AV_BENCH_ROOT = os.environ.get("AV_BENCHMARK_ROOT", os.path.join(os.path.dirname(__file__), "thirdparty", "av-benchmark"))
sys.path.insert(0, AV_BENCH_ROOT)
from av_bench.synchformer.synchformer import Synchformer, make_class_grid

_syncformer_ckpt_path = Path(AV_BENCH_ROOT) / 'weights' / 'synchformer_state_dict.pth'

# ─── 视频特征提取参数 ─────────────────────────────────────────────────────────
_SYNC_SIZE = 224
_SYNC_FPS = 25.0
V_SEGMENT_SIZE = 16    # 16 frames @ 25fps = 0.64s
V_STEP_SIZE = 8        # 8 frames = 0.32s
MIN_VIDEO_DURATION = V_SEGMENT_SIZE / _SYNC_FPS  # 0.64s

# ─── 音频特征提取参数 ─────────────────────────────────────────────────────────
A_SEGMENT_SIZE = 10240  # samples @ 16kHz
A_STEP_SIZE = 5120      # samples
A_SR = 16000

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  GT 视频特征提取（带缓存）
# ═══════════════════════════════════════════════════════════════════════════════

def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                dur = float(stream.get('duration', 0))
                if dur > 0:
                    return dur
        result2 = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        data2 = json.loads(result2.stdout)
        return float(data2.get('format', {}).get('duration', 0))
    except Exception:
        return 0.0


class SyncVideoDataset(Dataset):
    def __init__(self, items: list, duration_sec: float):
        self.items = items
        self.n_frames = max(V_SEGMENT_SIZE, int(_SYNC_FPS * duration_sec))
        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, idx):
        try:
            vid_id, video_path = self.items[idx]
            reader = StreamingMediaDecoder(video_path)
            reader.add_basic_video_stream(
                frames_per_chunk=self.n_frames,
                frame_rate=_SYNC_FPS, format='rgb24')
            reader.fill_buffer()
            chunk = reader.pop_chunks()[0]
            if chunk is None or chunk.shape[0] < V_SEGMENT_SIZE:
                return None
            chunk = self.sync_transform(chunk)
            return {'id': vid_id, 'sync_video': chunk}
        except Exception as e:
            log.warning(f'Video load error {self.items[idx][0]}: {e}')
            return None

    def __len__(self):
        return len(self.items)


def _collate_video(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    min_t = min(x['sync_video'].shape[0] for x in batch)
    min_t = max(min_t, V_SEGMENT_SIZE)
    for x in batch:
        t = x['sync_video'].shape[0]
        if t < min_t:
            pad = x['sync_video'][-1:].expand(min_t - t, -1, -1, -1)
            x['sync_video'] = torch.cat([x['sync_video'], pad], dim=0)
        else:
            x['sync_video'] = x['sync_video'][:min_t]
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


@torch.inference_mode()
def encode_video_with_sync(synchformer, x: torch.Tensor) -> torch.Tensor:
    """x: (B, T, C, H, W) → (B, S, tv, D)"""
    b, t, c, h, w = x.shape
    num_segments = (t - V_SEGMENT_SIZE) // V_STEP_SIZE + 1
    if num_segments < 1:
        return None
    segments = [x[:, i * V_STEP_SIZE: i * V_STEP_SIZE + V_SEGMENT_SIZE]
                for i in range(num_segments)]
    x = torch.stack(segments, dim=1)
    x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
    x = synchformer.extract_vfeats(x)
    x = rearrange(x, '(b s) 1 t d -> b s t d', b=b)
    return x


def extract_gt_video_features(
    synchformer, gt_map: dict, matched_ids: set, device,
    batch_size=4, num_workers=4, duration_bin=1.0
) -> dict:
    """提取 GT 视频的 Synchformer 特征"""
    # 收集有视频且在 matched_ids 中的样本
    id2video = {}
    for sid in matched_ids:
        if sid not in gt_map:
            continue
        vp = gt_map[sid].get('video_path', '')
        if vp and os.path.isfile(vp):
            id2video[sid] = vp

    if not id2video:
        print("  [WARNING] 没有可用的 GT 视频")
        return {}

    # 按时长分组
    duration_groups = defaultdict(list)
    for vid_id, vp in id2video.items():
        dur = get_video_duration(vp)
        if dur < MIN_VIDEO_DURATION:
            continue
        bin_dur = (int(dur / duration_bin) + 1) * duration_bin
        duration_groups[bin_dur].append((vid_id, vp))

    all_features = {}
    for bin_dur, group_items in sorted(duration_groups.items()):
        print(f"  [Video] {len(group_items)} videos @ {bin_dur:.1f}s")
        dataset = SyncVideoDataset(group_items, duration_sec=bin_dur)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_collate_video, pin_memory=True)
        for batch in tqdm(loader, desc=f'video {bin_dur:.1f}s', leave=False):
            if batch is None:
                continue
            ids = batch['id']
            sync_video = batch['sync_video'].to(device)
            feats = encode_video_with_sync(synchformer, sync_video)
            if feats is None:
                continue
            feats = feats.cpu()
            for i, vid_id in enumerate(ids):
                all_features[vid_id] = feats[i]
            del sync_video, feats

    print(f"  [Video] 提取了 {len(all_features)} 个视频特征")
    return all_features


# ═══════════════════════════════════════════════════════════════════════════════
#  生成音频特征提取
# ═══════════════════════════════════════════════════════════════════════════════

def pad_or_truncate(audio: torch.Tensor, max_spec_t: int):
    diff = max_spec_t - audio.shape[-1]
    if diff > 0:
        audio = torch.nn.functional.pad(audio, (0, diff), 'constant', 0.0)
    elif diff < 0:
        audio = audio[..., :max_spec_t]
    return audio


@torch.inference_mode()
def encode_audio_with_sync(synchformer, x: torch.Tensor, mel) -> torch.Tensor:
    """x: (B, T) → (B, S, ta, D)"""
    b, t = x.shape
    num_segments = (t - A_SEGMENT_SIZE) // A_STEP_SIZE + 1
    if num_segments < 1:
        return None
    segments = [x[:, i * A_STEP_SIZE: i * A_STEP_SIZE + A_SEGMENT_SIZE]
                for i in range(num_segments)]
    x = torch.stack(segments, dim=1)  # (B, S, segment_size)
    x = mel(x)
    x = torch.log(x + 1e-6)
    x = pad_or_truncate(x, 66)
    mean, std = -4.2677393, 4.5689974
    x = (x - mean) / (2 * std)
    x = synchformer.extract_afeats(x.unsqueeze(2))
    return x


def extract_gen_audio_features(
    synchformer, gen_map: dict, matched_ids: set, dur_map: dict,
    device, expected_length: int
) -> dict:
    """提取生成音频的 Synchformer 特征（支持 dur 裁剪）"""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=A_SR, n_fft=1024, hop_length=160,
        n_mels=128, f_min=0, f_max=8000
    ).to(device)

    resampler_cache = {}
    all_features = {}

    items = [(sid, gen_map[sid]) for sid in sorted(matched_ids) if sid in gen_map]
    for sid, wav_path in tqdm(items, desc='audio'):
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.mean(dim=0)  # mono
            if sr != A_SR:
                if sr not in resampler_cache:
                    resampler_cache[sr] = torchaudio.transforms.Resample(sr, A_SR)
                waveform = resampler_cache[sr](waveform)

            # dur 裁剪
            dur = dur_map.get(sid)
            if dur is not None:
                n = int(dur * A_SR)
                if n < len(waveform):
                    waveform = waveform[:n]

            # 截断到 expected_length
            waveform = waveform[:expected_length]
            if waveform.shape[0] < A_SEGMENT_SIZE:
                log.warning(f'  [SKIP] {sid}: 音频太短 ({waveform.shape[0]} < {A_SEGMENT_SIZE})')
                continue

            x = waveform.unsqueeze(0).to(device)  # (1, T)
            feat = encode_audio_with_sync(synchformer, x, mel)
            if feat is not None:
                all_features[sid] = feat[0].cpu()
        except Exception as e:
            log.warning(f'  [SKIP] {sid}: {e}')

    print(f"  [Audio] 提取了 {len(all_features)} 个音频特征")
    return all_features


# ═══════════════════════════════════════════════════════════════════════════════
#  DeSync 计算
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_to_segments(feats, target):
    """将特征 (S, t, D) 零填充到 target 段"""
    s = feats.shape[0]
    if s >= target:
        return feats
    pad = torch.zeros(target - s, *feats.shape[1:],
                      dtype=feats.dtype, device=feats.device)
    return torch.cat([feats, pad], dim=0)


# 2秒 ≈ 5段 (视频: (50-16)/8+1=5, 音频: (32000-10240)/5120+1=5)
MIN_SEGMENTS = 5


@torch.inference_mode()
def compute_desync(synchformer, video_feats, audio_feats, device, full_segments=14):
    """
    计算 DeSync 分数。
    video_feats: (S_v, tv, D)
    audio_feats: (S_a, ta, D)
    - >= 14 段：前14段 + 后14段取平均（原始 pos_emb）
    - 5~13 段：零填充到 14 段后计算（≈2s 以上）
    - < 5 段：跳过（<2s，太短）
    """
    sync_grid = make_class_grid(-2, 2, 21)

    s_v, s_a = video_feats.shape[0], audio_feats.shape[0]
    n = min(s_v, s_a)
    if n < MIN_SEGMENTS:
        return None

    if n >= full_segments:
        # 足够长：前14段 + 后14段，使用原始 pos_emb
        v = video_feats.unsqueeze(0).to(device)
        a = audio_feats.unsqueeze(0).to(device)

        logits1 = synchformer.compare_v_a(v[:, :full_segments], a[:, :full_segments])
        top1 = torch.argmax(logits1, dim=-1).cpu().numpy()[0]
        desync1 = abs(sync_grid[top1].item())

        logits2 = synchformer.compare_v_a(v[:, -full_segments:], a[:, -full_segments:])
        top2 = torch.argmax(logits2, dim=-1).cpu().numpy()[0]
        desync2 = abs(sync_grid[top2].item())

        return (desync1 + desync2) / 2.0
    else:
        # 5~13段：零填充到 14 段
        v = _pad_to_segments(video_feats[:n], full_segments).unsqueeze(0).to(device)
        a = _pad_to_segments(audio_feats[:n], full_segments).unsqueeze(0).to(device)

        logits = synchformer.compare_v_a(v, a)
        top = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        desync = abs(sync_grid[top].item())
        return desync


# ═══════════════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音频目录")
    parser.add_argument("--gt_jsonl", type=str, required=True, help="GT JSONL 文件")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="GT 视频特征缓存目录（默认 gen_dir/../metric_cache/syncformer）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--id_list", type=str, default=None, help="ID 过滤文件")
    parser.add_argument("--batch_size", type=int, default=4, help="视频特征提取 batch size")
    parser.add_argument("--audio_length", type=float, default=10.0,
                        help="音频最大长度（秒），默认 10.0")
    parser.add_argument("--force_gt_cache", action="store_true",
                        help="强制重新提取 GT 视频特征")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 缓存路径
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.gen_dir).parent / "metric_cache" / "syncformer")
    os.makedirs(args.cache_dir, exist_ok=True)
    gt_cache_path = Path(args.cache_dir) / "synchformer_video.pth"

    # 加载数据 — 确保同目录的 utils.py 优先于 av-benchmark 的 utils.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import build_dur_map, build_gt_map, get_gen_audio_map, load_id_list, match_gen_gt
    id_set = load_id_list(args.id_list)
    gt_map = build_gt_map(args.gt_jsonl)
    gen_map = get_gen_audio_map(args.gen_dir, id_set)
    matched = match_gen_gt(gen_map, gt_map)
    if not matched:
        print("ERROR: 无匹配样本")
        sys.exit(1)

    matched_ids = {sid for sid, _, _ in matched}
    dur_map = build_dur_map(args.gt_jsonl)

    # 加载 Synchformer 模型
    print(f"Loading Synchformer from {_syncformer_ckpt_path} ...")
    sync_model = Synchformer().to(device).eval()
    sd = torch.load(_syncformer_ckpt_path, weights_only=True)
    sync_model.load_state_dict(sd)
    print("Synchformer loaded")

    # ── GT 视频特征（带缓存）──
    gt_video_feats = None
    if not args.force_gt_cache and gt_cache_path.exists():
        print(f"[GT Cache] 加载 {gt_cache_path}")
        gt_video_feats = torch.load(gt_cache_path, weights_only=True)
        print(f"[GT Cache] 已加载 {len(gt_video_feats)} 条视频特征")

    if gt_video_feats is None:
        print("\n[GT] 提取 GT 视频 Synchformer 特征 ...")
        gt_video_feats = extract_gt_video_features(
            sync_model, gt_map, matched_ids, device,
            batch_size=args.batch_size, num_workers=4)
        torch.save(gt_video_feats, gt_cache_path)
        print(f"  [cache] Saved → {gt_cache_path}")

    # ── 生成音频特征 ──
    print("\n[GEN] 提取生成音频 Synchformer 特征 ...")
    expected_length = int(args.audio_length * A_SR)
    gen_audio_feats = extract_gen_audio_features(
        sync_model, gen_map, matched_ids, dur_map, device, expected_length)

    # ── 计算 DeSync ──
    print("\n[DeSync] 计算 AV 同步分数 ...")
    desync_scores = []
    errors = []
    for sid in sorted(matched_ids):
        if sid not in gt_video_feats or sid not in gen_audio_feats:
            continue
        score = compute_desync(sync_model, gt_video_feats[sid], gen_audio_feats[sid], device)
        if score is not None:
            desync_scores.append(score)
        else:
            errors.append(sid)

    if errors:
        print(f"  [WARNING] {len(errors)} 样本无法计算 DeSync")

    if not desync_scores:
        print("ERROR: 所有样本计算失败")
        sys.exit(1)

    mean_desync = float(np.mean(desync_scores))
    print(f"\n[Synchformer] {len(desync_scores)} samples evaluated")
    print(f"  AV DeSync (越低越好) : {mean_desync:.4f}")
    print(f"  Min / Max            : {min(desync_scores):.4f} / {max(desync_scores):.4f}")
    return mean_desync


if __name__ == "__main__":
    main()
