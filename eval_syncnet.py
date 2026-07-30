import os
#!/usr/bin/env python3
"""
SyncNet 评估脚本（LSE-D / LSE-C）
  - 从 GT JSONL 获取 video_path（GT 视频，提供嘴型）
  - 用 ffmpeg 将生成音频替换到 GT 视频的音轨 → 临时合成视频
  - 用 syncnet-python 自动做人脸检测 + crop + 打分
  - 缓存：GT 视频的人脸 crop 帧缓存到 ROI_CACHE_ROOT/{dataset}/{sid}/
    再次评估时跳过 S3FD 检测，只做音频切片 + 打分
  - FULL_FRAME_DATASETS（如 lrs3）：全程跳过 S3FD，直接把 GT 视频全帧
    center-crop + 缩放到 224×224 作为 crop，无需人脸检测

需要：
  pip install syncnet-python
  下载权重：
    weights/sfd_face_remapped.pth  （key 已重映射，原始 sfd_face.pth key 格式不兼容）
    weights/syncnet_v2.model

用法：
  python eval_syncnet.py --gen_dir /path/to/gen_wavs --gt_jsonl /path/to/gt.jsonl \
      --dataset chem

并行模式（多进程）：
  python eval_syncnet.py ... --num_workers 4
"""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import build_gt_map, get_gen_audio_map, match_gen_gt

# ─── ffmpeg 路径（pod 环境里 ffmpeg 不在 PATH，自动 fallback）─────────────────
def _find_ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    candidates = [
        "ffmpeg",
        os.environ.get("FFMPEG", "ffmpeg"),
        "ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "ffmpeg"  # 最终 fallback

FFMPEG_BIN = _find_ffmpeg()

# ─── Worker 级别的全局 pipeline（每个进程只加载一次模型）──────────────────────
_worker_pipeline = None


def _worker_init(gpu_queue, s3fd_weights, syncnet_weights, device_str):
    """ProcessPoolExecutor initializer：每个 worker 进程启动时调用一次，加载模型。"""
    global _worker_pipeline
    gpu_id = gpu_queue.get()  # 每个 worker 从队列取一个专属 GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from syncnet_python import SyncNetPipeline
    _worker_pipeline = SyncNetPipeline(
        s3fd_weights=s3fd_weights,
        syncnet_weights=syncnet_weights,
        device=device_str,
    )

DEFAULT_S3FD = os.environ.get("S3FD_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "sfd_face_remapped.pth"))
DEFAULT_SYNCNET = os.environ.get("SYNCNET_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "syncnet_v2.model"))
ROI_CACHE_ROOT = Path(os.environ.get("ROI_CACHE_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "roi")))

# 这些 dataset 全是正面人脸，直接用全帧作为 crop，跳过 S3FD 人脸检测
FULL_FRAME_DATASETS = {"lrs3"}


# ─── ffmpeg：将生成音频替换到视频音轨 ─────────────────────────────────────────

def mux_audio_to_video(video_path: str, audio_path: str, output_path: str) -> bool:
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


def trim_exclude_interval(input_path: str, output_path: str, edit_interval: list) -> bool:
    """
    用 ffmpeg 裁掉视频中 edit_interval 指定的时间段（音视频同步裁剪）。
    edit_interval: [[s1,e1], [s2,e2], ...] 或 [s1, e1]（单区间简写）。
    """
    # 统一为 [[s,e], ...] 格式
    if edit_interval and not isinstance(edit_interval[0], (list, tuple)):
        intervals = [edit_interval]
    else:
        intervals = edit_interval

    # 构建 select 表达式：排除所有区间
    excludes = "+".join(
        f"between(t\\,{float(s)}\\,{float(e)})" for s, e in intervals
    )
    vf = f"select='not({excludes})',setpts=N/FRAME_RATE/TB"
    af = f"aselect='not({excludes})',asetpts=N/SR/TB"
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vf", vf,
        "-af", af,
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return result.returncode == 0


# ─── ROI 缓存工具 ──────────────────────────────────────────────────────────────

def _has_roi_cache(sid_cache_dir: Path) -> bool:
    """检查是否有有效的 ROI 缓存（crop_timing.json + 至少一个 crop 帧目录）"""
    timing_file = sid_cache_dir / "crop_timing.json"
    if not timing_file.exists():
        return False
    crop0 = sid_cache_dir / "cropped" / "crop_00000"
    return crop0.exists() and len(list(crop0.glob("*.jpg"))) > 0


def _score_from_cache(pipeline, sid_cache_dir: Path, gen_wav: str):
    """
    用缓存的视觉帧 + 新生成音频打分，完全跳过 S3FD 人脸检测。
    返回 (best_min_dist, best_conf, error_str)
    """
    timings = json.loads((sid_cache_dir / "crop_timing.json").read_text())
    cfg = pipeline.cfg

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_audio = tmp.name
    r = subprocess.run(
        [FFMPEG_BIN, "-y", "-i", gen_wav,
         "-ac", "1", "-ar", str(cfg.audio_sample_rate), "-f", "wav", tmp_audio],
        capture_output=True, timeout=60
    )
    if r.returncode != 0:
        return None, None, "音频重采样失败"

    class Opt:
        pass

    offsets, confs, dists = [], [], []

    try:
        for i, t in enumerate(timings):
            crop_eval_dir = sid_cache_dir / "cropped" / f"crop_{i:05d}"
            if not crop_eval_dir.exists() or not list(crop_eval_dir.glob("*.jpg")):
                break

            audio_out = crop_eval_dir / "audio.wav"
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", tmp_audio,
                 "-ss", f"{t['start_sec']:.3f}",
                 "-to", f"{t['end_sec']:.3f}",
                 str(audio_out)],
                capture_output=True, timeout=30
            )

            opt = Opt()
            opt.tmp_dir = str(crop_eval_dir)
            opt.batch_size = cfg.batch_size
            opt.vshift = cfg.vshift

            try:
                off, conf, dist = pipeline.syncnet.evaluate(opt=opt)
                offsets.append(off)
                confs.append(conf)
                dists.append(dist)
            except Exception:
                pass
    finally:
        os.unlink(tmp_audio)

    if not offsets:
        return None, None, "缓存打分失败（无有效 crop）"
    return float(min(dists)), float(max(confs)), None


def _create_full_frame_roi(pipeline, gt_video: str, gen_wav: str, sid_cache_dir: Path):
    """
    正面人脸数据集（FULL_FRAME_DATASETS）专用：跳过 S3FD，把 GT 视频
    center-crop 成正方形后缩放到 224×224，直接作为 SyncNet 输入。
    首次调用时建立缓存（帧 jpg + crop_timing.json），后续只替换 audio.wav。
    返回 (best_min_dist, best_conf, error_str)
    """
    cfg = pipeline.cfg
    crop_eval_dir = sid_cache_dir / "cropped" / "crop_00000"

    if not _has_roi_cache(sid_cache_dir):
        # ── 首次：提取视觉帧，保存缓存 ──
        sid_cache_dir.mkdir(parents=True, exist_ok=True)
        crop_eval_dir.mkdir(parents=True, exist_ok=True)

        # center-crop 到正方形，再 scale 到 224×224，提取 25fps jpg
        r = subprocess.run(
            [FFMPEG_BIN, "-y", "-i", gt_video,
             "-vf", "crop=min(iw\\,ih):min(iw\\,ih),scale=224:224,fps=25",
             "-q:v", "2",
             str(crop_eval_dir / "%06d.jpg")],
            capture_output=True, timeout=120
        )
        frames = list(crop_eval_dir.glob("*.jpg"))
        if r.returncode != 0 or not frames:
            return None, None, "全帧提取失败"

        duration = len(frames) / cfg.frame_rate
        (sid_cache_dir / "crop_timing.json").write_text(
            json.dumps([{"start_sec": 0.0, "end_sec": round(duration, 3)}], indent=2))

    # ── 每次都替换 audio.wav（gen audio 每次不同）──
    r = subprocess.run(
        [FFMPEG_BIN, "-y", "-i", gen_wav,
         "-ac", "1", "-ar", str(cfg.audio_sample_rate), "-f", "wav",
         str(crop_eval_dir / "audio.wav")],
        capture_output=True, timeout=60
    )
    if r.returncode != 0:
        return None, None, "音频重采样失败"

    class Opt:
        pass

    opt = Opt()
    opt.tmp_dir = str(crop_eval_dir)
    opt.batch_size = cfg.batch_size
    opt.vshift = cfg.vshift

    try:
        _, conf, dist = pipeline.syncnet.evaluate(opt=opt)
        return float(dist), float(conf), None
    except Exception as e:
        return None, None, str(e)


def _run_inference_and_cache(pipeline, tmp_video: str, sid_cache_dir: Path):
    """
    首次运行：带 cache_dir 的完整 inference（含 S3FD），同时：
      1. monkey-patch _crop 拦截每个 track 的 start/end_sec → crop_timing.json
      2. inference 结束后清理大文件（video.avi、frames/、cropped/*.avi、speech.wav）
    返回 (best_min_dist, best_conf, success)
    """
    captured_timings = []
    original_crop = pipeline._crop

    def patched_crop(track, frames, audio_wav, base):
        ss = track["frame"][0] / pipeline.cfg.frame_rate
        to = (track["frame"][-1] + 1) / pipeline.cfg.frame_rate
        captured_timings.append({"start_sec": float(ss), "end_sec": float(to)})
        return original_crop(track, frames, audio_wav, base)

    pipeline._crop = patched_crop
    try:
        sid_cache_dir.mkdir(parents=True, exist_ok=True)
        results = pipeline.inference(video_path=tmp_video, cache_dir=str(sid_cache_dir))
    finally:
        try:
            del pipeline._crop
        except AttributeError:
            pipeline._crop = original_crop

    _, confidences, min_dists, best_conf, best_min_dist, _, success = results

    if success and captured_timings:
        (sid_cache_dir / "crop_timing.json").write_text(
            json.dumps(captured_timings, indent=2))

        for name in ["video.avi", "speech.wav"]:
            p = sid_cache_dir / name
            if p.exists():
                p.unlink()
        shutil.rmtree(sid_cache_dir / "frames", ignore_errors=True)
        cropped_dir = sid_cache_dir / "cropped"
        if cropped_dir.exists():
            for f in cropped_dir.glob("*.avi"):
                f.unlink()
            for f in cropped_dir.glob("*.wav"):
                f.unlink()

    return best_min_dist, best_conf, success


# ─── face_alignment fallback：SyncNet S3FD 失败时的替代人脸检测 ────────────────

def _fallback_fa_crop_cache(gt_video: str, sid_cache_dir: Path,
                             frame_rate: int = 25, crop_scale: float = 0.40,
                             device: str = "cuda") -> tuple:
    """
    当 SyncNet 内置 S3FD 检测失败时（动画脸、小人脸等），用 face_alignment
    重新检测人脸，按照 syncnet _crop 完全相同的逻辑裁剪 → 224×224 jpg，
    写入 cache 目录，供后续 _score_from_cache 使用。
    返回 (success: bool, error_str: str | None)
    """
    import cv2
    import face_alignment as fa_module
    import torch
    from scipy import signal as scipy_signal

    crop_dir = sid_cache_dir / "cropped" / "crop_00000"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 提取帧（25fps）到临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir()

        r = subprocess.run(
            [FFMPEG_BIN, "-y", "-i", gt_video,
             "-r", str(frame_rate), "-q:v", "2",
             str(frames_dir / "%06d.jpg")],
            capture_output=True, timeout=120
        )
        if r.returncode != 0:
            return False, "fallback: ffmpeg 帧提取失败"

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            return False, "fallback: 没有提取到帧"

        # 读取所有帧（保持原始 index 对应关系）
        frames_bgr = [cv2.imread(str(p)) for p in frame_paths]
        valid_read = [(i, f) for i, f in enumerate(frames_bgr) if f is not None]
        if not valid_read:
            return False, "fallback: 所有帧读取失败"

        # Step 2: face_alignment 批量检测
        # 指定 TORCH_HOME 到 vepfs 共享缓存，避免每次下载 s3fd 模型
        os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "torch_hub"))
        use_device = device if torch.cuda.is_available() else "cpu"
        fa = fa_module.FaceAlignment(
            fa_module.LandmarksType.TWO_D,
            device=use_device,
            face_detector="sfd",
        )

        # bboxes[i] = [x1,y1,x2,y2] 或 None，index 对应 frames_bgr
        bboxes = [None] * len(frames_bgr)
        batch_size = 16
        for batch_start in range(0, len(valid_read), batch_size):
            batch = valid_read[batch_start: batch_start + batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in batch]
            batch_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1).float() for f in batch_frames
            ])
            dets = fa.face_detector.detect_from_batch(batch_tensor)
            for j, orig_idx in enumerate(batch_indices):
                det = dets[j]
                if det is not None and len(det) > 0:
                    best = max(det, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    bboxes[orig_idx] = [float(best[0]), float(best[1]),
                                        float(best[2]), float(best[3])]

        valid_indices = [i for i, b in enumerate(bboxes) if b is not None]
        if not valid_indices:
            return False, "fallback: face_alignment 未检测到任何人脸"

        # Step 3: 与 syncnet _crop 完全相同的平滑 + 裁剪逻辑
        track_bboxes = [bboxes[i] for i in valid_indices]
        s_arr = np.array([max(b[3] - b[1], b[2] - b[0]) / 2 for b in track_bboxes], dtype=float)
        x_arr = np.array([(b[0] + b[2]) / 2 for b in track_bboxes], dtype=float)
        y_arr = np.array([(b[1] + b[3]) / 2 for b in track_bboxes], dtype=float)

        # medfilt kernel 需为奇数且 <= 序列长度
        k = min(13, len(s_arr))
        k = k if k % 2 == 1 else k - 1
        if k >= 3:
            s_arr = scipy_signal.medfilt(s_arr, k)
            x_arr = scipy_signal.medfilt(x_arr, k)
            y_arr = scipy_signal.medfilt(y_arr, k)

        # Step 4: 裁剪并保存 224×224 jpg（与 syncnet crop 目录格式完全一致）
        saved = 0
        for i, fidx in enumerate(valid_indices):
            img = frames_bgr[fidx]
            if img is None:
                continue
            bs = float(s_arr[i])
            cs = crop_scale
            pad = int(bs * (1 + 2 * cs))
            img_p = cv2.copyMakeBorder(
                img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110))
            my = float(y_arr[i]) + pad
            mx = float(x_arr[i]) + pad
            y1 = int(my - bs)
            y2 = int(my + bs * (1 + 2 * cs))
            x1 = int(mx - bs * (1 + cs))
            x2 = int(mx + bs * (1 + cs))
            crop = cv2.resize(img_p[y1:y2, x1:x2], (224, 224))
            cv2.imwrite(str(crop_dir / f"{saved + 1:06d}.jpg"), crop)
            saved += 1

        if saved == 0:
            return False, "fallback: 裁剪后无有效帧"

        # Step 5: crop_timing.json（格式与 syncnet 一致）
        start_sec = valid_indices[0] / frame_rate
        end_sec = (valid_indices[-1] + 1) / frame_rate
        (sid_cache_dir / "crop_timing.json").write_text(
            json.dumps([{"start_sec": float(start_sec), "end_sec": float(end_sec)}], indent=2)
        )

        return True, None


# ─── 单样本 SyncNet 打分（在子进程中运行）─────────────────────────────────────

def _score_single(args_tuple):
    sid, gen_wav, gt_video, dataset, edit_interval = args_tuple
    pipeline = _worker_pipeline  # 复用 worker 级别已加载的模型，无需重复加载
    try:
        # ── 有 edit_interval 时：跳过缓存，trim 后直接 inference ──
        if edit_interval is not None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_video = tmp.name
            ok = mux_audio_to_video(str(gt_video), str(gen_wav), tmp_video)
            if not ok:
                return sid, None, None, "ffmpeg mux 失败"

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp2:
                trimmed_video = tmp2.name
            ok = trim_exclude_interval(tmp_video, trimmed_video, edit_interval)
            os.unlink(tmp_video)
            if not ok:
                os.unlink(trimmed_video)
                return sid, None, None, f"ffmpeg trim edit_interval {edit_interval} 失败"

            try:
                results = pipeline.inference(video_path=trimmed_video)
                _, _, _, best_conf, best_min_dist, _, success = results
            finally:
                os.unlink(trimmed_video)

            if not success or best_min_dist is None:
                return sid, None, None, "SyncNet inference 失败（trimmed）"
            return sid, float(best_min_dist), float(best_conf), None

        # ── 无 edit_interval：走原有缓存逻辑 ──
        sid_cache_dir = ROI_CACHE_ROOT / dataset / sid

        if _has_roi_cache(sid_cache_dir):
            # ── 缓存命中：直接打分（S3FD / 全帧两种缓存格式一样）──
            best_min_dist, best_conf, err = _score_from_cache(
                pipeline, sid_cache_dir, str(gen_wav))
            if err:
                return sid, None, None, err

        elif dataset in FULL_FRAME_DATASETS:
            # ── 全帧模式：跳过 S3FD，直接用整帧 ──
            best_min_dist, best_conf, err = _create_full_frame_roi(
                pipeline, str(gt_video), str(gen_wav), sid_cache_dir)
            if err:
                return sid, None, None, err

        else:
            # ── 标准模式：完整 S3FD inference + 缓存 ──
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_video = tmp.name

            ok = mux_audio_to_video(str(gt_video), str(gen_wav), tmp_video)
            if not ok:
                return sid, None, None, "ffmpeg mux 失败"

            try:
                best_min_dist, best_conf, success = _run_inference_and_cache(
                    pipeline, tmp_video, sid_cache_dir)
            finally:
                if os.path.exists(tmp_video):
                    os.unlink(tmp_video)

            if not success or best_min_dist is None:
                # ── fallback：用 face_alignment S3FD 重建 ROI cache ──
                fb_ok, fb_err = _fallback_fa_crop_cache(
                    gt_video=str(gt_video),
                    sid_cache_dir=sid_cache_dir,
                    frame_rate=pipeline.cfg.frame_rate,
                    crop_scale=pipeline.cfg.crop_scale,
                    device=pipeline.device,
                )
                if not fb_ok:
                    return sid, None, None, f"SyncNet+fallback 均失败: {fb_err}"
                best_min_dist, best_conf, err = _score_from_cache(
                    pipeline, sid_cache_dir, str(gen_wav))
                if err:
                    return sid, None, None, f"fallback 打分失败: {err}"

        return sid, float(best_min_dist), float(best_conf), None

    except Exception as e:
        return sid, None, None, str(e)


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音频目录")
    parser.add_argument("--gt_jsonl", type=str, required=True, help="GT JSONL 文件")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset 名称，用于 ROI 缓存子目录（如 chem / grid / lrs3）")
    parser.add_argument("--s3fd_weights", type=str, default=DEFAULT_S3FD)
    parser.add_argument("--syncnet_weights", type=str, default=DEFAULT_SYNCNET)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="每张 GPU 的并行进程数（默认 2）")
    parser.add_argument("--ngpu", type=int, default=8,
                        help="使用的 GPU 数量（默认 8），会自动分配 worker 到各 GPU")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="指定 GPU 编号，逗号分隔，如 '0,1,2'（默认按 ngpu 从 0 开始）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu（默认 cuda）")
    parser.add_argument("--id_list", type=str, default=None, help="ID 过滤文件")
    args = parser.parse_args()

    for wpath in (args.s3fd_weights, args.syncnet_weights):
        if not Path(wpath).exists():
            print(f"ERROR: 权重文件不存在: {wpath}")
            sys.exit(1)

    from utils import load_id_list
    id_set = load_id_list(args.id_list)
    gt_map = build_gt_map(args.gt_jsonl)
    gen_map = get_gen_audio_map(args.gen_dir, id_set)
    matched = match_gen_gt(gen_map, gt_map)
    if not matched:
        print("ERROR: 无匹配样本")
        sys.exit(1)

    # 解析 GPU 列表
    if args.device == "cuda":
        if args.gpu_ids:
            gpu_list = [int(x) for x in args.gpu_ids.split(",")]
        else:
            gpu_list = list(range(args.ngpu))
    else:
        gpu_list = [None]

    total_workers = len(gpu_list) * args.num_workers

    tasks = []
    cached_count = 0
    full_frame_count = 0
    n_with_edit = 0
    for i, (sid, gen_path, gt_item) in enumerate(matched):
        gt_video = gt_item.get("video_path")
        edit_interval = gt_item.get("edit_interval", None)  # e.g. [1.0, 2.0]
        if edit_interval is not None:
            n_with_edit += 1
        if gt_video and Path(gt_video).exists():
            sid_cache_dir = ROI_CACHE_ROOT / args.dataset / sid
            if edit_interval is None and _has_roi_cache(sid_cache_dir):
                cached_count += 1
            elif edit_interval is None and args.dataset in FULL_FRAME_DATASETS:
                full_frame_count += 1
            tasks.append((sid, gen_path, gt_video, args.dataset, edit_interval))
        else:
            print(f"  [SKIP] {sid}: GT video 不存在 ({gt_video})")

    if not tasks:
        print("ERROR: 没有可用样本（需要 video_path 字段）")
        sys.exit(1)

    mode = "全帧模式（跳过S3FD）" if args.dataset in FULL_FRAME_DATASETS else "S3FD模式"
    print(f"\n[SyncNet] {len(tasks)} 个样本（{len(gpu_list)} GPU × {args.num_workers} workers = {total_workers} 并行）[{mode}]")
    print(f"  GPU 列表: {gpu_list}")
    print(f"  ROI 缓存命中: {cached_count}/{len(tasks)}")
    if full_frame_count:
        print(f"  全帧首次建缓存: {full_frame_count}/{len(tasks)}")
    if n_with_edit:
        print(f"  含 edit_interval: {n_with_edit}/{len(tasks)}（跳过缓存，trim 后评估）")
    print(f"  缓存目录: {ROI_CACHE_ROOT / args.dataset}")

    lse_d_list, lse_c_list, errors = [], [], []

    # 预填 GPU 队列：total_workers 个 worker，每个取一个 gpu_id
    gpu_queue = multiprocessing.Queue()
    for i in range(total_workers):
        gpu_queue.put(gpu_list[i % len(gpu_list)])

    with ProcessPoolExecutor(
        max_workers=total_workers,
        initializer=_worker_init,
        initargs=(gpu_queue, args.s3fd_weights, args.syncnet_weights, args.device),
    ) as executor:
        futures = {executor.submit(_score_single, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="SyncNet"):
            sid, lse_d, lse_c, err = future.result()
            if err:
                errors.append((sid, err))
            else:
                lse_d_list.append(lse_d)
                lse_c_list.append(lse_c)

    if errors:
        print(f"\n  [WARNING] {len(errors)} 样本出错：")
        for sid, err in errors[:5]:
            print(f"    [{sid}] {err}")

    if not lse_d_list:
        print("ERROR: 所有样本计算失败")
        sys.exit(1)

    mean_lse_d = float(np.mean(lse_d_list))
    mean_lse_c = float(np.mean(lse_c_list))

    print(f"\n[SyncNet] {len(lse_d_list)} samples evaluated")
    print(f"  LSE-D (越低越好) : {mean_lse_d:.4f}")
    print(f"  LSE-C (越高越好) : {mean_lse_c:.4f}")
    return mean_lse_d, mean_lse_c


if __name__ == "__main__":
    main()
