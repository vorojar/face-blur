#!/usr/bin/env python3
"""视频人脸打码工具 - 基于 MediaPipe FaceLandmarker + FaceDetector + OpenCV + FFmpeg"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
VENV_DIR = SCRIPT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python3"

def _ensure_venv():
    """自动创建虚拟环境并安装依赖，然后用 venv 的 python 重新执行自身。"""
    if VENV_DIR.exists() and VENV_PYTHON.exists():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

    print("首次运行，正在创建虚拟环境并安装依赖...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    subprocess.check_call([str(VENV_DIR / "bin" / "pip"), "install", "mediapipe", "opencv-python"])
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError:
    _ensure_venv()

import argparse
import platform
import time

import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector, FaceDetectorOptions,
    FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections,
)

LANDMARKER_MODEL_PATH = str(SCRIPT_DIR / "face_landmarker.task")
DETECTOR_MODEL_PATH = str(SCRIPT_DIR / "blaze_face_short_range.tflite")

FACE_OVAL_PATH = [c.start for c in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]


def parse_args():
    p = argparse.ArgumentParser(description="给视频中的人脸打马赛克/高斯模糊")
    p.add_argument("input", help="输入视频路径")
    p.add_argument("-o", "--output", help="输出视频路径（默认: 输入文件名_mosaic.mp4）")
    p.add_argument("-s", "--strength", type=int, default=80, help="模糊强度，越大越模糊 (默认: 80)")
    p.add_argument("-p", "--padding", type=float, default=0.3, help="人脸轮廓外扩比例 (默认: 0.3)")
    p.add_argument("--mode", choices=["gaussian", "mosaic"], default="gaussian", help="模糊模式 (默认: gaussian)")
    p.add_argument("--min-confidence", type=float, default=0.3, help="人脸检测最低置信度 (默认: 0.3)")
    p.add_argument("--min-face-size", type=int, default=40, help="人脸最小边长(像素)，低于此值不打码 (默认: 40)")
    p.add_argument("--detect-interval", type=int, default=3, help="每隔N帧检测一次人脸 (默认: 3)")
    p.add_argument("--preview", action="store_true", help="实时预览（按 q 退出）")
    return p.parse_args()


def make_output_path(input_path: str, output_path: str | None) -> str:
    if output_path:
        return output_path
    p = Path(input_path)
    return str(p.with_stem(p.stem + "_mosaic"))


def blur_face_region(frame, contour_pts, strength, mode, min_face_size=0):
    """只对人脸 ROI 区域做模糊，边缘柔化过渡。"""
    x, y, rw, rh = cv2.boundingRect(contour_pts)
    if rw == 0 or rh == 0:
        return
    if min(rw, rh) < min_face_size:
        return

    feather = max(3, min(rw, rh) // 8) | 1
    H, W = frame.shape[:2]
    # ROI 向外扩展 feather 大小的 margin，确保羽化过渡不被边界截断
    x0 = max(x - feather, 0)
    y0 = max(y - feather, 0)
    x1 = min(x + rw + feather, W)
    y1 = min(y + rh + feather, H)
    roi_w, roi_h = x1 - x0, y1 - y0

    roi = frame[y0:y1, x0:x1].copy()
    local_pts = contour_pts - np.array([x0, y0])

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_pts, 255)
    mask_float = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0

    if mode == "gaussian":
        k = strength | 1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        small = cv2.resize(roi, (max(1, roi_w // strength), max(1, roi_h // strength)), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    mask_3ch = np.stack([mask_float, mask_float, mask_float], axis=-1)
    target = frame[y0:y1, x0:x1]
    blended = (blurred.astype(np.float32) * mask_3ch + target.astype(np.float32) * (1 - mask_3ch))
    np.copyto(target, blended.astype(np.uint8))


def get_face_contour(landmarks, width, height, padding):
    """从 FaceLandmarker 结果提取人脸轮廓点并外扩。"""
    pts = np.array(
        [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in FACE_OVAL_PATH],
        dtype=np.int32,
    )
    center = pts.mean(axis=0)
    pts = ((pts - center) * (1 + padding) + center).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts


def bbox_to_ellipse_pts(bbox, width, height, padding, n_points=36):
    """将 FaceDetector 的矩形框转为椭圆轮廓点。"""
    bx, by, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    cx = bx + bw / 2
    cy = by + bh / 2
    rx = bw / 2 * (1 + padding)
    ry = bh / 2 * (1 + padding)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.stack([
        np.clip((cx + rx * np.cos(angles)).astype(np.int32), 0, width - 1),
        np.clip((cy + ry * np.sin(angles)).astype(np.int32), 0, height - 1),
    ], axis=-1)
    return pts


def contours_overlap(contour_a, contour_b):
    """判断两个轮廓的 bounding rect 是否有重叠（用于去重）。"""
    ax, ay, aw, ah = cv2.boundingRect(contour_a)
    bx, by, bw, bh = cv2.boundingRect(contour_b)
    overlap_x = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    overlap_y = max(0, min(ay+ah, by+bh) - max(ay, by))
    overlap_area = overlap_x * overlap_y
    smaller_area = min(aw * ah, bw * bh)
    return overlap_area > smaller_area * 0.3 if smaller_area > 0 else False



def detect_hw_encoder() -> tuple[str, list[str]]:
    """检测最佳硬件编码器，返回 (编码器名, 额外参数)。"""
    if platform.system() == "Darwin":
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                 "-c:v", "h264_videotoolbox", "-f", "null", "-"],
                capture_output=True, check=True,
            )
            return "h264_videotoolbox", ["-q:v", "65"]
        except subprocess.CalledProcessError:
            pass

    for enc in ["h264_nvenc", "h264_amf", "h264_qsv"]:
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                 "-c:v", enc, "-f", "null", "-"],
                capture_output=True, check=True,
            )
            if enc == "h264_nvenc":
                return enc, ["-preset", "p4", "-tune", "hq"]
            return enc, []
        except subprocess.CalledProcessError:
            continue

    return "libx264", ["-preset", "fast", "-crf", "18"]


def fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.1f}s"


def process_video(args):
    t_total_start = time.monotonic()
    input_path = args.input
    output_path = make_output_path(input_path, args.output)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    t_init_start = time.monotonic()
    encoder, enc_opts = detect_hw_encoder()
    t_init = time.monotonic() - t_init_start

    print(f"输入: {input_path}")
    print(f"分辨率: {width}x{height} | FPS: {fps:.1f} | 时长: {fmt_time(duration)} | 总帧数: {total_frames}")
    print(f"编码器: {encoder}")
    print(f"模式: {args.mode} | 模糊强度: {args.strength} | 外扩: {args.padding} | 最小人脸: {args.min_face_size}px | 检测间隔: {args.detect_interval}帧")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_path,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", encoder, *enc_opts,
    ]

    ffmpeg_cmd += [
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # FaceLandmarker: 精准轮廓（正脸/微侧脸）
    lm_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
        num_faces=10,
        min_face_detection_confidence=args.min_confidence,
        min_face_presence_confidence=args.min_confidence,
    )
    landmarker = FaceLandmarker.create_from_options(lm_options)

    # FaceDetector: 兜底检测（侧脸/极端角度 Landmarker 漏检时补充）
    fd_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=DETECTOR_MODEL_PATH),
        min_detection_confidence=args.min_confidence,
    )
    detector = FaceDetector.create_from_options(fd_options)

    frame_count = 0
    face_total = 0
    cached_contours = []
    t_detect_total = 0.0
    t_blur_total = 0.0

    t_process_start = time.monotonic()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            need_detect = (frame_count - 1) % args.detect_interval == 0

            if need_detect:
                t0 = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # 先用 Landmarker 获取精准轮廓
                lm_results = landmarker.detect(mp_image)
                cached_contours = []
                if lm_results.face_landmarks:
                    for landmarks in lm_results.face_landmarks:
                        cached_contours.append(get_face_contour(landmarks, width, height, args.padding))

                # 再用 Detector 补充漏检的脸
                fd_results = detector.detect(mp_image)
                if fd_results.detections:
                    for det in fd_results.detections:
                        fd_contour = bbox_to_ellipse_pts(det.bounding_box, width, height, args.padding)
                        already_covered = any(contours_overlap(fd_contour, c) for c in cached_contours)
                        if not already_covered:
                            cached_contours.append(fd_contour)

                face_total += len(cached_contours)
                t_detect_total += time.monotonic() - t0

            t0 = time.monotonic()
            for contour in cached_contours:
                blur_face_region(frame, contour, args.strength, args.mode, args.min_face_size)
            t_blur_total += time.monotonic() - t0

            ffmpeg_proc.stdin.write(frame.tobytes())

            if args.preview:
                cv2.imshow("Face Blur Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n预览被用户中断")
                    break

            if frame_count % 100 == 0 or frame_count == total_frames:
                pct = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"\r处理进度: {frame_count}/{total_frames} ({pct:.1f}%)", end="", flush=True)

    finally:
        cap.release()
        landmarker.close()
        detector.close()
        ffmpeg_proc.stdin.close()
        if args.preview:
            cv2.destroyAllWindows()

    t_process = time.monotonic() - t_process_start

    print(f"\n检测到人脸总次数: {face_total}")
    print("正在等待 FFmpeg 完成编码...")

    t_wait_start = time.monotonic()
    ffmpeg_proc.wait()
    t_wait = time.monotonic() - t_wait_start

    if ffmpeg_proc.returncode != 0:
        print(f"FFmpeg 错误: {ffmpeg_proc.stderr.read().decode()}", file=sys.stderr)
        sys.exit(1)

    t_total = time.monotonic() - t_total_start

    input_size = Path(input_path).stat().st_size
    output_size = Path(output_path).stat().st_size
    ratio = output_size / input_size * 100

    print(f"\n{'='*45}")
    print(f"  完成! 输出: {output_path}")
    print(f"  文件大小: {input_size/1024/1024:.1f}MB → {output_size/1024/1024:.1f}MB ({ratio:.0f}%)")
    print(f"{'='*45}")
    print(f"  初始化 (编码器检测+码率读取): {fmt_time(t_init)}")
    print(f"  帧处理 (检测+模糊+管道写入): {fmt_time(t_process)}")
    print(f"    ├─ 人脸检测: {fmt_time(t_detect_total)}")
    print(f"    └─ 模糊渲染: {fmt_time(t_blur_total)}")
    print(f"  FFmpeg 收尾: {fmt_time(t_wait)}")
    print(f"  总耗时: {fmt_time(t_total)}")
    if duration > 0:
        print(f"  处理速度: {duration/t_total:.1f}x 实时")
    print(f"{'='*45}")


if __name__ == "__main__":
    process_video(parse_args())
