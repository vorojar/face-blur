# Face Blur - 视频人脸打码工具

一键给视频中所有人脸打上高斯模糊，支持正脸、侧脸、多人脸，30秒视频处理只需约8秒。

## 特性

- **双模型检测** — FaceLandmarker 精准轮廓 + FaceDetector 侧脸兜底，不漏脸
- **人脸轮廓打码** — 沿 468 个面部关键点生成精准轮廓，不是粗暴的矩形框
- **边缘柔化** — 羽化过渡，模糊区域与原图自然融合
- **硬件加速** — 自动检测 Apple Silicon / NVIDIA / AMD / Intel 硬件编码器
- **零配置** — 首次运行自动创建虚拟环境、安装依赖，直接 `python3 face_blur.py video.mp4`
- **管道直出** — 原始帧直接管道喂给 FFmpeg，无中间文件，无二次编码，画质无损

## 快速开始

```bash
# 只需要系统装了 Python 3 和 FFmpeg
python3 face_blur.py your_video.mp4
```

首次运行会自动创建 `.venv` 并安装 `mediapipe` 和 `opencv-python`，之后直接运行。

## 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output` | 输出文件路径 | `输入文件名_mosaic.mp4` |
| `-s, --strength` | 模糊强度，越大越模糊 | `80` |
| `-p, --padding` | 人脸轮廓外扩比例 | `0.3` |
| `--mode` | `gaussian`（高斯模糊）或 `mosaic`（马赛克） | `gaussian` |
| `--detect-interval` | 每 N 帧检测一次（提速） | `3` |
| `--min-confidence` | 人脸检测最低置信度 | `0.3` |
| `--preview` | 实时预览处理效果 | - |

## 用法示例

```bash
# 默认高斯模糊
python3 face_blur.py video.mp4

# 马赛克模式
python3 face_blur.py video.mp4 --mode mosaic

# 加大模糊强度
python3 face_blur.py video.mp4 -s 120

# 扩大打码范围（覆盖更多额头/下巴）
python3 face_blur.py video.mp4 -p 0.5

# 指定输出路径
python3 face_blur.py video.mp4 -o output.mp4

# 实时预览（按 q 退出）
python3 face_blur.py video.mp4 --preview
```

## 输出示例

```
输入: video.mp4
分辨率: 1280x720 | FPS: 30.0 | 时长: 30.0s | 总帧数: 901
编码器: h264_videotoolbox
模式: gaussian | 模糊强度: 80 | 外扩: 0.3 | 检测间隔: 3帧
处理进度: 901/901 (100.0%)
检测到人脸总次数: 268

=============================================
  完成! 输出: video_mosaic.mp4
  文件大小: 2.1MB → 5.9MB (281%)
=============================================
  初始化 (编码器检测+码率读取): 268ms
  帧处理 (检测+模糊+管道写入): 7.9s
    ├─ 人脸检测: 2.5s
    └─ 模糊渲染: 4.4s
  FFmpeg 收尾: 8ms
  总耗时: 8.4s
  处理速度: 3.6x 实时
=============================================
```

## 技术架构

```
视频输入
  │
  ▼
OpenCV 逐帧读取
  │
  ├─ MediaPipe FaceLandmarker → 468点人脸轮廓（正脸/微侧脸）
  ├─ MediaPipe FaceDetector   → 矩形框转椭圆（侧脸兜底）
  ├─ 去重合并
  │
  ▼
高斯模糊 + 羽化 mask + alpha 混合
  │
  ▼
管道 (pipe) → FFmpeg 硬件加速编码 + 原音轨 copy
  │
  ▼
输出 MP4
```

## 硬件加速

| 平台 | 编码器 | 速度 |
|------|--------|------|
| macOS Apple Silicon | h264_videotoolbox | ~5x 实时 |
| NVIDIA GPU | h264_nvenc | ~8x 实时 |
| AMD GPU | h264_amf | ~5x 实时 |
| Intel | h264_qsv | ~5x 实时 |
| CPU | libx264 | ~1x 实时 |

## 依赖

- Python 3.10+
- FFmpeg（系统安装）
- mediapipe、opencv-python（自动安装到 .venv）

## License

MIT
