# Changelog

本项目变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### 新增
- GitHub Pages 落地页（`docs/index.html`）——双模型检测卖点、demo GIF 展示、参数表、架构图、硬件加速对比、一行安装 CTA。配色走黄黑/打码意象，字体 Space Grotesk + JetBrains Mono。部署路径：`main` 分支 `/docs` 文件夹。

## [2026-04-17]

### 修复
- FFmpeg 管道死锁与崩溃处理（3d27e5e）——stdin 写入阻塞时不再卡死主进程。

### 文档
- README 增加打码效果 GIF 演示（9dbbefd）。

## [2026-04-14]

### 新增
- ROI 扩展 feather margin 防止羽化硬边（3773398）——扩大模糊 mask 范围再羽化，消除边界硬切。
- 跳过远处小脸（`--min-face-size`）——避免对画面中低于阈值的人脸做无效处理。

## [2026-04-13]

### 新增
- 视频人脸打码工具首版（42f23d3）。
  - 高斯模糊 / 马赛克双模式。
  - 双模型检测：MediaPipe FaceLandmarker（468 点轮廓）+ FaceDetector（侧脸兜底）。
  - 硬件加速编码自动识别：VideoToolbox / NVENC / QSV / AMF / libx264。
  - 管道直出 FFmpeg，无中间文件，原音轨 copy。
  - 首次运行自动创建 `.venv` 并安装依赖。
