# GaussianTalker Streaming Interface

这是一个基于 FastAPI 和 WebSocket 的实时语音驱动视频生成接口。

## 1. 安装依赖

需要安装 FastAPI 相关依赖以及 `server_requirements.txt` 中的包。

```bash
pip install -r server_requirements.txt
```

注意：请确保你已经按照 GaussianTalker 的原始要求配置好了环境（PyTorch, MMCV 等）。

## 2. 准备模型

你需要准备好训练好的 GaussianTalker 模型路径，以及 DeepSpeech 模型路径。

DeepSpeech 模型通常位于 `~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb` 或者你可以指定其他路径。

## 3. 运行服务器

使用环境变量指定模型路径并启动服务器：

```bash
export GT_MODEL_PATH="/path/to/your/trained/model/output/ExpName"
export GT_DS_PATH="/path/to/deepspeech-0_1_0-b90017e8.pb" # 可选，如果默认路径存在则不需要

python server.py
# 或者使用 uvicorn
# uvicorn server:app --host 0.0.0.0 --port 8000
```

## 4. 使用

打开浏览器访问 `http://localhost:8000`。
点击 "Start Streaming"，允许麦克风权限。
对着麦克风说话，网页上应该会显示生成的视频流。

## 注意事项

- **性能**: 实时渲染需要较强的 GPU 性能。
- **DeepSpeech**: 确保 DeepSpeech 模型版本与代码兼容（代码默认适配 0.1.0 冻结图）。
- **音频采样率**: 前端会自动尝试采集并发送 16kHz 音频，但受限于浏览器和硬件，可能需要调整。
