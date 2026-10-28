# FeatherTalk 实时服务

FeatherTalk 在 OmniRT 内独立执行；无需安装或导入 OpenTalking。
安装依赖：`pip install -e ".[server,feathertalk]"`。

## 启动

```bash
export OMNIRT_FEATHERTALK_DEVICE=cuda:0  # cpu / mps / auto 也可
export OMNIRT_FEATHERTALK_DATASET=/srv/feathertalk/data/person
export OMNIRT_FEATHERTALK_CHECKPOINT=/srv/feathertalk/checkpoints/person/last.pth
export OMNIRT_FEATHERTALK_HUBERT_CHECKPOINT=/srv/feathertalk/feather_hubert.pth
bash scripts/start_feathertalk.sh
```

数据目录包含 `full_body_img/<数字>.jpg` 和 `landmarks/<同数字>.lms`。
模型定义、HuBERT、预处理工具和离线推理源码位于 `omnirt.models.feathertalk`。
权重和人物素材不打包到 Python 包中，使用现有 FeatherTalk 资产。
离线命令：`python -m omnirt.models.feathertalk.inference --help`。

## 服务接口

- `GET /v1/audio2video/models`：返回 feathertalk 的启用状态。
- `WS /v1/audio2video/feathertalk`（兼容别名 `/v1/avatar/feathertalk`）。
- `init`：包含 base64 `ref_image`；可选 `feathertalk_dataset`、
  `feathertalk_checkpoint`、`feathertalk_hubert_checkpoint` 覆盖环境配置。
  请求路径必须位于 `OMNIRT_ALLOWED_FRAME_ROOTS` 中（多个根路径用操作系统路径分隔符）。
  环境配置的路径属于服务端可信配置。
- `init_ok`：固定 16kHz 单声道、25fps、`slice_len=1`、`chunk_samples=640`。
  宽高来自源素材，人物坐标不进行预先缩放。
- 二进制输入：`AUDI` + 640 个 little-endian int16 采样点。
- 二进制输出：`VIDX` + uint32 帧数 + 每帧 uint32 长度和 JPEG。
- 可在每个音频块前发送 `audio_context`：`pcm_s16le` 为未来最多 6400 个采样点的
  base64 int16 PCM，`reset=true` 表示新一轮发言，响应 `audio_context_ok`。
  后续音频只用于编码上下文，不推进当前块时间。没有上下文时仍能推理，但边界口型可能下降。
- `idle` + `frame_index`：返回一个按素材往返播放的 JPEG 视频块，不改变发言状态。
- `close`：释放会话状态。断开连接同样清理；模型和参考素材按路径缓存复用，音频历史逐会话隔离。

## OpenTalking 接入

在 OpenTalking 中执行：

```bash
bash scripts/start_unified.sh --backend omnirt --model feathertalk --omnirt http://127.0.0.1:9000
```

现有 manifest 的 `metadata.feathertalk` 路径会转发给 OmniRT。如果两端共享文件系统，
请在 OmniRT 配置 `OMNIRT_ALLOWED_FRAME_ROOTS` 包含这些路径；不同机器应让 OmniRT
使用其环境配置中的资产路径，并从 OpenTalking manifest 删除本机专用的资产路径。
OpenTalking 客户端支持 audio_context 和 idle；音频块仍为 40ms，右侧预读约 400ms。

## 验证

`pytest tests/unit/test_feathertalk_runtime.py` 覆盖模型发现、会话、JPEG 推理协议、
上下文、idle、清理、路径限制和错误音频。真实权重需按上述环境变量另行部署。
