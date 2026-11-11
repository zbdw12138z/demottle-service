# demottle 服务

GPT Image 生图后处理：去掉平坦面（墙、天空、皮肤）上的斑驳、蜡感和脏色。进什么出什么：JPEG 进 JPEG 出（同量化表、同抽样，ICC / EXIF 带回），PNG 进 PNG 出，尺寸不变；纯算法，同一输入同一输出。

## 0. 机器要求（先确认）

- Linux 服务器 + NVIDIA GPU（实测 L4），驱动 ≥ 580；每卡配 8 核 CPU、32 GiB 内存
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

一条命令检查，能打印出显卡信息表就可以继续：

```bash
docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi
```

没有 NVIDIA GPU 的机器（如 Mac 笔记本）跑不了这个服务。

## 1. 上线

```bash
git clone https://github.com/zbdw12138z/demottle-service.git && cd demottle-service

docker build -t demottle:v17 .        # 首次构建要下载 PyTorch 等依赖，需要几分钟
docker run -d --gpus all -p 8000:8000 --restart unless-stopped --name demottle demottle:v17

docker exec demottle python3 tests/smoke_test.py --overload   # 自检：自动等预热（约 20 秒），最后一行 "25/25 项通过" 即可接流量
```

## 2. 调用

```bash
curl -s -X POST --data-binary @input.png 'http://127.0.0.1:8000/v1/demottle?bits=8' -o output.png
```

```python
import requests
r = requests.post('http://127.0.0.1:8000/v1/demottle', params={'bits': 8}, data=png_bytes, timeout=30)
r.raise_for_status(); out_png = r.content
```

- 请求体就是出图的原始字节（JPEG 或 PNG），响应体格式与输入相同（尺寸不变）：JPEG 进 JPEG 出（`image/jpeg`），PNG 进 PNG 出。要强制格式加 `format=jpeg` 或 `format=png`。`bits=16` 输出 16 位 PNG（供转 10 位 HEIC）。
- 健康检查用 `GET /readyz`（预热完成才返回 200），监控用 `GET /metrics`（Prometheus）。
- 返回 **429** 表示这张卡满了：退避 1 秒重试或换实例。客户端超时建议 ≥ 10 秒。
- 只送生图，不要送实拍照片，也不要送出图失败时兜底返回的原图。

## 容量与成本（L4 + 8 核 + 32 GiB，1536×2048）

| | 单张延迟 | 单卡吞吐 | 单张成本（按整机计） |
|---|---|---|---|
| 8 位输出 | 约 0.5 秒 | 2.3–3.0 张/秒（视机器） | 约 1.3–1.7 美元 / 万张 |

每卡保持 3 个左右在途请求即可吃满；扩容靠加容器（每容器 1 卡 1 进程）。在自家机器上复测：

```bash
pip install numpy opencv-python-headless
python3 tools/loadtest.py --url http://127.0.0.1:8000 --images '你的图/*.png' --concurrency 1,3
```

## 更多

- 接口全部参数、错误码、环境变量、运维排障：[`docs/integration.md`](docs/integration.md)
- 日志：`docker logs demottle`；版本（构建号）在 `/readyz` 和响应头 `X-Demottle-Version` 里，反馈问题时请带上它和请求 ID（响应头 `X-Request-ID`）
- 回滚到上一版效果：`docker run` 时加 `-e DEMOTTLE_PLAN_A=off`（与上一版 v16 逐位相同，不用重新构建）；更早的版本：`-e DEMOTTLE_SYN_CAP=off`（v15）、`-e DEMOTTLE_GATES=off`（v13）

## 许可

MIT（见 `LICENSE`）。人脸检测模型 `src/face_detection_yunet_2023mar.onnx` 来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，MIT 许可。
