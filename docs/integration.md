# demottle 服务接入说明

输入 GPT Image API 返回的原始图片字节（PNG），返回去斑驳后的 PNG（8 位或 16 位）。算法为 `demottle_v13`，`grain=0.5`：去斑驳后，按实拍照片的水平补回一半被抹掉的细纹理（非皮肤区）。同一输入，输出逐像素相同。

## 接口

`POST /v1/demottle?bits=8`

| 项 | 说明 |
|---|---|
| 请求体 | 图片原始字节。PNG 8 位 RGB / RGBA（主路径）；也接受 JPEG |
| `bits` | `8`（默认，带 ±0.5 级抖动防色阶）或 `16`（不抖动，供下游转 10 位 HEIC） |
| `grain` | 可选，0–1。不传 = 用服务配置（0.5）。只用于 A/B：同一服务按请求切换（`grain=0` = 只去斑驳、不补纹理），响应头会写明本次实际值 |
| 响应体 | `image/png`；尺寸不变；有 alpha 原样保留（16 位时 ×257） |
| 响应头 | `X-Demottle-Version: demottle_v13;build=<构建号>;grain=<本次实际值>`　`X-Demottle-Applied: 1/0`（0 = 量不到斑驳，像素原样返回）　`X-Process-Ms`（服务端耗时）　`X-Request-ID`（请求头带了就透传，否则服务生成） |
| 元数据 | PNG 输入的色彩块（sRGB / iCCP / gAMA / cHRM / pHYs）原样写回输出；其他元数据（文本、EXIF）不保留 |
| 错误（JSON：`{"error", "request_id"}`） | 400 解码失败或空请求体；413 请求体超过 64 MB 或像素数超上限（只读文件头判断，不解码）；415 非 PNG / JPEG 或非 8 位；422 `bits` 或 `grain` 非法；**429 服务繁忙**（在途请求已满，带 `Retry-After: 1`，请退避重试或换实例）；503 预热中；500 内部错误 |

```bash
curl -s -X POST --data-binary @input.png -H 'Content-Type: image/png' 'http://HOST:8000/v1/demottle?bits=8' -o output.png
```

```python
import requests
r = requests.post('http://HOST:8000/v1/demottle', params={'bits': 8}, data=png_bytes, timeout=30)
r.raise_for_status(); out_png = r.content
```

`GET /healthz`：进程存活。`GET /readyz`：预热完成后返回 200（含版本、构建号、配置），之前返回 503（负载均衡以它为准）。
`GET /metrics`：Prometheus 文本格式——`demottle_requests_total{code}`、`demottle_request_seconds`（直方图）、`demottle_stage_seconds{stage=decode|algo|encode}`、`demottle_inflight`、`demottle_applied_total`、`demottle_ready`、`demottle_build_info`、`demottle_mottle_sigma`（每张图量到的斑驳幅度 σL*，直方图；效果异常时先看它）、`demottle_grain_requests_total{grain}`（各 grain 值的请求数，A/B 分流核对用）。

## 部署

```bash
docker build -t demottle:v13 .
docker run -d --gpus all -p 8000:8000 --restart unless-stopped --name demottle demottle:v13
docker exec demottle python3 tests/smoke_test.py --overload   # 部署后一键自检（在容器内跑，宿主机无需装依赖），退出码 0 = 通过
```

冒烟测试检查：接口、确定性、与标准答案（`tests/golden/`，在 L4 上生成）最大差 ≤ 1 级、16 位、alpha、色彩块透传、各错误码、过载 429、监控指标。换显卡型号后若「与标准答案一致」一项不过，先看最大差是否为零星像素的 2–3 级（不同 GPU 的浮点细节），再联系算法负责人。

环境变量用 `docker run -e 名称=值` 设置。

| 项 | 要求 / 默认 |
|---|---|
| GPU | NVIDIA，驱动 ≥ 580（CUDA 13）；实测 L4（23 GB） |
| 每个容器 | 1 张卡、1 个进程；扩容靠多容器 |
| CPU / 内存 | 每卡 8 核、32 GiB（实测配置）。CPU 负责 PNG 解码 / 编码、人脸检测，单张约 0.2 s；核数更少时吞吐可能下降（未测），请用 `tools/loadtest.py` 自测 |
| 启动 / 停止 | 预热约 20 s（Triton 编译、CUDA Graph 捕获、按常见尺寸缓存）；预热失败进程直接退出。SIGTERM 后等在途请求完成再退出（最多 30 s） |
| 版本 | 构建号 = 包内源文件内容哈希，见 `service/BUILD_INFO`，也在 `/readyz`、响应头、`/metrics` 中 |
| `DEMOTTLE_GRAIN` | 0.5。设为 0 = 只去斑驳、不补纹理（回滚开关） |
| `DEMOTTLE_WORKERS` | 3（每卡并行处理的请求数；见下表） |
| `DEMOTTLE_MAX_INFLIGHT` | 8（处理中 + 排队的上限，超出返回 429；约等于最多排队 2 s） |
| `DEMOTTLE_WARM_SIZES` | `1536x2048,2048x1536`（宽x高）。其他尺寸首次请求会独占 GPU 多花约 1 s，之后正常 |
| `DEMOTTLE_PNG_LEVEL` | 1（多线程 PNG 编码；8 位输出约 2.4 MB / 张） |

## 性能与成本（L4，1536×2048 PNG 输入，按本 Dockerfile 构建的镜像实测）

| 输出 | 在途并发 | 延迟 p50 | p95 | 吞吐（张/s/卡） | 单张成本 |
|---|---|---|---|---|---|
| 8 位 | 1 | 510–557 ms | 553–572 ms¹ | 1.6–2.0 | $0.00020–0.00025 |
| 8 位 | 3 | 1.0–1.3 s | 1.2–1.6 s | 2.3–2.9 | $0.00014–0.00017 |
| 8 位 | 8 | 2.6–3.7 s | 3.0–4.6 s | 2.1–3.0 | $0.00013–0.00019 |
| 16 位 | 1 | 660–696 ms | 729–757 ms | 1.4–1.5 | $0.00027–0.00028 |
| 16 位 | 3 | 1.1–1.4 s | 1.3–1.6 s | 2.1–2.6 | $0.00015–0.00019 |

区间为不同机器上的四次实测（2026-10-05 / 10-07）。¹ 其中一台机器首轮 p95 为 1.1 s（刚启动），其余三台 ≤ 572 ms。并发 3 以上只增加排队，不增加吞吐。成本按 L4 + 8 核 + 32 GiB 容器 $0.000398 / s 折算（2026-10 云厂商按秒计价），自有 GPU 请按自家卡价换算：单张成本 = 每秒卡价 ÷ 吞吐。上线前请在自家 GPU 上用 `tools/loadtest.py` 复测。
单张约 0.5 s 的构成：PNG 解码约 75 ms（CPU）、算法约 380 ms（其中 GPU 约 310 ms，CPU 侧人脸检测等约 70 ms）、8 位 PNG 编码约 45 ms（CPU，8 线程）。单个请求内串行；多个请求之间 CPU 与 GPU 重叠，并发 ≥ 3 时 GPU 接近满载，吞吐由 GPU 决定。成本按整机（GPU + 8 核 + 32 GiB）计，只算 GPU 卡价约为一半。

建议：客户端超时 ≥ 10 s；每卡保持 3 个左右在途请求即可吃满吞吐，再多只会排队、拉高延迟。

## 运维与排障

| 场景 | 怎么办 |
|---|---|
| 看版本 | `GET /readyz` 的 `build`（运行代码内容哈希）；同一构建号 = 同一份运行代码；反馈问题时带上它和请求 ID |
| 日志 | `docker logs demottle`。每个请求一行：`rid=<请求 ID> <尺寸> <入KB>→<出KB> bits= grain= applied= sigmaL= <总ms> (decode / algo / encode 各 ms)`；错误为 WARNING，含请求 ID |
| 请求卡住 | 任一请求处理超过 `DEMOTTLE_STUCK_SECONDS`（默认 60）秒，服务自动把全部线程栈打到日志（关键字「疑似卡住」）；也可随时 `docker kill -s USR1 demottle` 导出。把这段日志发给算法负责人 |
| 429 增多 | 说明这张卡已满载（约 2.5 张/s）：加实例；不要调大 `DEMOTTLE_MAX_INFLIGHT`（只会让排队延迟变长） |
| 413 / 415 | 输入不合规（过大 / 非 8 位 / 非 PNG、JPEG）；在拒绝前已读完请求体，客户端收到的是正常错误码，不是连接重置。唯一例外：`Content-Length` 声明超过 64 MB 时直接拒绝并关闭连接 |
| 预热失败 | 进程退出（`--restart unless-stopped` 会自动重启）；日志里「预热失败」后的异常即原因（最常见：驱动版本不够、显存不足） |
| 回滚 | 带 `-e DEMOTTLE_GRAIN=0` 重新 `docker run`：只去斑驳、不补纹理（上一版效果，逐位相同） |

## 行为与注意事项

- 只处理生图；实拍照片会被平均改动约 1.8 级，不要送进来。
- 尺寸任意（≤ 1600 万像素）；常见尺寸以外首次请求较慢。
- 用户图片不落盘（JPEG 输入会写一个临时文件，处理完即删）。
