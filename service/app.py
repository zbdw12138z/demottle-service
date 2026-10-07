"""
service/app.py — demottle 去斑驳 HTTP 服务（demottle_v13，grain=0.5）

接口      POST /v1/demottle?bits=8|16[&grain=0–1]  请求体：图片原始字节（PNG，8 位 RGB / RGBA / 灰度；也接受 JPEG）
                                            返回：image/png（8 位或 16 位；alpha 原样保留；PNG 色彩块 sRGB/iCCP/gAMA/cHRM/pHYs 透传）
          GET  /healthz                     进程存活
          GET  /readyz                      预热完成后 200（含版本、构建号），否则 503
          GET  /metrics                     Prometheus 文本格式（含斑驳尺度 σ 分布、按 grain 的请求数）
          grain 不传用 DEMOTTLE_GRAIN；可按请求设置做线上 A/B（同一进程内互不干扰）
保护      在途请求（处理中 + 排队）≥ DEMOTTLE_MAX_INFLIGHT → 429（读完并丢弃请求体，不进 GPU 队列）；请求体 > DEMOTTLE_MAX_BYTES → 413；
          先读文件头尺寸 / 位深，像素数超限 → 413、非 8 位 → 415，都在解码之前
并发      单进程；N 个工作线程（DEMOTTLE_WORKERS），每线程独占一条 CUDA 流，CPU 段与另一线程的 GPU 段重叠。
          启动时每个线程按常见尺寸逐个预热；未预热的新尺寸独占 GPU 处理一次，之后即算已预热；预热失败进程退出
追踪      请求头 X-Request-ID 透传（无则生成），写入响应头与日志；请求超过 DEMOTTLE_STUCK_SECONDS（60）秒自动导出全部线程栈；
          kill -USR1 <pid> 随时导出线程栈
配置      DEMOTTLE_GRAIN=0.5  DEMOTTLE_WORKERS=3  DEMOTTLE_MAX_INFLIGHT=8  DEMOTTLE_MAX_BYTES=67108864
          DEMOTTLE_MAX_PIXELS=16000000  DEMOTTLE_WARM_SIZES=1536x2048,2048x1536（宽x高）  DEMOTTLE_PNG_LEVEL=1
启动      uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-graceful-shutdown 30
"""
import asyncio, faulthandler, json, logging, os, signal, struct, sys, tempfile, threading, time, uuid
import concurrent.futures as cf
from contextlib import asynccontextmanager
import numpy as np, cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import demottle_v13 as A  # noqa: E402
from service.png_fast import encode_png, png_color_chunks  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402

VERSION = 'demottle_v13'
_bi = os.path.join(os.path.dirname(__file__), 'BUILD_INFO')
BUILD = json.load(open(_bi)) if os.path.exists(_bi) else {'build': 'dev', 'date': ''}
CFG = dict(grain=float(os.environ.get('DEMOTTLE_GRAIN', '0.5')),
           workers=int(os.environ.get('DEMOTTLE_WORKERS', '3')),
           max_inflight=int(os.environ.get('DEMOTTLE_MAX_INFLIGHT', '8')),
           max_bytes=int(os.environ.get('DEMOTTLE_MAX_BYTES', str(64 << 20))),
           max_pixels=int(os.environ.get('DEMOTTLE_MAX_PIXELS', '16000000')),
           warm=[tuple(int(v) for v in s.split('x')) for s in os.environ.get('DEMOTTLE_WARM_SIZES', '1536x2048,2048x1536').split(',') if s],
           png_level=int(os.environ.get('DEMOTTLE_PNG_LEVEL', '1')))
log = logging.getLogger('demottle')
logging.basicConfig(level=os.environ.get('DEMOTTLE_LOG_LEVEL', 'INFO'), format='%(asctime)s %(levelname)s %(message)s')
faulthandler.register(signal.SIGUSR1, all_threads=True)          # kill -USR1 <pid>：导出全部线程栈到 stderr
_active = {}                                                       # 请求 ID → 开始时间（看门狗用）
STUCK_S = float(os.environ.get('DEMOTTLE_STUCK_SECONDS', '60'))


def _watchdog():
    """任一请求处理超过 STUCK_S 秒：打印全部线程栈（每个请求只打一次）"""
    dumped = set()
    while True:
        time.sleep(5)
        now = time.time()
        for rid, t0 in list(_active.items()):
            if now - t0 > STUCK_S and rid not in dumped:
                dumped.add(rid)
                log.error(f'rid={rid} 已处理 {now - t0:.0f}s，疑似卡住，导出线程栈：')
                faulthandler.dump_traceback(all_threads=True)


# ================================================================= 监控指标（Prometheus 文本格式，无额外依赖）
class Metrics:
    BUCKETS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    SIGMA_BUCKETS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)       # 斑驳尺度 σ_L（L*）：分布漂移提示生成模型变了、阈值可能失效

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {}                                   # 状态码 → 次数
        self.applied = {'1': 0, '0': 0}
        self.hist = [0] * (len(self.BUCKETS) + 1); self.lat_sum = 0.0; self.lat_n = 0
        self.stage = {k: [0.0, 0] for k in ('decode', 'algo', 'encode')}
        self.inflight = 0
        self.sigma = [0] * (len(self.SIGMA_BUCKETS) + 1)
        self.by_grain = {}

    def done(self, code, seconds=None, stages=None, applied=None, sigma=None, grain=None):
        with self.lock:
            self.requests[code] = self.requests.get(code, 0) + 1
            if seconds is not None:
                i = next((k for k, b in enumerate(self.BUCKETS) if seconds <= b), len(self.BUCKETS))
                self.hist[i] += 1; self.lat_sum += seconds; self.lat_n += 1
            for k, v in (stages or {}).items():
                self.stage[k][0] += v; self.stage[k][1] += 1
            if applied is not None:
                self.applied['1' if applied else '0'] += 1
            if sigma is not None:
                self.sigma[next((k for k, b in enumerate(self.SIGMA_BUCKETS) if sigma <= b), len(self.SIGMA_BUCKETS))] += 1
            if grain is not None:
                self.by_grain[grain] = self.by_grain.get(grain, 0) + 1

    def text(self):
        with self.lock:
            L = ['# HELP demottle_requests_total 请求数（按状态码）', '# TYPE demottle_requests_total counter']
            L += [f'demottle_requests_total{{code="{c}"}} {n}' for c, n in sorted(self.requests.items())]
            L += ['# HELP demottle_request_seconds 成功请求的服务端耗时', '# TYPE demottle_request_seconds histogram']
            acc = 0
            for b, n in zip(list(self.BUCKETS) + ['+Inf'], self.hist):
                acc += n; L.append(f'demottle_request_seconds_bucket{{le="{b}"}} {acc}')
            L += [f'demottle_request_seconds_sum {self.lat_sum:.4f}', f'demottle_request_seconds_count {self.lat_n}']
            L += ['# HELP demottle_stage_seconds 各阶段累计耗时', '# TYPE demottle_stage_seconds summary']
            for k, (s_, n) in self.stage.items():
                L += [f'demottle_stage_seconds_sum{{stage="{k}"}} {s_:.4f}', f'demottle_stage_seconds_count{{stage="{k}"}} {n}']
            L += ['# TYPE demottle_applied_total counter'] + [f'demottle_applied_total{{applied="{k}"}} {v}' for k, v in self.applied.items()]
            L += ['# HELP demottle_mottle_sigma 输入图斑驳尺度 σ_L（L*）分布', '# TYPE demottle_mottle_sigma histogram']
            acc = 0
            for b, n in zip(list(self.SIGMA_BUCKETS) + ['+Inf'], self.sigma):
                acc += n; L.append(f'demottle_mottle_sigma_bucket{{le="{b}"}} {acc}')
            L += ['# HELP demottle_grain_requests_total 成功请求数（按补纹理强度，供 A/B）', '# TYPE demottle_grain_requests_total counter']
            L += [f'demottle_grain_requests_total{{grain="{g}"}} {n}' for g, n in sorted(self.by_grain.items())]
            L += ['# TYPE demottle_inflight gauge', f'demottle_inflight {self.inflight}',
                  '# TYPE demottle_ready gauge', f'demottle_ready {1 if _ready.is_set() else 0}',
                  '# TYPE demottle_build_info gauge',
                  f'demottle_build_info{{version="{VERSION}",build="{BUILD["build"]}",grain="{CFG["grain"]}",workers="{CFG["workers"]}"}} 1']
            return '\n'.join(L) + '\n'


M = Metrics()


# ================================================================= 并发控制与预热
class RWLock:
    """共享 / 独占锁：常规请求共享；未预热尺寸独占 GPU（避免与其他线程的 CUDA Graph 捕获冲突）"""
    def __init__(self):
        self._c = threading.Condition(); self._r = 0; self._w = False

    def acquire(self, exclusive):
        with self._c:
            if exclusive:
                while self._w or self._r:
                    self._c.wait()
                self._w = True
            else:
                while self._w:
                    self._c.wait()
                self._r += 1

    def release(self, exclusive):
        with self._c:
            if exclusive:
                self._w = False
            else:
                self._r -= 1
            self._c.notify_all()


_tls = threading.local()
_gpu = RWLock()
_warmed = set()
_ready = threading.Event()
_pool = None
_barrier = None
_warm_lock = threading.Lock()
_inflight_lock = threading.Lock()


def _stream():
    if not hasattr(_tls, 'stream'):
        _tls.stream = A.torch.cuda.Stream() if A.torch is not None and A.torch.cuda.is_available() else None
    return _tls.stream


def _run(rgb, path, bits, grain=None):
    g = CFG['grain'] if grain is None else grain
    st = _stream()
    if st is None:                                       # 无 CUDA：v13 自动退回 v10（CPU，慢，仅供调试）
        return A.demottle_v13(rgb, path=path, out_bits=bits, verbose=False, grain=g)
    with A.torch.cuda.stream(st):
        out = A.demottle_v13(rgb, path=path, out_bits=bits, verbose=False, grain=g)
        st.synchronize()
    return out


def _warm_one(_):
    _barrier.wait()                                      # 保证每个工作线程各领一个预热任务
    with _warm_lock:                                     # 逐个预热，避免并发捕获 CUDA Graph
        rng = np.random.default_rng(1)
        for (w, h) in CFG['warm']:
            img = cv2.GaussianBlur((rng.random((h, w, 3)) * 60 + 90).astype(np.uint8), (0, 0), 12)
            for bits in (8, 16):
                _run(img, None, bits)
    return threading.get_ident()


def _warm_all():
    t = time.time()
    try:
        ids = {f.result() for f in [_pool.submit(_warm_one, i) for i in range(CFG['workers'])]}
    except Exception:
        log.exception('预热失败，进程退出（交给容器平台重启 / 报警）')
        os._exit(1)
    for (w, h) in CFG['warm']:
        _warmed.add((h, w))
    _ready.set()
    log.info(f'预热完成 {time.time() - t:.1f}s | {VERSION} build={BUILD["build"]} grain={CFG["grain"]} workers={len(ids)} '
             f'max_inflight={CFG["max_inflight"]} 尺寸 {CFG["warm"]}')


# ================================================================= 输入检查（解码前）与处理
def probe(data):
    """只读文件头：返回 (格式, 宽, 高, 位深)；无法识别返回 None"""
    if data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) >= 26 and data[12:16] == b'IHDR':
        w, h, depth = struct.unpack('>IIB', data[16:25])
        return 'png', w, h, depth
    if data[:3] == b'\xff\xd8\xff':
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1; continue
            mk = data[i + 1]
            if mk in (0xD8, 0x01, 0xFF) or 0xD0 <= mk <= 0xD7:
                i += 1 if mk == 0xFF else 2; continue
            seg = struct.unpack('>H', data[i + 2:i + 4])[0]
            if 0xC0 <= mk <= 0xCF and mk not in (0xC4, 0xC8, 0xCC):
                p, h, w = struct.unpack('>BHH', data[i + 4:i + 9])
                return 'jpeg', w, h, p
            i += 2 + seg
        return None
    return None


def _decode(data):
    info = probe(data)
    if info is None:
        raise HTTPException(415, '只支持 PNG / JPEG')
    fmt, w, h, depth = info
    if w * h > CFG['max_pixels']:
        raise HTTPException(413, f'像素数 {w}×{h} 超过上限 {CFG["max_pixels"]}')
    if depth != 8:
        raise HTTPException(415, f'只支持 8 位输入（收到 {depth} 位）')
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise HTTPException(400, '无法解码图片')
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    alpha = None
    if im.shape[2] == 4:
        alpha = im[..., 3].copy(); im = im[..., :3]
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), alpha, fmt


def _process(data, bits, grain):
    t0 = time.perf_counter()
    rgb, alpha, fmt = _decode(data)
    extra = png_color_chunks(data) if fmt == 'png' else []
    tmp = None
    if fmt == 'jpeg':                                    # JPEG：算法要读量化表（按文件路径），写临时文件
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False); tmp.write(data); tmp.close()
    key = rgb.shape[:2]
    exclusive = key not in _warmed
    t1 = time.perf_counter()
    _gpu.acquire(exclusive)
    try:
        out, ctx = _run(rgb, tmp.name if tmp else None, bits, grain)
        _warmed.add(key)
    finally:
        _gpu.release(exclusive)
        if tmp:
            os.unlink(tmp.name)
    t2 = time.perf_counter()
    if alpha is not None:
        out = np.dstack([out, alpha if bits == 8 else (alpha.astype(np.uint16) * 257)])
    png = encode_png(out, level=CFG['png_level'], extra_chunks=extra)   # 多线程 PNG 编码（标准 PNG）
    t3 = time.perf_counter()
    sig = ctx.get('sigma')
    sigma = float(sig[0]) * 100 if sig is not None else None
    return png, bool(ctx.get('ok', False)), dict(decode=t1 - t0, algo=t2 - t1, encode=t3 - t2), f'{key[1]}x{key[0]}', sigma


# ================================================================= HTTP
@asynccontextmanager
async def lifespan(_app):
    global _pool, _barrier
    _pool = cf.ThreadPoolExecutor(max_workers=CFG['workers'], thread_name_prefix='gpu')
    _barrier = threading.Barrier(CFG['workers'])
    threading.Thread(target=_warm_all, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()
    yield
    log.info('收到停止信号：等待在途请求完成')
    _pool.shutdown(wait=True)


app = FastAPI(title='demottle', version=f'{VERSION}+{BUILD["build"]}', lifespan=lifespan)


@app.get('/healthz')
def healthz():
    return {'ok': True}


@app.get('/readyz')
def readyz():
    if not _ready.is_set():
        raise HTTPException(503, '预热中')
    return {'ready': True, 'version': VERSION, 'build': BUILD['build'], 'build_date': BUILD.get('date', ''),
            'grain': CFG['grain'], 'workers': CFG['workers'], 'max_inflight': CFG['max_inflight']}


@app.get('/metrics')
def metrics():
    return PlainTextResponse(M.text(), media_type='text/plain; version=0.0.4')


def _err(code, msg, rid, close=False):
    M.done(str(code))
    log.warning(f'rid={rid} {code} {msg}')
    hdr = {'X-Request-ID': rid}
    if code == 429:
        hdr['Retry-After'] = '1'
    if close:
        hdr['Connection'] = 'close'
    return JSONResponse({'error': msg, 'request_id': rid}, status_code=code, headers=hdr)


async def _drain(request):
    """提前拒绝前读完并丢弃请求体（至多 max_bytes）：否则客户端仍在上传时连接被关，收到的是连接重置而不是错误码"""
    n = 0
    async for chunk in request.stream():
        n += len(chunk)
        if n > CFG['max_bytes']:
            break


@app.post('/v1/demottle')
async def demottle(request: Request, bits: int = 8, grain: float | None = None):
    rid = request.headers.get('X-Request-ID') or uuid.uuid4().hex[:16]
    cl = request.headers.get('content-length')
    if cl and cl.isdigit() and int(cl) > CFG['max_bytes']:          # 声明体积就超限：不读，直接拒绝并关闭连接
        return _err(413, f'请求体 {cl} 字节超过上限 {CFG["max_bytes"]}', rid, close=True)
    if bits not in (8, 16):
        await _drain(request); return _err(422, 'bits 只能是 8 或 16', rid)
    if grain is not None and not (0.0 <= grain <= 1.0):
        await _drain(request); return _err(422, 'grain 须在 0–1 之间', rid)
    g = CFG['grain'] if grain is None else round(float(grain), 3)
    if not _ready.is_set():
        await _drain(request); return _err(503, '预热中', rid)
    with _inflight_lock:                                 # 过载保护：在途（处理中 + 排队）满了直接拒绝，不进 GPU 队列
        busy = M.inflight >= CFG['max_inflight']
        if not busy:
            M.inflight += 1
    if busy:
        await _drain(request)
        return _err(429, f'服务繁忙（在途已达 {CFG["max_inflight"]}），请稍后重试', rid)
    try:
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > CFG['max_bytes']:
                return _err(413, f'请求体超过上限 {CFG["max_bytes"]}', rid, close=True)
        if not buf:
            return _err(400, '请求体为空', rid)
        t = time.perf_counter(); _active[rid] = time.time()
        try:
            png, applied, tm, size, sigma = await asyncio.get_running_loop().run_in_executor(_pool, _process, bytes(buf), bits, g)
        except HTTPException as e:
            return _err(e.status_code, e.detail, rid)
        except Exception:
            log.exception(f'rid={rid} 处理失败')
            return _err(500, '内部错误', rid)
        sec = time.perf_counter() - t
        M.done('200', sec, tm, applied, sigma, g)
        log.info(f'rid={rid} {size} {len(buf) // 1024}KB→{len(png) // 1024}KB bits={bits} grain={g} applied={int(applied)} '
                 f'sigmaL={sigma if sigma is None else round(sigma, 3)} {sec * 1000:.0f}ms '
                 f'(decode {tm["decode"] * 1000:.0f} algo {tm["algo"] * 1000:.0f} encode {tm["encode"] * 1000:.0f})')
        return Response(png, media_type='image/png', headers={
            'X-Demottle-Version': f'{VERSION};build={BUILD["build"]};grain={g}', 'X-Demottle-Applied': '1' if applied else '0',
            'X-Process-Ms': f'{sec * 1000:.0f}', 'X-Request-ID': rid})
    finally:
        _active.pop(rid, None)
        with _inflight_lock:
            M.inflight -= 1
