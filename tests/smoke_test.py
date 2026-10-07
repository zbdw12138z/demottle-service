"""
smoke_test.py — demottle 服务冒烟测试：部署后一键确认"接口通、结果对、保护生效"

用法  python3 tests/smoke_test.py --url http://HOST:8000 [--overload] [--write-golden]
检查  1 /healthz /readyz /metrics
      2 合成测试图（确定性生成，含斑驳）8 位：200、尺寸不变、X-Demottle-Applied=1、平均改动在合理范围、两次请求逐字节相同
      3 与标准答案 tests/golden/synth_8bit.png 比：最大差 ≤ 1 级（标准答案在 L4 上生成；--write-golden 重写）
      4 16 位输出；RGBA 输入 alpha 原样保留；PNG sRGB 块透传；请求 ID 透传
      5 错误码：空请求体 400、非图片 415、bits=7 / grain=2 422、文件头声明超大尺寸 413（不解码）、16 位输入 415；按请求设置 grain 生效
      6（--overload）并发超过 max_inflight：出现 429 且带 Retry-After，其余请求成功
退出码 0 = 全部通过
"""
import argparse, json, os, struct, sys, time, urllib.error, urllib.request, zlib
import concurrent.futures as cf
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden', 'synth_8bit.png')


def synth_image(h=2048, w=1536):
    """确定性合成图：光照渐变 + 柔边物体 + 60–200px 斑驳（亮度暗处偏黄）+ 细颗粒"""
    rng = np.random.default_rng(20261005)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    L = 55 + 20 * (x / w) - 10 * (y / h)
    L += 18 * (np.hypot(x - 0.62 * w, y - 0.40 * h) < 300) * 1.0
    L = cv2.GaussianBlur(L, (0, 0), 6)
    m = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 30); m /= m.std()
    L += 0.9 * m + 0.25 * rng.normal(size=(h, w)).astype(np.float32)
    a = 4 + 0.3 * m; b = 12 - 0.6 * m
    lab = np.stack([L, a, b], -1).astype(np.float32)
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return (rgb * 255 + 0.5).astype(np.uint8)


def png_bytes(rgb_or_rgba, srgb=False):
    im = rgb_or_rgba
    bgr = cv2.cvtColor(im, cv2.COLOR_RGBA2BGRA if im.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    b = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
    if srgb:                                               # 在 IHDR 之后插入 sRGB 块
        ch = b'sRGB' + b'\x00'
        blk = struct.pack('>I', 1) + ch + struct.pack('>I', zlib.crc32(ch) & 0xffffffff)
        b = b[:33] + blk + b[33:]
    return b


def call(url, data, bits=8, rid=None, path='/v1/demottle', grain=None):
    hdr = {'Content-Type': 'application/octet-stream'}
    if rid:
        hdr['X-Request-ID'] = rid
    q = f'?bits={bits}' + (f'&grain={grain}' if grain is not None else '')
    req = urllib.request.Request(f'{url}{path}{q}', data=data, method='POST', headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(), r.headers           # 响应头查找不区分大小写
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def decode(png):
    im = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is not None and im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA if im.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--overload', action='store_true'); ap.add_argument('--write-golden', action='store_true')
    a = ap.parse_args()
    results = []

    def check(name, ok, detail=''):
        results.append((name, bool(ok), detail)); print(f'{"通过" if ok else "失败"}  {name}  {detail}', flush=True)

    for _ in range(300):
        try:
            with urllib.request.urlopen(f'{a.url}/readyz', timeout=5) as r:
                info = json.loads(r.read()); break
        except Exception:
            time.sleep(1)
    else:
        print('服务未就绪'); sys.exit(1)
    with urllib.request.urlopen(f'{a.url}/healthz', timeout=5) as r:
        check('healthz', r.status == 200)
    check('readyz', info.get('ready') and info.get('version'), json.dumps(info, ensure_ascii=False))

    src = synth_image(); body = png_bytes(src)
    st, out1, h1 = call(a.url, body, rid='smoke-1')
    img = decode(out1) if st == 200 else None
    check('8 位：200 且尺寸不变', st == 200 and img is not None and img.shape == src.shape, f'状态 {st}')
    if img is None:
        sys.exit(1)
    delta = float(np.abs(img.astype(np.float32) - src).mean())
    check('8 位：已处理且改动合理', h1.get('X-Demottle-Applied') == '1' and 0.05 <= delta <= 3.0, f'applied={h1.get("X-Demottle-Applied")} 平均改动 {delta:.2f} 级')
    check('请求 ID 透传', h1.get('X-Request-ID') == 'smoke-1')
    st2, out2, _ = call(a.url, body)
    check('确定性：两次请求逐字节相同', st2 == 200 and out1 == out2)
    if a.write_golden:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True); open(GOLDEN, 'wb').write(out1)
        check('写标准答案', True, GOLDEN)
    elif os.path.exists(GOLDEN):
        g = decode(open(GOLDEN, 'rb').read())
        d = np.abs(img.astype(int) - g.astype(int))
        check('与标准答案一致（最大差 ≤ 1 级）', d.max() <= 1, f'最大差 {d.max()} 级，差异像素 {(d > 0).mean() * 100:.3f}%')
    else:
        check('与标准答案一致', False, f'缺少 {GOLDEN}')

    st, o16, _ = call(a.url, body, bits=16); im16 = decode(o16)
    check('16 位输出', st == 200 and im16 is not None and im16.dtype == np.uint16 and im16.shape == src.shape)
    rgba = np.dstack([src, (np.arange(src.shape[1]) % 256).astype(np.uint8)[None, :].repeat(src.shape[0], 0)])
    st, oa, _ = call(a.url, png_bytes(rgba)); ia = decode(oa)
    check('RGBA：alpha 原样保留', st == 200 and ia is not None and ia.shape[2] == 4 and np.array_equal(ia[..., 3], rgba[..., 3]))
    st, os_, _ = call(a.url, png_bytes(src, srgb=True))
    check('PNG sRGB 块透传', st == 200 and b'sRGB' in os_[:200])

    check('空请求体 → 400', call(a.url, b'')[0] == 400)
    check('非图片 → 415', call(a.url, b'hello world, not an image')[0] == 415)
    check('bits=7 → 422', call(a.url, body, bits=7)[0] == 422)
    st0, o0, h0 = call(a.url, body, grain=0); st1, o1, h1g = call(a.url, body, grain=1)
    check('按请求设置 grain（A/B）', st0 == 200 and st1 == 200 and 'grain=0.0' in h0.get('X-Demottle-Version', '')
          and 'grain=1.0' in h1g.get('X-Demottle-Version', '') and o0 != out1 and o1 != out1,
          f'{h0.get("X-Demottle-Version")} / {h1g.get("X-Demottle-Version")}')
    check('grain=2 → 422', call(a.url, body, grain=2)[0] == 422)
    big = bytearray(body[:33]); big[16:24] = struct.pack('>II', 40000, 40000)
    t = time.time(); code = call(a.url, bytes(big) + body[33:])[0]
    check('超大尺寸（只看文件头）→ 413', code == 413, f'{(time.time() - t) * 1000:.0f} ms')
    o16in = cv2.imencode('.png', (src.astype(np.uint16) * 257)[..., ::-1])[1].tobytes()
    check('16 位输入 → 415', call(a.url, o16in)[0] == 415)

    if a.overload:
        n = int(info.get('max_inflight', 8)) + 6
        with cf.ThreadPoolExecutor(n) as ex:
            rs = list(ex.map(lambda i: call(a.url, body), range(n)))
        codes = [r[0] for r in rs]
        ra = [r[2].get('Retry-After') for r in rs if r[0] == 429]
        check('过载 → 429 且带 Retry-After，其余成功', codes.count(429) >= 1 and all(c in (200, 429) for c in codes) and all(ra),
              f'{codes.count(200)} 个 200，{codes.count(429)} 个 429')

    with urllib.request.urlopen(f'{a.url}/metrics', timeout=5) as r:
        mt = r.read().decode()
    check('/metrics', all(k in mt for k in ('demottle_requests_total{code="200"}', 'demottle_build_info', 'demottle_mottle_sigma_bucket', 'demottle_grain_requests_total')))
    bad = [n for n, ok, _ in results if not ok]
    print(f'\n{len(results) - len(bad)}/{len(results)} 项通过' + (f'；失败：{bad}' if bad else ''))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
