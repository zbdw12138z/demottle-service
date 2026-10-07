"""
loadtest.py — demottle 服务压测：各并发下的单张延迟分位数（p50 / p95 / p99）与单卡吞吐、折算单张成本

用法  python3 tools/loadtest.py --url http://127.0.0.1:8000 --images 'data/jpeg/*.jpeg' [--concurrency 1,2,4,8] [--n 40]
                               [--bits 8] [--usd-per-s 0.000398]
说明  输入图先在本地转成 PNG 字节（与线上 API 原始 PNG 同类），请求前等 /readyz；
      每档并发先发 1 轮热身（不计），再计时 n 个请求。吞吐 = n / 墙钟时间；单张成本 = 每秒卡价 / 吞吐。
"""
import argparse, glob, json, time, urllib.request
import concurrent.futures as cf
import numpy as np, cv2


def post(url, data, bits):
    req = urllib.request.Request(f'{url}/v1/demottle?bits={bits}', data=data, method='POST', headers={'Content-Type': 'image/png'})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read(); srv = float(r.headers.get('X-Process-Ms', 'nan'))
    return time.perf_counter() - t, srv, len(body)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--url', default='http://127.0.0.1:8000'); ap.add_argument('--images', default='data/jpeg/*.jpeg')
    ap.add_argument('--concurrency', default='1,2,4,8'); ap.add_argument('--n', type=int, default=40); ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--usd-per-s', type=float, default=None)
    a = ap.parse_args()
    for _ in range(600):
        try:
            with urllib.request.urlopen(f'{a.url}/readyz', timeout=5) as r:
                info = json.loads(r.read()); break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit('服务未就绪')
    pngs = []
    for p in sorted(glob.glob(a.images)):
        im = cv2.imread(p)
        ok, buf = cv2.imencode('.png', im, [cv2.IMWRITE_PNG_COMPRESSION, 3]); pngs.append(buf.tobytes())
    print(f'服务 {info} | 输入 {len(pngs)} 张 PNG（{im.shape[1]}×{im.shape[0]}，平均 {np.mean([len(x) for x in pngs]) / 1e6:.1f} MB）| bits={a.bits}')
    print(f'{"并发":>4}{"请求":>6}{"p50":>9}{"p95":>9}{"p99":>9}{"服务端p50":>11}{"吞吐":>12}' + (f'{"单张成本":>16}' if a.usd_per_s else ''))
    res = {}
    for c in map(int, a.concurrency.split(',')):
        with cf.ThreadPoolExecutor(c) as ex:
            list(ex.map(lambda i: post(a.url, pngs[i % len(pngs)], a.bits), range(c)))          # 热身
            t = time.perf_counter()
            rs = list(ex.map(lambda i: post(a.url, pngs[i % len(pngs)], a.bits), range(a.n)))
            wall = time.perf_counter() - t
        lat = np.array([r[0] for r in rs]) * 1000; srv = np.array([r[1] for r in rs])
        thr = a.n / wall
        res[c] = dict(p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)), p99=float(np.percentile(lat, 99)),
                      srv_p50=float(np.nanmedian(srv)), throughput=thr)
        line = f'{c:>4}{a.n:>6}{res[c]["p50"]:>7.0f}ms{res[c]["p95"]:>7.0f}ms{res[c]["p99"]:>7.0f}ms{res[c]["srv_p50"]:>9.0f}ms{thr:>8.2f} 张/s'
        if a.usd_per_s:
            line += f'   ${a.usd_per_s / thr:.5f}（{a.usd_per_s / thr * 1e4:.1f} 美元/万张）'
        print(line, flush=True)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
