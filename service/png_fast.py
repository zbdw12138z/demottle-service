"""
png_fast.py — 多线程 PNG 编码（标准 PNG，任何解码器可读）

做法  1 逐行 Paeth 滤波（numpy 向量化，与 libpng 的 Paeth 定义一致）
      2 按行切成若干段，各段用 zlib 原始 deflate 独立压缩（Z_FULL_FLUSH 收尾，段间不共享字典；zlib 压缩时释放 GIL）
      3 拼成一条 zlib 流（头 + 各段 + 合并的 adler32），写入单个 IDAT
      与 pigz 的并行压缩同理；比单线程 zlib 快约线程数倍，体积与 level 1 相当
支持  uint8 / uint16，1 / 3 / 4 通道（RGB 顺序）；可附带输入 PNG 的色彩块（sRGB / iCCP / gAMA / cHRM / pHYs）原样写回
"""
import struct, zlib
import concurrent.futures as cf
import numpy as np

_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix='png')


def _chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)


COLOR_CHUNKS = (b'sRGB', b'iCCP', b'gAMA', b'cHRM', b'pHYs')


def png_color_chunks(data):
    """取输入 PNG 中位于 IDAT 之前的色彩相关块（原始字节，含长度与 CRC），供输出原样写回。
    输出总是 RGB(A)：输入为灰度 / 调色板时不带 iCCP（灰度 ICC 配置不能用于彩色图）"""
    rgb_in = len(data) > 25 and data[25] in (2, 6)
    out, i = [], 8
    while i + 8 <= len(data):
        n = struct.unpack('>I', data[i:i + 4])[0]; tag = data[i + 4:i + 8]
        if tag == b'IDAT' or tag == b'IEND' or i + 12 + n > len(data):
            break
        if tag in COLOR_CHUNKS and (tag != b'iCCP' or rgb_in):
            out.append(data[i:i + 12 + n])
        i += 12 + n
    return out


def _paeth_rows(rows, prev, bpp):
    """rows: (h, R) uint8；prev: 上一行（R,）或 None。返回带滤波类型字节（4 = Paeth）的 (h, 1 + R) uint8"""
    x = rows.astype(np.int16)
    h, R = x.shape
    up_ = np.empty_like(x)
    up_[1:] = x[:-1]
    up_[0] = prev if prev is not None else 0
    a = np.zeros_like(x); a[:, bpp:] = x[:, :-bpp]                 # 左
    c = np.zeros_like(x); c[:, bpp:] = up_[:, :-bpp]               # 左上
    pa = np.abs(up_ - c); pb = np.abs(a - c); pc = np.abs(a + up_ - 2 * c)
    pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, up_, c))
    out = np.empty((h, R + 1), np.uint8)
    out[:, 0] = 4
    out[:, 1:] = (x - pred).astype(np.uint8)                       # 模 256
    return out


def _deflate(buf, level):
    co = zlib.compressobj(level, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY)
    return co.compress(buf) + co.flush(zlib.Z_FULL_FLUSH)


def encode_png(img, level=1, parts=8, extra_chunks=()):
    """img: (H, W[, C]) uint8 / uint16，通道为 RGB / RGBA / 灰度；extra_chunks：原样插在 IHDR 之后的完整块"""
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape
    depth = 16 if img.dtype == np.uint16 else 8
    color = {1: 0, 3: 2, 4: 6}[C]
    raw = img.astype('>u2').view(np.uint8) if depth == 16 else np.ascontiguousarray(img, np.uint8)
    rows = raw.reshape(H, W * C * depth // 8)
    bpp = C * depth // 8
    edges = np.linspace(0, H, parts + 1).astype(int)

    def seg(i):                                          # 每段：滤波（带上一段最后一行）+ 压缩，均在线程内
        r0, r1 = edges[i], edges[i + 1]
        f = _paeth_rows(rows[r0:r1], rows[r0 - 1] if r0 > 0 else None, bpp).tobytes()
        return f, _deflate(f, level)
    res = list(_POOL.map(seg, range(parts)))
    comp = [c for _, c in res]
    adler = 1
    for f, _ in res:
        adler = zlib.adler32(f, adler)
    tail = zlib.compressobj(level, zlib.DEFLATED, -15).flush(zlib.Z_FINISH)   # 空的末块（BFINAL=1）
    zdata = b'\x78\x01' + b''.join(comp) + tail + struct.pack('>I', adler & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', W, H, depth, color, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr) + b''.join(extra_chunks) + _chunk(b'IDAT', zdata) + _chunk(b'IEND', b'')
