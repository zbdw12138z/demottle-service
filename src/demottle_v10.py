"""
demottle_v10.py — GPT Image 系生图的去斑驳 / 去蜡感 / 去棕色污渍后处理

问题
  生图按 token 块独立采样再解码，块间低频偏差在墙面、天空、皮肤等平坦表面上形成 60–200px 的
  斑驳（mottle）；斑驳暗处同时偏黄偏红，被人眼读成"脏"。原生 PNG 另带全图相位锁定的 16px 周期指纹。

原理
  只做减法，不生成内容：在"物体边缘"（堤坝）之间估一张平滑表面，与原图之差即斑驳候选；
  再用幅度、取向、质感等门控区分斑驳与真实结构，只减掉判为斑驳的部分（默认 80%）。
  纯算法，无训练、无模型权重；同一输入两次运行逐像素一致。

依赖   numpy, opencv-python, scipy, pillow
       face_detection_yunet_2023mar.onnx 放在同目录（人脸检测；缺失时退回 Haar，精度下降）
用法   python demottle_v10.py in.png out.png [--16bit]
       --16bit 输出 16 位 PNG、不加抖动，供下游转 10 位 HEIC
链路   优先处理 API 原始 PNG，全链路只编码一次（10 位 HEIC，或 JPEG q≥92 4:4:4）

流程
  1 指纹    PNG 输入：折叠估 16px 周期模板，四象限一致才整图精确减去
  2 分析    斑驳尺度 σ（中间调最平区块）、是否 JPEG、JPEG 量化步长、尺度因子 s = 短边/1536
  3 堤坝    判定"物体边缘"：梯度够陡 + 邻域反差够大 + 连成线；PNG 并入局部信噪比判据
  4 平面遍  堤坝内扩散 → 残差 m → 软物体钉住 → 幅度 / 取向门控 → 减去 80%
  5 皮肤遍  人脸锚定的肤色区：五官成堤；按质感门施加——毛孔清晰的皮肤不动，蜡感皮肤清亮度与色块
  6 细节    JPEG：平坦区细节层换成胶片式颗粒；PNG：保留原细节；8 位输出前 ±0.5 级抖动
  7 保险    量不到斑驳（无平坦区块或 σ 异常大）→ 原图返回

约定   内部颜色空间为 Lab，L 归一到 0–1，a/b 除以 128；像素类参数均乘 s，适配任意分辨率
"""
import numpy as np, cv2
from scipy.ndimage import distance_transform_edt


# ================================================================= 基础算子
def gauss(x, s):
    return cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)


def grad_mag(x, s):
    """先以 s 平滑再求梯度幅值"""
    xe = gauss(x, s)
    gx = cv2.Sobel(xe, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(xe, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def normconv(x, M, s):
    """归一化卷积：只用 M 覆盖的像素求加权平均"""
    return gauss(x * M, s) / np.maximum(gauss(M, s), 1e-6)


def hysteresis(strong, weak, min_area=0):
    """滞后连通：保留含强种子的弱连通块，面积不足 min_area 的丢弃"""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    seeds = np.unique(lbl[strong > 0]); seeds = seeds[seeds != 0]
    if min_area > 0:
        seeds = seeds[st[seeds, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(lbl, seeds)


def masked_diffuse(base, M, n_iter, s=1.0, state=None, return_state=False):
    """堤坝内扩散：M=1 处反复做归一化卷积，M=0 处（堤坝）保持原值。
    等效平滑半径约 s·√n_iter。state 用于从已有迭代状态续算（色度长扩散复用亮度的前段）。"""
    M3 = M[..., None]
    den = np.maximum(gauss(M, s), 1e-6)[..., None]          # 分母与迭代无关，只算一次
    SM = np.ascontiguousarray(base * M3 if state is None else state, dtype=np.float32)
    for _ in range(n_iter):
        num = gauss(SM, s)
        np.divide(num, den, out=SM)
        SM *= M3
    S = np.where(M3 > 0, SM, base)
    return (S, SM) if return_state else S


def coherence(band, s):
    """结构张量相干度：0 = 各向同性（斑驳），1 = 单一取向（雨丝、织物纹、木纹、发丝）"""
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = gauss(gx * gx, s), gauss(gy * gy, s), gauss(gx * gy, s)
    tr = jxx + jyy; det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.clip(tr * tr - 4 * det, 0, None))
    return disc / (tr + 1e-9)


def local_median(x, win, f=4):
    """局部中值（在 1/f 分辨率上算以提速）"""
    from scipy.ndimage import median_filter
    h, w = x.shape
    small = cv2.resize(x, (w // f, h // f), interpolation=cv2.INTER_AREA)
    small = median_filter(small, size=max(3, win // f), mode='reflect')
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def mad(v):
    """稳健标准差（中位数绝对偏差 × 1.4826）"""
    v = np.asarray(v, np.float32)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


# ================================================================= 1–2 指纹与分析
def jpeg_quant_steps(path):
    """读 JPEG 量化表的 DC 步长（亮度, 色度）；非 JPEG 返回 None"""
    if path is None:
        return None
    try:
        from PIL import Image
        q = Image.open(path).quantization
        if not q:
            return None
        return float(q[0][0]), float(q[min(1, max(q.keys()))][0])
    except Exception:
        return None


def blockiness(L):
    """8px 格线上的梯度 / 格内梯度；> 1.15 视为有 JPEG 块效应"""
    dx = np.abs(np.diff(L, axis=1)); dy = np.abs(np.diff(L, axis=0))
    on = np.concatenate([dx[:, 7::8].ravel(), dy[7::8, :].ravel()])
    off = np.concatenate([dx[:, 3::8].ravel(), dy[3::8, :].ravel()])
    return float(np.mean(on) / (np.mean(off) + 1e-9))


def sigma_by_tiles(band, B, frac=0.10, Lmap=None, lo=0.12, hi=0.90):
    """斑驳尺度 σ：分 B×B 区块算频带 MAD，取最平 frac 的中位数。
    给 Lmap 时只用中间调区块（lo–hi）：深暗部和高光被色调曲线压扁，量出来会偏小数倍。"""
    H, W = band.shape; vals = []
    for y in range(0, H - B + 1, B):
        for x in range(0, W - B + 1, B):
            if Lmap is not None:
                m = float(Lmap[y:y + B, x:x + B].mean())
                if m < lo or m > hi:
                    continue
            vals.append(mad(band[y:y + B, x:x + B].ravel()))
    if len(vals) < 12 and Lmap is not None:          # 中间调区块太少（整张极暗 / 极亮）：退回全图
        return sigma_by_tiles(band, B, frac)
    vals = np.sort(np.array(vals))
    k = max(3, int(len(vals) * frac))
    return float(np.median(vals[:k])), vals


def mottle_period(L, B, frac=0.10, pmin=24, pmax=200):
    """斑驳周期：最平区块上带通亮度的自相关，第一次过零距离 ≈ 周期 / 4"""
    band = gauss(L, 3) - gauss(L, 80)
    H, W = L.shape; tiles = []
    for y in range(0, H - B + 1, B):
        for x in range(0, W - B + 1, B):
            t = band[y:y + B, x:x + B]; tiles.append((mad(t.ravel()), y, x))
    tiles.sort(key=lambda t: t[0])
    tiles = tiles[:max(3, int(len(tiles) * frac))]
    acc = np.zeros(B // 2, np.float64); n = 0
    for _, y, x in tiles:
        t = band[y:y + B, x:x + B]; t = t - t.mean()
        for k in range(B // 2):
            acc[k] += (t[:, :B - k] * t[:, k:]).mean() + (t[:B - k, :] * t[k:, :]).mean()
        n += 2
    acc /= max(n, 1); acc /= acc[0] + 1e-12
    if not np.any(acc < 0):
        return 80.0                       # 区块内无过零：估不出，用经验值（实测 67–86px）
    zero = int(np.argmax(acc < 0))
    return float(np.clip(4 * zero, pmin, pmax))


def periodic_fingerprint(img_u8, P=16, flat_pct=30):
    """解码器周期指纹：全图相位锁定的 P×P 加性图案。
    在平坦像素上按 P 折叠估模板；四象限模板一致（相关 ≥ 0.6）且峰峰值 ≥ 0.8 级才逐通道整图减去。
    这是精确减法，不是滤波；经过重采样的图相位被打散，自动跳过。返回 (图, 相关, 峰峰值)。"""
    x = img_u8.astype(np.float32); H, W = x.shape[:2]
    g = x.mean(-1); hp = g - gauss(g, 4); fe = gauss(hp * hp, 8)
    flat = (fe <= np.percentile(fe, flat_pct)).astype(np.float32)
    def fold(hpc, y0, y1, x0, x1):
        h = hpc[y0:y1, x0:x1]; m = flat[y0:y1, x0:x1]; hh = (y1 - y0) // P * P; ww = (x1 - x0) // P * P
        num = (h * m)[:hh, :ww].reshape(hh // P, P, ww // P, P).sum((0, 2))
        den = m[:hh, :ww].reshape(hh // P, P, ww // P, P).sum((0, 2))
        return num / np.maximum(den, 1)
    q = [fold(hp, 0, H // 2, 0, W // 2), fold(hp, 0, H // 2, W // 2, W), fold(hp, H // 2, H, 0, W // 2), fold(hp, H // 2, H, W // 2, W)]
    corr = float(np.mean([np.corrcoef(q[i].ravel(), q[j].ravel())[0, 1] for i in range(4) for j in range(i + 1, 4)]))
    T0 = fold(hp, 0, H, 0, W); p2p = float(T0.max() - T0.min())
    if corr < 0.6 or p2p < 0.8:
        return img_u8, corr, 0.0
    out = x.copy()
    for c in range(3):
        hpc = x[..., c] - gauss(x[..., c], 4)
        T = fold(hpc, 0, H, 0, W); T -= T.mean()
        out[..., c] -= np.tile(T, (H // P + 1, W // P + 1))[:H, :W]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), corr, p2p


def analyze(img_u8, path=None):
    """全图统计量。返回 ctx：lab、尺度 s、是否 JPEG、量化步长 q、斑驳 σ（L/a/b）、周期、ok（是否可测）"""
    img = img_u8.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB); lab[..., 0] /= 100.0; lab[..., 1:] /= 128.0
    H, W = lab.shape[:2]; L = lab[..., 0]
    s = min(H, W) / 1536.0
    B = max(32, int(64 * s))
    q = jpeg_quant_steps(path)
    blk = blockiness(L)
    jpeg = (q is not None) or blk > 1.15
    sig = []; flat_vals = None
    for c in range(3):
        band = gauss(lab[..., c], 3 * s) - gauss(lab[..., c], 20 * s)
        v, vals = sigma_by_tiles(band, B, Lmap=L)
        sig.append(v)
        if c == 0: flat_vals = vals
    sig = np.array(sig, np.float32)
    if q is not None:
        qL, qC = q[0] / 255.0, q[1] / 255.0 * 1.3          # 量化步长换算到归一化 L / ab 单位
    else:
        qL, qC = (0.006, 0.010) if jpeg else (0.0, 0.0)
    period = mottle_period(L, max(64, int(128 * s)))
    k = max(3, len(flat_vals) // 10)
    ok = (sig[0] <= 0.015) and (flat_vals[:k].max() <= 0.02)   # 最平区块也很"花"→ 量不到斑驳，不处理
    return dict(lab=lab, H=H, W=W, s=s, jpeg=jpeg, blk=blk, q=(qL, qC), sigma=sig, period=period, ok=ok)


# ================================================================= 3 堤坝（物体边缘）
def unified_barrier(lab, ctx, lo=6.0, hi=9.0, dilate=5):
    """判定哪些像素是真实边缘（不参与扩散、不被处理）。三个条件同时满足：
      陡   梯度 ≥ lo / hi 个"斑驳台阶梯度"（u = 0.16·σ·8，斑驳自身能产生的梯度量级），亮度或色度任一；
      反差 15px 邻域峰峰值 > 3σ（PNG）/ 6σ（JPEG），且 > 3 倍量化步长——挡掉 JPEG 色阶等高线；
      成线 滞后连通后延伸长度 ≥ 24px——噪声触发的孤立小点不算边（紧凑小物体交给后面的软物体钉住）。
    PNG 另并入局部信噪比判据：细尺度梯度 / 周围中值，保住暗墙前虚焦的低反差物体（如暗叶子）。"""
    s = ctx['s']; sig = ctx['sigma']; qL, qC = ctx['q']
    es = 2.5 * s
    L = lab[..., 0]
    g = grad_mag(L, es); gc = np.maximum(grad_mag(lab[..., 1], es), grad_mag(lab[..., 2], es))
    if ctx['jpeg']:                                         # 8px 格线上的梯度减半，避免块边成堤
        H, W = L.shape; gridm = np.ones((H, W), np.float32)
        for k in range(7, W, 8): gridm[:, max(k - 1, 0):k + 2] = 0.5
        for k in range(7, H, 8): gridm[max(k - 1, 0):k + 2, :] *= 0.5
        g = g * gridm; gc = gc * gridm
    u = 0.16 * sig[0] * 8.0 + 1e-9
    uc = 0.16 * max(sig[1], sig[2]) * 8.0 + 1e-9
    r = np.maximum(g / u, gc / uc)
    if not ctx['jpeg']:
        gs = grad_mag(L, 1.6 * s)
        r_snr = gs / (local_median(gs, int(61 * s)) + 1e-6)
        gcs = np.maximum(grad_mag(lab[..., 1], 1.6 * s), grad_mag(lab[..., 2], 1.6 * s))
        r_snr = np.maximum(r_snr, gcs / (local_median(gcs, int(61 * s)) + 1e-6))
    else:
        r_snr = np.zeros_like(r)
    k = np.ones((int(15 * s) | 1,) * 2, np.uint8)
    def rng(x):
        b = gauss(x, es); return cv2.dilate(b, k) - cv2.erode(b, k)
    cm = 6.0 if ctx['jpeg'] else 3.0
    contrast = (rng(L) > max(cm * sig[0], 3.0 * qL)) | \
               (np.maximum(rng(lab[..., 1]), rng(lab[..., 2])) > max(cm * max(sig[1], sig[2]), 3.0 * qC))
    r = r * contrast; r_snr = r_snr * contrast
    bar = hysteresis((r > hi) | (r_snr > 4.0), (r > lo) | (r_snr > 2.0), int(40 * s * s))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(bar.astype(np.uint8), connectivity=8)
    ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    keep = np.where(ext >= max(12, int(24 * s)))[0]; keep = keep[keep != 0]
    bar = np.isin(lbl, keep)
    return cv2.dilate(bar.astype(np.uint8), np.ones((2 * int(dilate * s) + 1,) * 2, np.uint8)).astype(bool)


# ================================================================= 人脸与皮肤区域
_YUNET = None
def _yunet_path():
    """模型路径：环境变量 DEMOTTLE_YUNET 优先，其次脚本同目录"""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.environ.get('DEMOTTLE_YUNET', ''), os.path.join(here, 'face_detection_yunet_2023mar.onnx')):
        if p and os.path.exists(p):
            return p
    return None


def detect_faces(rgb_u8, score=0.7):
    """人脸框 (x, y, w, h)，原图坐标。YuNet 在长边 640 上检测；无模型时退回 Haar（正脸 + 双向侧脸）。"""
    global _YUNET
    H, W = rgb_u8.shape[:2]
    path = _yunet_path()
    if path is not None:
        try:
            if _YUNET is None:
                _YUNET = cv2.FaceDetectorYN.create(path, '', (320, 320), score, 0.3, 50)
            sc = 640.0 / max(H, W)
            small = cv2.resize(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
            _YUNET.setInputSize((small.shape[1], small.shape[0]))
            _, f = _YUNET.detect(small)
            return [] if f is None else [tuple(float(v) / sc for v in r[:4]) for r in f]
        except Exception:
            pass
    faces = []
    try:
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        ms = (int(0.08 * min(H, W)),) * 2
        for xml in ('haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml'):
            det = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
            faces += list(det.detectMultiScale(gray, 1.1, 5, minSize=ms))
        det = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        for (x, y, w, h) in det.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=ms):
            faces.append((W - x - w, y, w, h))
    except Exception:
        faces = []
    return faces


def face_anchored_skin(rgb_u8, s):
    """皮肤软掩码：YCrCb 肤色块 ∩ 人脸框（向下延伸到颈部）。无脸则返回全零——
    只凭肤色会把暖色墙、木头、头发误当皮肤。返回 (掩码, 人脸数)。"""
    H, W = rgb_u8.shape[:2]
    ycc = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((Cr > 133) & (Cr < 178) & (Cb > 77) & (Cb < 132) & (Y > 35)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((int(7 * s) | 1,) * 2, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(15 * s) | 1,) * 2, np.uint8))
    faces = detect_faces(rgb_u8)
    if len(faces) == 0:
        return np.zeros((H, W), np.float32), 0
    anchor = np.zeros((H, W), np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(anchor, (int(x - 0.3 * w), int(y - 0.3 * h)), (int(x + 1.3 * w), int(y + 1.8 * h)), 1, -1)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # 肤色块与脸框重叠 ≥ 脸框面积 15% 即保留（脸常与颈、手臂连成一块，按块内占比筛会整块丢掉）；
    # 随后与脸框取交集，框外部分不算皮肤
    A_anchor = float(anchor.sum()) + 1e-9
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.001 * H * W and float(anchor[lbl == i].sum()) / A_anchor >= 0.15]
    S = np.isin(lbl, keep).astype(np.float32) * anchor.astype(np.float32)
    return gauss(S, 6 * s), len(faces)


# ================================================================= 4 平面遍（皮肤遍复用）
def surface_sigma(m_s, M_s, sig_global, min_px=400, lo=0.7, hi=4.0):
    """按表面定尺子：堤坝之间每个连通区域用自己的残差估斑驳幅度（|m| 均值 × 1.2533，高斯下等于 σ），
    夹在全局值的 [lo, hi] 倍。斑驳偏强的墙面不会因此整面被当成"物体"。"""
    n, lbl = cv2.connectedComponents((M_s > 0).astype(np.uint8), connectivity=4)
    cnt = np.bincount(lbl.ravel(), minlength=n).astype(np.float64)
    out = np.empty(m_s.shape, np.float32)
    for c in range(m_s.shape[-1]):
        mean_abs = np.bincount(lbl.ravel(), np.abs(m_s[..., c]).ravel(), minlength=n) / np.maximum(cnt, 1)
        sc = np.clip(1.2533 * mean_abs, lo * sig_global[c], hi * sig_global[c])
        sc[cnt < min_px] = sig_global[c]; sc[0] = sig_global[c]
        out[..., c] = sc[lbl]
    return out


def region_pass(lab, ctx, block, sig_diff, T_near, T_far, far_px, pin_seed, fade_px,
                grain_sigma, restrict=None, chroma_boost=True, sig_override=None, gate='all'):
    """在 block（True = 不处理）之外估斑驳场 m_final；调用方做 base − 力度·m_final + grain。
      sig_diff        扩散平滑半径（px），决定能抓多大的斑驳
      T_near / T_far  幅度门限（σ 倍数）：贴近堤坝用 T_near，离堤坝 far_px 以外放宽到 T_far
      pin_seed        软物体钉住的种子门限（T² 的倍数）
      fade_px         物体下方斑驳的外推淡出距离
      grain_sigma     base / grain 分层尺度：grain（细节层）原样保留，只在 base 上做减法
      restrict        额外的 0–1 权重（平面遍传 1−皮肤，皮肤遍传皮肤掩码）
      chroma_boost    色度用 2.5 倍长的扩散（材质内色度应接近常数，可去得更干净）
      sig_override    给定则用固定 σ（皮肤遍沿用平面遍的值），否则按表面自估
      gate            'all'：亮度 + 色度联合判定；'L'：只看亮度（皮肤遍用）
    返回 dict(base, grain, m, w, M, dist_b, sig)。"""
    H, W, s = ctx['H'], ctx['W'], ctx['s']
    base = np.stack([gauss(lab[..., c], grain_sigma) for c in range(3)], -1)
    ds = 0.25                                                    # 扩散在 1/4 分辨率上做
    hs, ws = int(H * ds), int(W * ds)
    base_s = cv2.resize(base, (ws, hs), interpolation=cv2.INTER_AREA)
    # 堤坝降采样：max 池化保住细线的连续性（扩散不漏过细边），但不额外膨胀（否则小堤坝连片屏蔽墙面）
    bar_s = cv2.resize(cv2.dilate(block.astype(np.uint8), np.ones((3, 3), np.uint8)), (ws, hs), interpolation=cv2.INTER_NEAREST).astype(bool) | \
            (cv2.resize(block.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA) > 0.35)
    M_s = (~bar_s).astype(np.float32)
    if M_s.sum() < 200:
        return None
    n_iter = max(9, int((sig_diff * ds) ** 2))
    n_iter_c = int(n_iter * 2.5) if chroma_boost else n_iter
    # 两遍：第 1 遍找出软物体并钉住（从可处理区剔除），第 2 遍在剩余区域重估
    for p in range(2):
        S_s, SM_s = masked_diffuse(base_s, M_s, n_iter, 1.0, return_state=True)
        m_s = (base_s - S_s) * M_s[..., None]                    # 斑驳候选 = 原图 − 堤坝内平滑面
        if sig_override is None:
            flat = M_s > 0
            sig = np.array([mad(m_s[..., c][flat]) + 1e-9 for c in range(3)], np.float32)
            sigmap = surface_sigma(m_s, M_s, sig)
        else:
            sig = np.asarray(sig_override, np.float32)
            sigmap = np.broadcast_to(sig, m_s.shape)
        # 偏离度 d²（σ 单位）；三通道联合时色度全权重——暗墙前的绿叶靠色度才立得起来
        d2_s = np.sum((m_s / sigmap) ** 2, axis=-1) if gate == 'all' else (m_s[..., 0] / sigmap[..., 0]) ** 2
        if chroma_boost:   # 色度残差改用更长的扩散（从亮度的 n_iter 处续算）；判定 d² 仍用上面同尺度的值
            S_c = masked_diffuse(base_s[..., 1:], M_s, n_iter_c - n_iter, 1.0, state=SM_s[..., 1:].copy())
            m_s[..., 1:] = (base_s[..., 1:] - S_c) * M_s[..., None]
        if p == 0:
            # 软物体钉住：成片超出门限的偏离（虚焦物体、柔和阴影）视为物体，移出可处理区
            dist_s = distance_transform_edt(M_s > 0).astype(np.float32)
            T_s = T_near + (T_far - T_near) * np.clip(dist_s / (far_px * s * ds), 0, 1)
            soft = hysteresis(d2_s > pin_seed * T_s ** 2, d2_s > T_s ** 2, int(40 * s * s))
            M_s = M_s * (1 - cv2.dilate(soft.astype(np.uint8), np.ones((3, 3), np.uint8)))
    m = cv2.resize(m_s, (W, H), interpolation=cv2.INTER_LINEAR)
    d2 = cv2.resize(d2_s, (W, H), interpolation=cv2.INTER_LINEAR)
    M = cv2.resize(M_s, (W, H), interpolation=cv2.INTER_LINEAR)
    dist_b = distance_transform_edt(M > 0.5).astype(np.float32)
    T = T_near + (T_far - T_near) * np.clip(dist_b / (far_px * s), 0, 1)
    R1 = restrict if restrict is not None else np.ones((H, W), np.float32)
    # 取向门：斑驳没有取向，有取向的是真实纹理。检验对象是将被减掉的残差 m（不用原图带通——
    # 其中光照渐变的残余带方向，会把侧光墙面整面误判为纹理）；细频带另做一路
    fine_b = lab[..., 0] - gauss(lab[..., 0], 3 * s)
    c_res = cv2.resize(coherence(m_s[..., 0], 20 * s * ds), (W, H), interpolation=cv2.INTER_LINEAR)
    iso = gauss(np.minimum(np.clip((0.55 - c_res) / 0.20, 0, 1),
                           np.clip((0.60 - coherence(fine_b, 12 * s)) / 0.20, 0, 1)), 6 * s)
    R1 = R1 * iso
    # 亮度权重：幅度门 × 可处理区 × 限制项；min(w, blur(w)) 只向不处理区平滑过渡，不漫进物体
    w0 = np.clip(1 - d2 / T ** 2, 0, 1) * M * R1
    w = np.minimum(w0, gauss(w0, 10 * s))
    # 物体下方的斑驳：用周围已确认的斑驳外推填补，离开处理区后在 fade_px 内淡出（消除接缝）
    fill = np.stack([normconv(m[..., c] * w, w, 10 * s) for c in range(3)], -1)
    dist = distance_transform_edt(w < 0.5).astype(np.float32)
    fade = gauss(np.clip(1 - dist / (fade_px * s), 0, 1), 8 * s) * R1
    m_final = w[..., None] * m + (1 - w[..., None]) * fill * fade[..., None]
    # 色度权重：门限放宽 1.5 倍，但联合判定已是物体处（d² > 4T²）不放宽，否则暗墙前的绿叶会被去色
    d2_ab = (m[..., 1] / (sig[1] + 1e-9)) ** 2 + (m[..., 2] / (sig[2] + 1e-9)) ** 2
    wc0 = np.clip(1 - d2_ab / (1.5 * T) ** 2, 0, 1) * np.clip(1 - d2 / (2 * T) ** 2, 0, 1) * M * R1
    w_ab = np.maximum(w, np.minimum(wc0, gauss(wc0, 10 * s)))
    for c in (1, 2):
        m_final[..., c] = w_ab * m[..., c] + (1 - w_ab) * fill[..., c] * fade
    return dict(base=base, grain=lab - base, m=m_final, w=w, M=M, dist_b=dist_b, sig=sig)



# ================================================================= 主函数
def demottle_v10(img_u8, path=None, strength=0.8, skin_strength=0.85, clean_far=4.5, out_bits=8, T_near=3.0, T_far=6.0,
                far_px=40, pin_seed=4.0, fade_px=80, verbose=True, seed=0):
    """img_u8   RGB uint8
    path          源文件路径（用于读 JPEG 量化表；可为 None）
    strength      平面斑驳减去比例；留 20% 避免"腻子感"
    skin_strength 蜡感皮肤的最大减去比例（实际再乘质感门）
    clean_far     PNG 离堤坝远处的幅度门限（σ 倍数）；JPEG 用 T_far
    out_bits      8：返回 uint8（含抖动）；16：返回 uint16（不抖动）
    seed          颗粒与抖动的随机种子，固定则结果可复现
    返回 (RGB 图, ctx)"""
    rng = np.random.default_rng(seed)
    # 1 指纹：只在未压缩输入上做（JPEG 的 8px 块会被误判成指纹）
    fp_corr, fp_p2p = 0.0, 0.0
    if jpeg_quant_steps(path) is None:
        _L0 = cv2.cvtColor(img_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[..., 0] / 100.0
        if blockiness(_L0) <= 1.15:
            img_u8, fp_corr, fp_p2p = periodic_fingerprint(img_u8)
    # 2 分析
    ctx = analyze(img_u8, path)
    H, W, s = ctx['H'], ctx['W'], ctx['s']; lab = ctx['lab']; sig = ctx['sigma']
    if verbose:
        print(f"fingerprint 16px corr {fp_corr:.2f} → {'removed, p2p %.2f lvl' % fp_p2p if fp_p2p else 'skipped'}")
        print(f"scale {s:.2f} | {'JPEG' if ctx['jpeg'] else 'clean'} (blockiness {ctx['blk']:.2f}, qDC {ctx['q'][0]*255:.0f}) | "
              f"σ L/a/b {sig[0]:.4f}/{sig[1]:.4f}/{sig[2]:.4f} | period {ctx['period']:.0f}px | measurable {ctx['ok']}")
    if not ctx['ok']:
        if verbose: print("no measurable mottle → untouched")
        return img_u8.copy(), ctx
    sig_diff = float(np.clip(0.6 * ctx['period'], 30, 60)) * s          # 扩散半径随斑驳周期
    grain_sigma = (3.0 if ctx['jpeg'] else 1.6) * s                      # JPEG 细节层更厚（含块纹理）

    # 3 堤坝与皮肤区域
    barrier = unified_barrier(lab, ctx)
    S_skin, n_faces = face_anchored_skin(img_u8, s)

    # 4 平面遍：皮肤区域留给皮肤遍。离堤坝远处门限放宽——JPEG 到 T_far（块效应抹掉了软物体的边，
    #   斑驳重尾更厚），PNG 到 clean_far（再宽会开始削虚焦的低反差物体）
    T_far_eff = T_far if ctx['jpeg'] else (clean_far if clean_far else T_near)
    R = region_pass(lab, ctx, barrier, sig_diff, T_near, T_far_eff, far_px, pin_seed, fade_px, grain_sigma,
                    restrict=np.clip(1 - S_skin, 0, 1))
    if R is not None:
        out = R['base'] - strength * R['m'] + R['grain']
    else:
        out = lab.copy()
        R = dict(base=out.copy(), grain=np.zeros_like(out), dist_b=np.full((H, W), 1e3, np.float32))

    # 5 皮肤遍
    if n_faces > 0 and S_skin.max() > 0.5:
        L = out[..., 0]
        g = grad_mag(L, 2.5 * s)
        # 五官成堤：以皮肤自身梯度中位数为单位（随脸走，不随图走），眼、眉、唇、鼻翼在其 4–8 倍以上
        u_skin = float(np.median(g[S_skin > 0.5])) + 1e-9
        feat = hysteresis(g / u_skin > 8.0, g / u_skin > 4.0, int(60 * s * s))
        feat = cv2.dilate(feat.astype(np.uint8), np.ones((int(9 * s) | 1,) * 2, np.uint8)).astype(bool)
        block = feat | (S_skin < 0.5)
        # 质感门：3px 以下细纹理（毛孔、胡茬）局部 RMS ≥ 1.0% → 0（不动）；≤ 0.5% → 1（蜡感，全力）
        fine_rms = np.sqrt(gauss((out[..., 0] - gauss(out[..., 0], 3.0 * s)) ** 2, 25 * s)) * 100.0
        texture_gate = gauss(np.clip((1.0 - fine_rms) / 0.5, 0, 1), 10 * s)
        st = skin_strength * texture_gate
        sig_main = R.get('sig', None)
        # 两个尺度：脸颊级斑块（0.4 × 周期）与 15px 小块蜡感；门控只看亮度（σ 沿用平面遍）
        for sd, T_far_skin, fade in ((float(np.clip(0.4 * ctx['period'], 20, 40)) * s, 5.0, 20), (15.0 * s, 4.0, 12)):
            Rs = region_pass(out, ctx, block, sd, T_near, T_far_skin, 30, pin_seed, fade, 3.0 * s,
                             restrict=S_skin, chroma_boost=False, sig_override=sig_main, gate='L')
            if Rs is None:
                break
            out[..., 0] = Rs['base'][..., 0] - st * Rs['m'][..., 0] + Rs['grain'][..., 0]
            # 色度（赭黄 / 粉色块）同样只在蜡感皮肤上清；偏差超过 max(3σ, 2.5 ΔE) 的不动（雀斑、嘴唇、泛红）
            mc = Rs['m'][..., 1:] * 128.0
            dEm = np.hypot(mc[..., 0], mc[..., 1])
            act = (S_skin > 0.7) & (texture_gate > 0.5) & (dEm > 0)
            sig_c = float(np.median(dEm[act])) / 1.1774 if act.sum() > 2000 else 1.0   # 二维瑞利：中位数 → σ
            T_c = max(3.0 * sig_c, 2.5)
            w_c = st * np.clip(1 - (dEm / T_c) ** 2, 0, 1)
            for c in (1, 2):
                out[..., c] = Rs['base'][..., c] - w_c * Rs['m'][..., c] + Rs['grain'][..., c]
        if verbose:
            print(f"skin: {n_faces} face(s), mask {S_skin.mean()*100:.1f}%, feature-barrier in skin {(feat & (S_skin > 0.5)).sum() / max((S_skin > 0.5).sum(), 1) * 100:.1f}%")

    # 6 细节层
    Lref = np.clip(lab[..., 0], 0, 1)
    if ctx['jpeg']:
        # JPEG 平坦区的细节层是块纹理：离堤坝 24px 以上、非皮肤、无取向处换成胶片式颗粒
        # （0.7px 相关、暗部略多、纯黑不加）；全图另叠一层弱颗粒，处理区与未处理区纹理连续
        fine_L = lab[..., 0] - gauss(lab[..., 0], 3.0 * s)
        iso_fine = gauss(np.clip((0.55 - coherence(fine_L, 12 * s)) / 0.20, 0, 1), 6 * s)
        interior = (gauss(np.clip(R['dist_b'] / (24 * s), 0, 1), 10 * s) * (1 - S_skin) * iso_fine)[..., None]
        sel = R['dist_b'] > 24 * s
        flat_amp = np.array([mad(R['grain'][..., c][sel]) if sel.any() else 0 for c in range(3)], np.float32)
        amp = np.maximum(flat_amp * 0.6, np.array([0.0030, 0.0010, 0.0010], np.float32))
        n = np.stack([gauss(rng.normal(size=(H, W)).astype(np.float32), 0.7 * s) for _ in range(3)], -1)
        n /= n.std(axis=(0, 1), keepdims=True)
        shade = (np.clip(Lref / 0.08, 0, 1) * (1.2 - 0.4 * Lref))[..., None]
        syn = n * amp * shade; syn[..., 1:] *= 0.3
        out = out - R['grain'] * interior + syn * interior + 0.4 * syn * (1 - interior)
    if out_bits == 8:   # ±0.5 级抖动，防止抹平后的渐变量化成色阶；16 位输出不需要
        out[..., 0] += (rng.random((H, W)).astype(np.float32) - 0.5) * 0.004 * np.clip(Lref / 0.05, 0, 1)

    out[..., 0] *= 100.0; out[..., 1:] *= 128.0
    rgbf = np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2RGB), 0, 1)
    rgb = (rgbf * 255 + 0.5).astype(np.uint8)
    if out_bits == 16:
        if verbose:
            metrics(img_u8, rgb, ctx, barrier)
        return (rgbf * 65535 + 0.5).astype(np.uint16), ctx
    if verbose:
        metrics(img_u8, rgb, ctx, barrier)
    return rgb, ctx


def metrics(a_u8, b_u8, ctx, barrier):
    """打印自检：斑驳 σ 降幅（全区块口径）、堤坝处梯度保持率、平均改动（8 位级）"""
    s = ctx['s']; B = max(32, int(64 * s))
    La = cv2.cvtColor(a_u8.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0] / 100
    Lb = cv2.cvtColor(b_u8.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0] / 100
    ba = gauss(La, 3 * s) - gauss(La, 40 * s); bb = gauss(Lb, 3 * s) - gauss(Lb, 40 * s)
    sa, _ = sigma_by_tiles(ba, B); sb, _ = sigma_by_tiles(bb, B)
    ga, gb = grad_mag(La, 1.6 * s), grad_mag(Lb, 1.6 * s)
    keep = float(gb[barrier].sum() / (ga[barrier].sum() + 1e-9))
    print(f"metrics: mottle σ {sa:.4f} → {sb:.4f} ({(1-sb/sa)*100:.0f}% removed) | edge gradient kept {keep*100:.1f}% | mean |Δ| {np.abs(a_u8.astype(np.float32)-b_u8).mean():.2f} levels")


if __name__ == "__main__":
    import sys, time
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    bits = 16 if '--16bit' in sys.argv else 8
    im = cv2.cvtColor(cv2.imread(args[0]), cv2.COLOR_BGR2RGB)
    t = time.time(); out, _ = demottle_v10(im, path=args[0], out_bits=bits); print(f"{time.time()-t:.1f}s")
    cv2.imwrite(args[1], cv2.cvtColor(out, cv2.COLOR_RGB2BGR))     # --16bit 时输出文件须为 .png
