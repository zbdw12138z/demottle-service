"""
demottle_v13.py — 实验版：v12 + 可选的效果改进开关（默认全关，此时与 v12 逐位相同）

开关（均为 demottle_v13 的关键字参数）：
  chroma_mult  色度扩散迭代相对亮度的倍数（v10 = 2.5）；调大以去更大尺度的色斑
  fine_keep    平面遍减去的斑驳场中 6s 以下小尺度部分的保留比例（0 = 同 v10）；缓解 3–6s 亮度被抹过头
  skin_chroma_free  皮肤色度修正不受质感门（0 = 同 v10；>0 时色度权重 = skin_strength × max(质感门, 该值)）
                    质感门防的是"顺光细腻脸的亮度形体被削平"，与色度无关；生图皮肤细节多于实拍，质感门几乎全关
  chroma_pass  色度单独一遍的强度（0 = 关）：堤坝只看色度边缘，亮度纹理不再挡住色度扩散；取向门检验色度残差
  环境变量 DEMOTTLE_V13="key=val;key=val" 可覆盖以上开关（实验用，供不传参数的回归工具）
  grain        细纹理补偿强度（0 = 关；1 = 补到实拍水平；0.5 = 补足一半缺口能量）：非皮肤区，按
               "亮度 − gauss(亮度, 1.5s)"的局部 rms 与实拍照片平坦区水平的缺口补颗粒（补进能量 = grain × 缺口能量），实拍水平（按亮度分档：L* 21/40/60/80 → 0.445/0.363/0.354/0.275，tools/fine_level.py
               在实拍 / 生图配对上实测）；补的是确定性高斯颗粒（固定种子），只补缺口。皮肤不加（生图皮肤细节已多于实拍）

以下为 v12 的说明。
demottle_v12.py — demottle v10 算法的 GPU 后端（v11 的提速版，等价改写）

算法与 v10 完全相同，只换实现。相对 v11：
  逐位一致  高斯（复刻 OpenCV x86 浮点路径的 FMA 顺序与标量尾部）、Sobel 3×3、INTER_AREA 整数倍、扩散迭代、
           RGB→Lab（复刻 OpenCV 定点查表）、区块统计（σ、斑驳周期）、指纹百分位门限、有界精确距离变换
           ——全局尺子 σ 与 v10 一位不差，堤坝 / 软物体判定不再因浮点误差翻转（v11 在 P09-1 上曾因此差 2 级）
  仍有末位差 INTER_LINEAR（x86 的 OpenCV 走 IPP，闭源，无法逐位复刻；与无 IPP 的 OpenCV 一致）、
           非整数倍 INTER_AREA、Lab→RGB（OpenCV 的 gamma 用样条近似；只影响最终取整，≤ 0.03% 像素差 1 级）、
           均值 / 标准差等归约的求和顺序
  全 GPU   以上全部，加降采样、块效应度、指纹折叠与减法；距离变换超过用途上限的统一截到上限（不改变下游结果）
  更快     高斯用 Triton 直接卷积（反射下标在核内计算，不做填充拷贝）；扩散每步 2 个核（第二个融合除法与掩码），
           CUDA Graph 成批提交；最大 / 最小滤波拆成行、列两遍
  并行     人脸检测与肤色连通域在后台线程与 GPU 同时进行；颗粒 / 抖动随机数按 (种子, 尺寸) 缓存
  仍在 CPU 连通域（OpenCV；种子标记与查表在 GPU）、人脸检测（YuNet）、按表面定尺子（1/4 分辨率）
验收标准：与 v10 最大差 ≤ 1 级（8 位）。

依赖   v10 的全部依赖 + torch（CUDA 版，自带 Triton）；demottle_v10.py 与人脸模型须在同一目录
设备   有 CUDA 用 GPU；无 CUDA 自动退回 v10（CPU）
       DEMOTTLE_DEVICE=cpu   用 PyTorch 在 CPU 上跑（仅数值核对；不走 Triton / CUDA Graph）
       DEMOTTLE_TRITON=0     CUDA 上不用 Triton（退回 v11 的卷积写法，排查用）
       DEMOTTLE_GRAPHS=0     不用 CUDA Graph（排查用）
用法   python demottle_v13.py in.png out.png [--16bit]
"""
import os, threading, concurrent.futures as cf
import numpy as np, cv2
import demottle_v10 as V

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')     # cuBLAS 确定性（非整数倍降采样用到矩阵乘）
try:
    import torch
    import torch.nn.functional as F
except Exception:                                   # 未装 torch：整体退回 v10
    torch = None
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

DEV = None
_OPT_DEFAULT = dict(chroma_mult=2.5, fine_keep=0.0, grain=0.0, grain_tab=(0.445, 0.363, 0.354, 0.275), skin_chroma_free=0.0, chroma_pass=0.0)


def _opt():
    """本次调用的效果开关：按线程保存（服务里不同请求可用不同参数，互不干扰）"""
    return getattr(_TLS, 'opt', _OPT_DEFAULT)


_TLS = threading.local()                            # 每线程：YuNet 实例、扩散图
_LOCK = threading.Lock()                            # 共享缓存与 CUDA Graph 捕获
_POOL = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix='demottle')


def pick_device():
    """CUDA 优先；DEMOTTLE_DEVICE=cpu 时用 CPU 上的 PyTorch；否则返回 None（走 v10）"""
    if torch is None:
        return None
    want = os.environ.get('DEMOTTLE_DEVICE', '').lower()
    if want == 'cpu':
        return torch.device('cpu')
    if want == 'v10':
        return None
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False   # TF32 只有 10 位尾数，会让结果偏离 v10
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True       # 同一输入同一结果
        torch.backends.cudnn.benchmark = False
        return torch.device('cuda')
    return None


def _tri():
    return DEV is not None and DEV.type == 'cuda' and triton is not None and os.environ.get('DEMOTTLE_TRITON', '1') != '0'


# ================================================================= 张量工具
def up(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(DEV)


def dn(t):
    return t.detach().cpu().numpy()


_K, _IDX, _RS = {}, {}, {}


def _gk(s):
    """OpenCV GaussianBlur 的核：ksize = cvRound(8σ+1)|1（浮点图），系数取 getGaussianKernel"""
    k = int(round(float(s) * 8 + 1)) | 1
    key = (k, float(s), str(DEV))
    if key not in _K:
        _K[key] = torch.from_numpy(cv2.getGaussianKernel(k, float(s), cv2.CV_32F).ravel().copy()).to(DEV)
    return _K[key]


def _sym(n, before, after):
    """BORDER_REFLECT（fedcba|abcdef）对称延拓的下标，可超出边长"""
    key = (n, before, after, str(DEV))
    if key not in _IDX:
        i = np.arange(-before, n + after) % (2 * n)
        _IDX[key] = torch.from_numpy(np.where(i >= n, 2 * n - 1 - i, i).astype(np.int64)).to(DEV)
    return _IDX[key]


def _to4(x):
    return x[None, None] if x.dim() == 2 else x.permute(2, 0, 1).unsqueeze(1)


def _from4(y, like):
    return y[0, 0] if like.dim() == 2 else y.squeeze(1).permute(1, 2, 0).contiguous()


# ================================================================= Triton 核
if triton is not None:
    @triton.jit
    def _refl(j, n):
        n2 = 2 * n
        j = ((j % n2) + n2) % n2
        return tl.where(j >= n, n2 - 1 - j, j)

    @triton.jit
    def _conv_k(x, k, o, den, msk, n_lines, n, xs_c, xs_l, xs_p, os_c, os_l, os_p, ds_l, ds_p, C, TAIL0,
                K: tl.constexpr, SYM: tl.constexpr, EPI: tl.constexpr, BL: tl.constexpr, BP: tl.constexpr):
        """一维高斯（沿 pixel 方向），BORDER_REFLECT 下标在核内计算；运算顺序复刻 OpenCV x86 浮点路径，逐位一致：
          行（SYM=0，RowVec_32f）        s = 0；对 t = 0..K−1：s = fma(x[i+t−R], k[t], s)
          列（SYM=1，SymmColumnVec_32f） s = x[i]·k[R]；对 j = 1..R：s = fma(x[i+j] + x[i−j], k[R+j], s)
          每行最后 r = W·C mod 8 个元素是 OpenCV 的标量尾部（x86 实测）：行滤波 4 个一组的部分仍为 FMA、余下单个元素
          乘加分开舍入；列滤波尾部全部乘加分开舍入。TAIL0 为不融合区的起点（编译时关闭自动 FMA 融合，此处显式区分）
        EPI 时输出 s / den · msk（扩散迭代；IEEE 精确除法，与 numpy 一致）"""
        pl = tl.program_id(0); pp = tl.program_id(1); pc = tl.program_id(2)
        li = pl * BL + tl.arange(0, BL)[:, None]
        pi = pp * BP + tl.arange(0, BP)[None, :]
        m = (li < n_lines) & (pi < n)
        if SYM:
            tail = (li * C + pc) >= TAIL0
        else:
            tail = (pi * C + pc) >= TAIL0                # 行尾：4 个一组的部分 OpenCV 编译成 FMA，只有单个元素不融合
        xb = x + pc * xs_c + li * xs_l
        R: tl.constexpr = K // 2
        if SYM:
            has_tail = ((pl * BL + BL - 1) * C + pc) >= TAIL0
        else:
            has_tail = ((pp * BP + BP - 1) * C + pc) >= TAIL0
        if SYM:
            acc = tl.load(xb + pi * xs_p, mask=m, other=0.0) * tl.load(k + R)
            if has_tail:
                for j in range(1, R + 1):
                    a = tl.load(xb + _refl(pi + j, n) * xs_p, mask=m, other=0.0)
                    b = tl.load(xb + _refl(pi - j, n) * xs_p, mask=m, other=0.0)
                    kj = tl.load(k + R + j); ab = a + b
                    acc = tl.where(tail, acc + ab * kj, tl.fma(ab, kj, acc))
            else:
                for j in range(1, R + 1):
                    a = tl.load(xb + _refl(pi + j, n) * xs_p, mask=m, other=0.0)
                    b = tl.load(xb + _refl(pi - j, n) * xs_p, mask=m, other=0.0)
                    acc = tl.fma(a + b, tl.load(k + R + j), acc)
        else:
            acc = tl.load(xb + _refl(pi - R, n) * xs_p, mask=m, other=0.0) * tl.load(k)
            if has_tail:
                for t in range(1, K):
                    v = tl.load(xb + _refl(pi + (t - R), n) * xs_p, mask=m, other=0.0); kt = tl.load(k + t)
                    acc = tl.where(tail, acc + v * kt, tl.fma(v, kt, acc))
            else:
                for t in range(1, K):
                    acc = tl.fma(tl.load(xb + _refl(pi + (t - R), n) * xs_p, mask=m, other=0.0), tl.load(k + t), acc)
        if EPI:
            d = tl.load(den + li * ds_l + pi * ds_p, mask=m, other=1.0)
            w = tl.load(msk + li * ds_l + pi * ds_p, mask=m, other=0.0)
            acc = tl.div_rn(acc, d) * w
        tl.store(o + pc * os_c + li * os_l + pi * os_p, acc, mask=m)

    @triton.jit
    def _edt_col_k(g2, o, H, W, CAP2, BH: tl.constexpr, BW: tl.constexpr):
        """精确 EDT 第二遍：best(y) = min_k g²(y+k) + k²；块内所有像素都已不可能再变小时提前结束"""
        py = tl.program_id(0); px = tl.program_id(1)
        y = py * BH + tl.arange(0, BH)[:, None]
        x = px * BW + tl.arange(0, BW)[None, :]
        m = (y < H) & (x < W)
        best = tl.minimum(tl.load(g2 + y * W + x, mask=m, other=0), CAP2)
        lim = tl.max(tl.max(best, axis=1), axis=0)
        kk = 1
        while kk * kk < lim:
            a = tl.load(g2 + (y - kk) * W + x, mask=m & (y - kk >= 0), other=CAP2)
            b = tl.load(g2 + (y + kk) * W + x, mask=m & (y + kk < H), other=CAP2)
            best = tl.minimum(best, tl.minimum(a, b) + kk * kk)
            lim = tl.max(tl.max(best, axis=1), axis=0)
            kk += 1
        tl.store(o + y * W + x, best, mask=m)


def _conv_pass(x, k, axis, out=None, den=None, msk=None):
    """Triton 一维高斯：x 为 (H,W) 或 (H,W,C)，任意步长；axis=1 水平、0 垂直；返回连续张量"""
    if x.dim() == 2:
        H, W = x.shape; C = 1; sc, sh, sw = 0, x.stride(0), x.stride(1)
    else:
        H, W, C = x.shape; sc, sh, sw = x.stride(2), x.stride(0), x.stride(1)
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    if out.dim() == 2:
        oc, oh, ow = 0, out.stride(0), out.stride(1)
    else:
        oc, oh, ow = out.stride(2), out.stride(0), out.stride(1)
    epi = den is not None
    dh, dw = (den.stride(0), den.stride(1)) if epi else (0, 0)
    if axis == 1:
        nl, n, xl, xp, ol, op_, dl, dp, BL, BP = H, W, sh, sw, oh, ow, dh, dw, 4, 128
    else:
        nl, n, xl, xp, ol, op_, dl, dp, BL, BP = W, H, sw, sh, ow, oh, dw, dh, 128, 4
    grid = (triton.cdiv(nl, BL), triton.cdiv(n, BP), C)
    r8 = (W * C) % 8                                     # OpenCV 向量化覆盖 8 的整数倍，余下走标量
    tail0 = W * C - r8 + (4 * (r8 // 4) if axis == 1 else 0)
    _conv_k[grid](x, k, out, den if epi else out, msk if epi else out, nl, n, sc, xl, xp, oc, ol, op_, dl, dp, C, tail0,
                  K=k.numel(), SYM=(axis == 0), EPI=epi, BL=BL, BP=BP, enable_fp_fusion=False)
    return out


def gauss(x, s):
    """等价于 cv2.GaussianBlur(x, (0,0), s, borderType=BORDER_REFLECT)；先行后列"""
    k = _gk(s)
    if _tri():                                           # 与 OpenCV 逐位一致（超长核的矩阵乘更快但做不到逐位一致，不用）
        return _conv_pass(_conv_pass(x, k, 1), k, 0)
    r = k.numel() // 2
    y = _to4(x)
    y = F.conv2d(y.index_select(3, _sym(y.shape[3], r, r)), k.view(1, 1, 1, -1))
    y = F.conv2d(y.index_select(2, _sym(y.shape[2], r, r)), k.view(1, 1, -1, 1))
    return _from4(y, x)


def _sh(x, d, axis):
    """BORDER_REFLECT_101 下的平移：返回 x[i+d]（沿 axis）"""
    n = x.shape[axis]; i = torch.arange(n, device=x.device) + d
    i = torch.where(i < 0, -i, torch.where(i >= n, 2 * n - 2 - i, i))
    return x.index_select(axis, i)


def sobel(x):
    """等价于 cv2.Sobel(ksize=3)（BORDER_REFLECT_101），逐位一致：OpenCV 的小核滤波
      [−1,0,1]  s = x[+1] − x[−1]
      [1,2,1]   向量部分 fma(x, 2, x[−1] + x[+1])（2x 精确，等于 (x[−1]+x[+1]) + 2x）；每行最后 W mod 8 个元素为标量 (x[−1] + 2x) + x[+1]
    返回 (gx, gy)"""
    H, W = x.shape
    tail = torch.arange(W, device=x.device) >= W - W % 8
    def smooth(l, c, r):
        return torch.where(tail, (l + 2 * c) + r, (l + r) + 2 * c)
    dxr = _sh(x, 1, 1) - _sh(x, -1, 1)                  # 行：[−1,0,1]
    gx = smooth(_sh(dxr, -1, 0), dxr, _sh(dxr, 1, 0))  # 列：[1,2,1]
    smr = smooth(_sh(x, -1, 1), x, _sh(x, 1, 1))       # 行：[1,2,1]
    gy = _sh(smr, 1, 0) - _sh(smr, -1, 0)              # 列：[−1,0,1]
    return gx, gy


def grad_mag(x, s):
    gx, gy = sobel(gauss(x, s))
    return torch.sqrt(gx * gx + gy * gy)


def normconv(x, M, s):
    return gauss(x * M, s) / torch.clamp(gauss(M, s), min=1e-6)


def dilate(x, n):
    """方形 n×n 最大滤波（等价于 cv2.dilate 全 1 核）；拆成行、列两遍（max 与顺序无关，逐位相同）"""
    y = F.max_pool2d(x[None, None], (1, n), stride=1, padding=(0, n // 2))
    return F.max_pool2d(y, (n, 1), stride=1, padding=(n // 2, 0))[0, 0]


def erode(x, n):
    return -dilate(-x, n)


def dilate_mask(m, n):
    """布尔掩码膨胀（方形 n×n，n 为奇数；等价于 cv2.dilate 全 1 核）"""
    dt = torch.float16 if DEV.type == 'cuda' else torch.float32
    return dilate(m.to(dt), n) > 0


def _area_w(n_src, n_dst):
    """OpenCV INTER_AREA 的一维系数矩阵 (n_dst, n_src)：对单位阵做 resize 探测得到"""
    key = ('area', n_src, n_dst, str(DEV))
    with _LOCK:
        if key not in _RS:
            eye = np.eye(n_src, dtype=np.float32)
            _RS[key] = torch.from_numpy(np.ascontiguousarray(cv2.resize(eye, (n_dst, n_src), interpolation=cv2.INTER_AREA).T)).to(DEV)
        return _RS[key]


def _nearest_idx(n_src, n_dst):
    """OpenCV INTER_NEAREST 的取样下标：对下标序列做 resize 探测得到"""
    key = ('nn', n_src, n_dst, str(DEV))
    with _LOCK:
        if key not in _RS:
            src = np.arange(n_src, dtype=np.float32)[None, :]
            idx = cv2.resize(src, (n_dst, 1), interpolation=cv2.INTER_NEAREST)[0].astype(np.int64)
            _RS[key] = torch.from_numpy(idx).to(DEV)
        return _RS[key]


def resize_area(t, w, h):
    """缩小，与 cv2.INTER_AREA 一致。整数倍：OpenCV resizeAreaFast_ 的累加顺序（每行先求和——4 个一组 ((a+b)+c)+d——
    再逐行累加，最后乘 1/面积），逐位一致；非整数倍：用探测到的面积系数（矩阵乘，差 ~1e-7）"""
    H, W = t.shape[:2]
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        t = t[:h * fy, :w * fx]
        acc = None
        for dy in range(fy):
            row = t[dy::fy]
            k = 0; rs = None
            while k < fx:
                if k + 4 <= fx:
                    g = ((row[:, k::fx] + row[:, k + 1::fx]) + row[:, k + 2::fx]) + row[:, k + 3::fx]; k += 4
                else:
                    g = row[:, k::fx]; k += 1
                rs = g if rs is None else rs + g
            acc = rs if acc is None else acc + rs
        return acc * (1.0 / (fx * fy))
    Ay, Ax = _area_w(H, h), _area_w(W, w)
    if t.dim() == 2:
        return Ay @ (t @ Ax.t())
    return (Ay @ (t.permute(2, 0, 1) @ Ax.t())).permute(1, 2, 0).contiguous()


def resize_nearest_mask(m, w, h):
    H, W = m.shape
    return m.index_select(0, _nearest_idx(H, h)).index_select(1, _nearest_idx(W, w))


def _lin_tab(n_src, n_dst):
    """OpenCV INTER_LINEAR 的取样下标与系数（resize.cpp：fx = (float)((dx+0.5)·scale − 0.5)，越界夹取且系数归零）"""
    key = ('lin', n_src, n_dst, str(DEV))
    with _LOCK:
        if key not in _RS:
            scale = 1.0 / (float(n_dst) / n_src)
            i0 = np.empty(n_dst, np.int64); a1 = np.empty(n_dst, np.float32)
            for d in range(n_dst):
                f = np.float32((d + 0.5) * scale - 0.5); sx = int(np.floor(f)); f = np.float32(f - np.float32(sx))
                if sx < 0:
                    f, sx = np.float32(0), 0
                if sx >= n_src - 1:
                    f, sx = np.float32(0), n_src - 1
                i0[d] = sx; a1[d] = f
            a0 = (np.float32(1) - a1).astype(np.float32)
            i1 = np.minimum(i0 + 1, n_src - 1)
            _RS[key] = tuple(torch.from_numpy(v).to(DEV) for v in (i0, i1, a0, a1))
        return _RS[key]


def resize_linear(t, w, h):
    """放大，与 cv2.INTER_LINEAR（浮点）一致：先水平 S0·a0 + S1·a1，再垂直 S0·b0 + S1·b1，乘加分开舍入"""
    H, W = t.shape[:2]
    xi0, xi1, xa0, xa1 = _lin_tab(W, w); yi0, yi1, yb0, yb1 = _lin_tab(H, h)
    ex = (lambda v: v[:, None]) if t.dim() == 3 else (lambda v: v)
    hz = t.index_select(1, xi0) * ex(xa0) + t.index_select(1, xi1) * ex(xa1)
    ey = (lambda v: v[:, None, None]) if t.dim() == 3 else (lambda v: v[:, None])
    return hz.index_select(0, yi0) * ey(yb0) + hz.index_select(0, yi1) * ey(yb1)


def edt(fg, cap):
    """到最近 False 像素的精确欧氏距离；≥ cap 的统一记为 cap（调用处的用途在 cap 以内已饱和）。
    第一遍按行（扫描最近零点），第二遍按列取 min_k g²(y+k)+k²，整数运算，与 scipy 精确 EDT 一致。"""
    cap = int(cap)
    if not _tri():
        d = up(cv2.distanceTransform(dn(fg).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        return torch.clamp(d, max=float(cap))
    H, W = fg.shape
    idx = torch.arange(W, device=DEV, dtype=torch.int32).expand(H, W)
    big = 1 << 20
    left = torch.where(fg, torch.full_like(idx, -big), idx).cummax(dim=1).values
    right = torch.where(fg, torch.full_like(idx, big), idx).flip(1).cummin(dim=1).values.flip(1)
    g = torch.clamp(torch.minimum(idx - left, right - idx), max=cap)
    g2 = (g * g).contiguous(); best = torch.empty_like(g2)
    _edt_col_k[(triton.cdiv(H, 4), triton.cdiv(W, 128))](g2, best, H, W, cap * cap, BH=4, BW=128, enable_fp_fusion=False)
    return torch.sqrt(best.double()).float()


def median(v):
    """与 np.median 一致：偶数个取中间两数的均值"""
    v = v.reshape(-1); n = v.numel()
    lo = torch.kthvalue(v, (n + 1) // 2).values
    return lo if n % 2 else (lo + torch.kthvalue(v, n // 2 + 1).values) / 2


def mad(v):
    v = v.reshape(-1).float(); md = median(v)
    return float(1.4826 * median((v - md).abs()))


def _rowmedian(t):
    """逐行 np.median（float32）"""
    n = t.shape[1]
    lo = torch.kthvalue(t, (n + 1) // 2, dim=1).values
    return lo if n % 2 else (lo + torch.kthvalue(t, n // 2 + 1, dim=1).values) / 2


def _tiles(x, B):
    H, W = x.shape; ny, nx = (H - B) // B + 1, (W - B) // B + 1
    return x[:ny * B, :nx * B].reshape(ny, B, nx, B).permute(0, 2, 1, 3).reshape(ny * nx, B * B), ny, nx


def _tile_mads(t):
    md = _rowmedian(t)
    return _rowmedian((t - md[:, None]).abs()) * 1.4826


def local_median(x, win, f=4):
    """与 v10 相同：1/f 分辨率上做 scipy.median_filter(size, mode='reflect')，再双线性放大。
    逐窗精确求秩 size²//2 的元素（scipy 对偶数窗口的取法），分块以控制显存。"""
    h, w = x.shape
    small = resize_area(x, w // f, h // f)
    n = max(3, win // f); hb, ha = n // 2, n - 1 - n // 2
    P = small.index_select(0, _sym(small.shape[0], hb, ha)).index_select(1, _sym(small.shape[1], hb, ha))
    out = torch.empty_like(small); rows = max(1, int(6.4e7 // (small.shape[1] * n * n)))
    for r0 in range(0, small.shape[0], rows):
        r1 = min(small.shape[0], r0 + rows)
        win_ = P[r0:r1 + n - 1].unfold(0, n, 1).unfold(1, n, 1).reshape(r1 - r0, small.shape[1], n * n)
        out[r0:r1] = torch.kthvalue(win_, n * n // 2 + 1, dim=-1).values
    return resize_linear(out, w, h)


# ================================================================= 扩散（CUDA Graph 成批提交）
class _Diffuser:
    """固定形状的扩散器：静态缓冲 + 捕获好的 CUDA Graph（每线程一份，避免并发冲突）"""
    CH = 32

    def __init__(self, shape, s):
        h, w, C = shape
        self.k = _gk(s)
        self.A = torch.zeros(shape, device=DEV); self.B = torch.zeros(shape, device=DEV); self.T = torch.zeros(shape, device=DEV)
        self.den = torch.ones((h, w), device=DEV); self.M = torch.zeros((h, w), device=DEV)
        self.graphs = {}
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):                    # 预热：Triton 编译与启动缓存
            for i in range(2):
                self._step(i % 2)
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        for n, cur in ((self.CH, 0), (1, 0), (1, 1)):   # 一次捕获全部用得到的图：服务中途不再捕获（捕获期间与其他线程的同步有冲突风险）
            self._graph(n, cur)

    def _step(self, cur):
        src, dst = (self.A, self.B) if cur == 0 else (self.B, self.A)
        _conv_pass(src, self.k, 1, out=self.T)
        _conv_pass(self.T, self.k, 0, out=dst, den=self.den, msk=self.M)

    def _graph(self, n, cur):
        key = (n, cur)
        if key not in self.graphs:
            g = torch.cuda.CUDAGraph()
            with _LOCK:
                with torch.cuda.graph(g, capture_error_mode='thread_local'):
                    for i in range(n):
                        self._step((cur + i) % 2)
            self.graphs[key] = g
        return self.graphs[key]

    def run(self, SM, den, M, n):
        self.den.copy_(den); self.M.copy_(M); self.A.copy_(SM)
        cur = 0
        for _ in range(n // self.CH):
            self._graph(self.CH, cur).replay()         # CH 为偶数，结果仍在 A
        for _ in range(n % self.CH):
            self._graph(1, cur).replay(); cur ^= 1
        return (self.A if cur == 0 else self.B).clone()


def _diffuser(shape, s):
    d = getattr(_TLS, 'diff', None)
    if d is None:
        d = _TLS.diff = {}
    key = (tuple(shape), float(s))
    if key not in d:
        d[key] = _Diffuser(shape, s)
    return d[key]


def masked_diffuse(base, M, n_iter, s=1.0, state=None, return_state=False):
    """堤坝内扩散：SM ← gauss(SM)/gauss(M)·M，迭代 n_iter 次（同 v10）"""
    M3 = M[..., None]
    den = torch.clamp(gauss(M, s), min=1e-6)
    SM = (base * M3) if state is None else state
    if _tri() and n_iter > 0:
        if os.environ.get('DEMOTTLE_GRAPHS', '1') != '0':
            SM = _diffuser(SM.shape, s).run(SM, den, M, n_iter)
        else:
            k = _gk(s); SM = SM.contiguous(); M = M.contiguous(); den = den.contiguous()
            for _ in range(n_iter):
                SM = _conv_pass(_conv_pass(SM, k, 1), k, 0, den=den, msk=M)
    else:
        den3 = den[..., None]
        for _ in range(n_iter):
            SM = gauss(SM, s) / den3 * M3
    S = torch.where(M3 > 0, SM, base)
    return (S, SM) if return_state else S


def coherence(band, s):
    gx, gy = sobel(band)
    jxx, jyy, jxy = gauss(gx * gx, s), gauss(gy * gy, s), gauss(gx * gy, s)
    tr = jxx + jyy; det = jxx * jyy - jxy * jxy
    return torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0)) / (tr + 1e-9)


def clip01(x):
    return torch.clamp(x, 0, 1)


# ================================================================= Lab ↔ RGB
_LAB = {}


def _lab_tables():
    """OpenCV 浮点 sRGB→Lab 走 33³ 定点查表 + 三线性插值。格点处插值权重为 0，
    对格点颜色做 cvtColor 即可读出查表值；构造后抽样自检，不一致则退回 OpenCV（CPU）。"""
    key = str(DEV)
    with _LOCK:
        if key in _LAB:
            return _LAB[key]
        D = 33; g = np.arange(D, dtype=np.float32) / np.float32(32)
        R, G, B = np.meshgrid(g, g, g, indexing='ij')
        lab = cv2.cvtColor(np.stack([R, G, B], -1).reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        lut = np.stack([np.rint(lab[:, 0] / 100 * 16384), np.rint((lab[:, 1] + 128) / 256 * 16384),
                        np.rint((lab[:, 2] + 128) / 256 * 16384)], -1).astype(np.int32)
        iv = np.rint((np.arange(256, dtype=np.float32) / np.float32(255)) * np.float32(16384)).astype(np.int64)
        tab = (torch.from_numpy(iv >> 9).to(DEV), torch.from_numpy((iv >> 5) & 15).to(DEV), torch.from_numpy(lut).to(DEV))
        rng = np.random.default_rng(12345)
        sample = rng.integers(0, 256, (64, 1024, 3)).astype(np.uint8)
        ref = cv2.cvtColor(sample.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        _LAB[key] = tab
        got = _lab_raw(torch.from_numpy(sample).to(DEV), tab)
        if not np.array_equal(dn(got), ref):
            _LAB[key] = None
        return _LAB[key]


def _lab_raw(img_t, tab):
    """uint8 RGB → OpenCV 口径 Lab（L 0–100，a/b −128–127），逐位一致"""
    t, f, lut = tab
    idx = img_t.long()
    T = [t[idx[..., c]] for c in range(3)]; Fr = [f[idx[..., c]] for c in range(3)]
    acc = None
    for dr in (0, 1):
        wr = Fr[0] if dr else 16 - Fr[0]; ir = torch.clamp(T[0] + dr, max=32)
        for dg in (0, 1):
            wg = Fr[1] if dg else 16 - Fr[1]; ig = torch.clamp(T[1] + dg, max=32)
            for db in (0, 1):
                wb = Fr[2] if db else 16 - Fr[2]; ib = torch.clamp(T[2] + db, max=32)
                v = lut[(ir * 33 + ig) * 33 + ib] * (wr * wg * wb).to(torch.int32)[..., None]
                acc = v if acc is None else acc + v
    acc = (acc + 2048) >> 12
    a = acc.float() / 16384.0
    return torch.stack([a[..., 0] * 100.0, a[..., 1] * 256.0 - 128.0, a[..., 2] * 256.0 - 128.0], -1)


def rgb2lab(img_t):
    """uint8 RGB（GPU）→ 归一化 Lab（L/100，a、b/128），与 v10 analyze 完全相同"""
    tab = _lab_tables()
    if tab is None:
        lab = up(cv2.cvtColor(dn(img_t).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB))
    else:
        lab = _lab_raw(img_t, tab)
    return torch.stack([lab[..., 0] / 100.0, lab[..., 1] / 128.0, lab[..., 2] / 128.0], -1)


def lab2rgb01(out):
    """归一化 Lab → RGB [0,1]（OpenCV 浮点 Lab2RGB 的公式；OpenCV 的 gamma 用样条近似，差 ≤ 0.02 级）"""
    L = out[..., 0] * 100.0; a = out[..., 1] * 128.0; b = out[..., 2] * 128.0
    lth = 0.008856 * 903.3; fth = 7.787 * 0.008856 + 16.0 / 116.0
    fy_hi = (L + 16.0) / 116.0
    y = torch.where(L <= lth, L / 903.3, fy_hi * fy_hi * fy_hi)
    fy = torch.where(L <= lth, 7.787 * y + 16.0 / 116.0, fy_hi)
    fx = a / 500.0 + fy; fz = fy - b / 200.0
    X = torch.where(fx <= fth, (fx - 16.0 / 116.0) / 7.787, fx * fx * fx) * 0.950456
    Z = torch.where(fz <= fth, (fz - 16.0 / 116.0) / 7.787, fz * fz * fz) * 1.088754
    r = 3.240479 * X - 1.53715 * y - 0.498535 * Z
    g = -0.969256 * X + 1.875991 * y + 0.041556 * Z
    bb = 0.055648 * X - 0.204043 * y + 1.057311 * Z
    rgb = clip01(torch.stack([r, g, bb], -1))
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * torch.pow(rgb, 1.0 / 2.4) - 0.055)


# ================================================================= CPU 侧：连通域
def hysteresis(strong, weak, min_area=0, min_ext=0):
    """滞后连通（同 v10）：GPU 布尔张量进出。CPU 只做连通域（OpenCV），种子标记与按标签查表在 GPU。
    min_ext > 0 时同时按外接框长边过滤（与"先滞后、再对结果做连通域"等价：结果的连通块就是被保留的弱连通块本身）。"""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(dn(weak.to(torch.uint8)), connectivity=8)
    keep = np.ones(n, bool); keep[0] = False
    if min_area > 0:
        keep &= st[:, cv2.CC_STAT_AREA] >= min_area
    if min_ext > 0:
        keep &= np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) >= min_ext
    lbl_t = torch.from_numpy(lbl).to(DEV)
    seeded = torch.zeros(n, dtype=torch.bool, device=DEV)
    seeded[lbl_t[strong]] = True
    return (seeded & torch.from_numpy(keep).to(DEV))[lbl_t]


# ================================================================= 人脸与皮肤（后台线程）
def _detect_faces(rgb_u8, score=0.7):
    """同 V.detect_faces；YuNet 实例每线程一份（OpenCV DNN 对象不保证线程安全）"""
    H, W = rgb_u8.shape[:2]
    path = V._yunet_path()
    if path is not None:
        try:
            det = getattr(_TLS, 'yunet', None)
            if det is None:
                det = _TLS.yunet = cv2.FaceDetectorYN.create(path, '', (320, 320), score, 0.3, 50)
            sc = 640.0 / max(H, W)
            small = cv2.resize(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
            det.setInputSize((small.shape[1], small.shape[0]))
            _, f = det.detect(small)
            return [] if f is None else [tuple(float(v) / sc for v in r[:4]) for r in f]
        except Exception:
            pass
    return V.detect_faces(rgb_u8, score)


def _skin_cpu(rgb_u8, s):
    """V.face_anchored_skin 的 CPU 部分（未模糊的 0/1 掩码）；逐块重叠计数改为一次 bincount，结果相同"""
    H, W = rgb_u8.shape[:2]
    ycc = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((Cr > 133) & (Cr < 178) & (Cb > 77) & (Cb < 132) & (Y > 35)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((int(7 * s) | 1,) * 2, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(15 * s) | 1,) * 2, np.uint8))
    faces = _detect_faces(rgb_u8)
    if len(faces) == 0:
        return None, 0
    anchor = np.zeros((H, W), np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(anchor, (int(x - 0.3 * w), int(y - 0.3 * h)), (int(x + 1.3 * w), int(y + 1.8 * h)), 1, -1)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    a = anchor.astype(bool)
    A_anchor = float(a.sum()) + 1e-9
    ov = np.bincount(lbl[a], minlength=n).astype(np.float64)
    keep = (st[:, cv2.CC_STAT_AREA] >= 0.001 * H * W) & (ov / A_anchor >= 0.15); keep[0] = False
    return (keep[lbl] & a).astype(np.uint8), len(faces)


# ================================================================= 随机数（与 v10 同序列，按尺寸缓存）
_NOISE = {}


def _noise(seed, H, W, jpeg, bits):
    """v10 的随机数消费顺序：JPEG 先 3 张正态颗粒，8 位输出再 1 张均匀抖动"""
    key = (seed, H, W, jpeg, bits, str(DEV))
    with _LOCK:
        if key in _NOISE:
            return _NOISE[key]
    rng = np.random.default_rng(seed)
    nz = [up(rng.normal(size=(H, W)).astype(np.float32)) for _ in range(3)] if jpeg else None
    di = up(rng.random((H, W)).astype(np.float32)) if bits == 8 else None
    if DEV.type == 'cuda':
        torch.cuda.current_stream().synchronize()       # 后台线程上传：确保主线程使用前已落到显存
    with _LOCK:
        if len(_NOISE) >= 4:
            _NOISE.pop(next(iter(_NOISE)))
        _NOISE[key] = (nz, di)
    return _NOISE[key]


_GRAIN = {}


def _grain_field(seed, H, W, s):
    """补纹理用的颗粒场（确定性，按 (种子, 尺寸) 缓存）：0.6s 相关高斯颗粒，高通 2s（不给斑驳频带添能量），
    标定到"亮度 − 1.5s 高斯"口径下 rms = 1"""
    key = (seed, H, W, str(DEV))
    with _LOCK:
        if key in _GRAIN:
            return _GRAIN[key]
    g_rng = np.random.default_rng(seed + 7919)
    nz = gauss(up(g_rng.normal(size=(H, W)).astype(np.float32)), 0.6 * s)
    nz = nz - gauss(nz, 2.0 * s)
    nz = nz / torch.sqrt(torch.mean((nz - gauss(nz, 1.5 * s)) ** 2))
    with _LOCK:
        if len(_GRAIN) >= 4:
            _GRAIN.pop(next(iter(_GRAIN)))
        _GRAIN[key] = nz
    return nz


# ================================================================= 1–2 指纹与分析
def blockiness(L):
    dx = (L[:, 1:] - L[:, :-1]).abs(); dy = (L[1:, :] - L[:-1, :]).abs()
    on = torch.cat([dx[:, 7::8].reshape(-1), dy[7::8, :].reshape(-1)])
    off = torch.cat([dx[:, 3::8].reshape(-1), dy[3::8, :].reshape(-1)])
    return float(np.float32(on.mean().item()) / (np.float32(off.mean().item()) + np.float32(1e-9)))


def _percentile_thr(x, pct):
    """np.percentile(x, pct)（线性插值）的精确复现：两个次序统计量在 GPU 上取，插值按 numpy 规则在 CPU 算"""
    v = x.reshape(-1); n = v.numel()
    vi = (n - 1) * (pct / 100); lo = int(np.floor(vi)); hi = min(lo + 1, n - 1); t = float(vi - lo)
    a = np.float32(torch.kthvalue(v, lo + 1).values.item()); b = np.float32(torch.kthvalue(v, hi + 1).values.item())
    d = b - a
    return float(b - d * (1 - t)) if t >= 0.5 else float(a + d * t)


def periodic_fingerprint(img_t, P=16, flat_pct=30):
    """同 v10：返回 (uint8 图（GPU）, 相关, 峰峰值)"""
    x = img_t.float(); H, W = x.shape[:2]
    g = x.mean(-1)
    hp = g - gauss(g, 4); fe = gauss(hp * hp, 8)
    flat = (fe <= _percentile_thr(fe, flat_pct)).float()

    def fold(hpc, y0, y1, x0, x1):
        h = hpc[y0:y1, x0:x1]; m = flat[y0:y1, x0:x1]; hh = (y1 - y0) // P * P; ww = (x1 - x0) // P * P
        num = (h * m)[:hh, :ww].reshape(hh // P, P, ww // P, P).sum((0, 2))
        den = m[:hh, :ww].reshape(hh // P, P, ww // P, P).sum((0, 2))
        return num / torch.clamp(den, min=1)
    q = torch.stack([fold(hp, 0, H // 2, 0, W // 2), fold(hp, 0, H // 2, W // 2, W), fold(hp, H // 2, H, 0, W // 2),
                     fold(hp, H // 2, H, W // 2, W), fold(hp, 0, H, 0, W)])
    q = dn(q)
    corr = float(np.mean([np.corrcoef(q[i].ravel(), q[j].ravel())[0, 1] for i in range(4) for j in range(i + 1, 4)]))
    p2p = float(q[4].max() - q[4].min())
    if corr < 0.6 or p2p < 0.8:
        return img_t, corr, 0.0
    hpx = x - gauss(x, 4); out = x.clone()
    for c in range(3):
        T = fold(hpx[..., c], 0, H, 0, W); T = T - T.mean()
        out[..., c] -= T.repeat(H // P + 1, W // P + 1)[:H, :W]
    return torch.clamp(out + 0.5, 0, 255).to(torch.uint8), corr, p2p


def sigma_by_tiles(band, B, frac=0.10, Lmap=None, lo=0.12, hi=0.90):
    """同 v10：区块 MAD 在 GPU 上逐行求中位数；返回 (σ, 升序区块值 float64)"""
    tiles, ny, nx = _tiles(band, B)
    if Lmap is not None:
        lm = _tiles(Lmap, B)[0].mean(1)
        sel = (lm >= lo) & (lm <= hi)
        if int(sel.sum()) < 12:
            return sigma_by_tiles(band, B, frac)
        tiles = tiles[sel]
    vals = np.sort(dn(_tile_mads(tiles)).astype(np.float64))
    k = max(3, int(len(vals) * frac))
    return float(np.median(vals[:k])), vals


def mottle_period(band, B, frac=0.10, pmin=24, pmax=200):
    """同 v10：最平区块（按 MAD 升序，同值按行优先）上带通亮度的自相关，第一次过零距离 ≈ 周期 / 4"""
    tiles, ny, nx = _tiles(band, B)
    order = torch.sort(_tile_mads(tiles), stable=True).indices[:max(3, int(ny * nx * frac))]
    t = tiles[order].reshape(-1, B, B)
    t = t - t.mean((1, 2), keepdim=True)
    per = torch.stack([(t[:, :, :B - k] * t[:, :, k:]).mean((1, 2)) + (t[:, :B - k, :] * t[:, k:, :]).mean((1, 2))
                       for k in range(B // 2)], 1)              # (区块, 位移) float32
    per = dn(per)
    acc = np.zeros(B // 2, np.float64)
    for row in per:                                      # 与 v10 相同的逐区块 float64 累加顺序
        acc += row
    acc /= max(2 * len(per), 1); acc /= acc[0] + 1e-12
    if not np.any(acc < 0):
        return 80.0
    return float(np.clip(4 * int(np.argmax(acc < 0)), pmin, pmax))


def analyze(img_t, path=None, lab=None):
    """同 v10；lab 已算好（未做指纹减法时与输入相同）则直接用"""
    labt = rgb2lab(img_t) if lab is None else lab
    H, W = labt.shape[:2]; L = labt[..., 0]
    s = min(H, W) / 1536.0
    B = max(32, int(64 * s))
    q = V.jpeg_quant_steps(path)
    blk = blockiness(L)
    jpeg = (q is not None) or blk > 1.15
    sig = []; flat_vals = None
    for c in range(3):
        band = gauss(labt[..., c], 3 * s) - gauss(labt[..., c], 20 * s)
        v, vals = sigma_by_tiles(band, B, Lmap=L)
        sig.append(v)
        if c == 0: flat_vals = vals
    sig = np.array(sig, np.float32)
    if q is not None:
        qL, qC = q[0] / 255.0, q[1] / 255.0 * 1.3
    else:
        qL, qC = (0.006, 0.010) if jpeg else (0.0, 0.0)
    period = mottle_period(gauss(L, 3) - gauss(L, 80), max(64, int(128 * s)))
    k = max(3, len(flat_vals) // 10)
    ok = (sig[0] <= 0.015) and (flat_vals[:k].max() <= 0.02)
    return dict(labt=labt, H=H, W=W, s=s, jpeg=jpeg, blk=blk, q=(qL, qC), sigma=sig, period=period, ok=ok)


# ================================================================= 3 堤坝
def unified_barrier(ctx, lo=6.0, hi=9.0, dilate_px=5):
    s = ctx['s']; sig = ctx['sigma']; qL, qC = ctx['q']; labt = ctx['labt']
    es = 2.5 * s; L = labt[..., 0]; H, W = ctx['H'], ctx['W']
    g = grad_mag(L, es); gc = torch.maximum(grad_mag(labt[..., 1], es), grad_mag(labt[..., 2], es))
    if ctx['jpeg']:
        cx = torch.ones(W, device=DEV); ry = torch.ones(H, device=DEV)
        for k in range(7, W, 8): cx[max(k - 1, 0):k + 2] = 0.5
        for k in range(7, H, 8): ry[max(k - 1, 0):k + 2] = 0.5
        gridm = ry[:, None] * cx[None, :]
        g = g * gridm; gc = gc * gridm
    u = 0.16 * float(sig[0]) * 8.0 + 1e-9
    uc = 0.16 * float(max(sig[1], sig[2])) * 8.0 + 1e-9
    r = torch.maximum(g / u, gc / uc)
    if not ctx['jpeg']:
        gs = grad_mag(L, 1.6 * s)
        r_snr = gs / (local_median(gs, int(61 * s)) + 1e-6)
        gcs = torch.maximum(grad_mag(labt[..., 1], 1.6 * s), grad_mag(labt[..., 2], 1.6 * s))
        r_snr = torch.maximum(r_snr, gcs / (local_median(gcs, int(61 * s)) + 1e-6))
    else:
        r_snr = torch.zeros_like(r)
    n = int(15 * s) | 1
    def rng(x):
        b = gauss(x, es); return dilate(b, n) - erode(b, n)
    cm = 6.0 if ctx['jpeg'] else 3.0
    contrast = (rng(L) > max(cm * float(sig[0]), 3.0 * qL)) | \
               (torch.maximum(rng(labt[..., 1]), rng(labt[..., 2])) > max(cm * float(max(sig[1], sig[2])), 3.0 * qC))
    r = r * contrast; r_snr = r_snr * contrast
    bar = hysteresis((r > hi) | (r_snr > 4.0), (r > lo) | (r_snr > 2.0), int(40 * s * s), max(12, int(24 * s)))
    return dilate_mask(bar, 2 * int(dilate_px * s) + 1)


def chroma_barrier(ctx, lo=6.0, hi=9.0, dilate_px=5):
    """只看色度的堤坝（色度单独一遍用）：亮度纹理不再挡住色度扩散；判据同 unified_barrier 的色度部分"""
    s = ctx['s']; sig = ctx['sigma']; qL, qC = ctx['q']; labt = ctx['labt']
    es = 2.5 * s
    gc = torch.maximum(grad_mag(labt[..., 1], es), grad_mag(labt[..., 2], es))
    uc = 0.16 * float(max(sig[1], sig[2])) * 8.0 + 1e-9
    r = gc / uc
    if not ctx['jpeg']:
        gcs = torch.maximum(grad_mag(labt[..., 1], 1.6 * s), grad_mag(labt[..., 2], 1.6 * s))
        r_snr = gcs / (local_median(gcs, int(61 * s)) + 1e-6)
    else:
        r_snr = torch.zeros_like(r)
    n = int(15 * s) | 1
    def rng(x):
        b = gauss(x, es); return dilate(b, n) - erode(b, n)
    cm = 6.0 if ctx['jpeg'] else 3.0
    contrast = torch.maximum(rng(labt[..., 1]), rng(labt[..., 2])) > max(cm * float(max(sig[1], sig[2])), 3.0 * qC)
    r = r * contrast; r_snr = r_snr * contrast
    bar = hysteresis((r > hi) | (r_snr > 4.0), (r > lo) | (r_snr > 2.0), int(40 * s * s), max(12, int(24 * s)))
    return dilate_mask(bar, 2 * int(dilate_px * s) + 1)


# ================================================================= 4 平面遍（皮肤遍复用）
def region_pass(labt, ctx, block, sig_diff, T_near, T_far, far_px, pin_seed, fade_px,
                grain_sigma, restrict=None, chroma_boost=True, sig_override=None, gate='all'):
    H, W, s = ctx['H'], ctx['W'], ctx['s']
    base = gauss(labt, grain_sigma)
    ds = 0.25; hs, ws = int(H * ds), int(W * ds)
    base_s = resize_area(base, ws, hs)
    bar_s = resize_nearest_mask(dilate_mask(block, 3), ws, hs) | (resize_area(block.float(), ws, hs) > 0.35)
    M_s = (~bar_s).float()
    if float(M_s.sum()) < 200:
        return None
    n_iter = max(9, int((sig_diff * ds) ** 2))
    n_iter_c = int(n_iter * _opt()['chroma_mult']) if chroma_boost else n_iter
    for p in range(2):
        S_s, SM_s = masked_diffuse(base_s, M_s, n_iter, 1.0, return_state=True)
        m_s = (base_s - S_s) * M_s[..., None]
        if sig_override is None:
            flat = M_s > 0
            sig = np.array([mad(m_s[..., c][flat]) + 1e-9 for c in range(3)], np.float32)
            sigmap = up(V.surface_sigma(dn(m_s), dn(M_s), sig))
        else:
            sig = np.asarray(sig_override, np.float32)
            sigmap = up(sig).view(1, 1, 3).expand_as(m_s)
        if gate == 'all':
            d2_s = torch.sum((m_s / sigmap) ** 2, dim=-1)
        elif gate == 'ab':                               # 色度单独一遍：只按色度偏离判定
            d2_s = (m_s[..., 1] / sigmap[..., 1]) ** 2 + (m_s[..., 2] / sigmap[..., 2]) ** 2
        else:
            d2_s = (m_s[..., 0] / sigmap[..., 0]) ** 2
        if chroma_boost:
            S_c = masked_diffuse(base_s[..., 1:].contiguous(), M_s, n_iter_c - n_iter, 1.0, state=SM_s[..., 1:].clone())
            m_s = torch.cat([m_s[..., :1], (base_s[..., 1:] - S_c) * M_s[..., None]], -1)
        if p == 0:
            dist_s = edt(M_s > 0, far_px * s * ds + 2)
            T_s = T_near + (T_far - T_near) * clip01(dist_s / (far_px * s * ds))
            soft = hysteresis(d2_s > pin_seed * T_s ** 2, d2_s > T_s ** 2, int(40 * s * s))
            M_s = M_s * (1 - dilate_mask(soft, 3).float())
    m = resize_linear(m_s, W, H)
    d2 = resize_linear(d2_s, W, H)
    M = resize_linear(M_s, W, H)
    dist_b = edt(M > 0.5, max(far_px, 24) * s + 2)
    T = T_near + (T_far - T_near) * clip01(dist_b / (far_px * s))
    R1 = restrict if restrict is not None else torch.ones((H, W), device=DEV)
    fine_b = labt[..., 0] - gauss(labt[..., 0], 3 * s)
    if gate == 'ab':                                     # 取向门检验色度残差（彩色条纹等有取向的色度结构受保护），不看亮度纹理
        c_res = resize_linear(torch.maximum(coherence(m_s[..., 1].contiguous(), 20 * s * ds), coherence(m_s[..., 2].contiguous(), 20 * s * ds)), W, H)
        iso = gauss(clip01((0.55 - c_res) / 0.20), 6 * s)
    else:
        c_res = resize_linear(coherence(m_s[..., 0].contiguous(), 20 * s * ds), W, H)
        iso = gauss(torch.minimum(clip01((0.55 - c_res) / 0.20), clip01((0.60 - coherence(fine_b, 12 * s)) / 0.20)), 6 * s)
    R1 = R1 * iso
    w0 = clip01(1 - d2 / T ** 2) * M * R1
    w = torch.minimum(w0, gauss(w0, 10 * s))
    fill = torch.stack([normconv(m[..., c] * w, w, 10 * s) for c in range(3)], -1)
    dist = edt(w < 0.5, fade_px * s + 2)
    fade = gauss(clip01(1 - dist / (fade_px * s)), 8 * s) * R1
    m_final = w[..., None] * m + (1 - w[..., None]) * fill * fade[..., None]
    d2_ab = (m[..., 1] / (float(sig[1]) + 1e-9)) ** 2 + (m[..., 2] / (float(sig[2]) + 1e-9)) ** 2
    wc0 = clip01(1 - d2_ab / (1.5 * T) ** 2) * clip01(1 - d2 / (2 * T) ** 2) * M * R1
    w_ab = torch.maximum(w, torch.minimum(wc0, gauss(wc0, 10 * s)))
    m_final = torch.stack([m_final[..., 0]] + [w_ab * m[..., c] + (1 - w_ab) * fill[..., c] * fade for c in (1, 2)], -1)
    return dict(base=base, grain=labt - base, m=m_final, w=w, M=M, dist_b=dist_b, sig=sig)


# ================================================================= 主函数
def demottle_v13(img_u8, path=None, strength=0.8, skin_strength=0.85, clean_far=4.5, out_bits=8, T_near=3.0, T_far=6.0,
                 far_px=40, pin_seed=4.0, fade_px=80, verbose=True, seed=0, chroma_mult=2.5, fine_keep=0.0, grain=0.0,
                 skin_chroma_free=0.0, chroma_pass=0.0):
    """参数与返回值同 demottle_v10；有 CUDA 时在 GPU 上运行，否则直接调用 v10"""
    global DEV
    DEV = pick_device()
    if DEV is None:
        return V.demottle_v10(img_u8, path=path, strength=strength, skin_strength=skin_strength, clean_far=clean_far,
                              out_bits=out_bits, T_near=T_near, T_far=T_far, far_px=far_px, pin_seed=pin_seed,
                              fade_px=fade_px, verbose=verbose, seed=seed)
    env = dict(kv.split('=') for kv in os.environ.get('DEMOTTLE_V13', '').split(';') if '=' in kv)   # 实验用：回归工具不传参数时覆盖开关
    chroma_mult = float(env.get('chroma_mult', chroma_mult)); fine_keep = float(env.get('fine_keep', fine_keep))
    grain = float(env.get('grain', grain)); skin_chroma_free = float(env.get('skin_chroma_free', skin_chroma_free))
    chroma_pass = float(env.get('chroma_pass', chroma_pass))
    with torch.no_grad():
        _TLS.opt = dict(chroma_mult=float(chroma_mult), fine_keep=float(fine_keep), grain=float(grain), grain_tab=(0.445, 0.363, 0.354, 0.275),
                    skin_chroma_free=float(skin_chroma_free), chroma_pass=float(chroma_pass))
        return _run(img_u8, path, strength, skin_strength, clean_far, out_bits, T_near, T_far, far_px, pin_seed, fade_px, verbose, seed)


def _run(img_u8, path, strength, skin_strength, clean_far, out_bits, T_near, T_far, far_px, pin_seed, fade_px, verbose, seed):
    H, W = img_u8.shape[:2]; s = min(H, W) / 1536.0
    img_t = torch.from_numpy(np.ascontiguousarray(img_u8)).to(DEV)
    img_np = img_u8
    fp_corr, fp_p2p = 0.0, 0.0
    lab0 = None
    if V.jpeg_quant_steps(path) is None:
        lab0 = rgb2lab(img_t)
        if blockiness(lab0[..., 0]) <= 1.15:
            img_t, fp_corr, fp_p2p = periodic_fingerprint(img_t)
            if fp_p2p:
                lab0 = None; img_np = dn(img_t)
    skin_fut = _POOL.submit(_skin_cpu, img_np, s)                 # 人脸与肤色连通域：与 GPU 同时进行
    ctx = analyze(img_t, path, lab0)
    labt = ctx['labt']; sig = ctx['sigma']
    if verbose:
        print(f"device {DEV} | fingerprint 16px corr {fp_corr:.2f} → {'removed, p2p %.2f lvl' % fp_p2p if fp_p2p else 'skipped'}")
        print(f"scale {s:.2f} | {'JPEG' if ctx['jpeg'] else 'clean'} (blockiness {ctx['blk']:.2f}, qDC {ctx['q'][0]*255:.0f}) | "
              f"σ L/a/b {sig[0]:.4f}/{sig[1]:.4f}/{sig[2]:.4f} | period {ctx['period']:.0f}px | measurable {ctx['ok']}")
    if not ctx['ok']:
        skin_fut.result()
        if verbose: print("no measurable mottle → untouched")
        return img_np.copy(), _public(ctx)
    noise_fut = _POOL.submit(_noise, seed, H, W, ctx['jpeg'], out_bits)
    sig_diff = float(np.clip(0.6 * ctx['period'], 30, 60)) * s
    grain_sigma = (3.0 if ctx['jpeg'] else 1.6) * s

    barrier = unified_barrier(ctx)
    S_bin, n_faces = skin_fut.result()
    S_skin = gauss(up(S_bin), 6 * s) if n_faces > 0 else torch.zeros((H, W), device=DEV)

    T_far_eff = T_far if ctx['jpeg'] else (clean_far if clean_far else T_near)
    R = region_pass(labt, ctx, barrier, sig_diff, T_near, T_far_eff, far_px, pin_seed, fade_px, grain_sigma,
                    restrict=clip01(1 - S_skin))
    if R is not None:
        mf = R['m']
        if _opt()['fine_keep'] > 0:                        # 保留斑驳场中 6s 以下的小尺度部分（亮度）
            ml = mf[..., 0]
            mf = torch.cat([(ml - _opt()['fine_keep'] * (ml - gauss(ml, 6 * s)))[..., None], mf[..., 1:]], -1)
        out = R['base'] - strength * mf + R['grain']
    else:
        out = labt.clone()
        R = dict(base=out.clone(), grain=torch.zeros_like(out), dist_b=torch.full((H, W), 1e3, device=DEV))

    if n_faces > 0 and float(S_skin.max()) > 0.5:
        g = grad_mag(out[..., 0], 2.5 * s)
        u_skin = float(median(g[S_skin > 0.5])) + 1e-9
        feat = hysteresis(g / u_skin > 8.0, g / u_skin > 4.0, int(60 * s * s))
        feat = dilate_mask(feat, int(9 * s) | 1)
        block = feat | (S_skin < 0.5)
        fine_rms = torch.sqrt(gauss((out[..., 0] - gauss(out[..., 0], 3.0 * s)) ** 2, 25 * s)) * 100.0
        texture_gate = gauss(clip01((1.0 - fine_rms) / 0.5), 10 * s)
        st = skin_strength * texture_gate
        sig_main = R.get('sig', None)
        for sd, T_far_skin, fade in ((float(np.clip(0.4 * ctx['period'], 20, 40)) * s, 5.0, 20), (15.0 * s, 4.0, 12)):
            Rs = region_pass(out, ctx, block, sd, T_near, T_far_skin, 30, pin_seed, fade, 3.0 * s,
                             restrict=S_skin, chroma_boost=False, sig_override=sig_main, gate='L')
            if Rs is None:
                break
            L_new = Rs['base'][..., 0] - st * Rs['m'][..., 0] + Rs['grain'][..., 0]
            mc = Rs['m'][..., 1:] * 128.0
            dEm = torch.sqrt(mc[..., 0] ** 2 + mc[..., 1] ** 2)
            free = _opt()['skin_chroma_free'] > 0           # 皮肤色度不受质感门（质感门防的是亮度形体被削平）
            act = (S_skin > 0.7) & (dEm > 0) & (True if free else (texture_gate > 0.5))
            sig_c = float(median(dEm[act])) / 1.1774 if int(act.sum()) > 2000 else 1.0
            T_c = max(3.0 * sig_c, 2.5)
            st_c = skin_strength * torch.maximum(texture_gate, torch.full_like(texture_gate, _opt()['skin_chroma_free'])) if free else st
            w_c = st_c * clip01(1 - (dEm / T_c) ** 2)
            ab = [Rs['base'][..., c] - w_c * Rs['m'][..., c] + Rs['grain'][..., c] for c in (1, 2)]
            out = torch.stack([L_new] + ab, -1)
        if verbose:
            print(f"skin: {n_faces} face(s), mask {float(S_skin.mean())*100:.1f}%")

    if _opt()['chroma_pass'] > 0:                          # 色度单独一遍：色度堤坝之间，跨过亮度纹理去色斑（皮肤除外）
        Rc = region_pass(out, ctx, chroma_barrier(ctx), sig_diff, T_near, T_far_eff, far_px, pin_seed, fade_px, grain_sigma,
                         restrict=clip01(1 - S_skin), gate='ab')
        if Rc is not None:
            ab = Rc['base'][..., 1:] - strength * _opt()['chroma_pass'] * Rc['m'][..., 1:] + Rc['grain'][..., 1:]
            out = torch.cat([out[..., :1], ab], -1)

    nz, di = noise_fut.result()
    Lref = clip01(labt[..., 0])
    if ctx['jpeg']:
        fine_L = labt[..., 0] - gauss(labt[..., 0], 3.0 * s)
        iso_fine = gauss(clip01((0.55 - coherence(fine_L, 12 * s)) / 0.20), 6 * s)
        interior = (gauss(clip01(R['dist_b'] / (24 * s)), 10 * s) * (1 - S_skin) * iso_fine)[..., None]
        sel = R['dist_b'] > 24 * s
        flat_amp = np.array([mad(R['grain'][..., c][sel]) if bool(sel.any()) else 0 for c in range(3)], np.float32)
        amp = up(np.maximum(flat_amp * 0.6, np.array([0.0030, 0.0010, 0.0010], np.float32)))
        n = torch.stack([gauss(nz[c], 0.7 * s) for c in range(3)], -1)
        n = n / n.std(dim=(0, 1), keepdim=True, correction=0)
        shade = (clip01(Lref / 0.08) * (1.2 - 0.4 * Lref))[..., None]
        syn = n * amp * shade
        syn = torch.cat([syn[..., :1], syn[..., 1:] * 0.3], -1)
        out = out - R['grain'] * interior + syn * interior + 0.4 * syn * (1 - interior)
    if out_bits == 8:
        dith = (di - 0.5) * 0.004 * clip01(Lref / 0.05)
        out = torch.cat([(out[..., 0] + dith)[..., None], out[..., 1:]], -1)

    if _opt()['grain'] > 0:                                # 细纹理补偿：非皮肤区，把细节 rms 补到实拍水平（只补差额）
        Lc = out[..., 0]
        cur = torch.sqrt(gauss((Lc - gauss(Lc, 1.5 * s)) ** 2, 8 * s)) * 100.0
        # 目标：配对实拍原图平坦区的细纹理 rms（L*），按亮度分档插值（tools/fine_level.py 实测）
        Lstar = clip01(Lref) * 100.0
        xs = torch.tensor([21.0, 40.0, 60.0, 80.0], device=DEV); ys = torch.tensor(_opt()['grain_tab'], device=DEV)
        idx = torch.clamp(torch.searchsorted(xs, Lstar.contiguous()) - 1, 0, 2)
        t = clip01((Lstar - xs[idx]) / (xs[idx + 1] - xs[idx]))
        tgt = (ys[idx] + (ys[idx + 1] - ys[idx]) * t)
        tgt = tgt * clip01((Lstar - 4.0) / 8.0) * clip01((98.0 - Lstar) / 6.0)   # 纯黑 / 纯白不加
        add = torch.sqrt(_opt()['grain'] * torch.clamp(tgt ** 2 - cur ** 2, min=0)) / 100.0   # 补足缺口能量的 grain 倍
        zone = clip01(1 - S_skin)                        # 皮肤细节生图已多于实拍，不加
        nz = _grain_field(seed, H, W, s)
        out = torch.cat([(Lc + add * zone * nz)[..., None], out[..., 1:]], -1)
    rgbf = lab2rgb01(out)
    if out_bits == 16:
        res = dn((rgbf * 65535 + 0.5).to(torch.int32)).astype(np.uint16)
        rgb = None
    else:
        rgb = dn((rgbf * 255 + 0.5).to(torch.uint8)); res = rgb
    if verbose:
        if rgb is None:
            rgb = dn((rgbf * 255 + 0.5).to(torch.uint8))
        V.metrics(img_np, rgb, _public(ctx), dn(barrier))
    return res, _public(ctx)


class _Ctx(dict):
    """返回给调用方的 ctx：键与 v10 相同；'lab' 在首次读取时才从显存下载"""
    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        if k == 'lab' and callable(v):
            v = v(); dict.__setitem__(self, k, v)
        return v

    def get(self, k, d=None):
        return self[k] if k in self else d


def _public(ctx):
    labt = ctx['labt']
    c = _Ctx({k: v for k, v in ctx.items() if k != 'labt'})
    dict.__setitem__(c, 'lab', lambda: dn(labt))
    return c


def warmup(sizes=((2048, 1536),), bits=(8,)):
    """服务启动时调用：CUDA 初始化、Triton 编译、扩散图捕获、查表与随机数缓存（按常见尺寸）"""
    for (h, w) in sizes:
        rng = np.random.default_rng(1)
        img = (rng.random((h, w, 3)) * 40 + 100).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 20)
        for b in bits:
            demottle_v13(img, out_bits=b, verbose=False)
    if DEV is not None and DEV.type == 'cuda':
        torch.cuda.synchronize()


if __name__ == "__main__":
    import sys, time
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    bits = 16 if '--16bit' in sys.argv else 8
    im = cv2.cvtColor(cv2.imread(args[0]), cv2.COLOR_BGR2RGB)
    t = time.time(); out, _ = demottle_v13(im, path=args[0], out_bits=bits)
    if DEV is not None and DEV.type == 'cuda':
        torch.cuda.synchronize()
    print(f"{time.time()-t:.1f}s")
    cv2.imwrite(args[1], cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
