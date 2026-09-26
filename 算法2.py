# -*- coding: utf-8 -*-
"""
算法02 —— 低频段除杂增强版 (双分辨率 + 空间似然比低频掩码 + 频变可信度门限)
模块化可开关, 便于逐环节消融。

对应 diag*.py 实测问题:
 P0 次声/直流无法掩蔽 : 86.13Hz 栅格下 0-43Hz 全在被显式跳过的 bin0 -> 前端零相位高通 + 守护带静音
 P1 频率栅格过粗      : 整数倍抽取(352.8k->44.1k), bin 86.13->21.53 Hz, 低频段 bin 数 x4, 算力 /8
 P2 逐bin IPD方差~1/f : 低频段改用带内"相位斜率联合最小二乘" + 双假设空间似然比掩码
 P3 掩码核宽固定      : sigma(f) 由通道间相干度导出的 CRLB 决定
 P4 掩码下限/平滑     : 低频允许置零, 掩码时频二维平滑 + 锐化
 P5 模板自校准        : 一维搜索 DS 相干增益最大的延时尺度, 取代手调分位数
"""
import numpy as np
from scipy.signal import stft, istft, butter, sosfiltfilt, resample_poly
from scipy.ndimage import uniform_filter1d

FS0 = 352800.0
C = 343.0
SPACING = 0.008


def _butter(kind, fc, fs, order=4):
    return butter(order, np.asarray(fc) / (fs / 2), btype=kind, output='sos')


def front_end(six, fs=FS0, fsub=25.0, decim=8, aa_order=8, norm_ch=False):
    if fsub:
        six = sosfiltfilt(_butter('high', fsub, fs), six, axis=-1)
    fsd = fs / decim
    if decim > 1:
        six = sosfiltfilt(_butter('low', fsd / 2 * 0.9, fs, order=aa_order), six, axis=-1)[:, ::decim]
    if norm_ch:
        g = np.sqrt((six**2).mean(axis=1)); g = g.mean() / (g + 1e-12)
        six = six * g[:, None]
    return six.astype(np.float64), fsd


def phase_slope_itd(A, B, f, band, fs):
    """带内多 bin 加权最小二乘回归 dphi = -2*pi*f*tau, 逐帧输出 tau(样本)。A,B:(nbins,nframes)"""
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3:
        return np.zeros(A.shape[1]), np.zeros(A.shape[1])
    a, b = A[m], B[m]
    dphi = np.angle(b * np.conj(a))
    ang = (2 * np.pi * f[m])[:, None]
    w = np.abs(a) * np.abs(b)
    tau = np.sum(w * dphi * ang, axis=0) / (np.sum(w * ang * ang, axis=0) + 1e-30)
    return -tau * fs, np.sum(w, axis=0)


def template_tdoa(Z, f, fs, num_mics, band=(1200, 6400), qhi=0.65, mode='slope'):
    pair = []
    for i in range(num_mics - 1):
        if mode == 'ipd':
            tau = []
            for fi in np.where((f > 30) & (f < fs / 6))[0]:
                ipd = np.angle(Z[i + 1, fi] * np.conj(Z[i, fi]))
                tau.append(-ipd / (2 * np.pi * f[fi]) * fs)
            tau = np.array(tau)
        else:
            tau, _ = phase_slope_itd(Z[i], Z[i + 1], f, band, fs)
        tau = tau[np.isfinite(tau)]
        pair.append(float(np.quantile(tau, qhi)))
    pair = np.array(pair)
    return pair, np.concatenate([[0.0], np.cumsum(pair)])


def interferer_tdoa(Z, f, fs, num_mics, band, qlo=0.35):
    """低频干扰 ITD 模板: 低频能量被干扰主导 -> 帧级带内回归 ITD 的低分位数"""
    pair = []
    for i in range(num_mics - 1):
        tau, _ = phase_slope_itd(Z[i], Z[i + 1], f, band, fs)
        pair.append(float(np.quantile(tau[np.isfinite(tau)], qlo)))
    pair = np.array(pair)
    return pair, np.concatenate([[0.0], np.cumsum(pair)])


def coherence_gain(Z, f, fsd, cum, band=(1600, 6400)):
    """P5 无监督判据: 目标主导带内 DS 输出能量 / 平均通道能量 (相干增益, dB)"""
    m = (f >= band[0]) & (f <= band[1])
    A = np.zeros((int(m.sum()), Z.shape[2]), dtype=complex)
    for i in range(len(cum)):
        A += Z[i, m] * np.exp(1j * 2 * np.pi * f[m][:, None] * cum[i] / fsd)
    A /= len(cum)
    return 10 * np.log10((np.abs(A)**2).mean() / ((np.abs(Z[:, m])**2).mean() + 1e-30))


def coherence_sigma(Z, f, fs, num_mics):
    """P3: sigma_tau(f) = sigma_phi/(2*pi*f/fs), sigma_phi^2 = (1-g^2)/(2 g^2) 单次快拍 CRLB"""
    nf = Z.shape[1]; acc = np.zeros(nf)
    for i in range(num_mics - 1):
        Sxy = Z[i + 1] * np.conj(Z[i])
        Sxx = (np.abs(Z[i + 1])**2 + np.abs(Z[i])**2) / 2
        g2 = np.clip(np.abs(Sxy)**2 / (Sxx**2 + 1e-30), 1e-3, 1.0)
        acc += np.mean((1 - g2) / (2 * g2), axis=1)
    var = acc / (num_mics - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        st = np.sqrt(var) / (2 * np.pi * f / fs)
    st[0] = np.inf
    return np.nan_to_num(st, posinf=np.inf)


def lf_frame_gain(Z, f, fs, pair_t, band, sigma_min, kappa, n_avg_win=1):
    """低频段帧级方向判决(logistic), 输出逐帧增益"""
    num_mics = Z.shape[0]
    taus, ws = [], []
    for i in range(num_mics - 1):
        tau, w = phase_slope_itd(Z[i], Z[i + 1], f, band, fs)
        taus.append(tau); ws.append(np.maximum(w, 0))
    taus = np.array(taus); ws = np.array(ws)
    tau_bar = np.sum(taus * ws, axis=0) / (np.sum(ws, axis=0) + 1e-30)
    tau_bar = np.where(np.isfinite(tau_bar), tau_bar, 0.0)
    if n_avg_win > 1:
        tau_bar = uniform_filter1d(tau_bar, n_avg_win, mode='nearest')
    tgt = max(float(np.mean(pair_t)), 1e-3)
    z = (tau_bar - tgt / 2.0) / max(tgt / 2.0, sigma_min) * 3.0
    return 1.0 / (1.0 + np.exp(-kappa * z)), tau_bar


def lf_slm(Z, f, fsd, cum_t, cum_i, beta=1.0, gamma=1.0):
    """P2 低频逐时频单元空间似然比掩码: 目标/干扰两个导向矢量做匹配滤波, 掩码=归一化功率比.
    与帧级判决的区别: 目标与干扰同时存在时仍可在 TF 域分离(利用 W-disjoint 稀疏性)."""
    nf, nfr = len(f), Z.shape[2]
    num = Z.shape[0]
    mask = np.ones((nf, nfr))
    for fi in range(1, nf):
        at = np.exp(1j * 2 * np.pi * f[fi] * np.asarray(cum_t) / fsd)
        ai = np.exp(1j * 2 * np.pi * f[fi] * np.asarray(cum_i) / fsd)
        Xt = np.abs(at.conj() @ Z[:, fi]) / num
        Xi = np.abs(ai.conj() @ Z[:, fi]) / num
        pt, pi = Xt**(2 * beta) + 1e-30, Xi**(2 * gamma) + 1e-30
        mask[fi] = pt / (pt + pi)
    return mask


def lf_deflate(Z, f, fsd, cum_t, band, mu=1.0, nbin_smooth=5, return_diag=False):
    """P2(推荐) 低频子空间抵消: 在 LF 段用交叉谱矩阵主特征向量刻画相干干扰的空间签名,
    再经"目标无失真投影"去除其可去除分量 —— 等价于单约束 LCMV/GSC 固定波束形成器。
    与逐bin IPD 掩码不同: 不依赖 1/f 的相位分辨率, 低频仍有效。
      B = I - a_t a_t^H/||a_t||^2      (与目标正交的投影, 保证目标无失真)
      v = B v1 / ||B v1||              (干扰中可去除的部分)
      X' = X - mu * v v^H X
    返回 (deflated_Z, removable) ; removable = ||B v1||^2/||v1||^2 in [0,1] 越低越难消。"""
    num, nf, nfr = Z.shape
    a_t = np.exp(1j * 2 * np.pi * f[:, None] * np.asarray(cum_t)[None, :] / fsd)   # (nf, num)
    m = (f >= band[0]) & (f <= band[1])
    idx = np.where(m)[0]
    out = Z.copy(); remov = np.zeros(nf)
    for fi in idx:
        lo, hi = max(0, fi - nbin_smooth), min(nf, fi + nbin_smooth + 1)
        Zs = Z[:, lo:hi, :]                                        # (num, nb, nfr)
        R = np.einsum('ijk,ljk->il', Zs, Zs.conj()) / (Zs.shape[1] * Zs.shape[2])
        w, V = np.linalg.eigh(R)
        v1 = V[:, -1]                                              # 主导(相干干扰)空间签名
        at = a_t[fi]
        Bv = v1 - at * (np.dot(at.conj(), v1) / (np.dot(at.conj(), at) + 1e-30))
        r = float(np.dot(Bv.conj(), Bv).real / (np.dot(v1.conj(), v1).real + 1e-30))
        remov[fi] = r
        if r < 1e-6:
            continue
        v = Bv / np.sqrt(np.dot(Bv.conj(), Bv).real + 1e-30)
        out[:, fi, :] -= mu * v[:, None] * (v.conj() @ Z[:, fi, :])
    if return_diag:
        return out, remov, a_t
    return out, remov


def _wrap(x):
    """把相位差折回 [-pi, pi)。"""
    return (x + np.pi) % (2 * np.pi) - np.pi


def lf_joint(Z, f, fs, pair_t, pair_i, f_lf=1200.0, win=31,
             ks=1.5, kp=1.0, ki=1.5, ild_hi=200.0, target_band=(1600, 6400)):
    """阶段一低频掩码: 跨帧长时积分的"相位斜率假设检验 + 近场 ILD"联合似然。

    与 lf_slm(单快拍、逐 bin 导向匹配)的区别:
      1) 相位证据在 win 帧(默认31帧≈0.36s)上、**逐频率 bin** 累积交叉相位残差,
         ITD 方差按 ~1/sqrt(T) 下降。目标低频能量是稀疏持续线谱(43/48/146/309Hz...),
         宽带干扰占据其余 bin; 跨 bin 池化会被宽带干扰主导, 故判决保留 bin 级分辨;
      2) 增加逐通道对数幅度(ILD)线索: 近场目标 ch1->ch6 递减, 低频干扰反向递增,
         ILD 权重随频率衰减(实测 200Hz 以上梯度消失), 60Hz 以下仅靠 ILD;
      3) 输出有界软掩码(0,1), 不做守护带静音, 深度由外部 floor_lf 兜底。

    似然比(每 bin b、帧 t):
      l_ph = 0.5*mean_pairs(S_i-S_t)/var_phi_b   S_h = bin b 窗内逐帧加权残差均值
      l_il = 0.5*(d_i - d_t)/var_e               d_h = 窗内 ILD 向量与假设 h 模板距离
      mask = sigmoid(kp*l_ph + ki*w_ild(f_b)*l_il)
    """
    num, nf, nfr = Z.shape
    P = num - 1
    f = np.asarray(f)
    win = int(win)
    hw = win // 2

    # ---------- 预计算: 相邻对瞬时相位差 / 能量权重 / 两种假设下的残差 ----------
    dphi = np.stack([np.angle(Z[i + 1] * np.conj(Z[i])) for i in range(P)])     # (P,nf,nfr)
    wgt = np.stack([np.abs(Z[i]) * np.abs(Z[i + 1]) for i in range(P)])        # (P,nf,nfr)
    ang = (2 * np.pi * f / fs)[None, :, None]                                  # (1,nf,1)
    rt = _wrap(dphi + ang * np.asarray(pair_t)[:, None, None])
    ri = _wrap(dphi + ang * np.asarray(pair_i)[:, None, None])
    # 时间累积(cumsum 实现滑动窗 O(1) 查询)
    cw = np.concatenate([np.zeros((P, nf, 1)), np.cumsum(wgt, axis=2)], axis=2)
    ct = np.concatenate([np.zeros((P, nf, 1)), np.cumsum(wgt * rt ** 2, axis=2)], axis=2)
    ci = np.concatenate([np.zeros((P, nf, 1)), np.cumsum(wgt * ri ** 2, axis=2)], axis=2)

    # ---------- 相位噪声方差: 时间平均互谱给出的真 MSC -> var_phi=(1-g2)/(2g2),
    #            逐 bin(相邻 5 bin 平滑); 单快拍 |Sxy|^2/Sxx^2 的期望有偏、不能用 ----------
    g2 = []
    for i in range(P):
        Sxy = (Z[i + 1] * np.conj(Z[i])).mean(axis=1)
        Sxx = (np.abs(Z[i + 1]) ** 2).mean(axis=1)
        Syy = (np.abs(Z[i]) ** 2).mean(axis=1)
        msc = np.clip(np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-30), 1e-3, 1 - 1e-6)
        g2.append((1 - msc) / (2 * msc))
    var_phi_f = np.clip(np.mean(np.stack(g2), axis=0), 0.02, 4.0)                # (nf,)
    var_phi_f = uniform_filter1d(var_phi_f, 5, mode='nearest')

    # ---------- ILD 权重随频率衰减: 60Hz 以下仅 ILD, 200Hz 以上快速淡出 ----------
    b60 = int(np.searchsorted(f, 60, side='left'))
    b_lf = int(np.searchsorted(f, f_lf, side='right'))
    w_ild_f = np.zeros(nf)
    w_ild_f[1:b60] = 1.0
    w_ild_f[b60:int(np.searchsorted(f, 120, side='right'))] = 1.0
    w_ild_f[int(np.searchsorted(f, 120, side='right')):int(np.searchsorted(f, 250, side='right'))] = 0.5
    w_ild_f[int(np.searchsorted(f, 250, side='right')):int(np.searchsorted(f, 500, side='right'))] = 0.15

    # ---------- ILD: 60-ild_hi 带内逐通道对数能量, 窗内平均 ----------
    mb = (f >= 60) & (f <= ild_hi)
    E = np.log(np.sum(np.abs(Z[:, mb, :]) ** 2, axis=1) + 1e-20)               # (num,nfr)
    ce = np.concatenate([np.zeros((num, 1)), np.cumsum(E, axis=1)], axis=1)
    # 目标 ILD 模板: 目标主导带内逐帧归一化后长时平均
    mt = (f >= target_band[0]) & (f <= target_band[1])
    Et = np.log(np.sum(np.abs(Z[:, mt, :]) ** 2, axis=1) + 1e-20)
    tt = Et - Et.mean(axis=0, keepdims=True)
    tt = np.median(tt, axis=1)
    tt = tt - tt.mean()
    # 干扰 ILD 模板: 低频带窗均值归一化后的全程中位
    ti = np.median(E - E.mean(axis=0, keepdims=True), axis=1)
    ti = ti - ti.mean()
    mid = 0.5 * (tt + ti)
    var_e = max(float(np.mean((((E - E.mean(axis=0, keepdims=True)) - mid[:, None]) ** 2))), 1e-3)

    tt_idx = np.arange(nfr)
    t0 = np.maximum(0, tt_idx - hw)
    t1 = np.minimum(nfr, tt_idx + hw + 1)

    # ---------- ILD 似然(向量化滑窗) ----------
    Ew = (ce[:, t1] - ce[:, t0]) / (t1 - t0)[None, :]                      # (num,nfr)
    Ew -= Ew.mean(axis=0, keepdims=True)
    d_t = np.sum((Ew - tt[:, None]) ** 2, axis=0)
    d_i = np.sum((Ew - ti[:, None]) ** 2, axis=0)
    l_ild = 0.5 * (d_i - d_t) / var_e

    # ---------- 逐 bin 相位证据 ----------
    # 静态 L_s[b]: 全程积分 LLR。低频目标是持续线谱、干扰平稳宽带, 二者在"全程斜率"
    #              上稳定分离(实测目标线谱 bin +0.1..+0.5, 干扰线谱 bin -1..-4);
    # 动态 L[b,t]: 0.36s 滑窗 LLR; 单 bin 窗斜率噪声大, 以其对 L_s 的偏离量作时变修正。
    mask = np.ones((nf, nfr))
    L = np.zeros((nf, nfr))
    CW = cw[:, b60:b_lf, :][:, :, t1] - cw[:, b60:b_lf, :][:, :, t0]       # (P,nb,nfr)
    ST = (ct[:, b60:b_lf, :][:, :, t1] - ct[:, b60:b_lf, :][:, :, t0]) / (CW + 1e-30)
    SI = (ci[:, b60:b_lf, :][:, :, t1] - ci[:, b60:b_lf, :][:, :, t0]) / (CW + 1e-30)
    vp = np.maximum(var_phi_f[b60:b_lf], 0.03)                             # (nb,)
    L[b60:b_lf] = 0.5 * np.mean(SI - ST, axis=0) / vp[:, None]

    STs = (ct[:, b60:b_lf, nfr] - ct[:, b60:b_lf, 0]) / (cw[:, b60:b_lf, nfr] - cw[:, b60:b_lf, 0] + 1e-30)
    SIs = (ci[:, b60:b_lf, nfr] - ci[:, b60:b_lf, 0]) / (cw[:, b60:b_lf, nfr] - cw[:, b60:b_lf, 0] + 1e-30)
    Ls = np.zeros(nf)
    Ls[b60:b_lf] = 0.5 * np.mean(SIs - STs, axis=0) / vp

    z = ks * Ls[:, None] + kp * (L - Ls[:, None]) + ki * w_ild_f[:, None] * l_ild[None, :]
    z = np.clip(z, -50, 50)
    g = 1.0 / (1.0 + np.exp(-z))
    mask[1:b_lf, :] = g[1:b_lf, :]
    l_ph_store = {}
    for lo, hi in [(1, 60), (60, 120), (120, 250), (250, 500), (500, 800), (800, f_lf)]:
        i0 = int(np.searchsorted(f, lo, side='left'))
        i1 = int(np.searchsorted(f, hi, side='right'))
        l_ph_store[(lo, hi)] = L[i0:i1].mean(axis=0)
    return mask, dict(l_ild=l_ild, var_e=var_e, l_ph=l_ph_store, L=L, Ls=Ls)


def mwf_compose(Z, f, fsd, cum_t, mask, aligned,
                f_keep=100.0, f_mpdr=400.0, f_mask=800.0, f_mwf=2200.0,
                win=63, block=63, nbin=3, lam_lo=0.12, lam_hi=0.02):
    """阶段二: 分段多通道维纳方向滤波(单条抑制链, 按频段选最优证据)。
      <100Hz      : DS 直通(电小阵列空间签名共线, 保目标基音)
      100-400Hz   : 时变 MPDR —— 滑窗混合协方差 Rx(t), 对角加载, 纯空间零陷;
                    该带掩码不可靠而干扰空间统计可用, 无需 VAD/掩码
      400-800Hz   : joint 长时似然软掩码 × DS(阶段一证据最强的链路)
      800-2200Hz  : 分块掩码 MWF(秩1: MVDR + 后置维纳), 掩码加权 Phi_s/Phi_n,
                    块内协方差跟踪时变空间统计; 空间零陷优于标量掩码
      >2200Hz     : DS(hf sigma 在混叠下退化为静态衰减, 实测有害)
    对角加载量 = lam * trace(R)/M(相对加载, 与频带绝对能量无关)。
    """
    num, nf, nfr = Z.shape
    Y = aligned.copy()
    av_all = np.exp(-1j * 2 * np.pi * f[:, None] * cum_t[None, :] / fsd)   # (nf,num)
    i_keep = int(np.searchsorted(f, f_keep, side='right'))
    i_mpdr = int(np.searchsorted(f, f_mpdr, side='right'))
    i_mask = int(np.searchsorted(f, f_mask, side='right'))
    i_mwf = int(np.searchsorted(f, f_mwf, side='right'))
    hw = win // 2

    # ---------- 100-400Hz: 逐帧时变 MPDR ----------
    for fi in range(i_keep, i_mpdr):
        lo, hi = max(0, fi - nbin), min(nf, fi + nbin + 1)
        av = av_all[fi]
        for t in range(nfr):
            t0, t1 = max(0, t - hw), min(nfr, t + hw + 1)
            Xb = Z[:, lo:hi, t0:t1].reshape(num, -1)
            Rx = Xb @ Xb.conj().T / Xb.shape[1]
            Rx = 0.5 * (Rx + Rx.conj().T)
            Rinv = np.linalg.inv(Rx + lam_lo * np.trace(Rx).real / num * np.eye(num))
            u = Rinv @ av
            w = u / complex(av.conj() @ u)
            Y[fi, t] = w.conj() @ Z[:, fi, t]

    # ---------- 400-800Hz: joint 软掩码 ----------
    Y[i_mpdr:i_mask] = mask[i_mpdr:i_mask] * aligned[i_mpdr:i_mask]

    # ---------- 800-2200Hz: 分块掩码 MWF ----------
    for t0 in range(0, nfr, block):
        t1 = min(nfr, t0 + block)
        for fi in range(i_mask, i_mwf):
            lo, hi = max(0, fi - nbin), min(nf, fi + nbin + 1)
            Zs = Z[:, lo:hi, t0:t1]
            ww = mask[lo:hi, t0:t1][None, :, :]
            Ps = np.einsum('mkt,nkt->mn', Zs * ww, Zs.conj()) / max(float(ww.sum()), 1e-30)
            Pn = np.einsum('mkt,nkt->mn', Zs * (1 - ww), Zs.conj()) / (float((1 - ww).sum()) + 1e-30)
            Ps = 0.5 * (Ps + Ps.conj().T)
            Pn = 0.5 * (Pn + Pn.conj().T)
            av = av_all[fi]
            Rinv = np.linalg.inv(Pn + lam_hi * np.trace(Pn).real / num * np.eye(num))
            u = Rinv @ av
            den = complex(av.conj() @ u).real + 1e-30
            sig2 = max(0.0, complex(av.conj() @ Ps @ av).real
                       / complex(av.conj() @ av).real)
            G = min(1.0, sig2 / (sig2 + 1.0 / den))
            w = G * u / den
            Y[fi, t0:t1] = w.conj() @ Z[:, fi, t0:t1]
    return Y


def alg02(six, fs=FS0, num_mics=6, mic_spacing=SPACING, decim=8, fsub=25.0, norm_ch=False,
          nperseg=2048, ratio=4,
          use_ds=True, f_guard=90.0, f_lf=1200.0, lf_mode='slm', lf_sigmin=0.35, lf_kappa=2.5,
          lf_navgw=1, lf_beta=1.0, lf_band=(120, 1200), qlo=0.35, defl_mu=1.0, defl_nb=5,
          joint_win=31, joint_ks=1.5, joint_kp=1.0, joint_ki=1.5, joint_ild_hi=200.0,
          joint_lo_f=100.0, joint_lo_floor=0.8,
          mwf_keep=100.0, mwf_mpdr=400.0, mwf_mask=800.0, mwf_f=2200.0,
          mwf_win=63, mwf_block=63, mwf_nbin=3, mwf_lam_lo=0.12, mwf_lam_hi=0.02,
          hf_mode='sigma', hf_k=2.0, hf_floor=0.5, hf_cap=3.0,
          template_band=(1200, 6400), qhi=0.65, tmpl_mode='slope', tmpl_override=None,
          calib=None, calib_band=(1600, 6400),
          sm_f=3, sm_t=5, sharp=0.5, floor_lf=0.0, floor_hf=0.05, return_diag=False):
    n0 = six.shape[1]
    if lf_mode == 'mwf':
        # MWF 段(至 mwf_f)需要 joint 掩码作协方差权重, f_lf 必须覆盖该段
        f_lf = max(f_lf, mwf_f)
    x, fsd = front_end(six, fs=fs, fsub=fsub, decim=decim, norm_ch=norm_ch)
    nover = nperseg - nperseg // ratio
    Z = np.stack([stft(x[i], fs=fsd, nperseg=nperseg, noverlap=nover)[2] for i in range(num_mics)])
    f = np.fft.rfftfreq(nperseg, 1.0 / fsd)
    nf, nfr = len(f), Z.shape[2]

    if tmpl_override is not None:
        pair_t = np.asarray(tmpl_override, dtype=float)
    else:
        pair_t, _ = template_tdoa(Z, f, fsd, num_mics, template_band, qhi, tmpl_mode)
    cum_t = np.concatenate([[0.0], np.cumsum(pair_t)])
    if calib is not None:            # P5: 一维尺度搜索, 最大化无监督相干增益
        grid = np.linspace(calib[0], calib[1], 25) if not np.isscalar(calib) else calib
        best = max(grid, key=lambda s: coherence_gain(Z, f, fsd, cum_t * s, calib_band))
        cum_t = cum_t * best
        pair_t = pair_t * best
    stau = coherence_sigma(Z, f, fsd, num_mics)
    d_far = mic_spacing / C * fsd
    f_unamb = fsd / (2 * max(np.max(np.abs(pair_t)), d_far))

    # ---------- 低频子空间抵消(在 DS 之前) ----------
    Zds = Z
    remov = None
    if lf_mode.startswith('deflate'):
        Zds, remov = lf_deflate(Z, f, fsd, cum_t, (f_guard, f_lf), mu=defl_mu, nbin_smooth=defl_nb)
    # ---------- 延时求和 ----------
    if use_ds:
        aligned = np.zeros((nf, nfr), dtype=complex)
        for i in range(num_mics):
            aligned += Zds[i] * np.exp(1j * 2 * np.pi * f[:, None] * cum_t[i] / fsd)
        aligned /= num_mics
    else:
        aligned = Zds[0].copy()

    # ---------- 掩码 ----------
    mask = np.ones((nf, nfr))
    tau_bar = lf = None
    cum_i = None
    slm = None
    joint = jdiag = None
    if lf_mode in ('frame', 'both'):
        lf, tau_bar = lf_frame_gain(Z, f, fsd, pair_t, (f_guard, f_lf), lf_sigmin, lf_kappa, lf_navgw)
    if lf_mode in ('slm', 'both', 'deflate+slm', 'joint', 'deflate+joint', 'mwf'):
        _, cum_i = interferer_tdoa(Z, f, fsd, num_mics, lf_band, qlo)
        if lf_mode in ('joint', 'deflate+joint', 'mwf'):
            joint, jdiag = lf_joint(Zds, f, fsd, pair_t, np.diff(cum_i), f_lf=f_lf,
                                    win=joint_win, ks=joint_ks, kp=joint_kp, ki=joint_ki,
                                    ild_hi=joint_ild_hi, target_band=template_band)
        else:
            slm = lf_slm(Z, f, fsd, cum_t, cum_i, lf_beta)
    for fi in range(1, nf):
        freq = f[fi]
        # joint/mwf 模式不做守护带静音: 极低频由联合软掩码 + floor_lf 处理
        if lf_mode not in ('joint', 'deflate+joint', 'mwf') and f_guard and freq < f_guard:
            mask[fi] = 0.0
            continue
        if freq <= f_lf:
            if lf_mode == 'frame':
                mask[fi] = lf
            elif lf_mode == 'slm':
                mask[fi] = slm[fi]
            elif lf_mode in ('joint', 'deflate+joint', 'mwf'):
                mask[fi] = joint[fi]
            elif lf_mode == 'both':
                mask[fi] = slm[fi] * lf
            elif lf_mode == 'deflate+slm':
                mask[fi] = slm[fi]
            continue
        if freq > f_unamb or hf_mode == 'none' or lf_mode == 'mwf':
            continue
        if hf_mode == 'const':
            s = max(np.mean(np.abs(pair_t)) * 0.22, 3.0 / decim)
        else:
            s = np.clip(hf_k * stau[fi], hf_floor, hf_cap)
        dev = []
        for i in range(num_mics - 1):
            ipd = np.angle(Z[i + 1, fi] * np.conj(Z[i, fi]))
            raw = -ipd / (2 * np.pi * freq) * fsd
            dev.append(raw + np.round((pair_t[i] - raw) * freq / fsd) - pair_t[i])
        dev = np.array(dev)
        mask[fi] = np.exp(-np.mean(dev**2, axis=0) / (2 * s**2))

    # ---------- 平滑 / 锐化 / 下限 ----------
    if sm_f > 1:
        mask = uniform_filter1d(mask, sm_f, axis=0, mode='nearest')
    if sm_t > 1:
        mask = uniform_filter1d(mask, sm_t, axis=1, mode='nearest')
    if sharp != 1.0:
        mask = np.clip(mask, 0, 1) ** sharp
    mask = np.maximum(mask, np.where(f <= f_lf, floor_lf, floor_hf)[:, None])
    # joint/mwf: 极低频(<100Hz)相位/ILD 指纹在目标演奏帧同样误判, 强制保底以保目标基音
    if lf_mode in ('joint', 'deflate+joint', 'mwf') and joint_lo_floor is not None:
        mask[(f > 0) & (f <= joint_lo_f)] = np.maximum(
            mask[(f > 0) & (f <= joint_lo_f)], joint_lo_floor)
    mask[:1] = 0.0

    # ---------- 输出谱 ----------
    if lf_mode == 'mwf':
        spec_out = mwf_compose(Zds, f, fsd, cum_t, mask, aligned,
                               f_keep=mwf_keep, f_mpdr=mwf_mpdr, f_mask=mwf_mask, f_mwf=mwf_f,
                               win=mwf_win, block=mwf_block, nbin=mwf_nbin,
                               lam_lo=mwf_lam_lo, lam_hi=mwf_lam_hi)
    else:
        spec_out = aligned * mask
    _, out = istft(spec_out, fs=fsd, nperseg=nperseg, noverlap=nover)
    out = out[:x.shape[1]]
    if decim > 1:
        out = resample_poly(out, decim, 1)[:n0]
    if return_diag:
        return out, dict(f=f, fsd=fsd, mask=mask, pair_t=pair_t, cum_t=cum_t, stau=stau,
                         tau_bar=tau_bar, lf=lf, cum_i=cum_i, f_unamb=f_unamb, aligned=aligned, remov=remov,
                         joint=joint, jdiag=jdiag, spec_out=spec_out,
                         cg=coherence_gain(Z, f, fsd, cum_t, calib_band), Z=Z, x=x)
    return out
