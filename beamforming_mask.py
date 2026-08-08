# -*- coding: utf-8 -*-
"""
麦克风阵列盲源分离 - 基于通道间相位差(IPD)的方向性时频掩码
============================================================
适用场景: 6麦克风线阵(间距8mm), 目标音源在端射方向(1→6传播, 近场),
          干扰来自侧面(broadside, 近场). 目标与干扰的到达时间差(TDOA)
          差异极大(目标相邻对TDOA≈18-20样本, 干扰≈0), 时频域高度可分.

原理:
1. 能量加权IPD估计目标TDOA: 用1号麦克风(目标最近,能量最强)的能量作权重,
   在目标占优的时频单元估计各相邻对的瞬时相位差, 反推目标真实TDOA.
   (混合信号下GCC-PHAT会锁定强干扰方向, 不可用)
2. STFT分帧转到时频域
3. 延时求和波束形成: 按目标真实TDOA对齐各通道并相加, 目标方向无失真,
   同时让broadside干扰不同相而部分抵消
4. 方向性时频掩码: 每个时频单元, 用相邻对瞬时IPD估计该单元主导声源的TDOA,
   与目标TDOA比对生成软掩码(接近目标→1, 接近0即侧面干扰→0)
5. 掩码乘到延时求和输出上, 逐单元抑制侧面干扰
6. ISTFT重建时域信号

优点(相对MVDR/GSC):
- 逐时频单元判别方向, 不依赖协方差矩阵估计(单快拍下MVDR不稳定易过抑制)
- 利用语音时频稀疏性(W-disjoint正交性), 方向差异越大分离越彻底
- 延时求和保证目标方向无失真, 掩码只做幅度衰减不做相位扰动
- 能量加权IPD可在强干扰下提取目标TDOA(GCC失效的场景)
"""

import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d


# ==================== 目标TDOA估计(能量加权IPD) ====================

def estimate_target_tdoa(data, fs, num_mics, mic_spacing, c=343.0):
    """
    在强干扰存在下估计目标方向的TDOA.
    利用先验: 目标在1号麦克风能量最强(端射方向, 近场), 且目标TDOA为正
    (1→6传播), 干扰TDOA≈0(broadside). 用1号能量作权重, 收集所有无模糊
    时频单元的解包裹TDOA, 取加权高分位数(目标TDOA>干扰TDOA, 高分位偏向目标).

    Returns:
        tdoas: (num_mics,) 各麦克风相对1号的TDOA(样本, 正=i号晚于1号)
        pair_tdoas: (num_mics-1,) 相邻对目标TDOA(用于掩码判别模板)
    """
    nperseg = 4096
    noverlap = nperseg * 3 // 4
    stfts = []
    for i in range(num_mics):
        f, _, Z = stft(data[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        stfts.append(Z)
    stfts = np.array(stfts)  # (mics, freqs, frames)

    d_far = mic_spacing / c * fs  # 远场相邻对TDOA(样本)
    w = np.abs(stfts[0])  # 1号能量(目标最强)

    pair_tdoas = []
    for i in range(num_mics - 1):
        # 无模糊频率: 真实TDOA约2~3倍d_far, 取f < fs/(2*4*d_far)
        f_unamb = fs / (2 * 4 * d_far)
        freq_mask = (f > 30) & (f < f_unamb)
        # 收集所有无模糊时频单元的TDOA
        all_tdoa, all_w = [], []
        for fi in range(len(f)):
            if not freq_mask[fi]:
                continue
            ipd = np.angle(stfts[i + 1, fi] * np.conj(stfts[i, fi]))
            tr = -ipd / (2 * np.pi * f[fi]) * fs
            all_tdoa.append(tr)
            all_w.append(w[fi])
        all_tdoa = np.concatenate(all_tdoa)
        all_w = np.concatenate(all_w)
        # 加权65%分位数(目标TDOA为正且大于干扰, 高分位偏向目标)
        idx = np.argsort(all_tdoa)
        cumw = np.cumsum(all_w[idx]) / (all_w.sum() + 1e-10)
        k = np.searchsorted(cumw, 0.65)
        tdoa = float(all_tdoa[idx[min(k, len(idx) - 1)]])
        tdoa = max(tdoa, d_far)  # 下限保护
        pair_tdoas.append(tdoa)

    tdoas = np.zeros(num_mics)
    for i in range(1, num_mics):
        tdoas[i] = tdoas[i - 1] + pair_tdoas[i - 1]
    return tdoas, np.array(pair_tdoas)


# ==================== 核心: 延时求和 + IPD方向性掩码 ====================

def separate_target(data, fs=352800, num_mics=6, mic_spacing=0.008, c=343.0):
    """
    盲源分离主函数: 提取端射方向目标, 抑制侧面干扰.

    Args:
        data: (num_mics, num_samples) 多通道信号矩阵, 1号麦克风为参考(目标最近)
        fs: 采样率 (Hz)
        num_mics: 麦克风数
        mic_spacing: 麦克风间距 (m)
        c: 声速 (m/s)
    Returns:
        output: (num_samples,) 分离出的目标信号
    """
    n_samples = data.shape[1]

    # ---------- 1. 目标TDOA估计(能量加权IPD, 抗强干扰) ----------
    tdoas, pair_tdoa_target = estimate_target_tdoa(data, fs, num_mics, mic_spacing, c)
    # 相邻对最大无模糊TDOA(相位解包裹用)
    d_max = mic_spacing / c * fs

    # ---------- 2. STFT ----------
    nperseg = 4096
    noverlap = nperseg * 3 // 4
    stft_list = []
    for i in range(num_mics):
        f, t_stft, Zxx = stft(data[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        stft_list.append(Zxx)
    stft_list = np.array(stft_list)  # (mics, freqs, frames)
    n_freqs, n_frames = len(f), len(t_stft)

    # ---------- 3. 延时求和(目标方向对齐, 无失真) ----------
    # 目标从1号来, i号延迟 tdoas[i] 样本. 对齐需把i号"提前" tdoas[i],
    # 频域乘以 exp(+1j*2*pi*f*tdoa/fs) 补偿物理延迟, 使目标同相叠加.
    # 同时broadside干扰(TDOA≈0)被施加错误相位, 不同相而部分抵消.
    aligned = np.zeros((n_freqs, n_frames), dtype=complex)
    for i in range(num_mics):
        phase = np.exp(1j * 2 * np.pi * f[:, None] * tdoas[i] / fs)
        aligned += stft_list[i] * phase
    aligned /= num_mics

    # ---------- 4. IPD方向性掩码 ----------
    # 每个时频单元, 用相邻对瞬时相位差估计主导声源的TDOA.
    # 接近目标(端射, TDOA≈pair_tdoa_target)→掩码≈1;
    # 接近0(broadside侧面干扰)→掩码≈0.
    mask = np.ones((n_freqs, n_frames), dtype=float)
    # 判别宽度: 目标TDOA均值的0.22倍, 保证干扰方向(0)被强抑制
    sigma = max(np.mean(np.abs(pair_tdoa_target)) * 0.22, 3.0)
    # 无模糊频率上限: 目标TDOA必须 < 半个模糊周期 fs/(2f)
    # 即 f < fs / (2 * max_pair_tdoa), 高于此频率相位模糊无法区分方向
    f_limit = fs / (2 * np.max(np.abs(pair_tdoa_target)) + 1e-6)

    for fi in range(n_freqs):
        freq = f[fi]
        if freq < 20 or freq > fs // 2 - 20:
            continue  # 极低/极高频保留延时求和结果
        if freq > f_limit:
            continue  # 高频相位模糊, 无法可靠判别方向, 保留延时求和

        # 各相邻对瞬时TDOA(相位差法)
        period = fs / freq  # TDOA模糊周期(样本)
        tdoa_ests = []
        for i in range(num_mics - 1):
            ipd = np.angle(stft_list[i + 1, fi] * np.conj(stft_list[i, fi]))
            tdoa_raw = -ipd / (2 * np.pi * freq) * fs
            # 相位解包裹: 调整到目标模板附近(加模糊周期的整数倍)
            k = np.round((pair_tdoa_target[i] - tdoa_raw) / period)
            tdoa_fold = tdoa_raw + k * period
            tdoa_ests.append(tdoa_fold)
        tdoa_ests = np.array(tdoa_ests)  # (pairs, frames)

        # 各对偏差(相对目标模板), 几何平均成置信度
        dev = tdoa_ests - pair_tdoa_target[:, None]
        exp_dev = np.exp(-(dev ** 2) / (2 * sigma ** 2))  # (pairs, frames)
        mask[fi] = np.prod(exp_dev, axis=0) ** (1.0 / (num_mics - 1))

    # ---------- 5. 掩码时间平滑(减少音乐噪声/抖动) ----------
    mask = uniform_filter1d(mask, size=5, axis=1, mode='nearest')
    mask = np.clip(mask, 0.02, 1.0)  # 下限保护, 避免过度抑制

    # ---------- 6. 应用掩码 + ISTFT ----------
    output_stft = aligned * mask
    _, output = istft(output_stft, fs=fs, nperseg=nperseg, noverlap=noverlap)
    output = output[:n_samples]
    return output
