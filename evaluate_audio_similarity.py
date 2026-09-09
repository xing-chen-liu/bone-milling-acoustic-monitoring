#!/usr/bin/env python3
"""Evaluate TXT audio similarity using the six criteria in the supplied table."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("This script requires NumPy. Install it with: python -m pip install numpy") from exc

DEFAULT_SAMPLE_RATE = 352800
EPSILON = 1e-12

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate audio similarity using six metric groups.")
    parser.add_argument("reference", nargs="?", type=Path, default=Path("8.27/original_audio.txt"))
    parser.add_argument("candidate", nargs="?", type=Path, default=Path("8.27/output_audio_amplified_to_original_db.txt"))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--json", type=Path, default=None, help="Also write detailed results as JSON.")
    return parser.parse_args()

def read_audio(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read numeric samples from {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains no finite samples.")
    return values

def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal * signal)))

def db(value: float) -> float:
    return 20.0 * math.log10(max(value, EPSILON))

def align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    length = min(reference.size, candidate.size)
    reference, candidate = reference[:length], candidate[:length]
    size = 1 << (2 * length - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(reference, size) * np.conj(np.fft.rfft(candidate, size)), size)
    corr = np.concatenate((corr[-(length - 1):], corr[:length])) if length > 1 else corr
    lag = int(np.argmax(np.abs(corr)) - (length - 1))
    if lag > 0:
        return reference[lag:], candidate[:length - lag], lag
    if lag < 0:
        return reference[:length + lag], candidate[-lag:], lag
    return reference, candidate, 0

def correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = reference - np.mean(reference), candidate - np.mean(candidate)
    denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
    return float(np.dot(reference, candidate) / max(denominator, EPSILON))

def si_sdr(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = reference - np.mean(reference), candidate - np.mean(candidate)
    scale = float(np.dot(candidate, reference) / max(np.dot(reference, reference), EPSILON))
    target, noise = scale * reference, candidate - scale * reference
    return 10.0 * math.log10(max(np.dot(target, target), EPSILON) / max(np.dot(noise, noise), EPSILON))

def frame_signal(signal: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if signal.size < frame_size:
        signal = np.pad(signal, (0, frame_size - signal.size))
    count = 1 + (signal.size - frame_size) // hop
    indices = np.arange(frame_size)[None, :] + hop * np.arange(count)[:, None]
    return signal[indices] * np.hanning(frame_size)[None, :]

def spectral_features(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, float]:
    frames = frame_signal(signal, 4096, 2048)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) + EPSILON
    frequencies = np.fft.rfftfreq(4096, 1.0 / sample_rate)
    average = np.mean(spectrum, axis=0)
    return average, frequencies, float(frequencies[np.argmax(average[1:]) + 1])

def stft_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.abs(np.fft.rfft(frame_signal(reference, 2048, 1024), axis=1))
    can = np.abs(np.fft.rfft(frame_signal(candidate, 2048, 1024), axis=1))
    count = min(ref.shape[0], can.shape[0])
    ref, can = ref[:count], can[:count]
    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + EPSILON
    can /= np.linalg.norm(can, axis=1, keepdims=True) + EPSILON
    return float(np.mean(np.sum(ref * can, axis=1)))

def mfcc(signal: np.ndarray, sample_rate: int, count: int = 13) -> np.ndarray:
    frames = frame_signal(signal, 2048, 1024)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(2048, 1.0 / sample_rate)
    mel_low = 2595 * math.log10(1 + 20 / 700)
    mel_high = 2595 * math.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(mel_low, mel_high, 28)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.searchsorted(frequencies, hz_points).clip(0, power.shape[1] - 1)
    filters = np.zeros((26, power.shape[1]))
    for index in range(26):
        left, center, right = bins[index:index + 3]
        if center > left:
            filters[index - 1, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if right > center:
            filters[index - 1, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    log_energy = np.log(np.maximum(power @ filters.T, EPSILON))
    n, k = np.arange(count)[:, None], np.arange(26)[None, :]
    return (log_energy @ np.cos(math.pi * (k + 0.5) * n / 26).T).mean(axis=0)

def normalized_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.dot(reference / max(np.linalg.norm(reference), EPSILON), candidate / max(np.linalg.norm(candidate), EPSILON)))

def evaluate(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> dict[str, object]:
    reference, candidate, lag = align(reference, candidate)
    error = candidate - reference
    snr = 10 * math.log10(max(np.sum(reference * reference), EPSILON) / max(np.sum(error * error), EPSILON))
    ref_spectrum, _, ref_peak = spectral_features(reference, sample_rate)
    can_spectrum, _, can_peak = spectral_features(candidate, sample_rate)
    frequency_similarity = correlation(np.log(ref_spectrum), np.log(can_spectrum))
    waveform = {"pearson": correlation(reference, candidate), "ncc": normalized_similarity(reference, candidate), "rmse": float(np.sqrt(np.mean(error * error)))}
    metrics: dict[str, object] = {
        "time": {"lag_samples": lag, "lag_ms": lag * 1000 / sample_rate, "pass": abs(lag) <= 1},
        "waveform": waveform,
        "distortion": {"snr_db": snr, "si_sdr_db": si_sdr(reference, candidate)},
        "frequency": {"spectrum_correlation": frequency_similarity, "reference_peak_hz": ref_peak, "candidate_peak_hz": can_peak, "peak_error_hz": abs(ref_peak - can_peak)},
        "time_frequency": {"stft_similarity": stft_similarity(reference, candidate)},
        "perceptual": {"mfcc_similarity": normalized_similarity(mfcc(reference, sample_rate), mfcc(candidate, sample_rate))},
        "samples_compared": int(reference.size), "reference_rms_db": db(rms(reference)), "candidate_rms_db": db(rms(candidate)),
    }
    waveform["pass"] = waveform["pearson"] >= 0.995 and waveform["ncc"] >= 0.995
    metrics["distortion"]["pass"] = snr >= 40 and metrics["distortion"]["si_sdr_db"] >= 35
    metrics["frequency"]["pass"] = frequency_similarity >= 0.995
    metrics["time_frequency"]["pass"] = metrics["time_frequency"]["stft_similarity"] >= 0.990
    metrics["perceptual"]["pass"] = metrics["perceptual"]["mfcc_similarity"] >= 0.990
    metrics["pass"] = all(group["pass"] for group in metrics.values() if isinstance(group, dict) and "pass" in group)
    return metrics

def main() -> None:
    args = parse_args()
    result = evaluate(read_audio(args.reference), read_audio(args.candidate), args.sample_rate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n最终判定: {'PASS（高度一致）' if result['pass'] else 'FAIL（未达到全部标准）'}")
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入: {args.json}")

if __name__ == "__main__":
    main()
