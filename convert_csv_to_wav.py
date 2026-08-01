#!/usr/bin/env python3
"""
Convert separated source CSV files to WAV audio files.

Each separated CSV has six columns, one column per microphone. By default this script writes:
    <name>_6ch.wav   six-channel WAV, preserving all microphone channels
    <name>_mono.wav  mono WAV, averaging the six microphone channels for listening

The input CSV files are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import wave
from array import array
from pathlib import Path


DEFAULT_INPUTS = ("source_1_1to6.csv", "source_2_6to1.csv")
INT16_MAX = 32767
INT16_MIN = -32768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert separated six-channel source CSV files to WAV audio."
    )
    parser.add_argument(
        "--input-dir",
        default=Path("8.1") / "separated",
        type=Path,
        help="Directory containing separated source CSV files.",
    )
    parser.add_argument(
        "--sample-rate",
        default=44100,
        type=int,
        help="WAV sample rate in Hz. Default: 44100.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Scale each WAV so its peak reaches full 16-bit range.",
    )
    parser.add_argument(
        "--output-kind",
        default="both",
        choices=("both", "mono", "6ch"),
        help="Choose which WAV files to write. Default: both.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Optional suffix added before .wav, for example _352800Hz.",
    )
    return parser.parse_args()


def clamp_int16(value: float) -> int:
    sample = round(value * INT16_MAX)
    if sample > INT16_MAX:
        return INT16_MAX
    if sample < INT16_MIN:
        return INT16_MIN
    return sample


def read_peak(csv_path: Path) -> float:
    peak = 0.0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 6:
                raise ValueError(f"{csv_path} row {row_number} has {len(row)} columns, expected 6.")
            peak = max(peak, *(abs(float(cell)) for cell in row))
    return peak


def write_wavs(
    csv_path: Path,
    sample_rate: int,
    normalize: bool,
    output_kind: str,
    name_suffix: str,
) -> tuple[list[Path], int, float]:
    peak = read_peak(csv_path)
    gain = (1.0 / peak) if normalize and peak > 0 else 1.0

    write_6ch = output_kind in ("both", "6ch")
    write_mono = output_kind in ("both", "mono")
    written_paths: list[Path] = []

    six_channel_path = csv_path.with_name(f"{csv_path.stem}_6ch{name_suffix}.wav")
    mono_path = csv_path.with_name(f"{csv_path.stem}_mono{name_suffix}.wav")
    frame_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        six_channel_wav = wave.open(str(six_channel_path), "wb") if write_6ch else None
        mono_wav = wave.open(str(mono_path), "wb") if write_mono else None
        try:
            if six_channel_wav:
                six_channel_wav.setnchannels(6)
                six_channel_wav.setsampwidth(2)
                six_channel_wav.setframerate(sample_rate)
                written_paths.append(six_channel_path)

            if mono_wav:
                mono_wav.setnchannels(1)
                mono_wav.setsampwidth(2)
                mono_wav.setframerate(sample_rate)
                written_paths.append(mono_path)

            reader = csv.reader(csv_file)
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 6:
                    raise ValueError(f"{csv_path} row {row_number} has {len(row)} columns, expected 6.")

                channels = [float(cell) * gain for cell in row]

                if six_channel_wav:
                    six_channel_samples = array("h", (clamp_int16(value) for value in channels))
                    six_channel_wav.writeframesraw(six_channel_samples.tobytes())

                if mono_wav:
                    mono_sample = array("h", [clamp_int16(sum(channels) / len(channels))])
                    mono_wav.writeframesraw(mono_sample.tobytes())

                frame_count += 1
        finally:
            if six_channel_wav:
                six_channel_wav.close()
            if mono_wav:
                mono_wav.close()

    return written_paths, frame_count, peak


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir

    for name in DEFAULT_INPUTS:
        csv_path = input_dir / name
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing separated CSV: {csv_path}")

        written_paths, frame_count, peak = write_wavs(
            csv_path=csv_path,
            sample_rate=args.sample_rate,
            normalize=args.normalize,
            output_kind=args.output_kind,
            name_suffix=args.name_suffix,
        )
        duration = frame_count / args.sample_rate
        for path in written_paths:
            print(f"Wrote {path}")
        print(f"Frames: {frame_count}, duration: {duration:.3f}s, input peak: {peak:.6g}")


if __name__ == "__main__":
    main()
