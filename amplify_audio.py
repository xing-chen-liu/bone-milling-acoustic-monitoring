#!/usr/bin/env python3
"""
Amplify a mono text audio file and write both TXT samples and a WAV file.

The input TXT is expected to contain one floating-point sample per line. Samples
are treated as normalized audio values where -1.0 and 1.0 map to 16-bit WAV
minimum/maximum.
"""

from __future__ import annotations

import argparse
import math
import wave
from array import array
from pathlib import Path


INT16_MAX = 32767
INT16_MIN = -32768
DEFAULT_SAMPLE_RATE = 352800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Amplify mono TXT audio and export amplified TXT plus WAV."
    )
    parser.add_argument(
        "input_txt",
        type=Path,
        help="Input TXT file containing one floating-point audio sample per line.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference TXT audio. If supplied, auto gain matches input RMS dB to this file.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Manual gain multiplier. Used alone, or multiplied with the auto reference gain.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"WAV sample rate in Hz. Default: {DEFAULT_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Amplified TXT output path. Default: <input>_amplified.txt.",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=None,
        help="Amplified WAV output path. Default: <input>_amplified.wav.",
    )
    parser.add_argument(
        "--txt-precision",
        type=int,
        default=10,
        help="Decimal places written to the output TXT. Default: 10.",
    )
    return parser.parse_args()


def read_sample(text: str, path: Path, line_number: int) -> float:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{path} line {line_number} is empty.")
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{path} line {line_number} is not a number: {stripped!r}") from exc


def measure_txt_audio(path: Path) -> tuple[int, float, float, float]:
    count = 0
    sum_squares = 0.0
    peak = 0.0

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            sample = read_sample(line, path, line_number)
            count += 1
            sum_squares += sample * sample
            peak = max(peak, abs(sample))

    if count == 0:
        raise ValueError(f"{path} contains no samples.")

    rms = math.sqrt(sum_squares / count)
    db = 20.0 * math.log10(rms) if rms > 0.0 else float("-inf")
    return count, rms, db, peak


def clamp_int16(value: float) -> int:
    sample = round(value * INT16_MAX)
    if sample > INT16_MAX:
        return INT16_MAX
    if sample < INT16_MIN:
        return INT16_MIN
    return sample


def amplify_txt_to_outputs(
    input_txt: Path,
    output_txt: Path,
    output_wav: Path,
    gain: float,
    sample_rate: int,
    txt_precision: int,
) -> tuple[int, float]:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    peak_after = 0.0
    fmt = f"{{:.{txt_precision}f}}\n"

    with input_txt.open("r", encoding="utf-8-sig") as in_file, output_txt.open(
        "w", encoding="utf-8", newline=""
    ) as txt_file, wave.open(str(output_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frame_buffer = array("h")
        for line_number, line in enumerate(in_file, start=1):
            amplified = read_sample(line, input_txt, line_number) * gain
            txt_file.write(fmt.format(amplified))
            frame_buffer.append(clamp_int16(amplified))
            peak_after = max(peak_after, abs(amplified))
            count += 1

            if len(frame_buffer) >= 8192:
                wav_file.writeframesraw(frame_buffer.tobytes())
                del frame_buffer[:]

        if frame_buffer:
            wav_file.writeframesraw(frame_buffer.tobytes())

    return count, peak_after


def default_output_paths(input_txt: Path) -> tuple[Path, Path]:
    stem = input_txt.stem
    parent = input_txt.parent
    return parent / f"{stem}_amplified.txt", parent / f"{stem}_amplified.wav"


def main() -> None:
    args = parse_args()
    input_txt = args.input_txt
    default_txt, default_wav = default_output_paths(input_txt)
    output_txt = args.output_txt or default_txt
    output_wav = args.output_wav or default_wav

    input_count, input_rms, input_db, input_peak = measure_txt_audio(input_txt)

    auto_gain = 1.0
    reference_db = None
    if args.reference is not None:
        reference_count, reference_rms, reference_db, reference_peak = measure_txt_audio(args.reference)
        if input_rms == 0.0:
            raise ValueError("Cannot match reference dB because input RMS is zero.")
        auto_gain = reference_rms / input_rms
        print(f"Reference samples: {reference_count}")
        print(f"Reference RMS dB: {reference_db:.6f}")
        print(f"Reference peak: {reference_peak:.10f}")

    final_gain = auto_gain * args.gain
    sample_count, peak_after = amplify_txt_to_outputs(
        input_txt=input_txt,
        output_txt=output_txt,
        output_wav=output_wav,
        gain=final_gain,
        sample_rate=args.sample_rate,
        txt_precision=args.txt_precision,
    )
    _, output_rms, output_db, _ = measure_txt_audio(output_txt)

    print(f"Input samples: {input_count}")
    print(f"Input RMS dB: {input_db:.6f}")
    print(f"Input peak: {input_peak:.10f}")
    print(f"Auto gain: {auto_gain:.10f}")
    print(f"Manual gain: {args.gain:.10f}")
    print(f"Final gain: {final_gain:.10f}")
    print(f"Output samples: {sample_count}")
    print(f"Output RMS dB: {output_db:.6f}")
    if reference_db is not None:
        print(f"Output-reference dB error: {output_db - reference_db:.10f}")
    print(f"Output peak: {peak_after:.10f}")
    print(f"Wrote TXT: {output_txt}")
    print(f"Wrote WAV: {output_wav}")


if __name__ == "__main__":
    main()
