#!/usr/bin/env python3
"""
noise_hist.py

NPZ 内の 2D-array 全要素をヒストグラム化する。
複素配列は abs(array) を使用する。

生成画像:
  *_noise_hist_linear.png
  *_noise_hist_logx.png
  *_noise_hist_logy.png
  *_noise_hist_logxy.png

(delay, rate) = (0, 0) の画素値を縦の赤い破線で示す。\n凡例に全要素数、ピークを含むビンの度数、ピーク値以上の要素数を表示する。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create histograms of every value in a 2D array stored in NPZ."
    )
    parser.add_argument("input", type=Path, help="Input .npz file")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output base name. Default: INPUT_noise_hist",
    )
    parser.add_argument(
        "-k", "--key",
        default="delay_rate",
        help="Key of the 2D array in NPZ (default: delay_rate)",
    )
    parser.add_argument(
        "--delay-key",
        default="delay_sample",
        help="Key of delay coordinates (default: delay_sample)",
    )
    parser.add_argument(
        "--rate-key",
        default="rate_hz",
        help="Key of rate coordinates (default: rate_hz)",
    )
    parser.add_argument(
        "-b", "--bins",
        type=int,
        default=256,
        help="Number of histogram bins (default: 256)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution (default: 160)",
    )
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Generate only the linear-x/linear-y plot",
    )
    return parser.parse_args()


def load_values(
    path: Path,
    array_key: str,
    delay_key: str,
    rate_key: str,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    # mmap_mode does not accelerate arrays inside compressed NPZ; load once only.
    with np.load(path, allow_pickle=False) as npz:
        if array_key not in npz:
            available = ", ".join(npz.files)
            raise KeyError(
                f"2D-array key {array_key!r} not found. Available keys: {available}"
            )

        array = np.asarray(npz[array_key])
        if array.ndim != 2:
            raise ValueError(
                f"{array_key!r} must be a 2D array, but shape is {array.shape}"
            )

        # astype(copy=False) avoids a copy when already float32/float64.
        if np.iscomplexobj(array):
            values_2d = np.abs(array)
        else:
            values_2d = array

        # Remove NaN/Inf once. ravel() is normally a view.
        values = np.asarray(values_2d).ravel()
        finite = np.isfinite(values)
        if not finite.all():
            values = values[finite]

        if values.size == 0:
            raise ValueError("No finite values found in the 2D array")

        # Locate the exact delay=0 and rate=0 coordinates.
        if delay_key not in npz:
            raise KeyError(
                f"delay coordinate key {delay_key!r} not found in NPZ"
            )
        if rate_key not in npz:
            raise KeyError(
                f"rate coordinate key {rate_key!r} not found in NPZ"
            )

        delay = np.asarray(npz[delay_key]).ravel()
        rate = np.asarray(npz[rate_key]).ravel()

        delay_zero = np.flatnonzero(delay == 0)
        rate_zero = np.flatnonzero(rate == 0)

        if delay_zero.size == 0:
            raise ValueError(
                f"exact delay=0 was not found in {delay_key!r}"
            )
        if rate_zero.size == 0:
            raise ValueError(
                f"exact rate=0 was not found in {rate_key!r}"
            )
        if delay_zero.size > 1:
            raise ValueError(
                f"multiple exact delay=0 entries found in {delay_key!r}: "
                f"{delay_zero.tolist()}"
            )
        if rate_zero.size > 1:
            raise ValueError(
                f"multiple exact rate=0 entries found in {rate_key!r}: "
                f"{rate_zero.tolist()}"
            )

        delay_index = int(delay_zero[0])
        rate_index = int(rate_zero[0])

        if not (0 <= rate_index < array.shape[0]):
            raise IndexError(f"rate index {rate_index} is outside shape {array.shape}")
        if not (0 <= delay_index < array.shape[1]):
            raise IndexError(f"delay index {delay_index} is outside shape {array.shape}")

        zero_value = float(values_2d[rate_index, delay_index])

    return values, zero_value, (rate_index, delay_index)


def output_base(input_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        # Treat .png as a base name too, removing only the suffix.
        return requested.with_suffix("") if requested.suffix else requested
    return input_path.with_suffix("").with_name(input_path.stem + "_noise_hist")


def histogram_linear(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    return np.histogram(values, bins=bins)



def histogram_logx(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Histogram positive values using logarithmically spaced bin edges."""
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("Log-x histogram requires at least one positive value")

    vmin = float(positive.min())
    vmax = float(positive.max())

    if vmin == vmax:
        vmin *= 0.999
        vmax *= 1.001

    edges = np.geomspace(vmin, vmax, bins + 1)
    return np.histogram(positive, bins=edges)

def count_in_peak_bin(
    counts: np.ndarray,
    edges: np.ndarray,
    peak_value: float,
) -> int:
    """Return the histogram-bin count containing peak_value."""
    # np.histogram uses half-open bins [left, right), except the final bin.
    index = int(np.searchsorted(edges, peak_value, side="right") - 1)
    index = min(max(index, 0), counts.size - 1)
    return int(counts[index])

def draw_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    zero_value: float,
    total_count: int,
    peak_bin_count: int,
    count_ge_peak: int,
    output: Path,
    *,
    log_x: bool,
    log_y: bool,
    dpi: int,
) -> None:
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.stairs(counts, edges, fill=False, linewidth=1.0)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("2D-array value" if not log_x else "2D-array value (log scale)")
    ax.set_ylabel("Count" if not log_y else "Count (log scale)")
    ax.grid(True, which="both", alpha=0.25)

    # Mark the exact (delay, rate) = (0, 0) value.
    # A logarithmic x-axis cannot display zero or negative values.
    if zero_value > 0 or not log_x:
        legend_text = (
            f"Total count = {total_count:,}\n"
            f"Fringe-peak bin count = {peak_bin_count:,}\n"
            f"Count ≥ fringe peak = {count_ge_peak:,}\n"
            f"Fringe peak = {zero_value:.6g}"
        )
        ax.axvline(
            zero_value,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=legend_text,
        )
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)



def main() -> int:
    args = parse_args()
    start = time.perf_counter()

    if args.bins < 1:
        print("ERROR: --bins must be at least 1", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    try:
        values, zero_value, zero_index = load_values(
            args.input,
            args.key,
            args.delay_key,
            args.rate_key,
        )

        base = output_base(args.input, args.output)

        # Compute one linear-bin histogram for linear/log-y displays.
        linear_counts, linear_edges = histogram_linear(values, args.bins)

        total_count = int(values.size)
        count_ge_peak = int(np.count_nonzero(values >= zero_value))
        linear_peak_bin_count = count_in_peak_bin(
            linear_counts, linear_edges, zero_value
        )

        linear_path = Path(f"{base}_linear.png")
        draw_histogram(
            linear_counts,
            linear_edges,
            zero_value,
            total_count,
            linear_peak_bin_count,
            count_ge_peak,
            linear_path,
            log_x=False,
            log_y=False,
            dpi=args.dpi,
        )
        generated = [linear_path]

        if not args.linear_only:
            logy_path = Path(f"{base}_logy.png")
            draw_histogram(
                linear_counts,
                linear_edges,
                zero_value,
                total_count,
                linear_peak_bin_count,
                count_ge_peak,
                logy_path,
                log_x=False,
                log_y=True,
                dpi=args.dpi,
            )

            # For log-x displays, histogram positive values with
            # logarithmically spaced bin edges before plotting.
            log_counts, log_edges = histogram_logx(values, args.bins)
            log_peak_bin_count = count_in_peak_bin(
                log_counts, log_edges, zero_value
            )

            logx_path = Path(f"{base}_logx.png")
            draw_histogram(
                log_counts,
                log_edges,
                zero_value,
                total_count,
                log_peak_bin_count,
                count_ge_peak,
                logx_path,
                log_x=True,
                log_y=False,
                dpi=args.dpi,
            )

            logxy_path = Path(f"{base}_logxy.png")
            draw_histogram(
                log_counts,
                log_edges,
                zero_value,
                total_count,
                log_peak_bin_count,
                count_ge_peak,
                logxy_path,
                log_x=True,
                log_y=True,
                dpi=args.dpi,
            )
            generated.extend([logx_path, logy_path, logxy_path])

    except (OSError, KeyError, ValueError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"array elements : {values.size}")
    print(f"zero index     : rate={zero_index[0]}, delay={zero_index[1]}")
    print(f"zero value     : {zero_value:.12g}")
    print(f"peak bin count : {linear_peak_bin_count}")
    print(f"count >= peak  : {count_ge_peak}")
    for path in generated:
        print(f"created        : {path}")

    elapsed = time.perf_counter() - start
    print(f"elapsed        : {elapsed:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
