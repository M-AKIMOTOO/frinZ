#!/usr/bin/env python3
"""Relative flux calibration from frinZ --inband text outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from inband_txt_to_json import parse_inband_txt
from typing import Dict, List


def read_inband(path: Path) -> list[dict]:
    _metadata, rows = parse_inband_txt(path)
    return rows

def build_spectrum(target: Dict[int, dict], calibrator: Dict[int, dict], cal_flux_mjy: float) -> List[dict]:
    bands = sorted(set(target) & set(calibrator))
    if not bands:
        raise ValueError("target and calibrator have no common band numbers")

    out = []
    for band in bands:
        t = target[band]
        c = calibrator[band]
        if c["amp_percent"] == 0.0:
            flux = float("nan")
            flux_err = float("nan")
        else:
            flux = t["amp_percent"] / c["amp_percent"] * cal_flux_mjy
            frac_err = 0.0
            if t["snr"] > 0.0:
                frac_err += (1.0 / t["snr"]) ** 2
            if c["snr"] > 0.0:
                frac_err += (1.0 / c["snr"]) ** 2
            flux_err = flux * (frac_err**0.5)
        out.append(
            {
                "band": band,
                "center_mhz": t["center_mhz"],
                "band_start_mhz": t["band_start_mhz"],
                "band_end_mhz": t["band_end_mhz"],
                "target_source": t["source"],
                "calibrator_source": c["source"],
                "target_amp_percent": t["amp_percent"],
                "calibrator_amp_percent": c["amp_percent"],
                "target_snr": t["snr"],
                "calibrator_snr": c["snr"],
                "target_phase_deg": t["phase_deg"],
                "calibrator_phase_deg": c["phase_deg"],
                "flux_mjy": flux,
                "flux_err_mjy": flux_err,
            }
        )
    return out


def default_output_path(target_path: Path) -> Path:
    stem = target_path.name
    if stem.endswith("_inband.txt"):
        stem = stem[: -len("_inband.txt")]
    else:
        stem = target_path.stem
    return target_path.with_name(f"{stem}_relcal_spectrum.tsv")


def write_tsv(rows: List[dict], path: Path) -> None:
    header = [
        "band",
        "center_mhz",
        "band_start_mhz",
        "band_end_mhz",
        "target_source",
        "calibrator_source",
        "target_amp_percent",
        "calibrator_amp_percent",
        "target_snr",
        "calibrator_snr",
        "target_phase_deg",
        "calibrator_phase_deg",
        "flux_mjy",
        "flux_err_mjy",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write(
                "\t".join(
                    [
                        f"{row['band']}",
                        f"{row['center_mhz']:.6f}",
                        f"{row['band_start_mhz']:.6f}",
                        f"{row['band_end_mhz']:.6f}",
                        row["target_source"],
                        row["calibrator_source"],
                        f"{row['target_amp_percent']:.9g}",
                        f"{row['calibrator_amp_percent']:.9g}",
                        f"{row['target_snr']:.9g}",
                        f"{row['calibrator_snr']:.9g}",
                        f"{row['target_phase_deg']:.9g}",
                        f"{row['calibrator_phase_deg']:.9g}",
                        f"{row['flux_mjy']:.9g}",
                        f"{row['flux_err_mjy']:.9g}",
                    ]
                )
                + "\n"
            )


def write_plot(rows: List[dict], path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    freq = [r["center_mhz"] for r in rows]
    flux = [r["flux_mjy"] for r in rows]
    err = [r["flux_err_mjy"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(freq, flux, yerr=err, fmt="o-", capsize=3, lw=1.4)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Flux density (mJy)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a relative-calibrated spectrum from two frinZ --inband text files."
    )
    parser.add_argument("--target", required=True, type=Path, help="Target *_inband.txt file")
    parser.add_argument("--calibrator", required=True, type=Path, help="Calibrator *_inband.txt file")
    parser.add_argument(
        "--cal-flux-mjy",
        required=True,
        type=float,
        help="Calibrator flux density in mJy, e.g. 100 for 2016+386",
    )
    parser.add_argument("--output", type=Path, help="Output TSV path. Default: target stem + _relcal_spectrum.tsv")
    parser.add_argument("--ext", default="png", help="Plot extension: png, pdf, ps, eps, jpg, ... [default: png]")
    parser.add_argument("--nofig", action="store_true", help="Do not show the figure interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = read_inband(args.target)
    calibrator = read_inband(args.calibrator)
    rows = build_spectrum(target, calibrator, args.cal_flux_mjy)

    output = args.output or default_output_path(args.target)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(rows, output)

    plot_path = output.with_suffix(f".{args.ext.lstrip('.')}")
    write_plot(rows, plot_path, show=not args.nofig)

    print(f"Relative calibrated spectrum saved to: {output}")
    print(f"Spectrum plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
