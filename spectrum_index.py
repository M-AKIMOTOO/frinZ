#!/usr/bin/env python3
"""Combine relative-calibrated spectra and fit a spectral index."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List


def read_spectrum(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        required = ["band", "center_mhz", "flux_mjy", "flux_err_mjy"]
        for name in required:
            if name not in index:
                raise ValueError(f"{path} does not contain column {name}")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            flux = float(cols[index["flux_mjy"]])
            if flux <= 0.0 or not math.isfinite(flux):
                continue
            flux_err = float(cols[index["flux_err_mjy"]])
            rows.append(
                {
                    "input": path.name,
                    "band": int(cols[index["band"]]),
                    "center_mhz": float(cols[index["center_mhz"]]),
                    "flux_mjy": flux,
                    "flux_err_mjy": flux_err,
                }
            )
    if not rows:
        raise ValueError(f"no valid spectrum rows found: {path}")
    return rows


def parse_overlay_point(text: str) -> dict:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError(
            "--overlay-point format is freq_mhz,flux_mjy[,flux_err_mjy[,label]]"
        )
    freq = float(parts[0])
    flux = float(parts[1])
    err = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0
    label = parts[3] if len(parts) >= 4 and parts[3] else "overlay"
    if freq <= 0.0 or flux <= 0.0:
        raise ValueError("overlay frequency and flux must be positive")
    return {
        "label": label,
        "center_mhz": freq,
        "flux_mjy": flux,
        "flux_err_mjy": err,
    }


def weighted_linear_fit(rows: List[dict], weighted: bool) -> dict:
    xs = [math.log(r["center_mhz"]) for r in rows]
    ys = [math.log(r["flux_mjy"]) for r in rows]
    if weighted:
        ws = []
        for r in rows:
            if r["flux_err_mjy"] > 0.0 and math.isfinite(r["flux_err_mjy"]):
                sigma_log = r["flux_err_mjy"] / r["flux_mjy"]
                ws.append(1.0 / (sigma_log * sigma_log))
            else:
                ws.append(1.0)
    else:
        ws = [1.0] * len(rows)

    sw = sum(ws)
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    den = sw * sxx - sx * sx
    if den == 0.0:
        raise ValueError("cannot fit spectral index: singular design matrix")

    alpha = (sw * sxy - sx * sy) / den
    intercept = (sxx * sy - sx * sxy) / den

    residuals = [y - (alpha * x + intercept) for x, y in zip(xs, ys)]
    chi2 = sum(w * e * e for w, e in zip(ws, residuals))
    dof = max(0, len(rows) - 2)
    scale = chi2 / dof if dof > 0 else 0.0
    alpha_err = math.sqrt(sw / den * scale) if dof > 0 else float("nan")
    intercept_err = math.sqrt(sxx / den * scale) if dof > 0 else float("nan")

    return {
        "alpha": alpha,
        "alpha_err": alpha_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
        "chi2": chi2,
        "dof": dof,
    }


def write_combined(rows: List[dict], output: Path) -> None:
    with output.open("w", encoding="utf-8") as fh:
        fh.write("input\tband\tcenter_mhz\tflux_mjy\tflux_err_mjy\n")
        for r in rows:
            fh.write(
                f"{r['input']}\t{r['band']}\t{r['center_mhz']:.6f}\t"
                f"{r['flux_mjy']:.9g}\t{r['flux_err_mjy']:.9g}\n"
            )


def write_overlay_points(points: List[dict], output: Path) -> None:
    if not points:
        return
    with output.open("w", encoding="utf-8") as fh:
        fh.write("label\tcenter_mhz\tflux_mjy\tflux_err_mjy\n")
        for p in points:
            fh.write(
                f"{p['label']}\t{p['center_mhz']:.6f}\t"
                f"{p['flux_mjy']:.9g}\t{p['flux_err_mjy']:.9g}\n"
            )


def write_fit_summary(
    fit: dict, output: Path, rows: List[dict], weighted: bool, overlay_points: List[dict]
) -> None:
    with output.open("w", encoding="utf-8") as fh:
        fh.write("# Spectral index fit: S_mJy = exp(intercept) * frequency_MHz^alpha\n")
        fh.write("# Overlay points are plotted only and are not included in the fit.\n")
        fh.write(f"weighted\t{int(weighted)}\n")
        fh.write(f"n\t{len(rows)}\n")
        fh.write(f"overlay_n\t{len(overlay_points)}\n")
        fh.write(f"freq_min_mhz\t{min(r['center_mhz'] for r in rows):.6f}\n")
        fh.write(f"freq_max_mhz\t{max(r['center_mhz'] for r in rows):.6f}\n")
        fh.write(f"alpha\t{fit['alpha']:.9g}\n")
        fh.write(f"alpha_err\t{fit['alpha_err']:.9g}\n")
        fh.write(f"intercept\t{fit['intercept']:.9g}\n")
        fh.write(f"intercept_err\t{fit['intercept_err']:.9g}\n")
        fh.write(f"chi2\t{fit['chi2']:.9g}\n")
        fh.write(f"dof\t{fit['dof']}\n")


def write_plot(
    rows: List[dict], fit: dict, overlay_points: List[dict], output: Path, show: bool
) -> None:
    import matplotlib.pyplot as plt

    freq = [r["center_mhz"] for r in rows]
    flux = [r["flux_mjy"] for r in rows]
    err = [r["flux_err_mjy"] for r in rows]
    all_freq = freq + [p["center_mhz"] for p in overlay_points]
    fmin, fmax = min(all_freq), max(all_freq)
    grid = [fmin + (fmax - fmin) * i / 200 for i in range(201)]
    model = [math.exp(fit["intercept"]) * (f ** fit["alpha"]) for f in grid]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(freq, flux, yerr=err, fmt="o", capsize=3, label="32 MHz in-band")
    if overlay_points:
        overlay_freq = [p["center_mhz"] for p in overlay_points]
        overlay_flux = [p["flux_mjy"] for p in overlay_points]
        overlay_err = [p["flux_err_mjy"] for p in overlay_points]
        ax.errorbar(
            overlay_freq,
            overlay_flux,
            yerr=overlay_err,
            fmt="s",
            ms=7,
            capsize=4,
            color="black",
            label="512 MHz full-band",
        )
        for p in overlay_points:
            ax.annotate(
                p["label"],
                (p["center_mhz"], p["flux_mjy"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.plot(grid, model, "-", label=f"in-band fit: alpha = {fit['alpha']:.3f}")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Flux density (mJy)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine spectra and fit spectral index S ∝ ν^alpha."
    )
    parser.add_argument("spectra", nargs="+", type=Path, help="*_relcal_spectrum.tsv files")
    parser.add_argument("--output", type=Path, help="Combined TSV path")
    parser.add_argument("--ext", default="png", help="Plot extension [default: png]")
    parser.add_argument(
        "--overlay-point",
        action="append",
        default=[],
        metavar="FREQ,FLUX[,ERR[,LABEL]]",
        help="Overlay a point on the plot without using it in the fit. Repeatable.",
    )
    parser.add_argument(
        "--unweighted", action="store_true", help="Use unweighted fit instead of flux-error weighting"
    )
    parser.add_argument("--nofig", action="store_true", help="Do not show the figure interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[dict] = []
    for path in args.spectra:
        rows.extend(read_spectrum(path))
    rows.sort(key=lambda r: r["center_mhz"])
    overlay_points = [parse_overlay_point(text) for text in args.overlay_point]
    overlay_points.sort(key=lambda r: r["center_mhz"])

    output = args.output or args.spectra[0].with_name("combined_relcal_spectrum.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fit = weighted_linear_fit(rows, weighted=not args.unweighted)
    write_combined(rows, output)

    summary_path = output.with_name(f"{output.stem}_spectral_index.txt")
    write_fit_summary(fit, summary_path, rows, weighted=not args.unweighted, overlay_points=overlay_points)

    if overlay_points:
        overlay_path = output.with_name(f"{output.stem}_overlay_points.tsv")
        write_overlay_points(overlay_points, overlay_path)
        print(f"Overlay points saved to: {overlay_path}")

    plot_path = output.with_suffix(f".{args.ext.lstrip('.')}")
    write_plot(rows, fit, overlay_points, plot_path, show=not args.nofig)

    print(f"Combined spectrum saved to: {output}")
    print(f"Spectral index summary saved to: {summary_path}")
    print(f"Spectrum plot saved to: {plot_path}")
    print(f"alpha = {fit['alpha']:.6f} +/- {fit['alpha_err']:.6f}")


if __name__ == "__main__":
    main()
