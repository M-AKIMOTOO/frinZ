#!/usr/bin/env python3
"""Time-resolved flux spectra from frinZ --inband text outputs."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from inband_txt_to_json import parse_inband_txt
from typing import Dict, Iterable, List, Tuple


def read_inband(path: Path) -> list[dict]:
    _metadata, rows = parse_inband_txt(path)
    return rows

def weighted_mean(values: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    sw = 0.0
    sx = 0.0
    for value, sigma in values:
        if sigma > 0.0 and math.isfinite(sigma):
            w = 1.0 / (sigma * sigma)
        else:
            w = 1.0
        sw += w
        sx += w * value
    if sw <= 0.0:
        return float("nan"), float("nan")
    mean = sx / sw
    return mean, math.sqrt(1.0 / sw)


def calibrator_by_band(rows: List[dict]) -> Dict[int, dict]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["band"]].append(row)
    out = {}
    for band, band_rows in grouped.items():
        amp_mean, amp_err = weighted_mean(
            (
                r["amp_percent"],
                abs(r["amp_percent"] / r["snr"]) if r["snr"] > 0.0 else 0.0,
            )
            for r in band_rows
        )
        first = band_rows[0]
        out[band] = {
            "band": band,
            "amp_percent": amp_mean,
            "amp_err_percent": amp_err,
            "center_mhz": first["center_mhz"],
            "n": len(band_rows),
            "source": first["source"],
        }
    return out


def make_flux_rows(
    target_rows: List[dict],
    cal_rows: List[dict],
    cal_flux_mjy: float,
    band_label: str,
    snr_min: float,
    exclude_bands: set[int],
) -> List[dict]:
    cal = calibrator_by_band(cal_rows)
    out = []
    for row in target_rows:
        band = row["band"]
        if band in exclude_bands or row["snr"] < snr_min or band not in cal:
            continue
        cal_amp = cal[band]["amp_percent"]
        if cal_amp <= 0.0:
            continue
        flux = row["amp_percent"] / cal_amp * cal_flux_mjy
        target_amp_err = abs(row["amp_percent"] / row["snr"]) if row["snr"] > 0.0 else 0.0
        cal_amp_err = cal[band]["amp_err_percent"]
        frac_var = 0.0
        if target_amp_err > 0.0:
            frac_var += (target_amp_err / row["amp_percent"]) ** 2
        if cal_amp_err > 0.0:
            frac_var += (cal_amp_err / cal_amp) ** 2
        flux_err = flux * math.sqrt(frac_var) if frac_var > 0.0 else 0.0
        out.append(
            {
                "epoch": row["epoch"],
                "mjd": row["mjd"],
                "band_label": band_label,
                "band": band,
                "center_mhz": row["center_mhz"],
                "target_source": row["source"],
                "calibrator_source": cal[band]["source"],
                "target_amp_percent": row["amp_percent"],
                "cal_amp_percent": cal_amp,
                "target_snr": row["snr"],
                "flux_mjy": flux,
                "flux_err_mjy": flux_err,
            }
        )
    return out


def write_channel_tsv(rows: List[dict], output: Path) -> None:
    header = [
        "epoch",
        "mjd",
        "band_label",
        "band",
        "center_mhz",
        "target_source",
        "calibrator_source",
        "target_amp_percent",
        "cal_amp_percent",
        "target_snr",
        "flux_mjy",
        "flux_err_mjy",
    ]
    with output.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write(
                "\t".join(
                    [
                        r["epoch"],
                        f"{r['mjd']:.8f}",
                        r["band_label"],
                        str(r["band"]),
                        f"{r['center_mhz']:.6f}",
                        r["target_source"],
                        r["calibrator_source"],
                        f"{r['target_amp_percent']:.9g}",
                        f"{r['cal_amp_percent']:.9g}",
                        f"{r['target_snr']:.9g}",
                        f"{r['flux_mjy']:.9g}",
                        f"{r['flux_err_mjy']:.9g}",
                    ]
                )
                + "\n"
            )


def group_by_time(rows: List[dict]) -> Dict[float, List[dict]]:
    grouped: Dict[float, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["mjd"]].append(row)
    return grouped


def make_lightcurve(rows: List[dict]) -> List[dict]:
    out = []
    for mjd, group in sorted(group_by_time(rows).items()):
        mean, err = weighted_mean((r["flux_mjy"], r["flux_err_mjy"]) for r in group)
        first = group[0]
        out.append(
            {
                "epoch": first["epoch"],
                "mjd": mjd,
                "flux_mjy": mean,
                "flux_err_mjy": err,
                "nchan": len(group),
            }
        )
    return out


def fit_alpha(group: List[dict]) -> Tuple[float, float, int]:
    valid = [r for r in group if r["flux_mjy"] > 0.0 and r["center_mhz"] > 0.0]
    if len(valid) < 2:
        return float("nan"), float("nan"), len(valid)
    xs = [math.log(r["center_mhz"]) for r in valid]
    ys = [math.log(r["flux_mjy"]) for r in valid]
    ws = []
    for r in valid:
        if r["flux_err_mjy"] > 0.0:
            sigma_log = r["flux_err_mjy"] / r["flux_mjy"]
            ws.append(1.0 / (sigma_log * sigma_log))
        else:
            ws.append(1.0)
    sw = sum(ws)
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    den = sw * sxx - sx * sx
    if den <= 0.0:
        return float("nan"), float("nan"), len(valid)
    alpha = (sw * sxy - sx * sy) / den
    intercept = (sxx * sy - sx * sxy) / den
    residuals = [y - (alpha * x + intercept) for x, y in zip(xs, ys)]
    dof = max(0, len(valid) - 2)
    chi2 = sum(w * e * e for w, e in zip(ws, residuals))
    scale = chi2 / dof if dof > 0 else 0.0
    err = math.sqrt(sw / den * scale) if dof > 0 else float("nan")
    return alpha, err, len(valid)


def make_alpha_series(rows: List[dict]) -> List[dict]:
    out = []
    for mjd, group in sorted(group_by_time(rows).items()):
        labels = {r["band_label"] for r in group}
        if len(labels) < 2:
            continue
        alpha, err, nchan = fit_alpha(group)
        out.append(
            {
                "epoch": group[0]["epoch"],
                "mjd": mjd,
                "alpha": alpha,
                "alpha_err": err,
                "nchan": nchan,
            }
        )
    return out


def write_simple_tsv(rows: List[dict], output: Path, keys: List[str]) -> None:
    with output.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            vals = []
            for key in keys:
                v = r[key]
                vals.append(f"{v:.9g}" if isinstance(v, float) else str(v))
            fh.write("\t".join(vals) + "\n")


def minutes_from_start(rows: List[dict]) -> List[float]:
    if not rows:
        return []
    start = min(r["mjd"] for r in rows)
    return [(r["mjd"] - start) * 86400.0 / 60.0 for r in rows]


def plot_lightcurve(rows: List[dict], output: Path, ext: str, show: bool, label: str) -> None:
    import matplotlib.pyplot as plt

    t_lc = minutes_from_start(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.errorbar(
        t_lc,
        [r["flux_mjy"] for r in rows],
        yerr=[r["flux_err_mjy"] for r in rows],
        fmt="o-",
        ms=3,
        lw=1,
        capsize=2,
        label=label,
    )
    ax.set_xlabel("Time from start (min)")
    ax.set_ylabel("Flux density (mJy)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(f".{ext}"))
    if show:
        plt.show()
    plt.close(fig)


def plot_combined_lightcurve(
    combined: List[dict], c_lc: List[dict], x_lc: List[dict], output: Path, ext: str, show: bool
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for rows, label, marker in [
        (combined, "C+X weighted mean", "o"),
        (c_lc, "C", "s"),
        (x_lc, "X", "^"),
    ]:
        if not rows:
            continue
        t = minutes_from_start(rows)
        ax.errorbar(
            t,
            [r["flux_mjy"] for r in rows],
            yerr=[r["flux_err_mjy"] for r in rows],
            fmt=f"{marker}-",
            ms=3,
            lw=1,
            capsize=2,
            label=label,
        )
    ax.set_xlabel("Time from start (min)")
    ax.set_ylabel("Flux density (mJy)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(f".{ext}"))
    if show:
        plt.show()
    plt.close(fig)


def plot_outputs(
    channel_rows: List[dict],
    lightcurve: List[dict],
    c_lightcurve: List[dict],
    x_lightcurve: List[dict],
    alpha_rows: List[dict],
    output_prefix: Path,
    ext: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    # Light curves
    plot_combined_lightcurve(
        lightcurve,
        c_lightcurve,
        x_lightcurve,
        output_prefix.with_name(f"{output_prefix.name}_lightcurve"),
        ext,
        show,
    )
    plot_lightcurve(
        c_lightcurve,
        output_prefix.with_name(f"{output_prefix.name}_C_lightcurve"),
        ext,
        show,
        "C",
    )
    plot_lightcurve(
        x_lightcurve,
        output_prefix.with_name(f"{output_prefix.name}_X_lightcurve"),
        ext,
        show,
        "X",
    )

    # Dynamic spectrum as scatter; robust for C/X frequency gap.
    start = min(r["mjd"] for r in channel_rows)
    t = [(r["mjd"] - start) * 86400.0 / 60.0 for r in channel_rows]
    freq = [r["center_mhz"] for r in channel_rows]
    flux = [r["flux_mjy"] for r in channel_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(t, freq, c=flux, s=18, cmap="viridis")
    ax.set_xlabel("Time from start (min)")
    ax.set_ylabel("Frequency (MHz)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Flux density (mJy)")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_dynamic_spectrum.{ext}"))
    if show:
        plt.show()
    plt.close(fig)

    if alpha_rows:
        t_alpha = minutes_from_start(alpha_rows)
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.errorbar(t_alpha, [r["alpha"] for r in alpha_rows], yerr=[r["alpha_err"] for r in alpha_rows], fmt="o-", ms=3, lw=1, capsize=2)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("Time from start (min)")
        ax.set_ylabel("Spectral index alpha")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_prefix.with_name(f"{output_prefix.name}_spectral_index.{ext}"))
        if show:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Cyg X-3 light curve and time-resolved spectrum from --inband outputs.")
    p.add_argument("--c-target", type=Path, required=True)
    p.add_argument("--c-cal", type=Path, required=True)
    p.add_argument("--c-cal-flux-mjy", type=float, required=True)
    p.add_argument("--x-target", type=Path, required=True)
    p.add_argument("--x-cal", type=Path, required=True)
    p.add_argument("--x-cal-flux-mjy", type=float, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--exclude-bands", default="", help="Comma-separated band numbers to exclude, e.g. 0,15")
    p.add_argument("--ext", default="png")
    p.add_argument("--nofig", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exclude_bands = {int(x) for x in args.exclude_bands.split(",") if x.strip()}
    rows = []
    rows.extend(
        make_flux_rows(
            read_inband(args.c_target),
            read_inband(args.c_cal),
            args.c_cal_flux_mjy,
            "C",
            args.snr_min,
            exclude_bands,
        )
    )
    rows.extend(
        make_flux_rows(
            read_inband(args.x_target),
            read_inband(args.x_cal),
            args.x_cal_flux_mjy,
            "X",
            args.snr_min,
            exclude_bands,
        )
    )
    rows.sort(key=lambda r: (r["mjd"], r["center_mhz"]))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    channel_tsv = args.output_prefix.with_name(f"{args.output_prefix.name}_channel_flux.tsv")
    write_channel_tsv(rows, channel_tsv)

    c_rows = [r for r in rows if r["band_label"] == "C"]
    x_rows = [r for r in rows if r["band_label"] == "X"]

    lightcurve = make_lightcurve(rows)
    lc_tsv = args.output_prefix.with_name(f"{args.output_prefix.name}_lightcurve.tsv")
    write_simple_tsv(lightcurve, lc_tsv, ["epoch", "mjd", "flux_mjy", "flux_err_mjy", "nchan"])

    c_lightcurve = make_lightcurve(c_rows)
    c_lc_tsv = args.output_prefix.with_name(f"{args.output_prefix.name}_C_lightcurve.tsv")
    write_simple_tsv(c_lightcurve, c_lc_tsv, ["epoch", "mjd", "flux_mjy", "flux_err_mjy", "nchan"])

    x_lightcurve = make_lightcurve(x_rows)
    x_lc_tsv = args.output_prefix.with_name(f"{args.output_prefix.name}_X_lightcurve.tsv")
    write_simple_tsv(x_lightcurve, x_lc_tsv, ["epoch", "mjd", "flux_mjy", "flux_err_mjy", "nchan"])

    alpha_rows = make_alpha_series(rows)
    alpha_tsv = args.output_prefix.with_name(f"{args.output_prefix.name}_spectral_index.tsv")
    write_simple_tsv(alpha_rows, alpha_tsv, ["epoch", "mjd", "alpha", "alpha_err", "nchan"])

    plot_outputs(
        rows,
        lightcurve,
        c_lightcurve,
        x_lightcurve,
        alpha_rows,
        args.output_prefix,
        args.ext.lstrip("."),
        show=not args.nofig,
    )

    print(f"Channel flux TSV saved to: {channel_tsv}")
    print(f"Combined light curve TSV saved to: {lc_tsv}")
    print(f"C-band light curve TSV saved to: {c_lc_tsv}")
    print(f"X-band light curve TSV saved to: {x_lc_tsv}")
    print(f"Spectral index TSV saved to: {alpha_tsv}")
    print(f"Plots saved with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()
