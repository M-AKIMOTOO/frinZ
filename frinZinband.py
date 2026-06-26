#!/usr/bin/env python3
"""Plot light curves and spectra from frinZ --inband outputs.

Inputs are frinZ --inband text files. Both the old flat table and the newer
sectioned text format are supported.
"""

from __future__ import annotations

import argparse
import textwrap
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_bool(value: str) -> bool | str:
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def parse_header_value(token: str) -> Any:
    try:
        value = float(token)
        return int(value) if value.is_integer() else value
    except ValueError:
        return token


def normalize_meta_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def parse_metadata_line(text: str, metadata: dict[str, Any]) -> None:
    if text == "In-band fringe search":
        metadata["product"] = "inband_fringe_search"
        return
    if text.startswith("Epoch ") or ":" not in text:
        return
    key, value = text.split(":", 1)
    key = normalize_meta_key(key)
    value = value.strip()
    if key == "bandwidth":
        # Old v1 text: "bandwidth: 512.000 MHz, inband: 32 MHz, bands: 16, RBW: 1.000000 MHz"
        for part in text.split(","):
            if ":" not in part:
                continue
            pkey, pval = part.split(":", 1)
            pkey = normalize_meta_key(pkey)
            tokens = pval.strip().split()
            if not tokens:
                continue
            metadata[pkey] = parse_header_value(tokens[0])
            if len(tokens) > 1:
                metadata[f"{pkey}_unit"] = tokens[1]
    else:
        metadata[key] = parse_bool(value)


def parse_old_rows(lines: list[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 16:
            raise ValueError(f"invalid inband row: {line}")
        rows.append(
            {
                "epoch": f"{parts[0]} {parts[1]}",
                "label": parts[2],
                "source": parts[3],
                "band": int(parts[4]),
                "band_start_mhz": float(parts[5]),
                "band_end_mhz": float(parts[6]),
                "center_mhz": float(parts[7]),
                "length_s": float(parts[8]),
                "amp_percent": float(parts[9]),
                "snr": float(parts[10]),
                "phase_deg": float(parts[11]),
                "noise_percent": float(parts[12]),
                "res_delay_sample": float(parts[13]),
                "res_rate_hz": float(parts[14]),
                "mjd": float(parts[15]),
            }
        )
    return rows


def parse_sectioned_rows(section_lines: dict[str, list[str]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    time_rows = section_lines.get("times", [])
    channel_rows = section_lines.get("channels", [])
    data_rows = section_lines.get("data", [])
    if not time_rows or not channel_rows or not data_rows:
        raise ValueError("sectioned inband text requires @times, @channels, and @data")

    def split_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
        header = lines[0].split("\t")
        body = [line.split("\t") for line in lines[1:] if line.strip()]
        return header, body

    time_header, time_body = split_table(time_rows)
    channel_header, channel_body = split_table(channel_rows)
    data_header, data_body = split_table(data_rows)

    times: dict[int, dict[str, Any]] = {}
    for fields in time_body:
        rec = dict(zip(time_header, fields))
        idx = int(rec["time_index"])
        times[idx] = {"epoch": rec["epoch"], "mjd": float(rec["mjd"])}

    channels: dict[int, dict[str, Any]] = {}
    for fields in channel_body:
        rec = dict(zip(channel_header, fields))
        band = int(rec["band"])
        channels[band] = {
            "band": band,
            "band_start_mhz": float(rec["band_start_mhz"]),
            "band_end_mhz": float(rec["band_end_mhz"]),
            "center_mhz": float(rec["center_mhz"]),
        }

    rows: list[dict[str, Any]] = []
    label = str(metadata.get("label", ""))
    source = str(metadata.get("source", ""))
    length_s = float(metadata.get("length_s", 0.0))
    for fields in data_body:
        rec = dict(zip(data_header, fields))
        time_index = int(rec["time_index"])
        band = int(rec["band"])
        time = times[time_index]
        channel = channels[band]
        rows.append(
            {
                "epoch": time["epoch"],
                "label": label,
                "source": source,
                "band": band,
                "band_start_mhz": channel["band_start_mhz"],
                "band_end_mhz": channel["band_end_mhz"],
                "center_mhz": channel["center_mhz"],
                "length_s": length_s,
                "amp_percent": float(rec["amp_percent"]),
                "snr": float(rec["snr"]),
                "phase_deg": float(rec["phase_deg"]),
                "noise_percent": float(rec["noise_percent"]),
                "res_delay_sample": float(rec["res_delay_sample"]),
                "res_rate_hz": float(rec["res_rate_hz"]),
                "mjd": time["mjd"],
            }
        )
    return rows


def parse_inband_txt(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    old_data_lines: list[str] = []
    section_lines: dict[str, list[str]] = defaultdict(list)
    current_section: str | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parse_metadata_line(line[1:].strip(), metadata)
                continue
            if line.startswith("@"):
                current_section = line[1:].strip().lower()
                section_lines[current_section] = []
                continue
            if current_section is not None:
                section_lines[current_section].append(line)
            else:
                old_data_lines.append(line)

    rows = (
        parse_sectioned_rows(section_lines, metadata)
        if section_lines
        else parse_old_rows(old_data_lines, metadata)
    )
    if not rows:
        raise ValueError(f"no data rows found: {path}")
    return metadata, rows


def source_key(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        _meta, parsed = parse_inband_txt(path)
        for row in parsed:
            row = dict(row)
            row["file"] = path.name
            rows.append(row)
    if not rows:
        raise ValueError("no inband rows were read")
    return rows


def select_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    wanted = source_key(source)
    selected = [row for row in rows if source_key(str(row.get("source", ""))) == wanted]
    if not selected:
        available = sorted({str(row.get("source", "")) for row in rows})
        raise ValueError(f"source '{source}' not found. Available sources: {available}")
    return selected


def amp_sigma(row: dict[str, Any]) -> float:
    snr = float(row["snr"])
    amp = float(row["amp_percent"])
    return abs(amp / snr) if snr > 0.0 else 0.0


def weighted_mean(values: Iterable[tuple[float, float]]) -> tuple[float, float]:
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
    return sx / sw, math.sqrt(1.0 / sw)


def calibrator_table(rows: list[dict[str, Any]], snr_min: float) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["snr"]) >= snr_min:
            grouped[int(row["band"])].append(row)
    if not grouped:
        raise ValueError("no calibrator rows remain after SNR filtering")
    out: dict[int, dict[str, Any]] = {}
    for band, group in grouped.items():
        amp, err = weighted_mean((float(r["amp_percent"]), amp_sigma(r)) for r in group)
        first = group[0]
        out[band] = {
            "band": band,
            "amp_percent": amp,
            "amp_err_percent": err,
            "center_mhz": float(first["center_mhz"]),
            "band_start_mhz": float(first["band_start_mhz"]),
            "band_end_mhz": float(first["band_end_mhz"]),
            "n": len(group),
        }
    return out


def calibrate_band(
    rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    cal_flux_mjy: float,
    band_label: str,
    snr_min: float,
) -> list[dict[str, Any]]:
    cal = calibrator_table(cal_rows, snr_min)
    out: list[dict[str, Any]] = []
    for row in rows:
        if float(row["snr"]) < snr_min:
            continue
        band = int(row["band"])
        if band not in cal:
            continue
        cal_amp = float(cal[band]["amp_percent"])
        if cal_amp <= 0.0:
            continue
        amp = float(row["amp_percent"])
        flux = amp / cal_amp * cal_flux_mjy
        target_amp_err = amp_sigma(row)
        cal_amp_err = float(cal[band]["amp_err_percent"])
        frac_var = 0.0
        if target_amp_err > 0.0 and amp > 0.0:
            frac_var += (target_amp_err / amp) ** 2
        if cal_amp_err > 0.0:
            frac_var += (cal_amp_err / cal_amp) ** 2
        flux_err = flux * math.sqrt(frac_var) if frac_var > 0.0 else 0.0
        out.append(
            {
                "epoch": row["epoch"],
                "mjd": float(row["mjd"]),
                "band_label": band_label,
                "band": band,
                "center_mhz": float(row["center_mhz"]),
                "band_start_mhz": float(row["band_start_mhz"]),
                "band_end_mhz": float(row["band_end_mhz"]),
                "target_source": row["source"],
                "calibrator_flux_mjy": cal_flux_mjy,
                "target_amp_percent": amp,
                "cal_amp_percent": cal_amp,
                "target_snr": float(row["snr"]),
                "flux_mjy": flux,
                "flux_err_mjy": flux_err,
            }
        )
    return sorted(out, key=lambda r: (r["mjd"], r["center_mhz"]))


def group_by_epoch(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["epoch"])].append(row)
    return grouped


def mean_mjd(group: list[dict[str, Any]]) -> float:
    return sum(float(r["mjd"]) for r in group) / len(group)


def make_lightcurve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted(group_by_epoch(rows).items(), key=lambda item: mean_mjd(item[1]))
    for epoch, group in groups:
        flux, err = weighted_mean((float(r["flux_mjy"]), float(r["flux_err_mjy"])) for r in group)
        out.append(
            {
                "epoch": epoch,
                "mjd": mean_mjd(group),
                "flux_mjy": flux,
                "flux_err_mjy": err,
                "nchan": len(group),
            }
        )
    return out


def make_mean_spectrum(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["band_label"]), int(row["band"]))].append(row)
    out: list[dict[str, Any]] = []
    for (band_label, band), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        flux, err = weighted_mean((float(r["flux_mjy"]), float(r["flux_err_mjy"])) for r in group)
        first = group[0]
        out.append(
            {
                "band_label": band_label,
                "band": band,
                "center_mhz": float(first["center_mhz"]),
                "band_start_mhz": float(first["band_start_mhz"]),
                "band_end_mhz": float(first["band_end_mhz"]),
                "flux_mjy": flux,
                "flux_err_mjy": err,
                "ntime": len(group),
            }
        )
    return out


def fit_alpha(group: list[dict[str, Any]]) -> tuple[float, float, int]:
    valid = [r for r in group if float(r["flux_mjy"]) > 0.0 and float(r["center_mhz"]) > 0.0]
    if len(valid) < 2:
        return float("nan"), float("nan"), len(valid)
    xs = [math.log(float(r["center_mhz"])) for r in valid]
    ys = [math.log(float(r["flux_mjy"])) for r in valid]
    ws = []
    for r in valid:
        flux = float(r["flux_mjy"])
        err = float(r["flux_err_mjy"])
        ws.append(1.0 / ((err / flux) ** 2) if err > 0.0 else 1.0)
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


def make_alpha_series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted(group_by_epoch(rows).items(), key=lambda item: mean_mjd(item[1]))
    for epoch, group in groups:
        labels = {r["band_label"] for r in group}
        if len(labels) < 2:
            continue
        alpha, err, nchan = fit_alpha(group)
        out.append({"epoch": epoch, "mjd": mean_mjd(group), "alpha": alpha, "alpha_err": err, "nchan": nchan})
    return out


def format_tsv_value(key: str, value: Any) -> str:
    if key == "epoch":
        return str(value).replace(" ", "T", 1)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def write_tsv(rows: list[dict[str, Any]], path: Path, keys: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                values.append(format_tsv_value(key, row[key]))
            fh.write("\t".join(values) + "\n")


def mjd_ref(rows: list[dict[str, Any]]) -> int:
    return math.floor(min(float(r["mjd"]) for r in rows))


def x_mjd(rows: list[dict[str, Any]], ref: int) -> list[float]:
    return [float(r["mjd"]) - ref for r in rows]


def x_hour(rows: list[dict[str, Any]]) -> list[float]:
    return [(float(r["mjd"]) - math.floor(float(r["mjd"]))) * 24.0 for r in rows]


def first_mjd(rows: list[dict[str, Any]]) -> float:
    return min(float(r["mjd"]) for r in rows)


def set_axis_fontsize(ax, size: int = 15) -> None:
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)
    ax.tick_params(axis="both", which="major", labelsize=size)
    ax.tick_params(axis="both", which="minor", labelsize=size)


def set_colorbar_fontsize(cb, size: int = 15) -> None:
    cb.ax.yaxis.label.set_size(size)
    cb.ax.tick_params(labelsize=size)


def compress_png(path: Path, enabled: bool = True) -> None:
    if not enabled or path.suffix.lower() != ".png":
        return
    exe = shutil.which("pngquant")
    if exe is None:
        print(f"# pngquant not found; skipped PNG compression: {path}")
        return
    tmp = path.with_name(f"{path.stem}-fs8{path.suffix}")
    cmd = [exe, "--force", "--quality", "80-95", "--speed", "1", "--ext", "-fs8.png", str(path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if tmp.exists():
            tmp.replace(path)
    except Exception as exc:
        print(f"# pngquant failed; kept original PNG: {path} ({exc})")
        if tmp.exists():
            tmp.unlink()


def savefig(fig, path: Path, pngquant: bool) -> None:
    save_kwargs = {"dpi": 160}
    if path.suffix.lower() == ".ps":
        save_kwargs["orientation"] = "landscape"
    fig.savefig(path, **save_kwargs)
    compress_png(path, enabled=pngquant)


def plot_lightcurves(c_lc, x_lc, out_prefix: Path, exts: list[str], show: bool, ref: int, mjd0: float, pngquant: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    display_scale, display_unit = flux_display_scale(c_lc + x_lc)
    series = [(c_lc, "C", "s"), (x_lc, "X", "^")]
    for mode, xlabel, suffix in [
        ("mjd", f"MJD - {ref}", "lc_mjd"),
        ("hour", f"Hour on MJD {ref}", "lc_hrs"),
    ]:
        fig, ax = plt.subplots(figsize=(12.0, 7.0))
        for rows, label, marker in series:
            if not rows:
                continue
            x = x_mjd(rows, ref) if mode == "mjd" else x_hour(rows)
            ax.errorbar(
                x,
                [r["flux_mjy"] / display_scale for r in rows],
                yerr=[r["flux_err_mjy"] / display_scale for r in rows],
                fmt=f"{marker}-",
                ms=4,
                lw=1.2,
                capsize=2,
                label=label,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Flux density ({display_unit})")
        if display_unit == "Jy":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        set_axis_fontsize(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=15)
        fig.tight_layout()
        for ext in exts:
            savefig(fig, out_prefix.with_name(f"{out_prefix.name}_{suffix}.{ext}"), pngquant)
        if show:
            plt.show()
        plt.close(fig)


def plot_mean_spectrum(spectrum, out_prefix: Path, exts: list[str], show: bool, pngquant: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    display_scale, display_unit = flux_display_scale(spectrum)
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    freq_tick_values: list[float] = []
    for band_label, marker in [("C", "s"), ("X", "^")]:
        rows = [r for r in spectrum if r["band_label"] == band_label]
        if not rows:
            continue
        freq_tick_values.extend(freq_ticks(rows))
        ax.errorbar(
            [r["center_mhz"] for r in rows],
            [r["flux_mjy"] / display_scale for r in rows],
            yerr=[r["flux_err_mjy"] / display_scale for r in rows],
            fmt=f"{marker}-",
            ms=4,
            lw=1,
            capsize=2,
            label=band_label,
        )
    ax.set_xlabel("Frequency (MHz)")
    if freq_tick_values:
        ax.set_xticks(sorted(set(freq_tick_values)))
    ax.set_ylabel(f"Flux density ({display_unit})")
    if display_unit == "Jy":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    set_axis_fontsize(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=15)
    fig.tight_layout()
    for ext in exts:
        savefig(fig, out_prefix.with_name(f"{out_prefix.name}_sp.{ext}"), pngquant)
    if show:
        plt.show()
    plt.close(fig)


def plot_channel_lightcurves(c_rows, x_rows, out_prefix: Path, exts: list[str], show: bool, ref: int, pngquant: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter

    display_scale, display_unit = flux_display_scale(c_rows + x_rows)
    cmap = plt.get_cmap("turbo")

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 10.0), sharex=True, constrained_layout=True)
    for ax, rows in [(axes[0], x_rows), (axes[1], c_rows)]:
        freqs = [float(r["center_mhz"]) for r in rows]
        norm = Normalize(vmin=min(freqs), vmax=max(freqs))
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["band"])].append(row)
        for band in sorted(grouped):
            group = sorted(grouped[band], key=lambda r: float(r["mjd"]))
            freq = float(group[0]["center_mhz"])
            ax.errorbar(
                x_hour(group),
                [r["flux_mjy"] / display_scale for r in group],
                yerr=[r["flux_err_mjy"] / display_scale for r in group],
                color=cmap(norm(freq)),
                lw=1.0,
                alpha=0.9,
                capsize=0,
                label=f"{freq:.0f} MHz",
            )
        ax.set_ylabel(f"Flux density ({display_unit})")
        if display_unit == "Jy":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        set_axis_fontsize(ax)
        ax.grid(True, alpha=0.25)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.01)
        cb.set_label("Frequency (MHz)")
        cb.set_ticks(freq_ticks(rows))
        set_colorbar_fontsize(cb)
    axes[1].set_xlabel(f"Hour on MJD {ref}")
    for ax in axes:
        set_axis_fontsize(ax)
    for ext in exts:
        savefig(fig, out_prefix.with_name(f"{out_prefix.name}_chlc.{ext}"), pngquant)
    if show:
        plt.show()
    plt.close(fig)


def flux_display_scale(rows: list[dict[str, Any]]) -> tuple[float, str]:
    values = [float(r["flux_mjy"]) for r in rows if math.isfinite(float(r["flux_mjy"]))]
    if values and min(values) > 1000.0:
        return 1000.0, "Jy"
    return 1.0, "mJy"


def band_frequency_range(rows: list[dict[str, Any]]) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for row in rows:
        width = float(row["band_end_mhz"]) - float(row["band_start_mhz"])
        center = float(row["center_mhz"])
        lows.append(center - width / 2.0)
        highs.append(center + width / 2.0)
    return min(lows), max(highs)


def fmt_freq_range(rows: list[dict[str, Any]], unit: bool = True) -> str:
    low, high = band_frequency_range(rows)
    suffix = " MHz" if unit else ""
    return f"{low:.0f}-{high:.0f}{suffix}"


def freq_ticks(rows: list[dict[str, Any]], step_mhz: float = 128.0) -> list[float]:
    low, high = band_frequency_range(rows)
    ticks: list[float] = []
    value = low
    while value <= high + 1.0e-6:
        ticks.append(value)
        value += step_mhz
    if abs(ticks[-1] - high) > 1.0e-6:
        ticks.append(high)
    return ticks


def set_frequency_axis(ax, rows: list[dict[str, Any]], axis: str = "y") -> None:
    low, high = band_frequency_range(rows)
    ticks = freq_ticks(rows)
    if axis == "y":
        ax.set_ylim(low, high)
        ax.set_yticks(ticks)
    else:
        ax.set_xlim(low, high)
        ax.set_xticks(ticks)


def plot_channel_spectra(c_rows, x_rows, out_prefix: Path, exts: list[str], show: bool, ref: int, pngquant: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    all_hours = x_hour(c_rows + x_rows)
    norm = Normalize(vmin=min(all_hours), vmax=max(all_hours))
    cmap = plt.get_cmap("viridis")
    display_scale, display_unit = flux_display_scale(c_rows + x_rows)
    c_range = fmt_freq_range(c_rows)
    x_range = fmt_freq_range(x_rows)
    c_range_short = fmt_freq_range(c_rows, unit=False)
    x_range_short = fmt_freq_range(x_rows, unit=False)

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 10.0), sharex=True, constrained_layout=True)
    for ax, rows, label, freq_range in [(axes[0], x_rows, "X-band", x_range), (axes[1], c_rows, "C-band", c_range)]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["epoch"])].append(row)
        for _epoch, group in sorted(grouped.items(), key=lambda item: mean_mjd(item[1])):
            group = sorted(group, key=lambda r: float(r["band_start_mhz"]))
            hour = (float(group[0]["mjd"]) - math.floor(float(group[0]["mjd"]))) * 24.0
            xs: list[float] = []
            ys: list[float] = []
            for row in group:
                start = float(row["band_start_mhz"])
                end = float(row["band_end_mhz"])
                flux = float(row["flux_mjy"]) / display_scale
                xs.extend([start, end, float("nan")])
                ys.extend([flux, flux, float("nan")])
            ax.plot(xs, ys, color=cmap(norm(hour)), lw=0.8, alpha=0.35)
        ax.plot([], [], color="black", lw=1.5, label=f"{label} ({freq_range})")
        ax.legend(frameon=False, fontsize=15, loc="best")
        ax.set_ylabel(f"Flux density ({display_unit})")
        if display_unit == "Jy":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        set_axis_fontsize(ax)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel(f"Bandwidth offset (MHz; {c_range_short} & {x_range_short} MHz)")
    for ax in axes:
        ax.set_xlim(0.0, 512.0)
        ax.xaxis.set_major_locator(MultipleLocator(128.0))
        set_axis_fontsize(ax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, pad=0.01)
    cb.set_label(f"Hour on MJD {ref}")
    set_colorbar_fontsize(cb)
    for ext in exts:
        savefig(fig, out_prefix.with_name(f"{out_prefix.name}_chsp.{ext}"), pngquant)
    if show:
        plt.show()
    plt.close(fig)


def dynamic_grid(rows: list[dict[str, Any]]) -> tuple[list[float], list[float], Any]:
    import numpy as np

    times = sorted({float(r["mjd"]) for r in rows})
    freqs = sorted({float(r["center_mhz"]) for r in rows})
    time_index = {value: i for i, value in enumerate(times)}
    freq_index = {value: i for i, value in enumerate(freqs)}
    z = np.full((len(freqs), len(times)), np.nan, dtype=float)
    for row in rows:
        z[freq_index[float(row["center_mhz"])], time_index[float(row["mjd"])]] = float(row["flux_mjy"])
    return times, freqs, z


def edge_values(centers: list[float]) -> list[float]:
    if len(centers) == 1:
        step = 1.0
        return [centers[0] - step / 2.0, centers[0] + step / 2.0]
    edges = [centers[0] - (centers[1] - centers[0]) / 2.0]
    for left, right in zip(centers, centers[1:]):
        edges.append((left + right) / 2.0)
    edges.append(centers[-1] + (centers[-1] - centers[-2]) / 2.0)
    return edges


def plot_dynamic(c_rows, x_rows, out_prefix: Path, exts: list[str], show: bool, ref: int, mjd0: float, pngquant: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    display_scale, display_unit = flux_display_scale(c_rows + x_rows)
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.0), sharex=True, constrained_layout=True)
    for ax, rows, title in [(axes[0], x_rows, "X"), (axes[1], c_rows, "C")]:
        if not rows:
            ax.set_title(f"{title}: no data")
            continue
        times, freqs, z = dynamic_grid(rows)
        t_edges = [(v - math.floor(v)) * 24.0 for v in edge_values(times)]
        f_edges = edge_values(freqs)
        mesh = ax.pcolormesh(t_edges, f_edges, z / display_scale, shading="auto", cmap="viridis")
        ax.set_ylabel("Frequency (MHz)")
        set_frequency_axis(ax, rows, axis="y")
        set_axis_fontsize(ax)
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(display_unit)
        if display_unit == "Jy":
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        set_colorbar_fontsize(cb)
    axes[1].set_xlabel(f"Hour on MJD {ref}")
    for ax in axes:
        set_axis_fontsize(ax)
    for ext in exts:
        savefig(fig, out_prefix.with_name(f"{out_prefix.name}_dynamic_sp.{ext}"), pngquant)
    if show:
        plt.show()
    plt.close(fig)


def plot_alpha(alpha_rows, out_prefix: Path, exts: list[str], show: bool, ref: int, pngquant: bool) -> None:
    if not alpha_rows:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.errorbar(
        x_hour(alpha_rows),
        [r["alpha"] for r in alpha_rows],
        yerr=[r["alpha_err"] for r in alpha_rows],
        fmt="o-",
        ms=4,
        lw=1.2,
        capsize=2,
    )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"Hour on MJD {ref}")
    ax.set_ylabel("Spectral index α  (Sν ∝ ν^α)")
    set_axis_fontsize(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in exts:
        savefig(fig, out_prefix.with_name(f"{out_prefix.name}_spidx.{ext}"), pngquant)
    if show:
        plt.show()
    plt.close(fig)


def default_output_prefix(target: str, xdata: list[Path], cdata: list[Path]) -> Path:
    base = xdata[0] if xdata else cdata[0]
    safe_target = target.replace(" ", "_").replace("/", "-")
    return base.with_name(f"{base.stem}_{safe_target}_inband")


def parse_args() -> argparse.Namespace:
    epilog = """
Data processing summary:
  1. Read frinZ --inband text outputs for X and C bands. Each --xdata/--cdata
     argument may contain both target and calibrator files. Rows are selected by
     --target and --calib source names.

  2. Apply SNR filtering. Rows with SNR < --snr are excluded for both target and
     calibrator.

  3. Relative calibration is done independently for each frequency channel:
       target_flux_mJy(channel, time)
         = target_amp_percent(channel, time)
           / mean_calib_amp_percent(channel)
           * calib_flux_mJy
     where calib_flux_mJy is --xflux-calib for X band and --cflux-calib for C band.
     The calibrator amplitude is a weighted mean over calibrator time samples in
     the same channel. The amplitude error is estimated as amp/SNR.

  4. *_chflux.tsv contains the calibrated per-time, per-channel fluxes.

  5. Light curves:
       *_c_lc.tsv: C-band channels averaged at each epoch.
       *_x_lc.tsv: X-band channels averaged at each epoch.
     The average is weighted by 1/flux_err^2.
     Figures are written as *_lc_hrs.<ext> and *_lc_mjd.<ext>, showing C and X band
     light curves separately.
     The *_lc_hrs.<ext> x-axis is (fractional MJD) * 24 [hour], labeled as Hour on MJD <ref>.

  6. In-band channel comparison figures:
       *_chlc.<ext> compares the light curves of all in-band frequency channels
       after splitting the 512 MHz bandwidth into 32 MHz channels. Upper panel is X,
       lower panel is C. Each panel has its own frequency color bar with 128 MHz ticks.
       *_chsp.<ext> compares the per-epoch in-band spectra. The x-axis is
       bandwidth offset 0-512 MHz, and each 32 MHz channel is drawn as one
       horizontal bin: 0:32, 32:64, ..., 480:512 MHz. The axis label also shows
       the absolute frequency ranges, e.g. 6600-7112 & 8192-8704 MHz. Upper panel is X, lower panel is C.
       Line color indicates Hour on MJD <ref>. If all plotted fluxes exceed
       1000 mJy, flux-density graph displays are converted to Jy with %.1f tick labels.

  7. Mean spectrum:
       *_sp.tsv and *_sp.<ext> are made by averaging each frequency
       channel over time, again weighted by 1/flux_err^2.

  8. Dynamic spectrum:
       *_dynamic_sp.<ext> is not a time average. It is a color map of the
       calibrated target flux for each epoch and each in-band frequency channel.
       The upper panel is X band and the lower panel is C band.

  9. Spectral index time series:
       *_spidx.tsv and *_spidx.<ext> are made at each epoch by
       fitting log(flux_mJy) = alpha * log(freq_MHz) + constant using C+X channels.
       C and X are joined by epoch string, not by exact floating-point MJD equality.

Output extensions:
  --ext accepts one or more extensions. Example: --ext png eps ps

PNG compression:
  PNG outputs are compressed by pngquant by default when the pngquant command is
  available. Use --no-pngquant to keep matplotlib PNG files uncompressed.

Example:
  frinZinband.py
    --xdata target_x_inband.txt calibrator_x_inband.txt
    --cdata target_c_inband.txt calibrator_c_inband.txt
    --target CYGX-3 --calib 2016+386 --snr 30
    --xflux-calib 418 --cflux-calib 381 --nofig
"""
    p = argparse.ArgumentParser(
        description="Create frinZ --inband light curves, spectra, dynamic spectra, and spectral-index time series.",
        epilog=textwrap.dedent(epilog),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--xdata", nargs="+", type=Path, required=True, help="X-band frinZ --inband output(s), target and calibrator")
    p.add_argument("--cdata", nargs="+", type=Path, required=True, help="C-band frinZ --inband output(s), target and calibrator")
    p.add_argument("--target", required=True, help="Target source name")
    p.add_argument("--calib", required=True, help="Calibrator source name")
    p.add_argument("--snr", type=float, default=0.0, help="SNR threshold")
    p.add_argument("--xflux-calib", type=float, required=True, help="X-band calibrator flux density [mJy]")
    p.add_argument("--cflux-calib", type=float, required=True, help="C-band calibrator flux density [mJy]")
    p.add_argument("--ext", nargs="+", default=["png"], help="Graph extension(s): png, pdf, ps, eps, jpg, ... Example: --ext eps ps")
    p.add_argument("--output-prefix", type=Path, help="Output prefix [default: derived from xdata]")
    p.add_argument("--nofig", action="store_true", help="Do not show figures interactively")
    p.add_argument("--no-pngquant", action="store_true", help="Do not compress PNG outputs with pngquant")
    if len(sys.argv) == 1:
        p.print_help()
        raise SystemExit(0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_all = read_rows(args.xdata)
    c_all = read_rows(args.cdata)
    x_target = select_source(x_all, args.target)
    x_cal = select_source(x_all, args.calib)
    c_target = select_source(c_all, args.target)
    c_cal = select_source(c_all, args.calib)

    x_flux = calibrate_band(x_target, x_cal, args.xflux_calib, "X", args.snr)
    c_flux = calibrate_band(c_target, c_cal, args.cflux_calib, "C", args.snr)
    all_flux = sorted(c_flux + x_flux, key=lambda r: (r["mjd"], r["center_mhz"]))
    if not all_flux:
        raise ValueError("no calibrated target rows remain after SNR filtering")

    out_prefix = args.output_prefix or default_output_prefix(args.target, args.xdata, args.cdata)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    exts = [ext.lstrip(".") for ext in args.ext]
    ref = mjd_ref(all_flux)
    mjd0 = first_mjd(all_flux)

    c_lc = make_lightcurve(c_flux)
    x_lc = make_lightcurve(x_flux)
    spectrum = make_mean_spectrum(all_flux)
    alpha = make_alpha_series(all_flux)

    write_tsv(
        all_flux,
        out_prefix.with_name(f"{out_prefix.name}_chflux.tsv"),
        [
            "epoch",
            "mjd",
            "band_label",
            "band",
            "center_mhz",
            "target_source",
            "calibrator_flux_mjy",
            "target_amp_percent",
            "cal_amp_percent",
            "target_snr",
            "flux_mjy",
            "flux_err_mjy",
        ],
    )
    write_tsv(c_lc, out_prefix.with_name(f"{out_prefix.name}_c_lc.tsv"), ["epoch", "mjd", "flux_mjy", "flux_err_mjy", "nchan"])
    write_tsv(x_lc, out_prefix.with_name(f"{out_prefix.name}_x_lc.tsv"), ["epoch", "mjd", "flux_mjy", "flux_err_mjy", "nchan"])
    write_tsv(spectrum, out_prefix.with_name(f"{out_prefix.name}_sp.tsv"), ["band_label", "band", "center_mhz", "band_start_mhz", "band_end_mhz", "flux_mjy", "flux_err_mjy", "ntime"])
    write_tsv(alpha, out_prefix.with_name(f"{out_prefix.name}_spidx.tsv"), ["epoch", "mjd", "alpha", "alpha_err", "nchan"])

    plot_lightcurves(c_lc, x_lc, out_prefix, exts, show=not args.nofig, ref=ref, mjd0=mjd0, pngquant=not args.no_pngquant)
    plot_mean_spectrum(spectrum, out_prefix, exts, show=not args.nofig, pngquant=not args.no_pngquant)
    plot_channel_lightcurves(c_flux, x_flux, out_prefix, exts, show=not args.nofig, ref=ref, pngquant=not args.no_pngquant)
    plot_channel_spectra(c_flux, x_flux, out_prefix, exts, show=not args.nofig, ref=ref, pngquant=not args.no_pngquant)
    plot_dynamic(c_flux, x_flux, out_prefix, exts, show=not args.nofig, ref=ref, mjd0=mjd0, pngquant=not args.no_pngquant)
    plot_alpha(alpha, out_prefix, exts, show=not args.nofig, ref=ref, pngquant=not args.no_pngquant)

    print(f"Output prefix: {out_prefix}")
    print(f"MJD ref: {ref}")
    print(f"Hour zero MJD: {mjd0:.5f}")
    print(f"C rows: {len(c_flux)}, X rows: {len(x_flux)}, total rows: {len(all_flux)}")


if __name__ == "__main__":
    main()
