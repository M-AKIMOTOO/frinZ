#!/usr/bin/env python3
"""Editable frinZtable figure generator. Requires matplotlib.

Run: python3 graph.py
     python3 graph.py --output revised.eps --xlim 0.20 0.35
     python3 graph.py --sources NRAO530 4U1755-3 --title 'My observation'

The selected rows and graph settings are embedded in this single Python file,
so it can be moved and managed on its own. The original graph is not
overwritten: default output is *_replot.png.
Edit the constants below, SETTINGS, or the plotting code to tune it.
No tight bounding-box cropping is used, keeping the complete figure at 9:6.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

# Figure size is kept at 9:6; visual styling uses Matplotlib defaults.
FIGSIZE = (9, 6)
DPI = 200

# Embedded by frinZtable when the script is exported.
EMBEDDED = json.loads(__FRINZ_EMBEDDED_JSON__)
SETTINGS = EMBEDDED["settings"]
DATA_TSV = EMBEDDED["data_tsv"]


def numeric(text):
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def transform(value, log):
    if value is None or (log and value <= 0):
        return None
    return math.log10(value) if log else value


def output_with_suffix(path, suffix):
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def hours_since_reference(value, settings):
    if settings.get("hour_conversion") == "seconds":
        return value / 3600.0
    reference = settings["hour_reference_mjd"]
    mjd_reference = settings.get("reference_mjd") or 0.0
    return (value + mjd_reference - reference) * 24.0


def convert_axis_to_hours(ax, ax2, settings):
    """Convert plotted x coordinates while retaining the primary plot limits."""
    for axis in (ax, ax2):
        if axis is None:
            continue
        for line in axis.lines:
            line.set_xdata([hours_since_reference(float(x), settings) for x in line.get_xdata()])
        for collection in axis.collections:
            segments = collection.get_segments()
            if segments:
                collection.set_segments([
                    [(hours_since_reference(float(x), settings), y) for x, y in segment]
                    for segment in segments
                ])
    limits = ax.get_xlim()
    ax.set_xlim(tuple(hours_since_reference(float(x), settings) for x in limits))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="PNG, PDF, EPS, SVG, etc.; default: *_replot.png")
    parser.add_argument("--sources", nargs="+", help="Only these source names; default: all exported sources")
    parser.add_argument("--title")
    parser.add_argument("--xlabel")
    parser.add_argument("--ylabel")
    parser.add_argument("--xlim", type=float, nargs=2)
    parser.add_argument("--ylim", type=float, nargs=2)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--style", choices=("scatter", "linear", "scatter-dotted"),
                        help="Override the embedded time-series style")
    parser.add_argument("--no-error-bars", action="store_true")
    parser.add_argument("--connect", action="store_true")
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as mticker
    except ImportError as error:
        raise SystemExit("Install matplotlib in your Python environment: python3 -m pip install matplotlib") from error
    plt.style.use("default")
    base = Path(__file__).resolve()
    settings = SETTINGS
    output = args.output or base.with_name(base.stem + "_replot.png")
    rows = list(csv.DictReader(io.StringIO(DATA_TSV), delimiter="\t"))
    if args.sources is not None:
        rows = [r for r in rows if r["Source"] in args.sources]
    groups = {}
    for row in rows:
        groups.setdefault(row["Source"], []).append(row)
    groups = dict(sorted(groups.items()))
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    has_y2 = settings.get("mode") == "series" and settings.get("y2_label") is not None
    ax2 = ax.twinx() if has_y2 else None
    fig.subplots_adjust(left=0.14, right=0.84 if has_y2 else 0.97, bottom=0.15, top=0.90)
    log = settings["log_y"]
    mode = settings["mode"]
    parameter_label = settings["y_label"]
    style = (args.style or settings.get("series_style", "scatter")).replace("-", "_")
    show_markers = style != "linear"
    connect = style != "scatter" or args.connect
    line_style = ":" if style == "scatter_dotted" and not args.connect else "-" if connect else "None"
    error_bars = not args.no_error_bars
    edges = None
    if mode == "histogram":
        values = [transform(numeric(r["Y"]), log) for r in rows]
        values = [v for v in values if v is not None]
        if values:
            low, high = min(values), max(values)
            bins = args.bins if args.bins is not None else settings["bins"]
            if bins < 1:
                parser.error("--bins must be positive")
            if low == high:
                width = max(abs(low) * 0.05, 1e-6)
                edges = [low - width / 2, high + width / 2]
            else:
                width = (high - low) / bins
                edges = [low + i * width for i in range(bins + 1)]
                edges[-1] = high
    for source_index, (source, group) in enumerate(groups.items()):
        marker_index = settings["sources"].index(source) if source in settings["sources"] else source_index
        if mode == "series":
            points = [(numeric(r["X"]), transform(numeric(r["Y"]), log), numeric(r["Y"]), numeric(r["Noise"])) for r in group]
            points = sorted((p for p in points if p[0] is not None and p[1] is not None), key=lambda p: p[0])
            if points:
                series = ax.plot([p[0] for p in points], [p[1] for p in points], label=source,
                                 marker="o" if show_markers else None, linestyle=line_style)
                series_color = series[0].get_color()
            if ax2 is not None:
                points2 = [(numeric(r["X"]), transform(numeric(r.get("Y2")), log)) for r in group]
                points2 = sorted((p for p in points2 if p[0] is not None and p[1] is not None), key=lambda p: p[0])
                if points2:
                    ax2.plot([p[0] for p in points2], [p[1] for p in points2], label=f"{source} [Y2]",
                             marker="o" if show_markers else None, markerfacecolor="none",
                             linestyle=line_style)
            if points and error_bars and settings["noise_bars"]:
                for x, y, raw, noise in points:
                    if noise is None or noise < 0:
                        continue
                    low, high = transform(raw - noise, log), transform(raw + noise, log)
                    if low is not None and high is not None:
                        ax.vlines(x, low, high, color=series_color, alpha=0.65)
        elif mode == "histogram":
            values = [transform(numeric(r["Y"]), log) for r in group]
            values = [v for v in values if v is not None]
            if values and edges is not None:
                counts, _, patches = ax.hist(values, bins=edges, histtype="step")
                centers = [(left + right) / 2 for left, right in zip(edges, edges[1:])]
                ax.plot([x for x, count in zip(centers, counts) if count > 0],
                        [count for count in counts if count > 0],
                        color=patches[0].get_edgecolor() if patches else None,
                        label=source, marker="o", linestyle="None")
        elif mode == "sources":
            values = [numeric(r["Y"]) for r in group]
            values = [v for v in values if v is not None]
            if not values:
                continue
            raw = statistics.mean(values)
            y = transform(raw, log)
            if y is None:
                continue
            series = ax.plot([source_index], [y], marker="o", label=source)
            if error_bars and len(values) > 1:
                sem = statistics.stdev(values) / math.sqrt(len(values))
                low, high = transform(raw - sem, log), transform(raw + sem, log)
                if low is not None and high is not None:
                    ax.vlines(source_index, low, high, color=series[0].get_color())
    if mode == "sources":
        ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlabel(args.xlabel or (parameter_label if mode == "histogram" else "Source (mean ± standard error)" if mode == "sources" else settings["x_label"]))
    ax.set_ylabel(args.ylabel or ("Count" if mode == "histogram" else parameter_label))
    if ax2 is not None:
        ax2.set_ylabel(settings["y2_label"])
    ax.set_title(args.title if args.title is not None else settings["title"])
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    if mode != "sources":
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    if settings.get("x_axis_scale") == "date":
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _position: mdates.num2date(value - 40587.0).strftime("%Y-%m-%d\n%H:%M")
            )
        )
        fig.autofmt_xdate()
    if args.xlim:
        ax.set_xlim(args.xlim)
    if args.ylim:
        ax.set_ylim(args.ylim)
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    if handles:
        ax.legend(handles, labels)
    if settings.get("time_axis", False):
        outputs = [output, output_with_suffix(output, "hour")]
    else:
        outputs = [output]
    fig.savefig(outputs[0], dpi=args.dpi)
    print(f"Saved {outputs[0]} ({FIGSIZE[0]} × {FIGSIZE[1]} inches, {args.dpi} dpi)")
    if len(outputs) == 2:
        convert_axis_to_hours(ax, ax2, settings)
        ax.set_xlabel(settings["hour_label"])
        fig.savefig(outputs[1], dpi=args.dpi)
        print(f"Saved {outputs[1]} ({FIGSIZE[0]} × {FIGSIZE[1]} inches, {args.dpi} dpi)")
    plt.close(fig)


if __name__ == "__main__":
    main()
