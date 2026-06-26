#!/usr/bin/env python3
"""Inspect and quickly plot a compressed frinZ .npz file."""

import argparse
import sys
from pathlib import Path



def text(record, name):
    value = record[name]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\0")
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description="frinZ が出力した圧縮された自己記述型 .npz を表示・TSV変換します。",
        epilog="例: ./tools/npz_open.py --npz result_bptable.npz --output --ext pdf --nofig",
    )
    parser.add_argument(
        "--npz", required=True, type=Path, metavar="FILE",
        help="入力する frinZ の .npz ファイル",
    )
    parser.add_argument(
        "--output", action="store_true",
        help="入力ファイルと同じ stem の .tsv とグラフを保存",
    )
    parser.add_argument(
        "--ext", default="png", metavar="EXT",
        help="保存するグラフの拡張子 (既定: png; 例: pdf, ps, eps, jpg)",
    )
    parser.add_argument(
        "--nofig", action="store_true",
        help="plt.show() による画面表示を行わない",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    loaded = np.load(args.npz, allow_pickle=False)
    array = loaded["data"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
    if isinstance(loaded, np.lib.npyio.NpzFile) and array.dtype.names is None:
        data = array
        real = data.real
        imag = data.imag

        def npz_text(name):
            return loaded[name].tobytes().decode("utf-8", errors="replace")

        flag = npz_text("flag")
        axis0 = loaded["axis0"]
        axis1 = loaded["axis1"]
        axis0_name = npz_text("axis0_name") or "index"
        axis0_unit = npz_text("axis0_unit")
        axis1_name = npz_text("axis1_name") or "index"
        axis1_unit = npz_text("axis1_unit")
        fft_point = int(loaded["fft_point"].flat[0])
        pp = int(loaded["pp"].flat[0])
        fields = loaded.files
    else:
        record = array[0]
        flag = text(record, "flag")
        real = record["real"]
        imag = record["imag"]
        data = real + 1j * imag
        axis0 = record["axis0"]
        axis1 = record["axis1"]
        axis0_name = text(record, "axis0_name") or "index"
        axis0_unit = text(record, "axis0_unit")
        axis1_name = text(record, "axis1_name") or "index"
        axis1_unit = text(record, "axis1_unit")
        fft_point = int(record["fft_point"])
        pp = int(record["pp"])
        fields = array.dtype.names

    print(f"flag={flag}")
    print(f"shape={data.shape}, fft_point={fft_point}, pp={pp}")
    print(f"fields={fields}")

    def axis_label(name, unit):
        return f"{name} [{unit}]" if unit else name

    def image_geometry(values):
        image_values = values
        extent = None
        xlabel = axis_label(axis1_name, axis1_unit)
        ylabel = axis_label(axis0_name, axis0_unit)
        axes_match = axis0.size == values.shape[0] and axis1.size == values.shape[1]
        if axes_match:
            if "frequency" in axis0_name.lower():
                image_values = values.T
                extent = [axis0[0], axis0[-1], axis1[-1], axis1[0]]
                xlabel = axis_label(axis0_name, axis0_unit)
                ylabel = axis_label(axis1_name, axis1_unit)
            else:
                extent = [axis1[0], axis1[-1], axis0[-1], axis0[0]]
        return image_values, extent, xlabel, ylabel

    is_real = np.all(imag == 0)
    if data.ndim == 1 and is_real:
        x = axis0 if axis0.size == data.size else np.arange(data.size)
        xlabel = f"{axis0_name} [{axis0_unit}]" if axis0_unit else axis0_name
        fig, ax = plt.subplots()
        ax.plot(x, real)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(axis1_name if axis1_name != "index" else "Value")
        ax.grid(True)
    elif data.ndim == 1:
        x = axis0 if axis0.size == data.size else np.arange(data.size)
        xlabel = f"{axis0_name} [{axis0_unit}]" if axis0_unit else axis0_name
        fig, (ax_amp, ax_phase) = plt.subplots(2, 1, sharex=True)
        ax_amp.plot(x, np.abs(data))
        ax_amp.set_ylabel("Amplitude")
        ax_amp.grid(True)
        ax_phase.plot(x, np.angle(data, deg=True))
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_ylim(-180, 180)
        ax_phase.set_yticks(np.arange(-180, 181, 60))
        ax_phase.set_xlabel(xlabel)
        ax_phase.grid(True)
    elif is_real:
        image_data, extent, xlabel, ylabel = image_geometry(real)
        fig, ax = plt.subplots()
        image = ax.imshow(image_data, aspect="auto", extent=extent)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(image, ax=ax)
    else:
        amp_data, extent, xlabel, ylabel = image_geometry(np.abs(data))
        phase_data, phase_extent, _, _ = image_geometry(np.angle(data, deg=True))
        fig, (ax_amp, ax_phase) = plt.subplots(1, 2)
        amp = ax_amp.imshow(amp_data, aspect="auto", extent=extent)
        phase = ax_phase.imshow(
            phase_data,
            aspect="auto",
            extent=phase_extent,
            vmin=-180,
            vmax=180,
        )
        ax_amp.set_xlabel(xlabel)
        ax_phase.set_xlabel(xlabel)
        ax_amp.set_ylabel(ylabel)
        fig.colorbar(amp, ax=ax_amp)
        fig.colorbar(phase, ax=ax_phase, ticks=np.arange(-180, 181, 60))

    fig.tight_layout(pad=0.1)
    if args.output:
        output_stem = args.npz.with_suffix("")
        tsv_path = output_stem.with_suffix(".tsv")
        figure_path = output_stem.with_suffix(f".{args.ext.lstrip(chr(46))}")
        amplitude = np.abs(data)
        phase_deg = np.angle(data, deg=True)
        with tsv_path.open("w", encoding="utf-8") as stream:
            if data.ndim == 1:
                stream.write(f"{axis0_name}\treal\timag\tamplitude\tphase_deg\n")
                for x, value, amp, phase_value in zip(axis0, data, amplitude, phase_deg):
                    stream.write(f"{x:.12g}\t{value.real:.12g}\t{value.imag:.12g}\t{amp:.12g}\t{phase_value:.12g}\n")
            else:
                stream.write(f"{axis0_name}\t{axis1_name}\treal\timag\tamplitude\tphase_deg\n")
                for row, y in enumerate(axis0):
                    for column, x in enumerate(axis1):
                        value = data[row, column]
                        stream.write(f"{y:.12g}\t{x:.12g}\t{value.real:.12g}\t{value.imag:.12g}\t{amplitude[row, column]:.12g}\t{phase_deg[row, column]:.12g}\n")
        fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.02)
        print(f"TSV saved to: {tsv_path}")
        print(f"Figure saved to: {figure_path}")
    if not args.nofig:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
