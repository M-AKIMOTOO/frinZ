#!/usr/bin/env python3
"""Convert frinZ --inband text output to structured JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

VALUE_KEYS = [
    "amp_percent",
    "snr",
    "phase_deg",
    "noise_percent",
    "res_delay_sample",
    "res_rate_hz",
]

CHANNEL_KEYS = ["band", "band_start_mhz", "band_end_mhz", "center_mhz"]


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


def unique_or_none(rows: list[dict[str, Any]], key: str) -> Any | None:
    values = {row[key] for row in rows}
    return next(iter(values)) if len(values) == 1 else None


def compact_float(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def build_structured_json(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = unique_or_none(rows, "source")
    label = unique_or_none(rows, "label")
    length_s = unique_or_none(rows, "length_s")

    channels_by_band: dict[int, dict[str, Any]] = {}
    for row in rows:
        band = row["band"]
        channel = {key: compact_float(row[key]) for key in CHANNEL_KEYS}
        if band in channels_by_band and {
            key: channels_by_band[band][key] for key in CHANNEL_KEYS
        } != channel:
            raise ValueError(f"channel definition changed for band {band}")
        channels_by_band[band] = channel

    grouped_by_time: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_time[(row["epoch"], row["mjd"])].append(row)
    time_keys = sorted(grouped_by_time, key=lambda item: item[1])
    time_axis = [{"epoch": epoch, "mjd": mjd} for epoch, mjd in time_keys]

    row_by_time_band: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in rows:
        row_by_time_band[(row["epoch"], row["mjd"], row["band"])] = row

    channels = []
    for band in sorted(channels_by_band):
        channel = dict(channels_by_band[band])
        channel["series"] = {}
        for key in VALUE_KEYS:
            channel["series"][key] = []
            for epoch, mjd in time_keys:
                row = row_by_time_band.get((epoch, mjd, band))
                channel["series"][key].append(row[key] if row is not None else None)
        channels.append(channel)

    metadata = dict(metadata)
    if source is not None:
        metadata["source"] = source
    if label is not None:
        metadata["label"] = label
    if length_s is not None:
        metadata["length_s"] = compact_float(length_s)
    metadata["time_count"] = len(time_axis)
    metadata["channel_count"] = len(channels)
    metadata["row_count"] = len(rows)

    return {
        "format": "frinZ_inband_json_v3_channel_series",
        "metadata": metadata,
        "time_axis": time_axis,
        "value_keys": VALUE_KEYS,
        "channels": channels,
    }


def build_flat_json(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = [
        "epoch",
        "label",
        "source",
        "band",
        "band_start_mhz",
        "band_end_mhz",
        "center_mhz",
        "length_s",
        *VALUE_KEYS,
        "mjd",
    ]
    return {
        "format": "frinZ_inband_json_v1_flat",
        "metadata": metadata,
        "columns": columns,
        "rows": [[row[key] for key in columns] for row in rows],
    }


def default_output(path: Path) -> Path:
    return path.with_suffix(".json") if path.suffix else path.with_name(path.name + ".json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert frinZ --inband text output to channel-series JSON.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input *_inband.txt")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON path [default: input .json]")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON without indentation")
    parser.add_argument("--flat", action="store_true", help="Use old flat columns+rows JSON layout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, rows = parse_inband_txt(args.input)
    data = build_flat_json(metadata, rows) if args.flat else build_structured_json(metadata, rows)
    output = args.output or default_output(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        if args.compact:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"JSON saved to: {output}")
    print(f"format: {data['format']}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
