#!/usr/bin/env python3
"""Generate SVG speedup and efficiency charts from benchmark CSV output."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speedup and efficiency SVG charts from results.csv."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="results.csv",
        help="Benchmark CSV file produced by scripts/run_benchmarks.sh.",
    )
    parser.add_argument(
        "--output-dir",
        default="charts",
        help="Directory where SVG charts will be written.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "OK" or row.get("mode") == "sequential":
                continue
            rows.append(
                {
                    "n": int(row["n"]),
                    "threads": int(row["threads"]),
                    "scan_type": row["scan_type"],
                    "mode": row["mode"],
                    "speedup": float(row["speedup"]),
                    "efficiency": float(row["efficiency"]),
                }
            )
    return rows


def nice_number(value: float) -> float:
    if value <= 0:
        return 1.0

    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def y_ticks(max_value: float) -> tuple[float, list[float]]:
    upper = nice_number(max_value * 1.15)
    step = nice_number(upper / 5.0)
    upper = step * math.ceil(upper / step)
    return upper, [step * i for i in range(0, int(round(upper / step)) + 1)]


def format_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}K" if n % 1_000 == 0 else f"{n / 1_000:.1f}K"
    return str(n)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 14,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = "#111827",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def render_chart(
    rows: list[dict[str, object]],
    metric: str,
    mode: str,
    scan_type: str,
    output_path: Path,
) -> None:
    subset = [
        row
        for row in rows
        if row["mode"] == mode and row["scan_type"] == scan_type
    ]
    if not subset:
        return

    threads = sorted({int(row["threads"]) for row in subset})
    sizes = sorted({int(row["n"]) for row in subset})
    values = {
        (int(row["n"]), int(row["threads"])): float(row[metric])
        for row in subset
    }

    width = 900
    height = 560
    left = 82
    right = 34
    top = 78
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_value = max(values.values())
    y_max, ticks = y_ticks(max(max_value, 1.0 if metric == "speedup" else 0.25))

    def x_pos(thread: int) -> float:
        if len(threads) == 1:
            return left + plot_w / 2
        index = threads.index(thread)
        return left + (plot_w * index / (len(threads) - 1))

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max * plot_h)

    metric_title = "Speedup" if metric == "speedup" else "Efficiency"
    title = f"{metric_title}: {mode} {scan_type}"

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, title, size=24, weight="700"),
        svg_text(width / 2, 58, "Generated from results.csv", size=13, fill="#4b5563"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in ticks:
        y = y_pos(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(svg_text(left - 12, y + 4, f"{tick:.2g}", size=12, anchor="end"))

    for thread in threads:
        x = x_pos(thread)
        elements.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" '
            f'y2="{top + plot_h + 6}" stroke="#111827" stroke-width="1"/>'
        )
        elements.append(svg_text(x, top + plot_h + 28, str(thread), size=13))

    elements.append(svg_text(left + plot_w / 2, height - 22, "OpenMP threads", size=14))
    elements.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="14" fill="#111827" '
        f'transform="rotate(-90 22 {top + plot_h / 2:.1f})">{metric_title}</text>'
    )

    for size_index, n in enumerate(sizes):
        color = COLORS[size_index % len(COLORS)]
        points = [
            (x_pos(thread), y_pos(values[(n, thread)]))
            for thread in threads
            if (n, thread) in values
        ]
        if len(points) < 1:
            continue

        path_data = " ".join(
            f"{'M' if index == 0 else 'L'} {x:.1f} {y:.1f}"
            for index, (x, y) in enumerate(points)
        )
        elements.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" '
            'stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in points:
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.4" fill="{color}" '
                'stroke="#ffffff" stroke-width="1.4"/>'
            )

        legend_x = left + 16 + (size_index % 4) * 155
        legend_y = top - 14 + (size_index // 4) * 22
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            svg_text(
                legend_x + 32,
                legend_y + 4,
                f"n={format_size(n)}",
                size=12,
                anchor="start",
                fill="#374151",
            )
        )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No successful benchmark rows found in {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    modes = sorted({str(row["mode"]) for row in rows})
    scan_types = sorted({str(row["scan_type"]) for row in rows})

    for mode in modes:
        for scan_type in scan_types:
            for metric in ("speedup", "efficiency"):
                output_path = output_dir / f"{metric}_{mode}_{scan_type}.svg"
                render_chart(rows, metric, mode, scan_type, output_path)
                if output_path.exists():
                    generated.append(output_path)

    for path in generated:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
