#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


RMSE_RE = re.compile(r"\(([^)]+)\)")
SCREENSHOT_RE = re.compile(r"screenshot_step_(\d+)\.png$")
DEFAULT_COMPARE_SIZE = (320, 200)


@dataclass(frozen=True)
class ScreenshotDiffThresholds:
    rmse: float = 0.03
    phash: float = 5.0
    changed_fraction: float = 0.01


@dataclass(frozen=True)
class ScreenshotDiffResult:
    first_path: str
    second_path: str
    rmse: float
    phash: float
    changed_pixels: int
    total_pixels: int
    changed_fraction: float
    significance_score: float
    is_significant: bool
    triggered_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_step_number(path: Path) -> int:
    match = SCREENSHOT_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected screenshot name: {path}")
    return int(match.group(1))


def sorted_screenshot_paths(folder: Path) -> list[Path]:
    screenshots = [path for path in folder.iterdir() if path.is_file() and SCREENSHOT_RE.fullmatch(path.name)]
    return sorted(screenshots, key=extract_step_number)


def compare_screenshots(
    first_path: str | Path,
    second_path: str | Path,
    thresholds: ScreenshotDiffThresholds | None = None,
    compare_size: tuple[int, int] = DEFAULT_COMPARE_SIZE,
) -> ScreenshotDiffResult:
    first = Path(first_path)
    second = Path(second_path)
    if not first.exists():
        raise FileNotFoundError(first)
    if not second.exists():
        raise FileNotFoundError(second)

    if shutil.which("compare") is None:
        raise RuntimeError("ImageMagick `compare` command is required but was not found in PATH.")

    thresholds = thresholds or ScreenshotDiffThresholds()
    resize_geometry = f"{compare_size[0]}x{compare_size[1]}!"
    total_pixels = compare_size[0] * compare_size[1]

    rmse = _run_rmse_metric(first, second, resize_geometry)
    phash = _run_simple_metric("PHASH", first, second, resize_geometry)
    changed_pixels = int(round(_run_simple_metric("AE", first, second, resize_geometry)))
    changed_fraction = changed_pixels / total_pixels

    triggered_metrics = []
    if rmse >= thresholds.rmse:
        triggered_metrics.append("rmse")
    if phash >= thresholds.phash:
        triggered_metrics.append("phash")
    if changed_fraction >= thresholds.changed_fraction:
        triggered_metrics.append("changed_fraction")

    significance_score = max(
        rmse / thresholds.rmse if thresholds.rmse > 0 else 0.0,
        phash / thresholds.phash if thresholds.phash > 0 else 0.0,
        changed_fraction / thresholds.changed_fraction if thresholds.changed_fraction > 0 else 0.0,
    )

    return ScreenshotDiffResult(
        first_path=str(first),
        second_path=str(second),
        rmse=rmse,
        phash=phash,
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        changed_fraction=changed_fraction,
        significance_score=significance_score,
        is_significant=bool(triggered_metrics),
        triggered_metrics=tuple(triggered_metrics),
    )


def _run_rmse_metric(first: Path, second: Path, resize_geometry: str) -> float:
    output = _run_compare_command("RMSE", first, second, resize_geometry)
    match = RMSE_RE.search(output)
    if not match:
        raise RuntimeError(f"Could not parse RMSE output from ImageMagick compare: {output!r}")
    return float(match.group(1))


def _run_simple_metric(metric: str, first: Path, second: Path, resize_geometry: str) -> float:
    output = _run_compare_command(metric, first, second, resize_geometry).strip()
    if not output:
        raise RuntimeError(f"Could not parse {metric} output from ImageMagick compare.")
    return float(output)


def _run_compare_command(metric: str, first: Path, second: Path, resize_geometry: str) -> str:
    command = [
        "compare",
        "-colorspace",
        "Gray",
        "-resize",
        resize_geometry,
        "-metric",
        metric,
        str(first),
        str(second),
        "null:",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            "ImageMagick compare failed with "
            f"exit code {completed.returncode}: {(completed.stderr or completed.stdout).strip()}"
        )
    return completed.stderr or completed.stdout
