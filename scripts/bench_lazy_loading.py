#!/usr/bin/env python3
"""Benchmark: lazy vs eager loading for HDF5 files of varying sizes.

Generates test files with different array sizes, then compares:
  1. Eager load (preload="*") — reads everything from disk
  2. Lazy load (default) — defers array reads until access
  3. Metadata-only load — skips arrays entirely
  4. Selective preload — eagerly loads one field, leaves rest lazy

Usage:
    pixi run python scripts/bench_lazy_loading.py
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import versionable
from versionable import Versionable

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@dataclass
class SmallArrayFile(Versionable, version=1):
    """One small array + scalar metadata."""

    label: str
    config: npt.NDArray[np.float64]


@dataclass
class MultiArrayFile(Versionable, version=1):
    """Multiple arrays of equal size + scalar metadata."""

    label: str
    timestamps: npt.NDArray[np.float64]
    channel_a: npt.NDArray[np.float64]
    channel_b: npt.NDArray[np.float64]
    channel_c: npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BENCH_DIR = Path("_bench_lazy")


def _timeit(fn: Callable[[], object], *, runs: int = 5) -> float:
    """Return the median wall-clock time in seconds over *runs* calls."""
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[runs // 2]


_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB")
_BYTES_PER_UNIT = 1024


def _humanSize(nBytes: int) -> str:
    size = float(nBytes)
    for unit in _SIZE_UNITS[:-1]:
        if size < _BYTES_PER_UNIT:
            return f"{size:.1f} {unit}"
        size /= _BYTES_PER_UNIT
    return f"{size:.1f} {_SIZE_UNITS[-1]}"


def _generateFiles() -> list[tuple[str, Path, type, int]]:
    """Create test HDF5 files and return (description, path, cls, fileSize)."""
    BENCH_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    files: list[tuple[str, Path, type, int]] = []

    # --- Small file (~10 KB) ---
    nSmall = 1_250  # 1250 float64 ≈ 10 KB
    p = BENCH_DIR / "small_10kb.h5"
    obj = SmallArrayFile(label="small", config=rng.random(nSmall))
    versionable.save(obj, p)
    files.append(("Small (~10 KB, 1 array)", p, SmallArrayFile, p.stat().st_size))

    # --- Medium file (~10 MB, single array) ---
    nMed = 1_250_000  # 1.25M float64 ≈ 10 MB
    p = BENCH_DIR / "medium_10mb.h5"
    obj = SmallArrayFile(label="medium", config=rng.random(nMed))
    versionable.save(obj, p)
    files.append(("Medium (~10 MB, 1 array)", p, SmallArrayFile, p.stat().st_size))

    # --- Medium file (~10 MB, 4 arrays) ---
    nPerChannel = nMed // 4
    p = BENCH_DIR / "medium_10mb_multi.h5"
    obj = MultiArrayFile(
        label="medium-multi",
        timestamps=rng.random(nPerChannel),
        channel_a=rng.random(nPerChannel),
        channel_b=rng.random(nPerChannel),
        channel_c=rng.random(nPerChannel),
    )
    versionable.save(obj, p)
    files.append(("Medium (~10 MB, 4 arrays)", p, MultiArrayFile, p.stat().st_size))

    # --- Large file (~100 MB, 4 arrays) ---
    nLarge = 12_500_000 // 4  # total ≈ 100 MB across 4 arrays
    p = BENCH_DIR / "large_100mb_multi.h5"
    obj = MultiArrayFile(
        label="large-multi",
        timestamps=rng.random(nLarge),
        channel_a=rng.random(nLarge),
        channel_b=rng.random(nLarge),
        channel_c=rng.random(nLarge),
    )
    versionable.save(obj, p)
    files.append(("Large (~100 MB, 4 arrays)", p, MultiArrayFile, p.stat().st_size))

    # --- Extra large file (~1 GB, 4 arrays) ---
    nXL = 125_000_000 // 4  # total ≈ 1 GB across 4 arrays
    p = BENCH_DIR / "xl_1gb_multi.h5"
    obj = MultiArrayFile(
        label="xl-multi",
        timestamps=rng.random(nXL),
        channel_a=rng.random(nXL),
        channel_b=rng.random(nXL),
        channel_c=rng.random(nXL),
    )
    versionable.save(obj, p)
    files.append(("XL (~1 GB, 4 arrays)", p, MultiArrayFile, p.stat().st_size))

    return files


def _runBenchmarks(files: list[tuple[str, Path, type, int]]) -> None:
    """Run loading benchmarks and print a comparison table."""
    print()
    print("=" * 100)
    print(f"{'Scenario':<30} {'File size':>10}  {'Eager':>10}  {'Lazy':>10}  {'Meta-only':>10}  {'Selective':>10}")
    print("=" * 100)

    for desc, path, cls, fileSize in files:
        # 1. Eager: preload="*"
        tEager = _timeit(lambda c=cls, p=path: versionable.load(c, p, preload="*"))

        # 2. Lazy: default (no preload) — access only scalar fields
        def _lazyScalarOnly(c: type = cls, p: Path = path) -> None:
            obj = versionable.load(c, p)
            _ = obj.label  # only touch the scalar

        tLazy = _timeit(_lazyScalarOnly)

        # 3. Metadata-only
        tMeta = _timeit(lambda c=cls, p=path: versionable.load(c, p, metadataOnly=True))

        # 4. Selective preload (first array only)
        preloadField = ["timestamps"] if cls is MultiArrayFile else ["config"]
        tSelective = _timeit(lambda c=cls, p=path, f=preloadField: versionable.load(c, p, preload=f))

        print(
            f"{desc:<30} {_humanSize(fileSize):>10}"
            f"  {tEager * 1000:>8.1f}ms"
            f"  {tLazy * 1000:>8.1f}ms"
            f"  {tMeta * 1000:>8.1f}ms"
            f"  {tSelective * 1000:>8.1f}ms"
        )

    print()
    print("Eager     = preload='*' (all arrays read from disk)")
    print("Lazy      = default load, only scalar fields accessed")
    print("Meta-only = metadataOnly=True (arrays raise on access)")
    print("Selective = preload one array, rest lazy")
    print()


def _runLazyAccessBenchmark(files: list[tuple[str, Path, type, int]]) -> None:
    """Show cost of lazy access: load time + first array access time."""
    multiFiles = [(d, p, c, s) for d, p, c, s in files if c is MultiArrayFile]
    if not multiFiles:
        return

    print()
    print("=" * 100)
    print("Lazy load + selective access (MultiArrayFile only)")
    print("=" * 100)
    print(f"{'Scenario':<30} {'Load':>10}  {'1st array':>10}  {'All arrays':>10}  {'vs Eager':>10}")
    print("-" * 90)

    for desc, path, cls, _fileSize in multiFiles:
        # Eager baseline
        tEager = _timeit(lambda c=cls, p=path: versionable.load(c, p, preload="*"))

        # Lazy load (no array access)
        def _lazyLoad(c: type = cls, p: Path = path) -> object:
            return versionable.load(c, p)

        tLoad = _timeit(_lazyLoad)

        # Lazy load + access one array
        def _lazyOne(c: type = cls, p: Path = path) -> None:
            obj = versionable.load(c, p)
            _ = obj.timestamps  # triggers lazy read of one dataset

        tOne = _timeit(_lazyOne)

        # Lazy load + access all arrays
        def _lazyAll(c: type = cls, p: Path = path) -> None:
            obj = versionable.load(c, p)
            _ = obj.timestamps
            _ = obj.channel_a
            _ = obj.channel_b
            _ = obj.channel_c

        tAll = _timeit(_lazyAll)

        speedup = tEager / tAll if tAll > 0 else float("inf")
        print(f"{desc:<30}  {tLoad * 1000:>8.1f}ms  {tOne * 1000:>8.1f}ms  {tAll * 1000:>8.1f}ms  {speedup:>8.2f}x")

    print()
    print("Load       = lazy load, no array access")
    print("1st array  = lazy load + access timestamps only")
    print("All arrays = lazy load + access all 4 arrays")
    print("vs Eager   = eager time / all-arrays time (>1 = lazy is faster)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        print("Generating test files...")
        files = _generateFiles()
        for desc, path, _, fileSize in files:
            print(f"  {desc}: {_humanSize(fileSize)} ({path})")

        _runBenchmarks(files)
        _runLazyAccessBenchmark(files)
    finally:
        if BENCH_DIR.exists():
            shutil.rmtree(BENCH_DIR)
            print("Cleaned up benchmark files.")


# Below was the result on my machine (2026-04-02):
# ---

# Generating test files...
#   Small (~10 KB, 1 array): 17.3 KB (_bench_lazy/small_10kb.h5)
#   Medium (~10 MB, 1 array): 9.0 MB (_bench_lazy/medium_10mb.h5)
#   Medium (~10 MB, 4 arrays): 9.0 MB (_bench_lazy/medium_10mb_multi.h5)
#   Large (~100 MB, 4 arrays): 89.6 MB (_bench_lazy/large_100mb_multi.h5)
#   XL (~1 GB, 4 arrays): 895.6 MB (_bench_lazy/xl_1gb_multi.h5)

# ====================================================================================================
# Scenario                        File size       Eager        Lazy   Meta-only   Selective
# ====================================================================================================
# Small (~10 KB, 1 array)           17.3 KB       0.7ms       0.5ms       0.5ms       0.6ms
# Medium (~10 MB, 1 array)           9.0 MB      14.6ms       0.6ms       0.5ms      15.4ms
# Medium (~10 MB, 4 arrays)          9.0 MB      13.6ms       0.8ms       0.8ms       4.1ms
# Large (~100 MB, 4 arrays)         89.6 MB     129.1ms       0.8ms       0.8ms      33.9ms
# XL (~1 GB, 4 arrays)             895.6 MB    1220.6ms       0.8ms       0.8ms     298.2ms

# Eager     = preload='*' (all arrays read from disk)
# Lazy      = default load, only scalar fields accessed
# Meta-only = metadataOnly=True (arrays raise on access)
# Selective = preload one array, rest lazy


# ====================================================================================================
# Lazy load + selective access (MultiArrayFile only)
# ====================================================================================================
# Scenario                             Load   1st array  All arrays       Total    vs Eager
# ----------------------------------------------------------------------------------------------------
# Medium (~10 MB, 4 arrays)            0.8ms       3.9ms      15.1ms      15.1ms      0.88x
# Large (~100 MB, 4 arrays)            0.8ms      32.5ms     129.5ms     129.5ms      0.97x
# XL (~1 GB, 4 arrays)                 0.8ms     297.7ms    1191.6ms    1191.6ms      1.01x

# Load       = lazy load, no array access
# 1st array  = lazy load + access timestamps only
# All arrays = lazy load + access all 4 arrays
# vs Eager   = eager time / all-arrays time (>1 = lazy is faster)

# Cleaned up benchmark files.
