"""Summarize spaces between GPU kernel executions in an Nsight SQLite export.

This measures kernel-free time, not whole-GPU idle time: copies and other GPU
operations may run between kernels. It does not diagnose why a gap occurred.
"""

import argparse
import sqlite3
from pathlib import Path


def summarize(path: Path, threshold_ns: int = 10_000) -> tuple[int, int, int, int]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        )
        device_id = None
        busy_until = None
        kernel_count = gap_count = gap_total_ns = largest_gap_ns = 0
        for start, end, current_device_id in rows:
            if device_id is None:
                device_id = current_device_id
            elif current_device_id != device_id:
                raise ValueError("Trace has multiple GPUs; this summary requires one GPU")
            if end < start:
                raise ValueError("Trace contains a kernel ending before it starts")
            kernel_count += 1
            if busy_until is not None and start > busy_until:
                gap_ns = start - busy_until
                if gap_ns >= threshold_ns:
                    gap_count += 1
                    gap_total_ns += gap_ns
                    largest_gap_ns = max(largest_gap_ns, gap_ns)
            busy_until = end if busy_until is None else max(busy_until, end)
        return kernel_count, gap_count, gap_total_ns, largest_gap_ns
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_export", type=Path, help="Nsight Systems SQLite export")
    args = parser.parse_args()
    if not args.sqlite_export.is_file():
        parser.error(f"No SQLite export at {args.sqlite_export}")
    try:
        kernels, gaps, total_ns, largest_ns = summarize(args.sqlite_export)
    except (sqlite3.DatabaseError, ValueError) as error:
        parser.error(str(error))
    print(f"Recorded kernels: {kernels}")
    print(f"Kernel-free gaps >= 10 us: {gaps}")
    print(f"Combined duration of those gaps: {total_ns / 1_000_000:.3f} ms")
    print(f"Longest such gap: {largest_ns / 1_000:.1f} us")


if __name__ == "__main__":
    main()
