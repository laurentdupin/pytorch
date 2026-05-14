from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path


KEY_VALUE_RE = re.compile(r"([^ =]+)=([^ ]*)")


def parse_kv_line(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in KEY_VALUE_RE.finditer(line)}


def parse_gpu_timestamps(path: Path):
    by_name = collections.defaultdict(
        lambda: {"count": 0, "total_ns": 0, "max_ns": 0}
    )
    if not path or not path.exists():
        return by_name

    for line in path.read_text(errors="replace").splitlines():
        parts = parse_kv_line(line)
        name = parts.get("name")
        duration = parts.get("duration_ns")
        if not name or not duration:
            continue
        duration_ns = int(duration)
        row = by_name[name]
        row["count"] += 1
        row["total_ns"] += duration_ns
        row["max_ns"] = max(row["max_ns"], duration_ns)
    return by_name


def parse_op_hits(path: Path) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    if not path or not path.exists():
        return counts

    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        parts = parse_kv_line(line)
        label = parts.get("op") or line.strip()
        kernel = parts.get("kernel")
        caller = parts.get("caller")
        if kernel:
            label = f"{label} kernel={kernel}"
        if caller:
            label = f"{label} caller={caller}"
        counts[label] += 1
    return counts


def parse_plan(path: Path, selected_key: str = "selected"):
    counts: collections.Counter[str] = collections.Counter()
    shapes: collections.Counter[str] = collections.Counter()
    rejects: collections.Counter[str] = collections.Counter()
    if not path or not path.exists():
        return counts, shapes, rejects

    for line in path.read_text(errors="replace").splitlines():
        parts = parse_kv_line(line)
        if not parts:
            continue
        selected = parts.get(selected_key, "unknown")
        reject = parts.get("reject", "unknown")
        counts[f"selected={selected} reject={reject}"] += 1
        rejects[reject] += 1
        if "batch_heads" in parts:
            shape = (
                f"batch_heads={parts.get('batch_heads')} "
                f"target_len={parts.get('target_len')} "
                f"source_len={parts.get('source_len')} "
                f"head_dim={parts.get('head_dim')} "
                f"value_dim={parts.get('value_dim')} "
                f"query_tile={parts.get('query_tile')}"
            )
        elif "m" in parts:
            shape = (
                f"m={parts.get('m')} k={parts.get('k')} n={parts.get('n')} "
                f"m_tail={parts.get('m_tail')}"
            )
        else:
            shape = (
                f"n={parts.get('n')} cin={parts.get('cin')} "
                f"h={parts.get('h')} w={parts.get('w')} "
                f"cout={parts.get('cout')} kh={parts.get('kh')} "
                f"kw={parts.get('kw')} pointwise={parts.get('pointwise')}"
            )
        shapes[shape] += 1
    return counts, shapes, rejects


def parse_cpu_timeline_summary(path: Path):
    rows = []
    if not path or not path.exists():
        return rows

    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("event="):
            continue
        parts = parse_kv_line(line)
        event = parts.get("event")
        if not event:
            continue
        rows.append(
            {
                "event": event,
                "count": int(parts.get("count", "0")),
                "submitted": int(parts.get("submitted", "0")),
                "total_us": int(parts.get("total_us", "0")),
                "avg_us": int(parts.get("avg_us", "0")),
                "max_us": int(parts.get("max_us", "0")),
                "raw": line,
            }
        )
    rows.sort(key=lambda row: row["total_us"], reverse=True)
    return rows


def parse_sync_log(path: Path) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    if not path or not path.exists():
        return counts

    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        label = line.split(":", 1)[0].strip()
        counts[label] += 1
    return counts


def parse_buffer_copy_log(path: Path):
    by_reason_count: collections.Counter[str] = collections.Counter()
    by_reason_bytes: collections.Counter[str] = collections.Counter()
    by_shape_bytes: collections.Counter[str] = collections.Counter()
    by_pair_bytes: collections.Counter[str] = collections.Counter()
    logical_noop = 0
    if not path or not path.exists():
        return (
            by_reason_count,
            by_reason_bytes,
            by_shape_bytes,
            by_pair_bytes,
            logical_noop,
        )

    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("buffer_copy"):
            continue
        parts = parse_kv_line(line)
        reason = parts.get("reason", "unknown")
        bytes_ = int(parts.get("bytes", "0"))
        if parts.get("logical_noop") == "1":
            logical_noop += 1
        shape_key = (
            f"src={parts.get('src_sizes', '?')} dst={parts.get('dst_sizes', '?')} "
            f"src_strides={parts.get('src_strides', '?')} "
            f"dst_strides={parts.get('dst_strides', '?')} "
            f"dtype={parts.get('dtype', '?')}"
        )
        pair_key = (
            f"{parts.get('producer', 'unknown')} -> "
            f"{parts.get('consumer', 'unknown')} reason={reason}"
        )
        by_reason_count[reason] += 1
        by_reason_bytes[reason] += bytes_
        by_shape_bytes[shape_key] += bytes_
        by_pair_bytes[pair_key] += bytes_
    return (
        by_reason_count,
        by_reason_bytes,
        by_shape_bytes,
        by_pair_bytes,
        logical_noop,
    )


def parse_buffer_copy_aggregate(path: Path):
    by_reason_count: collections.Counter[str] = collections.Counter()
    by_reason_bytes: collections.Counter[str] = collections.Counter()
    by_shape_bytes: collections.Counter[str] = collections.Counter()
    by_pair_bytes: collections.Counter[str] = collections.Counter()
    logical_noop = 0
    if not path or not path.exists():
        return (
            by_reason_count,
            by_reason_bytes,
            by_shape_bytes,
            by_pair_bytes,
            logical_noop,
        )

    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("buffer_copy_aggregate"):
            continue
        parts = parse_kv_line(line)
        reason = parts.get("reason", "unknown")
        count = int(parts.get("count", "0"))
        bytes_ = int(parts.get("bytes", "0"))
        logical_noop += int(parts.get("logical_noop", "0"))
        shape_key = (
            f"src={parts.get('src_sizes', '?')} dst={parts.get('dst_sizes', '?')} "
            f"src_strides={parts.get('src_strides', '?')} "
            f"dst_strides={parts.get('dst_strides', '?')} "
            f"dtype={parts.get('dtype', '?')}"
        )
        pair_key = (
            f"{parts.get('producer', 'unknown')} -> "
            f"{parts.get('consumer', 'unknown')} reason={reason}"
        )
        by_reason_count[reason] += count
        by_reason_bytes[reason] += bytes_
        by_shape_bytes[shape_key] += bytes_
        by_pair_bytes[pair_key] += bytes_
    return (
        by_reason_count,
        by_reason_bytes,
        by_shape_bytes,
        by_pair_bytes,
        logical_noop,
    )


def print_top_gpu(by_name, limit: int) -> None:
    total_all = sum(data["total_ns"] for data in by_name.values())
    rows = []
    for name, data in by_name.items():
        count = data["count"]
        total = data["total_ns"]
        rows.append((total, count, data["max_ns"], name))
    rows.sort(reverse=True)

    print("top_gpu_kernels")
    for rank, (total, count, max_ns, name) in enumerate(rows[:limit], 1):
        avg = total / max(count, 1)
        share = (100.0 * total / total_all) if total_all else 0.0
        print(
            f"{rank:02d} {name} count={count} total_ms={total / 1e6:.3f} "
            f"avg_us={avg / 1e3:.3f} max_us={max_ns / 1e3:.3f} "
            f"share={share:.2f}%"
        )


def print_counter(title: str, counts: collections.Counter[str], limit: int) -> None:
    print(f"\n{title}")
    for rank, (label, count) in enumerate(counts.most_common(limit), 1):
        print(f"{rank:02d} {label} count={count}")


def print_cpu(rows, limit: int) -> None:
    print("\ntop_cpu_timeline")
    for rank, row in enumerate(rows[:limit], 1):
        print(
            f"{rank:02d} {row['event']} count={row['count']} "
            f"submitted={row['submitted']} total_ms={row['total_us'] / 1000:.3f} "
            f"avg_us={row['avg_us']} max_us={row['max_us']}"
        )


def print_bytes_counter(
    title: str, counts: collections.Counter[str], limit: int
) -> None:
    print(f"\n{title}")
    for rank, (label, bytes_) in enumerate(counts.most_common(limit), 1):
        print(f"{rank:02d} {label} bytes={bytes_} mb={bytes_ / 1048576.0:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-timestamps", type=Path)
    parser.add_argument("--op-hits", type=Path)
    parser.add_argument("--conv-plan", type=Path)
    parser.add_argument("--linear-plan", type=Path)
    parser.add_argument("--attention-plan", type=Path)
    parser.add_argument("--buffer-copy-log", type=Path)
    parser.add_argument("--buffer-copy-aggregate", type=Path)
    parser.add_argument("--sync-log", type=Path)
    parser.add_argument("--cpu-timeline-summary", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if args.gpu_timestamps:
        print_top_gpu(parse_gpu_timestamps(args.gpu_timestamps), args.top)
    if args.cpu_timeline_summary:
        print_cpu(parse_cpu_timeline_summary(args.cpu_timeline_summary), args.top)
    if args.op_hits:
        print_counter("op_hit_counts", parse_op_hits(args.op_hits), args.top)
    if args.conv_plan:
        counts, shapes, rejects = parse_plan(args.conv_plan)
        print_counter("conv_plan_decisions", counts, args.top)
        print_counter("conv_plan_shapes", shapes, args.top)
        print_counter("conv_plan_rejects", rejects, args.top)
    if args.linear_plan:
        counts, shapes, rejects = parse_plan(args.linear_plan)
        print_counter("linear_plan_decisions", counts, args.top)
        print_counter("linear_plan_shapes", shapes, args.top)
        print_counter("linear_plan_rejects", rejects, args.top)
    if args.attention_plan:
        counts, shapes, rejects = parse_plan(args.attention_plan)
        print_counter("attention_plan_decisions", counts, args.top)
        print_counter("attention_plan_shapes", shapes, args.top)
        print_counter("attention_plan_rejects", rejects, args.top)
    if args.buffer_copy_log:
        (
            by_reason_count,
            by_reason_bytes,
            by_shape_bytes,
            by_pair_bytes,
            logical_noop,
        ) = parse_buffer_copy_log(args.buffer_copy_log)
        print_counter("buffer_copy_reasons_by_count", by_reason_count, args.top)
        print_bytes_counter(
            "buffer_copy_reasons_by_bytes", by_reason_bytes, args.top
        )
        print_bytes_counter("buffer_copy_shapes_by_bytes", by_shape_bytes, args.top)
        print_bytes_counter(
            "buffer_copy_producer_consumer_by_bytes", by_pair_bytes, args.top
        )
        print(f"\nbuffer_copy_logical_noop_count count={logical_noop}")
    if args.buffer_copy_aggregate:
        (
            by_reason_count,
            by_reason_bytes,
            by_shape_bytes,
            by_pair_bytes,
            logical_noop,
        ) = parse_buffer_copy_aggregate(args.buffer_copy_aggregate)
        print_counter(
            "buffer_copy_aggregate_reasons_by_count", by_reason_count, args.top
        )
        print_bytes_counter(
            "buffer_copy_aggregate_reasons_by_bytes", by_reason_bytes, args.top
        )
        print_bytes_counter(
            "buffer_copy_aggregate_shapes_by_bytes", by_shape_bytes, args.top
        )
        print_bytes_counter(
            "buffer_copy_aggregate_producer_consumer_by_bytes",
            by_pair_bytes,
            args.top,
        )
        print(f"\nbuffer_copy_aggregate_logical_noop_count count={logical_noop}")
    if args.sync_log:
        print_counter("sync_events", parse_sync_log(args.sync_log), args.top)


if __name__ == "__main__":
    main()
