from __future__ import annotations

import argparse
import collections
import json
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


def parse_gpu_timestamp_sequence(path: Path):
    rows = []
    if not path or not path.exists():
        return rows

    for line in path.read_text(errors="replace").splitlines():
        parts = parse_kv_line(line)
        name = parts.get("name")
        duration = parts.get("duration_ns")
        if not name or not duration:
            continue
        rows.append(
            {
                "name": name,
                "duration_ns": int(duration),
                "global": parts.get("global", ""),
                "local": parts.get("local", ""),
            }
        )
    return rows


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


def classify_conv_role_from_parts(parts: dict[str, str]) -> str:
    input_shape = [
        int(value)
        for value in parts.get("input", "[]").strip("[]").split("x")
        if value
    ]
    weight_shape = [
        int(value)
        for value in parts.get("weight", "[]").strip("[]").split("x")
        if value
    ]
    stride = [
        int(value)
        for value in parts.get("stride", "[]").strip("[]").split("x")
        if value
    ]
    padding = [
        int(value)
        for value in parts.get("padding", "[]").strip("[]").split("x")
        if value
    ]
    dilation = [
        int(value)
        for value in parts.get("dilation", "[]").strip("[]").split("x")
        if value
    ]
    groups = int(parts.get("groups", "0"))
    n = input_shape[0] if input_shape else 0
    cin = input_shape[1] if len(input_shape) > 1 else 0
    h = input_shape[2] if len(input_shape) > 2 else 0
    w = input_shape[3] if len(input_shape) > 3 else 0
    cout = weight_shape[0] if weight_shape else 0
    kh = weight_shape[2] if len(weight_shape) > 2 else 0
    kw = weight_shape[3] if len(weight_shape) > 3 else 0
    if cin == 3 and cout == 384 and kh == 14 and kw == 14 and stride == [14, 14]:
        return "patch_embed"
    high_channel_384 = n == 1 and cin == 384 and cout == 384 and groups == 1
    small_spatial_decoder = 16 <= h <= 80 and 16 <= w <= 96
    if (
        high_channel_384
        and small_spatial_decoder
        and kh == 1
        and kw == 1
        and stride == [1, 1]
        and padding == [0, 0]
        and dilation == [1, 1]
    ):
        return "decoder_head_pointwise_1x1"
    if (
        high_channel_384
        and small_spatial_decoder
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and padding == [1, 1]
        and dilation == [1, 1]
    ):
        return "decoder_head_3x3_s2p1"
    if (
        high_channel_384
        and small_spatial_decoder
        and kh == 3
        and kw == 3
        and stride == [1, 1]
        and padding == [1, 1]
        and dilation == [1, 1]
    ):
        return "decoder_head_3x3_s1p1"
    if (
        kh == 3
        and kw == 3
        and stride == [1, 1]
        and padding == [1, 1]
        and dilation == [1, 1]
        and groups == 1
    ):
        return "other_3x3_s1p1"
    if kh == 1 and kw == 1 and groups == 1:
        return "other_pointwise_1x1"
    if groups == cin and groups == cout and groups > 1:
        return "depthwise"
    if len(input_shape) >= 4 and input_shape[2] <= 74 and input_shape[3] <= 114:
        return "decoder_head_generic"
    if kh == 3 and kw == 3 and groups == 1:
        return "other_3x3"
    return "other_generic"


def parse_conv_submit_op_hits(path: Path):
    rows = []
    if not path or not path.exists():
        return rows

    for line in path.read_text(errors="replace").splitlines():
        if "op=aten::convolution.submit" not in line:
            continue
        parts = parse_kv_line(line)
        kernel = parts.get("kernel")
        if not kernel:
            continue
        shape = (
            f"input={parts.get('input', '?')} "
            f"output={parts.get('output', '?')} "
            f"weight={parts.get('weight', '?')} "
            f"stride={parts.get('stride', '?')} "
            f"padding={parts.get('padding', '?')} "
            f"dilation={parts.get('dilation', '?')} "
            f"groups={parts.get('groups', '?')}"
        )
        rows.append(
            {
                "kernel": kernel,
                "shape": shape,
                "role": classify_conv_role_from_parts(parts),
            }
        )
    return rows


def conv_gpu_time_by_submit_shape(op_hits_path: Path, gpu_path: Path):
    submits = parse_conv_submit_op_hits(op_hits_path)
    timestamp_queues: dict[str, collections.deque[int]] = collections.defaultdict(
        collections.deque
    )
    for row in parse_gpu_timestamp_sequence(gpu_path):
        name = row["name"]
        if name.startswith("conv2d_buffer") or name.startswith("conv_transpose2d"):
            timestamp_queues[name].append(row["duration_ns"])

    by_kernel = collections.Counter()
    by_shape = collections.Counter()
    by_role = collections.Counter()
    unmatched = collections.Counter()
    for submit in submits:
        queue = timestamp_queues.get(submit["kernel"])
        if not queue:
            unmatched[submit["kernel"]] += 1
            continue
        duration_ns = queue.popleft()
        by_kernel[submit["kernel"]] += duration_ns
        by_shape[submit["shape"]] += duration_ns
        by_role[submit["role"]] += duration_ns
    return by_kernel, by_shape, by_role, unmatched


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
    by_pair_count: collections.Counter[str] = collections.Counter()
    by_pair_shape_bytes: collections.Counter[tuple[str, str]] = collections.Counter()
    large_transformer: collections.Counter[str] = collections.Counter()
    logical_noop = 0
    if not path or not path.exists():
        return (
            by_reason_count,
            by_reason_bytes,
            by_shape_bytes,
            by_pair_bytes,
            by_pair_count,
            by_pair_shape_bytes,
            large_transformer,
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
            f"{parts.get('consumer', 'unknown')} "
            f"producer_role={parts.get('producer_role', 'unknown')} "
            f"consumer_role={parts.get('consumer_role', 'unknown')} "
            f"reason={reason}"
        )
        by_reason_count[reason] += count
        by_reason_bytes[reason] += bytes_
        by_shape_bytes[shape_key] += bytes_
        by_pair_bytes[pair_key] += bytes_
        by_pair_count[pair_key] += count
        by_pair_shape_bytes[(pair_key, shape_key)] += bytes_

        src_sizes = parts.get("src_sizes", "")
        if (
            (
                src_sizes.startswith("[1,")
                and (
                    src_sizes.endswith(",1536]")
                    or src_sizes.endswith(",384]")
                    or src_sizes.endswith(",1152]")
                )
            )
            or (src_sizes.startswith("[6,") and src_sizes.endswith(",64]"))
        ):
            large_transformer[f"shape={src_sizes} {pair_key}"] += bytes_
    return (
        by_reason_count,
        by_reason_bytes,
        by_shape_bytes,
        by_pair_bytes,
        by_pair_count,
        by_pair_shape_bytes,
        large_transformer,
        logical_noop,
    )


def parse_clone_requirements(path: Path):
    by_reason_count: collections.Counter[str] = collections.Counter()
    by_reason_bytes: collections.Counter[str] = collections.Counter()
    by_pair_bytes: collections.Counter[str] = collections.Counter()
    large_mlp: collections.Counter[str] = collections.Counter()
    if not path or not path.exists():
        return by_reason_count, by_reason_bytes, by_pair_bytes, large_mlp

    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("clone_requirement"):
            continue
        parts = parse_kv_line(line)
        reason = parts.get("reason", "unknown")
        count = int(parts.get("count", "0"))
        bytes_ = int(parts.get("bytes", "0"))
        pair_key = (
            f"{parts.get('producer', 'unknown')} -> "
            f"{parts.get('consumer', 'unknown')} "
            f"producer_role={parts.get('producer_role', 'unknown')} "
            f"consumer_role={parts.get('consumer_role', 'unknown')} "
            f"sizes={parts.get('sizes', '?')}"
        )
        by_reason_count[reason] += count
        by_reason_bytes[reason] += bytes_
        by_pair_bytes[f"reason={reason} {pair_key}"] += bytes_
        sizes = parts.get("sizes", "")
        if sizes.startswith("[1,") and sizes.endswith(",1536]"):
            large_mlp[f"reason={reason} {pair_key}"] += bytes_
    return by_reason_count, by_reason_bytes, by_pair_bytes, large_mlp


def conv_aggregate_lines(path: Path) -> list[str]:
    if not path or not path.exists():
        return []
    text = path.read_text(errors="replace")
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        counters = data.get("vulkan_debug_counters", {})
        rows = counters.get("conv_aggregate_snapshot", [])
        return [row for row in rows if isinstance(row, str)]
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("conv_aggregate")
    ]


def parse_conv_aggregate(path: Path):
    rows = []
    by_kernel_count: collections.Counter[str] = collections.Counter()
    by_kernel_bytes: collections.Counter[str] = collections.Counter()
    by_shape_count: collections.Counter[str] = collections.Counter()
    by_shape_bytes: collections.Counter[str] = collections.Counter()
    by_role_count: collections.Counter[str] = collections.Counter()
    by_role_bytes: collections.Counter[str] = collections.Counter()
    by_method_count: collections.Counter[str] = collections.Counter()
    by_method_bytes: collections.Counter[str] = collections.Counter()
    role_buckets: dict[str, collections.Counter[str]] = {
        "patch_embed": collections.Counter(),
        "decoder_head_pointwise_1x1": collections.Counter(),
        "decoder_head_3x3_s2p1": collections.Counter(),
        "decoder_head_3x3_s1p1": collections.Counter(),
        "decoder_head_generic": collections.Counter(),
        "other_pointwise_1x1": collections.Counter(),
        "other_3x3_s1p1": collections.Counter(),
        "other_3x3": collections.Counter(),
        "other_generic": collections.Counter(),
    }
    for line in conv_aggregate_lines(path):
        parts = parse_kv_line(line)
        if not parts:
            continue
        count = int(parts.get("count", "0"))
        input_bytes = int(parts.get("input_bytes", "0"))
        output_bytes = int(parts.get("output_bytes", "0"))
        weight_bytes = int(parts.get("weight_bytes", "0"))
        bytes_ = input_bytes + output_bytes + weight_bytes
        kernel = parts.get("kernel", "unknown")
        role = parts.get("role", "unknown")
        method = (
            f"selected={parts.get('selected', 'unknown')} "
            f"reject={parts.get('reject', 'unknown')}"
        )
        shape = (
            f"input={parts.get('input', '?')} "
            f"weight={parts.get('weight', '?')} "
            f"stride={parts.get('stride', '?')} "
            f"padding={parts.get('padding', '?')} "
            f"dilation={parts.get('dilation', '?')} "
            f"groups={parts.get('groups', '?')}"
        )
        rows.append(
            {
                "count": count,
                "bytes": bytes_,
                "kernel": kernel,
                "role": role,
                "method": method,
                "shape": shape,
            }
        )
        by_kernel_count[kernel] += count
        by_kernel_bytes[kernel] += bytes_
        by_shape_count[shape] += count
        by_shape_bytes[shape] += bytes_
        by_role_count[role] += count
        by_role_bytes[role] += bytes_
        by_method_count[method] += count
        by_method_bytes[method] += bytes_
        if role in role_buckets:
            role_buckets[role][shape] += bytes_
    return {
        "rows": rows,
        "by_kernel_count": by_kernel_count,
        "by_kernel_bytes": by_kernel_bytes,
        "by_shape_count": by_shape_count,
        "by_shape_bytes": by_shape_bytes,
        "by_role_count": by_role_count,
        "by_role_bytes": by_role_bytes,
        "by_method_count": by_method_count,
        "by_method_bytes": by_method_bytes,
        "role_buckets": role_buckets,
    }


def linear_aggregate_lines(path: Path) -> list[str]:
    if not path or not path.exists():
        return []
    text = path.read_text(errors="replace")
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        counters = data.get("vulkan_debug_counters", {})
        rows = counters.get("linear_aggregate_snapshot", [])
        return [row for row in rows if isinstance(row, str)]
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("linear_aggregate")
    ]


def linear_submit_kernel_to_gpu_name(kernel: str) -> str:
    mapping = {
        "aten::linear.buffer_float_tiled_bias_vec2_gelu": (
            "mm_buffer_float_tiled_bias_vec2_gelu"
        ),
        "aten::linear.buffer_float_tiled_bias_vec2": (
            "mm_buffer_float_tiled_bias_vec2"
        ),
        "aten::linear.buffer_float_tiled_bias_gelu": (
            "mm_buffer_float_tiled_bias_gelu"
        ),
        "aten::linear.buffer_float_tiled_bias": "mm_buffer_float_tiled_bias",
        "aten::linear.buffer_float_tiled": "mm_buffer_float_tiled",
        "aten::linear.buffer_float_bias_gelu": "mm_buffer_float_bias_gelu",
        "aten::linear.buffer_float_bias": "mm_buffer_float_bias",
        "aten::linear.buffer_float": "mm_buffer_float",
    }
    return mapping.get(kernel, kernel)


def parse_linear_aggregate(path: Path):
    rows = []
    by_role_count: collections.Counter[str] = collections.Counter()
    by_role_bytes: collections.Counter[str] = collections.Counter()
    by_shape_count: collections.Counter[str] = collections.Counter()
    by_shape_bytes: collections.Counter[str] = collections.Counter()
    by_kernel_count: collections.Counter[str] = collections.Counter()
    by_kernel_bytes: collections.Counter[str] = collections.Counter()
    by_role_shape_count: collections.Counter[str] = collections.Counter()
    by_role_shape_bytes: collections.Counter[str] = collections.Counter()
    for line in linear_aggregate_lines(path):
        parts = parse_kv_line(line)
        if not parts:
            continue
        count = int(parts.get("count", "0"))
        input_bytes = int(parts.get("input_bytes", "0"))
        weight_bytes = int(parts.get("weight_bytes", "0"))
        output_bytes = int(parts.get("output_bytes", "0"))
        bytes_ = input_bytes + weight_bytes + output_bytes
        role = parts.get("role", "unknown")
        kernel = parts.get("submit_kernel", parts.get("kernel", "unknown"))
        gpu_kernel = linear_submit_kernel_to_gpu_name(kernel)
        shape = (
            f"m={parts.get('m', '?')} k={parts.get('k', '?')} "
            f"n={parts.get('n', '?')}"
        )
        role_shape = f"role={role} {shape}"
        rows.append(
            {
                "count": count,
                "bytes": bytes_,
                "role": role,
                "kernel": kernel,
                "gpu_kernel": gpu_kernel,
                "shape": shape,
                "role_shape": role_shape,
                "input_direct": parts.get("input_direct", "?"),
                "output_direct": parts.get("output_direct", "?"),
                "weight_packed": parts.get("weight_packed", "?"),
                "input_dtype": parts.get("input_dtype", "?"),
                "weight_dtype": parts.get("weight_dtype", "?"),
                "output_dtype": parts.get("output_dtype", "?"),
            }
        )
        by_role_count[role] += count
        by_role_bytes[role] += bytes_
        by_shape_count[shape] += count
        by_shape_bytes[shape] += bytes_
        by_kernel_count[kernel] += count
        by_kernel_bytes[kernel] += bytes_
        by_role_shape_count[role_shape] += count
        by_role_shape_bytes[role_shape] += bytes_
    return {
        "rows": rows,
        "by_role_count": by_role_count,
        "by_role_bytes": by_role_bytes,
        "by_shape_count": by_shape_count,
        "by_shape_bytes": by_shape_bytes,
        "by_kernel_count": by_kernel_count,
        "by_kernel_bytes": by_kernel_bytes,
        "by_role_shape_count": by_role_shape_count,
        "by_role_shape_bytes": by_role_shape_bytes,
    }


def estimate_linear_gpu_from_aggregate(rows, gpu_by_name):
    by_role = collections.Counter()
    by_shape = collections.Counter()
    by_kernel = collections.Counter()
    by_role_shape = collections.Counter()
    for row in rows:
        gpu = gpu_by_name.get(row["gpu_kernel"])
        if not gpu or not gpu["count"]:
            continue
        estimated_ns = int(row["count"] * (gpu["total_ns"] / gpu["count"]))
        by_role[row["role"]] += estimated_ns
        by_shape[row["shape"]] += estimated_ns
        by_kernel[row["kernel"]] += estimated_ns
        by_role_shape[row["role_shape"]] += estimated_ns
    return by_role, by_shape, by_kernel, by_role_shape


def estimate_conv_gpu_from_aggregate(rows, gpu_by_name):
    by_kernel = collections.Counter()
    by_shape = collections.Counter()
    by_role = collections.Counter()
    for row in rows:
        gpu = gpu_by_name.get(row["kernel"])
        if not gpu or not gpu["count"]:
            continue
        estimated_ns = int(row["count"] * (gpu["total_ns"] / gpu["count"]))
        by_kernel[row["kernel"]] += estimated_ns
        by_shape[row["shape"]] += estimated_ns
        by_role[row["role"]] += estimated_ns
    return by_kernel, by_shape, by_role


def print_time_counter(title: str, counts: collections.Counter[str], limit: int) -> None:
    total_all = sum(counts.values())
    print(f"\n{title}")
    for rank, (label, total_ns) in enumerate(counts.most_common(limit), 1):
        share = (100.0 * total_ns / total_all) if total_all else 0.0
        print(
            f"{rank:02d} {label} total_ms={total_ns / 1e6:.3f} "
            f"share={share:.2f}%"
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


def print_pair_shape_bytes(
    title: str, counts: collections.Counter[tuple[str, str]], limit: int
) -> None:
    print(f"\n{title}")
    for rank, ((pair, shape), bytes_) in enumerate(counts.most_common(limit), 1):
        print(
            f"{rank:02d} pair={pair} shape={shape} "
            f"bytes={bytes_} mb={bytes_ / 1048576.0:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-timestamps", type=Path)
    parser.add_argument("--op-hits", type=Path)
    parser.add_argument("--conv-plan", type=Path)
    parser.add_argument("--conv-aggregate", type=Path)
    parser.add_argument("--linear-plan", type=Path)
    parser.add_argument("--linear-aggregate", type=Path)
    parser.add_argument("--attention-plan", type=Path)
    parser.add_argument("--buffer-copy-log", type=Path)
    parser.add_argument("--buffer-copy-aggregate", type=Path)
    parser.add_argument("--clone-requirement", type=Path)
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
    if args.conv_aggregate:
        conv = parse_conv_aggregate(args.conv_aggregate)
        print_counter("conv_by_kernel", conv["by_kernel_count"], args.top)
        print_bytes_counter(
            "conv_by_kernel_estimated_bytes", conv["by_kernel_bytes"], args.top
        )
        print_counter("conv_by_shape", conv["by_shape_count"], args.top)
        print_bytes_counter(
            "conv_by_shape_estimated_bytes", conv["by_shape_bytes"], args.top
        )
        print_counter("conv_by_role", conv["by_role_count"], args.top)
        print_bytes_counter(
            "conv_by_role_estimated_bytes", conv["by_role_bytes"], args.top
        )
        print_counter("conv_by_method", conv["by_method_count"], args.top)
        print_bytes_counter(
            "conv_by_method_estimated_bytes", conv["by_method_bytes"], args.top
        )
        for role, bucket in conv["role_buckets"].items():
            print_bytes_counter(f"conv_{role}", bucket, args.top)
        if args.gpu_timestamps:
            if args.op_hits:
                by_kernel, by_shape, by_role, unmatched = (
                    conv_gpu_time_by_submit_shape(args.op_hits, args.gpu_timestamps)
                )
                print_time_counter("conv_by_kernel_gpu_time", by_kernel, args.top)
                print_time_counter("conv_by_shape_gpu_time", by_shape, args.top)
                print_time_counter("conv_by_role_gpu_time", by_role, args.top)
                print_counter(
                    "conv_submit_rows_without_timestamp", unmatched, args.top
                )
            else:
                gpu_by_name = parse_gpu_timestamps(args.gpu_timestamps)
                by_kernel, by_shape, by_role = estimate_conv_gpu_from_aggregate(
                    conv["rows"], gpu_by_name
                )
                print_time_counter(
                    "conv_by_kernel_estimated_gpu_time", by_kernel, args.top
                )
                print_time_counter(
                    "conv_by_shape_estimated_gpu_time", by_shape, args.top
                )
                print_time_counter(
                    "conv_by_role_estimated_gpu_time", by_role, args.top
                )
    if args.linear_plan:
        counts, shapes, rejects = parse_plan(args.linear_plan)
        print_counter("linear_plan_decisions", counts, args.top)
        print_counter("linear_plan_shapes", shapes, args.top)
        print_counter("linear_plan_rejects", rejects, args.top)
    if args.linear_aggregate:
        linear = parse_linear_aggregate(args.linear_aggregate)
        print_counter("linear_by_role", linear["by_role_count"], args.top)
        print_bytes_counter(
            "linear_by_role_estimated_bytes", linear["by_role_bytes"], args.top
        )
        print_counter("linear_by_shape", linear["by_shape_count"], args.top)
        print_bytes_counter(
            "linear_by_shape_estimated_bytes", linear["by_shape_bytes"], args.top
        )
        print_counter("linear_by_kernel", linear["by_kernel_count"], args.top)
        print_bytes_counter(
            "linear_by_kernel_estimated_bytes",
            linear["by_kernel_bytes"],
            args.top,
        )
        print_counter(
            "linear_by_role_and_shape",
            linear["by_role_shape_count"],
            args.top,
        )
        print_bytes_counter(
            "linear_by_role_and_shape_estimated_bytes",
            linear["by_role_shape_bytes"],
            args.top,
        )
        if args.gpu_timestamps:
            by_role, by_shape, by_kernel, by_role_shape = (
                estimate_linear_gpu_from_aggregate(
                    linear["rows"], parse_gpu_timestamps(args.gpu_timestamps)
                )
            )
            print_time_counter("linear_by_role_gpu_time", by_role, args.top)
            print_time_counter("linear_by_shape_gpu_time", by_shape, args.top)
            print_time_counter("linear_by_kernel_gpu_time", by_kernel, args.top)
            print_time_counter(
                "linear_by_role_and_shape_gpu_time", by_role_shape, args.top
            )
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
            by_pair_count,
            by_pair_shape_bytes,
            large_transformer,
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
        print_counter(
            "buffer_copy_aggregate_producer_consumer_by_count",
            by_pair_count,
            args.top,
        )
        print_pair_shape_bytes(
            "buffer_copy_aggregate_shapes_within_producer_consumer_by_bytes",
            by_pair_shape_bytes,
            args.top,
        )
        print_bytes_counter(
            "large_transformer_copy_suspects_by_bytes",
            large_transformer,
            args.top,
        )
        print(f"\nbuffer_copy_aggregate_logical_noop_count count={logical_noop}")
    if args.clone_requirement:
        (
            by_reason_count,
            by_reason_bytes,
            by_pair_bytes,
            large_mlp,
        ) = parse_clone_requirements(args.clone_requirement)
        print_counter(
            "clone_requirements_by_count", by_reason_count, args.top
        )
        print_bytes_counter(
            "clone_requirements_by_bytes", by_reason_bytes, args.top
        )
        print_bytes_counter(
            "clone_requirements_producer_consumer_by_bytes",
            by_pair_bytes,
            args.top,
        )
        print_bytes_counter(
            "large_mlp_clone_requirements_by_bytes", large_mlp, args.top
        )
    if args.sync_log:
        print_counter("sync_events", parse_sync_log(args.sync_log), args.top)


if __name__ == "__main__":
    main()
