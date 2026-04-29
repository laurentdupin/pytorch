#!/usr/bin/env python3

import argparse
import json
import math
import sys


def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line"] = line_number
            records.append(record)
    return records


def _shape(record):
    return tuple(record.get("sizes", ()))


def _sample_values(record):
    return record.get("sample_values", ())


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _max_sample_abs_diff(lhs, rhs):
    lhs_values = _sample_values(lhs)
    rhs_values = _sample_values(rhs)
    count = min(len(lhs_values), len(rhs_values))
    max_diff = 0.0
    for index in range(count):
        left = lhs_values[index]
        right = rhs_values[index]
        if not _is_number(left) or not _is_number(right):
            if left != right:
                return math.inf
            continue
        max_diff = max(max_diff, abs(float(left) - float(right)))
    if len(lhs_values) != len(rhs_values):
        return math.inf
    return max_diff


def _range_diff(lhs, rhs):
    fields = ("min", "max", "mean")
    max_diff = 0.0
    for field in fields:
        left = lhs.get(field)
        right = rhs.get(field)
        if not _is_number(left) or not _is_number(right):
            if left != right:
                return math.inf
            continue
        max_diff = max(max_diff, abs(float(left) - float(right)))
    return max_diff


def _format_record(record):
    return (
        f"line={record.get('_line')} op={record.get('op')} "
        f"route={record.get('route')} shape={_shape(record)} "
        f"dtype={record.get('dtype')} hash={record.get('sample_hash')} "
        f"min={record.get('min')} max={record.get('max')} "
        f"mean={record.get('mean')}"
    )


def compare_traces(args):
    vulkan = _load_jsonl(args.vulkan)
    reference = _load_jsonl(args.reference)
    count = min(len(vulkan), len(reference))

    for index in range(count):
        vk = vulkan[index]
        ref = reference[index]
        structural_mismatch = (
            vk.get("op") != ref.get("op")
            or _shape(vk) != _shape(ref)
            or vk.get("dtype") != ref.get("dtype")
        )
        sample_diff = _max_sample_abs_diff(vk, ref)
        range_diff = _range_diff(vk, ref)
        hash_mismatch = vk.get("sample_hash") != ref.get("sample_hash")
        diverged = (
            structural_mismatch
            or sample_diff > args.sample_atol
            or range_diff > args.range_atol
            or (args.compare_hash and hash_mismatch)
        )
        if diverged:
            print("First value trace divergence:")
            print(f"  index={index}")
            print(f"  structural_mismatch={int(structural_mismatch)}")
            print(f"  hash_mismatch={int(hash_mismatch)}")
            print(f"  sample_max_abs_diff={sample_diff}")
            print(f"  range_max_abs_diff={range_diff}")
            print(f"  vulkan:    {_format_record(vk)}")
            print(f"  reference: {_format_record(ref)}")
            return 1

    if len(vulkan) != len(reference):
        print(
            "Trace length mismatch after matching common prefix: "
            f"vulkan={len(vulkan)} reference={len(reference)} common={count}")
        return 1

    print(
        "Value traces match: "
        f"records={count} sample_atol={args.sample_atol} "
        f"range_atol={args.range_atol}")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compare Vulkan value-trace JSONL files by op sequence and samples.")
    parser.add_argument("--vulkan", required=True, help="Vulkan trace JSONL path")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference trace JSONL path, for example CPU/CUDA/DirectML")
    parser.add_argument("--sample-atol", type=float, default=1e-4)
    parser.add_argument("--range-atol", type=float, default=1e-4)
    parser.add_argument(
        "--compare-hash",
        action="store_true",
        help="Also require exact sampled hash equality")
    return compare_traces(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
