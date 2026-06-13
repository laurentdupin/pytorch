#!/usr/bin/env python3

import argparse
import json
import re
import sys


EXPECTED_CONTRACT_NAME = "ChannelCatContract"
EXPECTED_FAMILY = "Rank4Dim1BufferView"
EXPECTED_MATCHER = {
    "tensor_info": "ChannelCatTensorInfo",
    "reference_index": 0,
    "per_input_same_as_reference": [
        "dtype",
        "rank",
        "batch",
        "height",
        "width",
    ],
    "per_input_required_flags": [
        "is_vulkan",
        "is_contiguous",
        "has_buffer_storage",
        "supports_buffer_compute",
    ],
    "channel_axis": "channels",
    "aggregate": {
        "field": "channels",
        "result_name": "total_channels",
        "min": 1,
        "max_from_bounds": "channels.max_total",
        "multiple_of_from_bounds": "channels.multiple_of",
    },
}
SCALAR_TYPE_BY_DTYPE = {
    "float32": "at::kFloat",
    "int64": "at::kLong",
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _require_keys(mapping, keys, context):
    missing = sorted(key for key in keys if key not in mapping)
    _require(not missing, f"{context} missing required keys: {missing}")


def _require_non_empty_string(mapping, key, context):
    value = mapping.get(key)
    _require(isinstance(value, str) and value != "", f"{context}.{key} invalid")


def _cpp_string(value):
    return json.dumps(value)


def _cpp_bool(value):
    return "true" if value else "false"


def _cpp_identifier_fragment(value):
    parts = re.findall(r"[A-Za-z0-9]+", value)
    _require(parts, f"cannot build C++ identifier from {value!r}")
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _cpp_lower_identifier(value):
    raw_parts = re.findall(r"[A-Za-z0-9]+", value)
    parts = []
    for raw_part in raw_parts:
        parts.extend(
            re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+", raw_part)
        )
    _require(parts, f"cannot build C++ identifier from {value!r}")
    return "_".join(part.lower() for part in parts)


def _load_spec(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_bool(value, context):
    _require(isinstance(value, bool), f"{context} must be boolean")


def _validate_int(value, context):
    _require(isinstance(value, int), f"{context} must be integer")


def _validate_channel_cat_spec(spec):
    _require(
        spec.get("contract_name") == EXPECTED_CONTRACT_NAME,
        "expected ChannelCatContract spec",
    )
    _require(
        spec.get("family") == EXPECTED_FAMILY,
        "expected Rank4Dim1BufferView family",
    )
    _require_keys(
        spec,
        (
            "tuple_id",
            "writer_op",
            "route_label",
            "metadata",
            "shape_envelope",
            "bounds",
            "matcher",
        ),
        "ChannelCatContract spec",
    )
    for key in ("tuple_id", "writer_op", "route_label"):
        _require_non_empty_string(spec, key, "ChannelCatContract spec")

    envelope = _shape_envelope(spec, "multi_input_rank4_channel_cat")
    metadata = envelope["metadata"]
    _require(isinstance(metadata, dict), "metadata must be an object")
    _require_keys(
        metadata,
        (
            "evidence_id",
            "guard_id",
            "fallback_policy",
            "materialization_policy",
        ),
        "ChannelCatContract metadata",
    )
    for key in metadata:
        _require_non_empty_string(metadata, key, "ChannelCatContract metadata")
    _require(spec["metadata"] == metadata, "metadata must match shape_envelope")

    bounds = envelope["bounds"]
    _require(spec["bounds"] == bounds, "bounds must match shape_envelope")
    _require(isinstance(bounds, dict), "spec bounds must be an object")
    _require_keys(
        bounds,
        (
            "dtype",
            "rank",
            "dim",
            "input_count",
            "batch",
            "channels",
            "height",
            "width",
            "requires_vulkan",
            "requires_contiguous",
            "requires_buffer_storage",
            "requires_buffer_compute",
        ),
        "ChannelCatContract bounds",
    )
    _require(bounds["dtype"] in SCALAR_TYPE_BY_DTYPE, "unsupported dtype")
    for key in ("rank", "dim", "batch"):
        _validate_int(bounds[key], f"bounds.{key}")
    for key in (
        "requires_vulkan",
        "requires_contiguous",
        "requires_buffer_storage",
        "requires_buffer_compute",
    ):
        _validate_bool(bounds[key], f"bounds.{key}")

    channels = bounds["channels"]
    _require(isinstance(channels, dict), "bounds.channels must be an object")
    _require_keys(
        channels,
        ("min", "max_per_input", "multiple_of", "max_total"),
        "ChannelCatContract channels",
    )
    for key in channels:
        _validate_int(channels[key], f"bounds.channels.{key}")

    for key in ("input_count", "height", "width"):
        value = bounds[key]
        _require(isinstance(value, dict), f"bounds.{key} must be an object")
        _require_keys(value, ("min", "max"), f"ChannelCatContract {key}")
        _validate_int(value["min"], f"bounds.{key}.min")
        _validate_int(value["max"], f"bounds.{key}.max")

    _require(spec["matcher"] == EXPECTED_MATCHER, "unexpected matcher schema")
    _require(
        "total_channels" in envelope["aggregate_bounds"],
        "shape_envelope.aggregate_bounds.total_channels missing",
    )


def _validate_bound_pair(value, context):
    _require(isinstance(value, dict), f"{context} must be an object")
    _require_keys(value, ("min", "max"), context)
    _validate_int(value["min"], f"{context}.min")
    _validate_int(value["max"], f"{context}.max")


def _validate_contract_metadata(metadata, context):
    _require(isinstance(metadata, dict), f"{context} must be an object")
    _require_keys(
        metadata,
        (
            "evidence_id",
            "guard_id",
            "fallback_policy",
            "materialization_policy",
        ),
        context,
    )
    for key in metadata:
        _require_non_empty_string(metadata, key, context)


def _shape_envelope(spec, expected_role):
    envelope = spec.get("shape_envelope")
    _require(isinstance(envelope, dict), "shape_envelope must be an object")
    _require(envelope.get("version") == 1, "shape_envelope.version must be 1")
    _require(
        envelope.get("role") == expected_role,
        f"expected ShapeEnvelope role {expected_role}",
    )
    return envelope


def generate_channel_cat_header(spec, source_name):
    _validate_channel_cat_spec(spec)
    envelope = _shape_envelope(spec, "multi_input_rank4_channel_cat")
    metadata = envelope["metadata"]
    bounds = envelope["bounds"]
    channels = bounds["channels"]
    aggregate = envelope["aggregate_bounds"]["total_channels"]

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
        "#include <ATen/core/ScalarType.h>",
        "#include <cstdint>",
        "",
        "namespace at {",
        "namespace native {",
        "namespace vulkan {",
        "namespace ops {",
        "namespace utils {",
        "namespace generated {",
        "",
        f"constexpr const char* kChannelCatContractName = {_cpp_string(spec['contract_name'])};",
        (
            "constexpr const char* kChannelCatRank4Dim1BufferViewFamilyName = "
            f"{_cpp_string(spec['family'])};"
        ),
        (
            "constexpr const char* kChannelCatRank4Dim1BufferViewTupleId = "
            f"{_cpp_string(spec['tuple_id'])};"
        ),
        (
            "constexpr const char* kChannelCatRank4Dim1BufferViewWriterOp = "
            f"{_cpp_string(spec['writer_op'])};"
        ),
        (
            "constexpr const char* kChannelCatRank4Dim1BufferViewRouteLabel = "
            f"{_cpp_string(spec['route_label'])};"
        ),
        "",
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MinInputs = "
            f"{bounds['input_count']['min']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MaxInputs = "
            f"{bounds['input_count']['max']};"
        ),
        f"constexpr std::int64_t kChannelCatRank4Dim1Rank = {bounds['rank']};",
        f"constexpr std::int64_t kChannelCatRank4Dim1Dim = {bounds['dim']};",
        f"constexpr std::int64_t kChannelCatRank4Dim1Batch = {bounds['batch']};",
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MinInputChannels = "
            f"{channels['min']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MaxInputChannels = "
            f"{channels['max_per_input']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1ChannelMultiple = "
            f"{channels['multiple_of']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MinTotalChannels = "
            f"{aggregate['min']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MaxTotalChannels = "
            f"{channels['max_total']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MinHeight = "
            f"{bounds['height']['min']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MaxHeight = "
            f"{bounds['height']['max']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MinWidth = "
            f"{bounds['width']['min']};"
        ),
        (
            "constexpr std::int64_t kChannelCatRank4Dim1MaxWidth = "
            f"{bounds['width']['max']};"
        ),
        (
            "constexpr bool kChannelCatRank4Dim1RequiresVulkan = "
            f"{_cpp_bool(bounds['requires_vulkan'])};"
        ),
        (
            "constexpr bool kChannelCatRank4Dim1RequiresContiguous = "
            f"{_cpp_bool(bounds['requires_contiguous'])};"
        ),
        (
            "constexpr bool kChannelCatRank4Dim1RequiresBufferStorage = "
            f"{_cpp_bool(bounds['requires_buffer_storage'])};"
        ),
        (
            "constexpr bool kChannelCatRank4Dim1RequiresBufferCompute = "
            f"{_cpp_bool(bounds['requires_buffer_compute'])};"
        ),
        "",
        "struct ChannelCatRank4Dim1BufferViewSpec final {",
        "  const char* contract_name;",
        "  const char* family_name;",
        "  const char* tuple_id;",
        "  const char* writer_op;",
        "  const char* route_label;",
        "  const char* evidence_id;",
        "  const char* guard_id;",
        "  const char* fallback_policy;",
        "  const char* materialization_policy;",
        "  at::ScalarType dtype;",
        "  std::int64_t rank;",
        "  std::int64_t dim;",
        "  std::int64_t min_inputs;",
        "  std::int64_t max_inputs;",
        "  std::int64_t batch;",
        "  std::int64_t min_input_channels;",
        "  std::int64_t max_input_channels;",
        "  std::int64_t channel_multiple;",
        "  std::int64_t min_total_channels;",
        "  std::int64_t max_total_channels;",
        "  std::int64_t min_height;",
        "  std::int64_t max_height;",
        "  std::int64_t min_width;",
        "  std::int64_t max_width;",
        "  bool requires_vulkan;",
        "  bool requires_contiguous;",
        "  bool requires_buffer_storage;",
        "  bool requires_buffer_compute;",
        "};",
        "",
        "constexpr ChannelCatRank4Dim1BufferViewSpec",
        "    kChannelCatRank4Dim1BufferViewSpec = {",
        "        kChannelCatContractName,",
        "        kChannelCatRank4Dim1BufferViewFamilyName,",
        "        kChannelCatRank4Dim1BufferViewTupleId,",
        "        kChannelCatRank4Dim1BufferViewWriterOp,",
        "        kChannelCatRank4Dim1BufferViewRouteLabel,",
        f"        {_cpp_string(metadata['evidence_id'])},",
        f"        {_cpp_string(metadata['guard_id'])},",
        f"        {_cpp_string(metadata['fallback_policy'])},",
        f"        {_cpp_string(metadata['materialization_policy'])},",
        f"        {SCALAR_TYPE_BY_DTYPE[bounds['dtype']]},",
        "        kChannelCatRank4Dim1Rank,",
        "        kChannelCatRank4Dim1Dim,",
        "        kChannelCatRank4Dim1MinInputs,",
        "        kChannelCatRank4Dim1MaxInputs,",
        "        kChannelCatRank4Dim1Batch,",
        "        kChannelCatRank4Dim1MinInputChannels,",
        "        kChannelCatRank4Dim1MaxInputChannels,",
        "        kChannelCatRank4Dim1ChannelMultiple,",
        "        kChannelCatRank4Dim1MinTotalChannels,",
        "        kChannelCatRank4Dim1MaxTotalChannels,",
        "        kChannelCatRank4Dim1MinHeight,",
        "        kChannelCatRank4Dim1MaxHeight,",
        "        kChannelCatRank4Dim1MinWidth,",
        "        kChannelCatRank4Dim1MaxWidth,",
        "        kChannelCatRank4Dim1RequiresVulkan,",
        "        kChannelCatRank4Dim1RequiresContiguous,",
        "        kChannelCatRank4Dim1RequiresBufferStorage,",
        "        kChannelCatRank4Dim1RequiresBufferCompute};",
        "",
        "constexpr bool channel_cat_input_count_in_bounds(",
        "    const ChannelCatRank4Dim1BufferViewSpec& spec,",
        "    const std::int64_t input_count) {",
        "  return input_count >= spec.min_inputs && input_count <= spec.max_inputs;",
        "}",
        "",
        "inline bool channel_cat_reference_in_bounds(",
        "    const ChannelCatRank4Dim1BufferViewSpec& spec,",
        "    const ChannelCatTensorInfo& reference) {",
        "  return (!spec.requires_vulkan || reference.is_vulkan) &&",
        "      reference.dtype == spec.dtype && reference.rank == spec.rank &&",
        "      reference.batch == spec.batch &&",
        "      (!spec.requires_contiguous || reference.is_contiguous) &&",
        "      reference.height >= spec.min_height &&",
        "      reference.height <= spec.max_height &&",
        "      reference.width >= spec.min_width && reference.width <= spec.max_width;",
        "}",
        "",
        "inline bool channel_cat_input_in_bounds(",
        "    const ChannelCatRank4Dim1BufferViewSpec& spec,",
        "    const ChannelCatTensorInfo& reference,",
        "    const ChannelCatTensorInfo& tensor) {",
        "  return (!spec.requires_vulkan || tensor.is_vulkan) &&",
        "      tensor.dtype == reference.dtype && tensor.rank == reference.rank &&",
        "      tensor.batch == reference.batch &&",
        "      tensor.height == reference.height && tensor.width == reference.width &&",
        "      (!spec.requires_contiguous || tensor.is_contiguous) &&",
        "      (!spec.requires_buffer_storage || tensor.has_buffer_storage) &&",
        "      (!spec.requires_buffer_compute || tensor.supports_buffer_compute) &&",
        "      tensor.channels >= spec.min_input_channels &&",
        "      tensor.channels <= spec.max_input_channels &&",
        "      tensor.channels % spec.channel_multiple == 0;",
        "}",
        "",
        "constexpr bool channel_cat_total_channels_in_bounds(",
        "    const ChannelCatRank4Dim1BufferViewSpec& spec,",
        "    const std::int64_t total_channels) {",
        "  return total_channels >= spec.min_total_channels &&",
        "      total_channels <= spec.max_total_channels &&",
        "      total_channels % spec.channel_multiple == 0;",
        "}",
        "",
        "} // namespace generated",
        "} // namespace utils",
        "} // namespace ops",
        "} // namespace vulkan",
        "} // namespace native",
        "} // namespace at",
        "",
    ]
    return "\n".join(lines)


def _validate_generic_shape_envelope_spec(spec):
    _require_keys(
        spec,
        (
            "contract_name",
            "family",
            "tuple_id",
            "writer_op",
            "route_label",
            "metadata",
            "shape_envelope",
            "bounds",
        ),
        "ShapeEnvelope contract spec",
    )
    for key in ("contract_name", "family", "tuple_id", "writer_op", "route_label"):
        _require_non_empty_string(spec, key, "ShapeEnvelope contract spec")

    envelope = spec["shape_envelope"]
    _require(isinstance(envelope, dict), "shape_envelope must be an object")
    _require(envelope.get("version") == 1, "shape_envelope.version must be 1")
    _require_non_empty_string(envelope, "role", "shape_envelope")

    metadata = envelope.get("metadata")
    _validate_contract_metadata(metadata, "ShapeEnvelope metadata")
    _require(spec["metadata"] == metadata, "metadata must match shape_envelope")

    bounds = envelope.get("bounds")
    _require(spec["bounds"] == bounds, "bounds must match shape_envelope")
    _require(isinstance(bounds, dict), "ShapeEnvelope bounds must be an object")
    _require_keys(
        bounds,
        (
            "dtype",
            "rank",
            "ops",
            "alpha",
            "requires_vulkan",
            "requires_buffer_storage",
            "requires_buffer_compute",
        ),
        "ShapeEnvelope bounds",
    )
    _require(bounds["dtype"] in SCALAR_TYPE_BY_DTYPE, "unsupported dtype")
    _validate_bound_pair(bounds["rank"], "ShapeEnvelope bounds.rank")
    ops = bounds["ops"]
    _require(isinstance(ops, list) and ops, "ShapeEnvelope bounds.ops invalid")
    for index, op in enumerate(ops):
        _require(isinstance(op, str) and op != "", f"bounds.ops[{index}] invalid")
    _require(bounds["alpha"] == 1, "only alpha == 1 is supported by v0")
    for key in (
        "requires_vulkan",
        "requires_buffer_storage",
        "requires_buffer_compute",
    ):
        _validate_bool(bounds[key], f"bounds.{key}")

    attributes = envelope.get("attributes")
    _require(isinstance(attributes, dict), "shape_envelope.attributes invalid")
    _require_keys(
        attributes,
        ("op", "alpha", "inplace", "has_out"),
        "ShapeEnvelope attributes",
    )
    _require(attributes["op"].get("values") == ops, "attribute ops must match bounds")
    _require(
        attributes["alpha"].get("values") == [bounds["alpha"]],
        "attribute alpha must match bounds",
    )
    for key in ("inplace", "has_out"):
        values = attributes[key].get("values")
        _require(
            isinstance(values, list) and len(values) == 1 and isinstance(values[0], bool),
            f"attribute {key} must have one boolean value",
        )

    layout = envelope.get("layout")
    _require(isinstance(layout, dict), "shape_envelope.layout invalid")
    for key in (
        "requires_vulkan",
        "requires_buffer_storage",
        "requires_buffer_compute",
    ):
        _require(layout.get(key) == bounds[key], f"layout.{key} must match bounds")


def _simple_bounds_shape_envelope_fields(bounds):
    dtype_fields = []
    int_fields = []
    list_int_fields = []
    range_fields = []
    bool_fields = []
    unsupported = []
    for key, value in bounds.items():
        if isinstance(value, bool):
            bool_fields.append(key)
        elif isinstance(value, str) and key.endswith("_dtype"):
            dtype_fields.append(key)
        elif isinstance(value, int):
            int_fields.append(key)
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            list_int_fields.append(key)
        elif (
            isinstance(value, dict)
            and set(value) == {"min", "max"}
            and isinstance(value["min"], int)
            and isinstance(value["max"], int)
        ):
            range_fields.append(key)
        else:
            unsupported.append(key)
    if unsupported:
        return None
    if not dtype_fields or not range_fields:
        return None
    return {
        "dtype": dtype_fields,
        "int": int_fields,
        "list_int": list_int_fields,
        "range": range_fields,
        "bool": bool_fields,
    }


def _singular_field_name(field):
    return field[:-1] if field.endswith("s") else field


def _validate_generic_simple_bounds_shape_envelope_spec(spec):
    _require_keys(
        spec,
        (
            "contract_name",
            "family",
            "tuple_id",
            "writer_op",
            "route_label",
            "metadata",
            "shape_envelope",
            "bounds",
        ),
        "ShapeEnvelope simple-bounds contract spec",
    )
    for key in ("contract_name", "family", "tuple_id", "writer_op", "route_label"):
        _require_non_empty_string(
            spec, key, "ShapeEnvelope simple-bounds contract spec"
        )

    envelope = spec["shape_envelope"]
    _require(isinstance(envelope, dict), "shape_envelope must be an object")
    _require(envelope.get("version") == 1, "shape_envelope.version must be 1")
    _require_non_empty_string(envelope, "role", "shape_envelope")

    metadata = envelope.get("metadata")
    _validate_contract_metadata(metadata, "ShapeEnvelope metadata")
    _require(spec["metadata"] == metadata, "metadata must match shape_envelope")

    bounds = envelope.get("bounds")
    _require(spec["bounds"] == bounds, "bounds must match shape_envelope")
    _require(isinstance(bounds, dict), "ShapeEnvelope bounds must be an object")
    fields = _simple_bounds_shape_envelope_fields(bounds)
    _require(fields is not None, "unsupported simple ShapeEnvelope bounds")

    for key in fields["dtype"]:
        _require(bounds[key] in SCALAR_TYPE_BY_DTYPE, f"unsupported {key}")
    for key in fields["int"]:
        _validate_int(bounds[key], f"bounds.{key}")
    for key in fields["list_int"]:
        values = bounds[key]
        _require(values, f"bounds.{key} must not be empty")
        for index, value in enumerate(values):
            _validate_int(value, f"bounds.{key}[{index}]")
    for key in fields["range"]:
        _validate_bound_pair(bounds[key], f"bounds.{key}")
    for key in fields["bool"]:
        _validate_bool(bounds[key], f"bounds.{key}")
        attributes = envelope.get("attributes", {})
        if key in attributes:
            _require(
                attributes[key].get("values") == [bounds[key]],
                f"attribute {key} must match bounds",
            )
    return fields


def generate_generic_simple_bounds_shape_envelope_header(spec, source_name):
    fields = _validate_generic_simple_bounds_shape_envelope_spec(spec)
    envelope = spec["shape_envelope"]
    metadata = envelope["metadata"]
    bounds = envelope["bounds"]
    contract_prefix = _cpp_identifier_fragment(
        spec["contract_name"].removesuffix("Contract")
    )
    row_prefix = contract_prefix + _cpp_identifier_fragment(spec["family"])
    contract_func_prefix = _cpp_lower_identifier(
        spec["contract_name"].removesuffix("Contract")
    )
    role_func_prefix = _cpp_lower_identifier(envelope["role"])

    field_struct_lines = []
    initializer_lines = []
    constant_lines = []
    option_params = []
    option_checks = []
    range_params = []
    range_checks = []

    for key in fields["dtype"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  at::ScalarType {key};")
        initializer_lines.append(f"        k{row_prefix}{suffix},")
        option_params.append((f"const at::ScalarType {key}", key))
        option_checks.append(f"{key} == spec.{key}")
        constant_lines.append(
            f"constexpr at::ScalarType k{row_prefix}{suffix} = "
            f"{SCALAR_TYPE_BY_DTYPE[bounds[key]]};"
        )

    for key in fields["int"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  std::int64_t {key};")
        initializer_lines.append(f"        k{row_prefix}{suffix},")
        option_params.append((f"const std::int64_t {key}", key))
        option_checks.append(f"{key} == spec.{key}")
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}{suffix} = {bounds[key]};"
        )

    list_rank_helpers = []
    for key in fields["list_int"]:
        singular = _singular_field_name(key)
        singular_suffix = _cpp_identifier_fragment(singular)
        helper_name = (
            f"{contract_func_prefix}_{_cpp_lower_identifier(singular)}_in_bounds"
        )
        helper_checks = []
        for index, value in enumerate(bounds[key], start=1):
            field_name = f"{singular}_{index}"
            const_name = f"k{row_prefix}{singular_suffix}{index}"
            field_struct_lines.append(f"  std::int64_t {field_name};")
            initializer_lines.append(f"        {const_name},")
            constant_lines.append(
                f"constexpr std::int64_t {const_name} = {value};"
            )
            helper_checks.append(f"{singular} == spec.{field_name}")
        option_params.append((f"const std::int64_t {singular}", singular))
        option_checks.append(f"{helper_name}(spec, {singular})")
        helper_body = " ||\n      ".join(helper_checks)
        list_rank_helpers.extend(
            [
                f"constexpr bool {helper_name}(",
                f"    const {row_prefix}Spec& spec,",
                f"    const std::int64_t {singular}) {{",
                f"  return {helper_body};",
                "}",
                "",
            ]
        )

    for key in fields["range"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  std::int64_t min_{key};")
        field_struct_lines.append(f"  std::int64_t max_{key};")
        initializer_lines.append(f"        k{row_prefix}Min{suffix},")
        initializer_lines.append(f"        k{row_prefix}Max{suffix},")
        range_params.append((f"const std::int64_t {key}", key))
        range_checks.append(f"{key} <= spec.max_{key}")
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}Min{suffix} = "
            f"{bounds[key]['min']};"
        )
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}Max{suffix} = "
            f"{bounds[key]['max']};"
        )

    for key in fields["bool"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  bool {key};")
        initializer_lines.append(f"        k{row_prefix}{suffix},")
        option_params.append((f"const bool {key}", key))
        option_checks.append(f"{key} == spec.{key}")
        constant_lines.append(
            f"constexpr bool k{row_prefix}{suffix} = {_cpp_bool(bounds[key])};"
        )

    initializer_lines[-1] = initializer_lines[-1].rstrip(",") + "};"
    option_signature = []
    for param, _ in option_params:
        option_signature.append(f"    {param},")
    option_signature[-1] = option_signature[-1].rstrip(",") + ") {"
    option_body = " &&\n      ".join(option_checks)

    range_signature = []
    for param, _ in range_params:
        range_signature.append(f"    {param},")
    range_signature[-1] = range_signature[-1].rstrip(",") + ") {"
    range_body = " &&\n      ".join(range_checks)

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
        "#include <ATen/core/ScalarType.h>",
        "#include <cstdint>",
        "",
        "namespace at {",
        "namespace native {",
        "namespace vulkan {",
        "namespace ops {",
        "namespace utils {",
        "namespace generated {",
        "",
        f"constexpr const char* k{contract_prefix}ContractName = {_cpp_string(spec['contract_name'])};",
        f"constexpr const char* k{row_prefix}FamilyName = {_cpp_string(spec['family'])};",
        f"constexpr const char* k{row_prefix}TupleId = {_cpp_string(spec['tuple_id'])};",
        f"constexpr const char* k{row_prefix}WriterOp = {_cpp_string(spec['writer_op'])};",
        f"constexpr const char* k{row_prefix}RouteLabel = {_cpp_string(spec['route_label'])};",
        "",
    ]
    lines.extend(constant_lines)
    lines.extend(
        [
            "",
            f"struct {row_prefix}Spec final {{",
            "  const char* contract_name;",
            "  const char* family_name;",
            "  const char* tuple_id;",
            "  const char* writer_op;",
            "  const char* route_label;",
            "  const char* evidence_id;",
            "  const char* guard_id;",
            "  const char* fallback_policy;",
            "  const char* materialization_policy;",
        ]
    )
    lines.extend(field_struct_lines)
    lines.extend(
        [
            "};",
            "",
            f"constexpr {row_prefix}Spec",
            f"    k{row_prefix}Spec = {{",
            f"        k{contract_prefix}ContractName,",
            f"        k{row_prefix}FamilyName,",
            f"        k{row_prefix}TupleId,",
            f"        k{row_prefix}WriterOp,",
            f"        k{row_prefix}RouteLabel,",
            f"        {_cpp_string(metadata['evidence_id'])},",
            f"        {_cpp_string(metadata['guard_id'])},",
            f"        {_cpp_string(metadata['fallback_policy'])},",
            f"        {_cpp_string(metadata['materialization_policy'])},",
        ]
    )
    lines.extend(initializer_lines)
    lines.extend(
        [
            "",
        ]
    )
    lines.extend(list_rank_helpers)
    lines.extend(
        [
            f"constexpr bool {role_func_prefix}_options_match(",
            f"    const {row_prefix}Spec& spec,",
        ]
    )
    lines.extend(option_signature)
    lines.extend(
        [
            f"  return {option_body};",
            "}",
            "",
            f"constexpr bool {role_func_prefix}_in_bounds(",
            f"    const {row_prefix}Spec& spec,",
        ]
    )
    lines.extend(range_signature)
    lines.extend(
        [
            f"  return {range_body};",
            "}",
            "",
            "} // namespace generated",
            "} // namespace utils",
            "} // namespace ops",
            "} // namespace vulkan",
            "} // namespace native",
            "} // namespace at",
            "",
        ]
    )
    return "\n".join(lines)


def generate_generic_shape_envelope_header(spec, source_name):
    bounds = spec.get("shape_envelope", {}).get("bounds", {})
    if _simple_bounds_shape_envelope_fields(bounds) is not None:
        return generate_generic_simple_bounds_shape_envelope_header(
            spec, source_name
        )

    _validate_generic_shape_envelope_spec(spec)
    envelope = spec["shape_envelope"]
    metadata = envelope["metadata"]
    bounds = envelope["bounds"]
    attributes = envelope["attributes"]
    prefix = _cpp_identifier_fragment(spec["contract_name"].removesuffix("Contract"))
    prefix += _cpp_identifier_fragment(spec["family"])
    func_prefix = _cpp_lower_identifier(envelope["role"])

    op_values = bounds["ops"]
    op_constants = []
    op_fields = []
    op_initializers = []
    for index, op in enumerate(op_values):
        op_suffix = _cpp_identifier_fragment(op)
        op_constants.append(
            f"constexpr const char* k{prefix}Op{op_suffix} = {_cpp_string(op)};"
        )
        op_fields.append(f"  const char* op_{index};")
        op_initializers.append(f"        k{prefix}Op{op_suffix},")

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
        "#include <ATen/core/ScalarType.h>",
        "#include <cstdint>",
        "",
        "namespace at {",
        "namespace native {",
        "namespace vulkan {",
        "namespace ops {",
        "namespace utils {",
        "namespace generated {",
        "",
        f"constexpr const char* k{prefix}ContractName = {_cpp_string(spec['contract_name'])};",
        f"constexpr const char* k{prefix}FamilyName = {_cpp_string(spec['family'])};",
        f"constexpr const char* k{prefix}TupleId = {_cpp_string(spec['tuple_id'])};",
        f"constexpr const char* k{prefix}WriterOp = {_cpp_string(spec['writer_op'])};",
        f"constexpr const char* k{prefix}RouteLabel = {_cpp_string(spec['route_label'])};",
        f"constexpr const char* k{prefix}EvidenceId = {_cpp_string(metadata['evidence_id'])};",
        f"constexpr const char* k{prefix}GuardId = {_cpp_string(metadata['guard_id'])};",
        f"constexpr const char* k{prefix}FallbackPolicy = {_cpp_string(metadata['fallback_policy'])};",
        (
            f"constexpr const char* k{prefix}MaterializationPolicy = "
            f"{_cpp_string(metadata['materialization_policy'])};"
        ),
        f"constexpr at::ScalarType k{prefix}Dtype = {SCALAR_TYPE_BY_DTYPE[bounds['dtype']]};",
        f"constexpr std::int64_t k{prefix}MinRank = {bounds['rank']['min']};",
        f"constexpr std::int64_t k{prefix}MaxRank = {bounds['rank']['max']};",
        f"constexpr bool k{prefix}AlphaIsOne = {_cpp_bool(bounds['alpha'] == 1)};",
        f"constexpr bool k{prefix}HasOutput = {_cpp_bool(attributes['has_out']['values'][0])};",
        f"constexpr bool k{prefix}Inplace = {_cpp_bool(attributes['inplace']['values'][0])};",
        (
            f"constexpr bool k{prefix}RequiresVulkan = "
            f"{_cpp_bool(bounds['requires_vulkan'])};"
        ),
        (
            f"constexpr bool k{prefix}RequiresBufferStorage = "
            f"{_cpp_bool(bounds['requires_buffer_storage'])};"
        ),
        (
            f"constexpr bool k{prefix}RequiresBufferCompute = "
            f"{_cpp_bool(bounds['requires_buffer_compute'])};"
        ),
    ]
    lines.extend(op_constants)
    lines.extend(
        [
            "",
            f"struct {prefix}Spec final {{",
            "  const char* contract_name;",
            "  const char* family_name;",
            "  const char* tuple_id;",
            "  const char* writer_op;",
            "  const char* route_label;",
            "  const char* evidence_id;",
            "  const char* guard_id;",
            "  const char* fallback_policy;",
            "  const char* materialization_policy;",
            "  at::ScalarType dtype;",
            "  std::int64_t min_rank;",
            "  std::int64_t max_rank;",
            "  bool alpha_is_one;",
            "  bool has_output;",
            "  bool inplace;",
            "  bool requires_vulkan;",
            "  bool requires_buffer_storage;",
            "  bool requires_buffer_compute;",
        ]
    )
    lines.extend(op_fields)
    lines.extend(
        [
            "};",
            "",
            f"constexpr {prefix}Spec k{prefix}Spec = {{",
            f"        k{prefix}ContractName,",
            f"        k{prefix}FamilyName,",
            f"        k{prefix}TupleId,",
            f"        k{prefix}WriterOp,",
            f"        k{prefix}RouteLabel,",
            f"        k{prefix}EvidenceId,",
            f"        k{prefix}GuardId,",
            f"        k{prefix}FallbackPolicy,",
            f"        k{prefix}MaterializationPolicy,",
            f"        k{prefix}Dtype,",
            f"        k{prefix}MinRank,",
            f"        k{prefix}MaxRank,",
            f"        k{prefix}AlphaIsOne,",
            f"        k{prefix}HasOutput,",
            f"        k{prefix}Inplace,",
            f"        k{prefix}RequiresVulkan,",
            f"        k{prefix}RequiresBufferStorage,",
            f"        k{prefix}RequiresBufferCompute,",
        ]
    )
    lines.extend(op_initializers)
    lines.extend(
        [
            "};",
            "",
            f"constexpr bool {func_prefix}_rank_in_bounds(",
            f"    const {prefix}Spec& spec,",
            "    const std::int64_t rank) {",
            "  return rank >= spec.min_rank && rank <= spec.max_rank;",
            "}",
            "",
            f"constexpr bool {func_prefix}_dtype_matches(",
            f"    const {prefix}Spec& spec,",
            "    const at::ScalarType self_dtype,",
            "    const at::ScalarType other_dtype,",
            "    const at::ScalarType output_dtype) {",
            "  return self_dtype == spec.dtype && other_dtype == spec.dtype &&",
            "      output_dtype == spec.dtype;",
            "}",
            "",
            f"constexpr bool {func_prefix}_layout_matches(",
            f"    const {prefix}Spec& spec,",
            "    const bool self_is_vulkan,",
            "    const bool other_is_vulkan,",
            "    const bool self_supports_buffer_compute,",
            "    const bool other_supports_buffer_compute,",
            "    const bool buffer_route_selected) {",
            "  return buffer_route_selected &&",
            "      (!spec.requires_vulkan || (self_is_vulkan && other_is_vulkan)) &&",
            "      (!spec.requires_buffer_storage || buffer_route_selected) &&",
            "      (!spec.requires_buffer_compute ||",
            "       (self_supports_buffer_compute && other_supports_buffer_compute));",
            "}",
            "",
            f"constexpr bool {func_prefix}_attributes_match(",
            f"    const {prefix}Spec& spec,",
            "    const bool op_add,",
            "    const bool op_mul,",
            "    const bool alpha_is_one,",
            "    const bool has_output,",
            "    const bool inplace) {",
            "  return (op_add || op_mul) && alpha_is_one == spec.alpha_is_one &&",
            "      has_output == spec.has_output && inplace == spec.inplace;",
            "}",
            "",
            "} // namespace generated",
            "} // namespace utils",
            "} // namespace ops",
            "} // namespace vulkan",
            "} // namespace native",
            "} // namespace at",
            "",
        ]
    )
    return "\n".join(lines)


def generate_header(spec, source_name):
    key = (spec.get("contract_name"), spec.get("family"))
    if key == (EXPECTED_CONTRACT_NAME, EXPECTED_FAMILY):
        return generate_channel_cat_header(spec, source_name)
    if "shape_envelope" in spec:
        return generate_generic_shape_envelope_header(spec, source_name)
    raise RuntimeError(
        "unsupported contract spec for C++ generation: "
        f"{spec.get('contract_name')} {spec.get('family')}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    spec = _load_spec(args.spec)
    output = generate_header(spec, args.spec.replace("\\", "/"))
    if args.stdout:
        sys.stdout.buffer.write(output.encode("utf-8"))
        return
    raise RuntimeError("only --stdout is supported for the MVP generator")


if __name__ == "__main__":
    main()
