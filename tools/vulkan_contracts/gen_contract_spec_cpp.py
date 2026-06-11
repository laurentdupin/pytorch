#!/usr/bin/env python3

import argparse
import json
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
EXPECTED_EMBEDDING_LOOKUP_CONTRACT_NAME = "EmbeddingLookupContract"
EXPECTED_EMBEDDING_LOOKUP_FAMILY = "SmallBoundedLookup"
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


def _validate_embedding_lookup_spec(spec):
    _require(
        spec.get("contract_name") == EXPECTED_EMBEDDING_LOOKUP_CONTRACT_NAME,
        "expected EmbeddingLookupContract spec",
    )
    _require(
        spec.get("family") == EXPECTED_EMBEDDING_LOOKUP_FAMILY,
        "expected SmallBoundedLookup family",
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
        ),
        "EmbeddingLookupContract spec",
    )
    for key in ("tuple_id", "writer_op", "route_label"):
        _require_non_empty_string(spec, key, "EmbeddingLookupContract spec")

    envelope = _shape_envelope(spec, "embedding_lookup_small_bounded")
    metadata = envelope["metadata"]
    _validate_contract_metadata(metadata, "EmbeddingLookupContract metadata")
    _require(spec["metadata"] == metadata, "metadata must match shape_envelope")

    bounds = envelope["bounds"]
    _require(spec["bounds"] == bounds, "bounds must match shape_envelope")
    _require(isinstance(bounds, dict), "spec bounds must be an object")
    _require_keys(
        bounds,
        (
            "weight_dtype",
            "indices_dtype",
            "weight_rank",
            "index_ranks",
            "num_embeddings",
            "embedding_dim",
            "num_indices",
            "padding_idx_has_hint",
            "scale_grad_by_freq",
            "sparse",
        ),
        "EmbeddingLookupContract bounds",
    )
    _require(bounds["weight_dtype"] in SCALAR_TYPE_BY_DTYPE, "unsupported weight_dtype")
    _require(bounds["indices_dtype"] in SCALAR_TYPE_BY_DTYPE, "unsupported indices_dtype")
    _validate_int(bounds["weight_rank"], "bounds.weight_rank")
    index_ranks = bounds["index_ranks"]
    _require(isinstance(index_ranks, list), "bounds.index_ranks must be a list")
    _require(len(index_ranks) == 2, "bounds.index_ranks must have two entries")
    for index, rank in enumerate(index_ranks):
        _validate_int(rank, f"bounds.index_ranks[{index}]")

    for key in ("num_embeddings", "embedding_dim", "num_indices"):
        _validate_bound_pair(bounds[key], f"EmbeddingLookupContract bounds.{key}")

    for key in ("padding_idx_has_hint", "scale_grad_by_freq", "sparse"):
        _validate_bool(bounds[key], f"bounds.{key}")


def generate_embedding_lookup_header(spec, source_name):
    _validate_embedding_lookup_spec(spec)
    envelope = _shape_envelope(spec, "embedding_lookup_small_bounded")
    metadata = envelope["metadata"]
    bounds = envelope["bounds"]
    index_ranks = bounds["index_ranks"]

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
        f"constexpr const char* kEmbeddingLookupContractName = {_cpp_string(spec['contract_name'])};",
        (
            "constexpr const char* kEmbeddingLookupSmallBoundedLookupFamilyName = "
            f"{_cpp_string(spec['family'])};"
        ),
        (
            "constexpr const char* kEmbeddingLookupSmallBoundedLookupTupleId = "
            f"{_cpp_string(spec['tuple_id'])};"
        ),
        (
            "constexpr const char* kEmbeddingLookupSmallBoundedLookupWriterOp = "
            f"{_cpp_string(spec['writer_op'])};"
        ),
        (
            "constexpr const char* kEmbeddingLookupSmallBoundedLookupRouteLabel = "
            f"{_cpp_string(spec['route_label'])};"
        ),
        "",
        (
            "constexpr at::ScalarType kEmbeddingLookupSmallBoundedLookupWeightDtype = "
            f"{SCALAR_TYPE_BY_DTYPE[bounds['weight_dtype']]};"
        ),
        (
            "constexpr at::ScalarType kEmbeddingLookupSmallBoundedLookupIndicesDtype = "
            f"{SCALAR_TYPE_BY_DTYPE[bounds['indices_dtype']]};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupWeightRank = "
            f"{bounds['weight_rank']};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupIndexRank1 = "
            f"{index_ranks[0]};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupIndexRank2 = "
            f"{index_ranks[1]};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupMinNumEmbeddings = "
            f"{bounds['num_embeddings']['min']};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupMaxNumEmbeddings = "
            f"{bounds['num_embeddings']['max']};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupMinEmbeddingDim = "
            f"{bounds['embedding_dim']['min']};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupMaxEmbeddingDim = "
            f"{bounds['embedding_dim']['max']};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupMinNumIndices = "
            f"{bounds['num_indices']['min']};"
        ),
        (
            "constexpr std::int64_t kEmbeddingLookupSmallBoundedLookupMaxNumIndices = "
            f"{bounds['num_indices']['max']};"
        ),
        (
            "constexpr bool kEmbeddingLookupSmallBoundedLookupPaddingIdxHasHint = "
            f"{_cpp_bool(bounds['padding_idx_has_hint'])};"
        ),
        (
            "constexpr bool kEmbeddingLookupSmallBoundedLookupScaleGradByFreq = "
            f"{_cpp_bool(bounds['scale_grad_by_freq'])};"
        ),
        (
            "constexpr bool kEmbeddingLookupSmallBoundedLookupSparse = "
            f"{_cpp_bool(bounds['sparse'])};"
        ),
        "",
        "struct EmbeddingLookupSmallBoundedLookupSpec final {",
        "  const char* contract_name;",
        "  const char* family_name;",
        "  const char* tuple_id;",
        "  const char* writer_op;",
        "  const char* route_label;",
        "  const char* evidence_id;",
        "  const char* guard_id;",
        "  const char* fallback_policy;",
        "  const char* materialization_policy;",
        "  at::ScalarType weight_dtype;",
        "  at::ScalarType indices_dtype;",
        "  std::int64_t weight_rank;",
        "  std::int64_t index_rank_1;",
        "  std::int64_t index_rank_2;",
        "  std::int64_t min_num_embeddings;",
        "  std::int64_t max_num_embeddings;",
        "  std::int64_t min_embedding_dim;",
        "  std::int64_t max_embedding_dim;",
        "  std::int64_t min_num_indices;",
        "  std::int64_t max_num_indices;",
        "  bool padding_idx_has_hint;",
        "  bool scale_grad_by_freq;",
        "  bool sparse;",
        "};",
        "",
        "constexpr EmbeddingLookupSmallBoundedLookupSpec",
        "    kEmbeddingLookupSmallBoundedLookupSpec = {",
        "        kEmbeddingLookupContractName,",
        "        kEmbeddingLookupSmallBoundedLookupFamilyName,",
        "        kEmbeddingLookupSmallBoundedLookupTupleId,",
        "        kEmbeddingLookupSmallBoundedLookupWriterOp,",
        "        kEmbeddingLookupSmallBoundedLookupRouteLabel,",
        f"        {_cpp_string(metadata['evidence_id'])},",
        f"        {_cpp_string(metadata['guard_id'])},",
        f"        {_cpp_string(metadata['fallback_policy'])},",
        f"        {_cpp_string(metadata['materialization_policy'])},",
        "        kEmbeddingLookupSmallBoundedLookupWeightDtype,",
        "        kEmbeddingLookupSmallBoundedLookupIndicesDtype,",
        "        kEmbeddingLookupSmallBoundedLookupWeightRank,",
        "        kEmbeddingLookupSmallBoundedLookupIndexRank1,",
        "        kEmbeddingLookupSmallBoundedLookupIndexRank2,",
        "        kEmbeddingLookupSmallBoundedLookupMinNumEmbeddings,",
        "        kEmbeddingLookupSmallBoundedLookupMaxNumEmbeddings,",
        "        kEmbeddingLookupSmallBoundedLookupMinEmbeddingDim,",
        "        kEmbeddingLookupSmallBoundedLookupMaxEmbeddingDim,",
        "        kEmbeddingLookupSmallBoundedLookupMinNumIndices,",
        "        kEmbeddingLookupSmallBoundedLookupMaxNumIndices,",
        "        kEmbeddingLookupSmallBoundedLookupPaddingIdxHasHint,",
        "        kEmbeddingLookupSmallBoundedLookupScaleGradByFreq,",
        "        kEmbeddingLookupSmallBoundedLookupSparse};",
        "",
        "constexpr bool embedding_lookup_index_rank_in_bounds(",
        "    const EmbeddingLookupSmallBoundedLookupSpec& spec,",
        "    const std::int64_t index_rank) {",
        "  return index_rank == spec.index_rank_1 ||",
        "      index_rank == spec.index_rank_2;",
        "}",
        "",
        "constexpr bool embedding_lookup_small_bounded_options_match(",
        "    const EmbeddingLookupSmallBoundedLookupSpec& spec,",
        "    const at::ScalarType weight_dtype,",
        "    const at::ScalarType indices_dtype,",
        "    const std::int64_t weight_rank,",
        "    const std::int64_t index_rank,",
        "    const bool padding_idx_has_hint,",
        "    const bool scale_grad_by_freq,",
        "    const bool sparse) {",
        "  return weight_dtype == spec.weight_dtype &&",
        "      indices_dtype == spec.indices_dtype &&",
        "      weight_rank == spec.weight_rank &&",
        "      embedding_lookup_index_rank_in_bounds(spec, index_rank) &&",
        "      padding_idx_has_hint == spec.padding_idx_has_hint &&",
        "      scale_grad_by_freq == spec.scale_grad_by_freq &&",
        "      sparse == spec.sparse;",
        "}",
        "",
        "constexpr bool embedding_lookup_small_bounded_in_bounds(",
        "    const EmbeddingLookupSmallBoundedLookupSpec& spec,",
        "    const std::int64_t num_embeddings,",
        "    const std::int64_t embedding_dim,",
        "    const std::int64_t num_indices) {",
        "  return num_embeddings <= spec.max_num_embeddings &&",
        "      embedding_dim <= spec.max_embedding_dim &&",
        "      num_indices <= spec.max_num_indices;",
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


def generate_header(spec, source_name):
    key = (spec.get("contract_name"), spec.get("family"))
    if key == (EXPECTED_CONTRACT_NAME, EXPECTED_FAMILY):
        return generate_channel_cat_header(spec, source_name)
    if key == (
        EXPECTED_EMBEDDING_LOOKUP_CONTRACT_NAME,
        EXPECTED_EMBEDDING_LOOKUP_FAMILY,
    ):
        return generate_embedding_lookup_header(spec, source_name)
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
