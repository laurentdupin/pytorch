import argparse
import glob
import json
import os
import subprocess
import sys


GENERATED_CPP_MANIFEST_FILE = "generated_cpp_manifest.json"

CONTRACT_SPEC_REQUIRED_FIELDS = (
    "schema_version",
    "contract_name",
    "family",
    "tuple_id",
    "writer_op",
    "route_label",
    "bounds",
    "positive_cases",
    "negative_cases",
)

CONTRACT_SPEC_STRING_FIELDS = (
    "contract_name",
    "family",
    "tuple_id",
    "writer_op",
    "route_label",
)

CONTRACT_METADATA_FIELDS = (
    "evidence_id",
    "guard_id",
    "fallback_policy",
    "materialization_policy",
)

SHAPE_ENVELOPE_REQUIRED_FIELDS = (
    "version",
    "role",
    "inputs",
    "attributes",
    "bounds",
    "relationships",
    "aggregate_bounds",
    "layout",
    "capability_requirements",
    "metadata",
    "policies",
    "positive_cases",
    "negative_axes",
    "fuzz_hints",
)

SHAPE_ENVELOPE_POLICY_FIELDS = (
    "fallback",
    "readback",
    "copy",
)


def _require_non_empty_string(mapping, field, context):
    value = mapping[field]
    if not isinstance(value, str) or value == "":
        raise AssertionError(f"{context} {field} must be a non-empty string")


def _require_bool(value, context):
    if not isinstance(value, bool):
        raise AssertionError(f"{context} must be a boolean")


def _require_int(value, context):
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssertionError(f"{context} must be an integer")


def _require_mapping(value, context, allow_empty=False):
    if not isinstance(value, dict) or (not allow_empty and not value):
        raise AssertionError(f"{context} must be a non-empty object")


def _require_list(value, context, allow_empty=False):
    if not isinstance(value, list) or (not allow_empty and not value):
        raise AssertionError(f"{context} must be a non-empty list")


def _require_equal(actual, expected, context):
    if actual != expected:
        raise AssertionError(f"{context} mismatch: {actual!r} != {expected!r}")


def _validate_scalar(value, context):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, str) and value != "":
        return
    raise AssertionError(f"{context} must be a non-empty string, integer, or boolean")


def _validate_shape_field(field, context, require_bound=False):
    _require_mapping(field, context)
    has_bound = False
    if "values" in field:
        _require_list(field["values"], f"{context}.values")
        for index, value in enumerate(field["values"]):
            _validate_scalar(value, f"{context}.values[{index}]")
        has_bound = True
    for key in ("min", "max", "multiple_of"):
        if key in field:
            _require_int(field[key], f"{context}.{key}")
            has_bound = True
    if "min" in field and "max" in field and field["min"] > field["max"]:
        raise AssertionError(f"{context}.min must be <= max")
    if "multiple_of" in field and field["multiple_of"] <= 0:
        raise AssertionError(f"{context}.multiple_of must be positive")
    if "optional" in field:
        _require_bool(field["optional"], f"{context}.optional")
    for key in ("symbol", "field", "kind"):
        if key in field:
            _validate_scalar(field[key], f"{context}.{key}")
    if require_bound and not has_bound:
        raise AssertionError(f"{context} must include min/max, values, or multiple_of")


def _validate_bounds_tree(value, context):
    if isinstance(value, dict):
        if any(key in value for key in ("min", "max", "multiple_of", "values")):
            _validate_shape_field(value, context)
        for key, child in value.items():
            if key in ("min", "max", "multiple_of", "values", "optional"):
                continue
            _validate_bounds_tree(child, f"{context}.{key}")
    elif isinstance(value, list):
        _require_list(value, context)
        for index, child in enumerate(value):
            _validate_scalar(child, f"{context}[{index}]")
    else:
        _validate_scalar(value, context)


def _validate_contract_metadata(metadata, context):
    _require_mapping(metadata, context)
    require_fields(metadata, CONTRACT_METADATA_FIELDS, context)
    for field in CONTRACT_METADATA_FIELDS:
        _require_non_empty_string(metadata, field, context)


def _validate_shape_envelope_inputs(inputs, context):
    _require_mapping(inputs, f"{context} inputs")
    for input_name, input_spec in inputs.items():
        _require_mapping(input_spec, f"{context} inputs.{input_name}")
        _require_non_empty_string(input_spec, "kind", f"{context} inputs.{input_name}")
        for field_name in ("count", "dtype", "rank"):
            if field_name in input_spec:
                _validate_shape_field(
                    input_spec[field_name],
                    f"{context} inputs.{input_name}.{field_name}",
                    require_bound=True,
                )
        if "dims" in input_spec:
            _require_list(input_spec["dims"], f"{context} inputs.{input_name}.dims")
            for index, dim in enumerate(input_spec["dims"]):
                dim_context = f"{context} inputs.{input_name}.dims[{index}]"
                _validate_shape_field(dim, dim_context, require_bound=True)
                _require_non_empty_string(dim, "symbol", dim_context)


def _validate_shape_envelope_attributes(attributes, context):
    _require_mapping(attributes, f"{context} attributes", allow_empty=True)
    for attribute_name, attribute in attributes.items():
        _validate_shape_field(
            attribute,
            f"{context} attributes.{attribute_name}",
            require_bound=True,
        )


def _validate_shape_envelope_results(results, context):
    _require_mapping(results, f"{context} results", allow_empty=True)
    for result_name, result_spec in results.items():
        _require_mapping(result_spec, f"{context} results.{result_name}")
        for field_name in ("dtype", "rank"):
            if field_name in result_spec:
                _validate_shape_field(
                    result_spec[field_name],
                    f"{context} results.{result_name}.{field_name}",
                    require_bound=True,
                )
        if "dims" in result_spec:
            _require_list(result_spec["dims"], f"{context} results.{result_name}.dims")
            for index, dim in enumerate(result_spec["dims"]):
                dim_context = f"{context} results.{result_name}.dims[{index}]"
                _validate_shape_field(dim, dim_context, require_bound=True)
                _require_non_empty_string(dim, "symbol", dim_context)


def _validate_shape_envelope_relationships(
    relationships,
    context,
    input_names=(),
    result_names=(),
):
    _require_list(relationships, f"{context} relationships", allow_empty=True)
    input_names = set(input_names)
    result_names = set(result_names)
    for index, relationship in enumerate(relationships):
        rel_context = f"{context} relationships[{index}]"
        _require_mapping(relationship, rel_context)
        _require_non_empty_string(relationship, "type", rel_context)
        rel_type = relationship["type"]
        if rel_type not in ("equal", "sum_output", "product", "broadcast_compatible"):
            raise AssertionError(f"{rel_context} has unsupported type {rel_type!r}")
        if rel_type == "equal":
            require_fields(relationship, ("scope", "fields"), rel_context)
            _require_non_empty_string(relationship, "scope", rel_context)
            _require_list(relationship["fields"], f"{rel_context}.fields")
            for field_index, field in enumerate(relationship["fields"]):
                _validate_scalar(field, f"{rel_context}.fields[{field_index}]")
        elif rel_type == "sum_output":
            require_fields(relationship, ("input", "dim", "result"), rel_context)
            for field in ("input", "dim", "result"):
                _require_non_empty_string(relationship, field, rel_context)
        elif rel_type == "product":
            require_fields(relationship, ("input", "dims", "result"), rel_context)
            _require_non_empty_string(relationship, "input", rel_context)
            if isinstance(relationship["dims"], list):
                _require_list(relationship["dims"], f"{rel_context}.dims")
                for dim_index, dim in enumerate(relationship["dims"]):
                    _validate_scalar(dim, f"{rel_context}.dims[{dim_index}]")
            else:
                _validate_scalar(relationship["dims"], f"{rel_context}.dims")
            _require_non_empty_string(relationship, "result", rel_context)
        elif rel_type == "broadcast_compatible":
            require_fields(
                relationship,
                ("left", "right", "result", "align", "max_rank"),
                rel_context,
            )
            for field in ("left", "right", "result", "align"):
                _require_non_empty_string(relationship, field, rel_context)
            if relationship["left"] not in input_names:
                raise AssertionError(
                    f"{rel_context}.left references unknown input "
                    f"{relationship['left']!r}"
                )
            if relationship["right"] not in input_names:
                raise AssertionError(
                    f"{rel_context}.right references unknown input "
                    f"{relationship['right']!r}"
                )
            if relationship["result"] not in result_names:
                raise AssertionError(
                    f"{rel_context}.result references unknown result "
                    f"{relationship['result']!r}"
                )
            _require_equal(relationship["align"], "right", f"{rel_context}.align")
            _require_int(relationship["max_rank"], f"{rel_context}.max_rank")
            if relationship["max_rank"] > 4:
                raise AssertionError(f"{rel_context}.max_rank must be <= 4")


def _validate_shape_envelope_aggregate_bounds(aggregate_bounds, context):
    _require_mapping(aggregate_bounds, f"{context} aggregate_bounds", allow_empty=True)
    for aggregate_name, aggregate in aggregate_bounds.items():
        aggregate_context = f"{context} aggregate_bounds.{aggregate_name}"
        _validate_shape_field(aggregate, aggregate_context, require_bound=True)
        for field in ("input", "field"):
            if field in aggregate:
                _require_non_empty_string(aggregate, field, aggregate_context)


def _validate_shape_envelope_layout(layout, context):
    _require_mapping(layout, f"{context} layout", allow_empty=True)
    for field, value in layout.items():
        if isinstance(value, bool):
            continue
        _validate_scalar(value, f"{context} layout.{field}")


def _validate_shape_envelope_capabilities(capability_requirements, context):
    _require_mapping(
        capability_requirements,
        f"{context} capability_requirements",
        allow_empty=True,
    )
    for field, value in capability_requirements.items():
        if isinstance(value, bool):
            continue
        _validate_scalar(value, f"{context} capability_requirements.{field}")


def _negative_axis_names(envelope):
    names = set()
    for index, axis in enumerate(envelope["negative_axes"]):
        context = f"ShapeEnvelope negative_axes[{index}]"
        _require_mapping(axis, context)
        _require_non_empty_string(axis, "violates", context)
        if "adjacent" in axis:
            _require_non_empty_string(axis, "adjacent", context)
        if "value" in axis:
            _validate_scalar(axis["value"], f"{context}.value")
        names.add(axis["violates"])
    return names


def _relationship_types(envelope):
    return {relationship["type"] for relationship in envelope["relationships"]}


def _shape_envelope_rank_constraint(input_spec, context):
    if "rank" not in input_spec:
        raise AssertionError(f"{context} missing rank constraint")
    return input_spec["rank"]


def _shape_envelope_value_set(field, context):
    values = field.get("values")
    _require_list(values, f"{context}.values")
    return values


def _shape_envelope_numeric_bound(field, bound_name, context):
    if bound_name not in field:
        raise AssertionError(f"{context} missing {bound_name}")
    _require_int(field[bound_name], f"{context}.{bound_name}")
    return field[bound_name]


def _shape_envelope_multiple_of(field, context):
    if "multiple_of" not in field:
        raise AssertionError(f"{context} missing multiple_of")
    _require_int(field["multiple_of"], f"{context}.multiple_of")
    return field["multiple_of"]


def _single_value(field, context):
    values = _shape_envelope_value_set(field, context)
    if len(values) != 1:
        raise AssertionError(f"{context}.values must contain one value")
    return values[0]


def _dims_by_symbol(input_spec, context):
    dims = {}
    for dim in input_spec["dims"]:
        symbol = dim["symbol"]
        if symbol in dims:
            raise AssertionError(f"{context} duplicate dim symbol {symbol}")
        dims[symbol] = dim
    return dims


def _shape_envelope_negative_axes(envelope):
    axes = {}
    for axis in envelope["negative_axes"]:
        violates = axis["violates"]
        if violates in axes:
            raise AssertionError(f"duplicate ShapeEnvelope negative axis {violates}")
        axes[violates] = axis
    return axes


_MISSING_CASE_FIELD = object()


def _canonical_case_value(value):
    if isinstance(value, list):
        return tuple(_canonical_case_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (key, _canonical_case_value(child))
            for key, child in sorted(value.items())
        )
    return value


def _shape_envelope_derived_case_value(case, field):
    if "by_violates" in field:
        violates = case["violates"]
        if violates not in field["by_violates"]:
            raise AssertionError(
                f"case {case['name']} has unsupported violated axis {violates}"
            )
        return _shape_envelope_derived_case_value(
            case,
            field["by_violates"][violates],
        )

    op = field["op"]
    source = case[field["field"]]
    if op == "field":
        return source
    if op == "len":
        return len(source)
    if op == "product":
        return _product(source)
    if op == "shape_dim":
        return source[field.get("index", 0)][field["dim"]]
    if op == "sum_shape_dim":
        return sum(shape[field["dim"]] for shape in source)
    raise AssertionError(f"unsupported ShapeEnvelope case key op {op!r}")


def _shape_envelope_case_key(case, fields):
    key = []
    for field in fields:
        if isinstance(field, tuple):
            field_name, default = field
        elif isinstance(field, dict):
            key.append(_canonical_case_value(
                _shape_envelope_derived_case_value(case, field)
            ))
            continue
        else:
            field_name = field
            default = _MISSING_CASE_FIELD
        if field_name in case:
            value = case[field_name]
        elif default is not _MISSING_CASE_FIELD:
            value = default
        else:
            raise AssertionError(f"case {case['name']} missing field {field_name}")
        key.append(_canonical_case_value(value))
    return tuple(key)


def _dedupe_preserving_order(values):
    result = []
    seen = set()
    for value in values:
        canonical = _canonical_case_value(value)
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(value)
    return result


def _shape_envelope_boundary_values(field):
    if "values" in field:
        return list(_shape_envelope_value_set(field, "ShapeEnvelope field"))

    values = []
    has_min = "min" in field
    has_max = "max" in field
    if has_min:
        minimum = field["min"]
        if "multiple_of" in field:
            multiple = field["multiple_of"]
            aligned_minimum = ((minimum + multiple - 1) // multiple) * multiple
            values.append(aligned_minimum)
        else:
            values.append(minimum)
    if has_max:
        maximum = field["max"]
        if "multiple_of" in field:
            multiple = field["multiple_of"]
            aligned_maximum = (maximum // multiple) * multiple
            values.append(aligned_maximum)
        else:
            values.append(maximum)
    if not values and "multiple_of" in field:
        values.append(field["multiple_of"])
    if field.get("optional"):
        values.append(None)
    return _dedupe_preserving_order(values)


def _shape_envelope_value_candidates(field):
    if "values" in field:
        return list(_shape_envelope_value_set(field, "ShapeEnvelope field"))
    return _shape_envelope_boundary_values(field)


def _shape_envelope_symbolic_bounds(envelope):
    bounds = []
    for input_name in sorted(envelope["inputs"]):
        input_spec = envelope["inputs"][input_name]
        for field_name in ("count", "dtype", "rank"):
            if field_name in input_spec:
                field = input_spec[field_name]
                bounds.append(
                    {
                        "path": f"inputs.{input_name}.{field_name}",
                        "candidates": _shape_envelope_value_candidates(field),
                        "optional": field.get("optional", False),
                    }
                )
        for dim in input_spec.get("dims", ()):
            symbol = dim["symbol"]
            bounds.append(
                {
                    "path": f"inputs.{input_name}.dims.{symbol}",
                    "field": dim.get("field"),
                    "candidates": _shape_envelope_value_candidates(dim),
                    "optional": dim.get("optional", False),
                }
            )

    for result_name in sorted(envelope.get("results", {})):
        result_spec = envelope["results"][result_name]
        for field_name in ("dtype", "rank"):
            if field_name in result_spec:
                field = result_spec[field_name]
                bounds.append(
                    {
                        "path": f"results.{result_name}.{field_name}",
                        "candidates": _shape_envelope_value_candidates(field),
                        "optional": field.get("optional", False),
                    }
                )
        for dim in result_spec.get("dims", ()):
            symbol = dim["symbol"]
            bounds.append(
                {
                    "path": f"results.{result_name}.dims.{symbol}",
                    "field": dim.get("field"),
                    "candidates": _shape_envelope_value_candidates(dim),
                    "optional": dim.get("optional", False),
                }
            )

    for attribute_name in sorted(envelope["attributes"]):
        attribute = envelope["attributes"][attribute_name]
        bounds.append(
            {
                "path": f"attributes.{attribute_name}",
                "candidates": _shape_envelope_value_candidates(attribute),
                "optional": attribute.get("optional", False),
            }
        )
    for relationship in envelope["relationships"]:
        if relationship["type"] != "broadcast_compatible":
            continue
        bounds.extend(
            [
                {
                    "path": "relationships.broadcast_compatible.left",
                    "candidates": [relationship["left"]],
                    "optional": False,
                },
                {
                    "path": "relationships.broadcast_compatible.right",
                    "candidates": [relationship["right"]],
                    "optional": False,
                },
                {
                    "path": "relationships.broadcast_compatible.result",
                    "candidates": [relationship["result"]],
                    "optional": False,
                },
                {
                    "path": "relationships.broadcast_compatible.max_rank",
                    "candidates": [relationship["max_rank"]],
                    "optional": False,
                },
            ]
        )
    return bounds


def _shape_envelope_assignment_from_bounds(name, symbolic_bounds, index):
    values = {}
    for bound in symbolic_bounds:
        candidates = bound["candidates"]
        if not candidates:
            continue
        values[bound["path"]] = candidates[min(index, len(candidates) - 1)]
    return {
        "name": name,
        "kind": "legal_boundary",
        "values": values,
    }


def _shape_envelope_legal_assignments(envelope):
    symbolic_bounds = _shape_envelope_symbolic_bounds(envelope)
    if not symbolic_bounds:
        return []
    return [
        _shape_envelope_assignment_from_bounds(
            "generated_boundary_min",
            symbolic_bounds,
            0,
        ),
        _shape_envelope_assignment_from_bounds(
            "generated_boundary_max",
            symbolic_bounds,
            -1,
        ),
    ]


def _shape_envelope_adjacent_assignments(envelope):
    assignments = []
    for axis in envelope["negative_axes"]:
        assignments.append(
            {
                "name": f"generated_adjacent_{axis['violates']}",
                "kind": "adjacent_negative",
                "violates": axis["violates"],
                "adjacent": axis.get("adjacent"),
                "value": axis.get("value"),
            }
        )
    return assignments


def _generated_shape_envelope_assignment_cases(spec):
    envelope = spec["shape_envelope"]
    return {
        "legal_assignments": _shape_envelope_legal_assignments(envelope),
        "adjacent_negative_assignments": (
            _shape_envelope_adjacent_assignments(envelope)
        ),
    }


def _checked_in_shape_envelope_legal_cases(spec):
    return list(spec["positive_cases"])


def _checked_in_shape_envelope_adjacent_negative_cases(spec):
    return list(spec["negative_cases"])


def _validate_shape_envelope_common(file_name, spec, envelope):
    context = f"{file_name} ShapeEnvelope v1"
    _require_mapping(envelope, context)
    require_fields(envelope, SHAPE_ENVELOPE_REQUIRED_FIELDS, context)
    if envelope["version"] != 1:
        raise AssertionError(f"{context} version must be 1")
    _require_non_empty_string(envelope, "role", context)

    _validate_shape_envelope_inputs(envelope["inputs"], context)
    _validate_shape_envelope_results(envelope.get("results", {}), context)
    _validate_shape_envelope_attributes(envelope["attributes"], context)
    _validate_bounds_tree(envelope["bounds"], f"{context} bounds")
    _validate_shape_envelope_relationships(
        envelope["relationships"],
        context,
        input_names=envelope["inputs"],
        result_names=envelope.get("results", {}),
    )
    _validate_shape_envelope_aggregate_bounds(envelope["aggregate_bounds"], context)
    _validate_shape_envelope_layout(envelope["layout"], context)
    _validate_shape_envelope_capabilities(
        envelope["capability_requirements"],
        context,
    )
    _validate_contract_metadata(envelope["metadata"], f"{context} metadata")

    policies = envelope["policies"]
    _require_mapping(policies, f"{context} policies")
    require_fields(policies, SHAPE_ENVELOPE_POLICY_FIELDS, f"{context} policies")
    for field in SHAPE_ENVELOPE_POLICY_FIELDS:
        _require_non_empty_string(policies, field, f"{context} policies")

    positive_policy = envelope["positive_cases"]
    _require_mapping(positive_policy, f"{context} positive_cases")
    _require_non_empty_string(positive_policy, "source", f"{context} positive_cases")

    _require_list(envelope["negative_axes"], f"{context} negative_axes")
    negative_axes = _negative_axis_names(envelope)
    negative_cases = {case["violates"] for case in spec["negative_cases"]}
    missing_negative_axes = sorted(negative_cases - negative_axes)
    if missing_negative_axes:
        raise AssertionError(
            f"{context} missing negative_axes for {missing_negative_axes}"
        )

    _require_mapping(envelope["fuzz_hints"], f"{context} fuzz_hints", allow_empty=True)
    if "metadata" in spec:
        _require_equal(envelope["metadata"], spec["metadata"], f"{context} metadata")
        _require_equal(
            policies["fallback"],
            spec["metadata"]["fallback_policy"],
            f"{context} fallback policy",
        )
        _require_equal(
            policies["copy"],
            spec["metadata"]["materialization_policy"],
            f"{context} copy policy",
        )


def _validate_channel_cat_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} ChannelCat ShapeEnvelope"
    _require_equal(spec["contract_name"], "ChannelCatContract", f"{context} contract")
    _require_equal(spec["family"], "Rank4Dim1BufferView", f"{context} family")
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    inputs = envelope["inputs"]
    require_fields(inputs, ("tensors",), f"{context} inputs")
    tensors = inputs["tensors"]
    bounds = envelope["bounds"]
    _require_equal(tensors["count"], bounds["input_count"], f"{context} input count")
    _require_equal(
        _single_value(tensors["dtype"], f"{context} dtype"),
        bounds["dtype"],
        f"{context} dtype",
    )
    _require_equal(
        _single_value(tensors["rank"], f"{context} rank"),
        bounds["rank"],
        f"{context} rank",
    )

    dims = _dims_by_symbol(tensors, context)
    for symbol in ("N", "C", "H", "W"):
        if symbol not in dims:
            raise AssertionError(f"{context} missing symbolic dim {symbol}")
    _require_equal(dims["N"].get("values"), [bounds["batch"]], f"{context} N")
    _require_equal(dims["C"]["min"], bounds["channels"]["min"], f"{context} C min")
    _require_equal(
        dims["C"]["max"],
        bounds["channels"]["max_per_input"],
        f"{context} C max",
    )
    _require_equal(
        dims["C"]["multiple_of"],
        bounds["channels"]["multiple_of"],
        f"{context} C multiple_of",
    )
    _require_equal(dims["H"]["min"], bounds["height"]["min"], f"{context} H min")
    _require_equal(dims["H"]["max"], bounds["height"]["max"], f"{context} H max")
    _require_equal(dims["W"]["min"], bounds["width"]["min"], f"{context} W min")
    _require_equal(dims["W"]["max"], bounds["width"]["max"], f"{context} W max")

    relationships = _relationship_types(envelope)
    for rel_type in ("equal", "sum_output"):
        if rel_type not in relationships:
            raise AssertionError(f"{context} missing {rel_type} relationship")

    aggregate = envelope["aggregate_bounds"]["total_channels"]
    _require_equal(aggregate["min"], 1, f"{context} total channel min")
    _require_equal(
        aggregate["max"],
        bounds["channels"]["max_total"],
        f"{context} total channel max",
    )
    _require_equal(
        aggregate["multiple_of"],
        bounds["channels"]["multiple_of"],
        f"{context} total channel multiple_of",
    )

    layout = envelope["layout"]
    _require_equal(
        layout["requires_vulkan"],
        bounds["requires_vulkan"],
        f"{context} requires_vulkan",
    )
    _require_equal(
        layout["requires_contiguous"],
        bounds["requires_contiguous"],
        f"{context} requires_contiguous",
    )
    _require_equal(
        layout["requires_buffer_storage"],
        bounds["requires_buffer_storage"],
        f"{context} requires_buffer_storage",
    )
    _require_equal(
        envelope["capability_requirements"]["requires_buffer_compute"],
        bounds["requires_buffer_compute"],
        f"{context} requires_buffer_compute",
    )


def _validate_embedding_lookup_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} EmbeddingLookup ShapeEnvelope"
    _require_equal(spec["contract_name"], "EmbeddingLookupContract", f"{context} contract")
    _require_equal(spec["family"], "SmallBoundedLookup", f"{context} family")
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    inputs = envelope["inputs"]
    require_fields(inputs, ("weight", "indices"), f"{context} inputs")
    weight = inputs["weight"]
    indices = inputs["indices"]
    bounds = envelope["bounds"]
    _require_equal(
        _single_value(weight["dtype"], f"{context} weight dtype"),
        bounds["weight_dtype"],
        f"{context} weight dtype",
    )
    _require_equal(
        _single_value(weight["rank"], f"{context} weight rank"),
        bounds["weight_rank"],
        f"{context} weight rank",
    )
    _require_equal(
        _single_value(indices["dtype"], f"{context} indices dtype"),
        bounds["indices_dtype"],
        f"{context} indices dtype",
    )
    _require_equal(
        indices["rank"]["values"],
        bounds["index_ranks"],
        f"{context} index ranks",
    )

    weight_dims = _dims_by_symbol(weight, f"{context} weight")
    _require_equal(
        weight_dims["V"]["min"],
        bounds["num_embeddings"]["min"],
        f"{context} V min",
    )
    _require_equal(
        weight_dims["V"]["max"],
        bounds["num_embeddings"]["max"],
        f"{context} V max",
    )
    _require_equal(
        weight_dims["D"]["min"],
        bounds["embedding_dim"]["min"],
        f"{context} D min",
    )
    _require_equal(
        weight_dims["D"]["max"],
        bounds["embedding_dim"]["max"],
        f"{context} D max",
    )

    index_dims = _dims_by_symbol(indices, f"{context} indices")
    _require_equal(index_dims["I1"].get("optional"), True, f"{context} optional I1")
    _require_equal(
        index_dims["I0"]["max"],
        bounds["num_indices"]["max"],
        f"{context} I0 max",
    )
    _require_equal(
        index_dims["I1"]["max"],
        bounds["num_indices"]["max"],
        f"{context} I1 max",
    )

    relationships = _relationship_types(envelope)
    if "product" not in relationships:
        raise AssertionError(f"{context} missing product relationship")
    for attribute_name in ("padding_idx_has_hint", "scale_grad_by_freq", "sparse"):
        _require_equal(
            _single_value(
                envelope["attributes"][attribute_name],
                f"{context} {attribute_name}",
            ),
            bounds[attribute_name],
            f"{context} {attribute_name}",
        )
    _require_equal(
        envelope["layout"]["weight_requires_vulkan"],
        True,
        f"{context} weight layout",
    )
    _require_equal(
        envelope["layout"]["indices_requires_vulkan"],
        True,
        f"{context} indices layout",
    )


def _contiguous_strides(sizes):
    stride = 1
    strides = []
    for size in reversed(sizes):
        strides.insert(0, stride)
        stride *= size
    return strides


def _is_non_overlapping_dense_stride(sizes, strides):
    if len(sizes) != len(strides):
        return False
    dims = [index for index, size in enumerate(sizes) if size > 1]
    dims.sort(key=lambda index: strides[index])
    expected_stride = 1
    for dim in dims:
        if strides[dim] != expected_stride:
            return False
        expected_stride *= sizes[dim]
    return True


def _validate_safe_view_reshape_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} SafeViewReshape ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "SafeViewReshapeContract",
        f"{context} contract",
    )
    _require_equal(
        spec["family"],
        "ViewMaterializedDirectBuffer",
        f"{context} family",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    inputs = envelope["inputs"]
    require_fields(inputs, ("input", "output"), f"{context} inputs")
    bounds = envelope["bounds"]
    _require_equal(
        inputs["input"]["rank"],
        bounds["input_rank"],
        f"{context} input rank",
    )
    _require_equal(
        inputs["output"]["rank"],
        bounds["output_rank"],
        f"{context} output rank",
    )

    attributes = envelope["attributes"]
    _require_equal(
        _single_value(attributes["storage_offset"], f"{context} storage offset"),
        bounds["storage_offset"],
        f"{context} storage offset",
    )
    _require_equal(
        _single_value(
            attributes["output_stride_policy"],
            f"{context} output stride policy",
        ),
        bounds["output_stride"],
        f"{context} output stride policy",
    )
    _require_equal(
        _single_value(
            attributes["output_last_dim_multiple_of"],
            f"{context} output last dim multiple",
        ),
        bounds["output_last_dim_multiple_of"],
        f"{context} output last dim multiple",
    )

    relationships = _relationship_types(envelope)
    if "product" not in relationships or "equal" not in relationships:
        raise AssertionError(f"{context} missing product/equal relationship")
    _require_equal(
        envelope["layout"]["requires_vulkan"],
        bounds["requires_vulkan"],
        f"{context} requires_vulkan",
    )
    _require_equal(
        envelope["layout"]["output_storage"],
        "materialized_direct_buffer",
        f"{context} output storage",
    )

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            if _product(case["input_shape"]) != _product(case["output_shape"]):
                raise AssertionError(f"{case_context} product mismatch")
            if case["output_stride"] != _contiguous_strides(case["output_shape"]):
                raise AssertionError(f"{case_context} output stride is not contiguous")


def _validate_safe_view_reshape_alias_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} SafeViewReshapeAlias ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "SafeViewReshapeContract",
        f"{context} contract",
    )
    _require_equal(
        spec["family"],
        "ReshapeAliasDenseBufferDirect",
        f"{context} family",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    inputs = envelope["inputs"]
    require_fields(inputs, ("input", "output"), f"{context} inputs")
    input_spec = inputs["input"]
    bounds = envelope["bounds"]
    _require_equal(
        _single_value(input_spec["dtype"], f"{context} input dtype"),
        bounds["input_dtype"],
        f"{context} input dtype",
    )
    _require_equal(
        input_spec["rank"],
        bounds["input_rank"],
        f"{context} input rank",
    )
    _require_equal(
        inputs["output"]["rank"],
        bounds["output_rank"],
        f"{context} output rank",
    )

    attributes = envelope["attributes"]
    _require_equal(
        _single_value(attributes["storage_offset"], f"{context} storage offset"),
        bounds["storage_offset"],
        f"{context} storage offset",
    )
    _require_equal(
        _single_value(attributes["input_stride_policy"], f"{context} input stride"),
        bounds["input_stride"],
        f"{context} input stride",
    )
    _require_equal(
        _single_value(attributes["output_stride_policy"], f"{context} output stride"),
        bounds["output_stride"],
        f"{context} output stride",
    )
    _require_equal(
        _single_value(
            attributes["output_last_dim_multiple_of"],
            f"{context} output last dim multiple",
        ),
        bounds["output_last_dim_multiple_of"],
        f"{context} output last dim multiple",
    )

    relationships = _relationship_types(envelope)
    if "product" not in relationships or "equal" not in relationships:
        raise AssertionError(f"{context} missing product/equal relationship")
    _require_equal(
        envelope["layout"]["requires_vulkan"],
        bounds["requires_vulkan"],
        f"{context} requires_vulkan",
    )
    _require_equal(
        envelope["layout"]["input_storage"],
        bounds["input_storage"],
        f"{context} input storage",
    )
    _require_equal(
        envelope["layout"]["output_storage"],
        "materialized_direct_buffer",
        f"{context} output storage",
    )

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            if _product(case["input_shape"]) != _product(case["output_shape"]):
                raise AssertionError(f"{case_context} product mismatch")
            if not _is_non_overlapping_dense_stride(
                case["input_shape"],
                case["input_stride"],
            ):
                raise AssertionError(
                    f"{case_context} input stride is not non-overlapping dense"
                )
            output_stride_is_dense = _is_non_overlapping_dense_stride(
                case["output_shape"],
                case["output_stride"],
            )
            if case.get("violates") == "output_stride_policy":
                if output_stride_is_dense:
                    raise AssertionError(
                        f"{case_context} output stride unexpectedly dense"
                    )
            elif not output_stride_is_dense:
                raise AssertionError(
                    f"{case_context} output stride is not non-overlapping dense"
                )
            if (
                case.get("violates") != "output_last_dim_multiple_of" and
                case["output_shape"] and
                case["output_shape"][-1] % bounds["output_last_dim_multiple_of"] != 0
            ):
                raise AssertionError(f"{case_context} output last dim is unaligned")


def _validate_batch_norm_inference_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} BatchNormInference ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "BatchNormInferenceContract",
        f"{context} contract",
    )
    family_expectations = {
        "BufferFloat4D": {
            "tuple_id": "buffer_inference_4d_float",
            "materialization_policy": "batch_norm_inference_buffer_kernel",
            "requires_buffer_storage": True,
        },
        "MaterializedBufferFloat4D": {
            "tuple_id": "materialized_buffer_inference_4d_float",
            "materialization_policy": (
                "materialize_to_buffer_then_batch_norm_inference_buffer_kernel"
            ),
            "requires_buffer_storage": False,
            "requires_materialization": True,
        },
    }
    family = spec["family"]
    if family not in family_expectations:
        raise AssertionError(f"{context} unsupported family {family!r}")
    expectation = family_expectations[family]
    _require_equal(spec["tuple_id"], expectation["tuple_id"], f"{context} tuple_id")
    _require_equal(
        spec["metadata"]["materialization_policy"],
        expectation["materialization_policy"],
        f"{context} materialization policy",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    inputs = envelope["inputs"]
    require_fields(
        inputs,
        ("input", "running_mean", "running_var", "weight", "bias"),
        f"{context} inputs",
    )
    bounds = envelope["bounds"]
    for input_name in ("input", "running_mean", "running_var", "weight", "bias"):
        input_spec = inputs[input_name]
        expected_dtype = (
            bounds["input_dtype"]
            if input_name == "input"
            else bounds["parameter_dtype"]
        )
        expected_rank = (
            bounds["input_rank"]
            if input_name == "input"
            else bounds["parameter_rank"]
        )
        _require_equal(
            _single_value(input_spec["dtype"], f"{context} {input_name} dtype"),
            expected_dtype,
            f"{context} {input_name} dtype",
        )
        _require_equal(
            _single_value(input_spec["rank"], f"{context} {input_name} rank"),
            expected_rank,
            f"{context} {input_name} rank",
        )

    input_dims = _dims_by_symbol(inputs["input"], f"{context} input")
    for symbol in ("N", "C", "H", "W"):
        if symbol not in input_dims:
            raise AssertionError(f"{context} missing input dim {symbol}")
    for input_name in ("running_mean", "running_var", "weight", "bias"):
        dims = _dims_by_symbol(inputs[input_name], f"{context} {input_name}")
        if "C" not in dims:
            raise AssertionError(f"{context} missing {input_name} dim C")

    attributes = envelope["attributes"]
    _require_equal(
        _single_value(attributes["training"], f"{context} training"),
        bounds["training"],
        f"{context} training",
    )
    _require_equal(
        sorted(attributes["weight_has_value"]["values"]),
        [False, True],
        f"{context} weight optional",
    )
    _require_equal(
        sorted(attributes["bias_has_value"]["values"]),
        [False, True],
        f"{context} bias optional",
    )

    relationships = _relationship_types(envelope)
    if "equal" not in relationships:
        raise AssertionError(f"{context} missing feature-count relationship")
    layout = envelope["layout"]
    for field in (
        "requires_vulkan",
        "requires_contiguous",
        "requires_buffer_storage",
    ):
        _require_equal(layout[field], bounds[field], f"{context} {field}")
    _require_equal(
        envelope["capability_requirements"]["requires_buffer_compute"],
        bounds["requires_buffer_compute"],
        f"{context} requires_buffer_compute",
    )
    _require_equal(
        bounds["requires_buffer_storage"],
        expectation["requires_buffer_storage"],
        f"{context} requires_buffer_storage",
    )
    if "requires_materialization" in expectation:
        _require_equal(
            bounds.get("requires_materialization"),
            expectation["requires_materialization"],
            f"{context} requires_materialization",
        )

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            if len(case["input_shape"]) != bounds["input_rank"]:
                if case.get("violates") != "input_rank":
                    raise AssertionError(f"{case_context} input rank mismatch")
            if case["dtype"] != bounds["input_dtype"]:
                raise AssertionError(f"{case_context} dtype mismatch")
            if case["parameter_features"] != case["input_shape"][1]:
                if case.get("violates") != "feature_count.equal":
                    raise AssertionError(f"{case_context} feature count mismatch")


def _validate_no_overlap_conv_transpose2d_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} NoOverlapConvTranspose2D ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "NoOverlapConvTranspose2DContract",
        f"{context} contract",
    )
    _require_equal(
        spec["family"],
        "Kernel2Stride2FloatBuffer",
        f"{context} family",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    bounds = envelope["bounds"]
    inputs = envelope["inputs"]
    require_fields(inputs, ("input", "weight"), f"{context} inputs")
    input_spec = inputs["input"]
    weight_spec = inputs["weight"]
    _require_equal(
        _single_value(input_spec["dtype"], f"{context} input dtype"),
        bounds["input_dtype"],
        f"{context} input dtype",
    )
    _require_equal(
        _single_value(weight_spec["dtype"], f"{context} weight dtype"),
        bounds["weight_dtype"],
        f"{context} weight dtype",
    )
    _require_equal(
        _single_value(input_spec["rank"], f"{context} input rank"),
        bounds["input_rank"],
        f"{context} input rank",
    )
    _require_equal(
        _single_value(weight_spec["rank"], f"{context} weight rank"),
        bounds["weight_rank"],
        f"{context} weight rank",
    )

    input_dims = _dims_by_symbol(input_spec, f"{context} input")
    weight_dims = _dims_by_symbol(weight_spec, f"{context} weight")
    _require_equal(
        _single_value(input_dims["N"], f"{context} batch"),
        bounds["batch"],
        f"{context} batch",
    )
    _require_equal(
        input_dims["CI"]["min"],
        bounds["input_channels"]["min"],
        f"{context} input channel min",
    )
    _require_equal(
        weight_dims["CI"]["min"],
        bounds["input_channels"]["min"],
        f"{context} weight input channel min",
    )
    _require_equal(
        _single_value(weight_dims["KH"], f"{context} kernel_h"),
        bounds["kernel_h"],
        f"{context} kernel_h",
    )
    _require_equal(
        _single_value(weight_dims["KW"], f"{context} kernel_w"),
        bounds["kernel_w"],
        f"{context} kernel_w",
    )

    attributes = envelope["attributes"]
    for attribute_name in (
        "transposed",
        "options_quantized",
        "groups",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "output_padding_is_zero",
        "packed_quantized",
        "execution_is_buffer_direct",
        "bias_is_float",
    ):
        _require_equal(
            _single_value(attributes[attribute_name], f"{context} {attribute_name}"),
            bounds[attribute_name],
            f"{context} {attribute_name}",
        )

    relationships = _relationship_types(envelope)
    if "equal" not in relationships:
        raise AssertionError(f"{context} missing input/weight channel relationship")
    layout = envelope["layout"]
    _require_equal(
        layout["requires_vulkan"],
        bounds["requires_vulkan"],
        f"{context} requires_vulkan",
    )
    _require_equal(layout["input_storage"], "buffer", f"{context} input storage")
    _require_equal(layout["weight_storage"], "buffer", f"{context} weight storage")
    _require_equal(layout["bias_storage"], "buffer", f"{context} bias storage")
    _require_equal(
        layout["execution_storage"],
        "buffer_direct",
        f"{context} execution storage",
    )
    _require_equal(
        envelope["capability_requirements"]["input_supports_buffer_compute"],
        bounds["input_supports_buffer_compute"],
        f"{context} input supports buffer compute",
    )

    def require_pair(case, field, height_key, width_key, allowed_violation):
        if case.get("violates") == allowed_violation:
            return
        expected = [bounds[height_key], bounds[width_key]]
        if case[field] != expected:
            raise AssertionError(
                f"{context} {case['name']} {field} mismatch: "
                f"{case[field]} != {expected}"
            )

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            if len(case["input_shape"]) != bounds["input_rank"]:
                raise AssertionError(f"{case_context} input rank mismatch")
            if case["input_shape"][0] != bounds["batch"]:
                raise AssertionError(f"{case_context} batch mismatch")
            if (
                case.get("violates") != "input_channels.min" and
                case["input_shape"][1] < bounds["input_channels"]["min"]
            ):
                raise AssertionError(f"{case_context} input channels below min")
            if case["dtype"] != bounds["input_dtype"]:
                raise AssertionError(f"{case_context} dtype mismatch")
            if case["groups"] != bounds["groups"]:
                raise AssertionError(f"{case_context} groups mismatch")
            require_pair(case, "kernel_size", "kernel_h", "kernel_w", "kernel")
            require_pair(case, "stride", "stride_h", "stride_w", "stride")
            require_pair(case, "padding", "padding_h", "padding_w", "padding")
            require_pair(case, "dilation", "dilation_h", "dilation_w", "dilation")
            if (
                case.get("violates") != "output_padding" and
                case["output_padding"] != [0, 0]
            ):
                raise AssertionError(f"{case_context} output padding mismatch")


def _broadcast_output_shape(left, right):
    result = []
    max_rank = max(len(left), len(right))
    for offset in range(max_rank):
        left_index = len(left) - 1 - offset
        right_index = len(right) - 1 - offset
        left_dim = left[left_index] if left_index >= 0 else 1
        right_dim = right[right_index] if right_index >= 0 else 1
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            return None
        result.insert(0, max(left_dim, right_dim))
    return result


def _validate_elementwise_broadcast_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} ElementwiseBroadcast ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "ElementwiseBroadcastContract",
        f"{context} contract",
    )
    _require_equal(
        spec["family"],
        "FloatTensorTensorBufferBroadcast",
        f"{context} family",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    inputs = envelope["inputs"]
    require_fields(inputs, ("self", "other"), f"{context} inputs")
    bounds = envelope["bounds"]
    for input_name in ("self", "other"):
        input_spec = inputs[input_name]
        _require_equal(
            _single_value(input_spec["dtype"], f"{context} {input_name} dtype"),
            bounds["dtype"],
            f"{context} {input_name} dtype",
        )
        _require_equal(
            input_spec["rank"],
            bounds["rank"],
            f"{context} {input_name} rank",
        )

    results = envelope.get("results", {})
    require_fields(results, ("output",), f"{context} results")
    _require_equal(
        _single_value(results["output"]["dtype"], f"{context} output dtype"),
        bounds["dtype"],
        f"{context} output dtype",
    )
    _require_equal(
        results["output"]["rank"],
        bounds["rank"],
        f"{context} output rank",
    )

    attributes = envelope["attributes"]
    _require_equal(
        attributes["op"]["values"],
        bounds["ops"],
        f"{context} ops",
    )
    _require_equal(
        _single_value(attributes["alpha"], f"{context} alpha"),
        bounds["alpha"],
        f"{context} alpha",
    )
    _require_equal(
        _single_value(attributes["inplace"], f"{context} inplace"),
        False,
        f"{context} inplace",
    )
    _require_equal(
        _single_value(attributes["has_out"], f"{context} has_out"),
        False,
        f"{context} has_out",
    )

    relationships = [
        relationship
        for relationship in envelope["relationships"]
        if relationship["type"] == "broadcast_compatible"
    ]
    if len(relationships) != 1:
        raise AssertionError(f"{context} must have one broadcast_compatible relation")
    relationship = relationships[0]
    _require_equal(relationship["left"], "self", f"{context} broadcast left")
    _require_equal(relationship["right"], "other", f"{context} broadcast right")
    _require_equal(relationship["result"], "output", f"{context} broadcast result")
    _require_equal(relationship["align"], "right", f"{context} broadcast align")
    _require_equal(relationship["max_rank"], 4, f"{context} broadcast max_rank")

    layout = envelope["layout"]
    for field in (
        "requires_vulkan",
        "requires_buffer_storage",
        "requires_buffer_compute",
    ):
        _require_equal(layout[field], bounds[field], f"{context} {field}")
    _require_equal(
        envelope["capability_requirements"]["requires_buffer_compute"],
        bounds["requires_buffer_compute"],
        f"{context} requires_buffer_compute",
    )

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            if case["dtype"] != bounds["dtype"]:
                if case.get("violates") != "dtype":
                    raise AssertionError(f"{case_context} dtype mismatch")
            if case["op"] not in bounds["ops"]:
                if case.get("violates") != "op":
                    raise AssertionError(f"{case_context} op mismatch")
            output_shape = _broadcast_output_shape(
                case["self_shape"],
                case["other_shape"],
            )
            if case.get("violates") == "broadcast_compatible":
                if output_shape is not None:
                    raise AssertionError(
                        f"{case_context} unexpectedly broadcast-compatible"
                    )
            else:
                if output_shape is None:
                    raise AssertionError(f"{case_context} is not broadcast-compatible")
                _require_equal(
                    case["output_shape"],
                    output_shape,
                    f"{case_context} output shape",
                )
            if max(len(case["self_shape"]), len(case["other_shape"])) > bounds["rank"]["max"]:
                if case.get("violates") != "rank.max":
                    raise AssertionError(f"{case_context} rank exceeds bounds")


def validate_shape_envelope_spec(file_name, spec):
    envelope = spec.get("shape_envelope")
    if envelope is None:
        return None
    _validate_shape_envelope_common(file_name, spec, envelope)
    role = envelope["role"]
    _shape_envelope_role_adapter(role, file_name)["validate"](
        file_name,
        spec,
        envelope,
    )
    return envelope


def _channel_cat_base_shape(bounds):
    return [
        bounds["batch"],
        max(bounds["channels"]["min"], bounds["channels"]["multiple_of"] * 4),
        min(max(bounds["height"]["min"], 5), bounds["height"]["max"]),
        min(max(bounds["width"]["min"], 7), bounds["width"]["max"]),
    ]


def _channel_cat_expected_negative_policy():
    return {
        "expected_native_route": False,
        "expected_cpu_fallback": True,
    }


def _generated_channel_cat_legal_cases(spec):
    envelope = spec["shape_envelope"]
    bounds = envelope["bounds"]
    base_shape = _channel_cat_base_shape(bounds)
    min_input_count = bounds["input_count"]["min"]
    max_input_count = bounds["input_count"]["max"]
    min_channels = bounds["channels"]["min"]
    max_per_input_channels = bounds["channels"]["max_per_input"]
    max_total_channels = bounds["channels"]["max_total"]

    def make_case(name, input_shapes, dim=None):
        return {
            "name": name,
            "input_shapes": input_shapes,
            "dim": bounds["dim"] if dim is None else dim,
        }

    return [
        make_case(
            "generated_min_three_inputs",
            [
                [
                    bounds["batch"],
                    min_channels,
                    bounds["height"]["min"],
                    bounds["width"]["min"],
                ]
                for _ in range(min_input_count)
            ],
        ),
        make_case(
            "generated_interior_six_inputs",
            [list(base_shape) for _ in range(min(max_input_count, 6))],
        ),
        make_case(
            "generated_max_eight_inputs",
            [
                [
                    bounds["batch"],
                    min_channels,
                    min(bounds["height"]["min"] + 1, bounds["height"]["max"]),
                    min(bounds["width"]["min"] + 2, bounds["width"]["max"]),
                ]
                for _ in range(max_input_count)
            ],
        ),
        make_case(
            "generated_max_per_input_and_total_channels",
            [
                [
                    bounds["batch"],
                    max_per_input_channels,
                    min(bounds["height"]["min"] + 1, bounds["height"]["max"]),
                    min(bounds["width"]["min"] + 1, bounds["width"]["max"]),
                ]
                for _ in range(max_total_channels // max_per_input_channels)
            ],
        ),
        make_case(
            "generated_max_height",
            [
                [
                    bounds["batch"],
                    min_channels,
                    bounds["height"]["max"],
                    min(max(bounds["width"]["min"], 7), bounds["width"]["max"]),
                ]
                for _ in range(min_input_count)
            ],
        ),
    ]


def _generated_channel_cat_adjacent_negative_cases(spec):
    envelope = spec["shape_envelope"]
    bounds = envelope["bounds"]
    axes = _shape_envelope_negative_axes(envelope)
    base_shape = _channel_cat_base_shape(bounds)
    base_input_count = bounds["input_count"]["min"]
    cases = []

    def add_case(name, violates, input_shapes, dim=None):
        case = {
            "name": name,
            "input_shapes": input_shapes,
            "dim": bounds["dim"] if dim is None else dim,
            "violates": violates,
        }
        case.update(_channel_cat_expected_negative_policy())
        cases.append(case)

    if "input_count" in axes:
        value = axes["input_count"]["value"]
        add_case(
            "generated_input_count",
            "input_count",
            [list(base_shape) for _ in range(value)],
        )
    if "channels.multiple_of" in axes:
        value = axes["channels.multiple_of"]["value"]
        shape = list(base_shape)
        shape[1] = value
        add_case(
            "generated_channels_multiple_of",
            "channels.multiple_of",
            [list(shape) for _ in range(base_input_count)],
        )
    if "channels.max_per_input" in axes:
        value = axes["channels.max_per_input"]["value"]
        shape = list(base_shape)
        shape[1] = value
        add_case(
            "generated_channels_max_per_input",
            "channels.max_per_input",
            [list(shape) for _ in range(base_input_count)],
        )
    if "channels.max_total" in axes:
        value = axes["channels.max_total"]["value"]
        per_input = bounds["channels"]["max_per_input"]
        input_count = value // per_input
        if value % per_input != 0:
            input_count += 1
        shape = [
            bounds["batch"],
            per_input,
            bounds["height"]["min"] + 1,
            bounds["width"]["min"] + 2,
        ]
        add_case(
            "generated_channels_max_total",
            "channels.max_total",
            [list(shape) for _ in range(input_count)],
        )
    if "height.max" in axes:
        value = axes["height.max"]["value"]
        shape = list(base_shape)
        shape[2] = value
        add_case(
            "generated_height_max",
            "height.max",
            [list(shape) for _ in range(base_input_count)],
        )
    if "width.max" in axes:
        value = axes["width.max"]["value"]
        shape = list(base_shape)
        shape[3] = value
        add_case(
            "generated_width_max",
            "width.max",
            [list(shape) for _ in range(base_input_count)],
        )
    if "dim" in axes:
        value = axes["dim"]["value"]
        add_case(
            "generated_dim",
            "dim",
            [list(base_shape) for _ in range(base_input_count)],
            dim=value,
        )
    return cases


def _embedding_expected_negative_policy():
    return {
        "expected_native_route": False,
        "expected_sync_readback": True,
    }


def _generated_embedding_lookup_legal_cases(spec):
    envelope = spec["shape_envelope"]
    bounds = envelope["bounds"]
    indices_dtype = bounds["indices_dtype"]
    max_num_indices = bounds["num_indices"]["max"]

    def make_case(
        name,
        num_embeddings,
        embedding_dim,
        indices_shape,
    ):
        return {
            "name": name,
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "indices_shape": indices_shape,
            "indices_dtype": indices_dtype,
            "padding_idx": -1,
        }

    return [
        make_case(
            "generated_min_rank2",
            bounds["num_embeddings"]["min"],
            bounds["embedding_dim"]["min"],
            [1, 1],
        ),
        make_case(
            "generated_interior_rank2",
            min(
                max(bounds["num_embeddings"]["min"], 64),
                bounds["num_embeddings"]["max"],
            ),
            min(
                max(bounds["embedding_dim"]["min"], 24),
                bounds["embedding_dim"]["max"],
            ),
            [1, min(max(bounds["num_indices"]["min"], 8), max_num_indices)],
        ),
        make_case(
            "generated_boundary_rank2",
            bounds["num_embeddings"]["max"],
            bounds["embedding_dim"]["max"],
            [1, max_num_indices],
        ),
        make_case(
            "generated_boundary_rank1",
            bounds["num_embeddings"]["max"],
            bounds["embedding_dim"]["max"],
            [max_num_indices],
        ),
    ]


def _generated_embedding_lookup_adjacent_negative_cases(spec):
    envelope = spec["shape_envelope"]
    bounds = envelope["bounds"]
    axes = _shape_envelope_negative_axes(envelope)
    cases = []

    def add_case(
        name,
        violates,
        num_embeddings=None,
        embedding_dim=None,
        indices_shape=None,
        indices_dtype=None,
    ):
        case = {
            "name": name,
            "num_embeddings": (
                bounds["num_embeddings"]["max"]
                if num_embeddings is None
                else num_embeddings
            ),
            "embedding_dim": (
                bounds["embedding_dim"]["max"]
                if embedding_dim is None
                else embedding_dim
            ),
            "indices_shape": [1, 8] if indices_shape is None else indices_shape,
            "indices_dtype": (
                bounds["indices_dtype"] if indices_dtype is None else indices_dtype
            ),
            "padding_idx": -1,
            "violates": violates,
        }
        case.update(_embedding_expected_negative_policy())
        cases.append(case)

    if "num_indices" in axes:
        value = axes["num_indices"]["value"]
        add_case(
            "generated_num_indices",
            "num_indices",
            indices_shape=[1, value],
        )
    if "embedding_dim" in axes:
        value = axes["embedding_dim"]["value"]
        add_case(
            "generated_embedding_dim",
            "embedding_dim",
            embedding_dim=value,
        )
    if "num_embeddings" in axes:
        value = axes["num_embeddings"]["value"]
        add_case(
            "generated_num_embeddings",
            "num_embeddings",
            num_embeddings=value,
        )
    if "indices_dtype" in axes:
        value = axes["indices_dtype"]["value"]
        add_case(
            "generated_indices_dtype",
            "indices_dtype",
            indices_dtype=value,
        )
    return cases


def _safe_view_reshape_case(
    name,
    input_shape,
    output_shape,
    storage_offset=0,
    violates=None,
):
    case = {
        "name": name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "output_stride": _contiguous_strides(output_shape),
        "storage_offset": storage_offset,
    }
    if violates is not None:
        case["violates"] = violates
        case["expected_native_route"] = False
    return case


def _generated_safe_view_reshape_legal_cases(spec):
    envelope = spec["shape_envelope"]
    hints = envelope["fuzz_hints"]
    groups = hints["groups"]
    height, width = hints["spatial"]
    cases = []
    for channels in hints["channels"]:
        output_last_dim = channels * height * width // groups
        cases.append(
            _safe_view_reshape_case(
                f"generated_lotus_view_{channels}_groups{groups}",
                [1, channels, height, width],
                [1, groups, output_last_dim],
            )
        )
    return cases


def _generated_safe_view_reshape_adjacent_negative_cases(spec):
    axes = _shape_envelope_negative_axes(spec["shape_envelope"])
    cases = []
    if "input_rank.max" in axes:
        cases.append(
            _safe_view_reshape_case(
                "generated_input_rank_max",
                [1, 1, 1, 1, 16],
                [1, 4, 4],
                violates="input_rank.max",
            )
        )
    if "output_rank.max" in axes:
        cases.append(
            _safe_view_reshape_case(
                "generated_output_rank_max",
                [16],
                [1, 1, 1, 1, 4, 4],
                violates="output_rank.max",
            )
        )
    if "output_last_dim_multiple_of" in axes:
        cases.append(
            _safe_view_reshape_case(
                "generated_output_last_dim_multiple_of",
                [1, 12, 1, 1],
                [1, 4, 3],
                violates="output_last_dim_multiple_of",
            )
        )
    return cases


def _safe_view_reshape_alias_case(
    name,
    input_shape,
    input_stride,
    output_shape,
    output_stride,
    storage_offset=0,
    source_shape=None,
    input_permute=None,
    violates=None,
):
    case = {
        "name": name,
        "input_shape": input_shape,
        "input_stride": input_stride,
        "output_shape": output_shape,
        "output_stride": output_stride,
        "storage_offset": storage_offset,
    }
    if source_shape is not None:
        case["source_shape"] = source_shape
    if input_permute is not None:
        case["input_permute"] = input_permute
    if violates is not None:
        case["violates"] = violates
        case["expected_native_route"] = False
    return case


def _safe_view_reshape_alias_lotus_case(name, channels, height, width):
    source_shape = [1, channels, height, width]
    source_stride = _contiguous_strides(source_shape)
    input_permute = [0, 2, 3, 1]
    input_shape = [source_shape[index] for index in input_permute]
    input_stride = [source_stride[index] for index in input_permute]
    output_shape = [1, height * width, channels]
    output_stride = [height * width * channels, 1, height * width]
    return _safe_view_reshape_alias_case(
        name,
        input_shape,
        input_stride,
        output_shape,
        output_stride,
        source_shape=source_shape,
        input_permute=input_permute,
    )


def _generated_safe_view_reshape_alias_legal_cases(spec):
    cases = []
    for hint in spec["shape_envelope"]["fuzz_hints"]["lotus_cases"]:
        channels = hint["channels"]
        height = hint["height"]
        width = hint["width"]
        cases.append(
            _safe_view_reshape_alias_lotus_case(
                f"generated_lotus_alias_{channels}_{height}_{width}",
                channels,
                height,
                width,
            )
        )
    return cases


def _generated_safe_view_reshape_alias_adjacent_negative_cases(spec):
    axes = _shape_envelope_negative_axes(spec["shape_envelope"])
    cases = []
    if "input_rank.max" in axes:
        cases.append(
            _safe_view_reshape_alias_case(
                "generated_input_rank_max",
                [1, 1, 1, 1, 16],
                [16, 16, 16, 16, 1],
                [1, 4, 4],
                [16, 1, 4],
                violates="input_rank.max",
            )
        )
    if "output_rank.max" in axes:
        cases.append(
            _safe_view_reshape_alias_case(
                "generated_output_rank_max",
                [16],
                [1],
                [1, 1, 1, 1, 4, 4],
                [16, 16, 16, 16, 1, 4],
                violates="output_rank.max",
            )
        )
    if "output_last_dim_multiple_of" in axes:
        cases.append(
            _safe_view_reshape_alias_case(
                "generated_output_last_dim_multiple_of",
                [1, 12, 1, 1],
                [12, 1, 1, 1],
                [1, 4, 3],
                [12, 1, 4],
                violates="output_last_dim_multiple_of",
            )
        )
    if "output_stride_policy" in axes:
        cases.append(
            _safe_view_reshape_alias_case(
                "generated_output_stride_policy",
                [1, 16, 1, 1],
                [16, 1, 1, 1],
                [1, 4, 4],
                [16, 1, 1],
                violates="output_stride_policy",
            )
        )
    return cases


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


_BATCH_NORM_LEGAL_KEY_FIELDS = (
    "input_shape",
    "parameter_features",
    "dtype",
    "training",
    "has_weight",
    "has_bias",
    ("materialized_input", False),
    "expected_route_label",
    "expected_cpu_fallback",
    ("expected_contract_family", None),
    ("expected_contract_tuple_id", None),
    ("expected_contract_materialization_policy", None),
)

_BATCH_NORM_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "input_shape",
    "parameter_features",
    "dtype",
    "training",
    "has_weight",
    "has_bias",
    ("materialized_input", False),
    "expected_native_route",
    "expected_cpu_fallback",
    ("expected_error_regex", ""),
)

_BATCH_NORM_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.bias.dtype",
    "inputs.bias.rank",
    "inputs.bias.dims.C",
    "inputs.input.dtype",
    "inputs.input.rank",
    "inputs.input.dims.N",
    "inputs.input.dims.C",
    "inputs.input.dims.H",
    "inputs.input.dims.W",
    "inputs.running_mean.dtype",
    "inputs.running_mean.rank",
    "inputs.running_mean.dims.C",
    "inputs.running_var.dtype",
    "inputs.running_var.rank",
    "inputs.running_var.dims.C",
    "inputs.weight.dtype",
    "inputs.weight.rank",
    "inputs.weight.dims.C",
    "attributes.bias_has_value",
    "attributes.training",
    "attributes.weight_has_value",
)

_ELEMENTWISE_BROADCAST_LEGAL_KEY_FIELDS = (
    "self_shape",
    "other_shape",
    "output_shape",
    "op",
    "dtype",
    "expected_route_label",
    "expected_tensor_provenance_route",
    "expected_cpu_fallback",
    "expected_sync_readback",
)

_ELEMENTWISE_BROADCAST_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "self_shape",
    "other_shape",
    "output_shape",
    "op",
    "dtype",
    "expected_native_route",
    "expected_cpu_fallback",
    "expected_sync_readback",
    ("expected_error_regex", ""),
)

_ELEMENTWISE_BROADCAST_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.other.dtype",
    "inputs.other.rank",
    "inputs.other.dims.D0",
    "inputs.other.dims.D1",
    "inputs.other.dims.D2",
    "inputs.other.dims.D3",
    "inputs.self.dtype",
    "inputs.self.rank",
    "inputs.self.dims.D0",
    "inputs.self.dims.D1",
    "inputs.self.dims.D2",
    "inputs.self.dims.D3",
    "results.output.dtype",
    "results.output.rank",
    "results.output.dims.D0",
    "results.output.dims.D1",
    "results.output.dims.D2",
    "results.output.dims.D3",
    "attributes.alpha",
    "attributes.has_out",
    "attributes.inplace",
    "attributes.op",
    "relationships.broadcast_compatible.left",
    "relationships.broadcast_compatible.right",
    "relationships.broadcast_compatible.result",
    "relationships.broadcast_compatible.max_rank",
)

_NO_OVERLAP_CONV_TRANSPOSE2D_LEGAL_KEY_FIELDS = (
    "input_shape",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "output_padding",
    "bias",
    "dtype",
    "expected_route_label",
    "expected_cpu_fallback",
)

_NO_OVERLAP_CONV_TRANSPOSE2D_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "input_shape",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "output_padding",
    "bias",
    "dtype",
    "expected_native_route",
)

_NO_OVERLAP_CONV_TRANSPOSE2D_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.input.dtype",
    "inputs.input.rank",
    "inputs.input.dims.N",
    "inputs.input.dims.CI",
    "inputs.input.dims.H",
    "inputs.input.dims.W",
    "inputs.weight.dtype",
    "inputs.weight.rank",
    "inputs.weight.dims.CI",
    "inputs.weight.dims.CO",
    "inputs.weight.dims.KH",
    "inputs.weight.dims.KW",
    "attributes.transposed",
    "attributes.options_quantized",
    "attributes.groups",
    "attributes.kernel_h",
    "attributes.kernel_w",
    "attributes.stride_h",
    "attributes.stride_w",
    "attributes.padding_h",
    "attributes.padding_w",
    "attributes.dilation_h",
    "attributes.dilation_w",
    "attributes.output_padding_is_zero",
    "attributes.packed_quantized",
    "attributes.execution_is_buffer_direct",
    "attributes.bias_is_float",
)


SHAPE_ENVELOPE_ROLE_REGISTRY = {
    "batch_norm_inference_buffer_float_4d": {
        "validate": _validate_batch_norm_inference_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _BATCH_NORM_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": _BATCH_NORM_ASSIGNMENT_COVERAGE_FIELDS,
        "adjacent_negative_key_fields": _BATCH_NORM_ADJACENT_NEGATIVE_KEY_FIELDS,
    },
    "batch_norm_inference_materialized_buffer_float_4d": {
        "validate": _validate_batch_norm_inference_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _BATCH_NORM_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": _BATCH_NORM_ASSIGNMENT_COVERAGE_FIELDS,
        "adjacent_negative_key_fields": _BATCH_NORM_ADJACENT_NEGATIVE_KEY_FIELDS,
    },
    "elementwise_float_tensor_tensor_buffer_broadcast": {
        "validate": _validate_elementwise_broadcast_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _ELEMENTWISE_BROADCAST_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": (
            _ELEMENTWISE_BROADCAST_ASSIGNMENT_COVERAGE_FIELDS
        ),
        "adjacent_negative_key_fields": (
            _ELEMENTWISE_BROADCAST_ADJACENT_NEGATIVE_KEY_FIELDS
        ),
    },
    "no_overlap_conv_transpose2d_kernel2_stride2_float_buffer": {
        "validate": _validate_no_overlap_conv_transpose2d_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _NO_OVERLAP_CONV_TRANSPOSE2D_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": (
            _NO_OVERLAP_CONV_TRANSPOSE2D_ASSIGNMENT_COVERAGE_FIELDS
        ),
        "adjacent_negative_key_fields": (
            _NO_OVERLAP_CONV_TRANSPOSE2D_ADJACENT_NEGATIVE_KEY_FIELDS
        ),
    },
    "multi_input_rank4_channel_cat": {
        "validate": _validate_channel_cat_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _generated_channel_cat_legal_cases,
        "adjacent_negative_cases": _generated_channel_cat_adjacent_negative_cases,
        "legal_key_fields": (
            "input_shapes",
            "dim",
        ),
        "assignment_coverage_fields": (
            "inputs.tensors.count",
            "inputs.tensors.dtype",
            "inputs.tensors.rank",
            "inputs.tensors.dims.N",
            "inputs.tensors.dims.C",
            "inputs.tensors.dims.H",
            "inputs.tensors.dims.W",
            "attributes.dim",
        ),
        "adjacent_negative_key_fields": (
            "violates",
            {
                "by_violates": {
                    "input_count": {
                        "op": "len",
                        "field": "input_shapes",
                    },
                    "channels.multiple_of": {
                        "op": "shape_dim",
                        "field": "input_shapes",
                        "dim": 1,
                    },
                    "channels.max_per_input": {
                        "op": "shape_dim",
                        "field": "input_shapes",
                        "dim": 1,
                    },
                    "channels.max_total": {
                        "op": "sum_shape_dim",
                        "field": "input_shapes",
                        "dim": 1,
                    },
                    "height.max": {
                        "op": "shape_dim",
                        "field": "input_shapes",
                        "dim": 2,
                    },
                    "width.max": {
                        "op": "shape_dim",
                        "field": "input_shapes",
                        "dim": 3,
                    },
                    "dim": {
                        "op": "field",
                        "field": "dim",
                    },
                }
            },
            "expected_native_route",
            "expected_cpu_fallback",
        ),
    },
    "embedding_lookup_small_bounded": {
        "validate": _validate_embedding_lookup_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _generated_embedding_lookup_legal_cases,
        "adjacent_negative_cases": _generated_embedding_lookup_adjacent_negative_cases,
        "legal_key_fields": (
            "num_embeddings",
            "embedding_dim",
            "indices_shape",
            "indices_dtype",
            "padding_idx",
            ("scale_grad_by_freq", False),
            ("sparse", False),
        ),
        "assignment_coverage_fields": (
            "inputs.indices.dtype",
            "inputs.indices.rank",
            "inputs.indices.dims.I0",
            "inputs.indices.dims.I1",
            "inputs.weight.dtype",
            "inputs.weight.rank",
            "inputs.weight.dims.V",
            "inputs.weight.dims.D",
            "attributes.padding_idx_has_hint",
            "attributes.scale_grad_by_freq",
            "attributes.sparse",
        ),
        "adjacent_negative_key_fields": (
            "violates",
            {
                "by_violates": {
                    "num_indices": {
                        "op": "product",
                        "field": "indices_shape",
                    },
                    "embedding_dim": {
                        "op": "field",
                        "field": "embedding_dim",
                    },
                    "num_embeddings": {
                        "op": "field",
                        "field": "num_embeddings",
                    },
                    "indices_dtype": {
                        "op": "field",
                        "field": "indices_dtype",
                    },
                }
            },
            "expected_native_route",
            "expected_sync_readback",
        ),
    },
    "safe_view_materialized_direct_buffer": {
        "validate": _validate_safe_view_reshape_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _generated_safe_view_reshape_legal_cases,
        "adjacent_negative_cases": _generated_safe_view_reshape_adjacent_negative_cases,
        "legal_key_fields": (
            "input_shape",
            "output_shape",
            "output_stride",
            "storage_offset",
        ),
        "assignment_coverage_fields": (
            "inputs.input.rank",
            "inputs.output.rank",
            "attributes.storage_offset",
            "attributes.output_last_dim_multiple_of",
            "attributes.output_stride_policy",
        ),
        "adjacent_negative_key_fields": (
            "violates",
            "input_shape",
            "output_shape",
            "output_stride",
            "storage_offset",
            "expected_native_route",
        ),
    },
    "safe_reshape_alias_dense_buffer_direct": {
        "validate": _validate_safe_view_reshape_alias_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _generated_safe_view_reshape_alias_legal_cases,
        "adjacent_negative_cases": (
            _generated_safe_view_reshape_alias_adjacent_negative_cases
        ),
        "legal_key_fields": (
            "input_shape",
            "input_stride",
            "output_shape",
            "output_stride",
            "storage_offset",
        ),
        "assignment_coverage_fields": (
            "inputs.input.dtype",
            "inputs.input.rank",
            "inputs.output.rank",
            "attributes.storage_offset",
            "attributes.input_stride_policy",
            "attributes.output_last_dim_multiple_of",
            "attributes.output_stride_policy",
        ),
        "adjacent_negative_key_fields": (
            "violates",
            "input_shape",
            "input_stride",
            "output_shape",
            "output_stride",
            "storage_offset",
            "expected_native_route",
        ),
    },
}


def shape_envelope_role_registry():
    return SHAPE_ENVELOPE_ROLE_REGISTRY


def _shape_envelope_role_adapter(role, file_name=None):
    if role in SHAPE_ENVELOPE_ROLE_REGISTRY:
        return SHAPE_ENVELOPE_ROLE_REGISTRY[role]
    context = file_name if file_name is not None else "ShapeEnvelope"
    raise AssertionError(f"{context} unsupported ShapeEnvelope role {role!r}")


def generated_shape_envelope_legal_cases(spec):
    envelope = spec.get("shape_envelope")
    if envelope is None:
        return []
    return _shape_envelope_role_adapter(envelope["role"])["legal_cases"](spec)


def generated_shape_envelope_adjacent_negative_cases(spec):
    envelope = spec.get("shape_envelope")
    if envelope is None:
        return []
    return _shape_envelope_role_adapter(envelope["role"])[
        "adjacent_negative_cases"
    ](spec)


def generated_shape_envelope_assignment_cases(spec):
    envelope = spec.get("shape_envelope")
    if envelope is None:
        return {
            "legal_assignments": [],
            "adjacent_negative_assignments": [],
        }
    return _shape_envelope_role_adapter(envelope["role"])["assignment_cases"](spec)


def _legal_case_key(spec, case):
    adapter = _shape_envelope_role_adapter(spec["shape_envelope"]["role"])
    return _shape_envelope_case_key(case, adapter["legal_key_fields"])


def _adjacent_negative_key(spec, case):
    adapter = _shape_envelope_role_adapter(spec["shape_envelope"]["role"])
    return _shape_envelope_case_key(case, adapter["adjacent_negative_key_fields"])


def validate_generated_shape_envelope_legal_cases(file_name, spec):
    if "shape_envelope" not in spec:
        return []
    generated_cases = generated_shape_envelope_legal_cases(spec)
    generated_keys = {
        _legal_case_key(spec, case)
        for case in generated_cases
    }
    checked_in_keys = {
        _legal_case_key(spec, case)
        for case in spec["positive_cases"]
    }
    if generated_keys != checked_in_keys:
        missing = sorted(checked_in_keys - generated_keys)
        extra = sorted(generated_keys - checked_in_keys)
        raise AssertionError(
            f"{file_name} generated legal cases mismatch "
            f"missing={missing} extra={extra}"
        )
    return generated_cases


def validate_generated_shape_envelope_adjacent_negatives(file_name, spec):
    if "shape_envelope" not in spec:
        return []
    generated_cases = generated_shape_envelope_adjacent_negative_cases(spec)
    generated_keys = {
        _adjacent_negative_key(spec, case)
        for case in generated_cases
    }
    checked_in_keys = {
        _adjacent_negative_key(spec, case)
        for case in spec["negative_cases"]
    }
    if generated_keys != checked_in_keys:
        missing = sorted(checked_in_keys - generated_keys)
        extra = sorted(generated_keys - checked_in_keys)
        raise AssertionError(
            f"{file_name} generated adjacent negatives mismatch "
            f"missing={missing} extra={extra}"
        )
    return generated_cases


def contract_spec_dir(repo_root):
    return os.path.join(repo_root, "test", "vulkan_contract_specs")


def contract_spec_paths(repo_root):
    return sorted(
        path
        for path in glob.glob(os.path.join(contract_spec_dir(repo_root), "*.json"))
        if os.path.basename(path) != GENERATED_CPP_MANIFEST_FILE
    )


def load_contract_spec(repo_root, file_name):
    path = os.path.join(contract_spec_dir(repo_root), file_name)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_all_contract_specs(repo_root):
    return [
        (os.path.basename(path), load_contract_spec(repo_root, os.path.basename(path)))
        for path in contract_spec_paths(repo_root)
    ]


def require_fields(mapping, required_fields, context):
    missing = sorted(field for field in required_fields if field not in mapping)
    if missing:
        raise AssertionError(f"{context} missing required fields: {missing}")


def validate_contract_spec(file_name, spec):
    context = f"{file_name} contract spec"
    require_fields(spec, CONTRACT_SPEC_REQUIRED_FIELDS, context)
    if spec["schema_version"] != 1:
        raise AssertionError(f"{context} schema_version must be 1")
    for field in CONTRACT_SPEC_STRING_FIELDS:
        _require_non_empty_string(spec, field, context)

    if not isinstance(spec["bounds"], dict) or not spec["bounds"]:
        raise AssertionError(f"{context} bounds must be a non-empty object")

    case_names = []
    for section in ("positive_cases", "negative_cases"):
        cases = spec[section]
        if not isinstance(cases, list) or not cases:
            raise AssertionError(f"{context} {section} must be a non-empty list")
        for case in cases:
            if not isinstance(case, dict):
                raise AssertionError(f"{context} {section} case must be an object")
            case_context = f"{context} {section} case"
            require_fields(case, ("name",), case_context)
            _require_non_empty_string(case, "name", case_context)
            case_names.append(case["name"])
            if section == "negative_cases":
                require_fields(
                    case,
                    ("violates", "expected_native_route"),
                    f"{context} negative case",
                )
                _require_non_empty_string(case, "violates", f"{context} negative case")
                if case["expected_native_route"] is not False:
                    raise AssertionError(
                        f"{context} negative case expected_native_route must be false"
                    )

    if len(case_names) != len(set(case_names)):
        raise AssertionError(f"{context} case names must be unique")
    validate_shape_envelope_spec(file_name, spec)
    validate_generated_shape_envelope_legal_cases(file_name, spec)
    validate_generated_shape_envelope_adjacent_negatives(file_name, spec)


def validate_all_contract_specs(repo_root):
    specs = load_all_contract_specs(repo_root)
    if not specs:
        raise AssertionError("no Vulkan contract specs found")
    for file_name, spec in specs:
        validate_contract_spec(file_name, spec)
    return specs


def contract_spec_summary(repo_root):
    rows = []
    for file_name, spec in validate_all_contract_specs(repo_root):
        rows.append(
            {
                "file_name": file_name,
                "contract_name": spec["contract_name"],
                "family": spec["family"],
                "tuple_id": spec["tuple_id"],
                "positive_cases": len(spec["positive_cases"]),
                "negative_cases": len(spec["negative_cases"]),
                "shape_envelope_role": spec.get("shape_envelope", {}).get("role"),
            }
        )
    return rows


def format_contract_spec_summary_row(row):
    summary = (
        f"{row['file_name']}: {row['contract_name']} {row['family']} "
        f"{row['tuple_id']} positive_cases={row['positive_cases']} "
        f"negative_cases={row['negative_cases']}"
    )
    if row["shape_envelope_role"]:
        summary += f" shape_envelope={row['shape_envelope_role']}"
    return summary


def shape_envelope_summary(repo_root):
    rows = []
    for file_name, spec in validate_all_contract_specs(repo_root):
        envelope = spec.get("shape_envelope")
        if envelope is None:
            continue
        rows.append(
            {
                "file_name": file_name,
                "contract_name": spec["contract_name"],
                "family": spec["family"],
                "role": envelope["role"],
                "version": envelope["version"],
            }
        )
    return rows


def _repo_relative_path(repo_root, relative_path, context):
    if os.path.isabs(relative_path):
        raise AssertionError(f"{context} must be repository-relative")
    path_parts = relative_path.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in path_parts):
        raise AssertionError(f"{context} has invalid path components")
    return os.path.join(repo_root, *path_parts)


def generated_cpp_manifest_path(repo_root):
    return os.path.join(contract_spec_dir(repo_root), GENERATED_CPP_MANIFEST_FILE)


def load_generated_cpp_manifest(repo_root):
    with open(generated_cpp_manifest_path(repo_root), encoding="utf-8") as handle:
        return json.load(handle)


def _validate_generated_cpp_manifest_entry(entry, index):
    context = f"{GENERATED_CPP_MANIFEST_FILE} entries[{index}]"
    _require_mapping(entry, context)
    require_fields(entry, ("spec_file", "header", "markers"), context)
    for field in ("spec_file", "header"):
        _require_non_empty_string(entry, field, context)
    _require_list(entry["markers"], f"{context}.markers")
    for marker_index, marker in enumerate(entry["markers"]):
        if not isinstance(marker, str) or marker == "":
            raise AssertionError(
                f"{context}.markers[{marker_index}] must be a non-empty string"
            )


def generated_cpp_manifest_entries(repo_root):
    manifest = load_generated_cpp_manifest(repo_root)
    require_fields(manifest, ("schema_version", "entries"), GENERATED_CPP_MANIFEST_FILE)
    if manifest["schema_version"] != 1:
        raise AssertionError(f"{GENERATED_CPP_MANIFEST_FILE} schema_version must be 1")
    _require_list(manifest["entries"], f"{GENERATED_CPP_MANIFEST_FILE}.entries")
    seen_specs = set()
    seen_headers = set()
    for index, entry in enumerate(manifest["entries"]):
        _validate_generated_cpp_manifest_entry(entry, index)
        if entry["spec_file"] in seen_specs:
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} duplicate spec_file "
                f"{entry['spec_file']!r}"
            )
        if entry["header"] in seen_headers:
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} duplicate header "
                f"{entry['header']!r}"
            )
        seen_specs.add(entry["spec_file"])
        seen_headers.add(entry["header"])
    return manifest["entries"]


def validate_generated_cpp_manifest(repo_root):
    specs = dict(validate_all_contract_specs(repo_root))
    shape_envelope_specs = {
        file_name
        for file_name, spec in specs.items()
        if "shape_envelope" in spec
    }
    entries = generated_cpp_manifest_entries(repo_root)
    manifest_specs = {entry["spec_file"] for entry in entries}
    if manifest_specs != shape_envelope_specs:
        raise AssertionError(
            f"{GENERATED_CPP_MANIFEST_FILE} ShapeEnvelope coverage mismatch "
            f"missing={sorted(shape_envelope_specs - manifest_specs)} "
            f"extra={sorted(manifest_specs - shape_envelope_specs)}"
        )

    generator_path = os.path.join(
        repo_root,
        "tools",
        "vulkan_contracts",
        "gen_contract_spec_cpp.py",
    )
    rows = []
    for entry in entries:
        spec_file = entry["spec_file"]
        spec = specs.get(spec_file)
        if spec is None:
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} unknown spec_file {spec_file!r}"
            )
        if "shape_envelope" not in spec:
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} {spec_file} lacks ShapeEnvelope"
            )

        header_path = _repo_relative_path(
            repo_root,
            entry["header"],
            f"{GENERATED_CPP_MANIFEST_FILE} {spec_file} header",
        )
        if not os.path.isfile(header_path):
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} missing header {entry['header']}"
            )

        result = subprocess.run(
            [
                sys.executable,
                generator_path,
                "--spec",
                os.path.join("test", "vulkan_contract_specs", spec_file),
                "--stdout",
            ],
            check=False,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} generator failed for "
                f"{spec_file}: {stderr}"
            )

        with open(header_path, "rb") as handle:
            expected = handle.read()
        if result.stdout != expected:
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} generated header drift for "
                f"{entry['header']}"
            )

        header_text = expected.decode("utf-8")
        missing_markers = [
            marker
            for marker in entry["markers"]
            if marker not in header_text
        ]
        if missing_markers:
            raise AssertionError(
                f"{GENERATED_CPP_MANIFEST_FILE} {entry['header']} missing "
                f"markers: {missing_markers}"
            )

        rows.append(
            {
                "spec_file": spec_file,
                "contract_name": spec["contract_name"],
                "family": spec["family"],
                "shape_envelope_role": spec["shape_envelope"]["role"],
                "header": entry["header"],
                "marker_count": len(entry["markers"]),
            }
        )
    return rows


def generated_cpp_manifest_summary(repo_root):
    return validate_generated_cpp_manifest(repo_root)


def shape_envelope_adjacent_negative_summary(repo_root):
    rows = []
    for file_name, spec in validate_all_contract_specs(repo_root):
        if "shape_envelope" not in spec:
            continue
        generated_cases = generated_shape_envelope_adjacent_negative_cases(spec)
        rows.append(
            {
                "file_name": file_name,
                "contract_name": spec["contract_name"],
                "family": spec["family"],
                "role": spec["shape_envelope"]["role"],
                "generated_negative_cases": len(generated_cases),
            }
        )
    return rows


def shape_envelope_legal_case_summary(repo_root):
    rows = []
    for file_name, spec in validate_all_contract_specs(repo_root):
        if "shape_envelope" not in spec:
            continue
        generated_cases = generated_shape_envelope_legal_cases(spec)
        rows.append(
            {
                "file_name": file_name,
                "contract_name": spec["contract_name"],
                "family": spec["family"],
                "role": spec["shape_envelope"]["role"],
                "generated_legal_cases": len(generated_cases),
            }
        )
    return rows


def shape_envelope_fuzz_assignment_summary(repo_root):
    rows = []
    for file_name, spec in validate_all_contract_specs(repo_root):
        if "shape_envelope" not in spec:
            continue
        assignments = generated_shape_envelope_assignment_cases(spec)
        legal_assignments = assignments["legal_assignments"]
        adjacent_assignments = assignments["adjacent_negative_assignments"]
        if not legal_assignments:
            raise AssertionError(f"{file_name} has no legal fuzz assignments")
        if not adjacent_assignments:
            raise AssertionError(f"{file_name} has no adjacent fuzz assignments")
        rows.append(
            {
                "file_name": file_name,
                "contract_name": spec["contract_name"],
                "family": spec["family"],
                "role": spec["shape_envelope"]["role"],
                "legal_assignments": len(legal_assignments),
                "adjacent_negative_assignments": len(adjacent_assignments),
            }
        )
    return rows


def _shape_envelope_assignment_value_paths(assignments):
    paths = set()
    for assignment in assignments:
        paths.update(assignment.get("values", {}).keys())
    return tuple(sorted(paths))


def _shape_envelope_assignment_status(assignment, coverage_fields):
    paths = set(assignment.get("values", {}))
    covered_paths = paths & coverage_fields
    uncovered_paths = paths - coverage_fields
    if paths and not uncovered_paths:
        status = "covered"
    elif covered_paths:
        status = "partial"
    else:
        status = "unmapped"
    return {
        "name": assignment["name"],
        "status": status,
        "covered_paths": tuple(sorted(covered_paths)),
        "uncovered_paths": tuple(sorted(uncovered_paths)),
    }


def _shape_envelope_violated_axes(cases):
    axes = set()
    for case in cases:
        if "violates" not in case:
            raise AssertionError(f"case {case['name']} missing violates")
        axes.add(case["violates"])
    return axes


def validate_shape_envelope_fuzz_assignment_coverage(file_name, spec):
    if "shape_envelope" not in spec:
        return None
    envelope = spec["shape_envelope"]
    adapter = _shape_envelope_role_adapter(envelope["role"], file_name)
    assignments = generated_shape_envelope_assignment_cases(spec)
    legal_assignments = assignments["legal_assignments"]
    adjacent_assignments = assignments["adjacent_negative_assignments"]
    if not legal_assignments:
        raise AssertionError(f"{file_name} has no legal fuzz assignments")
    if not adjacent_assignments:
        raise AssertionError(f"{file_name} has no adjacent fuzz assignments")

    coverage_fields = set(adapter["assignment_coverage_fields"])
    legal_assignment_paths = set(
        _shape_envelope_assignment_value_paths(legal_assignments)
    )
    unknown_coverage_fields = coverage_fields - legal_assignment_paths
    if unknown_coverage_fields:
        raise AssertionError(
            f"{file_name} assignment coverage fields are not generated "
            f"assignment paths: {sorted(unknown_coverage_fields)}"
        )

    assignment_statuses = [
        _shape_envelope_assignment_status(assignment, coverage_fields)
        for assignment in legal_assignments
    ]
    coverage_gaps = [
        status for status in assignment_statuses
        if status["status"] != "covered"
    ]
    if coverage_gaps:
        raise AssertionError(
            f"{file_name} fuzz assignment coverage gaps: {coverage_gaps}"
        )

    generated_legal_cases = validate_generated_shape_envelope_legal_cases(
        file_name,
        spec,
    )
    generated_negative_cases = validate_generated_shape_envelope_adjacent_negatives(
        file_name,
        spec,
    )
    assignment_axes = _shape_envelope_violated_axes(adjacent_assignments)
    checked_in_axes = _shape_envelope_violated_axes(spec["negative_cases"])
    generated_negative_axes = _shape_envelope_violated_axes(generated_negative_cases)
    if assignment_axes != checked_in_axes:
        raise AssertionError(
            f"{file_name} adjacent assignment axes mismatch checked-in negatives "
            f"missing={sorted(checked_in_axes - assignment_axes)} "
            f"extra={sorted(assignment_axes - checked_in_axes)}"
        )
    if assignment_axes != generated_negative_axes:
        raise AssertionError(
            f"{file_name} adjacent assignment axes mismatch generated negatives "
            f"missing={sorted(generated_negative_axes - assignment_axes)} "
            f"extra={sorted(assignment_axes - generated_negative_axes)}"
        )

    return {
        "file_name": file_name,
        "contract_name": spec["contract_name"],
        "family": spec["family"],
        "role": envelope["role"],
        "legal_assignments": len(legal_assignments),
        "legal_assignment_paths": len(legal_assignment_paths),
        "covered_legal_assignment_paths": len(coverage_fields),
        "legal_assignment_status": "covered",
        "adjacent_negative_axes": len(assignment_axes),
        "generated_legal_cases": len(generated_legal_cases),
        "generated_negative_cases": len(generated_negative_cases),
    }


def shape_envelope_fuzz_assignment_coverage_summary(repo_root):
    rows = []
    for file_name, spec in validate_all_contract_specs(repo_root):
        row = validate_shape_envelope_fuzz_assignment_coverage(file_name, spec)
        if row is not None:
            rows.append(row)
    return rows


def iter_contract_cases(spec):
    for section, expect_native_route in (
        ("positive_cases", True),
        ("negative_cases", None),
    ):
        for case in spec[section]:
            if expect_native_route is None:
                yield section, case, case["expected_native_route"]
            else:
                yield section, case, expect_native_route


def iter_shape_envelope_contract_cases(spec):
    if "shape_envelope" not in spec:
        yield from iter_contract_cases(spec)
        return
    for case in generated_shape_envelope_legal_cases(spec):
        yield "generated_legal_cases", case, True
    for case in generated_shape_envelope_adjacent_negative_cases(spec):
        yield "generated_negative_cases", case, case["expected_native_route"]


def contract_case_id(spec, case):
    return f"{spec['contract_name']}_{spec['family']}_{case['name']}"


def contract_log_name(spec, case, suffix):
    safe_case_id = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in contract_case_id(spec, case)
    )
    return f"{safe_case_id}_{suffix}"


def expected_negative_flag(case, name, default=False):
    if case.get("expected_native_route", True):
        return default
    return case.get(name, default)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-shape-envelope", action="store_true")
    parser.add_argument("--validate-legal-cases", action="store_true")
    parser.add_argument("--validate-adjacent-negatives", action="store_true")
    parser.add_argument("--validate-fuzz-assignments", action="store_true")
    parser.add_argument("--validate-fuzz-assignment-coverage", action="store_true")
    parser.add_argument("--validate-generated-cpp-manifest", action="store_true")
    args = parser.parse_args()

    rows = contract_spec_summary(args.repo_root)
    if args.list or args.summary:
        for row in rows:
            print(format_contract_spec_summary_row(row))
    if args.validate:
        total_positive = sum(row["positive_cases"] for row in rows)
        total_negative = sum(row["negative_cases"] for row in rows)
        print(
            f"validated {len(rows)} Vulkan contract specs "
            f"positive_cases={total_positive} negative_cases={total_negative}"
        )
    if args.validate_shape_envelope:
        envelope_rows = shape_envelope_summary(args.repo_root)
        if not envelope_rows:
            raise AssertionError("no ShapeEnvelope v1 specs found")
        roles = ", ".join(
            f"{row['file_name']}:{row['role']}" for row in envelope_rows
        )
        print(f"validated {len(envelope_rows)} ShapeEnvelope v1 specs {roles}")
    if args.validate_legal_cases:
        legal_rows = shape_envelope_legal_case_summary(args.repo_root)
        if not legal_rows:
            raise AssertionError("no generated legal cases found")
        total_generated = sum(row["generated_legal_cases"] for row in legal_rows)
        roles = ", ".join(
            f"{row['file_name']}:{row['generated_legal_cases']}"
            for row in legal_rows
        )
        print(
            "validated "
            f"{len(legal_rows)} ShapeEnvelope legal-case generators "
            f"generated_cases={total_generated} {roles}"
        )
    if args.validate_adjacent_negatives:
        negative_rows = shape_envelope_adjacent_negative_summary(args.repo_root)
        if not negative_rows:
            raise AssertionError("no generated adjacent-negative cases found")
        total_generated = sum(
            row["generated_negative_cases"] for row in negative_rows
        )
        roles = ", ".join(
            f"{row['file_name']}:{row['generated_negative_cases']}"
            for row in negative_rows
        )
        print(
            "validated "
            f"{len(negative_rows)} ShapeEnvelope adjacent-negative generators "
            f"generated_cases={total_generated} {roles}"
        )
    if args.validate_fuzz_assignments:
        assignment_rows = shape_envelope_fuzz_assignment_summary(args.repo_root)
        if not assignment_rows:
            raise AssertionError("no generated fuzz assignments found")
        total_legal = sum(row["legal_assignments"] for row in assignment_rows)
        total_adjacent = sum(
            row["adjacent_negative_assignments"] for row in assignment_rows
        )
        roles = ", ".join(
            f"{row['file_name']}:legal={row['legal_assignments']}:"
            f"adjacent={row['adjacent_negative_assignments']}"
            for row in assignment_rows
        )
        print(
            "validated "
            f"{len(assignment_rows)} ShapeEnvelope fuzz assignment generators "
            f"legal_assignments={total_legal} "
            f"adjacent_negative_assignments={total_adjacent} {roles}"
        )
    if args.validate_fuzz_assignment_coverage:
        coverage_rows = shape_envelope_fuzz_assignment_coverage_summary(
            args.repo_root
        )
        if not coverage_rows:
            raise AssertionError("no generated fuzz assignment coverage found")
        total_legal_assignments = sum(
            row["legal_assignments"] for row in coverage_rows
        )
        total_legal_paths = sum(
            row["legal_assignment_paths"] for row in coverage_rows
        )
        total_adjacent_axes = sum(
            row["adjacent_negative_axes"] for row in coverage_rows
        )
        total_runtime_legal = sum(
            row["generated_legal_cases"] for row in coverage_rows
        )
        total_runtime_adjacent = sum(
            row["generated_negative_cases"] for row in coverage_rows
        )
        roles = ", ".join(
            f"{row['file_name']}:legal={row['legal_assignments']}:"
            f"status={row['legal_assignment_status']}:"
            f"paths={row['covered_legal_assignment_paths']}/"
            f"{row['legal_assignment_paths']}:"
            f"adjacent_axes={row['adjacent_negative_axes']}"
            for row in coverage_rows
        )
        print(
            "validated "
            f"{len(coverage_rows)} ShapeEnvelope fuzz assignment coverage "
            f"bridges legal_assignments={total_legal_assignments} "
            f"legal_paths={total_legal_paths} "
            f"adjacent_negative_axes={total_adjacent_axes} "
            f"runtime_legal_cases={total_runtime_legal} "
            f"runtime_adjacent_negative_cases={total_runtime_adjacent} {roles}"
        )
    if args.validate_generated_cpp_manifest:
        generated_rows = generated_cpp_manifest_summary(args.repo_root)
        if not generated_rows:
            raise AssertionError("no generated C++ manifest entries found")
        total_markers = sum(row["marker_count"] for row in generated_rows)
        entries = ", ".join(
            f"{row['spec_file']}:{os.path.basename(row['header'])}:"
            f"markers={row['marker_count']}"
            for row in generated_rows
        )
        print(
            "validated "
            f"{len(generated_rows)} generated ShapeEnvelope C++ helper headers "
            f"markers={total_markers} {entries}"
        )


if __name__ == "__main__":
    _main()
