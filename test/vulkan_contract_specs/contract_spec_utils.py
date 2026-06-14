import argparse
import glob
import json
import os
import re
import subprocess
import sys


GENERATED_CPP_MANIFEST_FILE = "generated_cpp_manifest.json"
TEMPORARY_EXCEPTIONS_FILE = os.path.join("docs", "vulkan", "TEMPORARY_EXCEPTIONS.md")
GENERIC_EXACT_TUPLE_EXCEPTION = "Exact Tuple Rows In Contract Tables"
CONTRACT_NAME_LITERAL_RE = re.compile(r"\"([A-Za-z0-9]+Contract)\"")

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

SPARSE_ROWSET_REQUIRED_FIELDS = (
    "name",
    "fields",
    "identity_fields",
    "rows",
)

SPARSE_ROWSET_NEGATIVE_KINDS = (
    "field_value_outside_rowset",
    "forbidden_cross_product",
    "adjacent_field",
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


def _validate_sparse_rowset_negative_axes(negative_axes, field_names, context):
    _require_list(negative_axes, context, allow_empty=True)
    field_names = set(field_names)
    for index, axis in enumerate(negative_axes):
        axis_context = f"{context}[{index}]"
        _require_mapping(axis, axis_context)
        require_fields(axis, ("violates", "kind"), axis_context)
        _require_non_empty_string(axis, "violates", axis_context)
        _require_non_empty_string(axis, "kind", axis_context)
        if axis["kind"] not in SPARSE_ROWSET_NEGATIVE_KINDS:
            raise AssertionError(
                f"{axis_context}.kind has unsupported value {axis['kind']!r}"
            )
        for mapping_field in ("fields", "row"):
            if mapping_field not in axis:
                continue
            _require_mapping(axis[mapping_field], f"{axis_context}.{mapping_field}")
            unknown_fields = sorted(set(axis[mapping_field]) - field_names)
            if unknown_fields:
                raise AssertionError(
                    f"{axis_context}.{mapping_field} has unknown fields "
                    f"{unknown_fields}"
                )
            for field, value in axis[mapping_field].items():
                _validate_scalar(value, f"{axis_context}.{mapping_field}.{field}")


def _sparse_rowset_field_values(rowset, field):
    return _dedupe_preserving_order(row[field] for row in rowset["rows"])


def _sparse_rowset_independent_cross_product(rowset):
    result = 1
    for field in rowset["identity_fields"]:
        result *= len(_sparse_rowset_field_values(rowset, field))
    return result


def _validate_sparse_rowset(rowset, context):
    _require_mapping(rowset, context)
    require_fields(rowset, SPARSE_ROWSET_REQUIRED_FIELDS, context)
    _require_non_empty_string(rowset, "name", context)
    _require_list(rowset["fields"], f"{context}.fields")
    _require_list(rowset["identity_fields"], f"{context}.identity_fields")
    fields = rowset["fields"]
    identity_fields = rowset["identity_fields"]
    for index, field in enumerate(fields):
        _require_non_empty_string({"field": field}, "field", f"{context}.fields[{index}]")
    if len(fields) != len(set(fields)):
        raise AssertionError(f"{context}.fields must be unique")
    field_names = set(fields)
    lookup_fields = rowset.get("lookup_fields", identity_fields)
    _require_list(lookup_fields, f"{context}.lookup_fields")
    for index, field in enumerate(lookup_fields):
        _require_non_empty_string(
            {"field": field},
            "field",
            f"{context}.lookup_fields[{index}]",
        )
        if field not in field_names:
            raise AssertionError(
                f"{context}.lookup_fields[{index}] unknown field {field!r}"
            )
    if len(lookup_fields) != len(set(lookup_fields)):
        raise AssertionError(f"{context}.lookup_fields must be unique")

    for index, field in enumerate(identity_fields):
        _require_non_empty_string(
            {"field": field},
            "field",
            f"{context}.identity_fields[{index}]",
        )
        if field not in field_names:
            raise AssertionError(
                f"{context}.identity_fields[{index}] unknown field {field!r}"
            )
    if len(identity_fields) != len(set(identity_fields)):
        raise AssertionError(f"{context}.identity_fields must be unique")
    label_field = rowset.get("label_field")
    if label_field is not None:
        _require_non_empty_string(rowset, "label_field", context)
        if label_field not in field_names:
            raise AssertionError(f"{context}.label_field unknown field {label_field!r}")

    _require_list(rowset["rows"], f"{context}.rows")
    seen_identity_keys = {}
    seen_labels = {}
    for index, row in enumerate(rowset["rows"]):
        row_context = f"{context}.rows[{index}]"
        _require_mapping(row, row_context)
        missing_fields = sorted(field_names - set(row))
        extra_fields = sorted(set(row) - field_names)
        if missing_fields or extra_fields:
            raise AssertionError(
                f"{row_context} field mismatch missing={missing_fields} "
                f"extra={extra_fields}"
            )
        for field in fields:
            _validate_scalar(row[field], f"{row_context}.{field}")
        identity_key = tuple(_canonical_case_value(row[field]) for field in identity_fields)
        if identity_key in seen_identity_keys:
            raise AssertionError(
                f"{row_context} duplicate identity with row "
                f"{seen_identity_keys[identity_key]}"
            )
        seen_identity_keys[identity_key] = index
        if label_field is not None:
            label = _canonical_case_value(row[label_field])
            if label in seen_labels:
                raise AssertionError(
                    f"{row_context} duplicate label {label!r} with row "
                    f"{seen_labels[label]}"
                )
            seen_labels[label] = index

    seen_lookup_keys = {}
    for index, row in enumerate(rowset["rows"]):
        lookup_key = tuple(_canonical_case_value(row[field]) for field in lookup_fields)
        if lookup_key in seen_lookup_keys:
            raise AssertionError(
                f"{context}.rows[{index}] duplicate lookup with row "
                f"{seen_lookup_keys[lookup_key]}"
            )
        seen_lookup_keys[lookup_key] = index

    _validate_sparse_rowset_negative_axes(
        rowset.get("negative_axes", []),
        fields,
        f"{context}.negative_axes",
    )
    independent_cross_product = _sparse_rowset_independent_cross_product(rowset)
    return {
        "rowset_name": rowset["name"],
        "fields": tuple(fields),
        "identity_fields": tuple(identity_fields),
        "lookup_fields": tuple(lookup_fields),
        "label_field": label_field or "",
        "row_count": len(rowset["rows"]),
        "independent_identity_cross_product": independent_cross_product,
        "sparse_cross_product_gap": independent_cross_product - len(rowset["rows"]),
        "negative_axes": len(rowset.get("negative_axes", [])),
    }


def validate_shape_envelope_sparse_rowsets(file_name, spec):
    envelope = spec.get("shape_envelope")
    if envelope is None:
        return []
    rowsets = envelope.get("sparse_rowsets", [])
    _require_list(
        rowsets,
        f"{file_name} ShapeEnvelope v1 sparse_rowsets",
        allow_empty=True,
    )
    rows = []
    for index, rowset in enumerate(rowsets):
        row = _validate_sparse_rowset(
            rowset,
            f"{file_name} ShapeEnvelope v1 sparse_rowsets[{index}]",
        )
        row.update(
            {
                "file_name": file_name,
                "contract_name": spec.get("contract_name", ""),
                "family": spec.get("family", ""),
                "role": envelope.get("role", ""),
            }
        )
        rows.append(row)
    return rows


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
    validate_shape_envelope_sparse_rowsets(file_name, spec)
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


def _validate_small_metadata_padded_conv2d_shape_envelope(
    file_name,
    spec,
    envelope,
):
    context = f"{file_name} SmallMetadataPaddedConv2D ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "SmallMetadataPaddedConv2DContract",
        f"{context} contract",
    )
    _require_equal(
        spec["family"],
        "MaterializedBufferInput2x2",
        f"{context} family",
    )
    _require_equal(
        spec["tuple_id"],
        "input_1x16x721x1281_weight_32x16x2x2_stride1",
        f"{context} tuple",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    bounds = envelope["bounds"]
    for key, expected in (
        ("input_dtype", "float32"),
        ("weight_dtype", "float32"),
        ("input_rank", 4),
        ("weight_rank", 4),
        ("batch", 1),
        ("groups", 1),
        ("kernel_h", 2),
        ("kernel_w", 2),
        ("stride_h", 1),
        ("stride_w", 1),
        ("padding_h", 0),
        ("padding_w", 0),
        ("dilation_h", 1),
        ("dilation_w", 1),
        ("input_channels", 16),
        ("input_height", 721),
        ("input_width", 1281),
        ("output_channels", 32),
    ):
        _require_equal(bounds[key], expected, f"{context} {key}")
    for key, expected in (
        ("transposed", False),
        ("options_quantized", False),
        ("output_padding_is_zero", True),
        ("requires_vulkan", True),
        ("input_has_buffer_storage", True),
        ("input_is_width_packed", True),
        ("input_has_direct_buffer_layout", False),
        ("input_supports_buffer_compute", True),
        ("weight_defined", True),
    ):
        _require_equal(bounds[key], expected, f"{context} {key}")

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
    for symbol, bound_key in (
        ("N", "batch"),
        ("CI", "input_channels"),
        ("H", "input_height"),
        ("W", "input_width"),
    ):
        expected = bounds[bound_key]
        _require_equal(
            _single_value(input_dims[symbol], f"{context} input {symbol}"),
            expected,
            f"{context} input {symbol}",
        )
    for symbol, bound_key in (
        ("CO", "output_channels"),
        ("CI", "input_channels"),
        ("KH", "kernel_h"),
        ("KW", "kernel_w"),
    ):
        expected = bounds[bound_key]
        _require_equal(
            _single_value(weight_dims[symbol], f"{context} weight {symbol}"),
            expected,
            f"{context} weight {symbol}",
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
        "input_is_width_packed",
        "input_has_direct_buffer_layout",
        "requires_input_materialization",
    ):
        expected = (
            True
            if attribute_name == "requires_input_materialization"
            else bounds[attribute_name]
        )
        _require_equal(
            _single_value(attributes[attribute_name], f"{context} {attribute_name}"),
            expected,
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
    _require_equal(
        layout["input_layout"],
        "width_packed_small_metadata",
        f"{context} input layout",
    )
    _require_equal(
        layout["input_has_direct_buffer_layout"],
        bounds["input_has_direct_buffer_layout"],
        f"{context} direct buffer layout",
    )
    _require_equal(layout["weight_storage"], "vulkan_tensor", f"{context} weight")
    _require_equal(
        layout["execution_storage"],
        "materialized_buffer",
        f"{context} execution storage",
    )
    _require_equal(
        layout["requires_input_materialization"],
        True,
        f"{context} materialization",
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
                case.get("violates") != "input_channels"
                and case["input_shape"][1] != bounds["input_channels"]
            ):
                raise AssertionError(f"{case_context} input channels mismatch")
            if (
                case.get("violates") != "input_height"
                and case["input_shape"][2] != bounds["input_height"]
            ):
                raise AssertionError(f"{case_context} input height mismatch")
            if (
                case.get("violates") != "input_width"
                and case["input_shape"][3] != bounds["input_width"]
            ):
                raise AssertionError(f"{case_context} input width mismatch")
            if (
                case.get("violates") != "output_channels"
                and case["out_channels"] != bounds["output_channels"]
            ):
                raise AssertionError(f"{case_context} output channels mismatch")
            if case["dtype"] != bounds["input_dtype"]:
                if case.get("violates") != "dtype":
                    raise AssertionError(f"{case_context} dtype mismatch")
            if case["groups"] != bounds["groups"]:
                raise AssertionError(f"{case_context} groups mismatch")
            require_pair(case, "kernel_size", "kernel_h", "kernel_w", "kernel")
            require_pair(case, "stride", "stride_h", "stride_w", "stride")
            require_pair(case, "padding", "padding_h", "padding_w", "padding")
            require_pair(case, "dilation", "dilation_h", "dilation_w", "dilation")
            if case["output_padding"] != [0, 0]:
                raise AssertionError(f"{case_context} output padding mismatch")
            if (
                case["input_has_direct_buffer_layout"]
                != bounds["input_has_direct_buffer_layout"]
                and case.get("violates") != "input_has_direct_buffer_layout"
            ):
                raise AssertionError(f"{case_context} direct layout mismatch")


def _validate_linear_gelu_bridge_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} LinearGeluBridge ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "LinearGeluBridgeContract",
        f"{context} contract",
    )
    _require_equal(
        spec["family"],
        "BackboneMlpHidden384To1536",
        f"{context} family",
    )
    _require_equal(
        spec["tuple_id"],
        "backbone_mlp_hidden384_to1536_rows_ge512",
        f"{context} tuple",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    bounds = envelope["bounds"]
    for key, expected in (
        ("input_rank", [2, 3]),
        ("flattened_rank", 2),
        ("flattened_features", 384),
        ("weight_height", 384),
        ("weight_width", 1536),
        ("rank3_batch", 1),
    ):
        _require_equal(bounds[key], expected, f"{context} {key}")
    _require_equal(bounds["flattened_rows"]["min"], 512, f"{context} rows min")
    for key, expected in (
        ("bias_defined", True),
        ("can_run_float_buffer_linear", True),
        ("inference_mode_enabled", False),
        ("has_output", False),
        ("post_op_is_none", True),
        ("alpha_is_one", True),
        ("beta_is_one", True),
        ("may_defer", True),
        ("may_consume_gelu_none", True),
        ("may_consume_gelu_tanh", True),
    ):
        _require_equal(bounds[key], expected, f"{context} {key}")

    inputs = envelope["inputs"]
    require_fields(
        inputs,
        ("input", "flattened_input", "packed_weight"),
        f"{context} inputs",
    )
    input_spec = inputs["input"]
    flattened = inputs["flattened_input"]
    packed_weight = inputs["packed_weight"]
    _require_equal(input_spec["rank"]["values"], bounds["input_rank"], f"{context} rank")
    _require_equal(
        _single_value(flattened["rank"], f"{context} flattened rank"),
        bounds["flattened_rank"],
        f"{context} flattened rank",
    )
    _require_equal(
        _single_value(packed_weight["rank"], f"{context} packed rank"),
        2,
        f"{context} packed rank",
    )

    input_dims = _dims_by_symbol(input_spec, f"{context} input")
    flattened_dims = _dims_by_symbol(flattened, f"{context} flattened")
    packed_dims = _dims_by_symbol(packed_weight, f"{context} packed")
    _require_equal(
        input_dims["B"]["values"],
        [bounds["rank3_batch"]],
        f"{context} rank3 batch",
    )
    _require_equal(input_dims["B"].get("optional"), True, f"{context} optional B")
    _require_equal(
        input_dims["R"]["min"],
        bounds["flattened_rows"]["min"],
        f"{context} input rows",
    )
    _require_equal(
        _single_value(input_dims["F"], f"{context} input features"),
        bounds["flattened_features"],
        f"{context} input features",
    )
    _require_equal(
        flattened_dims["FR"]["min"],
        bounds["flattened_rows"]["min"],
        f"{context} flattened rows",
    )
    _require_equal(
        _single_value(flattened_dims["F"], f"{context} flattened features"),
        bounds["flattened_features"],
        f"{context} flattened features",
    )
    _require_equal(
        _single_value(packed_dims["F"], f"{context} packed input features"),
        bounds["weight_height"],
        f"{context} packed input features",
    )
    _require_equal(
        _single_value(packed_dims["O"], f"{context} packed output features"),
        bounds["weight_width"],
        f"{context} packed output features",
    )

    attributes = envelope["attributes"]
    for attribute_name in (
        "rank3_batch",
        "bias_defined",
        "can_run_float_buffer_linear",
        "inference_mode_enabled",
        "has_output",
        "post_op_is_none",
        "alpha_is_one",
        "beta_is_one",
        "may_defer",
        "may_consume_gelu_none",
        "may_consume_gelu_tanh",
    ):
        _require_equal(
            _single_value(attributes[attribute_name], f"{context} {attribute_name}"),
            bounds[attribute_name],
            f"{context} {attribute_name}",
        )

    relationships = _relationship_types(envelope)
    if "equal" not in relationships or len(envelope["relationships"]) != 3:
        raise AssertionError(f"{context} expected three equal relationships")
    layout = envelope["layout"]
    _require_equal(layout["requires_vulkan"], True, f"{context} requires_vulkan")
    _require_equal(layout["input_storage"], "buffer", f"{context} input storage")
    _require_equal(
        layout["packed_weight_storage"],
        "buffer",
        f"{context} packed weight storage",
    )
    _require_equal(layout["bias_storage"], "buffer", f"{context} bias storage")
    _require_equal(
        layout["execution_storage"],
        "deferred_placeholder",
        f"{context} execution storage",
    )
    _require_equal(
        envelope["capability_requirements"]["can_run_float_buffer_linear"],
        bounds["can_run_float_buffer_linear"],
        f"{context} can run float buffer linear",
    )

    def flattened_rows(shape):
        return _product(shape[:-1])

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            if (
                len(case["input_shape"]) not in bounds["input_rank"]
                and case.get("violates") != "input_rank"
            ):
                raise AssertionError(f"{case_context} input rank mismatch")
            if (
                flattened_rows(case["input_shape"]) < bounds["flattened_rows"]["min"]
                and case.get("violates") != "flattened_rows.min"
            ):
                raise AssertionError(f"{case_context} flattened rows below min")
            if (
                case["input_shape"][-1] != bounds["flattened_features"]
                and case.get("violates") != "flattened_features"
            ):
                raise AssertionError(f"{case_context} input feature mismatch")
            if (
                case["weight_shape"][1] != bounds["weight_height"]
                and case.get("violates") != "flattened_features"
            ):
                raise AssertionError(f"{case_context} weight height mismatch")
            if (
                case["weight_shape"][0] != bounds["weight_width"]
                and case.get("violates") != "weight_width"
            ):
                raise AssertionError(f"{case_context} weight width mismatch")
            if (
                len(case["input_shape"]) == 3
                and case["input_shape"][0] != bounds["rank3_batch"]
                and case.get("violates") != "rank3_batch"
            ):
                raise AssertionError(f"{case_context} rank3 batch mismatch")
            for key in (
                "bias_defined",
                "inference_mode_enabled",
                "has_output",
                "post_op_is_none",
                "alpha_is_one",
                "beta_is_one",
            ):
                if case[key] != bounds[key] and case.get("violates") != key:
                    raise AssertionError(f"{case_context} {key} mismatch")


def _validate_gqa_repeat_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} GQARepeat ShapeEnvelope"
    _require_equal(spec["contract_name"], "GQARepeatContract", f"{context} contract")
    _require_equal(
        spec["family"],
        "Batch1Heads4Factor4Sequence100To116Dim128",
        f"{context} family",
    )
    _require_equal(
        spec["tuple_id"],
        "gqa_repeat_batch1_heads4_factor4_sequence100_to_116_dim128",
        f"{context} tuple",
    )
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    bounds = envelope["bounds"]
    for key, expected in (
        ("dtype", "float32"),
        ("rank", 4),
        ("batch", 1),
        ("source_heads", 4),
        ("target_heads", 16),
        ("repeat_factor", 4),
        ("target_sequence", 1),
        ("head_dim", 128),
        ("requires_vulkan", True),
        ("requires_buffer_storage", True),
        ("enable_gqa", True),
    ):
        _require_equal(bounds[key], expected, f"{context} {key}")
    _require_equal(
        bounds["source_sequence"],
        {"min": 100, "max": 116},
        f"{context} source sequence",
    )

    inputs = envelope["inputs"]
    require_fields(inputs, ("source",), f"{context} inputs")
    source = inputs["source"]
    _require_equal(
        _single_value(source["dtype"], f"{context} source dtype"),
        bounds["dtype"],
        f"{context} source dtype",
    )
    _require_equal(
        _single_value(source["rank"], f"{context} source rank"),
        bounds["rank"],
        f"{context} source rank",
    )
    dims = _dims_by_symbol(source, f"{context} source")
    _require_equal(dims["N"].get("values"), [bounds["batch"]], f"{context} N")
    _require_equal(
        dims["H"].get("values"),
        [bounds["source_heads"]],
        f"{context} H",
    )
    _require_equal(
        dims["S"]["min"],
        bounds["source_sequence"]["min"],
        f"{context} S min",
    )
    _require_equal(
        dims["S"]["max"],
        bounds["source_sequence"]["max"],
        f"{context} S max",
    )
    _require_equal(dims["D"].get("values"), [bounds["head_dim"]], f"{context} D")

    attributes = envelope["attributes"]
    for key in ("target_heads", "repeat_factor", "target_sequence", "enable_gqa"):
        _require_equal(
            _single_value(attributes[key], f"{context} {key}"),
            bounds[key],
            f"{context} {key}",
        )

    layout = envelope["layout"]
    _require_equal(
        layout["requires_vulkan"],
        bounds["requires_vulkan"],
        f"{context} requires_vulkan",
    )
    _require_equal(layout["source_storage"], "buffer", f"{context} source storage")
    _require_equal(layout["output_storage"], "buffer", f"{context} output storage")

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            for shape_name in ("key_shape", "value_shape"):
                shape = case[shape_name]
                if len(shape) != bounds["rank"]:
                    raise AssertionError(f"{case_context} {shape_name} rank mismatch")
                if shape[0] != bounds["batch"]:
                    raise AssertionError(f"{case_context} {shape_name} batch mismatch")
                if shape[1] != bounds["source_heads"]:
                    raise AssertionError(f"{case_context} {shape_name} heads mismatch")
                if (
                    shape[2] < bounds["source_sequence"]["min"]
                    and case.get("violates") != "source_sequence.min"
                ):
                    raise AssertionError(
                        f"{case_context} {shape_name} sequence below min"
                    )
                if (
                    shape[2] > bounds["source_sequence"]["max"]
                    and case.get("violates") != "source_sequence.max"
                ):
                    raise AssertionError(
                        f"{case_context} {shape_name} sequence above max"
                    )
                if shape[3] != bounds["head_dim"]:
                    raise AssertionError(
                        f"{case_context} {shape_name} head dim mismatch"
                    )
            if case["query_shape"][1] != bounds["target_heads"]:
                raise AssertionError(f"{case_context} query heads mismatch")
            if case["query_shape"][2] != bounds["target_sequence"]:
                raise AssertionError(f"{case_context} query sequence mismatch")
            if case["enable_gqa"] != bounds["enable_gqa"]:
                raise AssertionError(f"{case_context} enable_gqa mismatch")
            if case["dtype"] != bounds["dtype"]:
                raise AssertionError(f"{case_context} dtype mismatch")


def _small_spatial_pointwise_conv_rowset(envelope, context):
    rowsets = envelope.get("sparse_rowsets", [])
    if len(rowsets) != 1:
        raise AssertionError(f"{context} must have one sparse rowset")
    rowset = rowsets[0]
    _require_equal(rowset["name"], "projection_rows", f"{context} rowset name")
    _require_equal(
        rowset["fields"],
        ["family", "input_c", "input_h", "input_w", "output_c", "tuple_id"],
        f"{context} rowset fields",
    )
    _require_equal(
        rowset["identity_fields"],
        ["family", "input_c", "input_h", "input_w", "output_c"],
        f"{context} rowset identity fields",
    )
    _require_equal(
        rowset["lookup_fields"],
        ["input_c", "input_h", "input_w", "output_c"],
        f"{context} rowset lookup fields",
    )
    _require_equal(rowset["label_field"], "tuple_id", f"{context} rowset label")
    return rowset


def _validate_small_spatial_pointwise_conv_shape_envelope(
    file_name,
    spec,
    envelope,
):
    context = f"{file_name} SmallSpatialPointwiseConv ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "SmallSpatialPointwiseConvContract",
        f"{context} contract",
    )
    _require_equal(spec["family"], "SparseProjectionRows", f"{context} family")
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")

    bounds = envelope["bounds"]
    for key, expected in (
        ("dtype", "float32"),
        ("input_rank", 4),
        ("weight_rank", 4),
        ("batch", 1),
        ("groups", 1),
        ("kernel_h", 1),
        ("kernel_w", 1),
        ("stride_h", 1),
        ("stride_w", 1),
        ("padding_h", 0),
        ("padding_w", 0),
        ("dilation_h", 1),
        ("dilation_w", 1),
    ):
        _require_equal(bounds[key], expected, f"{context} {key}")
    for key in ("requires_vulkan", "requires_buffer_storage", "requires_buffer_compute"):
        _require_equal(bounds[key], True, f"{context} {key}")

    rowset = _small_spatial_pointwise_conv_rowset(envelope, context)
    rows = rowset["rows"]
    if len(rows) != 39:
        raise AssertionError(f"{context} expected 39 sparse rows")
    family_counts = {}
    row_keys = set()
    tuple_ids = set()
    lookup_keys = set()
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
        row_keys.add(
            (
                row["family"],
                row["input_c"],
                row["input_h"],
                row["input_w"],
                row["output_c"],
            )
        )
        lookup_keys.add(
            (
                row["input_c"],
                row["input_h"],
                row["input_w"],
                row["output_c"],
            )
        )
        tuple_ids.add(row["tuple_id"])
    _require_equal(
        family_counts,
        {
            "DepthVisionProjection": 10,
            "OCRProjection": 13,
            "DiffusionProjection": 16,
        },
        f"{context} family counts",
    )
    _require_equal(len(tuple_ids), 39, f"{context} tuple ids")
    _require_equal(len(lookup_keys), 39, f"{context} lookup keys")

    positive_keys = {
        (
            case["expected_contract_family"],
            case["input_shape"][1],
            case["input_shape"][2],
            case["input_shape"][3],
            case["out_channels"],
        )
        for case in spec["positive_cases"]
    }
    if positive_keys != row_keys:
        missing = sorted(row_keys - positive_keys)
        extra = sorted(positive_keys - row_keys)
        raise AssertionError(
            f"{context} positives do not match sparse rowset "
            f"missing={missing} extra={extra}"
        )

    for case in spec["positive_cases"]:
        case_context = f"{context} positive {case['name']}"
        _require_equal(case["dtype"], bounds["dtype"], f"{case_context} dtype")
        _require_equal(case["kernel_size"], [1, 1], f"{case_context} kernel")
        _require_equal(case["stride"], [1, 1], f"{case_context} stride")
        _require_equal(case["padding"], [0, 0], f"{case_context} padding")
        _require_equal(case["dilation"], [1, 1], f"{case_context} dilation")
        _require_equal(case["groups"], 1, f"{case_context} groups")
        _require_equal(
            case["expected_route_label"],
            "aten::convolution.buffer_float_1x1_skip.small_spatial_pointwise",
            f"{case_context} route label",
        )
        _require_equal(case["expected_cpu_fallback"], False, f"{case_context} fallback")

    for case in spec["negative_cases"]:
        case_context = f"{context} negative {case['name']}"
        if case["violates"].startswith("projection_rows."):
            key = (
                case["input_shape"][1],
                case["input_shape"][2],
                case["input_shape"][3],
                case["out_channels"],
            )
            if key in lookup_keys:
                raise AssertionError(f"{case_context} unexpectedly matches rowset")


def _validate_kv_cache_append_shape_envelope(file_name, spec, envelope):
    context = f"{file_name} KVCacheAppend ShapeEnvelope"
    _require_equal(
        spec["contract_name"],
        "KVCacheAppendContract",
        f"{context} contract",
    )
    family = spec["family"]
    if family not in ("SequenceAppend", "InitialCache"):
        raise AssertionError(f"{context} unsupported family {family!r}")
    _require_equal(envelope["bounds"], spec["bounds"], f"{context} bounds")
    bounds = envelope["bounds"]
    attributes = envelope["attributes"]
    _require_equal(
        _single_value(attributes["dim"], f"{context} dim"),
        bounds["dim"],
        f"{context} dim",
    )

    if family == "SequenceAppend":
        inputs = envelope["inputs"]
        require_fields(inputs, ("cache", "token"), f"{context} inputs")
        cache = inputs["cache"]
        token = inputs["token"]
        _require_equal(
            _single_value(cache["dtype"], f"{context} cache dtype"),
            bounds["cache_dtype"],
            f"{context} cache dtype",
        )
        _require_equal(
            _single_value(token["dtype"], f"{context} token dtype"),
            bounds["token_dtype"],
            f"{context} token dtype",
        )
        _require_equal(
            _single_value(cache["rank"], f"{context} cache rank"),
            bounds["cache_rank"],
            f"{context} cache rank",
        )
        _require_equal(
            _single_value(token["rank"], f"{context} token rank"),
            bounds["token_rank"],
            f"{context} token rank",
        )
        cache_dims = _dims_by_symbol(cache, f"{context} cache")
        token_dims = _dims_by_symbol(token, f"{context} token")
        _require_equal(
            _single_value(cache_dims["N"], f"{context} cache batch"),
            bounds["batch"],
            f"{context} cache batch",
        )
        _require_equal(
            _single_value(token_dims["N"], f"{context} token batch"),
            bounds["batch"],
            f"{context} token batch",
        )
        _require_equal(
            _single_value(cache_dims["H"], f"{context} cache heads"),
            bounds["heads"],
            f"{context} cache heads",
        )
        _require_equal(
            _single_value(token_dims["H"], f"{context} token heads"),
            bounds["heads"],
            f"{context} token heads",
        )
        _require_equal(
            cache_dims["S"]["min"],
            bounds["source_sequence"]["min"],
            f"{context} source min",
        )
        _require_equal(
            cache_dims["S"]["max"],
            bounds["source_sequence"]["max"],
            f"{context} source max",
        )
        _require_equal(
            _single_value(token_dims["T"], f"{context} token sequence"),
            bounds["token_sequence"],
            f"{context} token sequence",
        )
        _require_equal(
            _single_value(cache_dims["D"], f"{context} cache head_dim"),
            bounds["head_dim"],
            f"{context} cache head_dim",
        )
        _require_equal(
            _single_value(token_dims["D"], f"{context} token head_dim"),
            bounds["head_dim"],
            f"{context} token head_dim",
        )
        if "equal" not in _relationship_types(envelope):
            raise AssertionError(f"{context} missing equality relationships")
        _require_equal(
            envelope["layout"]["cache_requires_vulkan"],
            bounds["cache_requires_vulkan"],
            f"{context} cache_requires_vulkan",
        )
        _require_equal(
            envelope["layout"]["token_requires_vulkan"],
            bounds["token_requires_vulkan"],
            f"{context} token_requires_vulkan",
        )

        for section in ("positive_cases", "negative_cases"):
            for case in spec[section]:
                case_context = f"{context} {section} {case['name']}"
                if len(case["cache_shape"]) != bounds["cache_rank"]:
                    raise AssertionError(f"{case_context} cache rank mismatch")
                if len(case["token_shape"]) != bounds["token_rank"]:
                    raise AssertionError(f"{case_context} token rank mismatch")
                if case["dim"] != bounds["dim"]:
                    raise AssertionError(f"{case_context} dim mismatch")
                if case["dtype"] != bounds["cache_dtype"]:
                    raise AssertionError(f"{case_context} dtype mismatch")
                if case.get("violates") not in (
                    "source_sequence.min",
                    "source_sequence.max",
                ):
                    source_sequence = case["cache_shape"][2]
                    if not (
                        bounds["source_sequence"]["min"]
                        <= source_sequence
                        <= bounds["source_sequence"]["max"]
                    ):
                        raise AssertionError(
                            f"{case_context} source sequence mismatch"
                        )
                if (
                    case.get("violates") != "token_sequence" and
                    case["token_shape"][2] != bounds["token_sequence"]
                ):
                    raise AssertionError(f"{case_context} token sequence mismatch")
                if (
                    case.get("violates") != "heads" and
                    (
                        case["cache_shape"][1] != bounds["heads"] or
                        case["token_shape"][1] != bounds["heads"]
                    )
                ):
                    raise AssertionError(f"{case_context} heads mismatch")
                if (
                    case.get("violates") != "head_dim" and
                    (
                        case["cache_shape"][3] != bounds["head_dim"] or
                        case["token_shape"][3] != bounds["head_dim"]
                    )
                ):
                    raise AssertionError(f"{case_context} head dim mismatch")
        return

    inputs = envelope["inputs"]
    require_fields(inputs, ("empty", "value"), f"{context} inputs")
    empty = inputs["empty"]
    value = inputs["value"]
    _require_equal(
        _single_value(empty["rank"], f"{context} empty rank"),
        bounds["empty_rank"],
        f"{context} empty rank",
    )
    _require_equal(
        _single_value(value["dtype"], f"{context} value dtype"),
        bounds["value_dtype"],
        f"{context} value dtype",
    )
    _require_equal(
        _single_value(value["rank"], f"{context} value rank"),
        bounds["value_rank"],
        f"{context} value rank",
    )
    empty_dims = _dims_by_symbol(empty, f"{context} empty")
    value_dims = _dims_by_symbol(value, f"{context} value")
    _require_equal(
        _single_value(empty_dims["E"], f"{context} empty dim0"),
        bounds["empty_dim0"],
        f"{context} empty dim0",
    )
    _require_equal(
        _single_value(value_dims["N"], f"{context} value batch"),
        bounds["batch"],
        f"{context} value batch",
    )
    _require_equal(
        _single_value(value_dims["H"], f"{context} value heads"),
        bounds["heads"],
        f"{context} value heads",
    )
    _require_equal(
        value_dims["S"]["min"],
        bounds["sequence"]["min"],
        f"{context} sequence min",
    )
    _require_equal(
        value_dims["S"]["max"],
        bounds["sequence"]["max"],
        f"{context} sequence max",
    )
    _require_equal(
        _single_value(value_dims["D"], f"{context} value head_dim"),
        bounds["head_dim"],
        f"{context} value head_dim",
    )
    _require_equal(
        envelope["layout"]["empty_requires_vulkan"],
        bounds["empty_requires_vulkan"],
        f"{context} empty_requires_vulkan",
    )
    _require_equal(
        envelope["layout"]["value_requires_vulkan"],
        bounds["value_requires_vulkan"],
        f"{context} value_requires_vulkan",
    )

    for section in ("positive_cases", "negative_cases"):
        for case in spec[section]:
            case_context = f"{context} {section} {case['name']}"
            if case.get("violates") != "empty_shape" and case["empty_shape"] != [0]:
                raise AssertionError(f"{case_context} empty shape mismatch")
            if len(case["value_shape"]) != bounds["value_rank"]:
                raise AssertionError(f"{case_context} value rank mismatch")
            if case["dim"] != bounds["dim"]:
                raise AssertionError(f"{case_context} dim mismatch")
            if case["dtype"] != bounds["value_dtype"]:
                raise AssertionError(f"{case_context} dtype mismatch")
            if case.get("violates") not in ("sequence.min", "sequence.max"):
                sequence = case["value_shape"][2]
                if not (bounds["sequence"]["min"] <= sequence <= bounds["sequence"]["max"]):
                    raise AssertionError(f"{case_context} sequence mismatch")
            if (
                case.get("violates") != "heads" and
                case["value_shape"][1] != bounds["heads"]
            ):
                raise AssertionError(f"{case_context} heads mismatch")
            if (
                case.get("violates") != "head_dim" and
                case["value_shape"][3] != bounds["head_dim"]
            ):
                raise AssertionError(f"{case_context} head dim mismatch")


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

_SMALL_METADATA_PADDED_CONV2D_LEGAL_KEY_FIELDS = (
    "input_shape",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "output_padding",
    "dtype",
    "input_is_width_packed",
    "input_has_direct_buffer_layout",
    "expected_route_label",
    "expected_post_materialization_route_label",
    "expected_cpu_fallback",
    "expected_sync_readback",
)

_SMALL_METADATA_PADDED_CONV2D_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "input_shape",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "output_padding",
    "dtype",
    "input_is_width_packed",
    "input_has_direct_buffer_layout",
    "expected_native_route",
    "expected_guard_route_label",
    "expected_cpu_fallback",
)

_SMALL_METADATA_PADDED_CONV2D_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.input.dtype",
    "inputs.input.rank",
    "inputs.input.dims.N",
    "inputs.input.dims.CI",
    "inputs.input.dims.H",
    "inputs.input.dims.W",
    "inputs.weight.dtype",
    "inputs.weight.rank",
    "inputs.weight.dims.CO",
    "inputs.weight.dims.CI",
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
    "attributes.input_is_width_packed",
    "attributes.input_has_direct_buffer_layout",
    "attributes.requires_input_materialization",
)

_LINEAR_GELU_BRIDGE_LEGAL_KEY_FIELDS = (
    "input_shape",
    "weight_shape",
    "bias_defined",
    "inference_mode_enabled",
    "has_output",
    "post_op_is_none",
    "alpha_is_one",
    "beta_is_one",
    "gelu_approximate",
    "expected_defer",
    "expected_hit",
    "expected_materialize",
)

_LINEAR_GELU_BRIDGE_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "input_shape",
    "weight_shape",
    "bias_defined",
    "inference_mode_enabled",
    "has_output",
    "post_op_is_none",
    "alpha_is_one",
    "beta_is_one",
    "gelu_approximate",
    "expected_native_route",
    ("runtime_supported", True),
)

_LINEAR_GELU_BRIDGE_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.flattened_input.rank",
    "inputs.flattened_input.dims.FR",
    "inputs.flattened_input.dims.F",
    "inputs.input.rank",
    "inputs.input.dims.B",
    "inputs.input.dims.R",
    "inputs.input.dims.F",
    "inputs.packed_weight.rank",
    "inputs.packed_weight.dims.F",
    "inputs.packed_weight.dims.O",
    "attributes.rank3_batch",
    "attributes.bias_defined",
    "attributes.can_run_float_buffer_linear",
    "attributes.inference_mode_enabled",
    "attributes.has_output",
    "attributes.post_op_is_none",
    "attributes.alpha_is_one",
    "attributes.beta_is_one",
    "attributes.may_defer",
    "attributes.may_consume_gelu_none",
    "attributes.may_consume_gelu_tanh",
)

_GQA_REPEAT_LEGAL_KEY_FIELDS = (
    "query_shape",
    "key_shape",
    "value_shape",
    "dropout_p",
    "is_causal",
    "scale",
    "enable_gqa",
    "dtype",
    "expected_route_label",
    "expected_cpu_fallback",
)

_GQA_REPEAT_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "query_shape",
    "key_shape",
    "value_shape",
    "dropout_p",
    "is_causal",
    "scale",
    "enable_gqa",
    "dtype",
    "expected_native_route",
    "expected_runtime_error",
)

_GQA_REPEAT_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.source.dtype",
    "inputs.source.rank",
    "inputs.source.dims.N",
    "inputs.source.dims.H",
    "inputs.source.dims.S",
    "inputs.source.dims.D",
    "attributes.target_heads",
    "attributes.repeat_factor",
    "attributes.target_sequence",
    "attributes.enable_gqa",
)

_SMALL_SPATIAL_POINTWISE_CONV_LEGAL_KEY_FIELDS = (
    "input_shape",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "dtype",
    "expected_route_label",
    "expected_contract_family",
    "expected_contract_tuple_id",
    "expected_cpu_fallback",
)

_SMALL_SPATIAL_POINTWISE_CONV_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "input_shape",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "dtype",
    "expected_native_route",
    ("expected_error_regex", ""),
    ("expected_cpu_fallback", False),
)

_SMALL_SPATIAL_POINTWISE_CONV_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.input.dtype",
    "inputs.input.rank",
    "inputs.input.dims.N",
    "inputs.input.dims.CI",
    "inputs.input.dims.H",
    "inputs.input.dims.W",
    "inputs.weight.dtype",
    "inputs.weight.rank",
    "inputs.weight.dims.CO",
    "inputs.weight.dims.CI",
    "inputs.weight.dims.KH",
    "inputs.weight.dims.KW",
    "attributes.groups",
    "attributes.kernel_h",
    "attributes.kernel_w",
    "attributes.stride_h",
    "attributes.stride_w",
    "attributes.padding_h",
    "attributes.padding_w",
    "attributes.dilation_h",
    "attributes.dilation_w",
    "attributes.execution_storage",
)

_KV_CACHE_APPEND_SEQUENCE_LEGAL_KEY_FIELDS = (
    "cache_shape",
    "token_shape",
    "dim",
    "dtype",
    "expected_route_label",
    "expected_cpu_fallback",
)

_KV_CACHE_APPEND_SEQUENCE_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "cache_shape",
    "token_shape",
    "dim",
    "dtype",
    "force_buffer_view",
    "expected_native_route",
    "expected_cpu_fallback",
)

_KV_CACHE_APPEND_SEQUENCE_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.cache.dtype",
    "inputs.cache.rank",
    "inputs.cache.dims.N",
    "inputs.cache.dims.H",
    "inputs.cache.dims.S",
    "inputs.cache.dims.D",
    "inputs.token.dtype",
    "inputs.token.rank",
    "inputs.token.dims.N",
    "inputs.token.dims.H",
    "inputs.token.dims.T",
    "inputs.token.dims.D",
    "attributes.dim",
)

_KV_CACHE_APPEND_INITIAL_LEGAL_KEY_FIELDS = (
    "empty_shape",
    "value_shape",
    "dim",
    "dtype",
    "expected_route_label",
    "expected_cpu_fallback",
)

_KV_CACHE_APPEND_INITIAL_ADJACENT_NEGATIVE_KEY_FIELDS = (
    "violates",
    "empty_shape",
    "value_shape",
    "dim",
    "dtype",
    "force_buffer_view",
    "expected_native_route",
    "expected_cpu_fallback",
)

_KV_CACHE_APPEND_INITIAL_ASSIGNMENT_COVERAGE_FIELDS = (
    "inputs.empty.rank",
    "inputs.empty.dims.E",
    "inputs.value.dtype",
    "inputs.value.rank",
    "inputs.value.dims.N",
    "inputs.value.dims.H",
    "inputs.value.dims.S",
    "inputs.value.dims.D",
    "attributes.dim",
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
    "small_spatial_pointwise_conv_sparse_projection_rows": {
        "validate": _validate_small_spatial_pointwise_conv_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _SMALL_SPATIAL_POINTWISE_CONV_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": (
            _SMALL_SPATIAL_POINTWISE_CONV_ASSIGNMENT_COVERAGE_FIELDS
        ),
        "adjacent_negative_key_fields": (
            _SMALL_SPATIAL_POINTWISE_CONV_ADJACENT_NEGATIVE_KEY_FIELDS
        ),
    },
    "small_metadata_padded_conv2d_materialized_buffer_input_2x2": {
        "validate": _validate_small_metadata_padded_conv2d_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _SMALL_METADATA_PADDED_CONV2D_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": (
            _SMALL_METADATA_PADDED_CONV2D_ASSIGNMENT_COVERAGE_FIELDS
        ),
        "adjacent_negative_key_fields": (
            _SMALL_METADATA_PADDED_CONV2D_ADJACENT_NEGATIVE_KEY_FIELDS
        ),
    },
    "linear_gelu_bridge_backbone_mlp_hidden384_to1536": {
        "validate": _validate_linear_gelu_bridge_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _LINEAR_GELU_BRIDGE_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": (
            _LINEAR_GELU_BRIDGE_ASSIGNMENT_COVERAGE_FIELDS
        ),
        "adjacent_negative_key_fields": (
            _LINEAR_GELU_BRIDGE_ADJACENT_NEGATIVE_KEY_FIELDS
        ),
    },
    "gqa_repeat_batch1_heads4_factor4_sequence100_to116_dim128": {
        "validate": _validate_gqa_repeat_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _GQA_REPEAT_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": _GQA_REPEAT_ASSIGNMENT_COVERAGE_FIELDS,
        "adjacent_negative_key_fields": _GQA_REPEAT_ADJACENT_NEGATIVE_KEY_FIELDS,
    },
    "kv_cache_append_sequence_append": {
        "validate": _validate_kv_cache_append_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _KV_CACHE_APPEND_SEQUENCE_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": (
            _KV_CACHE_APPEND_SEQUENCE_ASSIGNMENT_COVERAGE_FIELDS
        ),
        "adjacent_negative_key_fields": (
            _KV_CACHE_APPEND_SEQUENCE_ADJACENT_NEGATIVE_KEY_FIELDS
        ),
    },
    "kv_cache_append_initial_cache": {
        "validate": _validate_kv_cache_append_shape_envelope,
        "assignment_cases": _generated_shape_envelope_assignment_cases,
        "legal_cases": _checked_in_shape_envelope_legal_cases,
        "adjacent_negative_cases": (
            _checked_in_shape_envelope_adjacent_negative_cases
        ),
        "legal_key_fields": _KV_CACHE_APPEND_INITIAL_LEGAL_KEY_FIELDS,
        "assignment_coverage_fields": (
            _KV_CACHE_APPEND_INITIAL_ASSIGNMENT_COVERAGE_FIELDS
        ),
        "adjacent_negative_key_fields": (
            _KV_CACHE_APPEND_INITIAL_ADJACENT_NEGATIVE_KEY_FIELDS
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


def shape_envelope_sparse_rowset_summary(repo_root):
    rows = []
    for file_name, spec in validate_all_contract_specs(repo_root):
        rows.extend(validate_shape_envelope_sparse_rowsets(file_name, spec))
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


def _contract_source_paths(repo_root):
    patterns = (
        os.path.join(
            repo_root,
            "aten",
            "src",
            "ATen",
            "native",
            "vulkan",
            "planning",
            "ExecutionContracts*.cpp",
        ),
        os.path.join(
            repo_root,
            "aten",
            "src",
            "ATen",
            "native",
            "vulkan",
            "planning",
            "generated",
            "ExecutionContracts*Spec.h",
        ),
    )
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def _repo_relative_display(repo_root, path):
    return os.path.relpath(path, repo_root).replace(os.sep, "/")


def execution_contract_source_summary(repo_root):
    contracts = {}
    for path in _contract_source_paths(repo_root):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for contract_name in sorted(set(CONTRACT_NAME_LITERAL_RE.findall(text))):
            contracts.setdefault(contract_name, set()).add(
                _repo_relative_display(repo_root, path)
            )
    return [
        {
            "contract_name": contract_name,
            "source_files": tuple(sorted(source_files)),
        }
        for contract_name, source_files in sorted(contracts.items())
    ]


def _active_temporary_exception_sections(repo_root):
    path = os.path.join(repo_root, TEMPORARY_EXCEPTIONS_FILE)
    with open(path, encoding="utf-8") as handle:
        doc = handle.read()
    if "## Active Exceptions" not in doc:
        return []
    active = doc.split("## Active Exceptions", 1)[1]
    active = active.split("## Rules For New Exceptions", 1)[0]
    return [
        {
            "title": match.group(1),
            "body": match.group(2),
        }
        for match in re.finditer(r"^### (.+?)\n(.*?)(?=^### |\Z)", active, re.S | re.M)
    ]


def _temporary_exception_tokens(file_name, spec, contract_name):
    tokens = {contract_name}
    if file_name:
        tokens.add(file_name)
        tokens.add(os.path.splitext(file_name)[0])
    if spec is not None:
        for field in ("contract_name", "family", "tuple_id", "route_label"):
            tokens.add(spec[field])
        envelope = spec.get("shape_envelope")
        if envelope is not None:
            tokens.add(envelope["role"])
    return tuple(sorted(token for token in tokens if token))


def _temporary_exception_link(sections, tokens, allow_generic):
    generic_section = None
    for section in sections:
        haystack = f"{section['title']}\n{section['body']}"
        if section["title"] == GENERIC_EXACT_TUPLE_EXCEPTION:
            generic_section = section
            continue
        if any(token in haystack for token in tokens):
            return section["title"], "specific"
    if allow_generic and generic_section is not None:
        return generic_section["title"], "generic"
    return "", "none"


def _coverage_category(spec, has_generated_cpp):
    if spec.get("source_status") == "schema_only":
        return "schema_only_spec"
    if "shape_envelope" in spec and has_generated_cpp:
        return "generated_shape_envelope"
    if "shape_envelope" in spec:
        return "shape_envelope_without_generated_header"
    return "json_spec_without_shape_envelope"


def _coverage_category_is_debt(category):
    return category in {
        "json_spec_without_shape_envelope",
        "shape_envelope_without_generated_header",
        "live_contract_without_json_spec",
    }


def contract_coverage_census(repo_root):
    specs = validate_all_contract_specs(repo_root)
    generated_by_spec = {
        entry["spec_file"]: entry
        for entry in generated_cpp_manifest_entries(repo_root)
    }
    source_rows = execution_contract_source_summary(repo_root)
    sources_by_contract = {
        row["contract_name"]: row["source_files"]
        for row in source_rows
    }
    exception_sections = _active_temporary_exception_sections(repo_root)

    spec_rows = []
    for file_name, spec in specs:
        generated_entry = generated_by_spec.get(file_name)
        generated_header = ""
        has_generated_cpp = False
        if generated_entry is not None:
            generated_header = generated_entry["header"]
            has_generated_cpp = os.path.isfile(
                _repo_relative_path(
                    repo_root,
                    generated_header,
                    f"{GENERATED_CPP_MANIFEST_FILE} {file_name} header",
                )
            )
        category = _coverage_category(spec, has_generated_cpp)
        temporary_exception, temporary_exception_scope = _temporary_exception_link(
            exception_sections,
            _temporary_exception_tokens(file_name, spec, spec["contract_name"]),
            allow_generic=True,
        )
        spec_rows.append(
            {
                "row_kind": "spec",
                "category": category,
                "file_name": file_name,
                "contract_name": spec["contract_name"],
                "family": spec["family"],
                "tuple_id": spec["tuple_id"],
                "shape_envelope_role": spec.get("shape_envelope", {}).get("role", ""),
                "generated_cpp_header": generated_header if has_generated_cpp else "",
                "source_files": sources_by_contract.get(spec["contract_name"], ()),
                "temporary_exception": temporary_exception,
                "temporary_exception_scope": temporary_exception_scope,
                "exact_row_debt": _coverage_category_is_debt(category),
            }
        )

    spec_contracts = {spec["contract_name"] for _, spec in specs}
    live_contract_rows = []
    for source_row in source_rows:
        contract_name = source_row["contract_name"]
        if contract_name in spec_contracts:
            continue
        temporary_exception, temporary_exception_scope = _temporary_exception_link(
            exception_sections,
            _temporary_exception_tokens("", None, contract_name),
            allow_generic=True,
        )
        live_contract_rows.append(
            {
                "row_kind": "live_contract",
                "category": "live_contract_without_json_spec",
                "file_name": "",
                "contract_name": contract_name,
                "family": "",
                "tuple_id": "",
                "shape_envelope_role": "",
                "generated_cpp_header": "",
                "source_files": source_row["source_files"],
                "temporary_exception": temporary_exception,
                "temporary_exception_scope": temporary_exception_scope,
                "exact_row_debt": True,
            }
        )

    return {
        "spec_rows": spec_rows,
        "live_contract_rows": live_contract_rows,
    }


def _contract_coverage_all_rows(census):
    return census["spec_rows"] + census["live_contract_rows"]


def contract_coverage_census_summary(repo_root):
    census = contract_coverage_census(repo_root)
    rows = _contract_coverage_all_rows(census)
    categories = {
        "generated_shape_envelope": 0,
        "shape_envelope_without_generated_header": 0,
        "json_spec_without_shape_envelope": 0,
        "schema_only_spec": 0,
        "live_contract_without_json_spec": 0,
    }
    for row in rows:
        categories[row["category"]] += 1
    return {
        "specs": len(census["spec_rows"]),
        "live_contract_without_json_spec": categories[
            "live_contract_without_json_spec"
        ],
        "generated_shape_envelope": categories["generated_shape_envelope"],
        "shape_envelope_without_generated_header": categories[
            "shape_envelope_without_generated_header"
        ],
        "json_spec_without_shape_envelope": categories[
            "json_spec_without_shape_envelope"
        ],
        "schema_only_spec": categories["schema_only_spec"],
        "temporary_exception_specific": sum(
            row["temporary_exception_scope"] == "specific" for row in rows
        ),
        "temporary_exception_generic": sum(
            row["temporary_exception_scope"] == "generic" for row in rows
        ),
        "temporary_exception_missing": sum(
            row["temporary_exception_scope"] == "none" for row in rows
        ),
        "exact_row_debt": sum(row["exact_row_debt"] for row in rows),
    }


def format_contract_coverage_census_summary(summary):
    return (
        "contract coverage census "
        f"specs={summary['specs']} "
        f"generated_shape_envelope={summary['generated_shape_envelope']} "
        "shape_envelope_without_generated_header="
        f"{summary['shape_envelope_without_generated_header']} "
        f"json_spec_without_shape_envelope="
        f"{summary['json_spec_without_shape_envelope']} "
        f"schema_only_spec={summary['schema_only_spec']} "
        f"live_contract_without_json_spec="
        f"{summary['live_contract_without_json_spec']} "
        f"temporary_exception_specific="
        f"{summary['temporary_exception_specific']} "
        f"temporary_exception_generic={summary['temporary_exception_generic']} "
        f"temporary_exception_missing={summary['temporary_exception_missing']} "
        f"exact_row_debt={summary['exact_row_debt']}"
    )


def _format_optional_field(name, value):
    if not value:
        return ""
    return f"{name}={json.dumps(value)}"


def format_contract_coverage_census_row(row):
    fields = [
        f"{row['row_kind']}:{row['category']}",
        f"contract={row['contract_name']}",
    ]
    for name in (
        "file_name",
        "family",
        "tuple_id",
        "shape_envelope_role",
        "generated_cpp_header",
    ):
        formatted = _format_optional_field(name, row[name])
        if formatted:
            fields.append(formatted)
    if row["source_files"]:
        fields.append(f"sources={json.dumps(list(row['source_files']))}")
    if row["temporary_exception"]:
        fields.append(
            f"temporary_exception={json.dumps(row['temporary_exception'])}"
        )
    fields.append(f"temporary_exception_scope={row['temporary_exception_scope']}")
    fields.append(f"exact_row_debt={'yes' if row['exact_row_debt'] else 'no'}")
    return " ".join(fields)


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
    parser.add_argument("--validate-sparse-rowsets", action="store_true")
    parser.add_argument("--validate-legal-cases", action="store_true")
    parser.add_argument("--validate-adjacent-negatives", action="store_true")
    parser.add_argument("--validate-fuzz-assignments", action="store_true")
    parser.add_argument("--validate-fuzz-assignment-coverage", action="store_true")
    parser.add_argument("--validate-generated-cpp-manifest", action="store_true")
    parser.add_argument("--contract-coverage-census", action="store_true")
    parser.add_argument("--validate-contract-coverage-census", action="store_true")
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
    if args.validate_sparse_rowsets:
        sparse_rows = shape_envelope_sparse_rowset_summary(args.repo_root)
        total_rows = sum(row["row_count"] for row in sparse_rows)
        total_independent = sum(
            row["independent_identity_cross_product"] for row in sparse_rows
        )
        total_sparse_gap = sum(row["sparse_cross_product_gap"] for row in sparse_rows)
        rowsets = ", ".join(
            f"{row['file_name']}:{row['rowset_name']}:rows={row['row_count']}:"
            f"cross_product={row['independent_identity_cross_product']}:"
            f"gap={row['sparse_cross_product_gap']}"
            for row in sparse_rows
        )
        rowset_details = f" {rowsets}" if rowsets else ""
        print(
            "validated "
            f"{len(sparse_rows)} ShapeEnvelope sparse rowsets "
            f"rows={total_rows} independent_cross_product={total_independent} "
            f"sparse_gap={total_sparse_gap}{rowset_details}"
        )
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
    if args.contract_coverage_census or args.validate_contract_coverage_census:
        census = contract_coverage_census(args.repo_root)
        summary = contract_coverage_census_summary(args.repo_root)
        prefix = "validated " if args.validate_contract_coverage_census else ""
        print(prefix + format_contract_coverage_census_summary(summary))
        if args.contract_coverage_census:
            for row in _contract_coverage_all_rows(census):
                print(format_contract_coverage_census_row(row))


if __name__ == "__main__":
    _main()
