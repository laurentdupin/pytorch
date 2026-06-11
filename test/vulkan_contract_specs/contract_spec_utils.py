import argparse
import glob
import json
import os


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


def _validate_shape_envelope_relationships(relationships, context):
    _require_list(relationships, f"{context} relationships", allow_empty=True)
    for index, relationship in enumerate(relationships):
        rel_context = f"{context} relationships[{index}]"
        _require_mapping(relationship, rel_context)
        _require_non_empty_string(relationship, "type", rel_context)
        rel_type = relationship["type"]
        if rel_type not in ("equal", "sum_output", "product"):
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


def _single_value(field, context):
    values = field.get("values")
    _require_list(values, f"{context}.values")
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


def _validate_shape_envelope_common(file_name, spec, envelope):
    context = f"{file_name} ShapeEnvelope v1"
    _require_mapping(envelope, context)
    require_fields(envelope, SHAPE_ENVELOPE_REQUIRED_FIELDS, context)
    if envelope["version"] != 1:
        raise AssertionError(f"{context} version must be 1")
    _require_non_empty_string(envelope, "role", context)

    _validate_shape_envelope_inputs(envelope["inputs"], context)
    _validate_shape_envelope_attributes(envelope["attributes"], context)
    _validate_bounds_tree(envelope["bounds"], f"{context} bounds")
    _validate_shape_envelope_relationships(envelope["relationships"], context)
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


def validate_shape_envelope_spec(file_name, spec):
    envelope = spec.get("shape_envelope")
    if envelope is None:
        return None
    _validate_shape_envelope_common(file_name, spec, envelope)
    role = envelope["role"]
    if role == "multi_input_rank4_channel_cat":
        _validate_channel_cat_shape_envelope(file_name, spec, envelope)
    elif role == "embedding_lookup_small_bounded":
        _validate_embedding_lookup_shape_envelope(file_name, spec, envelope)
    else:
        raise AssertionError(f"{file_name} unsupported ShapeEnvelope role {role!r}")
    return envelope


def contract_spec_dir(repo_root):
    return os.path.join(repo_root, "test", "vulkan_contract_specs")


def contract_spec_paths(repo_root):
    return sorted(glob.glob(os.path.join(contract_spec_dir(repo_root), "*.json")))


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


if __name__ == "__main__":
    _main()
