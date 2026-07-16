#!/usr/bin/env python3

import argparse
import json
import re
import sys


GENERATED_CPP_FUNCTION_RE = re.compile(
    r"^[ \t]*(?:constexpr|inline)\s+(?:[^\n(]*?\s+)?"
    r"([a-z][a-z0-9_]*)\s*\(",
    re.MULTILINE,
)


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


def _generated_cpp_function_spans(output):
    functions = {}
    for match in GENERATED_CPP_FUNCTION_RE.finditer(output):
        name = match.group(1)
        _require(name not in functions, f"duplicate generated C++ helper {name!r}")
        open_brace = output.find("{", match.end())
        _require(open_brace >= 0, f"generated C++ helper {name!r} has no body")

        depth = 0
        close_brace = None
        for index in range(open_brace, len(output)):
            if output[index] == "{":
                depth += 1
            elif output[index] == "}":
                depth -= 1
                if depth == 0:
                    close_brace = index + 1
                    break
        _require(close_brace is not None, f"generated C++ helper {name!r} is unclosed")

        end = close_brace
        for _ in range(2):
            if output.startswith("\n", end):
                end += 1
        functions[name] = {
            "body": output[open_brace:close_brace],
            "span": (match.start(), end),
        }
    return functions


def _prune_generated_cpp_helpers(output, entry_points):
    _require(
        isinstance(entry_points, list),
        "generated_cpp_entry_points must be a list",
    )
    for index, entry_point in enumerate(entry_points):
        _require(
            isinstance(entry_point, str)
            and re.fullmatch(r"[a-z][a-z0-9_]*", entry_point),
            f"generated_cpp_entry_points[{index}] invalid",
        )
    _require(
        len(entry_points) == len(set(entry_points)),
        "generated_cpp_entry_points must be unique",
    )

    functions = _generated_cpp_function_spans(output)
    missing = sorted(set(entry_points) - set(functions))
    _require(not missing, f"generated C++ entry points not emitted: {missing}")

    dependencies = {}
    for name, function in functions.items():
        dependencies[name] = {
            candidate
            for candidate in functions
            if candidate != name
            and re.search(
                rf"\b{re.escape(candidate)}\s*\(",
                function["body"],
            )
        }

    live = set(entry_points)
    pending = list(entry_points)
    while pending:
        name = pending.pop()
        for dependency in dependencies[name] - live:
            live.add(dependency)
            pending.append(dependency)

    for name in sorted(
        set(functions) - live,
        key=lambda candidate: functions[candidate]["span"][0],
        reverse=True,
    ):
        start, end = functions[name]["span"]
        output = output[:start] + output[end:]
    return output


def _validate_bool(value, context):
    _require(isinstance(value, bool), f"{context} must be boolean")


def _validate_int(value, context):
    _require(isinstance(value, int), f"{context} must be integer")


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


def _broadcast_compatible_relationships(envelope, context):
    relationships = envelope.get("relationships", [])
    _require(isinstance(relationships, list), f"{context}.relationships invalid")
    matches = []
    for index, relationship in enumerate(relationships):
        if relationship.get("type") != "broadcast_compatible":
            continue
        relationship_context = f"{context}.relationships[{index}]"
        _require_keys(
            relationship,
            ("left", "right", "result", "align", "max_rank"),
            relationship_context,
        )
        for key in ("left", "right", "result", "align"):
            _require_non_empty_string(relationship, key, relationship_context)
        _require(
            relationship["align"] == "right",
            f"{relationship_context}.align must be right",
        )
        _validate_int(relationship["max_rank"], f"{relationship_context}.max_rank")
        _require(
            relationship["max_rank"] > 0,
            f"{relationship_context}.max_rank must be positive",
        )
        matches.append(relationship)
    _require(
        len(matches) <= 1,
        f"{context} supports at most one broadcast_compatible relationship",
    )
    return matches


def _product_equal_relationship(envelope, context):
    relationships = envelope.get("relationships", [])
    _require(isinstance(relationships, list), f"{context}.relationships invalid")
    product_by_result = {}
    equals = []
    for index, relationship in enumerate(relationships):
        relationship_type = relationship.get("type")
        if relationship_type == "product":
            relationship_context = f"{context}.relationships[{index}]"
            _require_keys(
                relationship,
                ("input", "dims", "result"),
                relationship_context,
            )
            for key in ("input", "result"):
                _require_non_empty_string(relationship, key, relationship_context)
            _require(
                relationship["dims"] == "all",
                f"{relationship_context}.dims must be all",
            )
            result = relationship["result"]
            _require(
                result not in product_by_result,
                f"{relationship_context}.result duplicated",
            )
            product_by_result[result] = relationship
        elif relationship_type == "equal":
            relationship_context = f"{context}.relationships[{index}]"
            _require_keys(relationship, ("scope", "fields"), relationship_context)
            _require(
                relationship["scope"] == "derived",
                f"{relationship_context}.scope must be derived",
            )
            fields = relationship["fields"]
            _require(
                isinstance(fields, list) and len(fields) == 2,
                f"{relationship_context}.fields must contain two fields",
            )
            for field_index, field in enumerate(fields):
                _require(
                    isinstance(field, str) and field != "",
                    f"{relationship_context}.fields[{field_index}] invalid",
                )
            equals.append(relationship)

    matches = []
    for relationship in equals:
        fields = relationship["fields"]
        if all(field in product_by_result for field in fields):
            matches.append(
                {
                    "left_input": product_by_result[fields[0]]["input"],
                    "right_input": product_by_result[fields[1]]["input"],
                }
            )
    _require(
        len(matches) <= 1,
        f"{context} supports at most one product/equal relationship",
    )
    return matches[0] if matches else None


def _product_value_relationships(envelope, context):
    relationships = envelope.get("relationships", [])
    _require(isinstance(relationships, list), f"{context}.relationships invalid")
    matches = []
    seen_results = set()
    for index, relationship in enumerate(relationships):
        if relationship.get("type") != "product":
            continue
        relationship_context = f"{context}.relationships[{index}]"
        _require_keys(
            relationship,
            ("input", "dims", "result"),
            relationship_context,
        )
        for key in ("input", "result"):
            _require_non_empty_string(relationship, key, relationship_context)
        _require(
            relationship["dims"] == "all",
            f"{relationship_context}.dims must be all",
        )
        result = relationship["result"]
        _require(
            result not in seen_results,
            f"{relationship_context}.result duplicated",
        )
        seen_results.add(result)
        matches.append(
            {
                "input": relationship["input"],
                "result": result,
            }
        )
    return matches


def _sum_output_relationship(envelope, input_name, dim_name, result_name, context):
    relationships = envelope.get("relationships", [])
    _require(isinstance(relationships, list), f"{context}.relationships invalid")
    matches = []
    for index, relationship in enumerate(relationships):
        if relationship.get("type") != "sum_output":
            continue
        relationship_context = f"{context}.relationships[{index}]"
        _require_keys(
            relationship,
            ("input", "dim", "result"),
            relationship_context,
        )
        for key in ("input", "dim", "result"):
            _require_non_empty_string(relationship, key, relationship_context)
        if (
            relationship["input"] == input_name
            and relationship["dim"] == dim_name
            and relationship["result"] == result_name
        ):
            matches.append(relationship)
    _require(len(matches) == 1, f"{context} sum_output relationship missing")
    return matches[0]


SCALAR_EQUAL_HELPER_SCOPES = {
    "input_weight_channels",
    "batch",
    "heads",
    "head_dim",
    "square_scores",
}

MULTI_FIELD_EQUAL_HELPER_SCOPES = {
    "feature_count",
}


def _symbol_field_for_input_details(envelope, field_path, context):
    parts = field_path.split(".")
    _require(len(parts) == 2, f"{context} field path {field_path!r} invalid")
    input_name, symbol = parts
    inputs = envelope.get("inputs", {})
    _require(isinstance(inputs, dict), f"{context}.inputs invalid")
    input_spec = inputs.get(input_name)
    _require(isinstance(input_spec, dict), f"{context}.{input_name} missing")
    dims = input_spec.get("dims")
    _require(isinstance(dims, list), f"{context}.{input_name}.dims invalid")
    for dim in dims:
        if dim.get("symbol") == symbol:
            _require_non_empty_string(
                dim, "field", f"{context}.{input_name}.{symbol}"
            )
            kind = input_spec.get("kind")
            _require(
                isinstance(kind, str) and kind != "",
                f"{context}.{input_name}.kind invalid",
            )
            return {
                "input": input_name,
                "symbol": symbol,
                "field": dim["field"],
                "kind": kind,
            }
    raise RuntimeError(f"{context} field path {field_path!r} does not map to a dim")


def _symbol_field_for_input(envelope, field_path, context):
    details = _symbol_field_for_input_details(envelope, field_path, context)
    return details["input"], details["symbol"], details["field"]


def _scalar_equal_relationships(envelope, context):
    relationships = envelope.get("relationships", [])
    _require(isinstance(relationships, list), f"{context}.relationships invalid")
    matches = []
    seen_scopes = set()
    for index, relationship in enumerate(relationships):
        if relationship.get("type") != "equal":
            continue
        scope = relationship.get("scope")
        if scope not in SCALAR_EQUAL_HELPER_SCOPES:
            continue
        relationship_context = f"{context}.relationships[{index}]"
        _require_keys(relationship, ("scope", "fields"), relationship_context)
        fields = relationship["fields"]
        _require(
            isinstance(fields, list) and len(fields) == 2,
            f"{relationship_context}.fields must contain two fields",
        )
        left = _symbol_field_for_input(envelope, fields[0], relationship_context)
        right = _symbol_field_for_input(envelope, fields[1], relationship_context)
        if left[1] != right[1]:
            _require(
                scope == "square_scores"
                and left[0] == right[0]
                and left[2] == right[2],
                f"{relationship_context}.fields must reference the same symbol",
            )
        _require(scope not in seen_scopes, f"{relationship_context}.scope duplicated")
        seen_scopes.add(scope)
        matches.append(
            {
                "scope": scope,
                "left_input": left[0],
                "left_field": left[2],
                "right_input": right[0],
                "right_field": right[2],
            }
        )
    return matches


def _multi_field_equal_relationships(envelope, context):
    relationships = envelope.get("relationships", [])
    _require(isinstance(relationships, list), f"{context}.relationships invalid")
    attributes = envelope.get("attributes", {})
    _require(isinstance(attributes, dict), f"{context}.attributes invalid")
    matches = []
    seen_scopes = set()
    for index, relationship in enumerate(relationships):
        if relationship.get("type") != "equal":
            continue
        scope = relationship.get("scope")
        if scope not in MULTI_FIELD_EQUAL_HELPER_SCOPES:
            continue
        relationship_context = f"{context}.relationships[{index}]"
        _require_keys(relationship, ("scope", "fields"), relationship_context)
        fields = relationship["fields"]
        _require(
            isinstance(fields, list) and len(fields) >= 2,
            f"{relationship_context}.fields must contain at least two fields",
        )
        participants = []
        for field_index, field in enumerate(fields):
            details = _symbol_field_for_input_details(
                envelope, field, relationship_context
            )
            _require(
                details["kind"] in ("tensor", "optional_tensor"),
                f"{relationship_context}.fields[{field_index}] kind unsupported",
            )
            optional = details["kind"] == "optional_tensor"
            if optional:
                attribute_name = f"{details['input']}_has_value"
                attribute = attributes.get(attribute_name)
                _require(
                    isinstance(attribute, dict),
                    f"{relationship_context}.{attribute_name} attribute missing",
                )
                values = attribute.get("values")
                _require(
                    isinstance(values, list)
                    and values
                    and all(isinstance(value, bool) for value in values),
                    f"{relationship_context}.{attribute_name}.values invalid",
                )
            participants.append(
                {
                    "input": details["input"],
                    "symbol": details["symbol"],
                    "field": details["field"],
                    "optional": optional,
                }
            )
        symbol = participants[0]["symbol"]
        _require(
            not participants[0]["optional"],
            f"{relationship_context}.fields[0] must be a required reference",
        )
        for field_index, participant in enumerate(participants[1:], start=1):
            _require(
                participant["symbol"] == symbol,
                f"{relationship_context}.fields[{field_index}] must reference {symbol}",
            )
        _require(scope not in seen_scopes, f"{relationship_context}.scope duplicated")
        seen_scopes.add(scope)
        matches.append(
            {
                "scope": scope,
                "participants": participants,
            }
        )
    return matches


def _scalar_equal_parameter_name(input_name, field_name, used_names):
    if input_name == "input" and field_name not in used_names:
        candidate = field_name
    else:
        candidate = f"{input_name}_{field_name}"
    base_candidate = candidate
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_candidate}_{suffix}"
        suffix += 1
    _require(
        candidate not in used_names,
        f"duplicate scalar equal parameter {candidate}",
    )
    used_names.add(candidate)
    return candidate


def _scalar_equal_helper_lines(role_func_prefix, relationships):
    helper_lines = []
    for relationship in relationships:
        used_names = set()
        left_name = _scalar_equal_parameter_name(
            _cpp_lower_identifier(relationship["left_input"]),
            _cpp_lower_identifier(relationship["left_field"]),
            used_names,
        )
        right_name = _scalar_equal_parameter_name(
            _cpp_lower_identifier(relationship["right_input"]),
            _cpp_lower_identifier(relationship["right_field"]),
            used_names,
        )
        helper_name = (
            f"{role_func_prefix}_{_cpp_lower_identifier(relationship['scope'])}_equal"
        )
        helper_lines.extend(
            [
                f"constexpr bool {helper_name}(",
                f"    const std::int64_t {left_name},",
                f"    const std::int64_t {right_name}) {{",
                f"  return {left_name} == {right_name};",
                "}",
                "",
            ]
        )
    return helper_lines


def _multi_field_equal_parameter_name(participant, used_names):
    candidate = (
        f"{_cpp_lower_identifier(participant['input'])}_"
        f"{_cpp_lower_identifier(participant['field'])}"
    )
    _require(
        candidate not in used_names,
        f"duplicate multi-field equal parameter {candidate}",
    )
    used_names.add(candidate)
    return candidate


def _multi_field_equal_helper_lines(role_func_prefix, relationships):
    helper_lines = []
    for relationship in relationships:
        used_names = set()
        parameters = []
        value_names = []
        for participant in relationship["participants"]:
            value_name = _multi_field_equal_parameter_name(participant, used_names)
            if participant["optional"]:
                has_value_name = (
                    f"{_cpp_lower_identifier(participant['input'])}_has_value"
                )
                _require(
                    has_value_name not in used_names,
                    f"duplicate multi-field equal parameter {has_value_name}",
                )
                used_names.add(has_value_name)
                parameters.append(f"    const bool {has_value_name},")
                parameters.append(f"    const std::int64_t {value_name},")
                value_names.append((value_name, has_value_name))
            else:
                parameters.append(f"    const std::int64_t {value_name},")
                value_names.append((value_name, None))
        parameters[-1] = parameters[-1].rstrip(",") + ") {"
        reference_name = value_names[0][0]
        conditions = []
        for value_name, has_value_name in value_names[1:]:
            if has_value_name is None:
                conditions.append(f"{reference_name} == {value_name}")
            else:
                conditions.append(
                    f"(!{has_value_name} || {reference_name} == {value_name})"
                )
        helper_name = (
            f"{role_func_prefix}_{_cpp_lower_identifier(relationship['scope'])}_equal"
        )
        body = " &&\n      ".join(conditions) if conditions else "true"
        helper_lines.append(f"constexpr bool {helper_name}(")
        helper_lines.extend(parameters)
        helper_lines.append(f"  return {body};")
        helper_lines.extend(
            [
                "}",
                "",
            ]
        )
    return helper_lines


def _simple_bounds_shape_envelope_fields(bounds):
    dtype_fields = []
    int_fields = []
    list_int_fields = []
    range_fields = []
    min_range_fields = []
    bool_fields = []
    unsupported = []
    for key, value in bounds.items():
        if isinstance(value, bool):
            bool_fields.append(key)
        elif isinstance(value, str) and (key == "dtype" or key.endswith("_dtype")):
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
        elif (
            isinstance(value, dict)
            and set(value) == {"min"}
            and isinstance(value["min"], int)
        ):
            min_range_fields.append(key)
        else:
            unsupported.append(key)
    if unsupported:
        return None
    if not (
        int_fields
        or list_int_fields
        or range_fields
        or min_range_fields
        or bool_fields
    ):
        return None
    return {
        "dtype": dtype_fields,
        "int": int_fields,
        "list_int": list_int_fields,
        "range": range_fields,
        "min_range": min_range_fields,
        "bool": bool_fields,
    }


def _shape_layout_simple_bounds_shape_envelope_fields(bounds):
    int_fields = []
    multiple_of_fields = []
    range_fields = []
    bool_fields = []
    string_fields = []
    unsupported = []
    for key, value in bounds.items():
        if isinstance(value, bool):
            bool_fields.append(key)
        elif isinstance(value, int) and key.endswith("_multiple_of"):
            multiple_of_fields.append(key)
        elif isinstance(value, int):
            int_fields.append(key)
        elif (
            isinstance(value, dict)
            and set(value) == {"min", "max"}
            and isinstance(value["min"], int)
            and isinstance(value["max"], int)
        ):
            range_fields.append(key)
        elif isinstance(value, str):
            string_fields.append(key)
        else:
            unsupported.append(key)
    if unsupported:
        return None
    if not (int_fields or multiple_of_fields or range_fields):
        return None
    return {
        "int": int_fields,
        "multiple_of": multiple_of_fields,
        "range": range_fields,
        "bool": bool_fields,
        "string": string_fields,
    }


def _singular_field_name(field):
    return field[:-1] if field.endswith("s") else field


def _bound_path_value(bounds, path):
    value = bounds
    for part in path.split("."):
        _require(isinstance(value, dict) and part in value, f"bounds path {path} invalid")
        value = value[part]
    return value


def _variadic_tensor_list_input(spec):
    envelope = spec.get("shape_envelope")
    if not isinstance(envelope, dict):
        return None
    inputs = envelope.get("inputs")
    if not isinstance(inputs, dict):
        return None
    candidates = [
        (name, input_spec)
        for name, input_spec in inputs.items()
        if isinstance(input_spec, dict)
        and input_spec.get("kind") == "variadic_tensor_list"
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _required_flag_field(flag):
    for prefix in ("is_", "has_", "supports_"):
        if flag.startswith(prefix):
            return flag.removeprefix(prefix)
    return flag


def _required_flag_bound_key(flag):
    return f"requires_{_required_flag_field(flag)}"


def _validate_generic_variadic_tensor_list_shape_envelope_spec(spec):
    variadic = _variadic_tensor_list_input(spec)
    if variadic is None:
        return None
    input_name, input_spec = variadic
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
            "matcher",
        ),
        "ShapeEnvelope variadic tensor-list contract spec",
    )
    for key in ("contract_name", "family", "tuple_id", "writer_op", "route_label"):
        _require_non_empty_string(
            spec, key, "ShapeEnvelope variadic tensor-list contract spec"
        )

    envelope = spec["shape_envelope"]
    _require(envelope.get("version") == 1, "shape_envelope.version must be 1")
    _require_non_empty_string(envelope, "role", "shape_envelope")
    metadata = envelope.get("metadata")
    _validate_contract_metadata(metadata, "ShapeEnvelope metadata")
    _require(spec["metadata"] == metadata, "metadata must match shape_envelope")

    bounds = envelope.get("bounds")
    _require(spec["bounds"] == bounds, "bounds must match shape_envelope")
    _require(isinstance(bounds, dict), "ShapeEnvelope bounds must be an object")

    matcher = spec["matcher"]
    _require(isinstance(matcher, dict), "matcher must be an object")
    _require_keys(
        matcher,
        (
            "tensor_info",
            "reference_index",
            "per_input_same_as_reference",
            "per_input_required_flags",
            "channel_axis",
            "aggregate",
        ),
        "ShapeEnvelope variadic tensor-list matcher",
    )
    _require_non_empty_string(matcher, "tensor_info", "matcher")
    _validate_int(matcher["reference_index"], "matcher.reference_index")
    for key in ("per_input_same_as_reference", "per_input_required_flags"):
        _require(
            isinstance(matcher[key], list) and matcher[key],
            f"matcher.{key} must be a non-empty list",
        )
        for index, value in enumerate(matcher[key]):
            _require(isinstance(value, str) and value, f"matcher.{key}[{index}] invalid")
    _require_non_empty_string(matcher, "channel_axis", "matcher")

    aggregate_matcher = matcher["aggregate"]
    _require(isinstance(aggregate_matcher, dict), "matcher.aggregate invalid")
    _require_keys(
        aggregate_matcher,
        ("field", "result_name", "min", "max_from_bounds", "multiple_of_from_bounds"),
        "matcher.aggregate",
    )
    _require(
        aggregate_matcher["field"] == matcher["channel_axis"],
        "matcher.aggregate.field must match matcher.channel_axis",
    )
    _validate_int(aggregate_matcher["min"], "matcher.aggregate.min")

    count = input_spec.get("count")
    _validate_bound_pair(count, f"shape_envelope.inputs.{input_name}.count")
    _require(bounds.get("input_count") == count, "bounds.input_count must match count")
    _require(bounds["dtype"] in SCALAR_TYPE_BY_DTYPE, "unsupported dtype")
    dtype_values = input_spec.get("dtype", {}).get("values")
    _require(dtype_values == [bounds["dtype"]], "input dtype must match bounds")
    rank_values = input_spec.get("rank", {}).get("values")
    _require(rank_values == [bounds["rank"]], "input rank must match bounds")
    for key in ("rank", "dim"):
        _validate_int(bounds[key], f"bounds.{key}")
    attr_dim = envelope.get("attributes", {}).get("dim", {}).get("values")
    _require(attr_dim == [bounds["dim"]], "attribute dim must match bounds")

    dims = input_spec.get("dims")
    _require(isinstance(dims, list) and dims, "variadic input dims invalid")
    dims_by_field = {}
    for dim in dims:
        _require(isinstance(dim, dict), "variadic input dim invalid")
        _require_non_empty_string(dim, "field", "variadic input dim")
        dims_by_field[dim["field"]] = dim
    channel_field = matcher["channel_axis"]
    _require(channel_field in dims_by_field, "matcher.channel_axis missing from dims")

    fixed_fields = [
        dim["field"]
        for dim in dims
        if dim["field"] != channel_field and "values" in dim
    ]
    range_fields = [
        dim["field"]
        for dim in dims
        if dim["field"] != channel_field and "min" in dim and "max" in dim
    ]
    _require(len(fixed_fields) == 1, "expected one fixed non-channel dim")
    _require(len(range_fields) == 2, "expected two ranged non-channel dims")
    fixed_field = fixed_fields[0]
    fixed_dim = dims_by_field[fixed_field]
    _require(
        fixed_dim.get("values") == [bounds[fixed_field]],
        f"bounds.{fixed_field} must match shape_envelope dim values",
    )
    for field in range_fields:
        _validate_bound_pair(bounds[field], f"bounds.{field}")
        _require(
            bounds[field]["min"] == dims_by_field[field]["min"]
            and bounds[field]["max"] == dims_by_field[field]["max"],
            f"bounds.{field} must match shape_envelope dim bounds",
        )

    channel_dim = dims_by_field[channel_field]
    channel_dim_symbol = channel_dim.get("symbol", channel_field)
    channels = bounds[channel_field]
    _require(isinstance(channels, dict), f"bounds.{channel_field} must be an object")
    _require_keys(
        channels,
        ("min", "max_per_input", "multiple_of", "max_total"),
        f"bounds.{channel_field}",
    )
    _require(
        channels["min"] == channel_dim["min"]
        and channels["max_per_input"] == channel_dim["max"]
        and channels["multiple_of"] == channel_dim["multiple_of"],
        f"bounds.{channel_field} must match shape_envelope channel dim",
    )
    for key in channels:
        _validate_int(channels[key], f"bounds.{channel_field}.{key}")

    aggregate_bounds = envelope.get("aggregate_bounds", {})
    result_name = aggregate_matcher["result_name"]
    _sum_output_relationship(
        envelope,
        input_name,
        channel_dim_symbol,
        result_name,
        "ShapeEnvelope variadic tensor-list",
    )
    aggregate = aggregate_bounds.get(result_name)
    _require(isinstance(aggregate, dict), f"aggregate_bounds.{result_name} invalid")
    _require_keys(
        aggregate,
        ("input", "field", "min", "max", "multiple_of"),
        f"aggregate_bounds.{result_name}",
    )
    _require(
        aggregate["input"] == input_name and aggregate["field"] == channel_field,
        f"aggregate_bounds.{result_name} input/field mismatch",
    )
    _require(
        aggregate["min"] == aggregate_matcher["min"],
        "aggregate min must match matcher.aggregate.min",
    )
    _require(
        aggregate["max"] == _bound_path_value(bounds, aggregate_matcher["max_from_bounds"]),
        "aggregate max must match matcher.aggregate.max_from_bounds",
    )
    _require(
        aggregate["multiple_of"]
        == _bound_path_value(bounds, aggregate_matcher["multiple_of_from_bounds"]),
        "aggregate multiple_of must match matcher.aggregate.multiple_of_from_bounds",
    )

    for key in matcher["per_input_required_flags"]:
        bound_key = _required_flag_bound_key(key)
        _require(bound_key in bounds, f"bounds.{bound_key} missing")
        _validate_bool(bounds[bound_key], f"bounds.{bound_key}")
    return {
        "metadata": metadata,
        "bounds": bounds,
        "matcher": matcher,
        "channel_field": channel_field,
        "fixed_field": fixed_field,
        "range_fields": range_fields,
        "aggregate": aggregate,
    }


def generate_generic_variadic_tensor_list_shape_envelope_header(spec, source_name):
    validated = _validate_generic_variadic_tensor_list_shape_envelope_spec(spec)
    _require(validated is not None, "expected variadic tensor-list ShapeEnvelope")
    metadata = validated["metadata"]
    bounds = validated["bounds"]
    matcher = validated["matcher"]
    channel_field = validated["channel_field"]
    fixed_field = validated["fixed_field"]
    range_fields = validated["range_fields"]
    aggregate = validated["aggregate"]
    channels = bounds[channel_field]

    contract_prefix = _cpp_identifier_fragment(
        spec["contract_name"].removesuffix("Contract")
    )
    row_prefix = contract_prefix + _cpp_identifier_fragment(spec["family"])
    bounds_prefix = f"{contract_prefix}Rank{bounds['rank']}Dim{bounds['dim']}"
    func_prefix = _cpp_lower_identifier(spec["contract_name"].removesuffix("Contract"))
    singular_channel = _singular_field_name(channel_field)
    singular_channel_title = _cpp_identifier_fragment(singular_channel)
    channel_title = _cpp_identifier_fragment(channel_field)
    fixed_field_title = _cpp_identifier_fragment(fixed_field)
    range_titles = {field: _cpp_identifier_fragment(field) for field in range_fields}

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
        "#include <ATen/ArrayRef.h>",
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
        (
            f"constexpr const char* k{row_prefix}FamilyName = "
            f"{_cpp_string(spec['family'])};"
        ),
        (
            f"constexpr const char* k{row_prefix}TupleId = "
            f"{_cpp_string(spec['tuple_id'])};"
        ),
        (
            f"constexpr const char* k{row_prefix}WriterOp = "
            f"{_cpp_string(spec['writer_op'])};"
        ),
        (
            f"constexpr const char* k{row_prefix}RouteLabel = "
            f"{_cpp_string(spec['route_label'])};"
        ),
        "",
        (
            f"constexpr std::int64_t k{bounds_prefix}MinInputs = "
            f"{bounds['input_count']['min']};"
        ),
        (
            f"constexpr std::int64_t k{bounds_prefix}MaxInputs = "
            f"{bounds['input_count']['max']};"
        ),
        f"constexpr std::int64_t k{bounds_prefix}Rank = {bounds['rank']};",
        f"constexpr std::int64_t k{bounds_prefix}Dim = {bounds['dim']};",
        f"constexpr std::int64_t k{bounds_prefix}{fixed_field_title} = {bounds[fixed_field]};",
        (
            f"constexpr std::int64_t k{bounds_prefix}MinInput{channel_title} = "
            f"{channels['min']};"
        ),
        (
            f"constexpr std::int64_t k{bounds_prefix}MaxInput{channel_title} = "
            f"{channels['max_per_input']};"
        ),
        (
            f"constexpr std::int64_t k{bounds_prefix}{singular_channel_title}Multiple = "
            f"{channels['multiple_of']};"
        ),
        (
            f"constexpr std::int64_t k{bounds_prefix}MinTotal{channel_title} = "
            f"{aggregate['min']};"
        ),
        (
            f"constexpr std::int64_t k{bounds_prefix}MaxTotal{channel_title} = "
            f"{channels['max_total']};"
        ),
    ]
    for field in range_fields:
        title = range_titles[field]
        lines.extend(
            [
                (
                    f"constexpr std::int64_t k{bounds_prefix}Min{title} = "
                    f"{bounds[field]['min']};"
                ),
                (
                    f"constexpr std::int64_t k{bounds_prefix}Max{title} = "
                    f"{bounds[field]['max']};"
                ),
            ]
        )
    for flag in matcher["per_input_required_flags"]:
        bound_key = _required_flag_bound_key(flag)
        lines.append(
            f"constexpr bool k{bounds_prefix}Requires{_cpp_identifier_fragment(_required_flag_field(flag))} = "
            f"{_cpp_bool(bounds[bound_key])};"
        )

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
            "  at::ScalarType dtype;",
            "  std::int64_t rank;",
            "  std::int64_t dim;",
            "  std::int64_t min_inputs;",
            "  std::int64_t max_inputs;",
            f"  std::int64_t {fixed_field};",
            f"  std::int64_t min_input_{channel_field};",
            f"  std::int64_t max_input_{channel_field};",
            f"  std::int64_t {singular_channel}_multiple;",
            f"  std::int64_t min_total_{channel_field};",
            f"  std::int64_t max_total_{channel_field};",
        ]
    )
    for field in range_fields:
        lines.append(f"  std::int64_t min_{field};")
        lines.append(f"  std::int64_t max_{field};")
    for flag in matcher["per_input_required_flags"]:
        lines.append(f"  bool {_required_flag_bound_key(flag)};")

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
            f"        {SCALAR_TYPE_BY_DTYPE[bounds['dtype']]},",
            f"        k{bounds_prefix}Rank,",
            f"        k{bounds_prefix}Dim,",
            f"        k{bounds_prefix}MinInputs,",
            f"        k{bounds_prefix}MaxInputs,",
            f"        k{bounds_prefix}{fixed_field_title},",
            f"        k{bounds_prefix}MinInput{channel_title},",
            f"        k{bounds_prefix}MaxInput{channel_title},",
            f"        k{bounds_prefix}{singular_channel_title}Multiple,",
            f"        k{bounds_prefix}MinTotal{channel_title},",
            f"        k{bounds_prefix}MaxTotal{channel_title},",
        ]
    )
    for field in range_fields:
        title = range_titles[field]
        lines.append(f"        k{bounds_prefix}Min{title},")
        lines.append(f"        k{bounds_prefix}Max{title},")
    for index, flag in enumerate(matcher["per_input_required_flags"]):
        suffix = _cpp_identifier_fragment(_required_flag_field(flag))
        terminator = "};" if index == len(matcher["per_input_required_flags"]) - 1 else ","
        lines.append(f"        k{bounds_prefix}Requires{suffix}{terminator}")

    tensor_info = matcher["tensor_info"]
    aggregate_result = matcher["aggregate"]["result_name"]
    aggregate_sum_helper = (
        f"{func_prefix}_{_cpp_lower_identifier(aggregate_result)}_sum"
    )
    lines.extend(
        [
            "",
            f"constexpr bool {func_prefix}_input_count_in_bounds(",
            f"    const {row_prefix}Spec& spec,",
            "    const std::int64_t input_count) {",
            "  return input_count >= spec.min_inputs && input_count <= spec.max_inputs;",
            "}",
            "",
            f"inline bool {func_prefix}_reference_in_bounds(",
            f"    const {row_prefix}Spec& spec,",
            f"    const {tensor_info}& reference) {{",
            "  return (!spec.requires_vulkan || reference.is_vulkan) &&",
            "      reference.dtype == spec.dtype && reference.rank == spec.rank &&",
            f"      reference.{fixed_field} == spec.{fixed_field} &&",
            "      (!spec.requires_contiguous || reference.is_contiguous) &&",
            f"      reference.{range_fields[0]} >= spec.min_{range_fields[0]} &&",
            f"      reference.{range_fields[0]} <= spec.max_{range_fields[0]} &&",
            f"      reference.{range_fields[1]} >= spec.min_{range_fields[1]} && reference.{range_fields[1]} <= spec.max_{range_fields[1]};",
            "}",
            "",
            f"inline bool {func_prefix}_input_in_bounds(",
            f"    const {row_prefix}Spec& spec,",
            f"    const {tensor_info}& reference,",
            f"    const {tensor_info}& tensor) {{",
            "  return (!spec.requires_vulkan || tensor.is_vulkan) &&",
            "      tensor.dtype == reference.dtype && tensor.rank == reference.rank &&",
            f"      tensor.{fixed_field} == reference.{fixed_field} &&",
            f"      tensor.{range_fields[0]} == reference.{range_fields[0]} && tensor.{range_fields[1]} == reference.{range_fields[1]} &&",
            "      (!spec.requires_contiguous || tensor.is_contiguous) &&",
            "      (!spec.requires_buffer_storage || tensor.has_buffer_storage) &&",
            "      (!spec.requires_buffer_compute || tensor.supports_buffer_compute) &&",
            f"      tensor.{channel_field} >= spec.min_input_{channel_field} &&",
            f"      tensor.{channel_field} <= spec.max_input_{channel_field} &&",
            f"      tensor.{channel_field} % spec.{singular_channel}_multiple == 0;",
            "}",
            "",
            f"inline std::int64_t {aggregate_sum_helper}(",
            f"    const {row_prefix}Spec& spec,",
            f"    const ArrayRef<{tensor_info}> tensors) {{",
            "  static_cast<void>(spec);",
            f"  std::int64_t {aggregate_result} = 0;",
            f"  for (const {tensor_info}& tensor : tensors) {{",
            f"    {aggregate_result} += tensor.{aggregate['field']};",
            "  }",
            f"  return {aggregate_result};",
            "}",
            "",
            f"constexpr bool {func_prefix}_total_{channel_field}_in_bounds(",
            f"    const {row_prefix}Spec& spec,",
            f"    const std::int64_t total_{channel_field}) {{",
            f"  return total_{channel_field} >= spec.min_total_{channel_field} &&",
            f"      total_{channel_field} <= spec.max_total_{channel_field} &&",
            f"      total_{channel_field} % spec.{singular_channel}_multiple == 0;",
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
    for key in fields["min_range"]:
        _validate_int(bounds[key]["min"], f"bounds.{key}.min")
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
    broadcast_relationships = _broadcast_compatible_relationships(
        envelope, "ShapeEnvelope simple-bounds"
    )
    product_value_relationships = _product_value_relationships(
        envelope, "ShapeEnvelope simple-bounds"
    )
    scalar_equal_relationships = _scalar_equal_relationships(
        envelope, "ShapeEnvelope simple-bounds"
    )
    multi_field_equal_relationships = _multi_field_equal_relationships(
        envelope, "ShapeEnvelope simple-bounds"
    )

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

    for key in fields["min_range"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  std::int64_t min_{key};")
        initializer_lines.append(f"        k{row_prefix}Min{suffix},")
        range_params.append((f"const std::int64_t {key}", key))
        range_checks.append(f"{key} >= spec.min_{key}")
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}Min{suffix} = "
            f"{bounds[key]['min']};"
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

    relationship_helpers = []
    for relationship in broadcast_relationships:
        suffix = "BroadcastCompatible"
        field_struct_lines.append("  std::int64_t broadcast_compatible_max_rank;")
        initializer_lines.append(f"        k{row_prefix}{suffix}MaxRank,")
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}{suffix}MaxRank = "
            f"{relationship['max_rank']};"
        )
        relationship_helpers.extend(
            [
                f"inline bool {role_func_prefix}_broadcast_compatible(",
                f"    const {row_prefix}Spec& spec,",
                "    const IntArrayRef left_sizes,",
                "    const IntArrayRef right_sizes) {",
                (
                    "  const std::int64_t left_rank = "
                    "static_cast<std::int64_t>(left_sizes.size());"
                ),
                (
                    "  const std::int64_t right_rank = "
                    "static_cast<std::int64_t>(right_sizes.size());"
                ),
                "  if (left_rank > spec.broadcast_compatible_max_rank ||",
                "      right_rank > spec.broadcast_compatible_max_rank) {",
                "    return false;",
                "  }",
                (
                    "  const std::int64_t max_rank = "
                    "left_rank > right_rank ? left_rank : right_rank;"
                ),
                "  for (std::int64_t axis = 0; axis < max_rank; ++axis) {",
                "    const std::int64_t left_axis = left_rank - 1 - axis;",
                "    const std::int64_t right_axis = right_rank - 1 - axis;",
                "    const std::int64_t left_dim = left_axis >= 0 ? left_sizes[left_axis] : 1;",
                "    const std::int64_t right_dim = right_axis >= 0 ? right_sizes[right_axis] : 1;",
                "    if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {",
                "      return false;",
                "    }",
                "  }",
                "  return true;",
                "}",
                "",
            ]
        )

    product_helpers = []
    for relationship in product_value_relationships:
        input_name = _cpp_lower_identifier(relationship["input"])
        result_name = _cpp_lower_identifier(relationship["result"])
        product_helpers.extend(
            [
                f"inline std::int64_t {role_func_prefix}_{input_name}_{result_name}(",
                f"    const {row_prefix}Spec& spec,",
                f"    const IntArrayRef {input_name}_sizes) {{",
                "  static_cast<void>(spec);",
                "  std::int64_t product = 1;",
                f"  for (const std::int64_t size : {input_name}_sizes) {{",
                "    product *= size;",
                "  }",
                "  return product;",
                "}",
                "",
            ]
        )

    scalar_equal_helpers = _scalar_equal_helper_lines(
        role_func_prefix, scalar_equal_relationships
    )
    multi_field_equal_helpers = _multi_field_equal_helper_lines(
        role_func_prefix, multi_field_equal_relationships
    )

    initializer_lines[-1] = initializer_lines[-1].rstrip(",") + "};"
    option_signature = []
    for param, _ in option_params:
        option_signature.append(f"    {param},")
    option_signature[-1] = option_signature[-1].rstrip(",") + ") {"
    option_body = " &&\n      ".join(option_checks)

    if range_params:
        range_spec_signature = [f"    const {row_prefix}Spec& spec,"]
        range_signature = []
        for param, _ in range_params:
            range_signature.append(f"    {param},")
        range_signature[-1] = range_signature[-1].rstrip(",") + ") {"
        range_body = " &&\n      ".join(range_checks)
    else:
        range_spec_signature = [f"    const {row_prefix}Spec&) {{"]
        range_signature = []
        range_body = "true"

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
    ]
    if relationship_helpers or product_helpers:
        lines.append("#include <ATen/ArrayRef.h>")
    lines.extend(
        [
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
        f"constexpr const char* k{row_prefix}ContractName = {_cpp_string(spec['contract_name'])};",
        f"constexpr const char* k{row_prefix}FamilyName = {_cpp_string(spec['family'])};",
        f"constexpr const char* k{row_prefix}TupleId = {_cpp_string(spec['tuple_id'])};",
        f"constexpr const char* k{row_prefix}WriterOp = {_cpp_string(spec['writer_op'])};",
        f"constexpr const char* k{row_prefix}RouteLabel = {_cpp_string(spec['route_label'])};",
        "",
        ]
    )
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
            f"        k{row_prefix}ContractName,",
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
    lines.extend(relationship_helpers)
    lines.extend(product_helpers)
    lines.extend(scalar_equal_helpers)
    lines.extend(multi_field_equal_helpers)
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
        ]
    )
    lines.extend(range_spec_signature)
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


def _validate_generic_shape_layout_simple_bounds_shape_envelope_spec(spec):
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
        "ShapeEnvelope shape/layout simple-bounds contract spec",
    )
    for key in ("contract_name", "family", "tuple_id", "writer_op", "route_label"):
        _require_non_empty_string(
            spec, key, "ShapeEnvelope shape/layout simple-bounds contract spec"
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
    fields = _shape_layout_simple_bounds_shape_envelope_fields(bounds)
    _require(fields is not None, "unsupported shape/layout ShapeEnvelope bounds")

    for key in fields["int"]:
        _validate_int(bounds[key], f"bounds.{key}")
    for key in fields["multiple_of"]:
        _validate_int(bounds[key], f"bounds.{key}")
        _require(bounds[key] > 0, f"bounds.{key} must be positive")
    for key in fields["range"]:
        _validate_bound_pair(bounds[key], f"bounds.{key}")
    for key in fields["bool"]:
        _validate_bool(bounds[key], f"bounds.{key}")
    for key in fields["string"]:
        _require_non_empty_string(bounds, key, "ShapeEnvelope bounds")
    return fields


def generate_generic_shape_layout_simple_bounds_shape_envelope_header(
    spec, source_name
):
    fields = _validate_generic_shape_layout_simple_bounds_shape_envelope_spec(spec)
    envelope = spec["shape_envelope"]
    metadata = envelope["metadata"]
    bounds = envelope["bounds"]
    contract_prefix = _cpp_identifier_fragment(
        spec["contract_name"].removesuffix("Contract")
    )
    row_prefix = contract_prefix + _cpp_identifier_fragment(spec["family"])
    role_func_prefix = _cpp_lower_identifier(envelope["role"])
    product_equal_relationship = _product_equal_relationship(
        envelope, "ShapeEnvelope shape/layout simple-bounds"
    )

    field_struct_lines = []
    initializer_lines = []
    constant_lines = []
    helper_lines = []

    for key in fields["range"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  std::int64_t min_{key};")
        field_struct_lines.append(f"  std::int64_t max_{key};")
        initializer_lines.append(f"        k{row_prefix}Min{suffix},")
        initializer_lines.append(f"        k{row_prefix}Max{suffix},")
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}Min{suffix} = "
            f"{bounds[key]['min']};"
        )
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}Max{suffix} = "
            f"{bounds[key]['max']};"
        )
        helper_lines.extend(
            [
                f"constexpr bool {role_func_prefix}_{key}_in_bounds(",
                f"    const {row_prefix}Spec& spec,",
                f"    const std::int64_t {key}) {{",
                f"  return {key} >= spec.min_{key} && {key} <= spec.max_{key};",
                "}",
                "",
            ]
        )

    for key in fields["int"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  std::int64_t {key};")
        initializer_lines.append(f"        k{row_prefix}{suffix},")
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}{suffix} = {bounds[key]};"
        )
        helper_lines.extend(
            [
                f"constexpr bool {role_func_prefix}_{key}_matches(",
                f"    const {row_prefix}Spec& spec,",
                f"    const std::int64_t {key}) {{",
                f"  return {key} == spec.{key};",
                "}",
                "",
            ]
        )

    for key in fields["multiple_of"]:
        suffix = _cpp_identifier_fragment(key)
        value_name = key.removesuffix("_multiple_of")
        field_struct_lines.append(f"  std::int64_t {key};")
        initializer_lines.append(f"        k{row_prefix}{suffix},")
        constant_lines.append(
            f"constexpr std::int64_t k{row_prefix}{suffix} = {bounds[key]};"
        )
        helper_lines.extend(
            [
                f"constexpr bool {role_func_prefix}_{value_name}_multiple_matches(",
                f"    const {row_prefix}Spec& spec,",
                f"    const bool has_{value_name},",
                f"    const std::int64_t {value_name}) {{",
                f"  return !has_{value_name} || {value_name} % spec.{key} == 0;",
                "}",
                "",
            ]
        )

    bool_check_lines = []
    bool_signature_lines = []
    for key in fields["bool"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  bool {key};")
        initializer_lines.append(f"        k{row_prefix}{suffix},")
        constant_lines.append(
            f"constexpr bool k{row_prefix}{suffix} = {_cpp_bool(bounds[key])};"
        )
        bool_signature_lines.append(f"    const bool {key},")
        bool_check_lines.append(f"{key} == spec.{key}")
    if bool_signature_lines:
        bool_signature_lines[-1] = bool_signature_lines[-1].rstrip(",") + ") {"
        helper_lines.extend(
            [
                f"constexpr bool {role_func_prefix}_policies_match(",
                f"    const {row_prefix}Spec& spec,",
            ]
        )
        helper_lines.extend(bool_signature_lines)
        helper_lines.extend(
            [
                f"  return {' && '.join(bool_check_lines)};",
                "}",
                "",
            ]
        )

    for key in fields["string"]:
        suffix = _cpp_identifier_fragment(key)
        field_struct_lines.append(f"  const char* {key};")
        initializer_lines.append(f"        k{row_prefix}{suffix},")
        constant_lines.append(
            f"constexpr const char* k{row_prefix}{suffix} = "
            f"{_cpp_string(bounds[key])};"
        )

    if product_equal_relationship:
        left_name = _cpp_lower_identifier(product_equal_relationship["left_input"])
        right_name = _cpp_lower_identifier(product_equal_relationship["right_input"])
        helper_lines.extend(
            [
                f"inline bool {role_func_prefix}_product_equal(",
                f"    const {row_prefix}Spec& spec,",
                f"    const IntArrayRef {left_name}_sizes,",
                f"    const IntArrayRef {right_name}_sizes) {{",
                f"  if (!spec.product_equal) {{",
                "    return true;",
                "  }",
                "  auto product_of_sizes = [](const IntArrayRef sizes) {",
                "    std::int64_t product = 1;",
                "    for (const std::int64_t size : sizes) {",
                "      product *= size;",
                "    }",
                "    return product;",
                "  };",
                (
                    f"  return product_of_sizes({left_name}_sizes) == "
                    f"product_of_sizes({right_name}_sizes);"
                ),
                "}",
                "",
            ]
        )

    initializer_lines[-1] = initializer_lines[-1].rstrip(",") + "};"

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
    ]
    if product_equal_relationship:
        lines.append("#include <ATen/ArrayRef.h>")
    lines.extend(
        [
        "#include <cstdint>",
        "",
        "namespace at {",
        "namespace native {",
        "namespace vulkan {",
        "namespace ops {",
        "namespace utils {",
        "namespace generated {",
        "",
        f"constexpr const char* k{row_prefix}ContractName = {_cpp_string(spec['contract_name'])};",
        f"constexpr const char* k{row_prefix}FamilyName = {_cpp_string(spec['family'])};",
        f"constexpr const char* k{row_prefix}TupleId = {_cpp_string(spec['tuple_id'])};",
        f"constexpr const char* k{row_prefix}WriterOp = {_cpp_string(spec['writer_op'])};",
        f"constexpr const char* k{row_prefix}RouteLabel = {_cpp_string(spec['route_label'])};",
        "",
        ]
    )
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
            f"        k{row_prefix}ContractName,",
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
    lines.extend([""])
    lines.extend(helper_lines)
    lines.extend(
        [
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


def _sparse_rowsets(spec):
    envelope = spec.get("shape_envelope")
    if not isinstance(envelope, dict):
        return []
    rowsets = envelope.get("sparse_rowsets", [])
    if rowsets is None:
        return []
    return rowsets


def _factorized_groups(spec):
    envelope = spec.get("shape_envelope")
    if not isinstance(envelope, dict):
        return []
    groups = envelope.get("factorized_groups", [])
    if groups is None:
        return []
    return groups


def _validate_sparse_row_value(value, context):
    _require(
        isinstance(value, (str, int, bool)) and not (isinstance(value, str) and value == ""),
        f"{context} must be a non-empty string, integer, or boolean",
    )


def _sparse_row_field_types(rows, fields, context):
    field_types = {}
    for field in fields:
        current_type = None
        for row_index, row in enumerate(rows):
            value = row[field]
            value_type = bool if isinstance(value, bool) else type(value)
            if value_type not in (str, int, bool):
                raise RuntimeError(
                    f"{context}.rows[{row_index}].{field} has unsupported type"
                )
            if current_type is None:
                current_type = value_type
            _require(
                value_type is current_type,
                f"{context}.{field} must use one scalar type across all rows",
            )
        field_types[field] = current_type
    return field_types


def _cpp_type_for_row_field(field_type):
    if field_type is str:
        return "const char*"
    if field_type is bool:
        return "bool"
    if field_type is int:
        return "std::int64_t"
    raise RuntimeError(f"unsupported sparse row field type {field_type!r}")


def _cpp_row_value(value):
    if isinstance(value, str):
        return _cpp_string(value)
    if isinstance(value, bool):
        return _cpp_bool(value)
    if isinstance(value, int):
        return str(value)
    raise RuntimeError(f"unsupported sparse row value {value!r}")


def _cpp_row_field_compare(field, field_type):
    if field_type is str:
        return f"std::string_view(row.{field}) == {field}"
    return f"row.{field} == {field}"


ROW_MATCH_ARGUMENT_TYPES = {
    "bool": bool,
    "int64": int,
    "string": str,
}


def _cpp_type_for_row_match_argument(argument_type):
    if argument_type == "bool":
        return "bool"
    if argument_type == "int64":
        return "std::int64_t"
    if argument_type == "string":
        return "const char*"
    raise RuntimeError(f"unsupported row_match argument type {argument_type!r}")


def _validate_sparse_row_match(row_match, rowset, field_types, context):
    if row_match is None:
        return None
    _require(isinstance(row_match, dict), f"{context} must be an object")
    _require_keys(row_match, ("arguments",), context)
    arguments = row_match["arguments"]
    _require(isinstance(arguments, list) and arguments, f"{context}.arguments invalid")

    argument_names = set()
    argument_types = {}
    field_names = set(rowset["fields"])
    for index, argument in enumerate(arguments):
        argument_context = f"{context}.arguments[{index}]"
        _require(isinstance(argument, dict), f"{argument_context} must be an object")
        _require_keys(argument, ("name", "type"), argument_context)
        _require_non_empty_string(argument, "name", argument_context)
        name = argument["name"]
        _require(name not in argument_names, f"{argument_context}.name duplicate")
        argument_names.add(name)

        argument_type = argument["type"]
        _require(
            argument_type in ROW_MATCH_ARGUMENT_TYPES,
            f"{argument_context}.type unsupported",
        )
        argument_types[name] = argument_type

        has_field = "field" in argument
        has_range = "min_field" in argument or "max_field" in argument
        _require(
            has_field != has_range,
            f"{argument_context} must use either field or min_field/max_field",
        )
        expected_type = ROW_MATCH_ARGUMENT_TYPES[argument_type]
        if has_field:
            _require_non_empty_string(argument, "field", argument_context)
            field = argument["field"]
            _require(field in field_names, f"{argument_context}.field unknown")
            _require(
                field_types[field] is expected_type,
                f"{argument_context}.field type mismatch",
            )
            continue

        _require_keys(argument, ("min_field", "max_field"), argument_context)
        _require(
            argument_type == "int64",
            f"{argument_context} range arguments must be int64",
        )
        for key in ("min_field", "max_field"):
            _require_non_empty_string(argument, key, argument_context)
            field = argument[key]
            _require(field in field_names, f"{argument_context}.{key} unknown")
            _require(
                field_types[field] is int,
                f"{argument_context}.{key} must reference integer row fields",
            )

    conditional_equal = row_match.get("conditional_equal", [])
    _require(
        isinstance(conditional_equal, list),
        f"{context}.conditional_equal must be a list",
    )
    for index, relationship in enumerate(conditional_equal):
        relationship_context = f"{context}.conditional_equal[{index}]"
        _require(
            isinstance(relationship, dict),
            f"{relationship_context} must be an object",
        )
        _require_keys(
            relationship,
            ("flag_field", "left", "right"),
            relationship_context,
        )
        for key in ("flag_field", "left", "right"):
            _require_non_empty_string(relationship, key, relationship_context)
        flag_field = relationship["flag_field"]
        _require(flag_field in field_names, f"{relationship_context}.flag_field unknown")
        _require(
            field_types[flag_field] is bool,
            f"{relationship_context}.flag_field must reference a boolean row field",
        )
        left = relationship["left"]
        right = relationship["right"]
        _require(left in argument_names, f"{relationship_context}.left unknown")
        _require(right in argument_names, f"{relationship_context}.right unknown")
        _require(
            argument_types[left] == argument_types[right],
            f"{relationship_context} argument type mismatch",
        )
        _require(
            argument_types[left] != "string",
            f"{relationship_context} string equality is not supported",
        )
    return row_match


def _validate_factorized_projection_groups(groups, context):
    _require(isinstance(groups, list), f"{context} must be a list")
    validated = []
    names = set()
    for group_index, group in enumerate(groups):
        group_context = f"{context}[{group_index}]"
        _require(isinstance(group, dict), f"{group_context} must be an object")
        _require_keys(
            group,
            (
                "name",
                "family",
                "tuple_id",
                "metadata",
                "channel_pairs",
                "spatial_pairs",
                "cardinality",
                "validated_corpus_count",
                "extrapolated_shape_count",
                "expansion_ratio",
            ),
            group_context,
        )
        _require_non_empty_string(group, "name", group_context)
        _require(group["name"] not in names, f"{group_context}.name duplicate")
        names.add(group["name"])
        for key in ("family", "tuple_id"):
            _require_non_empty_string(group, key, group_context)
        _validate_contract_metadata(group["metadata"], f"{group_context}.metadata")

        channel_pairs = group["channel_pairs"]
        spatial_pairs = group["spatial_pairs"]
        _require(
            isinstance(channel_pairs, list) and channel_pairs,
            f"{group_context}.channel_pairs invalid",
        )
        _require(
            isinstance(spatial_pairs, list) and spatial_pairs,
            f"{group_context}.spatial_pairs invalid",
        )
        channel_keys = set()
        for index, pair in enumerate(channel_pairs):
            pair_context = f"{group_context}.channel_pairs[{index}]"
            _require(isinstance(pair, dict), f"{pair_context} must be an object")
            _require_keys(pair, ("input_c", "output_c", "proof_class"), pair_context)
            for key in ("input_c", "output_c"):
                _validate_sparse_row_value(pair[key], f"{pair_context}.{key}")
                _require(isinstance(pair[key], int), f"{pair_context}.{key} must be int")
            _require_non_empty_string(pair, "proof_class", pair_context)
            key = (pair["input_c"], pair["output_c"])
            _require(key not in channel_keys, f"{pair_context} duplicate")
            channel_keys.add(key)

        spatial_keys = set()
        for index, pair in enumerate(spatial_pairs):
            pair_context = f"{group_context}.spatial_pairs[{index}]"
            _require(isinstance(pair, dict), f"{pair_context} must be an object")
            _require_keys(pair, ("input_h", "input_w"), pair_context)
            for key in ("input_h", "input_w"):
                _validate_sparse_row_value(pair[key], f"{pair_context}.{key}")
                _require(isinstance(pair[key], int), f"{pair_context}.{key} must be int")
            key = (pair["input_h"], pair["input_w"])
            _require(key not in spatial_keys, f"{pair_context} duplicate")
            spatial_keys.add(key)

        cardinality = group["cardinality"]
        _require(isinstance(cardinality, int), f"{group_context}.cardinality invalid")
        _require(
            cardinality == len(channel_pairs) * len(spatial_pairs),
            f"{group_context}.cardinality mismatch",
        )
        for key in ("validated_corpus_count", "extrapolated_shape_count"):
            _require(isinstance(group[key], int), f"{group_context}.{key} invalid")
        _require(
            group["validated_corpus_count"] + group["extrapolated_shape_count"]
            == cardinality,
            f"{group_context} proof counts must sum to cardinality",
        )
        _require(
            isinstance(group["expansion_ratio"], (int, float)),
            f"{group_context}.expansion_ratio invalid",
        )
        validated.append(group)
    return validated


def _cpp_row_match_argument_condition(argument):
    name = argument["name"]
    if "field" in argument:
        field = argument["field"]
        if argument["type"] == "string":
            return f"std::string_view(row.{field}) == {name}"
        return f"row.{field} == {name}"
    return (
        f"({name} >= row.{argument['min_field']} && "
        f"{name} <= row.{argument['max_field']})"
    )


def _sparse_row_match_helper_lines(rowset_prefix, func_prefix, rowset_func, row_match):
    if row_match is None:
        return []

    signature_params = [f"    const {rowset_prefix}Row& row"]
    for argument in row_match["arguments"]:
        cpp_type = _cpp_type_for_row_match_argument(argument["type"])
        param_type = cpp_type if cpp_type == "const char*" else f"const {cpp_type}"
        signature_params.append(f"    {param_type} {argument['name']}")

    lines = [
        f"inline bool {func_prefix}_{rowset_func}_row_matches(",
    ]
    for index, parameter in enumerate(signature_params):
        suffix = "," if index + 1 < len(signature_params) else ") {"
        lines.append(parameter + suffix)

    conditions = [
        _cpp_row_match_argument_condition(argument)
        for argument in row_match["arguments"]
    ]
    for relationship in row_match.get("conditional_equal", []):
        conditions.append(
            f"(!row.{relationship['flag_field']} || "
            f"{relationship['left']} == {relationship['right']})"
        )

    for index, condition in enumerate(conditions):
        if index == 0:
            prefix = "  return "
        else:
            prefix = "      "
        suffix = " &&" if index + 1 < len(conditions) else ";"
        lines.append(prefix + condition + suffix)
    lines.extend(["}", ""])
    return lines


def _factorized_group_helper_lines(contract_prefix, spec_prefix, func_prefix, groups):
    lines = []
    for group in groups:
        group_prefix = contract_prefix + _cpp_identifier_fragment(group["name"])
        group_func = func_prefix + "_" + _cpp_lower_identifier(group["name"])
        metadata = group["metadata"]
        channel_pairs = group["channel_pairs"]
        spatial_pairs = group["spatial_pairs"]

        lines.extend(
            [
                f"constexpr const char* k{group_prefix}FamilyName = {_cpp_string(group['family'])};",
                f"constexpr const char* k{group_prefix}TupleId = {_cpp_string(group['tuple_id'])};",
                f"constexpr const char* k{group_prefix}EvidenceId = {_cpp_string(metadata['evidence_id'])};",
                f"constexpr const char* k{group_prefix}GuardId = {_cpp_string(metadata['guard_id'])};",
                f"constexpr const char* k{group_prefix}FallbackPolicy = {_cpp_string(metadata['fallback_policy'])};",
                (
                    f"constexpr const char* k{group_prefix}MaterializationPolicy = "
                    f"{_cpp_string(metadata['materialization_policy'])};"
                ),
                (
                    f"constexpr std::int64_t k{group_prefix}Cardinality = "
                    f"{group['cardinality']};"
                ),
                (
                    f"constexpr std::int64_t k{group_prefix}ValidatedCorpusCount = "
                    f"{group['validated_corpus_count']};"
                ),
                (
                    f"constexpr std::int64_t k{group_prefix}ExtrapolatedShapeCount = "
                    f"{group['extrapolated_shape_count']};"
                ),
                (
                    f"constexpr double k{group_prefix}ExpansionRatio = "
                    f"{group['expansion_ratio']};"
                ),
                "",
                f"struct {group_prefix}ChannelPair final {{",
                "  std::int64_t input_c;",
                "  std::int64_t output_c;",
                "  const char* proof_class;",
                "};",
                "",
                f"struct {group_prefix}SpatialPair final {{",
                "  std::int64_t input_h;",
                "  std::int64_t input_w;",
                "};",
                "",
                (
                    f"constexpr ExecutionContractMetadata "
                    f"k{group_prefix}Metadata = {{"
                    f"k{spec_prefix}ContractName, "
                    f"k{group_prefix}FamilyName, "
                    f"k{group_prefix}TupleId, "
                    f"k{group_prefix}EvidenceId, "
                    f"k{group_prefix}GuardId, "
                    f"k{group_prefix}FallbackPolicy, "
                    f"k{group_prefix}MaterializationPolicy}};"
                ),
                "",
                f"constexpr {group_prefix}ChannelPair k{group_prefix}ChannelPairs[] = {{",
            ]
        )
        for index, pair in enumerate(channel_pairs):
            suffix = "," if index + 1 < len(channel_pairs) else ""
            lines.append(
                "    {"
                f"{pair['input_c']}, {pair['output_c']}, "
                f"{_cpp_string(pair['proof_class'])}}}"
                f"{suffix}"
            )
        lines.extend(
            [
                "};",
                "",
                f"constexpr {group_prefix}SpatialPair k{group_prefix}SpatialPairs[] = {{",
            ]
        )
        for index, pair in enumerate(spatial_pairs):
            suffix = "," if index + 1 < len(spatial_pairs) else ""
            lines.append(
                f"    {{{pair['input_h']}, {pair['input_w']}}}{suffix}"
            )
        lines.extend(
            [
                "};",
                "",
                f"inline bool {group_func}_channel_pair_matches(",
                "    const std::int64_t input_c,",
                "    const std::int64_t output_c) {",
                f"  for (const auto& pair : k{group_prefix}ChannelPairs) {{",
                "    if (pair.input_c == input_c && pair.output_c == output_c) {",
                "      return true;",
                "    }",
                "  }",
                "  return false;",
                "}",
                "",
                f"inline bool {group_func}_spatial_pair_matches(",
                "    const std::int64_t input_h,",
                "    const std::int64_t input_w) {",
                f"  for (const auto& pair : k{group_prefix}SpatialPairs) {{",
                "    if (pair.input_h == input_h && pair.input_w == input_w) {",
                "      return true;",
                "    }",
                "  }",
                "  return false;",
                "}",
                "",
                f"inline bool {group_func}_matches(",
                "    const std::int64_t input_c,",
                "    const std::int64_t input_h,",
                "    const std::int64_t input_w,",
                "    const std::int64_t output_c) {",
                f"  return {group_func}_channel_pair_matches(input_c, output_c) &&",
                f"      {group_func}_spatial_pair_matches(input_h, input_w);",
                "}",
                "",
                f"inline const char* {group_func}_family_name() {{",
                f"  return k{group_prefix}FamilyName;",
                "}",
                "",
                f"inline const char* {group_func}_tuple_id() {{",
                f"  return k{group_prefix}TupleId;",
                "}",
                "",
                (
                    f"inline const ExecutionContractMetadata* "
                    f"{group_func}_metadata() {{"
                ),
                f"  return &k{group_prefix}Metadata;",
                "}",
                "",
            ]
        )
    return lines


def _validate_generic_sparse_rowset_shape_envelope_spec(spec):
    rowsets = _sparse_rowsets(spec)
    if not rowsets:
        return None
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
        "ShapeEnvelope sparse-rowset contract spec",
    )
    for key in ("contract_name", "family", "tuple_id", "writer_op", "route_label"):
        _require_non_empty_string(spec, key, "ShapeEnvelope sparse-rowset contract spec")

    envelope = spec["shape_envelope"]
    _require(envelope.get("version") == 1, "shape_envelope.version must be 1")
    _require_non_empty_string(envelope, "role", "shape_envelope")
    metadata = envelope.get("metadata")
    _validate_contract_metadata(metadata, "ShapeEnvelope metadata")
    _require(spec["metadata"] == metadata, "metadata must match shape_envelope")

    bounds = envelope.get("bounds")
    _require(spec["bounds"] == bounds, "bounds must match shape_envelope")
    _require(isinstance(bounds, dict), "ShapeEnvelope bounds must be an object")

    _require(len(rowsets) == 1, "sparse-rowset generator v0 supports one rowset")
    rowset = rowsets[0]
    _require(isinstance(rowset, dict), "sparse_rowsets[0] must be an object")
    _require_keys(
        rowset,
        ("name", "fields", "identity_fields", "lookup_fields", "label_field", "rows"),
        "sparse_rowsets[0]",
    )
    _require_non_empty_string(rowset, "name", "sparse_rowsets[0]")
    for key in ("fields", "identity_fields", "lookup_fields", "rows"):
        _require(isinstance(rowset[key], list) and rowset[key], f"sparse_rowsets[0].{key} invalid")
    fields = rowset["fields"]
    identity_fields = rowset["identity_fields"]
    lookup_fields = rowset["lookup_fields"]
    for key, values in (
        ("fields", fields),
        ("identity_fields", identity_fields),
        ("lookup_fields", lookup_fields),
    ):
        for index, value in enumerate(values):
            _require(
                isinstance(value, str) and value,
                f"sparse_rowsets[0].{key}[{index}] invalid",
            )
        _require(
            len(values) == len(set(values)),
            f"sparse_rowsets[0].{key} must be unique",
        )
    field_names = set(fields)
    for key, values in (("identity_fields", identity_fields), ("lookup_fields", lookup_fields)):
        unknown = sorted(set(values) - field_names)
        _require(not unknown, f"sparse_rowsets[0].{key} unknown fields {unknown}")
    label_field = rowset["label_field"]
    _require(
        isinstance(label_field, str) and label_field in field_names,
        "sparse_rowsets[0].label_field invalid",
    )

    rows = rowset["rows"]
    identity_keys = set()
    lookup_keys = set()
    labels = set()
    for row_index, row in enumerate(rows):
        _require(isinstance(row, dict), f"sparse_rowsets[0].rows[{row_index}] invalid")
        missing = sorted(field_names - set(row))
        extra = sorted(set(row) - field_names)
        _require(
            not missing and not extra,
            f"sparse_rowsets[0].rows[{row_index}] field mismatch "
            f"missing={missing} extra={extra}",
        )
        for field in fields:
            _validate_sparse_row_value(row[field], f"sparse_rowsets[0].rows[{row_index}].{field}")
        identity_key = tuple(row[field] for field in identity_fields)
        lookup_key = tuple(row[field] for field in lookup_fields)
        label = row[label_field]
        _require(
            identity_key not in identity_keys,
            f"sparse_rowsets[0].rows[{row_index}] duplicate identity",
        )
        _require(
            lookup_key not in lookup_keys,
            f"sparse_rowsets[0].rows[{row_index}] duplicate lookup",
        )
        _require(
            label not in labels,
            f"sparse_rowsets[0].rows[{row_index}] duplicate label",
        )
        identity_keys.add(identity_key)
        lookup_keys.add(lookup_key)
        labels.add(label)

    field_types = _sparse_row_field_types(rows, fields, "sparse_rowsets[0]")
    row_match = _validate_sparse_row_match(
        rowset.get("row_match"),
        rowset,
        field_types,
        "sparse_rowsets[0].row_match",
    )
    factorized_groups = _validate_factorized_projection_groups(
        _factorized_groups(spec),
        "shape_envelope.factorized_groups",
    )

    return {
        "metadata": metadata,
        "bounds": bounds,
        "rowset": rowset,
        "field_types": field_types,
        "row_match": row_match,
        "factorized_groups": factorized_groups,
    }


def generate_generic_sparse_rowset_shape_envelope_header(spec, source_name):
    validated = _validate_generic_sparse_rowset_shape_envelope_spec(spec)
    _require(validated is not None, "expected sparse-rowset ShapeEnvelope")
    envelope = spec["shape_envelope"]
    metadata = validated["metadata"]
    rowset = validated["rowset"]
    field_types = validated["field_types"]
    row_match = validated["row_match"]
    factorized_groups = validated["factorized_groups"]
    rows = rowset["rows"]
    fields = rowset["fields"]
    lookup_fields = rowset["lookup_fields"]
    label_field = rowset["label_field"]

    contract_prefix = _cpp_identifier_fragment(
        spec["contract_name"].removesuffix("Contract")
    )
    spec_prefix = contract_prefix + _cpp_identifier_fragment(spec["family"])
    rowset_prefix = contract_prefix + _cpp_identifier_fragment(rowset["name"])
    func_prefix = _cpp_lower_identifier(spec["contract_name"].removesuffix("Contract"))
    role_func_prefix = _cpp_lower_identifier(envelope["role"])
    rowset_func = _cpp_lower_identifier(rowset["name"])
    scalar_equal_relationships = _scalar_equal_relationships(
        envelope, "ShapeEnvelope sparse-rowset"
    )

    row_field_lines = [
        f"  {_cpp_type_for_row_field(field_types[field])} {field};"
        for field in fields
    ]
    row_field_lines.append("  ExecutionContractMetadata metadata;")

    row_initializers = []
    for row in rows:
        values = [f"        {_cpp_row_value(row[field])}," for field in fields]
        materialization_policy = (
            _cpp_row_value(row["materialization_policy"])
            if "materialization_policy" in row
            else f"k{spec_prefix}MaterializationPolicy"
        )
        values.append(
            "        ExecutionContractMetadata{"
            f"k{spec_prefix}ContractName, "
            f"{_cpp_row_value(row['family'])}, "
            f"{_cpp_row_value(row[label_field])}, "
            f"k{spec_prefix}EvidenceId, "
            f"k{spec_prefix}GuardId, "
            f"k{spec_prefix}FallbackPolicy, "
            f"{materialization_policy}}}"
        )
        row_initializers.extend(["    {", *values, "    },"])
    row_initializers[-1] = row_initializers[-1].rstrip(",")

    lookup_params = []
    lookup_checks = []
    for field in lookup_fields:
        field_type = _cpp_type_for_row_field(field_types[field])
        param_type = field_type if field_type == "const char*" else f"const {field_type}"
        lookup_params.append(f"    {param_type} {field},")
        lookup_checks.append(_cpp_row_field_compare(field, field_types[field]))
    lookup_params[-1] = lookup_params[-1].rstrip(",") + ") {"
    lookup_condition = " &&\n        ".join(lookup_checks)
    row_match_lines = _sparse_row_match_helper_lines(
        rowset_prefix,
        func_prefix,
        rowset_func,
        row_match,
    )
    factorized_group_lines = _factorized_group_helper_lines(
        contract_prefix,
        spec_prefix,
        func_prefix,
        factorized_groups,
    )
    scalar_equal_helpers = _scalar_equal_helper_lines(
        role_func_prefix, scalar_equal_relationships
    )

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
        "#include <ATen/native/vulkan/planning/ExecutionContracts.h>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "namespace at {",
        "namespace native {",
        "namespace vulkan {",
        "namespace ops {",
        "namespace utils {",
        "namespace generated {",
        "",
        f"constexpr const char* k{spec_prefix}ContractName = {_cpp_string(spec['contract_name'])};",
        f"constexpr const char* k{spec_prefix}FamilyName = {_cpp_string(spec['family'])};",
        f"constexpr const char* k{spec_prefix}TupleId = {_cpp_string(spec['tuple_id'])};",
        f"constexpr const char* k{spec_prefix}WriterOp = {_cpp_string(spec['writer_op'])};",
        f"constexpr const char* k{spec_prefix}RouteLabel = {_cpp_string(spec['route_label'])};",
        f"constexpr const char* k{spec_prefix}EvidenceId = {_cpp_string(metadata['evidence_id'])};",
        f"constexpr const char* k{spec_prefix}GuardId = {_cpp_string(metadata['guard_id'])};",
        f"constexpr const char* k{spec_prefix}FallbackPolicy = {_cpp_string(metadata['fallback_policy'])};",
        (
            f"constexpr const char* k{spec_prefix}MaterializationPolicy = "
            f"{_cpp_string(metadata['materialization_policy'])};"
        ),
        f"constexpr std::int64_t k{rowset_prefix}RowCount = {len(rows)};",
        "",
        f"struct {rowset_prefix}Row final {{",
    ]
    lines.extend(row_field_lines)
    lines.extend(
        [
            "};",
            "",
            f"struct {spec_prefix}Spec final {{",
            "  const char* contract_name;",
            "  const char* family_name;",
            "  const char* tuple_id;",
            "  const char* writer_op;",
            "  const char* route_label;",
            "  const char* evidence_id;",
            "  const char* guard_id;",
            "  const char* fallback_policy;",
            "  const char* materialization_policy;",
            "  std::int64_t row_count;",
            "};",
            "",
            f"constexpr {spec_prefix}Spec",
            f"    k{spec_prefix}Spec = {{",
            f"        k{spec_prefix}ContractName,",
            f"        k{spec_prefix}FamilyName,",
            f"        k{spec_prefix}TupleId,",
            f"        k{spec_prefix}WriterOp,",
            f"        k{spec_prefix}RouteLabel,",
            f"        k{spec_prefix}EvidenceId,",
            f"        k{spec_prefix}GuardId,",
            f"        k{spec_prefix}FallbackPolicy,",
            f"        k{spec_prefix}MaterializationPolicy,",
            f"        k{rowset_prefix}RowCount}};",
            "",
            f"constexpr {rowset_prefix}Row k{rowset_prefix}Rows[] = {{",
        ]
    )
    lines.extend(row_initializers)
    lines.extend(
        [
            "};",
            "",
        ]
    )
    lines.extend(scalar_equal_helpers)
    lines.extend(factorized_group_lines)
    lines.extend(row_match_lines)
    lines.extend(
        [
            f"inline const {rowset_prefix}Row* {func_prefix}_{rowset_func}_find(",
        ]
    )
    lines.extend(lookup_params)
    lines.extend(
        [
            f"  for (const {rowset_prefix}Row& row : k{rowset_prefix}Rows) {{",
            f"    if ({lookup_condition}) {{",
            "      return &row;",
            "    }",
            "  }",
            "  return nullptr;",
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
    if _sparse_rowsets(spec):
        return generate_generic_sparse_rowset_shape_envelope_header(
            spec, source_name
        )
    if _variadic_tensor_list_input(spec) is not None:
        return generate_generic_variadic_tensor_list_shape_envelope_header(
            spec, source_name
        )
    if _simple_bounds_shape_envelope_fields(bounds) is not None:
        return generate_generic_simple_bounds_shape_envelope_header(
            spec, source_name
        )
    if _shape_layout_simple_bounds_shape_envelope_fields(bounds) is not None:
        return generate_generic_shape_layout_simple_bounds_shape_envelope_header(
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
    broadcast_relationships = _broadcast_compatible_relationships(
        envelope, "ShapeEnvelope"
    )

    op_values = bounds["ops"]
    op_constants = []
    op_fields = []
    op_initializers = []
    op_match_params = []
    for index, op in enumerate(op_values):
        op_suffix = _cpp_identifier_fragment(op)
        op_constants.append(
            f"constexpr const char* k{prefix}Op{op_suffix} = {_cpp_string(op)};"
        )
        op_fields.append(f"  const char* op_{index};")
        op_initializers.append(f"        k{prefix}Op{op_suffix},")
        op_match_params.append(f"op_{_cpp_lower_identifier(op)}")
    op_match_param_lines = [
        f"    const bool {param}," for param in op_match_params
    ]
    op_match_condition = " || ".join(op_match_params)

    relationship_constants = []
    relationship_fields = []
    relationship_initializers = []
    relationship_helpers = []
    for relationship in broadcast_relationships:
        relationship_constants.append(
            f"constexpr std::int64_t k{prefix}BroadcastCompatibleMaxRank = "
            f"{relationship['max_rank']};"
        )
        relationship_fields.append(
            "  std::int64_t broadcast_compatible_max_rank;"
        )
        relationship_initializers.append(
            f"        k{prefix}BroadcastCompatibleMaxRank,"
        )
        relationship_helpers.extend(
            [
                f"inline bool {func_prefix}_broadcast_compatible(",
                f"    const {prefix}Spec& spec,",
                "    const IntArrayRef left_sizes,",
                "    const IntArrayRef right_sizes) {",
                (
                    "  const std::int64_t left_rank = "
                    "static_cast<std::int64_t>(left_sizes.size());"
                ),
                (
                    "  const std::int64_t right_rank = "
                    "static_cast<std::int64_t>(right_sizes.size());"
                ),
                "  if (left_rank > spec.broadcast_compatible_max_rank ||",
                "      right_rank > spec.broadcast_compatible_max_rank) {",
                "    return false;",
                "  }",
                (
                    "  const std::int64_t max_rank = "
                    "left_rank > right_rank ? left_rank : right_rank;"
                ),
                "  for (std::int64_t axis = 0; axis < max_rank; ++axis) {",
                "    const std::int64_t left_axis = left_rank - 1 - axis;",
                "    const std::int64_t right_axis = right_rank - 1 - axis;",
                "    const std::int64_t left_dim = left_axis >= 0 ? left_sizes[left_axis] : 1;",
                "    const std::int64_t right_dim = right_axis >= 0 ? right_sizes[right_axis] : 1;",
                "    if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {",
                "      return false;",
                "    }",
                "  }",
                "  return true;",
                "}",
                "",
            ]
        )

    lines = [
        "// Generated by tools/vulkan_contracts/gen_contract_spec_cpp.py",
        f"// Source: {source_name}",
        "// Do not edit by hand.",
        "",
        "#pragma once",
        "",
        "#include <ATen/ArrayRef.h>",
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
    lines.extend(relationship_constants)
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
    lines.extend(relationship_fields)
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
    lines.extend(relationship_initializers)
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
        ]
    )
    lines.extend(op_match_param_lines)
    lines.extend(
        [
            "    const bool alpha_is_one,",
            "    const bool has_output,",
            "    const bool inplace) {",
            f"  return ({op_match_condition}) && alpha_is_one == spec.alpha_is_one &&",
            "      has_output == spec.has_output && inplace == spec.inplace;",
            "}",
            "",
        ]
    )
    lines.extend(relationship_helpers)
    lines.extend(
        [
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
    if "shape_envelope" in spec:
        output = generate_generic_shape_envelope_header(spec, source_name)
        return _prune_generated_cpp_helpers(
            output,
            spec.get("generated_cpp_entry_points"),
        )
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
