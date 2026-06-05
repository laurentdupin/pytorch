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


def _require_non_empty_string(mapping, field, context):
    value = mapping[field]
    if not isinstance(value, str) or value == "":
        raise AssertionError(f"{context} {field} must be a non-empty string")


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
            }
        )
    return rows


def format_contract_spec_summary_row(row):
    return (
        f"{row['file_name']}: {row['contract_name']} {row['family']} "
        f"{row['tuple_id']} positive_cases={row['positive_cases']} "
        f"negative_cases={row['negative_cases']}"
    )


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


if __name__ == "__main__":
    _main()
