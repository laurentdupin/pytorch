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
    args = parser.parse_args()

    specs = load_all_contract_specs(args.repo_root)
    for file_name, spec in specs:
        require_fields(spec, CONTRACT_SPEC_REQUIRED_FIELDS, file_name)
        if args.list:
            print(
                f"{file_name}: {spec['contract_name']} "
                f"{spec['family']} {spec['tuple_id']}"
            )


if __name__ == "__main__":
    _main()
