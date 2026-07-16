#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = Path("docs/vulkan/cleanup_ledger.json")
INVENTORY_PATH = Path("docs/vulkan/generated/cleanup_surface_inventory.json")
REGISTER_PATH = Path("aten/src/ATen/native/vulkan/ops/Register.cpp")
ENV_REGISTRY_PATH = Path("aten/src/ATen/native/vulkan/api/Env.cpp")
PYTHON_API_PATHS = (
    Path("torch/vulkan/__init__.py"),
    Path("torch/backends/vulkan/__init__.py"),
)
ENV_SCAN_ROOTS = (
    Path("aten/src/ATen/native/vulkan"),
    Path("torch/vulkan"),
    Path("scripts/benchmarks"),
)
ENV_SOURCE_SUFFIXES = (".cpp", ".h", ".py")
RETIRED_CODE_SUFFIXES = (".cpp", ".h", ".glsl", ".py")
STATES = ("Active", "Migration", "Compatibility", "Delete-ready")
ENV_NAME_RE = re.compile(r"PYTORCH_VULKAN_[A-Z0-9_]+")
CPP_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.S)
CPP_ENV_CALL_RE = re.compile(
    r"(?P<callee>(?:(?:\w+)::)*\w*(?:env|getenv)\w*)\s*\(\s*"
    r'"(?P<name>PYTORCH_VULKAN_[A-Z0-9_]+)"',
    re.I,
)


class InventoryError(RuntimeError):
    pass


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _cpp_string_value(token: str) -> str:
    return ast.literal_eval(token)


def _matching_cpp_delimiter(
    text: str, open_offset: int, opener: str, closer: str
) -> int:
    depth = 0
    state = "code"
    index = open_offset
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and next_char == "/":
                state = "code"
                index += 1
        elif state in ("string", "char"):
            quote = '"' if state == "string" else "'"
            if char == "\\":
                index += 1
            elif char == quote:
                state = "code"
        elif char == "/" and next_char == "/":
            state = "line_comment"
            index += 1
        elif char == "/" and next_char == "*":
            state = "block_comment"
            index += 1
        elif char == '"':
            state = "string"
        elif char == "'":
            state = "char"
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise InventoryError(f"Unclosed C++ delimiter at offset {open_offset}")


def _normalize_schema(schema: str) -> str:
    return " ".join(schema.split())


def _operator_name(schema: str, namespace: str) -> str:
    match = re.match(r"([^\s(]+)\s*\(", schema)
    if not match:
        raise InventoryError(f"Could not parse operator schema: {schema!r}")
    name = match.group(1)
    return name if "::" in name else f"{namespace}::{name}"


def discover_operator_schemas(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / REGISTER_PATH
    text = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    library_re = re.compile(
        r"TORCH_LIBRARY\s*\(\s*(?P<namespace>\w+)\s*,\s*(?P<var>\w+)\s*\)\s*\{"
    )
    for library_match in library_re.finditer(text):
        namespace = library_match.group("namespace")
        variable = library_match.group("var")
        block_open = text.find("{", library_match.start())
        block_close = _matching_cpp_delimiter(text, block_open, "{", "}")
        block = text[block_open + 1 : block_close]
        block_start = block_open + 1
        definition_re = re.compile(rf"\b{re.escape(variable)}\.def\s*\(")
        for definition_match in definition_re.finditer(block):
            call_open = block_start + definition_match.end() - 1
            call_close = _matching_cpp_delimiter(text, call_open, "(", ")")
            call = text[call_open + 1 : call_close]
            schema = _normalize_schema(
                "".join(
                    _cpp_string_value(token)
                    for token in CPP_STRING_RE.findall(call)
                )
            )
            if not schema:
                continue
            name = _operator_name(schema, namespace)
            records.append(
                {
                    "id": f"operator_schema:{name}",
                    "kind": "operator_schema",
                    "name": name,
                    "schema": schema,
                    "source": _relative(path, repo_root),
                    "line": _line_number(text, call_open),
                }
            )
    return records


def discover_custom_classes(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / REGISTER_PATH
    text = path.read_text(encoding="utf-8")
    class_re = re.compile(
        r"torch::selective_class_<.*?>\s*\(\s*"
        r'"(?P<namespace>[^"]+)"\s*,\s*'
        r'TORCH_SELECTIVE_CLASS\(\s*"(?P<name>[^"]+)"\s*\)',
        re.S,
    )
    records = []
    for match in class_re.finditer(text):
        qualified_name = f"{match.group('namespace')}::{match.group('name')}"
        records.append(
            {
                "id": f"custom_class:{qualified_name}",
                "kind": "custom_class",
                "name": qualified_name,
                "source": _relative(path, repo_root),
                "line": _line_number(text, match.start()),
            }
        )
    library_re = re.compile(
        r"TORCH_LIBRARY\s*\(\s*(?P<namespace>\w+)\s*,\s*(?P<var>\w+)\s*\)\s*\{"
    )
    for library_match in library_re.finditer(text):
        namespace = library_match.group("namespace")
        variable = library_match.group("var")
        block_open = text.find("{", library_match.start())
        block_close = _matching_cpp_delimiter(text, block_open, "{", "}")
        block = text[block_open + 1 : block_close]
        class_re = re.compile(
            rf"\b{re.escape(variable)}\.class_<.*?>\s*\(\s*"
            r'"(?P<name>[^"]+)"',
            re.S,
        )
        for match in class_re.finditer(block):
            qualified_name = f"{namespace}::{match.group('name')}"
            records.append(
                {
                    "id": f"custom_class:{qualified_name}",
                    "kind": "custom_class",
                    "name": qualified_name,
                    "source": _relative(path, repo_root),
                    "line": _line_number(text, block_open + 1 + match.start()),
                }
            )
    return records


def _python_constant_values(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        value = statement.value
        if value is None:
            continue
        strings: tuple[str, ...] = ()
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            strings = (value.value,)
        elif isinstance(value, (ast.Tuple, ast.List)) and all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in value.elts
        ):
            strings = tuple(element.value for element in value.elts)
        if not strings:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = strings
    return values


def _resolved_python_strings(
    node: ast.AST | None, constants: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        return constants.get(node.id, ())
    return ()


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _python_env_accesses(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    constants = _python_constant_values(tree)
    accesses: list[dict[str, Any]] = []
    relative_path = _relative(path, repo_root)

    def add(names: tuple[str, ...], line: int, access: str) -> None:
        for name in names:
            if ENV_NAME_RE.fullmatch(name):
                accesses.append(
                    {
                        "name": name,
                        "source": relative_path,
                        "line": line,
                        "access": access,
                    }
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if _is_os_environ(node.func.value) and node.func.attr in {
                "get",
                "pop",
                "setdefault",
            }:
                names = _resolved_python_strings(
                    node.args[0] if node.args else None, constants
                )
                access = {
                    "get": "read",
                    "pop": "delete",
                    "setdefault": "read_write",
                }[node.func.attr]
                add(names, node.lineno, access)
            elif (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "getenv"
            ):
                add(
                    _resolved_python_strings(
                        node.args[0] if node.args else None, constants
                    ),
                    node.lineno,
                    "read",
                )
        elif isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            access = "read"
            if isinstance(node.ctx, ast.Store):
                access = "write"
            elif isinstance(node.ctx, ast.Del):
                access = "delete"
            add(_resolved_python_strings(node.slice, constants), node.lineno, access)
    return accesses


def _env_registry(repo_root: Path) -> dict[str, dict[str, str]]:
    path = repo_root / ENV_REGISTRY_PATH
    text = path.read_text(encoding="utf-8")
    entry_re = re.compile(
        r'\{"(?P<name>PYTORCH_VULKAN_[A-Z0-9_]+)",\s*'
        r"VulkanEnvFlagKind::(?P<kind>\w+),\s*"
        r'"(?P<reason>(?:\\.|[^"\\])*)",\s*'
        r'"(?P<coverage>(?:\\.|[^"\\])*)"\}',
        re.S,
    )
    return {
        match.group("name"): {
            "kind": match.group("kind"),
            "reason": _normalize_schema(match.group("reason")),
            "coverage": _normalize_schema(match.group("coverage")),
        }
        for match in entry_re.finditer(text)
    }


def discover_environment_accesses(repo_root: Path) -> list[dict[str, Any]]:
    sites: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relative_root in ENV_SCAN_ROOTS:
        root = repo_root / relative_root
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in ENV_SOURCE_SUFFIXES:
                continue
            if path.suffix == ".py":
                for access in _python_env_accesses(path, repo_root):
                    sites[access["name"]].append(
                        {key: value for key, value in access.items() if key != "name"}
                    )
                continue
            text = path.read_text(encoding="utf-8")
            for match in CPP_ENV_CALL_RE.finditer(text):
                sites[match.group("name")].append(
                    {
                        "source": _relative(path, repo_root),
                        "line": _line_number(text, match.start()),
                        "access": "read",
                        "callee": match.group("callee"),
                    }
                )

    registry = _env_registry(repo_root)
    records = []
    for name in sorted(sites):
        unique_sites = sorted(
            {json.dumps(site, sort_keys=True) for site in sites[name]}
        )
        record: dict[str, Any] = {
            "id": f"environment_variable:{name}",
            "kind": "environment_variable",
            "name": name,
            "sites": [json.loads(site) for site in unique_sites],
        }
        if name in registry:
            record["registry"] = registry[name]
        records.append(record)
    return records


def discover_python_entrypoints(repo_root: Path) -> list[dict[str, Any]]:
    records = []
    for relative_path in PYTHON_API_PATHS:
        path = repo_root / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_name = relative_path.with_suffix("").as_posix().replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        exports: list[ast.Constant] = []
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in statement.targets
            ):
                continue
            if not isinstance(statement.value, (ast.List, ast.Tuple)):
                raise InventoryError(f"{relative_path}: __all__ must be a literal list")
            exports.extend(statement.value.elts)
        for export in exports:
            if not isinstance(export, ast.Constant) or not isinstance(
                export.value, str
            ):
                raise InventoryError(f"{relative_path}: __all__ must contain strings")
            qualified_name = f"{module_name}.{export.value}"
            records.append(
                {
                    "id": f"python_entrypoint:{qualified_name}",
                    "kind": "python_entrypoint",
                    "name": qualified_name,
                    "source": relative_path.as_posix(),
                    "line": export.lineno,
                }
            )
    return records


def discover_surfaces(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    records = [
        *discover_operator_schemas(repo_root),
        *discover_custom_classes(repo_root),
        *discover_environment_accesses(repo_root),
        *discover_python_entrypoints(repo_root),
    ]
    by_id: dict[str, dict[str, Any]] = {}
    duplicates = []
    for record in records:
        surface_id = record["id"]
        if surface_id in by_id:
            duplicates.append(surface_id)
        by_id[surface_id] = record
    if duplicates:
        raise InventoryError(
            f"Duplicate discovered surfaces: {sorted(set(duplicates))}"
        )
    return [by_id[surface_id] for surface_id in sorted(by_id)]


def load_ledger(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    with (repo_root / LEDGER_PATH).open(encoding="utf-8") as handle:
        return json.load(handle)


def _ledger_location_matches(
    location: str, repo_root: Path, *, allow_glob: bool
) -> list[str]:
    if not isinstance(location, str) or not location:
        raise InventoryError("Ledger locations must be non-empty strings")
    path = Path(location)
    if path.is_absolute() or ".." in path.parts or "\\" in location:
        raise InventoryError(
            f"Ledger location must be a normalized relative path: {location!r}"
        )

    has_glob = any(character in location for character in "*?[")
    if has_glob and not allow_glob:
        raise InventoryError(f"Ledger document cannot be a glob: {location!r}")
    matches = sorted(repo_root.glob(location)) if has_glob else [repo_root / path]
    matches = [match for match in matches if match.exists()]
    if not matches:
        raise InventoryError(f"Ledger location does not exist: {location!r}")
    if not allow_glob and not matches[0].is_file():
        raise InventoryError(f"Ledger document is not a file: {location!r}")
    return [_relative(match, repo_root) for match in matches]


def resolve_ledger_locations(
    ledger: dict[str, Any], repo_root: Path = REPO_ROOT
) -> dict[str, dict[str, list[str]]]:
    resolved: dict[str, dict[str, list[str]]] = {}
    for entry in ledger.get("entries", []):
        entry_id = entry.get("id")
        paths = entry.get("paths", [])
        documents = entry.get("documents", [])
        if not isinstance(paths, list):
            raise InventoryError(f"Ledger entry {entry_id!r} paths must be a list")
        if not isinstance(documents, list):
            raise InventoryError(
                f"Ledger entry {entry_id!r} documents must be a list"
            )
        resolved_paths = {
            match
            for location in paths
            for match in _ledger_location_matches(
                location, repo_root, allow_glob=True
            )
        }
        resolved_documents = {
            match
            for location in documents
            for match in _ledger_location_matches(
                location, repo_root, allow_glob=False
            )
        }
        resolved[entry_id] = {
            "resolved_paths": sorted(resolved_paths),
            "resolved_documents": sorted(resolved_documents),
        }
    return resolved


def validate_scope_decisions(
    ledger: dict[str, Any], repo_root: Path = REPO_ROOT
) -> None:
    decisions = ledger.get("scope_decisions")
    if not isinstance(decisions, list):
        raise InventoryError("Ledger scope_decisions must be a list")

    for decision in decisions:
        if decision.get("status") != "deleted":
            continue
        decision_id = decision.get("id")
        removed_paths = decision.get("removed_paths", [])
        scan_roots = decision.get("scan_roots")
        forbidden_symbols = decision.get("forbidden_code_symbols")
        if not isinstance(decision_id, str) or not decision_id:
            raise InventoryError("Every deleted scope decision needs an id")
        if not isinstance(removed_paths, list):
            raise InventoryError(
                f"Deleted scope decision {decision_id!r} has invalid removed_paths"
            )
        if not isinstance(scan_roots, list) or not scan_roots:
            raise InventoryError(
                f"Deleted scope decision {decision_id!r} needs scan_roots"
            )
        if not isinstance(forbidden_symbols, list) or not forbidden_symbols:
            raise InventoryError(
                f"Deleted scope decision {decision_id!r} needs "
                "forbidden_code_symbols"
            )

        restored_paths = [
            relative_path
            for relative_path in removed_paths
            if (repo_root / relative_path).exists()
        ]
        if restored_paths:
            raise InventoryError(
                f"Deleted scope decision {decision_id!r} restored paths: "
                f"{restored_paths}"
            )

        hits = []
        for relative_root in scan_roots:
            root = repo_root / relative_root
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in RETIRED_CODE_SUFFIXES:
                    continue
                text = path.read_text(encoding="utf-8")
                for symbol in forbidden_symbols:
                    offset = text.find(symbol)
                    if offset >= 0:
                        hits.append(
                            {
                                "symbol": symbol,
                                "source": _relative(path, repo_root),
                                "line": _line_number(text, offset),
                            }
                        )
        if hits:
            raise InventoryError(
                f"Deleted scope decision {decision_id!r} restored symbols: "
                f"{hits}"
            )


def classify_surfaces(
    surfaces: list[dict[str, Any]], ledger: dict[str, Any]
) -> list[dict[str, Any]]:
    if ledger.get("states") != list(STATES):
        raise InventoryError(f"Ledger states must be exactly {list(STATES)}")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise InventoryError("Ledger entries must be a list")

    classification: dict[str, dict[str, str]] = {}
    required_fields = {
        "Active": ("why",),
        "Migration": ("replacement", "delete_when", "baseline"),
        "Compatibility": ("evidence", "delete_when"),
        "Delete-ready": ("deletion_wave", "preserve"),
    }
    for entry in entries:
        entry_id = entry.get("id")
        state = entry.get("state")
        members = entry.get("surfaces", [])
        if not isinstance(entry_id, str) or not entry_id:
            raise InventoryError("Every ledger entry needs an id")
        if state not in STATES:
            raise InventoryError(
                f"Ledger entry {entry_id!r} has invalid state {state!r}"
            )
        if not isinstance(entry.get("summary"), str) or not entry["summary"]:
            raise InventoryError(f"Ledger entry {entry_id!r} needs a summary")
        if not isinstance(members, list):
            raise InventoryError(f"Ledger entry {entry_id!r} surfaces must be a list")
        if not members and not entry.get("paths") and not entry.get("documents"):
            raise InventoryError(
                f"Ledger entry {entry_id!r} needs surfaces, paths, or documents"
            )
        missing_fields = [
            field for field in required_fields[state] if not entry.get(field)
        ]
        if missing_fields:
            raise InventoryError(
                f"Ledger entry {entry_id!r} is missing {missing_fields}"
            )
        for surface_id in members:
            if surface_id in classification:
                previous = classification[surface_id]["ledger_entry"]
                raise InventoryError(
                    f"Surface {surface_id!r} is in both {previous!r} and {entry_id!r}"
                )
            classification[surface_id] = {
                "state": state,
                "ledger_entry": entry_id,
            }

    discovered_ids = {surface["id"] for surface in surfaces}
    classified_ids = set(classification)
    unclassified = sorted(discovered_ids - classified_ids)
    stale = sorted(classified_ids - discovered_ids)
    if unclassified or stale:
        parts = []
        if unclassified:
            parts.append(f"unclassified surfaces: {unclassified}")
        if stale:
            parts.append(f"stale ledger surfaces: {stale}")
        raise InventoryError("; ".join(parts))

    return [
        {**surface, **classification[surface["id"]]}
        for surface in surfaces
    ]


def build_inventory(
    repo_root: Path = REPO_ROOT, ledger: dict[str, Any] | None = None
) -> dict[str, Any]:
    ledger = ledger if ledger is not None else load_ledger(repo_root)
    validate_scope_decisions(ledger, repo_root)
    surfaces = classify_surfaces(discover_surfaces(repo_root), ledger)
    resolved_locations = resolve_ledger_locations(ledger, repo_root)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for surface in surfaces:
        counts["by_kind"][surface["kind"]] += 1
        counts["by_state"][surface["state"]] += 1
    entry_summaries = []
    for entry in ledger["entries"]:
        entry_summaries.append(
            {
                key: entry[key]
                for key in (
                    "id",
                    "state",
                    "summary",
                    "paths",
                    "documents",
                )
                if key in entry
            }
            | resolved_locations[entry["id"]]
            | {"surface_count": len(entry.get("surfaces", []))}
        )
    return {
        "schema_version": 1,
        "generator": _relative(Path(__file__).resolve(), repo_root),
        "ledger": LEDGER_PATH.as_posix(),
        "baseline_policy": ledger["baseline_policy"],
        "compatibility_audit": ledger["compatibility_audit"],
        "scope_decisions": ledger["scope_decisions"],
        "counts": {key: dict(sorted(value.items())) for key, value in counts.items()},
        "ledger_entries": entry_summaries,
        "surfaces": surfaces,
    }


def inventory_text(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"


def check_inventory(repo_root: Path = REPO_ROOT) -> None:
    expected = inventory_text(build_inventory(repo_root))
    path = repo_root / INVENTORY_PATH
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
    if actual != expected:
        raise InventoryError(
            f"{INVENTORY_PATH} is stale; run "
            "python tools/vulkan_cleanup/generate_surface_inventory.py --write"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--print-surfaces", action="store_true")
    args = parser.parse_args()

    try:
        if args.print_surfaces:
            for surface in discover_surfaces():
                print(surface["id"])
            return 0
        if args.write:
            path = REPO_ROOT / INVENTORY_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(inventory_text(build_inventory()), encoding="utf-8")
            return 0
        check_inventory()
        return 0
    except (InventoryError, OSError, ValueError, SyntaxError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
