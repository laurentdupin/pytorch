#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SPEC_DIR = Path("test") / "vulkan_contract_specs"
PROOF_DIR = Path("test") / "vulkan_contract_proofs"
GENERATED_CPP_MANIFEST = SPEC_DIR / "generated_cpp_manifest.json"
DEFAULT_ACCEPTED_MANIFEST = PROOF_DIR / "accepted_contract_rows_manifest.json"
DEFAULT_PROOF_MANIFEST = PROOF_DIR / "contract_proof_manifest.json"

COVERED_PROOF_CONTRACTS = (
    "AttentionProbabilityMaterializationContract",
    "PatchEmbedFeatureMapToTokensContract",
    "PatchEmbedFloatBufferConvRoute",
    "SmallSpatialPointwiseConvContract",
    "TokenPrefixCatAddContract",
)

SOURCE_HINTS: dict[str, dict[str, list[str]]] = {
    "AttentionProbabilityMaterializationContract": {
        "matcher_sources": [
            "aten/src/ATen/native/vulkan/ops/Softmax.cpp",
            "aten/src/ATen/native/vulkan/planning/ExecutionContractsSDPAExecutionPolicy.cpp",
        ],
        "route_policy_sources": [
            "aten/src/ATen/native/vulkan/planning/RoutePolicy.cpp",
        ],
        "transition_contract_sources": [
            "aten/src/ATen/native/vulkan/planning/TransitionContracts.cpp",
            "aten/src/ATen/native/vulkan/planning/TransitionPlanner.cpp",
        ],
    },
    "PatchEmbedFeatureMapToTokensContract": {
        "matcher_sources": [
            "aten/src/ATen/native/vulkan/ops/VisionBlocks.cpp",
        ],
        "route_policy_sources": [
            "scripts/benchmarks/benchmark_depth_anything.py",
        ],
        "transition_contract_sources": [
            "aten/src/ATen/native/vulkan/planning/TransitionContracts.cpp",
        ],
    },
    "PatchEmbedFloatBufferConvRoute": {
        "matcher_sources": [
            "aten/src/ATen/native/vulkan/ops/Convolution.cpp",
        ],
        "route_policy_sources": [
            "aten/src/ATen/native/vulkan/ops/Convolution.cpp",
        ],
        "transition_contract_sources": [
            "aten/src/ATen/native/vulkan/planning/TransitionContracts.cpp",
        ],
    },
    "SmallSpatialPointwiseConvContract": {
        "matcher_sources": [
            "aten/src/ATen/native/vulkan/planning/ExecutionContractsSmallSpatialPointwiseConv.cpp",
            "aten/src/ATen/native/vulkan/ops/Convolution.cpp",
        ],
        "route_policy_sources": [
            "aten/src/ATen/native/vulkan/ops/Convolution.cpp",
        ],
        "transition_contract_sources": [],
    },
    "TokenPrefixCatAddContract": {
        "matcher_sources": [
            "aten/src/ATen/native/vulkan/ops/VisionBlocks.cpp",
        ],
        "route_policy_sources": [
            "scripts/benchmarks/benchmark_depth_anything.py",
        ],
        "transition_contract_sources": [
            "aten/src/ATen/native/vulkan/planning/TransitionContracts.cpp",
        ],
    },
}

PROOF_TEMPLATES: dict[str, dict[str, Any]] = {
    "AttentionProbabilityMaterializationContract": {
        "proof_status": "validated_bounded_transition_rows",
        "positive_runtime_or_proof_cases": [
            "attention_probability_materialization_contract.json positive_cases",
            "agent_space/attention_probability_owner_path_no_clone_diagnostic.json",
            "agent_space/attention_probability_materialization_current_runtime_diagnostics.json",
        ],
        "adjacent_negative_coverage": [
            "attention_probability_materialization_contract.json negative_cases",
            "ShapeEnvelope adjacent-negative generator",
        ],
        "fallback_readback_copy_budget": {
            "cpu_fallback": 0,
            "sync_readback": 0,
            "host_transfer": 0,
        },
        "expiry": "broader attention probability producer/consumer layout proof replaces finite rows",
        "migration_target": "generated LayoutTransitionContract or attention RegionContract policy",
    },
    "PatchEmbedFeatureMapToTokensContract": {
        "proof_status": "validated_bounded_layout_transition_rows",
        "positive_runtime_or_proof_cases": [
            "patch_embed_feature_map_to_tokens_contract.json positive_cases",
            "test_patch_embed_feature_map_to_tokens_contract_matches_reference",
        ],
        "adjacent_negative_coverage": [
            "patch_embed_feature_map_to_tokens_contract.json negative_cases",
            "ShapeEnvelope adjacent-negative generator",
        ],
        "fallback_readback_copy_budget": {
            "cpu_fallback": 0,
            "sync_readback": 0,
            "host_transfer": 0,
        },
        "expiry": "broader patch-token layout transition proof covers adjacent feature-map pairs",
        "migration_target": "PatchEmbedFeatureMapToTokensContract parameterized layout-transition table",
    },
    "PatchEmbedFloatBufferConvRoute": {
        "proof_status": "validated_bounded_execution_plan_rows",
        "positive_runtime_or_proof_cases": [
            "patch_embed_float_buffer_conv_route_contract.json positive_cases",
            "test_patch_embed_float_buffer_conv_route_matches_cpu",
        ],
        "adjacent_negative_coverage": [
            "patch_embed_float_buffer_conv_route_contract.json negative_cases",
            "ShapeEnvelope adjacent-negative generator",
        ],
        "fallback_readback_copy_budget": {
            "cpu_fallback": 0,
            "sync_readback": 0,
            "vulkan_weight_cpu_materialization": 0,
        },
        "expiry": "broader kernel14 stride14 conv route proof covers adjacent image sizes and layouts",
        "migration_target": "PatchEmbedFloatBufferConvRoute generated execution-plan contract",
    },
    "SmallSpatialPointwiseConvContract": {
        "proof_status": "validated_sparse_rows_and_factorized_group",
        "positive_runtime_or_proof_cases": [
            "small_spatial_pointwise_conv_contract.json positive_cases",
            "agent_space/dav2_adjacent_pointwise_factorized_36_cross_gpu_isolated_proof.json",
            "agent_space/dav2_midres_pointwise_8shape_parity.json",
            "agent_space/dav2_midres_pointwise_downstream_4shape_parity.json",
            "agent_space/dav2_midres_pointwise_next_blocker_parity.json",
            "agent_space/dav2_midres_pointwise_1024_square_parity.json",
        ],
        "adjacent_negative_coverage": [
            "small_spatial_pointwise_conv_contract.json negative_cases",
            "ShapeEnvelope adjacent-negative generator",
        ],
        "fallback_readback_copy_budget": {
            "cpu_fallback": 0,
            "sync_readback": 0,
            "host_transfer": 0,
        },
        "expiry": "broader pointwise layout/channel/spatial proof replaces exact sparse rows",
        "migration_target": "generated pointwise KernelFamilyContract with row-level proof ledger",
    },
    "TokenPrefixCatAddContract": {
        "proof_status": "validated_bounded_fused_route_rows",
        "positive_runtime_or_proof_cases": [
            "token_prefix_cat_add_contract.json positive_cases",
            "agent_space/token_prefix_cat_add_observed_envelope_parity.json",
        ],
        "adjacent_negative_coverage": [
            "token_prefix_cat_add_contract.json negative_cases",
            "ShapeEnvelope adjacent-negative generator",
        ],
        "fallback_readback_copy_budget": {
            "cpu_fallback": 0,
            "sync_readback": 0,
            "host_transfer": 0,
        },
        "expiry": "broader token prefix/position-add region proof replaces finite rows",
        "migration_target": "TokenPrefixCatAddContract or token-preparation RegionContract",
    },
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def run_git(repo_root: Path, args: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def read_repo_bytes(repo_root: Path, rel_path: str | Path, git_ref: str | None) -> bytes | None:
    rel = str(rel_path).replace("\\", "/")
    if git_ref is None:
        path = repo_root / rel
        if not path.exists():
            return None
        return path.read_bytes()
    try:
        return run_git(repo_root, ["show", f"{git_ref}:{rel}"])
    except subprocess.CalledProcessError:
        return None


def read_repo_json(repo_root: Path, rel_path: str | Path, git_ref: str | None) -> Any | None:
    data = read_repo_bytes(repo_root, rel_path, git_ref)
    if data is None:
        return None
    return json.loads(data.decode("utf-8"))


def list_contract_spec_files(repo_root: Path, git_ref: str | None) -> list[str]:
    if git_ref is None:
        files = [
            path.name
            for path in (repo_root / SPEC_DIR).glob("*.json")
            if path.name != GENERATED_CPP_MANIFEST.name
        ]
        return sorted(files)
    output = run_git(repo_root, ["ls-tree", "-r", "--name-only", git_ref, str(SPEC_DIR).replace("\\", "/")])
    files = []
    for line in output.decode("utf-8").splitlines():
        path = Path(line)
        if path.suffix == ".json" and path.name != GENERATED_CPP_MANIFEST.name:
            files.append(path.name)
    return sorted(files)


def sha256_repo_file(repo_root: Path, rel_path: str | Path, git_ref: str | None) -> str | None:
    data = read_repo_bytes(repo_root, rel_path, git_ref)
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()


def combined_source_digest(repo_root: Path, files: list[str], git_ref: str | None) -> dict[str, Any]:
    file_hashes = []
    missing = []
    for rel in files:
        digest = sha256_repo_file(repo_root, rel, git_ref)
        if digest is None:
            missing.append(rel)
        else:
            file_hashes.append({"path": rel, "sha256": digest})
    payload = {"files": file_hashes, "missing": missing}
    payload["combined_sha256"] = digest_value(payload)
    payload["known"] = bool(files) and not missing
    return payload


def generated_header_for_spec(repo_root: Path, spec_file: str, git_ref: str | None) -> str | None:
    manifest = read_repo_json(repo_root, GENERATED_CPP_MANIFEST, git_ref)
    if not manifest:
        return None
    for entry in manifest.get("entries", []):
        if entry.get("spec_file") == spec_file:
            return entry.get("header")
    return None


def spec_policies(spec: dict[str, Any]) -> dict[str, Any]:
    shape = spec.get("shape_envelope", {})
    metadata = spec.get("metadata", {})
    policies = copy.deepcopy(shape.get("policies", {}))
    if "fallback_policy" in metadata:
        policies.setdefault("fallback", metadata["fallback_policy"])
    if "materialization_policy" in metadata:
        policies.setdefault("materialization", metadata["materialization_policy"])
    transition = spec.get("transition_contract")
    if transition:
        policies["transition_contract"] = {
            key: transition.get(key)
            for key in (
                "contract_type",
                "reason",
                "kind",
                "outcome",
                "physical_copy",
                "host_transfer",
                "sync_required",
                "queue_submit_required",
                "copy_budget",
            )
            if key in transition
        }
    return policies


def common_entry_fields(
    repo_root: Path,
    spec_file: str,
    spec: dict[str, Any],
    git_ref: str | None,
) -> dict[str, Any]:
    header = generated_header_for_spec(repo_root, spec_file, git_ref)
    hints = SOURCE_HINTS.get(spec.get("contract_name", ""), {})
    spec_path = SPEC_DIR / spec_file
    generated_helper_hash = sha256_repo_file(repo_root, header, git_ref) if header else None
    dependencies = {
        "json_spec": {
            "path": str(spec_path).replace("\\", "/"),
            "sha256": sha256_repo_file(repo_root, spec_path, git_ref),
        },
        "generated_cpp_helper": {
            "path": header,
            "sha256": generated_helper_hash,
            "known": header is not None,
        },
        "matcher_source": combined_source_digest(repo_root, hints.get("matcher_sources", []), git_ref),
        "route_policy_dependency": combined_source_digest(
            repo_root,
            hints.get("route_policy_sources", []),
            git_ref,
        ),
        "transition_contract_dependency": combined_source_digest(
            repo_root,
            hints.get("transition_contract_sources", []),
            git_ref,
        ),
        "expected_counter_policy": {
            "policy": spec_policies(spec),
        },
    }
    dependencies["expected_counter_policy"]["sha256"] = digest_value(
        dependencies["expected_counter_policy"]["policy"]
    )
    dependencies["entry_dependency_digest"] = digest_value(dependencies)
    return {
        "spec_file": spec_file,
        "contract_name": spec["contract_name"],
        "family": spec["family"],
        "top_level_tuple_id": spec["tuple_id"],
        "writer_op": spec.get("writer_op"),
        "route_labels": sorted(
            {
                value
                for value in (
                    spec.get("route_label"),
                    spec.get("metadata", {}).get("route_label"),
                )
                if value
            }
        ),
        "fallback_policy": spec.get("metadata", {}).get("fallback_policy"),
        "materialization_policy": spec.get("metadata", {}).get("materialization_policy"),
        "expected_counter_policy": dependencies["expected_counter_policy"]["policy"],
        "generated_helper_source": header,
        "dependency_digests": dependencies,
    }


def entry_digest_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in entry.items()
        if key not in ("dependency_digests", "admission_digest")
    }


def finalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry["admission_digest"] = digest_value(entry_digest_payload(entry))
    return entry


def identity_for_entry(entry: dict[str, Any]) -> str:
    return entry["admission_key"]


def generate_entries_for_spec(
    repo_root: Path,
    spec_file: str,
    spec: dict[str, Any],
    git_ref: str | None,
) -> list[dict[str, Any]]:
    base = common_entry_fields(repo_root, spec_file, spec, git_ref)
    shape = spec.get("shape_envelope")
    if not shape:
        entry = copy.deepcopy(base)
        transition = spec.get("transition_contract", {})
        entry.update(
            {
                "admission_kind": "transition_reason_bucket",
                "tuple_id": spec.get("tuple_id"),
                "row_fields": {
                    "reason": transition.get("reason"),
                    "kind": transition.get("kind"),
                    "producer_schema": transition.get("producer_schema"),
                    "consumer_schema": transition.get("consumer_schema"),
                },
                "extrapolation_class": "schema_only_transition_bucket",
                "cardinality_estimate": len(spec.get("positive_cases", [])),
            }
        )
        entry["admission_key"] = (
            f"{entry['contract_name']}|{entry['family']}|"
            f"{entry['admission_kind']}|{entry['tuple_id']}"
        )
        return [finalize_entry(entry)]

    entries: list[dict[str, Any]] = []
    for rowset in shape.get("sparse_rowsets", []):
        rowset_name = rowset["name"]
        for row in rowset.get("rows", []):
            tuple_id = row.get(rowset.get("label_field", "tuple_id"), row.get("tuple_id", spec["tuple_id"]))
            entry = copy.deepcopy(base)
            route_labels = set(entry["route_labels"])
            for case in spec.get("positive_cases", []):
                if case.get("expected_contract_tuple_id") == tuple_id and case.get("expected_route_label"):
                    route_labels.add(case["expected_route_label"])
            entry.update(
                {
                    "admission_kind": "exact_sparse_row",
                    "rowset_name": rowset_name,
                    "tuple_id": tuple_id,
                    "row_fields": copy.deepcopy(row),
                    "extrapolation_class": row.get(
                        "proof_class",
                        row.get("extrapolation_class", "exact_sparse_row"),
                    ),
                    "cardinality_estimate": 1,
                    "route_labels": sorted(route_labels),
                }
            )
            entry["admission_key"] = (
                f"{entry['contract_name']}|{entry['family']}|"
                f"{entry['admission_kind']}|{rowset_name}|{tuple_id}"
            )
            entries.append(finalize_entry(entry))

    for group in shape.get("factorized_groups", []):
        entry = copy.deepcopy(base)
        tuple_id = group.get("tuple_id", group["name"])
        entry.update(
            {
                "admission_kind": "factorized_group",
                "rowset_name": group["name"],
                "tuple_id": tuple_id,
                "row_fields": copy.deepcopy(group),
                "extrapolation_class": "factorized_group",
                "cardinality_estimate": group.get("cardinality"),
            }
        )
        entry["admission_key"] = (
            f"{entry['contract_name']}|{entry['family']}|"
            f"{entry['admission_kind']}|{group['name']}|{tuple_id}"
        )
        entries.append(finalize_entry(entry))

    if not entries:
        entry = copy.deepcopy(base)
        entry.update(
            {
                "admission_kind": "bounded_envelope",
                "tuple_id": spec.get("tuple_id"),
                "row_fields": copy.deepcopy(shape.get("bounds", spec.get("bounds", {}))),
                "extrapolation_class": "bounded_envelope",
                "cardinality_estimate": len(spec.get("positive_cases", [])),
            }
        )
        entry["admission_key"] = (
            f"{entry['contract_name']}|{entry['family']}|"
            f"{entry['admission_kind']}|{entry['tuple_id']}"
        )
        entries.append(finalize_entry(entry))
    return entries


def generate_accepted_manifest(repo_root: Path, git_ref: str | None = None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for spec_file in list_contract_spec_files(repo_root, git_ref):
        spec = read_repo_json(repo_root, SPEC_DIR / spec_file, git_ref)
        if spec is None:
            continue
        entries.extend(generate_entries_for_spec(repo_root, spec_file, spec, git_ref))
    entries.sort(key=lambda entry: entry["admission_key"])
    summaries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        contract = entry["contract_name"]
        summary = summaries.setdefault(
            contract,
            {
                "entry_count": 0,
                "cardinality_estimate": 0,
                "exact_sparse_row_count": 0,
                "factorized_group_count": 0,
                "bounded_envelope_count": 0,
                "transition_reason_bucket_count": 0,
            },
        )
        summary["entry_count"] += 1
        if isinstance(entry.get("cardinality_estimate"), int):
            summary["cardinality_estimate"] += entry["cardinality_estimate"]
        kind_count_key = f"{entry['admission_kind']}_count"
        summary[kind_count_key] = summary.get(kind_count_key, 0) + 1
    for contract, summary in summaries.items():
        contract_entries = [entry for entry in entries if entry["contract_name"] == contract]
        summary["admission_digest"] = digest_value(
            [
                {
                    "admission_key": entry["admission_key"],
                    "admission_digest": entry["admission_digest"],
                }
                for entry in contract_entries
            ]
        )
        summary["dependency_digest"] = digest_value(
            [
                {
                    "admission_key": entry["admission_key"],
                    "dependency_digest": entry["dependency_digests"]["entry_dependency_digest"],
                }
                for entry in contract_entries
            ]
        )
    manifest = {
        "schema_version": 1,
        "generator": "tools/vulkan_contract_codegen/compare_contract_admission.py",
        "git_ref": git_ref or "WORKTREE",
        "entry_count": len(entries),
        "covered_contracts_v1": list(COVERED_PROOF_CONTRACTS),
        "summaries": dict(sorted(summaries.items())),
        "entries": entries,
    }
    manifest["manifest_digest"] = digest_value(
        {
            "entries": [
                {
                    "admission_key": entry["admission_key"],
                    "admission_digest": entry["admission_digest"],
                    "dependency_digest": entry["dependency_digests"]["entry_dependency_digest"],
                }
                for entry in entries
            ],
            "summaries": manifest["summaries"],
        }
    )
    return manifest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_manifests(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    base_entries = {identity_for_entry(entry): entry for entry in base.get("entries", [])}
    current_entries = {identity_for_entry(entry): entry for entry in current.get("entries", [])}
    added_keys = sorted(set(current_entries) - set(base_entries))
    removed_keys = sorted(set(base_entries) - set(current_entries))
    common_keys = sorted(set(base_entries) & set(current_entries))
    metadata_changes = []
    dependency_changes = []
    for key in common_keys:
        base_entry = base_entries[key]
        current_entry = current_entries[key]
        if base_entry.get("admission_digest") != current_entry.get("admission_digest"):
            metadata_changes.append(
                {
                    "admission_key": key,
                    "before": base_entry.get("admission_digest"),
                    "after": current_entry.get("admission_digest"),
                }
            )
        if (
            base_entry.get("dependency_digests", {}).get("entry_dependency_digest")
            != current_entry.get("dependency_digests", {}).get("entry_dependency_digest")
        ):
            dependency_changes.append(
                {
                    "admission_key": key,
                    "before": base_entry.get("dependency_digests", {}).get("entry_dependency_digest"),
                    "after": current_entry.get("dependency_digests", {}).get("entry_dependency_digest"),
                }
            )
    cardinality_increases = []
    exact_row_debt_changes = []
    contracts = sorted(set(base.get("summaries", {})) | set(current.get("summaries", {})))
    for contract in contracts:
        before = base.get("summaries", {}).get(contract, {})
        after = current.get("summaries", {}).get(contract, {})
        if after.get("cardinality_estimate", 0) > before.get("cardinality_estimate", 0):
            cardinality_increases.append(
                {
                    "contract_name": contract,
                    "before": before.get("cardinality_estimate", 0),
                    "after": after.get("cardinality_estimate", 0),
                }
            )
        if after.get("exact_sparse_row_count", 0) != before.get("exact_sparse_row_count", 0):
            exact_row_debt_changes.append(
                {
                    "contract_name": contract,
                    "before": before.get("exact_sparse_row_count", 0),
                    "after": after.get("exact_sparse_row_count", 0),
                }
            )
    return {
        "schema_version": 1,
        "base_manifest_digest": base.get("manifest_digest"),
        "current_manifest_digest": current.get("manifest_digest"),
        "newly_admitted_rows": [current_entries[key] for key in added_keys],
        "removed_rows": [base_entries[key] for key in removed_keys],
        "metadata_changes": metadata_changes,
        "dependency_hash_changes": dependency_changes,
        "cardinality_increases": cardinality_increases,
        "exact_row_debt_changes": exact_row_debt_changes,
        "summary": {
            "newly_admitted_rows": len(added_keys),
            "removed_rows": len(removed_keys),
            "metadata_changes": len(metadata_changes),
            "dependency_hash_changes": len(dependency_changes),
            "cardinality_increases": len(cardinality_increases),
            "exact_row_debt_changes": len(exact_row_debt_changes),
        },
    }


def compare_has_deltas(report: dict[str, Any]) -> bool:
    return any(value for value in report["summary"].values())


def proof_entry_for_admission(entry: dict[str, Any]) -> dict[str, Any]:
    template = PROOF_TEMPLATES[entry["contract_name"]]
    return {
        "admission_key": entry["admission_key"],
        "contract_name": entry["contract_name"],
        "family": entry["family"],
        "tuple_id": entry["tuple_id"],
        "admission_kind": entry["admission_kind"],
        "proof_status": template["proof_status"],
        "positive_runtime_or_proof_cases": template["positive_runtime_or_proof_cases"],
        "adjacent_negative_coverage": template["adjacent_negative_coverage"],
        "fallback_readback_copy_budget": template["fallback_readback_copy_budget"],
        "expiry": template["expiry"],
        "migration_target": template["migration_target"],
        "admission_digest": entry["admission_digest"],
        "dependency_digest": entry["dependency_digests"]["entry_dependency_digest"],
        "cardinality_estimate": entry.get("cardinality_estimate"),
        "exact_row_debt": entry["admission_kind"] == "exact_sparse_row",
    }


def generate_proof_manifest(accepted_manifest: dict[str, Any]) -> dict[str, Any]:
    entries = [
        proof_entry_for_admission(entry)
        for entry in accepted_manifest["entries"]
        if entry["contract_name"] in COVERED_PROOF_CONTRACTS
    ]
    entries.sort(key=lambda entry: entry["admission_key"])
    covered = {}
    for contract in COVERED_PROOF_CONTRACTS:
        accepted_summary = accepted_manifest["summaries"].get(contract, {})
        contract_entries = [entry for entry in entries if entry["contract_name"] == contract]
        covered[contract] = {
            "proof_entry_count": len(contract_entries),
            "accepted_entry_count": accepted_summary.get("entry_count", 0),
            "cardinality_estimate": accepted_summary.get("cardinality_estimate", 0),
            "exact_sparse_row_count": accepted_summary.get("exact_sparse_row_count", 0),
            "factorized_group_count": accepted_summary.get("factorized_group_count", 0),
            "admission_digest": accepted_summary.get("admission_digest"),
            "dependency_digest": accepted_summary.get("dependency_digest"),
        }
    manifest = {
        "schema_version": 1,
        "source_accepted_manifest_digest": accepted_manifest["manifest_digest"],
        "coverage_scope": "v1_high_risk_contracts",
        "covered_contracts": covered,
        "proof_entries": entries,
        "explicit_debt": [
            {
                "scope": "broader_shape_envelope_contracts",
                "status": "todo",
                "reason": "v1 gates high-risk contracts first; remaining contracts still have spec/governance coverage but not proof-carrying dependency ledgers",
                "migration_target": "extend proof manifest coverage contract by contract",
            }
        ],
    }
    manifest["proof_manifest_digest"] = digest_value(
        {
            "covered_contracts": covered,
            "proof_entries": entries,
        }
    )
    return manifest


def validate_proof_manifest(
    accepted_manifest: dict[str, Any],
    proof_manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    accepted_by_key = {entry["admission_key"]: entry for entry in accepted_manifest["entries"]}
    proof_by_key = {
        entry["admission_key"]: entry
        for entry in proof_manifest.get("proof_entries", [])
    }
    covered_contracts = set(proof_manifest.get("covered_contracts", {}))
    for entry in accepted_manifest["entries"]:
        if entry["contract_name"] not in covered_contracts:
            continue
        proof = proof_by_key.get(entry["admission_key"])
        if proof is None:
            errors.append(f"missing proof entry for {entry['admission_key']}")
            continue
        if proof.get("admission_digest") != entry.get("admission_digest"):
            errors.append(f"stale admission digest for {entry['admission_key']}")
        if proof.get("dependency_digest") != entry["dependency_digests"]["entry_dependency_digest"]:
            errors.append(f"stale dependency digest for {entry['admission_key']}")
        if proof.get("cardinality_estimate") != entry.get("cardinality_estimate"):
            errors.append(f"cardinality changed for {entry['admission_key']}")
        if proof.get("exact_row_debt") != (entry["admission_kind"] == "exact_sparse_row"):
            errors.append(f"exact-row debt flag changed for {entry['admission_key']}")
    for key, proof in proof_by_key.items():
        if proof["contract_name"] in covered_contracts and key not in accepted_by_key:
            errors.append(f"proof entry no longer has accepted row {key}")
    for contract, proof_summary in proof_manifest.get("covered_contracts", {}).items():
        accepted_summary = accepted_manifest["summaries"].get(contract)
        if not accepted_summary:
            errors.append(f"covered contract missing from accepted manifest {contract}")
            continue
        for field in (
            "entry_count",
            "cardinality_estimate",
            "exact_sparse_row_count",
            "factorized_group_count",
            "admission_digest",
            "dependency_digest",
        ):
            proof_field = "accepted_entry_count" if field == "entry_count" else field
            if proof_summary.get(proof_field) != accepted_summary.get(field):
                errors.append(
                    f"{contract} proof summary {proof_field} stale: "
                    f"{proof_summary.get(proof_field)!r} != {accepted_summary.get(field)!r}"
                )
    return errors


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Vulkan Contract Admission Delta Report", ""]
    for field, count in report["summary"].items():
        lines.append(f"- {field}: {count}")
    lines.append("")
    if report["newly_admitted_rows"]:
        lines.append("## Newly Admitted Rows")
        for entry in report["newly_admitted_rows"][:80]:
            lines.append(
                f"- `{entry['contract_name']}` `{entry['admission_kind']}` "
                f"`{entry['tuple_id']}` cardinality={entry.get('cardinality_estimate')}"
            )
        if len(report["newly_admitted_rows"]) > 80:
            lines.append(f"- ... {len(report['newly_admitted_rows']) - 80} more")
        lines.append("")
    if report["cardinality_increases"]:
        lines.append("## Cardinality Increases")
        for item in report["cardinality_increases"]:
            lines.append(
                f"- `{item['contract_name']}`: {item['before']} -> {item['after']}"
            )
        lines.append("")
    if report["exact_row_debt_changes"]:
        lines.append("## Exact-Row Debt Changes")
        for item in report["exact_row_debt_changes"]:
            lines.append(
                f"- `{item['contract_name']}`: {item['before']} -> {item['after']}"
            )
        lines.append("")
    if report["dependency_hash_changes"]:
        lines.append("## Dependency Hash Changes")
        for item in report["dependency_hash_changes"][:80]:
            lines.append(f"- `{item['admission_key']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate_accepted_manifest(repo_root: Path, checked_path: Path) -> list[str]:
    checked = load_json(checked_path)
    current = generate_accepted_manifest(repo_root)
    report = compare_manifests(checked, current)
    if compare_has_deltas(report):
        return [json.dumps(report["summary"], sort_keys=True)]
    if checked.get("manifest_digest") != current.get("manifest_digest"):
        return ["manifest digest mismatch without row-level deltas"]
    return []


def self_test() -> None:
    base = {
        "manifest_digest": "base",
        "summaries": {
            "ExampleContract": {
                "cardinality_estimate": 1,
                "exact_sparse_row_count": 1,
            }
        },
        "entries": [
            {
                "admission_key": "ExampleContract|row|a",
                "contract_name": "ExampleContract",
                "admission_kind": "exact_sparse_row",
                "admission_digest": "admit_a",
                "dependency_digests": {"entry_dependency_digest": "dep_a"},
            }
        ],
    }
    current = copy.deepcopy(base)
    current["manifest_digest"] = "current"
    current["entries"].append(
        {
            "admission_key": "ExampleContract|row|b",
            "contract_name": "ExampleContract",
            "admission_kind": "exact_sparse_row",
            "admission_digest": "admit_b",
            "dependency_digests": {"entry_dependency_digest": "dep_b"},
        }
    )
    current["entries"][0]["admission_digest"] = "admit_a2"
    current["entries"][0]["dependency_digests"]["entry_dependency_digest"] = "dep_a2"
    current["summaries"]["ExampleContract"] = {
        "cardinality_estimate": 2,
        "exact_sparse_row_count": 2,
    }
    report = compare_manifests(base, current)
    expected = {
        "newly_admitted_rows": 1,
        "removed_rows": 0,
        "metadata_changes": 1,
        "dependency_hash_changes": 1,
        "cardinality_increases": 1,
        "exact_row_debt_changes": 1,
    }
    if report["summary"] != expected:
        raise AssertionError(report["summary"])

    proof = {
        "covered_contracts": {
            "ExampleContract": {
                "accepted_entry_count": 1,
                "cardinality_estimate": 1,
                "exact_sparse_row_count": 1,
                "factorized_group_count": 0,
                "admission_digest": None,
                "dependency_digest": None,
            }
        },
        "proof_entries": [],
    }
    errors = validate_proof_manifest(current, proof)
    if not errors or "missing proof entry" not in errors[0]:
        raise AssertionError(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--git-ref")
    parser.add_argument("--write-accepted-manifest")
    parser.add_argument("--write-proof-manifest")
    parser.add_argument("--accepted-manifest", default=str(DEFAULT_ACCEPTED_MANIFEST))
    parser.add_argument("--proof-manifest", default=str(DEFAULT_PROOF_MANIFEST))
    parser.add_argument("--baseline-manifest")
    parser.add_argument("--current-manifest")
    parser.add_argument("--validate-accepted-manifest", action="store_true")
    parser.add_argument("--validate-proof-manifest", action="store_true")
    parser.add_argument("--report-json")
    parser.add_argument("--report-md")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    if args.self_test:
        self_test()
        print("validated contract admission compare self-test")
        return

    if args.write_accepted_manifest:
        manifest = generate_accepted_manifest(repo_root, args.git_ref)
        write_json(Path(args.write_accepted_manifest), manifest)
        print(
            "wrote accepted contract manifest "
            f"entries={manifest['entry_count']} digest={manifest['manifest_digest']}"
        )

    if args.write_proof_manifest:
        accepted = load_json(Path(args.accepted_manifest)) if Path(args.accepted_manifest).exists() else generate_accepted_manifest(repo_root)
        proof = generate_proof_manifest(accepted)
        write_json(Path(args.write_proof_manifest), proof)
        print(
            "wrote proof manifest "
            f"entries={len(proof['proof_entries'])} digest={proof['proof_manifest_digest']}"
        )

    if args.validate_accepted_manifest:
        errors = validate_accepted_manifest(repo_root, Path(args.accepted_manifest))
        if errors:
            raise SystemExit("accepted manifest is stale: " + "; ".join(errors))
        print("validated accepted contract row manifest")

    if args.validate_proof_manifest:
        accepted = generate_accepted_manifest(repo_root)
        proof = load_json(Path(args.proof_manifest))
        errors = validate_proof_manifest(accepted, proof)
        if errors:
            raise SystemExit("proof manifest is stale:\n" + "\n".join(errors))
        print(
            "validated proof manifest "
            f"covered_contracts={len(proof.get('covered_contracts', {}))} "
            f"proof_entries={len(proof.get('proof_entries', []))}"
        )

    if args.baseline_manifest or args.current_manifest:
        if not (args.baseline_manifest and args.current_manifest):
            raise SystemExit("--baseline-manifest and --current-manifest are required together")
        base = load_json(Path(args.baseline_manifest))
        current = load_json(Path(args.current_manifest))
        report = compare_manifests(base, current)
        if args.report_json:
            write_json(Path(args.report_json), report)
        if args.report_md:
            Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report_md).write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
