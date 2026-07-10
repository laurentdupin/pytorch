from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


EVIDENCE_SCHEMA = "VulkanGraphExportEvidence.v1"
TEMPLATE_STATUS = "template_not_measured"
MEASURED_STATUS = "measured"


class ExternalGraphEvidenceSetupError(RuntimeError):
    pass


def parse_input_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError(f"Invalid input shape {value!r}") from error
    if len(shape) != 4 or any(dim <= 0 for dim in shape):
        raise ValueError(
            "Input shapes must be four positive comma-separated dimensions"
        )
    return shape


def require_external_assets(
    external_root: str | None,
    checkpoint: str | None,
) -> tuple[Path, Path]:
    if not external_root:
        raise ExternalGraphEvidenceSetupError(
            "External model assets are required; pass --external-root"
        )
    if not checkpoint:
        raise ExternalGraphEvidenceSetupError(
            "External checkpoint is required; pass --checkpoint"
        )
    root = Path(external_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise ExternalGraphEvidenceSetupError(
            f"External model root does not exist: {root}"
        )
    if not checkpoint_path.is_file():
        raise ExternalGraphEvidenceSetupError(
            f"External checkpoint does not exist: {checkpoint_path}"
        )
    return root, checkpoint_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def template_payload(artifact_type: str) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "artifact_type": artifact_type,
        "status": TEMPLATE_STATUS,
        "not_measured_reason": "external_assets_and_device_measurement_required",
        "source_git_sha": None,
        "external_assets": {
            "adapter": None,
            "checkpoint": None,
        },
        "cases": [],
    }


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute()
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return False


def validate_evidence_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != EVIDENCE_SCHEMA:
        errors.append("schema must equal VulkanGraphExportEvidence.v1")
    if payload.get("artifact_type") not in {"export_census", "parity"}:
        errors.append("artifact_type must be export_census or parity")
    status = payload.get("status")
    if status not in {TEMPLATE_STATUS, MEASURED_STATUS}:
        errors.append("status must be template_not_measured or measured")
        return errors
    if status == TEMPLATE_STATUS:
        if not payload.get("not_measured_reason"):
            errors.append("template evidence requires not_measured_reason")
        if _contains_absolute_path(payload):
            errors.append("template evidence must not contain absolute paths")
        return errors
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        errors.append("measured evidence requires at least two cases")
        return errors
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"cases[{index}] must be an object")
            continue
        for field in ("name", "input_shape", "timing", "guard"):
            if field not in case:
                errors.append(f"cases[{index}].{field} is required")
    return errors


def write_evidence(path: Path, payload: dict[str, Any]) -> None:
    errors = validate_evidence_payload(payload)
    if errors:
        raise ValueError("Invalid graph evidence: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
