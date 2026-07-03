from __future__ import annotations

import contextlib
import json
import os
import re
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ADMISSION_ENV = "PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG"
OP_HIT_ENV = "PYTORCH_VULKAN_OP_HIT_LOG"


@dataclass
class VulkanModelProbeConfig:
    mode: str
    out_path: Path
    policy_path: Path | None = None
    max_records: int | None = None
    model: dict[str, Any] | None = None


def create_vulkan_model_probe(
    torch_module: Any,
    *,
    mode: str,
    out_path: str | Path | None,
    policy_path: str | Path | None = None,
    max_records: int | None = None,
    model: dict[str, Any] | None = None,
) -> "VulkanModelProbe":
    path = Path(out_path or "agent_space/vulkan_model_probe.probe.jsonl").resolve()
    policy = Path(policy_path).resolve() if policy_path else None
    return VulkanModelProbe(
        torch_module,
        VulkanModelProbeConfig(
            mode=mode,
            out_path=path,
            policy_path=policy,
            max_records=max_records,
            model=model,
        ),
    )


class VulkanModelProbe:
    def __init__(self, torch_module: Any, config: VulkanModelProbeConfig) -> None:
        self.torch = torch_module
        self.config = config
        self.enabled = config.mode != "off"
        self.run_id = f"vulkan_probe_{uuid.uuid4().hex[:12]}"
        self.record_index = 0
        self.performance_valid = True
        self.records_written = 0
        self.total_ops_seen = 0
        self.vulkan_input_ops = 0
        self.tainted_input_ops = 0
        self.cpu_substituted_ops = 0
        self.rewrapped_ops = 0
        self.exception_ops = 0
        self.aborted_exception: dict[str, Any] | None = None
        self._mode: Any = None
        self._previous_env: dict[str, str | None] = {}
        self._taints: dict[int, dict[str, Any]] = {}
        self._policy = self._load_policy(config.policy_path)
        self._candidate_specs = self._load_candidate_specs()
        self._counter_fns = self._resolve_counter_fns()
        self._op_counts: Counter[str] = Counter()
        self._family_counts: Counter[str] = Counter()
        self._action_counts: Counter[str] = Counter()
        self._failure_counts: Counter[str] = Counter()
        self._taint_depth_max = 0

    @property
    def summary_path(self) -> Path:
        return self.config.out_path.with_suffix(".probe_summary.json")

    @property
    def admission_path(self) -> Path:
        return self.config.out_path.with_suffix(".admission.jsonl")

    @property
    def op_hit_path(self) -> Path:
        return self.config.out_path.with_suffix(".op_hits.log")

    def __enter__(self) -> "VulkanModelProbe":
        if not self.enabled:
            return self
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except Exception as exc:
            self.enabled = False
            self.aborted_exception = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "phase": "import_torch_dispatch_mode",
            }
            self.write_summary()
            return self

        self.config.out_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.config.out_path, self.admission_path, self.op_hit_path):
            path.write_text("", encoding="utf-8")
        self._set_env(ADMISSION_ENV, str(self.admission_path))
        self._set_env(OP_HIT_ENV, str(self.op_hit_path))
        owner = self

        class ProbeMode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func: Any,
                types: Any,
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                return owner._dispatch(func, args, kwargs or {})

        self._mode = ProbeMode()
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._mode is not None:
            self._mode.__exit__(exc_type, exc, tb)
        self._restore_env()
        if exc is not None and self.aborted_exception is None:
            self.aborted_exception = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:1000],
                "phase": "context_exit",
            }
        self.write_summary()

    def _set_env(self, key: str, value: str) -> None:
        if key not in self._previous_env:
            self._previous_env[key] = os.environ.get(key)
        os.environ[key] = value

    def _restore_env(self) -> None:
        for key, previous in self._previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def _load_policy(self, path: Path | None) -> dict[str, Any]:
        if path is None:
            return {"rules": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"rules": [], "load_error": f"{type(exc).__name__}: {exc}"}

    def _load_candidate_specs(self) -> list[dict[str, Any]]:
        root = Path(__file__).resolve().parents[2]
        specs: list[dict[str, Any]] = []
        spec_dir = root / "test" / "vulkan_contract_specs"
        for path in sorted(spec_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            specs.append(
                {
                    "contract_name": data.get("contract_name"),
                    "family": data.get("family"),
                    "tuple_id": data.get("tuple_id"),
                    "writer_op": data.get("writer_op"),
                    "route_label": data.get("route_label"),
                    "spec_file": str(path),
                    "shape_envelope": (data.get("shape_envelope") or {}).get("role"),
                }
            )
        return specs

    def _resolve_counter_fns(self) -> dict[str, Any]:
        ops = getattr(getattr(self.torch, "ops", None), "vulkan_prepack", None)
        if ops is None:
            return {}
        return {
            "cpu_fallback_count": getattr(ops, "cpu_fallback_count", None),
            "sync_readback_count": getattr(ops, "sync_readback_count", None),
            "submit_origin_counters": getattr(ops, "submit_origin_counters", None),
        }

    def _counter_value(self, name: str) -> int:
        fn = self._counter_fns.get(name)
        if fn is None:
            return 0
        try:
            return int(fn())
        except Exception:
            return 0

    def _submit_value(self) -> list[int]:
        fn = self._counter_fns.get("submit_origin_counters")
        if fn is None:
            return []
        try:
            return [int(value) for value in fn()]
        except Exception:
            return []

    def _without_probe_reentry(self) -> Any:
        disable = getattr(getattr(self.torch, "_C", None), "_DisableTorchDispatch", None)
        if disable is None:
            return contextlib.nullcontext()
        return disable()

    def _execute_without_probe_reentry(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        with self._without_probe_reentry():
            return func(*args, **kwargs)

    def _snapshot_counters(self) -> dict[str, Any]:
        return {
            "cpu_fallback_count": self._counter_value("cpu_fallback_count"),
            "sync_readback_count": self._counter_value("sync_readback_count"),
            "submit_origin_counters": self._submit_value(),
        }

    def _counter_delta(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        before_submit = before.get("submit_origin_counters") or []
        after_submit = after.get("submit_origin_counters") or []
        submit_names = {
            "total": 0,
            "normal_cmd_submit_frequency": 1,
            "stack_planned_recording_submit": 2,
            "tensor_cpu_readback": 6,
            "host_upload": 7,
            "retire_queue_drain": 9,
        }
        width = min(len(before_submit), len(after_submit))
        submit_delta = {
            name: max(0, after_submit[index] - before_submit[index])
            if index < width
            else 0
            for name, index in submit_names.items()
        }
        return {
            "cpu_fallback_count": max(
                0,
                int(after.get("cpu_fallback_count", 0))
                - int(before.get("cpu_fallback_count", 0)),
            ),
            "sync_readback_count": max(
                0,
                int(after.get("sync_readback_count", 0))
                - int(before.get("sync_readback_count", 0)),
            ),
            "submit_origin_delta": submit_delta,
        }

    def _file_offset(self, path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    def _read_new_text_lines(self, path: Path, offset: int) -> list[str]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                return [line.rstrip("\n") for line in handle if line.strip()]
        except OSError:
            return []

    def _dispatch(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.total_ops_seen += 1
        op = self._op(func)
        input_tensors = flatten_tensors(self.torch, args) + flatten_tensors(
            self.torch, kwargs
        )
        has_vulkan = any(is_vulkan_tensor(self.torch, tensor) for tensor in input_tensors)
        input_taints = self._input_taints(input_tensors)
        should_record = has_vulkan or bool(input_taints)
        if has_vulkan:
            self.vulkan_input_ops += 1
        if input_taints:
            self.tainted_input_ops += 1
        policy = self._policy_for_op(op, args, kwargs, has_vulkan, input_taints)
        preexecute_guard = self._preexecute_cpu_continuation_guard(
            op,
            args,
            kwargs,
            has_vulkan,
        )
        if preexecute_guard is not None:
            policy = preexecute_guard
        if policy["action"] == "record_only":
            result = self._execute_without_probe_reentry(func, args, kwargs)
            if should_record:
                self._record_op(
                    op,
                    args,
                    kwargs,
                    result,
                    input_taints,
                    policy,
                    attempted_vulkan=has_vulkan,
                    skipped_vulkan=False,
                    cpu_continuation_used=False,
                    rewrapped_to_vulkan=False,
                    execution_result="ok",
                    exception=None,
                    before_counters=self._snapshot_counters(),
                    after_counters=self._snapshot_counters(),
                    admission_offset=self._file_offset(self.admission_path),
                    op_hit_offset=self._file_offset(self.op_hit_path),
                )
            return result

        before_counters = self._snapshot_counters() if should_record else {}
        admission_offset = self._file_offset(self.admission_path)
        op_hit_offset = self._file_offset(self.op_hit_path)
        if policy["action"] == "cpu_substitute_then_rewrap_vulkan":
            result, rewrapped = self._cpu_substitute(func, args, kwargs, rewrap=True)
            if should_record:
                self._record_op(
                    op,
                    args,
                    kwargs,
                    result,
                    input_taints,
                    policy,
                    attempted_vulkan=False,
                    skipped_vulkan=True,
                    cpu_continuation_used=True,
                    rewrapped_to_vulkan=rewrapped,
                    execution_result="cpu_substituted",
                    exception=None,
                    before_counters=before_counters,
                    after_counters=self._snapshot_counters(),
                    admission_offset=admission_offset,
                    op_hit_offset=op_hit_offset,
                )
            return result

        try:
            result = self._execute_without_probe_reentry(func, args, kwargs)
        except Exception as exc:
            if self._can_continue_after_exception(exc):
                policy = dict(policy)
                policy["action"] = "cpu_substitute_then_rewrap_vulkan"
                policy["known_bad_reason"] = self._known_bad_reason(exc)
                result, rewrapped = self._cpu_substitute(func, args, kwargs, rewrap=True)
                if should_record:
                    self._record_op(
                        op,
                        args,
                        kwargs,
                        result,
                        input_taints,
                        policy,
                        attempted_vulkan=has_vulkan,
                        skipped_vulkan=False,
                        cpu_continuation_used=True,
                        rewrapped_to_vulkan=rewrapped,
                        execution_result="cpu_substituted",
                        exception=exc,
                        before_counters=before_counters,
                        after_counters=self._snapshot_counters(),
                        admission_offset=admission_offset,
                        op_hit_offset=op_hit_offset,
                    )
                return result
            if should_record:
                self._record_op(
                    op,
                    args,
                    kwargs,
                    None,
                    input_taints,
                    policy,
                    attempted_vulkan=has_vulkan,
                    skipped_vulkan=False,
                    cpu_continuation_used=False,
                    rewrapped_to_vulkan=False,
                    execution_result="exception",
                    exception=exc,
                    before_counters=before_counters,
                    after_counters=self._snapshot_counters(),
                    admission_offset=admission_offset,
                    op_hit_offset=op_hit_offset,
                )
            raise

        if should_record:
            self._record_op(
                op,
                args,
                kwargs,
                result,
                input_taints,
                policy,
                attempted_vulkan=has_vulkan,
                skipped_vulkan=False,
                cpu_continuation_used=False,
                rewrapped_to_vulkan=False,
                execution_result="ok",
                exception=None,
                before_counters=before_counters,
                after_counters=self._snapshot_counters(),
                admission_offset=admission_offset,
                op_hit_offset=op_hit_offset,
            )
        return result

    def _op(self, func: Any) -> dict[str, Any]:
        schema = getattr(func, "_schema", None)
        schema_text = str(schema) if schema is not None else str(func)
        name = schema_text.split("(", 1)[0]
        overload = getattr(func, "_overloadname", "")
        return {
            "name": name,
            "schema": schema_text,
            "overload": str(overload),
            "callsite": self._callsite(),
        }

    def _callsite(self) -> dict[str, Any]:
        stack = traceback.extract_stack(limit=64)
        fallback = None
        for frame in reversed(stack[:-3]):
            path = frame.filename.replace("\\", "/")
            lower = path.lower()
            if "torch/utils/_python_dispatch.py" in lower:
                continue
            if "scripts/benchmarks/vulkan_model_probe.py" in lower:
                continue
            if "/torch/" in lower or "\\torch\\" in lower:
                continue
            item = {
                "file": path,
                "line": int(frame.lineno),
                "function": frame.name,
                "site": f"{path}:{frame.lineno}:{frame.name}",
            }
            if "scripts/benchmarks" not in lower:
                return item
            if fallback is None:
                fallback = item
        return fallback or {"site": "unknown"}

    def _policy_for_op(
        self,
        op: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        has_vulkan: bool,
        input_taints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.config.mode == "record":
            return {"mode": self.config.mode, "action": "execute"}
        if self.config.mode != "continue_cpu_to_vulkan_safe":
            return {"mode": self.config.mode, "action": "execute"}
        op_name = op["name"]
        family = operator_family(op_name)
        for index, rule in enumerate(self._policy.get("rules", [])):
            if rule.get("op") not in (None, op_name):
                continue
            if rule.get("operator_family") not in (None, family):
                continue
            action = rule.get("action", "cpu_substitute_then_rewrap_vulkan")
            return {
                "mode": self.config.mode,
                "action": action,
                "policy_rule_id": str(rule.get("id", f"rule_{index}")),
                "known_bad_reason": str(rule.get("reason", "")),
            }
        return {
            "mode": self.config.mode,
            "action": "execute",
            "input_tainted": bool(input_taints),
            "has_vulkan_input": has_vulkan,
        }

    def _preexecute_cpu_continuation_guard(
        self,
        op: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        has_vulkan: bool,
    ) -> dict[str, Any] | None:
        if not has_vulkan or self.config.mode not in {
            "record",
            "continue_cpu_to_vulkan_safe",
        }:
            return None
        if not self._is_attention_score_softmax(op, args, kwargs):
            return None
        return {
            "mode": self.config.mode,
            "action": "cpu_substitute_then_rewrap_vulkan",
            "policy_rule_id": "probe_guard_attention_score_softmax",
            "known_bad_reason": "attention_score_softmax_preexecute_cpu_continuation",
            "preexecute_guard": True,
            "guarded_op": op["name"],
        }

    def _is_attention_score_softmax(
        self,
        op: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        if op["name"] not in {"aten::_softmax", "aten::softmax.int"}:
            return False
        tensors = flatten_tensors(self.torch, args[:1])
        if not tensors:
            return False
        tensor = tensors[0]
        if not is_vulkan_tensor(self.torch, tensor):
            return False
        rank = int(tensor.dim())
        if rank < 3:
            return False
        dim = kwargs.get("dim")
        if dim is None and len(args) > 1:
            dim = args[1]
        try:
            dim = int(dim)
        except Exception:
            return False
        if dim < 0:
            dim += rank
        shape = [int(value) for value in tensor.shape]
        return dim == rank - 1 and shape[-1] > 1 and shape[-1] == shape[-2]

    def _can_continue_after_exception(self, exc: Exception) -> bool:
        if self.config.mode != "continue_cpu_to_vulkan_safe":
            return False
        text = str(exc)
        if "VK_ERROR_DEVICE_LOST" in text or "DeviceLost" in text:
            return False
        if "Vulkan failure" in text and "failure_class=RouteHardFail" in text:
            return True
        if "Vulkan" in text and "not implemented" in text.lower():
            return True
        return False

    def _known_bad_reason(self, exc: Exception) -> str:
        text = str(exc)
        reason = re.search(r"reason=([^ ]+)", text)
        if reason:
            return reason.group(1)
        failure = re.search(r"failure_class=([^ ]+)", text)
        if failure:
            return failure.group(1)
        return "vulkan_exception_cpu_continuation"

    def _cpu_substitute(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        rewrap: bool,
    ) -> tuple[Any, bool]:
        with self._without_probe_reentry():
            cpu_args = map_tensors(self.torch, args, lambda tensor: tensor_to_cpu(tensor))
            cpu_kwargs = map_tensors(
                self.torch,
                kwargs,
                lambda tensor: tensor_to_cpu(tensor),
            )
            cpu_result = func(*cpu_args, **cpu_kwargs)
        if not rewrap:
            return cpu_result, False
        rewrapped_any = False

        def rewrap_tensor(tensor: Any) -> Any:
            nonlocal rewrapped_any
            try:
                if str(tensor.device) == "cpu":
                    out = tensor.to("vulkan")
                    rewrapped_any = rewrapped_any or is_vulkan_tensor(self.torch, out)
                    return out
            except Exception:
                return tensor
            return tensor

        with self._without_probe_reentry():
            return map_tensors(self.torch, cpu_result, rewrap_tensor), rewrapped_any

    def _input_taints(self, tensors: list[Any]) -> list[dict[str, Any]]:
        taints: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tensor in tensors:
            taint = self._taints.get(id(tensor))
            if taint is None:
                continue
            for taint_id in taint.get("taint_ids", []):
                if taint_id in seen:
                    continue
                seen.add(taint_id)
                taints.append(taint)
        return taints

    def _mark_outputs_tainted(
        self,
        result: Any,
        *,
        input_taints: list[dict[str, Any]],
        reason: str,
        force_new_taint: bool,
    ) -> tuple[list[str], int]:
        tensors = flatten_tensors(self.torch, result)
        if not tensors:
            return [], 0
        inherited_ids: list[str] = []
        max_depth = 0
        for taint in input_taints:
            inherited_ids.extend(str(item) for item in taint.get("taint_ids", []))
            max_depth = max(max_depth, int(taint.get("continuation_depth", 0)))
        if force_new_taint or not inherited_ids:
            inherited_ids.append(f"{self.run_id}_taint_{self.record_index}")
        inherited_ids = sorted(set(inherited_ids))
        depth = max_depth + 1
        self._taint_depth_max = max(self._taint_depth_max, depth)
        taint = {
            "taint_ids": inherited_ids,
            "taint_reason": reason,
            "continuation_depth": depth,
        }
        for tensor in tensors:
            self._taints[id(tensor)] = taint
        return inherited_ids, depth

    def _record_op(
        self,
        op: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        input_taints: list[dict[str, Any]],
        policy: dict[str, Any],
        *,
        attempted_vulkan: bool,
        skipped_vulkan: bool,
        cpu_continuation_used: bool,
        rewrapped_to_vulkan: bool,
        execution_result: str,
        exception: Exception | None,
        before_counters: dict[str, Any],
        after_counters: dict[str, Any],
        admission_offset: int,
        op_hit_offset: int,
    ) -> None:
        if self.config.max_records is not None and self.records_written >= int(
            self.config.max_records
        ):
            return
        input_taint_ids = sorted(
            {
                str(taint_id)
                for taint in input_taints
                for taint_id in taint.get("taint_ids", [])
            }
        )
        output_taint_ids: list[str] = []
        depth = max(
            [int(taint.get("continuation_depth", 0)) for taint in input_taints]
            or [0]
        )
        output_tainted = bool(input_taints)
        taint_reason = "propagated_cpu_substitution" if input_taints else ""
        if cpu_continuation_used:
            output_tainted = True
            taint_reason = "cpu_substitution"
            output_taint_ids, depth = self._mark_outputs_tainted(
                result,
                input_taints=input_taints,
                reason=taint_reason,
                force_new_taint=True,
            )
        elif input_taints:
            output_taint_ids, depth = self._mark_outputs_tainted(
                result,
                input_taints=input_taints,
                reason=taint_reason,
                force_new_taint=False,
            )
        self.performance_valid = self.performance_valid and not output_tainted
        if cpu_continuation_used:
            self.cpu_substituted_ops += 1
        if rewrapped_to_vulkan:
            self.rewrapped_ops += 1
        if execution_result == "exception":
            self.exception_ops += 1
        op_name = op["name"]
        family = operator_family(op_name)
        self._op_counts[op_name] += 1
        self._family_counts[family] += 1
        self._action_counts[str(policy.get("action", "execute"))] += 1
        parsed_exception = parse_vulkan_exception(exception)
        if parsed_exception.get("failure_class"):
            self._failure_counts[str(parsed_exception["failure_class"])] += 1
        inputs = tensor_summaries(self.torch, args, self._taints, prefix="args")
        inputs.extend(tensor_summaries(self.torch, kwargs, self._taints, prefix="kwargs"))
        outputs = tensor_summaries(self.torch, result, self._taints, prefix="outputs")
        admission_records = self._read_admission_records(admission_offset)
        op_hits = self._read_op_hits(op_hit_offset)
        first_tensor = first_tensor_summary(inputs)
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "record_index": self.record_index,
            "event": "vulkan_model_probe_op",
            "model": dict(self.config.model or {}),
            "op": op,
            "inputs": inputs,
            "outputs": outputs,
            "attrs": summarize_attrs(args, kwargs),
            "candidate_contracts": self._candidate_contracts_for_op(op_name, op_hits),
            "admission": {
                "observed": bool(admission_records),
                "records": admission_records,
            },
            "op_hits": op_hits,
            "counters_delta": self._counter_delta(before_counters, after_counters),
            "probe_policy": policy,
            "execution": {
                "attempted_vulkan": bool(attempted_vulkan),
                "skipped_vulkan": bool(skipped_vulkan),
                "cpu_continuation_used": bool(cpu_continuation_used),
                "rewrapped_to_vulkan": bool(rewrapped_to_vulkan),
                "result": execution_result,
                "exception_type": type(exception).__name__ if exception else None,
                "exception_message": str(exception)[:1000] if exception else None,
                "performance_valid": False if output_tainted else self.performance_valid,
            },
            "taint": {
                "input_tainted": bool(input_taints),
                "input_taint_ids": input_taint_ids,
                "output_tainted": bool(output_tainted),
                "output_taint_ids": output_taint_ids,
                "taint_reason": taint_reason,
                "continuation_depth": int(depth),
            },
            "classification": {
                "operator_family": family,
                "dtype": first_tensor.get("dtype", "unknown"),
                "rank": first_tensor.get("rank", "unknown"),
                "layout_storage": first_tensor.get("layout", "unknown"),
                "shape_envelope": self._shape_envelope_for_op(op_name),
                "execution_phase": execution_phase(op["callsite"]),
                "failure_class": parsed_exception.get("failure_class"),
                "route_reject_reason": parsed_exception.get("reason"),
                "reusability": reusability_for(family, parsed_exception),
            },
        }
        self.config.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.records_written += 1
        self.record_index += 1

    def _read_admission_records(self, offset: int) -> list[dict[str, Any]]:
        records = []
        for line in self._read_new_text_lines(self.admission_path, offset):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"raw": line})
        return records

    def _read_op_hits(self, offset: int) -> list[str]:
        hits = []
        for line in self._read_new_text_lines(self.op_hit_path, offset):
            match = re.search(r"op=([^ ]+)", line)
            hits.append(match.group(1) if match else line)
        return hits

    def _candidate_contracts_for_op(
        self,
        op_name: str,
        op_hits: list[str],
    ) -> list[dict[str, Any]]:
        candidates = []
        family = operator_family(op_name)
        for spec in self._candidate_specs:
            writer_op = spec.get("writer_op")
            route_label = spec.get("route_label")
            if writer_op and (writer_op == op_name or writer_op in op_name):
                item = dict(spec)
                item["match_basis"] = "writer_op"
                candidates.append(item)
                continue
            if route_label and route_label in op_hits:
                item = dict(spec)
                item["match_basis"] = "route_label"
                candidates.append(item)
                continue
            if family and family in str(spec.get("shape_envelope", "")):
                item = dict(spec)
                item["match_basis"] = "shape_envelope_role"
                candidates.append(item)
        return candidates[:8]

    def _shape_envelope_for_op(self, op_name: str) -> str:
        for spec in self._candidate_specs:
            if spec.get("writer_op") == op_name:
                return str(spec.get("shape_envelope") or "")
        return ""

    def write_summary(self) -> None:
        if not self.enabled:
            return
        payload = self.summary()
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": self.config.mode,
            "record_path": str(self.config.out_path),
            "summary_path": str(self.summary_path),
            "admission_log_path": str(self.admission_path),
            "op_hit_log_path": str(self.op_hit_path),
            "policy_path": str(self.config.policy_path) if self.config.policy_path else None,
            "policy_load_error": self._policy.get("load_error"),
            "model": dict(self.config.model or {}),
            "performance_valid": False,
            "untainted_outputs": bool(self.performance_valid),
            "performance_note": (
                "probe mode adds dispatch/logging overhead; do not use timings "
                "from this run as performance data"
            ),
            "total_ops_seen": int(self.total_ops_seen),
            "records_written": int(self.records_written),
            "vulkan_input_ops": int(self.vulkan_input_ops),
            "tainted_input_ops": int(self.tainted_input_ops),
            "cpu_substituted_ops": int(self.cpu_substituted_ops),
            "rewrapped_to_vulkan_ops": int(self.rewrapped_ops),
            "exception_ops": int(self.exception_ops),
            "max_taint_depth": int(self._taint_depth_max),
            "aborted_exception": self.aborted_exception,
            "top_ops": self._op_counts.most_common(20),
            "top_operator_families": self._family_counts.most_common(20),
            "policy_actions": self._action_counts.most_common(),
            "failure_classes": self._failure_counts.most_common(),
        }


def is_vulkan_tensor(torch_module: Any, value: Any) -> bool:
    return torch_module.is_tensor(value) and str(value.device).startswith("vulkan")


def flatten_tensors(torch_module: Any, value: Any) -> list[Any]:
    tensors: list[Any] = []
    if torch_module.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_tensors(torch_module, item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_tensors(torch_module, item))
    return tensors


def map_tensors(torch_module: Any, value: Any, fn: Any) -> Any:
    if torch_module.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(torch_module, item, fn) for item in value)
    if isinstance(value, list):
        return [map_tensors(torch_module, item, fn) for item in value]
    if isinstance(value, dict):
        return {key: map_tensors(torch_module, item, fn) for key, item in value.items()}
    return value


def tensor_to_cpu(tensor: Any) -> Any:
    try:
        if str(tensor.device).startswith("vulkan"):
            return tensor.cpu()
    except Exception:
        return tensor
    return tensor


def tensor_summaries(
    torch_module: Any,
    value: Any,
    taints: dict[int, dict[str, Any]],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    def visit(item: Any, path: str) -> None:
        if torch_module.is_tensor(item):
            summaries.append(tensor_summary(item, path, taints.get(id(item))))
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, prefix)
    return summaries


def tensor_summary(tensor: Any, path: str, taint: dict[str, Any] | None) -> dict[str, Any]:
    out = {
        "arg_path": path,
        "shape": [int(dim) for dim in tensor.shape],
        "rank": int(tensor.dim()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "layout": str(tensor.layout).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "is_vulkan": str(tensor.device).startswith("vulkan"),
        "tainted": taint is not None,
        "taint_ids": list(taint.get("taint_ids", [])) if taint else [],
    }
    try:
        out["stride"] = [int(dim) for dim in tensor.stride()]
    except Exception as exc:
        out["stride_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    try:
        out["storage_offset"] = int(tensor.storage_offset())
    except Exception as exc:
        out["storage_offset_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


def first_tensor_summary(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return inputs[0] if inputs else {}


def summarize_attrs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": [summarize_value(item) for item in args],
        "kwargs": {key: summarize_value(value) for key, value in kwargs.items()},
    }


def summarize_value(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"tensor": True}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [summarize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): summarize_value(item) for key, item in value.items()}
    return repr(value)[:200]


def operator_family(op_name: str) -> str:
    if "scaled_dot_product_attention" in op_name:
        return "attention"
    if "convolution" in op_name or "conv" in op_name:
        return "conv"
    if "grid_sampler" in op_name or "grid_sample" in op_name:
        return "grid_sample"
    if "cat" in op_name:
        return "cat"
    if "view" in op_name or "reshape" in op_name:
        return "view_reshape"
    if any(name in op_name for name in ("addmm", "linear", "mm", "bmm", "matmul")):
        return "linear_mm"
    if any(name in op_name for name in ("add.", "mul.", "sub.", "div.")):
        return "elementwise_broadcast"
    if "batch_norm" in op_name or "native_batch_norm" in op_name:
        return "batch_norm"
    return "unknown"


def execution_phase(callsite: dict[str, Any]) -> str:
    site = str(callsite.get("site", "")).lower()
    if "depth_anything" in site or "dpt.py" in site or "dinov2.py" in site:
        return "depth_vision_model"
    if "transformers" in site:
        return "transformers_model"
    if "paddle" in site:
        return "paddle_model"
    if "benchmark" in site:
        return "benchmark_harness"
    return "unknown"


def parse_vulkan_exception(exc: Exception | None) -> dict[str, Any]:
    if exc is None:
        return {}
    text = str(exc)
    out: dict[str, Any] = {}
    for key in ("failure_class", "op", "reason", "caller"):
        match = re.search(rf"{key}=([^ ]+)", text)
        if match:
            out[key] = match.group(1)
    return out


def reusability_for(family: str, parsed_exception: dict[str, Any]) -> str:
    reason = str(parsed_exception.get("reason", ""))
    if reason.startswith("KnownBad"):
        return "candidate_contract_gap"
    if family != "unknown":
        return "operator_family_probe"
    return "unclassified"


@contextlib.contextmanager
def maybe_vulkan_model_probe(
    torch_module: Any,
    *,
    mode: str,
    out_path: str | Path | None,
    policy_path: str | Path | None = None,
    max_records: int | None = None,
    model: dict[str, Any] | None = None,
) -> Any:
    probe = create_vulkan_model_probe(
        torch_module,
        mode=mode,
        out_path=out_path,
        policy_path=policy_path,
        max_records=max_records,
        model=model,
    )
    with probe:
        yield probe
