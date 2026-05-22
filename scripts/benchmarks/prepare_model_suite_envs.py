from __future__ import annotations

import argparse
import json
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ENV_ROOT = REPO_ROOT / "agent_space" / "venvs"
DEFAULT_PIP_CACHE = REPO_ROOT / "agent_space" / "pip_cache"


ENV_SPECS: dict[str, dict[str, Any]] = {
    "diffusers": {
        "tasks": ["lotus"],
        "packages": [
            "diffusers==0.34.0",
            "transformers==4.55.0",
            "huggingface_hub==0.34.3",
            "safetensors",
            "Pillow",
        ],
    },
    "diffusers_cpu": {
        "tasks": ["lotus"],
        "packages": [
            "torch",
            "diffusers>=0.38.0",
            "transformers>=5.8.0",
            "huggingface_hub>=1.16.1",
            "accelerate",
            "safetensors",
            "Pillow",
        ],
    },
    "transformers": {
        "tasks": ["hy_mt", "gemma"],
        "packages": [
            "transformers==4.55.0",
            "huggingface_hub==0.34.3",
            "safetensors",
            "sentencepiece",
            "accelerate",
        ],
    },
    "transformers_cpu": {
        "tasks": ["hy_mt", "gemma"],
        "packages": [
            "torch",
            "transformers>=5.8.0",
            "huggingface_hub>=1.16.1",
            "safetensors",
            "sentencepiece",
            "accelerate",
            "protobuf",
            "hf_xet",
        ],
    },
    "paddleocr": {
        "tasks": ["paddleocr"],
        "packages": [
            "paddleocr==3.5.0",
            "transformers>=5.8.0",
            "safetensors",
            "sympy",
        ],
        "no_deps_packages": [
            "torchvision",
        ],
    },
    "paddleocr_cpu": {
        "tasks": ["paddleocr"],
        "packages": [
            "torch",
            "torchvision",
            "paddleocr==3.5.0",
            "transformers>=5.8.0",
            "safetensors",
            "sympy",
        ],
    },
}


def venv_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def run_command(cmd: list[str], *, stream: bool = False) -> subprocess.CompletedProcess[str]:
    if not stream:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    print(f"\n[model-suite-env] running: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    print(f"[model-suite-env] finished with exit code {returncode}", flush=True)
    return subprocess.CompletedProcess(cmd, returncode, "".join(output), None)


def create_env(env_dir: Path) -> None:
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if venv_python(env_dir).exists():
        return
    venv.EnvBuilder(with_pip=True).create(env_dir)


def install_packages(
    env_dir: Path,
    packages: list[str],
    *,
    upgrade_pip: bool,
    no_deps_packages: list[str] | None = None,
) -> dict[str, Any]:
    python = venv_python(env_dir)
    DEFAULT_PIP_CACHE.mkdir(parents=True, exist_ok=True)
    commands = []
    if upgrade_pip:
        commands.append([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    commands.append(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--cache-dir",
            str(DEFAULT_PIP_CACHE),
            *packages,
        ],
    )
    if no_deps_packages:
        commands.append(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--cache-dir",
                str(DEFAULT_PIP_CACHE),
                *no_deps_packages,
            ],
        )
    results = []
    for cmd in commands:
        result = run_command(cmd, stream=True)
        results.append(
            {
                "command": cmd,
                "returncode": result.returncode,
                "output_tail": result.stdout[-4000:],
            }
        )
        if result.returncode != 0:
            break
    return {"ok": all(item["returncode"] == 0 for item in results), "runs": results}


def probe_env(env_dir: Path) -> dict[str, Any]:
    python = venv_python(env_dir)
    if not python.exists():
        return {"exists": False, "python": str(python)}
    code = r"""
import importlib.metadata
import importlib.util
import json
import sys

packages = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "huggingface_hub": "huggingface-hub",
    "paddleocr": "paddleocr",
    "torch_directml": "torch-directml",
}

payload = {
    "python": sys.executable,
    "python_version": sys.version,
    "packages": {},
}

for module_name, package_name in packages.items():
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    payload["packages"][module_name] = {
        "available": importlib.util.find_spec(module_name) is not None,
        "version": version,
    }

print(json.dumps(payload, indent=2, sort_keys=True))
"""
    result = run_command([str(python), "-c", code])
    payload: dict[str, Any] = {
        "exists": True,
        "python": str(python),
        "returncode": result.returncode,
    }
    if result.returncode == 0:
        payload.update(json.loads(result.stdout))
    else:
        payload["output_tail"] = result.stdout[-4000:]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and probe isolated virtual environments for benchmark suite dependencies."
    )
    parser.add_argument(
        "--env",
        action="append",
        choices=[*ENV_SPECS.keys(), "all"],
        default=None,
        help="Environment to operate on. May be repeated.",
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        default=DEFAULT_ENV_ROOT,
        help="Directory that contains benchmark virtual environments.",
    )
    parser.add_argument("--create", action="store_true", help="Create selected virtual environments.")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the selected environment's non-torch benchmark dependencies.",
    )
    parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="Upgrade pip in the selected virtual environment before installing packages.",
    )
    parser.add_argument("--probe", action="store_true", help="Probe selected environments.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "agent_space" / "model_suite_env_probe.json",
        help="JSON output path for environment preparation/probe results.",
    )
    return parser.parse_args()


def selected_envs(raw_envs: list[str]) -> list[str]:
    if "all" in raw_envs:
        return list(ENV_SPECS)
    deduped: list[str] = []
    for name in raw_envs:
        if name not in deduped:
            deduped.append(name)
    return deduped


def main() -> None:
    args = parse_args()
    env_names = selected_envs(args.env or ["all"])
    payload: dict[str, Any] = {
        "env_root": str(args.env_root.resolve()),
        "pip_cache": str(DEFAULT_PIP_CACHE.resolve()),
        "environments": {},
    }

    for name in env_names:
        spec = ENV_SPECS[name]
        env_dir = args.env_root / name
        row: dict[str, Any] = {
            "tasks": spec["tasks"],
            "packages": spec["packages"],
            "no_deps_packages": spec.get("no_deps_packages", []),
            "path": str(env_dir.resolve()),
            "python": str(venv_python(env_dir).resolve()),
        }
        if args.create:
            create_env(env_dir)
            row["created"] = True
        if args.install:
            if not venv_python(env_dir).exists():
                create_env(env_dir)
            row["install"] = install_packages(
                env_dir,
                spec["packages"],
                upgrade_pip=args.upgrade_pip,
                no_deps_packages=spec.get("no_deps_packages"),
            )
        if args.probe:
            row["probe"] = probe_env(env_dir)
        payload["environments"][name] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
