#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_max_jobs="4"
python_exe=""
vulkan_sdk="${VULKAN_SDK:-}"
out_dir="$repo_root/dist-vulkan"
venv_dir=""
max_jobs="${MAX_JOBS:-}"
clean=0
clean_venv=0
dry_run=0
no_venv=0
fp16=1
relaxed_precision=0
build_version=""
build_number="1"
build_python=""
tools_dir=""
venv_status="disabled"
pending_bootstrap=0
pending_create=0

usage() {
  cat <<'EOF'
Usage: ./scripts/deepdesktop/build-vulkan-wheel.sh [options]

Options:
  --python PATH           Base Python interpreter to use
  --vulkan-sdk PATH       Vulkan SDK root (must contain bin/glslc or a versioned child)
  --out-dir PATH          Wheel output directory (default: ./dist-vulkan)
  --venv-dir PATH         Build venv directory (default: ./.build-venvs/pyXY)
  --max-jobs N            Compile parallelism (default: min(CPU count - 1, 4))
  --clean                 Delete ./build and output directory before building
  --clean-venv            Recreate the build venv before installing dependencies
  --dry-run               Validate paths and show the planned build without building
  --no-venv               Use the base Python environment directly
  --no-fp16               Disable USE_VULKAN_FP16_INFERENCE
  --relaxed-precision     Enable USE_VULKAN_RELAXED_PRECISION
  --build-version VER     Override wheel version via PYTORCH_BUILD_VERSION
  --build-number N        Build number used with --build-version (default: 1)
EOF
}

resolve_vulkan_sdk() {
  local candidate="$1"

  if [[ -z "$candidate" ]]; then
    return 1
  fi

  if [[ -x "$candidate/bin/glslc" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  local versioned
  versioned="$(find "$candidate" -mindepth 1 -maxdepth 2 -type f -path '*/bin/glslc' | sort -r | head -n1 || true)"
  if [[ -n "$versioned" ]]; then
    dirname "$(dirname "$versioned")"
    return 0
  fi

  return 1
}

resolve_work_path() {
  local candidate="$1"

  if [[ -z "$candidate" ]]; then
    return 1
  fi

  if [[ "$candidate" = /* ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s\n' "$repo_root/$candidate"
  fi
}

hash_file() {
  local path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
    return 0
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
    return 0
  fi

  if command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$path" | awk '{print $NF}'
    return 0
  fi

  echo "No SHA-256 implementation found in PATH." >&2
  return 1
}

test_build_python_environment() {
  local python_path="$1"
  "$python_path" -c 'import build, cmake, ninja, numpy, yaml, requests, packaging, six, typing_extensions' >/dev/null 2>&1
}

ensure_build_python() {
  local base_python="$1"
  local version_tag=""
  local version_string=""
  local resolved_venv_dir=""
  local venv_python=""
  local state_path=""
  local requirements_path="$repo_root/requirements-build.txt"
  local requirements_hash=""
  local state_version="1"
  local current_state=""
  local existing_state=""

  if [[ ! -f "$requirements_path" ]]; then
    echo "Unable to locate $requirements_path" >&2
    exit 1
  fi

  version_tag="$("$base_python" -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')"
  version_string="$("$base_python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"

  if [[ "$no_venv" -eq 1 ]]; then
    "$base_python" -m build --version >/dev/null
    build_python="$base_python"
    tools_dir="$(dirname "$base_python")"
    venv_status="disabled"
    pending_bootstrap=0
    pending_create=0
    printf '%s\n' "$version_string"
    return 0
  fi

  if [[ -n "$venv_dir" ]]; then
    resolved_venv_dir="$(resolve_work_path "$venv_dir")"
  else
    resolved_venv_dir="$repo_root/.build-venvs/$version_tag"
  fi

  venv_python="$resolved_venv_dir/bin/python"
  tools_dir="$resolved_venv_dir/bin"
  state_path="$resolved_venv_dir/.deepdesktop-build-state"
  requirements_hash="$(hash_file "$requirements_path")"
  current_state="$(printf 'state_version=%s\nbase_python=%s\nrequirements_hash=%s' "$state_version" "$base_python" "$requirements_hash")"

  pending_create=0
  pending_bootstrap=0
  if [[ "$clean_venv" -eq 1 || ! -x "$venv_python" ]]; then
    pending_create=1
    pending_bootstrap=1
    venv_status="missing or reset"
  elif [[ ! -f "$state_path" ]]; then
    if test_build_python_environment "$venv_python"; then
      printf '%s' "$current_state" > "$state_path"
      venv_status="reused"
    else
      pending_bootstrap=1
      venv_status="missing state"
    fi
  else
    existing_state="$(cat "$state_path")"
    if [[ "$existing_state" != "$current_state" ]]; then
      pending_bootstrap=1
      venv_status="state mismatch"
    elif ! "$venv_python" -m build --version >/dev/null 2>&1; then
      pending_bootstrap=1
      venv_status="missing build frontend"
    else
      venv_status="reused"
    fi
  fi

  if [[ "$dry_run" -eq 1 ]]; then
    build_python="$venv_python"
    venv_dir="$resolved_venv_dir"
    printf '%s\n' "$version_string"
    return 0
  fi

  if [[ "$clean_venv" -eq 1 && -d "$resolved_venv_dir" ]]; then
    rm -rf "$resolved_venv_dir"
  fi

  if [[ ! -x "$venv_python" ]]; then
    mkdir -p "$resolved_venv_dir"
    "$base_python" -m venv "$resolved_venv_dir"
  fi

  if [[ "$pending_bootstrap" -eq 1 ]]; then
    "$venv_python" -m pip install --upgrade -r "$requirements_path" build wheel
    "$venv_python" -m build --version >/dev/null
    printf '%s' "$current_state" > "$state_path"
    venv_status="bootstrapped"
  fi

  build_python="$venv_python"
  venv_dir="$resolved_venv_dir"
  printf '%s\n' "$version_string"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      python_exe="$2"
      shift 2
      ;;
    --vulkan-sdk)
      vulkan_sdk="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --venv-dir)
      venv_dir="$2"
      shift 2
      ;;
    --max-jobs)
      max_jobs="$2"
      shift 2
      ;;
    --clean)
      clean=1
      shift
      ;;
    --clean-venv)
      clean_venv=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --no-venv)
      no_venv=1
      shift
      ;;
    --no-fp16)
      fp16=0
      shift
      ;;
    --relaxed-precision)
      relaxed_precision=1
      shift
      ;;
    --build-version)
      build_version="$2"
      shift 2
      ;;
    --build-number)
      build_number="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$python_exe" ]]; then
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_exe="$repo_root/.venv/bin/python"
  else
    python_exe="${PYTHON:-python3}"
  fi
fi

if [[ -x "$python_exe" ]]; then
  python_exe="$(cd "$(dirname "$python_exe")" && pwd)/$(basename "$python_exe")"
else
  python_exe="$(command -v "$python_exe" || true)"
fi

if [[ -z "$python_exe" || ! -x "$python_exe" ]]; then
  echo "Python interpreter not found." >&2
  exit 1
fi

python_version="$(ensure_build_python "$python_exe")"

if [[ "$dry_run" -eq 0 || ( "$pending_bootstrap" -eq 0 && "$pending_create" -eq 0 ) ]]; then
  if [[ ! -x "$tools_dir/cmake" ]] && ! command -v cmake >/dev/null 2>&1; then
    echo "cmake was not found for the selected build environment." >&2
    exit 1
  fi

  if [[ ! -x "$tools_dir/ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
    echo "ninja was not found for the selected build environment." >&2
    exit 1
  fi
fi

resolved_vulkan_sdk="$(resolve_vulkan_sdk "$vulkan_sdk" || true)"
if [[ -z "$resolved_vulkan_sdk" ]]; then
  echo "Unable to resolve a Vulkan SDK root containing bin/glslc. Pass --vulkan-sdk." >&2
  exit 1
fi

if [[ -z "$max_jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    max_jobs="$(nproc)"
  else
    max_jobs="$default_max_jobs"
  fi

  if [[ "$max_jobs" -gt 1 ]]; then
    max_jobs="$((max_jobs - 1))"
  fi

  if [[ "$max_jobs" -gt "$default_max_jobs" ]]; then
    max_jobs="$default_max_jobs"
  fi
fi

if [[ "$clean" -eq 1 ]]; then
  rm -rf "$repo_root/build" "$out_dir"
fi

mkdir -p "$out_dir"

export VULKAN_SDK="$resolved_vulkan_sdk"
export PATH="$tools_dir:$resolved_vulkan_sdk/bin:$PATH"
export CMAKE_GENERATOR="Ninja"
export CMAKE_BUILD_TYPE="Release"
export CMAKE_BUILD_PARALLEL_LEVEL="$max_jobs"
export MAX_JOBS="$max_jobs"
export USE_VULKAN="1"
export USE_VULKAN_FP16_INFERENCE="$fp16"
export USE_VULKAN_RELAXED_PRECISION="$relaxed_precision"
export USE_CUDA="0"
export USE_ROCM="0"
export USE_DISTRIBUTED="0"
export USE_GLOO="0"
export USE_MPI="0"
export USE_TENSORPIPE="0"
export USE_XPU="0"
export BUILD_TEST="0"
export BUILD_BINARY="0"

if [[ -n "$build_version" ]]; then
  export PYTORCH_BUILD_VERSION="$build_version"
  export PYTORCH_BUILD_NUMBER="$build_number"
fi

echo "Repo root     : $repo_root"
echo "Base Python   : $python_exe"
echo "Build Python  : $build_python"
echo "Python ver.   : $python_version"
if [[ "$no_venv" -eq 1 ]]; then
  echo "Build venv    : disabled"
elif [[ "$dry_run" -eq 1 && ( "$pending_bootstrap" -eq 1 || "$pending_create" -eq 1 ) ]]; then
  echo "Build venv    : $venv_dir"
  echo "Venv status   : would prepare ($venv_status)"
else
  echo "Build venv    : $venv_dir"
  echo "Venv status   : $venv_status"
fi
echo "Vulkan SDK    : $resolved_vulkan_sdk"
echo "Output dir    : $out_dir"
echo "Max jobs      : $max_jobs"
echo "CMake jobs    : $CMAKE_BUILD_PARALLEL_LEVEL"
echo "FP16 shaders  : $fp16"
echo "Relaxed prec. : $relaxed_precision"

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run only. Wheel build was not executed."
  exit 0
fi

"$build_python" -m build --wheel --no-isolation --outdir "$out_dir"

latest_wheel="$(find "$out_dir" -maxdepth 1 -type f -name '*.whl' | sort | tail -n1 || true)"
if [[ -z "$latest_wheel" ]]; then
  echo "Build completed without producing a wheel." >&2
  exit 1
fi

echo "Wheel generated: $latest_wheel"
