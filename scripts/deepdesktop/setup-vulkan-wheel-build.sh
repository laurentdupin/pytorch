#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
vulkan_sdk="${VULKAN_SDK:-}"
clean_venv=0
dry_run=0
python_inputs=()

usage() {
  cat <<'EOF'
Usage: ./scripts/deepdesktop/setup-vulkan-wheel-build.sh [options]

Options:
  --vulkan-sdk PATH       Vulkan SDK root (must contain bin/glslc or a versioned child)
  --python PATH           Python interpreter or command to bootstrap (repeatable)
  --clean-venv            Recreate matching build venvs before installing dependencies
  --dry-run               Validate paths and print planned actions without changing anything
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

resolve_python() {
  local candidate="$1"

  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
    return 0
  fi

  local resolved
  resolved="$(command -v "$candidate" || true)"
  if [[ -n "$resolved" && -x "$resolved" ]]; then
    printf '%s\n' "$resolved"
    return 0
  fi

  return 1
}

test_build_python_environment() {
  local python_path="$1"
  "$python_path" -c 'import build, cmake, ninja, numpy, yaml, requests, packaging, six, typing_extensions' >/dev/null 2>&1
}

ensure_build_venv() {
  local base_python="$1"
  local version_tag=""
  local version_string=""
  local venv_dir=""
  local venv_python=""
  local state_path=""
  local requirements_path="$repo_root/requirements-build.txt"
  local requirements_hash=""
  local state_version="1"
  local current_state=""
  local existing_state=""
  local state_reason="ready"
  local pending_create=0
  local pending_bootstrap=0

  version_tag="$("$base_python" -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')"
  version_string="$("$base_python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  venv_dir="$repo_root/.build-venvs/$version_tag"
  venv_python="$venv_dir/bin/python"
  state_path="$venv_dir/.deepdesktop-build-state"
  requirements_hash="$(hash_file "$requirements_path")"
  current_state="$(printf 'state_version=%s\nbase_python=%s\nrequirements_hash=%s' "$state_version" "$base_python" "$requirements_hash")"

  if [[ "$clean_venv" -eq 1 || ! -x "$venv_python" ]]; then
    pending_create=1
    pending_bootstrap=1
    state_reason="missing or reset"
  elif [[ ! -f "$state_path" ]]; then
    if test_build_python_environment "$venv_python"; then
      if [[ "$dry_run" -eq 0 ]]; then
        printf '%s' "$current_state" > "$state_path"
      fi
      state_reason="reused"
    else
      pending_bootstrap=1
      state_reason="missing state"
    fi
  else
    existing_state="$(cat "$state_path")"
    if [[ "$existing_state" != "$current_state" ]]; then
      pending_bootstrap=1
      state_reason="state mismatch"
    elif ! "$venv_python" -m build --version >/dev/null 2>&1; then
      pending_bootstrap=1
      state_reason="missing build frontend"
    else
      state_reason="reused"
    fi
  fi

  if [[ "$dry_run" -eq 0 ]]; then
    if [[ "$clean_venv" -eq 1 && -d "$venv_dir" ]]; then
      rm -rf "$venv_dir"
    fi

    if [[ ! -x "$venv_python" ]]; then
      mkdir -p "$venv_dir"
      "$base_python" -m venv "$venv_dir"
    fi

    if [[ "$pending_bootstrap" -eq 1 ]]; then
      "$venv_python" -m pip install --upgrade -r "$requirements_path" build wheel
      "$venv_python" -m build --version >/dev/null
      printf '%s' "$current_state" > "$state_path"
      state_reason="bootstrapped"
    fi
  fi

  printf '%s|%s|%s|%s\n' "$version_string" "$venv_dir" "$state_reason" "$base_python"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vulkan-sdk)
      vulkan_sdk="$2"
      shift 2
      ;;
    --python)
      python_inputs+=("$2")
      shift 2
      ;;
    --clean-venv)
      clean_venv=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
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

if [[ ${#python_inputs[@]} -eq 0 ]]; then
  python_inputs=("python3.10" "python3.12" "python3.14")
fi

resolved_vulkan_sdk="$(resolve_vulkan_sdk "$vulkan_sdk" || true)"
if [[ -z "$resolved_vulkan_sdk" ]]; then
  echo "Unable to resolve a Vulkan SDK root containing bin/glslc. Pass --vulkan-sdk." >&2
  exit 1
fi

echo "Repo root      : $repo_root"
echo "Resolved SDK   : $resolved_vulkan_sdk"
echo "Shell export   : export VULKAN_SDK=\"$resolved_vulkan_sdk\""

for candidate in "${python_inputs[@]}"; do
  resolved_python="$(resolve_python "$candidate" || true)"
  if [[ -z "$resolved_python" ]]; then
    echo
    echo "Python $candidate"
    echo "  Interpreter : missing"
    echo "  Action      : install or expose this interpreter before bootstrapping"
    continue
  fi

  result="$(ensure_build_venv "$resolved_python")"
  version_string="${result%%|*}"
  rest="${result#*|}"
  venv_dir="${rest%%|*}"
  rest="${rest#*|}"
  state_reason="${rest%%|*}"

  echo
  echo "Python $version_string"
  echo "  Interpreter : $resolved_python"
  echo "  Venv dir    : $venv_dir"
  if [[ "$dry_run" -eq 1 && "$state_reason" != "reused" ]]; then
    echo "  Venv step   : would prepare ($state_reason)"
  else
    echo "  Venv step   : $state_reason"
  fi
done

if [[ "$dry_run" -eq 1 ]]; then
  echo
  echo "Dry run only. No venv changes were executed."
fi
