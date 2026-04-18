# DeepDesktop Vulkan Wheel Scripts

This directory contains the PyTorch fork scripts used to prepare and build
DeepDesktop Vulkan wheels.

## Layout

- `setup-vulkan-wheel-build.ps1`
  Windows setup entrypoint. Resolves the Vulkan SDK, optionally installs
  Python `3.10`, `3.12`, and `3.14` with `winget`, and prepares the matching
  build venvs under `.build-venvs/`.
- `setup-vulkan-wheel-build.cmd`
  Thin Windows wrapper for `setup-vulkan-wheel-build.ps1`.
- `setup-vulkan-wheel-build.sh`
  Linux/macOS setup entrypoint. Resolves the Vulkan SDK and prepares build
  venvs from Python interpreters you provide. It does not install Python for
  you.
- `build-vulkan-wheel.ps1`
  Windows wheel build entrypoint. Reuses or creates a build venv, configures
  the Vulkan environment, then runs `python -m build --wheel --no-isolation`.
- `build-vulkan-wheel.cmd`
  Thin Windows wrapper for `build-vulkan-wheel.ps1`.
- `build-vulkan-wheel.sh`
  Linux/macOS wheel build entrypoint with the same role as the PowerShell
  script.
- `windows/vs2022-cmake.cmd`
  Runs `cmake` inside a Visual Studio 2022 developer environment.
- `windows/vs2022-python.cmd`
  Runs the repo `.venv` Python inside a Visual Studio 2022 developer
  environment.
- `windows/vs18-python.cmd`
  Compatibility alias that forwards to `windows/vs2022-python.cmd`.
- `windows/restart-torch-python-build.cmd`
  Resumes the `torch_python` target build with output appended to
  `.build-logs/windows/torch_python-build-seq.log`.
- `windows/resume-torch-python-build.ps1`
  PowerShell version of the same resume helper.

## Common Directories

- `.build-venvs/`
  One build venv per Python minor version, for example `py310`, `py312`,
  `py314`.
- `dist-vulkan/`
  Default wheel output root.
- `.build-logs/windows/`
  Windows helper log output.

## Windows Setup

Prepare the default Python versions and the Vulkan SDK:

```powershell
.\scripts\deepdesktop\setup-vulkan-wheel-build.cmd `
  -VulkanSdk "<VULKAN_SDK_ROOT>"
```

Useful flags:

- `-PythonVersions 3.12,3.14`
  Only prepare specific versions.
- `-CleanVenv`
  Recreate existing `.build-venvs`.
- `-DryRun`
  Print the planned actions without changing anything.
- `-SkipPythonInstall`
  Fail if a requested Python version is missing instead of calling `winget`.
- `-SkipVulkanSdkEnv`
  Do not persist the resolved `VULKAN_SDK` to the user environment.

Notes:

- The script accepts either the actual SDK folder that contains `Bin\glslc.exe`
  or a parent folder that contains versioned SDK subdirectories.
- Python `3.14` may be installed and usable even if `py -3.14` does not
  enumerate it. The script also checks the standard install locations.

## Linux/macOS Setup

Prepare build venvs from existing interpreters:

```bash
./scripts/deepdesktop/setup-vulkan-wheel-build.sh \
  --vulkan-sdk /path/to/VulkanSDK \
  --python python3.10 \
  --python python3.12 \
  --python python3.14
```

Useful flags:

- `--clean-venv`
  Recreate matching `.build-venvs`.
- `--dry-run`
  Validate and print the planned actions without changing anything.

Notes:

- This script does not install Python. Use system package management or your
  own Python install method first.
- It prints the `export VULKAN_SDK=...` line you should use in your shell.

## Windows Wheel Build

Build a wheel for one Python version:

```powershell
.\scripts\deepdesktop\build-vulkan-wheel.cmd `
  -PythonExe "<PYTHON_3_12_EXE>" `
  -VulkanSdk "<VULKAN_SDK_ROOT>" `
  -OutDir "dist-vulkan\py312"
```

Useful flags:

- `-VenvDir PATH`
  Override the default build venv location.
- `-Clean`
  Delete `build/` and the chosen output directory first.
- `-CleanVenv`
  Recreate the build venv before bootstrapping dependencies.
- `-DryRun`
  Validate the environment without building.
- `-NoVenv`
  Use the base Python environment directly instead of `.build-venvs/pyXY`.
- `-DisableFp16`
  Set `USE_VULKAN_FP16_INFERENCE=0`.
- `-RelaxedPrecision`
  Set `USE_VULKAN_RELAXED_PRECISION=1`.
- `-BuildVersion VER`
  Set `PYTORCH_BUILD_VERSION`.
- `-BuildNumber N`
  Set `PYTORCH_BUILD_NUMBER`.
- `-MaxJobs N`
  Override compile parallelism.

What the script sets for the build:

- `USE_VULKAN=1`
- `USE_CUDA=0`
- `USE_ROCM=0`
- `USE_DISTRIBUTED=0`
- `BUILD_TEST=0`
- `BUILD_BINARY=0`
- `CMAKE_GENERATOR=Ninja`

The script reuses an existing build venv if it already contains the required
build packages. It does not reinstall just because the internal state file is
missing.

## Linux/macOS Wheel Build

Build a wheel from an existing interpreter:

```bash
./scripts/deepdesktop/build-vulkan-wheel.sh \
  --python python3.12 \
  --vulkan-sdk /path/to/VulkanSDK \
  --out-dir dist-vulkan/py312
```

Useful flags match the Windows build script closely:

- `--venv-dir PATH`
- `--clean`
- `--clean-venv`
- `--dry-run`
- `--no-venv`
- `--no-fp16`
- `--relaxed-precision`
- `--build-version VER`
- `--build-number N`
- `--max-jobs N`

## Windows Helper Scripts

These are not the main DeepDesktop wheel entrypoints, but they remain useful
for lower-level Windows work inside the fork:

- `windows/vs2022-cmake.cmd --version`
  Quick way to run `cmake` after Visual Studio environment setup.
- `windows/vs2022-python.cmd -c "import sys; print(sys.executable)"`
  Runs the repo `.venv` Python with the Visual Studio build environment
  initialized.
- `windows/restart-torch-python-build.cmd`
  Resume the `torch_python` target and append logs to
  `.build-logs/windows/torch_python-build-seq.log`.
- `windows/resume-torch-python-build.ps1`
  Same goal as the `.cmd` version, but callable from PowerShell workflows.

## Recommended Flow

Windows:

1. Run `setup-vulkan-wheel-build.cmd`.
2. Dry-run `build-vulkan-wheel.cmd` for each Python version.
3. Run the real build for each version into separate output folders.

Linux/macOS:

1. Install the Python versions you want to target.
2. Run `setup-vulkan-wheel-build.sh`.
3. Dry-run `build-vulkan-wheel.sh`.
4. Run the real build for each version into separate output folders.
