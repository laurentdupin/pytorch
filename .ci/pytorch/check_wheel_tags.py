import os
import re
import subprocess
import sys
from pathlib import Path


def check_mac_wheel_minos() -> None:
    """Check that dylib minos matches the wheel platform tag on macOS.

    Parses the platform tag from the .whl filename in PYTORCH_FINAL_PACKAGE_DIR,
    then verifies each installed dylib's minos (from otool -l) matches.
    """
    if sys.platform != "darwin":
        return

    wheel_dir = os.getenv("PYTORCH_FINAL_PACKAGE_DIR", "")
    if not wheel_dir or not os.path.isdir(wheel_dir):
        print("PYTORCH_FINAL_PACKAGE_DIR not set, skipping wheel minos check")
        return

    whls = list(Path(wheel_dir).glob("*.whl"))
    if not whls:
        print(f"No .whl files in {wheel_dir}, skipping wheel minos check")
        return

    import torch

    torch_dir = Path(os.path.dirname(torch.__file__))
    dylibs = list(torch_dir.rglob("*.dylib"))
    if not dylibs:
        print("No .dylib files found, skipping minos check")
        return

    for whl in whls:
        print(f"Checking wheel tag minos for: {whl.name}")

        m = re.search(r"macosx_(\d+)_(\d+)_(\w+)\.whl$", whl.name)
        if not m:
            print(f"No macOS platform tag in {whl.name}, skipping")
            continue

        expected_minos = f"{m.group(1)}.{m.group(2)}"
        print(f"Expected minos from platform tag: {expected_minos}")

        mismatches = []
        for dylib in dylibs:
            try:
                result = subprocess.run(
                    ["otool", "-l", str(dylib)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception:
                continue

            minos = None
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                s = line.strip()
                if "LC_BUILD_VERSION" in s:
                    for j in range(i + 1, min(i + 6, len(lines))):
                        if lines[j].strip().startswith("minos"):
                            minos = lines[j].strip().split()[1]
                            break
                    break
                if "LC_VERSION_MIN_MACOSX" in s:
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if lines[j].strip().startswith("version"):
                            minos = lines[j].strip().split()[1]
                            break
                    break

            if minos and minos != expected_minos:
                mismatches.append(
                    f"{dylib.name}: minos={minos}, expected={expected_minos}"
                )

        if mismatches:
            raise RuntimeError(
                f"minos/platform tag mismatch in {len(mismatches)} dylib(s):\n"
                + "\n".join(f"  {m}" for m in mismatches)
            )
        print(
            f"OK: All {len(dylibs)} dylib(s) have minos matching "
            f"platform tag ({expected_minos})"
        )


if __name__ == "__main__":
    check_mac_wheel_minos()
