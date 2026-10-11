"""Check that the built wheel ships the library and nothing else.

The package claims (`pyproject.toml`) that a distribution contains only `capacity_lens/` plus the licence:
`experiments/` is the evidence and `reference/` is the documentation, and neither belongs in something a user
installs. That claim is checkable, so it is checked -- here and in CI -- rather than trusted.

Usage:
    python tools/check_wheel.py                 # auto-detect exactly one wheel under dist/
    python tools/check_wheel.py path/to/x.whl   # check a specific wheel
    python tools/check_wheel.py --selftest      # prove the checker itself can fail
"""

import argparse
import glob
import io
import os
import sys
import zipfile

FORBIDDEN_PREFIXES = ("experiments/", "reference/", "tests/", "tools/", "examples/", ".github/")
REQUIRED_SUFFIX = "capacity_lens/__init__.py"


def check(wheel):
    """Returns (ok, message). `wheel` is a path or any file-like object; the function needs no writing."""
    names = zipfile.ZipFile(wheel).namelist()
    label = wheel if isinstance(wheel, str) else "<in-memory archive>"
    leaked = [n for n in names if n.startswith(FORBIDDEN_PREFIXES)]
    if leaked:
        return False, f"{label}: ships {len(leaked)} non-library entries, e.g. {leaked[:3]}"
    if not any(n.endswith(REQUIRED_SUFFIX) for n in names):
        return False, f"{label}: does not contain {REQUIRED_SUFFIX} (not a capacity-lens wheel?)"
    licence = [n for n in names if os.path.basename(n).startswith(("LICENSE", "COPYING"))]
    return True, (f"{label}: {len(names)} entries, library only "
                  f"(licence present: {bool(licence)})")


def _archive(entries):
    """Build a zip in memory, so the selftest never touches the filesystem (it must run in confined sandboxes)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name in entries:
            z.writestr(name, "")
    buffer.seek(0)
    return buffer


def run_selftest():
    """The checker must be able to say no: three synthetic archives, two of which must be rejected."""
    ok = []
    good = _archive(["capacity_lens/__init__.py", "capacity_lens/preflight.py",
                     "capacity_lens-0.6.1.dist-info/METADATA",
                     "capacity_lens-0.6.1.dist-info/licenses/LICENSE"])
    passed, _ = check(good)
    ok.append(("a library-only wheel is accepted", passed))
    bad = _archive(["capacity_lens/__init__.py", "experiments/mnist_demo.py"])
    passed, _ = check(bad)
    ok.append(("a wheel that leaks experiments/ is rejected", not passed))
    empty = _archive(["something_else/__init__.py"])
    passed, _ = check(empty)
    ok.append(("an unrelated wheel is rejected", not passed))
    print("[check_wheel selftest]")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="check that a built wheel ships only the library")
    ap.add_argument("wheel", nargs="?", help="path to a .whl (default: exactly one under dist/)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return 0 if run_selftest() else 1

    candidates = [args.wheel] if args.wheel else sorted(glob.glob("dist/*.whl"))
    if not candidates:
        print("no wheel found: build one first (python -m pip wheel . --no-deps -w dist/)")
        return 1
    if len(candidates) > 1:
        print(f"more than one wheel found, pass one explicitly: {candidates}")
        return 1
    ok, message = check(candidates[0])
    print(("[check_wheel] OK   " if ok else "[check_wheel] FAIL ") + message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
