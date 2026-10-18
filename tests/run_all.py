"""Dependency-free test runner: `python tests/run_all.py`.

pytest is not installed in every environment this project runs in (the verified GPU environment has torch
but no pytest), so the suite is plain `test_*` functions with asserts, runnable two ways:

    pytest tests/            # when pytest is available (recommended, gives -k/-x and reporting)
    python tests/run_all.py  # no dependencies beyond the project's own requirements

Each test module is imported, every module-level `test_*` callable is executed, and a summary is printed.
An optional substring filter is supported: `python tests/run_all.py capacity`.
"""

import importlib
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "experiments"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODULES = [
    "test_capacity",
    "test_bottleneck",
    "test_preflight",
    "test_quant_distill",
    "test_training",
    "test_geometry",
    "test_compress_advisor",
    "test_experiment_criteria",
]


def main(pattern=None):
    passed, failed, errors = 0, [], []
    for name in MODULES:
        if pattern and pattern not in name:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            errors.append((name, "<import>", traceback.format_exc()))
            continue
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            label = f"{name}::{attr}"
            try:
                fn()
                passed += 1
                print(f"  PASS  {label}")
            except Exception as exc:                      # noqa: BLE001 - the runner reports, never swallows
                failed.append((label, exc))
                print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
    print()
    for name, where, tb in errors:
        print(f"  ERROR importing {name}{where}:\n{tb}")
    print(f"[tests] {passed} passed, {len(failed)} failed, {len(errors)} import errors")
    if failed:
        print("        failed: " + ", ".join(label for label, _ in failed))
    return 0 if (not failed and not errors) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
