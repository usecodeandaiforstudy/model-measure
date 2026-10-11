"""Reproduce every published verdict offline, from the recorded artifacts, in one command.

What this is for. The project's rule is that a verdict must be rebuildable from the recorded jsonl with zero
training ("MIXED", "--verdict-only" in CONTRIBUTING). Until now that was true script by script; this tool does
it for the whole evidence base at once and writes a diffable snapshot, so a reader can check the claims without
a GPU, without the datasets, and without trusting the prose.

How it works, and why this way:
  * It imports each experiment module **in process** and calls its `verdict_only()` when it has one, capturing
    stdout with contextlib.redirect_stdout. It deliberately does NOT spawn subprocesses: under a confined
    sandbox a child process cannot open the pipes that output capture needs, and a tool whose correctness
    depends on the sandbox mode is not a reproduction tool.
  * Modules without `verdict_only()` are classified rather than assumed: the early-generation scripts (see
    CONTRIBUTING, "two generations") are reported as historical, and a module that only exposes callable
    criteria functions is reported as such, with the instruction to run it directly (its main path is offline).
  * Every recorded `<topic>_report.txt` verdict line is printed alongside, so the recomputed text and the
    recorded text can be compared by eye or by `--check`.

Exit code: 0 when every module that could be replayed succeeded; 1 otherwise (so it can gate a release).
Artifacts: tools/reproduce_all_snapshot.json (the machine-readable snapshot, meant to be diffed between runs).

Usage:
  python tools/reproduce_all.py                # replay everything and write the snapshot
  python tools/reproduce_all.py --check        # also compare with the previous snapshot and report changes
  python tools/reproduce_all.py --reports-only # no imports at all: just the recorded verdict lines
  python tools/reproduce_all.py --strict       # treat "historical" and "criteria-only" as failures too
"""

import argparse
import contextlib
import glob
import importlib
import io
import json
import os
import re
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
DATA = os.path.join(EXP, "data")
SNAPSHOT = os.path.join(ROOT, "tools", "reproduce_all_snapshot.json")

# The nine scripts that predate --selftest / --verdict-only / relay mode. They are kept untouched so that their
# recorded artifacts stay coupled to the code that produced them (CONTRIBUTING.md).
HISTORICAL = {
    "cifar_demo.py", "cp3_demo.py", "cp3_dyn.py", "cp3_dyn_sens.py", "cp3_multilevel.py",
    "cph_post.py", "mnist_demo.py", "opt_demo.py", "transformer_demo.py",
}


VERDICT_WORDS = ("SUPPORT", "REFUTED", "MIXED", "UNDECIDABLE", "RESTRICTED", "USABLE", "PARTIAL", "PASS", "FAIL")


def report_verdict_lines(topic):
    """The recorded verdict lines for a module, from its report file.

    Two details learned the hard way here: the report is named after the CAMPAIGN (`scale_axes_report.txt` for
    `scale_axes_demo.py`), so the trailing `_demo`/`_post` has to come off; and the project's report files do not
    follow one format -- several interleave their jsonl rows with the verdict lines -- so this looks for the
    project's own verdict vocabulary rather than the literal word "verdict".
    """
    stem = re.sub(r"_(demo|post)$", "", topic)
    for cand in (f"{stem}_report.txt", f"{stem}s_report.txt", f"{topic}_report.txt"):
        path = os.path.join(DATA, cand)
        if os.path.exists(path):
            break
    else:
        return None
    out = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if len(s) > 3 and not s.startswith("{") and any(w in s for w in VERDICT_WORDS):
            out.append(s)
    return out or None


def call_verdict_only(module):
    """Call module.verdict_only() in process; returns (ok, captured_text, error)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            module.verdict_only()
        return True, buf.getvalue().strip(), None
    except SystemExit as exc:                      # verdict_only() raises SystemExit when nothing is recorded
        return False, buf.getvalue().strip(), f"SystemExit: {exc}"
    except Exception:
        return False, buf.getvalue().strip(), traceback.format_exc(limit=3).strip().splitlines()[-1]


def classify(path):
    """What kind of offline replay does this module support, decided by introspection, not by a hardcoded map."""
    name = os.path.basename(path)
    topic = name[:-3]
    if name in HISTORICAL:
        return topic, "historical", None
    try:
        sys.path.insert(0, EXP)
        sys.path.insert(0, ROOT)
        module = importlib.import_module(topic)
    except Exception:
        return topic, "import-failed", traceback.format_exc(limit=2).strip().splitlines()[-1]
    if callable(getattr(module, "verdict_only", None)):
        return topic, "replayable", module
    criteria = [n for n in dir(module) if n.endswith("_verdict") and callable(getattr(module, n))]
    if criteria:
        return topic, "criteria-only", (module, criteria)
    return topic, "no-offline-entry", None


def main():
    ap = argparse.ArgumentParser(description="reproduce every published verdict offline from the recorded artifacts")
    ap.add_argument("--check", action="store_true", help="diff against the previous snapshot")
    ap.add_argument("--reports-only", action="store_true", help="print the recorded verdict lines only")
    ap.add_argument("--strict", action="store_true", help="count historical/criteria-only modules as failures")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 108)
    print("[REPRODUCE ALL] every verdict rebuilt from experiments/data/*.jsonl -- zero training, no GPU needed")
    print("=" * 108)

    snapshot = dict(tool="tools/reproduce_all.py", when=time.strftime("%Y-%m-%d %H:%M:%S"), modules=[])
    failures = []
    counts = {}

    for path in sorted(glob.glob(os.path.join(EXP, "*.py"))):
        topic, kind, payload = classify(path)
        counts[kind] = counts.get(kind, 0) + 1
        recorded = report_verdict_lines(topic)
        entry = dict(topic=topic, kind=kind, recorded=recorded)
        head = f"  {topic:28s} [{kind}]"

        if args.reports_only:
            print(head)
        elif kind == "replayable":
            ok, text, err = call_verdict_only(payload)
            entry["replayed"] = text or None
            entry["ok"] = bool(ok)
            print(f"{head}\n      replayed: {text.splitlines()[-1].strip() if text else '(no output)'}")
            if not ok:
                failures.append(f"{topic}: {err}")
                print(f"      !! {err}")
        elif kind == "criteria-only":
            module, criteria = payload
            entry["criteria"] = criteria
            print(f"{head} criteria: {', '.join(criteria)} -- run the script directly (its main path is offline)")
        elif kind == "historical":
            print(f"{head} kept untouched by design (CONTRIBUTING: two generations)")
        else:
            entry["error"] = payload if isinstance(payload, str) else None
            print(f"{head} {payload if isinstance(payload, str) else 'no offline entry point'}")
            if kind == "import-failed":
                failures.append(topic)

        if recorded:
            for line in recorded:
                print(f"      recorded: {line[:160]}")
        else:
            print("      recorded: (no verdict line in the report file)")
        snapshot["modules"].append(entry)

    total = sum(counts.values())
    print()
    print(f"[reproduce] {total} modules: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if failures:
        print(f"[reproduce] FAILURES: {failures}")
    else:
        print("[reproduce] every module that has an offline entry point replayed successfully")

    changed = []
    if args.check and os.path.exists(SNAPSHOT):
        old = json.load(open(SNAPSHOT, encoding="utf-8"))
        old_by = {m["topic"]: m.get("replayed") for m in old.get("modules", [])}
        for m in snapshot["modules"]:
            if m["topic"] in old_by and old_by[m["topic"]] != m.get("replayed"):
                changed.append(m["topic"])
        print(f"[reproduce] --check against {os.path.basename(SNAPSHOT)}: "
              + (f"CHANGED verdicts: {changed}" if changed else "no verdict changed since the snapshot"))
    if changed:
        failures.extend(changed)
    os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
    json.dump(snapshot, open(SNAPSHOT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[reproduce] snapshot written: {os.path.relpath(SNAPSHOT, ROOT)} "
          f"({os.path.getsize(SNAPSHOT)} bytes, {time.time() - t0:.1f}s)")
    strict_fail = args.strict and (counts.get("historical") or counts.get("criteria-only") or counts.get("no-offline-entry"))
    return 1 if (failures or strict_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
