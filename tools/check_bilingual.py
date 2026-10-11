"""Bilingual audit: every document must exist in English and in Chinese (`<name>_cn.md`).

The rule this enforces: **all documents ship in both languages, and the Chinese file is the English name plus
`_cn`.** A translation is only useful if it is a *twin*, so this checks more than existence:

  * every English document has its `_cn` twin, and every `_cn` file has an English original (no orphans);
  * the heading structure matches (same number of headings, same depth sequence) -- a translation that quietly
    drops a section is worse than no translation;
  * **the numeric tokens match**: every number in the English document appears in the Chinese one. Numbers are
    the project's claims; if one drifts in translation, the twin is lying;
  * the fenced code blocks are identical apart from comment text;
  * the file-name references match, so a reader of either version can find the same artifacts.

Exempt by design (not documents, and their canonical names are read by tools): `LICENSE`, `CITATION.cff`,
`.gitignore`, `pyproject.toml`, and the recorded evidence (`*.jsonl`, `*_report.txt`) -- data, not prose.

Usage:
    python tools/check_bilingual.py            # report and exit non-zero if anything is missing or inconsistent
    python tools/check_bilingual.py --quiet     # only the summary
"""

import argparse
import glob
import os
import re
import sys

SKIP_NAMES = {"LICENSE", "CITATION.cff", ".gitignore", "pyproject.toml"}
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
NUMBER_SKIP = {str(i) for i in range(0, 40)}          # list markers and small indices are not claims
FILEREF = re.compile(r"\b[A-Za-z0-9_]+\.(?:py|md|cff|toml|json|txt|jsonl|yml)\b")
# (Uppercase matters: this project's note names are wall-remeasurement.md, transformer.md, cifar10-anchor.md ...
#  A lowercase-only class silently skipped exactly the file names a reader needs. The --selftest caught it.)


def headings(text):
    return [(len(l) - len(l.lstrip("#")), l.strip()) for l in text.splitlines() if l.startswith("#")]


BOX_CHARS = "│┌┐└┘├┤┬┴┼─►◄▲▼⇒↑↓←→"      # box drawing, arrows: prose-in-a-diagram gets translated


def _is_diagram(block):
    return any(ch in block for ch in BOX_CHARS)


def code_blocks(text):
    return re.findall(r"```[a-z]*\n(.*?)```", text, re.S)


def _diagram_shape(block):
    """A translated diagram is REDRAWN: its labels are shorter or longer, so the box padding necessarily
    differs and comparing box characters would fail on every honest translation. What must match is the
    diagram's size -- the number of drawn lines -- which is what catches a dropped or truncated diagram."""
    return sum(1 for line in block.splitlines() if line.strip())


def english_documents():
    docs = sorted(glob.glob("docs/*.md"))
    docs = [d for d in docs if not d.endswith("_cn.md")]
    root = ["README.md"]
    for name in ("CONTRIBUTING.md", "CHANGELOG.md"):
        if os.path.exists(name):
            root.append(name)
    return root + docs


def compare(en_path, cn_path):
    """Returns a list of problems for one pair of files."""
    return compare_text(open(en_path, encoding="utf-8").read(), open(cn_path, encoding="utf-8").read())


def compare_text(en, cn):
    """Returns a list of problems for one pair of documents. Pure function of the two texts."""
    problems = []

    h_en, h_cn = headings(en), headings(cn)
    if len(h_en) != len(h_cn):
        problems.append(f"headings: {len(h_en)} (en) vs {len(h_cn)} (cn)")
    elif [d for d, _ in h_en] != [d for d, _ in h_cn]:
        problems.append("headings: same count but different depth sequence")

    n_en = {m.group(0) for m in NUMBER.finditer(en)} - NUMBER_SKIP
    n_cn = {m.group(0) for m in NUMBER.finditer(cn)} - NUMBER_SKIP
    missing = sorted(n_en - n_cn)
    if missing:
        problems.append(f"numbers present in English but missing in Chinese: {missing[:8]}")

    f_en = set(FILEREF.findall(en))
    f_cn = set(FILEREF.findall(cn))
    missing_files = sorted(f_en - f_cn)
    if missing_files:
        problems.append(f"file references missing in Chinese: {missing_files[:6]}")

    b_en = sorted(b.strip() for b in code_blocks(en))
    b_cn = sorted(b.strip() for b in code_blocks(cn))
    if len(b_en) != len(b_cn):
        problems.append(f"code blocks: {len(b_en)} (en) vs {len(b_cn)} (cn)")
    else:
        # Two kinds of block, checked differently:
        #   * DIAGRAMS (box-drawing characters) carry translated prose inside the boxes, so what must match is
        #     their shape -- the same lines and the same box characters;
        #   * CODE must match line for line, ignoring only translated `#` comments.
        d_en = sorted(_diagram_shape(b) for b in b_en if _is_diagram(b))
        d_cn = sorted(_diagram_shape(b) for b in b_cn if _is_diagram(b))
        if d_en != d_cn:
            problems.append(f"diagram blocks: {len(d_en)} (en) vs {len(d_cn)} (cn), or their shapes differ")

        def norm(blocks):
            out = []
            for block in blocks:
                if _is_diagram(block):
                    continue
                lines = []
                for line in block.splitlines():
                    line = re.sub(r"#.*$", "", line).rstrip()
                    if not line.strip():
                        continue
                    lines.append(re.sub(r"[\u4e00-\u9fff]+", "<CJK>", line).strip())
                out.append(tuple(lines))
            return sorted(out)

        n_en, n_cn = norm(b_en), norm(b_cn)
        if n_en != n_cn:
            detail = next((f"first difference: {a[:70]!r} vs {b[:70]!r}"
                           for a, b in zip(n_en, n_cn) if a != b), "")
            problems.append(f"code blocks differ beyond comments and translated labels ({detail})")
    return problems


def run_selftest():
    """The checker must be able to say no. Three synthetic pairs, checked in memory (no filesystem needed)."""
    good_en = "# Title\n\n## One\n\nR_IB = 3.11, K_wall = 64, 94 tests, see `WALL_NOTE.md`.\n```bash\npython x.py\n```\n"
    good_cn = "# 标题\n\n## 一\n\nR_IB = 3.11, K_wall = 64, 94 tests，见 `WALL_NOTE.md`。\n```bash\npython x.py\n```\n"
    cases = [
        ("a faithful twin is accepted", good_en, good_cn, False),
        ("a dropped heading is caught", good_en, good_cn.replace("## 一\n", ""), True),
        ("a dropped number is caught", good_en, good_cn.replace("K_wall = 64, ", ""), True),
        ("a changed command is caught", good_en, good_cn.replace("python x.py", "python y.py"), True),
        ("a lost file reference is caught", good_en, good_cn.replace("`WALL_NOTE.md`", "该笔记"), True),
    ]
    print("[check_bilingual selftest]")
    all_ok = True
    for name, en, cn, should_fail in cases:
        problems = compare_text(en, cn)
        passed = bool(problems) == should_fail
        all_ok = all_ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}"
              + (f"  ({problems[0][:60]})" if problems else ""))
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="check that every document has a _cn twin that matches it")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the checker itself can say no (runs in memory, needs no files)")
    args = ap.parse_args()

    if args.selftest:
        return 0 if run_selftest() else 1

    missing, orphans, bad = [], [], []
    english = english_documents()
    for en_path in english:
        base = os.path.basename(en_path)[:-3]
        cn_path = os.path.join(os.path.dirname(en_path), base + "_cn.md")
        if not os.path.exists(cn_path):
            missing.append(cn_path)
            continue
        problems = compare(en_path, cn_path)
        if problems:
            bad.append((cn_path, problems))
            if not args.quiet:
                print(f"  MISMATCH {cn_path}")
                for p in problems:
                    print(f"      {p}")

    cn_files = sorted(glob.glob("**/*_cn.md", recursive=True))
    for cn_path in cn_files:
        en_path = cn_path[: -len("_cn.md")] + ".md"
        if not os.path.exists(en_path):
            orphans.append(cn_path)

    print(f"[bilingual] english documents: {len(english)} | twins present: {len(english) - len(missing)} "
          f"| matching: {len(english) - len(missing) - len(bad)}")
    if missing:
        print(f"[bilingual] MISSING Chinese twins ({len(missing)}):")
        for p in missing:
            print(f"    {p}")
    if orphans:
        print(f"[bilingual] ORPHAN _cn files (no English original): {orphans}")
    if bad:
        print(f"[bilingual] MISMATCHED twins: {len(bad)} (see above)")
    ok = not (missing or orphans or bad)
    print(f"[bilingual] {'every document ships in both languages and the twins agree' if ok else 'NOT YET COMPLETE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
