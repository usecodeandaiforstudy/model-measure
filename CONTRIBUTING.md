# Contributing

This repository is a **measurement record** as much as a library: `capacity_lens/` is the product,
`experiments/` is the evidence for every claim the product makes, and `docs/` is the written
record. Contributions are welcome under the rules below — they exist because the project's value comes from
claims that can be checked, not from features that look useful.

## Getting set up

```bash
pip install -e ".[dev]"          # library + torchvision (experiments) + pytest
python tests/run_all.py          # tests without pytest (the verified GPU environment has no pytest)
pytest tests/                    # same suite, when pytest is available
```

## What CI checks before anything is merged

`.github/workflows/tests.yml` runs four gates, all offline (no GPU, no datasets, no network):

1. **the suite** — `python tests/run_all.py` and `pytest tests/` on Python 3.9, 3.11 and 3.13;
2. **every script's `--selftest`** — the pre-registered criteria live in the experiment scripts, so they are run;
3. **the evidence** — `python tools/reproduce_all.py --check` rebuilds every published verdict from the recorded
   jsonl and fails if one no longer rebuilds or differs from the committed snapshot. **If you change a criterion
   without recording the change, this is what catches it** (report files are append-only, so the old verdict
   stays visible next to the new one);
4. **the distribution** — build a wheel, run `python tools/check_wheel.py` (the wheel must ship the library and
   nothing else; run its `--selftest` to see it can say no) and run `examples/quickstart.py`.

The experiment scripts run from a checkout (`python experiments/<name>_demo.py`) and need their datasets
placed manually under `experiments/data/` — see the README's data-layout section. **The project never
downloads data and never calls the network**, and every script must keep working offline.

## The rules a change has to satisfy

1. **Criteria before data.** Any new experiment writes its pre-registration into the module docstring
   *before* it is run: definitions, the grid, the criteria, the verdict rule. Editing those after seeing
   results is only acceptable as an explicitly recorded amendment (see `docs/class-count-axis.md`
   §6.1 and §7 for two examples of corrections that were kept in the record rather than hidden).
2. **Criteria are pure functions, and they are tested.** Thresholds and verdict assembly live in small
   functions that take plain data and return plain results, so they can be unit-tested without training;
   each script exposes them through `--selftest`, and the cross-script ones are covered in
   `tests/test_experiment_criteria.py`.
3. **No re-rolling to pass a gate.** If a seed fails a gate, it is reported as failed. There is no
   "run it again until green" — that is noise selection, and it is the failure mode the G0 gate exists to
   prevent.
4. **Negative results are archived as-is.** A falsified expectation is a result: it goes into the note with
   its mechanism, into `docs/falsified.md`, and into the ledger patch as a new entry.
5. **Relay mode.** Every unit of work is cached per (configuration, seed, K) under
   `experiments/data/<campaign>/<fingerprint>/`, and the fingerprint covers only the *base* grids, so an
   extended run reuses the cached base units. Long runs must survive interruption, and offline
   re-verdicts (`--verdict-only`) must reproduce the published numbers without retraining.
6. **English only, in code and in output.** Identifiers, docstrings, comments, `print` output and exception
   messages. Every document in `docs/` ships in both languages: `<name>.md` plus `<name>_cn.md`.
7. **Status labels are mandatory.** A public function carries `VERIFIED` / `PARTIALLY VERIFIED` /
   `CANDIDATE` in its docstring, with the measured caveat spelled out. **Do not ship a verdict-shaped API
   whose direction has not been verified** — `capacity_lens/monitor.py` records the precedent: its
   `H(Z) > R_IB ⇒ overfit` flag was removed rather than left in place with a warning, because a caller
   cannot tell a heuristic from a rule.
8. **No fabricated metadata.** No invented repository URLs, DOIs, author lists or affiliations. If a field
   is unknown, leave it out.

## Adding an experiment
Name it `experiments/<topic>_demo.py`, and give it the standard shape:

```
docstring      — the pre-registration (design, grids, criteria, verdict rule, iron rules)
pure functions — the criteria and the verdict assembly, unit-testable
--selftest     — runs those unit tests with fixture data, no training
--quick        — a smoke configuration (tiny grids) for wiring checks
--fresh        — recompute instead of the default relay mode
--verdict-only — rebuild the verdict from the recorded jsonl, zero training
artifacts      — <topic>.jsonl, <topic>_report.txt, <topic>/<fingerprint>/ checkpoints
```

**Two generations of scripts, and why the older ones stay as they are.** The nine earliest experiments
(`cifar_demo.py`, `cp3_demo.py`, `cp3_dyn.py`, `cp3_dyn_sens.py`, `cp3_multilevel.py`, `cph_post.py`,
`mnist_demo.py`, `opt_demo.py`, `transformer_demo.py`) predate `--selftest` / `--verdict-only` / relay mode.
They are **deliberately left untouched**: their recorded artifacts (jsonl, reports, checkpoints) were produced
by exactly this code, so rewriting them would decouple the numbers from the code that made them — which is the
one thing this project refuses to do. Read them as the historical record, run them only to reproduce their own
artifacts, and give every new script the full shape above. A closing audit (2026-10-05) verified that all
twenty-three later scripts expose the offline paths. Run `python tools/reproduce_all.py` after adding one: it
replays every published verdict from the recorded jsonl and will tell you whether yours re-verdicts cleanly
(and whether it needs an import-time write, which it must not).

Then write `docs/<topic>.md` (lowercase kebab-case: design, results, verdict, what it means, honest boundaries) plus its `_cn` twin
and add a ledger line to `docs/ledger.md` plus a row to the README's verification
matrix. A new campaign without a note and a matrix row does not exist as far as the project is concerned.

## Documentation layout

- `README.md` — the single entry point: capabilities, boundaries, quickstart, the verification matrix.
  Every claim there must be traceable to a note or a ledger entry.
- `docs/*.md` — one document per campaign (lowercase kebab-case): measurements, verdict and honest boundaries.
- `docs/framework.md` — the theoretical frame and the stance that survived.
- `docs/verification-plan.md`, `docs/ledger.md` — the pre-registered criteria, and the 33-row verdict ledger that indexes every campaign.
- Local working material (pre-translation snapshots, retired process documents, old run dumps) lives in `reference/`, which is gitignored and never committed.
- `artifacts.md` (in `docs/`) — the run-artifact manifest, and the place where anything unusual about
  the shipped evidence is disclosed (two such disclosures are recorded there rather than "fixed" in the data).

**Every document ships in both languages.** The Chinese file is the English name plus **`_cn`**
(`README.md` → `README_cn.md`, `docs/transformer.md` → `docs/transformer_cn.md`), and the two must be
**twins**: same heading structure, same numbers, same file references, same code blocks. Numbers are the
project's claims, so a number that drifts in translation makes the translation a lie — which is why
`python tools/check_bilingual.py` checks exactly those things mechanically (and CI runs it). Deliberately exempt,
because their canonical filenames are read by tools and their content is not prose: `LICENSE`, `CITATION.cff`,
`.gitignore`, `pyproject.toml`, and the recorded evidence (`*.jsonl`, `*_report.txt`).

**Two histories, two files, and why the CHANGELOG stays at the root.** `CHANGELOG.md` is the *library's* history
(API added/removed/fixed, with the reason and the measurement behind each change) and belongs at the repository
root like any other package's: readers, reviewers and release tooling look for it there. The *research's* history
— verdicts, superseded readings, criterion flaws — lives in `ledger.md` and the notes, and is
never folded into the CHANGELOG. Keeping these two apart is deliberate; a change to one is not a change to the
other. Pre-release milestones are fine in the CHANGELOG even though nothing has been published: it records what
the package did at each version, not when it shipped.

## Code style

Plain PyTorch, no framework beyond it; no global state at import time (logging is set up lazily inside the
entry point, so the modules stay importable for tests); no downloads; deterministic seeds everywhere a
result is recorded. Keep functions small, keep the math in the module that owns it (`capacity.py` for
capacity formulas, `preflight.py` for judging rules), and prefer an explicit `None` with a comment over a
silent fallback.
