"""Command line for capacity_dataset. Reads csv or npy, prints a JSON/Markdown report."""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

from .report import build_report, dup_report, review_price, saturation_report


def _load(args):
    p = pathlib.Path(args.data)
    if p.suffix == ".npy":
        X = torch.as_tensor(__import__("numpy").load(str(p)), dtype=torch.float32)
    elif p.suffix == ".csv":
        import csv
        rows, labels = [], []
        with open(p, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            head = next(rd)
            li = head.index(args.label_col) if args.label_col in head else len(head) - 1
            for r in rd:
                vals = [float(v) for i, v in enumerate(r) if i != li]
                rows.append(vals)
                labels.append(int(float(r[li])))
        X = torch.tensor(rows, dtype=torch.float32)
        return X, torch.tensor(labels, dtype=torch.long)
    else:
        raise SystemExit(f"unsupported data file: {p.suffix} (use .npy or .csv)")
    y = None
    if args.labels:
        y = torch.as_tensor(__import__("numpy").load(args.labels), dtype=torch.long)
    return X, y


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capacity_dataset", description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("dup", "sat", "price", "report", "codec"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True, help="X matrix: .npy or .csv")
        s.add_argument("--labels", help="labels: .npy (or use --label-col with csv)")
        s.add_argument("--label-col", default="label")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--epochs", type=int, default=25)
        s.add_argument("--noisy", help="noisy labels .npy (price subcommand only)")
        s.add_argument("--json", help="write the report to this path")
    args = ap.parse_args(argv)

    if args.selftest:
        from .report import selftest
        fails = selftest()
        print(f"[capacity_dataset] {len(fails)} failures")
        for f in fails:
            print("   " + f)
        return 1 if fails else 0
    if not args.cmd:
        ap.print_help()
        return 2

    X, y = _load(args)
    if args.cmd == "dup":
        out = dup_report(X, y)
    elif args.cmd == "sat":
        out = saturation_report(X, y, seed=args.seed, epochs=args.epochs)
    elif args.cmd == "codec":
        from .codec import latent_report
        # the CLI can only measure the LATENT (no encoder/decoder object is available from a file), so it
        # reports capacity/information/efficiency and explicitly refuses to attribute the loss.
        out = latent_report(X, y if y is not None else torch.zeros(len(X), dtype=torch.long))
        out["verdict"] = "UNDECIDED (CLI cannot attribute: pass enc/dec to codec_report in Python)"
    elif args.cmd == "price":
        if not args.noisy:
            raise SystemExit("price needs both --labels (true) and --noisy (noisy) .npy files")
        import numpy as np
        y_true = torch.as_tensor(np.load(args.labels), dtype=torch.long) if args.labels else None
        y_noisy = torch.as_tensor(np.load(args.noisy), dtype=torch.long)
        out = review_price(X, y_true, y_noisy, seed=args.seed, epochs=max(5, args.epochs // 2))
        if out is None:
            raise SystemExit("price needs --labels (true) as well")
    else:
        out = build_report(X, y, seed=args.seed, epochs=args.epochs)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
