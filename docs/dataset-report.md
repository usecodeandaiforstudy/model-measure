# Dataset report

`capacity_dataset` measures three things about a classification dataset and returns one recommendation. It
only reads the published `capacity_lens` API (no formula is reimplemented) and needs nothing beyond torch.

| block | what it reports | cost |
| --- | --- | --- |
| duplicate report | exact duplicate rate, label collisions, H(y), R_IB, K_min | seconds, no training |
| saturation | level, parameter knee, sample knee | one short training run |
| review price | rows per caught mislabel (requires true labels) | one training run |

One recommendation follows from a fixed table: `DEDUP FIRST`, `TOO EASY`, `USABLE`, `UNREACHABLE`.

## What it does not do

- no automatic cleaning: deleting high-loss rows measured +0.0010 accuracy, so it is not offered;
- no label-noise detection: three approaches failed against their own null controls;
- no single difficulty score: the knee measures how fast a task saturates, not how high it goes;
- no precision or recall without ground truth: `review_price` returns `None` when given none.

## Codec attribution

`codec_report(enc, dec, X, y, reference_acc=...)` returns `LOSS IN ENCODER`, `LOSS IN DECODER` or `NEITHER`.
The reference must be measured on the raw input; an accuracy ceiling derived from the probe's lower bound is
invalid (predicted 0.500, achieved 0.643 in one measurement). Without a reference the verdict is `UNDECIDED`.

## Demo

    python -m capacity_dataset.demo

Runs on synthetic data with known defects and writes `demo_report.md` in the current directory.
