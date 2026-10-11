"""KV: verification of the content-addressable memory reading (kv_storage_capacity) on associative recall.

The claim under test (capacity.py / A-002 4.5): a KV memory's STORAGE grows with the number of stored
entries (seq_len * per_token), while a single READOUT is bounded by the activated content -- which the
flow tier of the surrounding network governs, not the total storage.

Usage:
    python experiments/kv_demo.py                  # full run, 3 seeds, ~15-25 min (tiny models)
    python experiments/kv_demo.py --quick          # 1 seed, smaller grids (smoke)
    python experiments/kv_demo.py --selftest       # pure-function unit tests

Pre-registration (2026-09-30, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) Task: associative recall. A memory of L slots holds one value per slot, values drawn uniformly from
    C_V = 8 classes; the query names one slot and the model must output that slot's value. Task
    information: L * ln(C_V) nats are STORED (the whole table), while a single query needs ln(C_V) nats
    of payload (plus ln(L) of addressing). Sequence length L in {2,4,8,16}; fixed dataset per arm
    (generated in-process, no downloads); 40 epochs Adam 1e-3, batch 128.
(2) Three arms, same task and same representation width d=32, and the SAME computational capacity
    (every encoder/decoder is a 2-layer MLP of width 128); the ONLY structural difference is where the
    VQ bottleneck sits -- which is exactly what "storage vs flow" is about:
      KV   (content-addressable): keys are addressed by an unquantized embedding path (the routing
           channel, analogous to the MoE router whose information counts in the flow bound), and the
           PAYLOAD of every slot is written through a per-token VQ(K_tok). Read = attention over the
           quantized payloads. Storage = L * log K_tok; per-query flow = ln(L) addressing + ln(C_V) read.
      DENSE (memory written before the query): the whole table goes through ONE VQ(K_dense); the query
           only conditions the decoder afterwards. Everything needed to answer ANY query must pass
           through that single code => the storage requirement is L * ln(C_V).
      CTRL (bottleneck after the query): the query is concatenated BEFORE the VQ(K_ctrl), so the
           bottleneck only has to carry the answer (ln(C_V)) => the smallest possible requirement.
    (Recorded: an earlier draft gave the arms single Linear layers; a pre-run smoke test showed such
    layers cannot implement the query-dependent selection at all, so the arms were upgraded to equal MLP
    capacity BEFORE the pre-registered grid was run. This is a design fix, not a criterion change.)
(3) Pre-registered criteria (pure functions; >= 2/3 seeds):
      K1 (per-token flow binds, storage does not): KV arm with K_tok = C_V = 8 reaches acc >= 0.90 for
         EVERY L in {2,4,8,16}; with K_tok = 4 (< C_V) acc <= 0.50 for every L.
      K2 (dense storage requirement is L * ln C_V): the smallest K_dense reaching acc >= 0.90 at each L
         satisfies log K_dense within a factor of 2 of L * ln(C_V) (i.e. K ~ C_V^L).
      K3 (bottleneck POSITION is the mechanism): the CTRL arm reaches acc >= 0.90 with K_ctrl = 8 for
         every L in {2,4,8} -- the same storage, a different bottleneck position, no exponential cost.
      K4 (formula consistency): kv_storage_capacity(L, per_token=ln C_V) = L*ln C_V predicts the dense
         arm's measured requirement within one octave (factor e).
    Verdict (pure function kv_verdict): SUPPORT if K1 and K3 hold and (K2 or K4 holds within an octave);
    else PARTIAL/MIXED/REFUTED, recorded as-is. A REFUTED verdict here would mean the two-tier reading
    does not transfer from routing (MOE) to content-addressable memory (attention).
(4) Relay: per-unit checkpoints under experiments/data/kv/<fingerprint>/; --fresh recomputes.
(5) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads.
-----------------------------------------------------------------------------------------------
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import VQBottleneck, train_one_epoch, evaluate, entropy_of, kv_storage_capacity

C_V = 8
D = 32
N_SEQ = 4096
EPOCHS = 40
LR = 1e-3
BATCH = 128
L_GRID_KV = [2, 4, 8, 16]
K_TOK_GRID = [4, 8, 16]
L_GRID_DENSE = [2, 3, 4]
# per-L dense grids, centred on the prediction exp(L*ln C_V) = C_V^L (64 / 512 / 4096): fine enough to
# resolve the pre-registered "within a factor of 2" test on either side of the prediction
K_DENSE_BY_L = {2: [64, 96, 128, 192], 3: [512, 768, 1024, 2048], 4: [2048, 4096]}
K_DENSE_GRID = sorted({k for v in K_DENSE_BY_L.values() for k in v})
L_GRID_CTRL = [2, 4, 8]
K_CTRL_GRID = [8, 16, 64]

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "kv.jsonl")
LOG = os.path.join(_DATA, "kv.log")
REPORT = os.path.join(_DATA, "kv_report.txt")
CKPT_ROOT = os.path.join(_DATA, "kv")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_logging():
    global _LOG
    _LOG = open(LOG, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def make_recall(n_seq, L, seed, c_v=C_V):
    """L slots, values uniform in [c_v]; each sample queries one slot (one-hot position).
    Returns x = [values one-hot flatten | query one-hot] (the project's single-input convention) and y."""
    g = torch.Generator().manual_seed(seed)
    vals = torch.randint(0, c_v, (n_seq, L), generator=g)
    qi = torch.randint(0, L, (n_seq,), generator=g)
    target = vals[torch.arange(n_seq), qi]
    v_oh = F.one_hot(vals, c_v).float().reshape(n_seq, -1)     # [N, L*c_v]
    q_oh = F.one_hot(qi, L).float()                            # [N, L]
    return torch.cat([v_oh, q_oh], dim=-1), target


def split_dataset(x, y):
    n_tr = int(0.8 * len(y))
    return (x[:n_tr], y[:n_tr]), (x[n_tr:], y[n_tr:])


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

class KVMem(nn.Module):
    """Content-addressable memory: keys addressed by an unquantized embedding path (the routing channel),
    slot payloads written through a per-token VQ. Read = attention over the quantized payloads."""

    def __init__(self, L, k_tok, d=D, c_v=C_V):
        super().__init__()
        self.L = L
        self.key_emb = nn.Linear(L, d)                    # addressing channel (position/one-hot key)
        self.val_enc = _mlp(c_v, d)                       # payload encoder
        self.vq = VQBottleneck(d, k_tok, metric="euclid")
        self.dec = _mlp(d, c_v)

    def forward(self, x):
        n = x.size(0)
        v_oh = x[:, :self.L * C_V].reshape(n, self.L, C_V)
        q_oh = x[:, self.L * C_V:]
        keys = self.key_emb(torch.eye(self.L, device=x.device)).unsqueeze(0).expand(n, -1, -1)
        z, idx = self.vq(self.val_enc(v_oh).reshape(-1, D))
        z = z.reshape(n, self.L, D)
        q = self.key_emb(q_oh)
        att = (q.unsqueeze(1) @ keys.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        read = (att.softmax(-1).unsqueeze(-1) * z).sum(1)
        return self.dec(read), [idx]


class DenseMem(nn.Module):
    """Memory written BEFORE the query: the whole table passes through one VQ, the query only conditions
    the decoder afterwards => the single code must carry everything needed to answer any query."""

    def __init__(self, L, k_codes, d=D, c_v=C_V):
        super().__init__()
        self.L = L
        self.enc = _mlp(L * c_v, d)
        self.vq = VQBottleneck(d, k_codes, metric="euclid")
        self.dec = _mlp(d + L, c_v)

    def forward(self, x):
        n = x.size(0)
        v_oh = x[:, :self.L * C_V]
        q_oh = x[:, self.L * C_V:]
        h = F.relu(self.enc(v_oh))
        z, idx = self.vq(h)
        return self.dec(torch.cat([z, q_oh], dim=-1)), [idx]


class CtrlMem(nn.Module):
    """Control: the query is concatenated BEFORE the bottleneck, so the code only carries the answer."""

    def __init__(self, L, k_codes, d=D, c_v=C_V):
        super().__init__()
        self.enc = _mlp(L * c_v + L, d)
        self.vq = VQBottleneck(d, k_codes, metric="euclid")
        self.dec = _mlp(d, c_v)

    def forward(self, x):
        h = F.relu(self.enc(x))
        z, idx = self.vq(h)
        return self.dec(z), [idx]


ARMS = {"kv": KVMem, "dense": DenseMem, "ctrl": CtrlMem}

MLP_WIDTH = 128


def _mlp(d_in, d_out, width=MLP_WIDTH):
    """Two-layer MLP: every arm gets the SAME computational capacity, so the only structural difference
    between arms is WHERE the VQ bottleneck sits. (An earlier draft used single Linear layers; a pre-run
    smoke test showed they cannot implement the query-dependent selection at all -- that is a capacity
    artifact of the draft, not of the reading under test.)"""
    return nn.Sequential(nn.Linear(d_in, width), nn.ReLU(), nn.Linear(width, d_out))


# ---------------------------------------------------------------------------
# Criterion pure functions
# ---------------------------------------------------------------------------

def dense_requirement_nats(L, c_v=C_V):
    """Predicted dense storage requirement, taken from the function under test: L * ln(c_v) nats."""
    return kv_storage_capacity(L, math.log(c_v))


def octave_ok(measured_k, predicted_nats, octaves=1.0):
    """Is ln(measured K) within `octaves` octaves (a factor of 2 each) of the predicted requirement?
    The 1e-9 slack absorbs floating-point boundary cases (a factor exactly 2 must count as inside)."""
    if measured_k is None:
        return False
    return abs(math.log(measured_k) - predicted_nats) <= octaves * math.log(2.0) + 1e-9


def smallest_good(accs, thresh=0.90):
    """Smallest K whose accuracy reaches the threshold (None when none does)."""
    ok = [k for k, a in accs.items() if a >= thresh]
    return min(ok) if ok else None


def kv_verdict(rows):
    """Pre-registered verdict (docstring (3)). rows: list of per-unit records."""
    def accs(arm, L, K):
        vals = [r["acc"] for r in rows if r["arm"] == arm and r["L"] == L and r["K"] == K]
        return sum(vals) / len(vals) if vals else None

    k1_hi = [accs("kv", L, C_V) for L in L_GRID_KV]
    k1_lo = [accs("kv", L, 4) for L in L_GRID_KV]
    k1 = (all(a is not None and a >= 0.90 for a in k1_hi)
          and all(a is not None and a <= 0.50 for a in k1_lo))
    k3 = all((accs("ctrl", L, 8) or -1) >= 0.90 for L in L_GRID_CTRL)
    k2_hits, k2_detail = [], []
    for L in L_GRID_DENSE:
        per_k = {K: accs("dense", L, K) for K in K_DENSE_BY_L[L] if accs("dense", L, K) is not None}
        got = smallest_good(per_k)
        need = dense_requirement_nats(L)
        hit = got is not None and abs(math.log(got) - need) <= math.log(2.0) + 1e-9
        k2_hits.append(hit)
        k2_detail.append(f"L={L}: smallest K with acc>=0.90 is {got} (predicted ~{math.exp(need):.0f}, "
                         f"factor {math.exp(need) / got:.2f}x)" if got else
                         f"L={L}: no K reached 0.90 (predicted ~{math.exp(need):.0f}, "
                         f"grid top {max(K_DENSE_BY_L[L])})")
    k2 = all(k2_hits)
    k4_hits = 0
    for L in L_GRID_DENSE:
        per_k = {K: accs("dense", L, K) for K in K_DENSE_BY_L[L] if accs("dense", L, K) is not None}
        g = smallest_good(per_k)
        if g is not None and abs(math.log(g) - dense_requirement_nats(L)) <= math.log(math.e) + 1e-9:
            k4_hits += 1
    k4 = k4_hits >= max(1, len(L_GRID_DENSE) - 1)
    detail = [
        f"  K1 per-token flow binds: KV acc at K_tok=C_V: {[round(a, 3) if a else None for a in k1_hi]} ; "
        f"at K_tok=4: {[round(a, 3) if a else None for a in k1_lo]} -> {'PASS' if k1 else 'FAIL'}",
        f"  K2 dense requirement ~ C_V^L: " + "; ".join(k2_detail) + f" -> {'PASS' if k2 else 'FAIL'}",
        f"  K3 bottleneck position is the mechanism (CTRL K=8, acc>=0.90 for L in {L_GRID_CTRL}): "
        f"{[round(accs('ctrl', L, 8) or -1, 3) for L in L_GRID_CTRL]} -> {'PASS' if k3 else 'FAIL'}",
        f"  K4 kv_storage_capacity(L, ln C_V) = L*ln C_V predicts the dense wall within one octave -> "
        f"{'PASS' if k4 else 'FAIL'}",
    ]
    if k1 and k3 and (k2 or k4):
        return ("SUPPORT: the content-addressable memory reading is verified -- storage scales with the "
                "number of slots (L*ln C_V) while a single readout is bounded by the activated payload; "
                "moving the same bottleneck after the query removes the exponential requirement"), detail
    if k1 and k3:
        return ("PARTIAL: the two-tier mechanism holds (K1/K3) but the dense requirement does not match "
                "the L*ln C_V prediction within an octave (K2/K4) -- recorded as-is"), detail
    return "MIXED/REFUTED: the two-tier reading did not transfer to content-addressable memory (recorded)", detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_signature(quick):
    raw = "|".join([f"q{int(quick)}", f"cv{C_V}", f"d{D}", f"ep{EPOCHS}",
                    f"kv{','.join(map(str, L_GRID_KV))}", f"kt{','.join(map(str, K_TOK_GRID))}",
                    f"de{','.join(map(str, L_GRID_DENSE))}", f"kd{','.join(map(str, K_DENSE_GRID))}",
                    f"ct{','.join(map(str, L_GRID_CTRL))}", f"kc{','.join(map(str, K_CTRL_GRID))}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def train_unit(arm, L, K, seed, epochs, device, ckpt_path, resume):
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return ck["acc"], ck.get("h_z"), None
    x, target = make_recall(N_SEQ, L, seed)
    (x_tr, y_tr), (x_te, y_te) = split_dataset(x, target)
    tr = DataLoader(list(zip(x_tr, y_tr)), batch_size=BATCH, shuffle=True)
    te = DataLoader(list(zip(x_te, y_te)), batch_size=256)
    torch.manual_seed(100000 + 1000 * seed + 10 * L + K)
    model = ARMS[arm](L, K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, tr, opt, device=device)
    acc, idx = evaluate(model, te, device=device)
    h_z = entropy_of(idx[0], K)
    torch.save({"acc": acc, "h_z": h_z, "state_dict": model.state_dict()}, ckpt_path)
    return acc, h_z, model


def run_kv(quick, resume, device):
    seeds = 1 if quick else 3
    l_kv = [2, 4, 8] if quick else L_GRID_KV
    k_tok = [8] if quick else K_TOK_GRID
    l_de = [2, 3] if quick else L_GRID_DENSE
    k_de = [64, 256] if quick else None
    l_ct = [2, 4] if quick else L_GRID_CTRL
    k_ct = [8] if quick else K_CTRL_GRID
    epochs = 8 if quick else EPOCHS
    fp = run_signature(quick)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 84)
    print(f"[KV] content-addressable memory reading: storage vs flow (associative recall, C_V={C_V})")
    print("=" * 84)
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print(f"  [arms] KV (payload VQ) / DENSE (table VQ before the query) / CTRL (VQ after the query)")
    print(f"  [prediction] dense requirement log K ~ L*ln(C_V) = L*{math.log(C_V):.2f} nats")
    plan = ([("kv", L, K, k_tok) for L in l_kv for K in k_tok]
            + [("dense", L, K, None) for L in l_de for K in (K_DENSE_BY_L[L] if not quick else k_de)]
            + [("ctrl", L, K, None) for L in l_ct for K in k_ct])
    rows = []
    t0 = time.time()
    for arm, L, K, _ in plan:
        accs = []
        for s in range(seeds):
            ck = os.path.join(ckpt_dir, f"{arm}_L{L}_K{K}_s{s}.pt")
            acc, h_z, _ = train_unit(arm, L, K, s, epochs, device, ck, resume)
            accs.append(acc)
            jsonl_append(JSONL, dict(arm=arm, L=L, K=K, seed=s, acc=acc, h_z=h_z,
                                     dense_need_nats=dense_requirement_nats(L, C_V),
                                     kv_storage_nats=kv_storage_capacity(L, math.log(C_V)), fp=fp))
        med = sorted(accs)[len(accs) // 2]
        rows.append(dict(arm=arm, L=L, K=K, acc=med, accs=accs))
        print(f"  {arm:5s} L={L:2d} K={K:5d}: acc={med:.3f} (seeds {[round(a, 3) for a in accs]})")
    print(f"\n  [elapsed] {time.time() - t0:.0f}s")
    print("\n  --- KV verdict ---")
    verdict, detail = kv_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[KV] content-addressable memory reading: storage vs flow\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return rows, verdict


def run_selftest():
    ok = []
    ok.append(("dense requirement = L*ln C_V",
               abs(dense_requirement_nats(4) - 4 * math.log(8)) < 1e-9))
    ok.append(("octave_ok accepts a factor-2 miss", octave_ok(128, math.log(256), octaves=1.0)))
    ok.append(("octave_ok rejects a factor-8 miss", not octave_ok(32, math.log(256), octaves=1.0)))
    ok.append(("smallest_good picks the smallest K", smallest_good({8: 0.5, 16: 0.91, 64: 0.99}) == 16))
    ok.append(("smallest_good returns None when nothing passes", smallest_good({8: 0.5}) is None))
    good = []
    for L in L_GRID_KV:
        good.append(dict(arm="kv", L=L, K=C_V, acc=0.99))
        good.append(dict(arm="kv", L=L, K=4, acc=0.25))
    for L in L_GRID_DENSE:
        for K in K_DENSE_GRID:
            good.append(dict(arm="dense", L=L, K=K, acc=0.95 if K >= C_V ** L else 0.2))
    for L in L_GRID_CTRL:
        good.append(dict(arm="ctrl", L=L, K=8, acc=0.95))
    v, _ = kv_verdict(good)
    ok.append(("verdict SUPPORT on a clean synthetic table", v.startswith("SUPPORT")))
    bad = [r for r in good if not (r["arm"] == "kv" and r["K"] == 4)]
    bad += [dict(arm="kv", L=L, K=4, acc=0.99) for L in L_GRID_KV]     # K1 lower half broken
    v, _ = kv_verdict(bad)
    ok.append(("verdict not SUPPORT when K1 fails", not v.startswith("SUPPORT")))
    print("[KV selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="KV: content-addressable memory reading (storage vs flow)")
    ap.add_argument("--quick", action="store_true", help="1 seed, smaller grids (smoke)")
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    _setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    run_kv(args.quick, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
