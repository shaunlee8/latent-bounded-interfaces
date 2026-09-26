"""Network table: logs/netx_*.log (emulated link) and reallink*_*.log (shaped
link) -> the rows of Table 2 and the appendix network tables.

Per scheme: counted bytes/step, critical-path bytes per link, charged
operations per step (emu-ops lines), the schedule-independent bound
T(BW, RTT) >= max(T0, bytes_link/BW + n_ops*RTT), the measured step at
each bandwidth (median over repeats, with spread), the bandwidth at
which the measured step is within 5% of the network-free step (log-log
interpolation), and the RTT slope at 1 and 10 Gbit/s.

Usage: python scripts/netx_analyze.py --logs RUN_ROOT/logs --md OUT.md --json OUT.json
(defaults read LBI_RUN_ROOT, the checkpoint/log root of the paper runs).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

from scripts.link_emu import MBPS_TO_BYTES_PER_S

SWEEP_RTT_MS = 5.0   # the RTT of the bandwidth sweep (the "sync, RTT 5" column)
LOW_RTT_MS = 0.1     # the RTT of the bandwidth-to-95% column

STEP_RE = re.compile(r"fwd\+bwd:\s*([0-9.]+) ms/step")
COUNT_RE = re.compile(r"counted comm/step:\s*([0-9.]+) MB")
OPS_RE = re.compile(r"emu-ops rank=(\d+) mode=(\w+) (.*)")
NAME_RE = re.compile(r"netx_(?P<sc>[A-Za-z0-9_]+?)_bw(?P<bw>[0-9.]+)_rtt(?P<rtt>[0-9.]+)_(?P<mode>\w+)_rep(?P<rep>\d+)\.log$")


def parse(path):
    txt = open(path, errors="replace").read()
    m = STEP_RE.search(txt)
    if not m:
        return None
    rec = {"step": float(m.group(1))}
    c = COUNT_RE.search(txt)
    rec["counted_mb"] = float(c.group(1)) if c else None
    ops = {}
    for r, mode, body in OPS_RE.findall(txt):
        d = {}
        for part in body.split():
            k, rest = part.split(":", 1)
            n, b = rest.split(",")
            d[k] = (int(n.split("=")[1]), float(b.split("=")[1]))
        ops[int(r)] = d
    rec["ops"] = ops
    return rec


def steps_in_window(name, txt_path):
    """How many steps the emu-ops counters cover: the pipeline baseline resets
    them after warmup (iters steps), the region-parallel step before warmup
    (warmup + iters). Logs without the printed counts fall back to the
    defaults (5; N + N for window rows, 3 + 5 otherwise)."""
    txt = open(txt_path, errors="replace").read()
    w = re.search(r"warmup=(\d+)", txt)
    it = re.search(r"iters=(\d+)", txt)
    if name.startswith("pipe"):
        return int(it.group(1)) if it else 5
    if w and it:
        return int(w.group(1)) + int(it.group(1))
    m = re.search(r"canvas_accum=(\d+)", txt)
    acc = int(m.group(1)) if m else 1
    return 2 * acc if acc > 1 else 8


def main():
    ap = argparse.ArgumentParser()
    root = os.environ.get("LBI_RUN_ROOT", ".")
    ap.add_argument("--logs", default=os.path.join(root, "logs"))
    ap.add_argument("--md", default=os.path.join(root, "netx_table.md"))
    ap.add_argument("--json", default=os.path.join(root, "netx_table.json"))
    a = ap.parse_args()

    runs = defaultdict(list)  # (sc, bw, rtt, mode) -> [step,...]
    opsby = {}                # sc -> per-rank per-step bytes/ops (from any emulated run)
    for p in sorted(glob.glob(os.path.join(a.logs, "netx_*.log"))):
        m = NAME_RE.search(p)
        if not m:
            continue
        rec = parse(p)
        if rec is None:
            continue
        sc, bw, rtt, mode = m["sc"], float(m["bw"]), float(m["rtt"]), m["mode"]
        runs[(sc, bw, rtt, mode)].append(rec["step"])
        if rec["ops"] and sc not in opsby:
            n = steps_in_window(sc, p)
            per_rank = {}
            for r, d in rec["ops"].items():
                per_rank[r] = {k: (v[0] / n, v[1] / n) for k, v in d.items()}
            opsby[sc] = per_rank

    def med(key):
        v = runs.get(key)
        return (st.median(v), (max(v) - min(v)) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)

    schemes = sorted({k[0] for k in runs})
    bws = sorted({k[1] for k in runs if k[1] > 0})
    out = {"schemes": {}}
    lines = ["# Network table (emulated link; medians over repeats, spread = max-min)", ""]
    for sc in schemes:
        t0, _, _ = med((sc, 0.0, 0.0, "sync"))
        if t0 is None:
            t0, _, _ = med((sc, 0.0, 0.0, "overlap"))
        if t0 is None and sc.startswith("pipeM4_"):
            t0, _, _ = med(("pipeM4", 0.0, 0.0, "sync"))
        row = {"T0_ms": t0, "sweep": {}, "rtt": {}, "overlap": {}}
        # bound inputs: per-link critical bytes = max over ranks of P2P (send + p2p_send);
        # collective bytes reported separately (latency-tolerant for rp, absent for pipe)
        crit = coll = nops = 0.0
        if sc in opsby:
            for r, d in opsby[sc].items():
                p2p = d.get("send", (0, 0))[1] + d.get("p2p_send", (0, 0))[1]
                ops = d.get("send", (0, 0))[0] + d.get("p2p_send", (0, 0))[0] + d.get("broadcast", (0, 0))[0]
                crit = max(crit, p2p); nops = max(nops, ops)
                coll = max(coll, d.get("all_reduce", (0, 0))[1])
        # region-parallel: the serial chain is the K-1 message hops, the
        # Jacobian all-reduce, the seed broadcast, and one shared-parameter
        # reduce per step unless the window fuses it (5 or 6 at K=4).
        if sc.startswith("rp"):
            fused = any(re.search(r"fuse_shared=1", open(p, errors="replace").read())
                        for p in glob.glob(os.path.join(a.logs, f"netx_{sc}_*.log"))[:1])
            nops = 5.0 if fused else 6.0
        row["critical_bytes_per_link"] = crit
        row["collective_bytes_per_rank"] = coll
        row["charged_ops_per_step"] = nops
        lines.append(f"## {sc}: T0 = {t0} ms | critical bytes/link/step = {crit/1e6:.3f} MB | "
                     f"collective bytes/rank/step = {coll/1e6:.3f} MB | charged ops/step = {nops:.1f}")
        lines.append("| BW (Mbit/s) | measured sync, RTT 5 (ms) | spread | n | RTT 0.1 (ms) | bytes-only bound (ms) | bound + ops*RTT5 (ms) | overlap-mode (ms) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for bw in bws:
            s, sp, n = med((sc, bw, SWEEP_RTT_MS, "sync"))
            ov, _, _ = med((sc, bw, SWEEP_RTT_MS, "overlap"))
            bps = bw * MBPS_TO_BYTES_PER_S
            s01, _, _ = med((sc, bw, LOW_RTT_MS, "sync"))
            b1 = max(t0 or 0, ((crit + coll) / bps) * 1e3) if t0 else None          # schedule-independent
            b2 = max(t0 or 0, ((crit + coll) / bps + nops * SWEEP_RTT_MS / 1e3) * 1e3) if t0 else None  # + serial-RTT model
            row["sweep"][bw] = {"measured": s, "spread": sp, "n": n, "rtt01": s01, "bound_bytes": b1, "bound_bytes_rtt": b2, "overlap": ov}
            f = lambda x: "-" if x is None else f"{x:.1f}"
            lines.append(f"| {bw:g} | {f(s)} | {f(sp)} | {n} | {f(s01)} | {f(b1)} | {f(b2)} | {f(ov)} |")
        # Bandwidth to 95% per RTT: the first bandwidth where the step is within
        # 5% of the network-free step T0, log-log interpolated over the sweep.
        row["bw95_mbps"] = {}
        for rtt in (SWEEP_RTT_MS, LOW_RTT_MS):
            b95 = None
            if t0:
                pts = [(bw, med((sc, bw, rtt, "sync"))[0]) for bw in bws + [100000.0]]
                pts = [(bw, v) for bw, v in pts if v]
                thr = 1.05 * t0
                for (bw_lo, s_lo), (bw_hi, s_hi) in zip(pts, pts[1:]):
                    if s_lo > thr >= s_hi:
                        x = (math.log(s_lo) - math.log(thr)) / (math.log(s_lo) - math.log(s_hi)) if s_lo != s_hi else 1
                        b95 = math.exp(math.log(bw_lo) + x * (math.log(bw_hi) - math.log(bw_lo)))
                        break
                if b95 is None and len(pts) >= 2 and pts[0][1] <= thr:
                    b95 = pts[0][0]
            row["bw95_mbps"][rtt] = b95
            floor = med((sc, 100000.0, rtt, "sync"))[0]
            # bandwidth-isolating variant: within 5% of the latency-only floor at this RTT
            # (the 100 Gbit/s point), which removes the RTT term and the host-sleep
            # granularity (~0.5 ms per charged op at RTT 0.1) from the statistic
            bf = None
            if floor:
                thr = 1.05 * floor
                for (bw_lo, s_lo), (bw_hi, s_hi) in zip(pts, pts[1:]):
                    if s_lo > thr >= s_hi:
                        x = (math.log(s_lo) - math.log(thr)) / (math.log(s_lo) - math.log(s_hi)) if s_lo != s_hi else 1
                        bf = math.exp(math.log(bw_lo) + x * (math.log(bw_hi) - math.log(bw_lo)))
                        break
                if bf is None and len(pts) >= 2 and pts[0][1] <= thr:
                    bf = pts[0][0]
            row.setdefault("bw95_floor_mbps", {})[rtt] = bf
            lines.append(f"\n95% bandwidth at RTT {rtt:g} ms: vs network-free step "
                         f"{('%.0f Mbit/s' % b95) if b95 else 'not reached (latency floor)'}"
                         + (f" | vs latency-only floor ({floor:.1f} ms at 100 Gbit/s): {('%.0f Mbit/s' % bf) if bf else 'not reached'}" if floor else ""))
        # RTT slope
        for bw in (1000.0, 10000.0):
            pts = [(rtt, med((sc, bw, rtt, "sync"))[0]) for rtt in (0.1, 1.0, 5.0, 20.0, 50.0)]
            pts = [(r, s) for r, s in pts if s]
            if len(pts) >= 2:
                slope = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
                row["rtt"][bw] = {"points": pts, "slope_ms_per_ms": slope}
                lines.append(f"RTT at {bw:g} Mbit/s: " + ", ".join(f"{r:g}->{s:.0f}" for r, s in pts) + f" | slope {slope:.1f} ms/ms")
        lines.append("")
        out["schemes"][sc] = row

    # real-link rows
    real = {}
    for p in sorted(glob.glob(os.path.join(a.logs, "reallink*_*.log"))):
        m = re.search(r"(reallink\d?)_(\w+?)_(sock|\d+rtt\d+[a-z]*|\d+)(?:_rep\d+)?\.log$", p)
        rec = parse(p)
        if m and rec:
            real.setdefault(m.group(1), {}).setdefault(m.group(2), {}).setdefault(m.group(3), []).append(rec["step"])
    for sess, d1 in sorted(real.items()):
        lines.append(f"## Shaped loopback link, log set {sess} (NCCL sockets over lo, tc tbf + netem; one shared pipe for all ranks)")
        lines.append("| scheme | socket unshaped | 10 G | 1 G | 100 M |")
        lines.append("|---|---|---|---|---|")
        for sc, d in d1.items():
            def g(k):
                v = d.get(k)
                if not v: return "-"
                return f"{st.median(v):.0f}" + (f" (n={len(v)}, spread {max(v)-min(v):.0f})" if len(v) > 1 else "")
            lines.append(f"| {sc} | {g('sock')} | {g('10000')} | {g('1000')} | {g('100')} |")
            extra = [k for k in d if 'rtt' in k]
            for k in extra:
                lines.append(f"| {sc} @ {k} | | | {g(k)} | |")
        lines.append("")
    out["reallink"] = real
    open(a.md, "w").write("\n".join(lines) + "\n")
    json.dump(out, open(a.json, "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
