"""The appendix timeline figure: every GPU kernel of one region-parallel
training step (K = 4 devices, Mamba-3, r = 16), read from an Nsight Systems
trace exported to sqlite, drawn as a bar on its device's row. Kernels are
classed into five phases: region forward, Jacobian construction (from the
device's first construction kernel to the end of its last), region-local
backward, chain messages (NCCL SendRecv), and end-of-step reductions (NCCL
AllReduce and Broadcast, including waits). A bracket over the last device's
row gives its critical path.

Usage: python scripts/appendix/make_timeline_figure.py --db TRACE.sqlite [--out FIG.pdf]"""
import argparse
import sqlite3

# The construction's two kernels: the fused recurrence JVP and the tilelang
# chunked-scan kernel (nsys short name "kernel_kernel").
CONSTRUCTION = ("recurrence_jvp_kernel", "kernel_kernel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="timeline_sharded_step.pdf")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c = sqlite3.connect(a.db)
    rows = c.execute("""SELECT k.deviceId, k.start, k.end, s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id""").fetchall()
    # step boundaries: a step begins with device 0's embedding gather (the first
    # kernel of its forward, right after the previous step's f32 all-reduces); take
    # the last complete step in the trace
    d0 = sorted((s, e, n) for d, s, e, n in rows if d == 0)
    gathers = [s for s, e, n in d0 if n == "vectorized_gather_kernel"]
    S, E = gathers[-2], gathers[-1]
    period = (E - S) / 1e6
    win = [(d, max(s, S), min(e, E), n) for d, s, e, n in rows if e > S and s < E]

    def phases(dev):
        cons = [(s, e) for d, s, e, n in win if d == dev and any(k in n for k in CONSTRUCTION)]
        on, off = min(s for s, e in cons), max(e for s, e in cons)
        return on, off

    bounds = {d: phases(d) for d in range(4)}

    def cls(dev, s, n):
        if "SendRecv" in n:
            return "chain"
        if "nccl" in n.lower():
            return "reduce"
        on, off = bounds[dev]
        if s < on:
            return "forward"
        if s < off:
            return "construction"
        return "backward"

    colors = {"forward": "#0072B2", "construction": "#E69F00", "backward": "#56B4E9", "chain": "#009E73", "reduce": "#999999"}
    z = {"forward": 1, "construction": 2, "backward": 1, "chain": 3, "reduce": 3}
    labels = {"forward": "region forward", "construction": "Jacobian construction", "backward": "region-local backward",
              "chain": "chain message (NCCL send/recv)", "reduce": "collective (NCCL all-reduce / broadcast, incl. waits)"}
    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=200)
    for d, s, e, n in win:
        k = cls(d, s, n)
        ax.barh(3 - d, (e - s) / 1e6, left=(s - S) / 1e6, height=0.62 if k not in ("chain", "reduce") else 0.3,
                color=colors[k], zorder=z[k], linewidth=0)
    # critical-path bracket over device 3's row
    on3, off3 = bounds[3]
    # the end-of-step reduction is the long canvas-gradient all-reduce (the short ones after the
    # construction are the interface-adjoint collectives of the scan)
    red3 = max(((e - s, s) for d, s, e, n in win if d == 3 and "AllReduce" in n and s > off3))[1]
    segs = [("chain fill", 0.0, (on3 - S) / 1e6), ("Jacobian construction", (on3 - S) / 1e6, (off3 - S) / 1e6),
            ("local backward", (off3 - S) / 1e6, (red3 - S) / 1e6), ("reductions", (red3 - S) / 1e6, period)]
    y = -0.62
    for lab, t0, t1 in segs:
        ax.annotate("", xy=(t0, y), xytext=(t1, y), arrowprops=dict(arrowstyle="|-|", color="#1f1f1d", linewidth=0.8, shrinkA=0, shrinkB=0, mutation_scale=3))
        ax.text((t0 + t1) / 2, y - 0.22, f"{lab}\n{t1 - t0:.1f} ms", ha="center", va="top", fontsize=6.5, color="#1f1f1d")
    for d in range(4):
        sends = [s for dd, s, e, n in win if dd == d and "SendRecv" in n]
        print(f"device {d}: first send/recv {round((min(sends)-S)/1e6,1)} ms, construction {round((bounds[d][0]-S)/1e6,1)}-{round((bounds[d][1]-S)/1e6,1)} ms")
    print(f"step period {period:.1f} ms; device 3 critical path: " + ", ".join(f"{lab} {t1-t0:.1f}" for lab, t0, t1 in segs))
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([f"device {i}" for i in range(4)], fontsize=8)
    ax.set_ylim(-1.75, 3.55)
    ax.set_xlabel("time within one training step (ms)", fontsize=8)
    ax.tick_params(labelsize=8); ax.set_xlim(0, period)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(color=colors[k], label=labels[k]) for k in ("forward", "construction", "backward", "chain", "reduce")],
              fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, framealpha=0.95, edgecolor="#e6e6e2")
    fig.tight_layout()
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
