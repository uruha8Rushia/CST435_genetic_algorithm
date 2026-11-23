"""Scalability benchmark for the CST435 genetic algorithm.

Runs the sequential and OpenMP programs over several population sizes and
thread counts, then reports execution time, speedup and efficiency:

    speedup    S(p) = T_sequential / T_parallel(p)
    efficiency E(p) = S(p) / p

Each configuration is run several times and the median time is used.
The script also checks that every parallel run ends with the same total
fitness as the sequential run (same seed, so the results must match).

Usage (run from the folder that contains population.txt):
    python bench.py --seq ./sequential --par ./parallel
    python bench.py --seq sequential.exe --par parallel.exe --pops 1000 2000 4000 --threads 1 2 4 8

Writes results.csv and results.md, plus speedup.png and efficiency.png when
matplotlib is installed.
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys


def run(cmd):
    """Run one program and return (seconds, total_fitness) from its RESULT line."""
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("RESULT,"):
            fields = line.split(",")
            return float(fields[5]), int(fields[6])
    sys.exit("No RESULT line in output of: " + " ".join(cmd) + "\n" + out)


def median_run(cmd, runs):
    times, totals = [], set()
    for _ in range(runs):
        seconds, total = run(cmd)
        times.append(seconds)
        totals.add(total)
    if len(totals) != 1:
        sys.exit("Results differ between runs of: " + " ".join(cmd))
    return statistics.median(times), totals.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="path to the sequential executable")
    ap.add_argument("--par", required=True, help="path to the OpenMP executable")
    ap.add_argument("--pops", type=int, nargs="+", default=[1000, 2000, 4000], help="population sizes")
    ap.add_argument("--iters", type=int, default=500, help="generations per run")
    ap.add_argument("--threads", type=int, nargs="+", default=None,
                    help="thread counts (default: 1, 2, 4, ... up to the number of CPU cores)")
    ap.add_argument("--runs", type=int, default=3, help="repetitions per configuration (median is used)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--input", default="population.txt", help="population file")
    args = ap.parse_args()

    if args.threads is None:
        cores = os.cpu_count() or 1
        args.threads = sorted({1, cores} | {t for t in (2, 4, 8, 16, 32) if t <= cores})

    common = ["-i", str(args.iters), "-s", str(args.seed), "-f", args.input, "-q"]
    rows = []
    for pop in args.pops:
        t_seq, fit_seq = median_run([args.seq, "-p", str(pop)] + common, args.runs)
        print(f"population {pop}: sequential {t_seq:.4f} s")
        for p in args.threads:
            t_par, fit_par = median_run([args.par, "-p", str(pop), "-t", str(p)] + common, args.runs)
            if fit_par != fit_seq:
                sys.exit(f"Mismatch at population {pop}, {p} threads: {fit_par} vs {fit_seq}")
            speedup = t_seq / t_par
            rows.append({
                "population": pop, "iterations": args.iters, "threads": p,
                "t_sequential_s": round(t_seq, 6), "t_parallel_s": round(t_par, 6),
                "speedup": round(speedup, 3), "efficiency": round(speedup / p, 3),
            })
            print(f"    {p:>3} threads: {t_par:.4f} s  speedup {speedup:.2f}  efficiency {speedup / p:.0%}")

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open("results.md", "w") as f:
        f.write(f"CPU cores: {os.cpu_count()}, iterations: {args.iters}, "
                f"median of {args.runs} runs. All parallel results matched the sequential result.\n\n")
        f.write("| Population | Threads | Sequential (s) | Parallel (s) | Speedup | Efficiency |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['population']} | {r['threads']} | {r['t_sequential_s']:.4f} | "
                    f"{r['t_parallel_s']:.4f} | {r['speedup']:.2f} | {r['efficiency']:.0%} |\n")
    print("\nWrote results.csv and results.md")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the graphs (pip install matplotlib)")
        return

    for metric, label, ideal in (("speedup", "Speedup", lambda t: t), ("efficiency", "Efficiency", lambda t: 1.0)):
        fig, ax = plt.subplots(figsize=(6, 4))
        for pop in args.pops:
            data = [r for r in rows if r["population"] == pop]
            ax.plot([r["threads"] for r in data], [r[metric] for r in data], marker="o", label=f"population {pop}")
        ax.plot(args.threads, [ideal(t) for t in args.threads], linestyle="--", color="grey", label="ideal")
        ax.set_xlabel("Threads")
        ax.set_ylabel(label)
        ax.set_xticks(args.threads)
        ax.set_title(f"{label} vs threads ({args.iters} generations)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{metric}.png", dpi=150)
        plt.close(fig)
    print("Wrote speedup.png and efficiency.png")


if __name__ == "__main__":
    main()
