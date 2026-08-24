#!/usr/bin/env python
"""Minimal reproduction experiment: TuRBO-1 vs. baselines on the 10D Ackley function.

    python scripts/reproduce_minimal.py

------------------------------------------------------------------------------------
1. TARGETED PAPER RESULT
------------------------------------------------------------------------------------
Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization",
NeurIPS 2019 (arXiv 1910.01739) -- Appendix A, Figure 8.

Two claims, both quoted verbatim:

  (C1) "TuRBO-1 and TuRBO-5 outperform other methods on Ackley and consistently find
        solutions close to the global optimum."                            -- App. A
  (C2) "TuRBO consistently finds excellent solutions, outperforming the other methods
        on most problems."                                                 -- Sect. 3

Why this target and not a headline benchmark: every PRECISE number the paper reports
(robot pushing "mean and median reward of around 9.4", Sect. 3.1; rover "about 2",
Sect. 3.2; the 1.284 vs 1.174 nats of Sect. 3.6) depends on external simulators that
this paper does not specify -- see REPRODUCTION_NOTES.md §6. Appendix A's synthetic
suite is the only part of the evaluation that is fully specified in the paper, so it is
the only part that can be checked honestly.

------------------------------------------------------------------------------------
WHAT THIS EXPERIMENT CANNOT DO
------------------------------------------------------------------------------------
Appendix A reports Figure 8 as a PLOT. The paper states no numeric value for Ackley-10,
so there is no published number to match against. This experiment therefore checks the
two claims above operationally:

  (C1) -> is the mean final value close to the known global optimum of Ackley (0.0)?
  (C2) -> does TuRBO beat every baseline it is run against, by more than one standard
          error?

The "close to the optimum" threshold below is OUR operationalization of the paper's
words, not a paper-stated quantity. It is declared in the config so it can be argued
with. A MATCH here means "consistent with the paper's claims", never "reproduces the
paper's numbers" -- the paper reports none for this function.

------------------------------------------------------------------------------------
DIFFERENCES FROM THE PAPER'S SETUP
------------------------------------------------------------------------------------
  * Functions:  Ackley-10 only. App. A runs four (Ackley, Levy, Rastrigin, Hartmann6).
  * Baselines:  random search, Nelder-Mead and BFGS. App. A also compares BOBYQA,
                CMA-ES, EBO, BOCK, BOHAMIANN, GP-TS and HeSBO-TS; those need optional
                or unimplemented dependencies (PAPER_SPEC.md §2, App. B).
  * Algorithms: TuRBO-1 only. App. A also reports TuRBO-5.
  * Hardware:   CPU, float64. App. G used an NVIDIA RTX 2080 TI.
  * Everything else follows App. A exactly: 500 evaluations, q=10, 20 initial points
    from a Latin hypercube design, 30 replications, mean +/- one standard error.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import baselines as B  # noqa: E402
from src.benchmarks import get_benchmark  # noqa: E402
from src.evaluate import best_so_far, run_replications  # noqa: E402


# ---------------------------------------------------------------------------------
# PINNED CONFIGURATION -- App. A. Do not edit to make a run pass.
# ---------------------------------------------------------------------------------
CONFIG: dict = {
    "paper": "arXiv:1910.01739, Appendix A / Figure 8",
    "benchmark": "ackley10",
    "domain": "[-5, 10]^10",             # App. A
    "global_minimum": 0.0,               # standard Ackley optimum, at the origin
    "max_evals": 500,                    # App. A: "a total of n = 500 function evaluations"
    "batch_size": 10,                    # App. A: "50 batches of size q = 10"
    "n_init": 20,                        # App. A: "20 initial points from a Latin hypercube design"
    "n_replications": 30,                # App. A: "we use 30 runs"
    "base_seed": 0,
    "algorithm": "turbo-1",
    "baselines": ["random_search", "nelder_mead", "bfgs"],
    # Operationalization of App. A's "close to the global optimum". OURS, not the paper's.
    "close_to_optimum_threshold": 1.0,
    # A win must clear one standard error, matching how Fig. 8 is drawn
    # (Sect. 3: "mean performances with one standard error").
    "margin_in_standard_errors": 1.0,
}


@dataclass
class ArmResult:
    """Final best value per replication for one optimizer."""

    name: str
    finals: np.ndarray
    seconds: float

    @property
    def mean(self) -> float:
        return float(self.finals.mean())

    @property
    def sem(self) -> float:
        n = len(self.finals)
        return float(self.finals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.finals))


def environment() -> dict:
    """Record everything needed to explain a different result on another machine."""

    def version(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:  # pragma: no cover - reporting only
            return "not installed"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:  # pragma: no cover - reporting only
        commit, dirty = "unknown", False

    import torch

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "torch": version("torch"),
        "gpytorch": version("gpytorch"),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "dtype": "float64",
        "cuda_available": torch.cuda.is_available(),
    }


def run_turbo(cfg: dict) -> tuple[ArmResult, np.ndarray]:
    """TuRBO-1 under the App. A protocol. Returns the arm result and per-rep traces."""
    t0 = time.perf_counter()
    results = run_replications(
        benchmark=cfg["benchmark"],
        algorithm=cfg["algorithm"],
        n_replications=cfg["n_replications"],
        base_seed=cfg["base_seed"],
        n_init=cfg["n_init"],
        max_evals=cfg["max_evals"],
        batch_size=cfg["batch_size"],
    )
    seconds = time.perf_counter() - t0
    finals = np.array([r.best_value for r in results], dtype=np.float64)
    traces = np.vstack([r.trace for r in results])
    return ArmResult("TuRBO-1", finals, seconds), traces


def run_baseline(name: str, cfg: dict) -> tuple[ArmResult, np.ndarray]:
    """One App. B baseline, on the same objective and the same evaluation budget."""
    bench = get_benchmark(cfg["benchmark"])
    finals, traces = [], []
    t0 = time.perf_counter()
    for i in range(cfg["n_replications"]):
        seed = cfg["base_seed"] + i
        if name == "random_search":
            out = B.random_search(bench, cfg["max_evals"], seed=seed)
        else:
            out = getattr(B, name)(bench, cfg["max_evals"], seed=seed, n_init=cfg["n_init"])
        finals.append(out["best_value"])
        trace = best_so_far(out["fX"])
        # Baselines may stop early (converged simplex); pad with the final value so all
        # traces share the evaluation axis of Fig. 8.
        if len(trace) < cfg["max_evals"]:
            trace = np.concatenate([trace, np.full(cfg["max_evals"] - len(trace), trace[-1])])
        traces.append(trace[: cfg["max_evals"]])
    return ArmResult(name, np.array(finals, dtype=np.float64),
                     time.perf_counter() - t0), np.vstack(traces)


def classify(turbo: ArmResult, baselines: list[ArmResult], cfg: dict) -> tuple[str, list[str]]:
    """Label the outcome MATCH / PARTIAL MATCH / NOT REPRODUCED, with reasons."""
    reasons: list[str] = []
    margin = cfg["margin_in_standard_errors"]

    # (C1) close to the global optimum
    gap = turbo.mean - cfg["global_minimum"]
    near_optimum = gap <= cfg["close_to_optimum_threshold"]
    reasons.append(
        f"C1 close to optimum: TuRBO-1 mean {turbo.mean:.4f} is {gap:.4f} from the optimum "
        f"{cfg['global_minimum']:.1f} (our threshold {cfg['close_to_optimum_threshold']}) "
        f"-> {'PASS' if near_optimum else 'FAIL'}"
    )

    # (C2) outperforms every baseline by more than one standard error
    beaten = []
    for b in baselines:
        wins = (turbo.mean + margin * turbo.sem) < (b.mean - margin * b.sem)
        beaten.append(wins)
        reasons.append(
            f"C2 vs {b.name}: TuRBO {turbo.mean:.4f}+/-{turbo.sem:.4f} vs "
            f"{b.mean:.4f}+/-{b.sem:.4f} -> {'PASS' if wins else 'FAIL'}"
        )
    beats_all = all(beaten)

    if near_optimum and beats_all:
        verdict = "MATCH"
    elif near_optimum or any(beaten):
        verdict = "PARTIAL MATCH"
    else:
        verdict = "NOT REPRODUCED"
    return verdict, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replications", type=int, default=None,
                    help="override App. A's 30 runs (documented as a deviation if used)")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results",
                    help="default: <repo>/results, so the run works from any directory")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    deviations: list[str] = []
    if args.replications is not None and args.replications != CONFIG["n_replications"]:
        deviations.append(
            f"n_replications={args.replications} instead of App. A's {CONFIG['n_replications']}; "
            "error bars are NOT the paper's"
        )
        cfg["n_replications"] = args.replications

    env = environment()
    print("=" * 78)
    print("MINIMAL REPRODUCTION -- TuRBO vs baselines on Ackley-10")
    print("Target:", cfg["paper"])
    print("=" * 78)
    print("\n[environment]")
    for k, v in env.items():
        print(f"  {k:18s} {v}")
    if env["git_dirty"]:
        print("  WARNING: working tree is dirty; this run is not tied to a clean commit")
    print("\n[pinned config]")
    for k, v in cfg.items():
        print(f"  {k:28s} {v}")
    print(f"\n[seeds] {cfg['base_seed']}..{cfg['base_seed'] + cfg['n_replications'] - 1}")

    print("\n[running] TuRBO-1 ...", flush=True)
    turbo, turbo_traces = run_turbo(cfg)
    print(f"  done in {turbo.seconds:.1f}s")

    baseline_arms, baseline_traces = [], {}
    for name in cfg["baselines"]:
        print(f"[running] {name} ...", flush=True)
        arm, traces = run_baseline(name, cfg)
        baseline_arms.append(arm)
        baseline_traces[name] = traces
        print(f"  done in {arm.seconds:.1f}s")

    verdict, reasons = classify(turbo, baseline_arms, cfg)

    print("\n" + "-" * 78)
    print(f"{'optimizer':<16}{'mean':>12}{'+/- 1 SE':>12}{'median':>12}{'best':>12}{'worst':>12}")
    print("-" * 78)
    for arm in [turbo, *baseline_arms]:
        print(f"{arm.name:<16}{arm.mean:>12.4f}{arm.sem:>12.4f}{arm.median:>12.4f}"
              f"{arm.finals.min():>12.4f}{arm.finals.max():>12.4f}")
    print("-" * 78)
    print(f"{'global optimum':<16}{cfg['global_minimum']:>12.4f}")

    print("\n[claim checks]")
    for r in reasons:
        print("  " + r)

    print("\n" + "=" * 78)
    print(f"OUTCOME: {verdict}")
    print("=" * 78)
    print("Scope: consistent-with-claims only. App. A reports Figure 8 as a plot and states")
    print("no numeric value for Ackley-10, so no published number is being matched.")
    if deviations:
        print("\nDeviations from the pinned protocol:")
        for d in deviations:
            print("  - " + d)

    # ---- raw metrics --------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "target": cfg["paper"],
        "claims": {
            "C1": "TuRBO-1 and TuRBO-5 ... consistently find solutions close to the "
                  "global optimum. (App. A)",
            "C2": "TuRBO consistently finds excellent solutions, outperforming the other "
                  "methods on most problems. (Sect. 3)",
        },
        "verdict": verdict,
        "verdict_scope": "consistent-with-claims; the paper reports no number for Ackley-10",
        "reasons": reasons,
        "config": cfg,
        "environment": env,
        "deviations_from_paper": deviations + [
            "Ackley-10 only; App. A runs four synthetic functions",
            "baselines limited to random search, Nelder-Mead, BFGS (App. B lists more)",
            "TuRBO-1 only; App. A also reports TuRBO-5",
            "CPU float64; App. G used an NVIDIA RTX 2080 TI",
        ],
        "results": {
            arm.name: {
                "final_values": arm.finals.tolist(),
                "mean": arm.mean,
                "standard_error": arm.sem,
                "median": arm.median,
                "seconds": arm.seconds,
            }
            for arm in [turbo, *baseline_arms]
        },
    }
    json_path = args.output_dir / f"reproduce_minimal_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    npz_path = args.output_dir / f"reproduce_minimal_{stamp}_traces.npz"
    np.savez_compressed(npz_path, turbo_1=turbo_traces,
                        **{f"baseline_{k}": v for k, v in baseline_traces.items()})

    print(f"\n[raw metrics] {json_path}")
    print(f"[raw traces ] {npz_path}  (per-replication best-so-far, shape "
          f"{turbo_traces.shape} = reps x evaluations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
