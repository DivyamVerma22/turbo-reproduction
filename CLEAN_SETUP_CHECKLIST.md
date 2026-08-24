# CLEAN_SETUP_CHECKLIST.md

From `git clone` to a passing smoke test on a machine that has never seen this project.

Verified by cloning this repository into an empty directory and running the suite there — see
§6 for exactly what that check covers and what it cannot.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| **Python ≥ 3.10** | Developed on 3.11.0. The code uses PEP 604 `X \| Y` annotations evaluated at runtime in dataclass fields, so 3.9 will fail at import. |
| **git** | Only for cloning. The reproduction script also calls `git rev-parse` to stamp results with a commit, and degrades to `"unknown"` if git is absent. |
| ~2 GB disk | Almost entirely the PyTorch wheel. |
| **No GPU** | Nothing in `src/` requests one. A CPU-only torch build is sufficient and much smaller. |

**Not required:** no environment variables, no API keys, no network access at run time, no external
data files, no database, no Docker, no build step.

---

## 1. Clone

```bash
git clone <repository-url> paper-reproduction
cd paper-reproduction
```

Run every command below **from the repository root**. (The reproduction script resolves its own
paths, so it works from anywhere, but `pytest` expects the root.)

## 2. Create an isolated environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3. Install dependencies

Install the CPU build of PyTorch first, so pip does not pull a multi-gigabyte CUDA build:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On a machine where you *want* the default (possibly CUDA) build, skip the first line — the code runs
identically on CPU either way.

Versions in `requirements.txt` are pinned to exactly what was used to produce the numbers in
`VERIFICATION.md` and `results/`. Newer versions very likely work; if you change one, re-run the
tests and treat any difference as a finding rather than noise.

**Optional extras** — only for two Appendix B baselines. Leave them out unless you need them:

```bash
pip install cma nlopt     # CMA-ES and BOBYQA
```

Without them, `src/baselines.py` raises a clear `ImportError` naming the missing package. Everything
else, including the whole test suite and the reproduction script, runs without them.

## 4. Smoke test (about 20 seconds)

```bash
pytest tests/test_smoke.py -q
```

Expected: **15 passed**. This walks the minimal end-to-end path in stages — tiny fixture → GP forward
pass → marginal likelihood → one Adam step → one full TuRBO iteration → complete `Turbo1` and
`TurboM` runs — so a broken install fails at the earliest stage rather than opaquely.

## 5. Full test suite (about 20 seconds)

```bash
pytest tests/ -q
```

Expected: **193 passed**. A handful of `DeprecationWarning`/`GPInputWarning` lines from torch and
GPyTorch are normal.

Useful variants:

```bash
pytest tests/test_trust_region.py -q                          # one module
pytest tests/test_trust_region.py::test_volume_invariant -q   # a single test
```

## 6. Reproduce the paper claim (about 10 minutes)

```bash
python scripts/reproduce_minimal.py
```

Runs Appendix A's pinned protocol on Ackley-10 (500 evaluations, `q=10`, 20 initial LHD points, 30
replications) for TuRBO-1 against random search, Nelder-Mead and BFGS. Prints the environment, the
pinned config, the seed range, a comparison table, and an outcome of **MATCH / PARTIAL MATCH /
NOT REPRODUCED**. Writes raw per-replication metrics and best-so-far traces to `results/`.

For a quick check instead, `--replications 5` finishes in ~2 minutes and labels itself as a deviation
from the paper protocol in both the console output and the saved JSON.

**Read the verdict narrowly.** Appendix A reports Figure 8 as a plot and states no numeric value for
Ackley-10, so MATCH means "consistent with the paper's claims", never "reproduces the paper's
numbers". The full reasoning is in the script's header.

---

## 7. Expect small numerical differences on your machine

Runs are reproducible **on a fixed machine**: a seed determines the whole trajectory, including the
Thompson draws (`src/thompson.py` forks and seeds torch's global RNG, because GPyTorch samples from
it and accepts no generator argument).

Across machines, expect small differences. GP fitting runs 50 Adam steps of a Cholesky/CG solve in
float64, and different BLAS builds, CPU instruction sets and thread counts reorder those
floating-point reductions. The *statistics* should agree within the reported standard error; do not
expect identical trajectories. `scripts/reproduce_minimal.py` logs `platform`, `processor`,
`torch_threads` and every library version precisely so a difference can be attributed.

If you need to eliminate thread-count variation: `export OMP_NUM_THREADS=1` before running.

---

## 8. What is deliberately absent

| Missing | Why |
|---|---|
| `.paper2code_work/` | Untracked scratch from a local authoring tool: the parsed paper and a copy of the official reference implementation. Nothing in `src/`, `tests/` or `scripts/` reads it. To verify quotations, use the paper (`arxiv.org/abs/1910.01739`) and the official code (`github.com/uber-research/TuRBO`) directly — every citation names its section or `file:line`. |
| Robot pushing, rover, lunar lander, cosmological constants | Not reproducible from the paper. Their simulators are external, or the paper omits the parameters and bounds. `src/benchmarks.py` declares them and raises `NotImplementedError` with the specific reason rather than silently substituting a stand-in. See `REPRODUCTION_NOTES.md` §6. |
| A packaging file (`pyproject.toml`) | The project is run in place; `conftest.py` puts the repo root on `sys.path` for pytest, and the reproduction script resolves its own root. There is nothing to build or install. |
| `results/` on a fresh clone | Created by the reproduction script. The canonical run is committed as evidence for the verdict recorded in `VERIFICATION.md`. |

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You ran pytest from outside the repository root. `cd` to the root; `conftest.py` handles the path from there. |
| `ImportError: cma_es requires pycma` / `bobyqa requires nlopt` | Expected — optional extras. Install them (§3) or avoid those two baselines. |
| pip pulls gigabytes of CUDA packages | You skipped the CPU index in §3. |
| `NotImplementedError` naming Wang et al. 2018 or a missing simulator | Expected — that benchmark is not reproducible from the paper (§8). |
| Tests pass but the reproduction run prints `git_dirty: true` | You have uncommitted changes; the run is not tied to a clean commit. Harmless for a local check, but do not cite such a run as evidence. |
| Slightly different numbers from `VERIFICATION.md` | Expected across machines — see §7. Differences beyond one standard error are worth investigating. |

## 10. Where to look next

| File | Contents |
|---|---|
| `CLAUDE.md` | Orientation: architecture, the load-bearing ordering constraints, the six paper/official-code conflicts. |
| `PAPER_SPEC.md` | The implementation contract: equations, shapes, hyperparameters with per-value citations, and a 26-row ambiguity table. |
| `REPRODUCTION_NOTES.md` | Every deviation from the paper and every undetermined choice, with the flag that restores paper-literal behavior. |
| `VERIFICATION.md` | What has been checked, what has not, and the audit findings. |
| `PERFORMANCE.md` | Profiling results and why no optimization was applied. |
