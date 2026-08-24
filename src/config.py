"""Single source of truth for every numerical constant in the reproduction.

Constants are grouped by **provenance**, which is the distinction that matters here and
must stay visible at the point of definition (CLAUDE.md rules 3-4):

    SPECIFIED          stated in the paper                     -> cite section/appendix
    FROM_OFFICIAL_CODE only in uber-research/TuRBO@master       -> cite file:line
    ASSUMPTION         chosen here, with no source              -> cite REPRODUCTION_NOTES

Collapsing these into one undifferentiated block of numbers would erase the difference
between "the authors specified this" and "the authors' code happened to do this", which is
exactly what `PAPER_SPEC.md` §10 exists to preserve.

Nothing here changes a value: this module gathers the constants that previously lived in
`trust_region.py`, `gp.py`, `turbo_1.py` and `evaluate.py`. Those modules re-export the
names they published before, so existing imports keep working.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "TR_DEFAULTS",
    "HYPER_BOUNDS",
    "GP_INIT",
    "SYNTHETIC_PROTOCOL",
    "SUCCESS_TOL",
    "ADAM_LR",
    "N_TRAINING_STEPS",
    "MAX_CHOLESKY_SIZE",
    "N_CAND_PER_DIM",
    "N_CAND_MAX",
    "PERTURB_NUMERATOR",
    "CENTER_STAT",
    "USE_PREDICTIVE_DRAWS",
    "SOBOL_SEED_MAX",
    "TORCH_SEED_MAX",
]


# --- SPECIFIED: trust-region schedule (App. D) ---------------------------------------
# "In all experiments, we use the following hyperparameters for TuRBO-1: tau_succ = 3,
#  tau_fail = ceil(d/q), L_min = 2^-7, L_max = 1.6, and L_init = 0.8"
# tau_fail is dimension- and batch-dependent, so it lives in trust_region.default_failtol.
TR_DEFAULTS: Final[dict[str, float | int]] = {
    "succtol": 3,             # tau_succ, App. D
    "length_min": 2.0**-7,    # L_min,    App. D
    "length_max": 1.6,        # L_max,    App. D
    "length_init": 0.8,       # L_init,   App. D
}

# --- SPECIFIED: candidate set (App. D) -----------------------------------------------
# "a candidate set of size min{100d, 5000}" and "we use the value in the Sobol sequence
#  with probability min{1, 20/d}"
N_CAND_PER_DIM: Final[int] = 100
N_CAND_MAX: Final[int] = 5000
PERTURB_NUMERATOR: Final[float] = 20.0

# --- SPECIFIED: GP hyperparameter box constraints (App. C) ---------------------------
# "(lengthscale) lambda_i in [0.005, 2.0], (signal variance) s^2 in [0.05, 20.0],
#  (noise variance) sigma^2 in [0.0005, 0.1]"
#
# NOTE (PAPER_SPEC.md §10 A5): the official code uses Interval(5e-4, 0.2) for the noise
# (gp.py L48). We follow the PAPER's 0.1 -- see REPRODUCTION_NOTES.md §2 P3.
HYPER_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "lengthscale": (0.005, 2.0),
    "outputscale": (0.05, 20.0),
    "noise": (0.0005, 0.1),
}

# --- SPECIFIED: App. A synthetic protocol ---------------------------------------------
# "a budget of 50 batches of size q = 10 which results in a total of n = 500 function
#  evaluations. All methods use 20 initial points from a Latin hypercube design (LHD)
#  except for TuRBO-5, where we use 10 initial points in each local region. To compute
#  confidence intervals on the results, we use 30 runs."
SYNTHETIC_PROTOCOL: Final[dict[str, int]] = {
    "max_evals": 500,
    "batch_size": 10,
    "n_init": 20,
    "n_init_per_tr": 10,
    "n_trust_regions": 5,
    "n_replications": 30,
}

# --- FROM_OFFICIAL_CODE: GP fitting ---------------------------------------------------
# The paper says only "optimizing the log-marginal likelihood" (App. C) -- it names no
# optimizer, learning rate, step count or initialization. All of these come from
# uber-research/TuRBO. See REPRODUCTION_NOTES.md §4 C1-C2.
ADAM_LR: Final[float] = 0.1                     # gp.py L85
N_TRAINING_STEPS: Final[int] = 50               # turbo_1.py L61
MAX_CHOLESKY_SIZE: Final[int] = 2000            # turbo_1.py L60; App. C's CG/Lanczos switch
GP_INIT: Final[dict[str, float]] = {            # gp.py L79-L81
    "covar_module.outputscale": 1.0,
    "covar_module.base_kernel.lengthscale": 0.5,
    "likelihood.noise": 0.005,
}

# --- FROM_OFFICIAL_CODE: behavioral defaults where paper and code differ --------------
# Each of these is a documented paper/official-code conflict. The default reproduces the
# official code; the paper-literal alternative is reachable through the noted argument.
# See REPRODUCTION_NOTES.md §1.

# D1: Sect. 2 says a success is "a candidate that improves upon x*" with no tolerance;
# the code requires a relative margin (turbo_1.py L138). success_tol=0.0 restores the paper.
SUCCESS_TOL: Final[float] = 1e-3

# D2: App. C says values are "standardized" (conventionally mean/std); the code centers on
# the median (turbo_1.py L159). center_stat="mean" restores the ordinary reading.
CENTER_STAT: Final[str] = "median"

# D3: Sect. 2 writes f ~ GP(mu, k), the LATENT posterior; the code samples the predictive
# distribution, adding sigma^2 noise (turbo_1.py L216). use_predictive=False restores it.
USE_PREDICTIVE_DRAWS: Final[bool] = True

# --- ASSUMPTION -----------------------------------------------------------------------
# Upper bound of the per-batch scramble seed drawn for SobolEngine. The official code uses
# np.random.randint(int(1e6)) (turbo_1.py L191); we draw the same range from the run's own
# Generator so a run stays reproducible from one seed. REPRODUCTION_NOTES.md §4 C5/C11.
SOBOL_SEED_MAX: Final[int] = 1_000_000

# Upper bound of the per-draw seed used to fork and seed torch's global RNG in
# thompson_draws. GPyTorch samples from the global generator, so the draw must be
# seeded explicitly for a run to be reproducible. REPRODUCTION_NOTES.md §4 C11.
TORCH_SEED_MAX: Final[int] = 2**31 - 1
