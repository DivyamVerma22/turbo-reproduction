"""TuRBO reproduction — Eriksson et al., NeurIPS 2019 (arXiv 1910.01739).

"Scalable Global Optimization via Local Bayesian Optimization".

`PAPER_SPEC.md` is the implementation contract for this package. Every module here
cites the paper section it implements. See `REPRODUCTION_NOTES.md` for deviations.
"""

from .turbo_1 import Turbo1
from .turbo_m import TurboM

__all__ = ["Turbo1", "TurboM"]
