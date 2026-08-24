"""Make the repository root importable so `from src... import ...` works under pytest.

Without this, a bare `pytest tests/...` fails with ModuleNotFoundError: No module named
'src' (only `python -m pytest`, which puts the CWD on sys.path, would work). CLAUDE.md
documents the bare form as the rule-9 "smallest relevant test" command, so both must work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
