"""Make `src/` importable so tests can `import market`.

If nba-forecast imports through a `src.` prefix instead, drop this file and
change the test imports to `from src.market ...`.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
