# src/daaq/utils/paths.py
from pathlib import Path

# Anchors to the project root regardless of where the script is called from
# __file__ = src/daaq/utils/paths.py
# .parents[0] = src/daaq/utils/
# .parents[1] = src/daaq/
# .parents[2] = src/
# .parents[3] = daaq root

DAAQ_ROOT = Path(__file__).resolve().parents[3]
COLORMAPS_DIR = DAAQ_ROOT / "data" / "colormaps"