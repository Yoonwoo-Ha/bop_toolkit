#!/usr/bin/env python3
"""Convert raw BOP data to the complete COCO layout consumed by RACE6D."""

from pathlib import Path
import sys

# A checkout's script must use its matching library, not a different installed
# bop_toolkit_lib. This also permits invoking the script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bop_toolkit_lib.race6d_coco import main


if __name__ == "__main__":
    main()
