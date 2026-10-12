#!/usr/bin/env python3
"""Create a separate P-aligned training mesh/keypoint bundle from BOP models."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bop_toolkit_lib.race6d_models import main

if __name__ == "__main__":
    main()
