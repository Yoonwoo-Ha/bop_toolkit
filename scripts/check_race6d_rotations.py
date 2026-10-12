"""Check raw BOP pose annotations without changing them or exporting images."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bop_toolkit_lib.race6d_rotations import main


if __name__ == "__main__":
    main()
