"""Run this working copy, never the original agent package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import main

if __name__ == "__main__":
    raise SystemExit(main())
