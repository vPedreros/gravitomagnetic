"""
Pytest configuration — adds python/ to sys.path so all scripts are importable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
