"""Standalone numerical reproduction tools for KS, KPP and cubic reaction experiments."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deepparticle'))
_dll_handles = []
if os.name == 'nt':
    library = Path(sys.prefix) / 'Library' / 'bin'
    if library.is_dir():
        _dll_handles.append(os.add_dll_directory(str(library)))

