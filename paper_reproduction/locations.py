"""Protect package inputs while creating independent experiment outputs."""
from pathlib import Path
from . import ROOT


def fresh_directory(path: Path, *, create: bool = False) -> Path:
    path = Path(path).expanduser().resolve()
    protected = [ROOT / name for name in ('deepparticle', 'output', 'paper_reproduction', '.git', '.ol-sync')]
    if path == ROOT or any(path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Choose an output directory outside the package input/code directories.')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Use a new or empty output directory: {path}')
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
