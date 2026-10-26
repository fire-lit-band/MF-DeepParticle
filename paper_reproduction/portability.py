"""Resolve archived paths and verify portable identities without changing inputs.

Saved arrays, checkpoints and manifests retain their original bytes.  Legacy
absolute paths are interpreted only as repository-relative identities: this
module never opens files at their old machine locations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Mapping


_LEGACY_REPOSITORY = "overleaf-6ab213532624b316eadcbdcc"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def resolve_recorded_path(
    value: str | Path,
    *,
    root: Path,
    allowed_root: Path | None = None,
    base: Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Map a saved path into root, rejecting traversal and outside symlinks.

    Relative paths use ``base`` (the repository root by default).  Both slash
    styles are accepted.  An absolute path already inside this checkout is
    accepted; an archived absolute path must contain the original repository
    directory name.  Only its suffix is used, never its original disk path.
    ``allowed_root`` can further restrict a caller to one dataset directory.
    """
    root = Path(root).resolve(strict=True)
    base = root if base is None else Path(base).resolve()
    allowed = root if allowed_root is None else Path(allowed_root).resolve()
    if not base.is_relative_to(root) or not allowed.is_relative_to(root):
        raise ValueError("Path base and allowed directory must be inside the repository")
    text = str(value).replace("\\", "/")
    if not text or "\x00" in text:
        raise ValueError("Empty or invalid recorded path")
    parts = PurePosixPath(text).parts
    if ".." in parts:
        raise ValueError(f"Parent traversal is not allowed in a recorded path: {value}")
    windows = PureWindowsPath(text)
    if windows.drive and not windows.is_absolute():
        raise ValueError(f"Drive-relative paths are not supported: {value}")
    native = Path(text)
    absolute = native.is_absolute() or windows.is_absolute()
    if absolute:
        # Only inspect a native absolute path if it is lexically in this root.
        # Do not resolve an old path that could still exist on the host machine.
        if native.is_absolute() and native.is_relative_to(root):
            candidate = native
        else:
            markers = [i for i, part in enumerate(parts)
                       if part.casefold() == _LEGACY_REPOSITORY.casefold()]
            if len(markers) != 1 or markers[0] == len(parts) - 1:
                raise ValueError(f"Absolute recorded path is outside the repository: {value}")
            candidate = root.joinpath(*parts[markers[0] + 1:])
    else:
        candidate = base.joinpath(*parts)
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(root) or not resolved.is_relative_to(allowed):
        raise ValueError(f"Recorded path escapes its allowed repository directory: {value}")
    return resolved


def verified_source_identity(source: str | Path, *, root: Path) -> dict[str, str]:
    """Verify published source bytes and return the unchanged original identity.

    ``source_provenance.json`` links the original source SHA256 stored in data
    and checkpoints to the published source SHA256.  Publication edits to these
    two source files are documentation-only, independently checked during export.
    A missing or unfinished provenance entry is an error, not a bypass.
    """
    root = Path(root).resolve(strict=True)
    path = resolve_recorded_path(source, root=root)
    relative = path.relative_to(root).as_posix()
    provenance = json.loads((root / "paper_reproduction/source_provenance.json").read_text(encoding="utf-8"))
    record = provenance.get(relative)
    if not isinstance(record, dict):
        raise ValueError(f"Source is absent from the publication provenance: {relative}")
    original, published = record.get("original_sha256"), record.get("published_sha256")
    if not isinstance(original, str) or not _SHA256.fullmatch(original):
        raise ValueError(f"Invalid original source SHA256: {relative}")
    if not isinstance(published, str) or not _SHA256.fullmatch(published):
        raise ValueError(f"Missing or invalid published source SHA256: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != published:
        raise ValueError(f"Published physical source changed: {relative}")
    return {"source_file": relative, "source_sha256": original}


def canonical_protocol(protocol: Mapping, *, root: Path) -> dict:
    """Normalize only the verified source location; preserve all other fields."""
    result = dict(protocol)
    if "source_file" in result:
        identity = verified_source_identity(result["source_file"], root=root)
        if result.get("source_sha256") != identity["source_sha256"]:
            raise ValueError("Original physical source SHA256 differs from publication provenance")
        result["source_file"] = identity["source_file"]
    return result


def protocols_match(actual: Mapping, expected: Mapping, *, root: Path) -> bool:
    """Compare every protocol field after validating and normalizing its source."""
    return canonical_protocol(actual, root=root) == canonical_protocol(expected, root=root)
