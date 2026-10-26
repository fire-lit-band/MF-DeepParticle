"""Install the separately distributed, checksum-verified numerical data archive."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Path to reproduction-data-v1.zip')
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve(strict=True)
    info = json.loads((ROOT / 'paper_reproduction/data_archive.json').read_text(encoding='utf-8'))
    if archive.stat().st_size != info['bytes'] or sha256(archive) != info['sha256']:
        raise ValueError('The data archive does not match data_archive.json')
    manifest = json.loads((ROOT / 'paper_reproduction/retained_data.json').read_text(encoding='utf-8'))
    expected = {r['path']: r for r in manifest['files'] if r['path'].endswith('.npz')}
    installed = skipped = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(expected) or {r.filename for r in members} != set(expected):
            raise ValueError('Archive members do not match the retained numerical inputs')
        for member in members:
            record = expected[member.filename]
            destination = (ROOT / member.filename).resolve()
            if not destination.is_relative_to(ROOT) or member.is_dir() or member.file_size != record['bytes']:
                raise ValueError(f'Invalid data member: {member.filename}')
            if destination.exists():
                if destination.is_file() and sha256(destination) == record['sha256']:
                    skipped += 1
                    continue
                raise FileExistsError(f'Refusing to replace different existing data: {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.data-install-', delete=False) as target:
                    temporary = Path(target.name)
                    with bundle.open(member) as source:
                        shutil.copyfileobj(source, target)
                if sha256(temporary) != record['sha256']:
                    raise ValueError(f'Data hash mismatch: {member.filename}')
                os.replace(temporary, destination)
                temporary = None
                installed += 1
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    print(json.dumps({'status': 'passed', 'installed': installed, 'already_present': skipped,
                      'data_files': len(expected)}, indent=2))


if __name__ == '__main__':
    main()
