"""Verify the current public files against their SHA-256 manifest."""
import csv
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / 'SHA256SUMS.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or invalid release path: ' + row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Checksum mismatch: ' + row['path'])
    print(f'All {len(rows)} public-file checksums match.')


if __name__ == '__main__':
    main()
