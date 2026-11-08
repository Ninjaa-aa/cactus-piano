"""Fetch the pinned, small CC BY 3.0 Salamander evaluation subset."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    dest = ROOT / 'data'
    dest.mkdir(exist_ok=True)
    base = f"https://raw.githubusercontent.com/Tonejs/audio/{protocol['source_commit']}/salamander/"
    manifest = []
    for split in ('development', 'evaluation'):
        for note in protocol[split]:
            name = note + '.mp3'
            path = dest / name
            url = base + name
            if not path.exists():
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read(2_000_001)
                if len(data) > 2_000_000 or len(data) < 1000:
                    raise ValueError(f'Unexpected download size: {name}')
                path.write_bytes(data)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append(dict(file=name, split=split, url=url, sha256=digest, bytes=path.stat().st_size))
    existing = dest / 'manifest.json'
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError('Source checksum drift: preserve the existing manifest and investigate.')
    existing.write_text(json.dumps(manifest, indent=2) + '\n')
    with urllib.request.urlopen(base + 'README', timeout=30) as response:
        (dest / 'SOURCE_README.txt').write_bytes(response.read())
    print(f'{len(manifest)} pinned recordings; {sum(r["bytes"] for r in manifest):,} bytes')

if __name__ == '__main__':
    main()
