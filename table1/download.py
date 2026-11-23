"""Download, verify, and extract the released numerical files."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-only', action='store_true')
    args = parser.parse_args(); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
    for item in assets['assets']:
        if args.evaluation_only and item['name'] not in ('table1_test.zip','table1_checkpoints.zip'):
            continue
        archive = root/item['name']
        if not archive.exists() or digest(archive) != item['sha256']:
            part = archive.with_suffix(archive.suffix+'.part')
            request = urllib.request.Request(item['url'], headers={'User-Agent':'PhySMI'})
            with urllib.request.urlopen(request, timeout=120) as response, part.open('wb') as f:
                while True:
                    block = response.read(1024*1024)
                    if not block: break
                    f.write(block)
            if digest(part) != item['sha256']: raise RuntimeError('Download checksum mismatch: '+item['name'])
            part.replace(archive)
        with ZipFile(archive) as z:
            for info in z.infolist():
                target = (root/info.filename).resolve()
                if not target.is_relative_to(root) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe archive member')
            z.extractall(root)
        print('Ready:', item['name'], flush=True)
    manifest = json.loads((ROOT/'dataset.json').read_text(encoding='utf-8'))
    files = [] if args.evaluation_only else manifest['train'] + [r for pool in manifest['priors'] for r in pool]
    for case in manifest['test']:
        files.extend({'file':case[k], 'sha256':case[k+'_sha256']} for k in ('input','reference','nnls'))
    for item in files:
        if digest(root/item['file']) != item['sha256']: raise RuntimeError('Extracted checksum mismatch: '+item['file'])
    print('All extracted data verified.', flush=True)

if __name__ == '__main__': main()
