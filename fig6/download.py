"""Download the image library and verify its numerical files."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import urllib.request
import time
from simulation import ROOT,read,file_digest

def download(data):
    root=Path(data).resolve();root.mkdir(parents=True,exist_ok=True)
    for asset in read(ROOT/'release.json')['assets']:
        archive=root/asset['name']
        if not archive.exists() or file_digest(archive)!=asset['sha256']:
            part=archive.with_suffix('.zip.part')
            for attempt in range(5):
                try:
                    request=urllib.request.Request(asset['url'],headers={'User-Agent':'PhySMI'})
                    with urllib.request.urlopen(request,timeout=120) as response,part.open('wb') as stream:
                        while True:
                            block=response.read(1048576)
                            if not block:break
                            stream.write(block)
                    if file_digest(part)!=asset['sha256']:raise ValueError('Archive checksum mismatch')
                    part.replace(archive);break
                except (OSError,ValueError):
                    if attempt==4:raise
                    print('Retrying download...',flush=True);time.sleep(2**attempt)
        with ZipFile(archive) as z:
            for entry in z.infolist():
                if not (root/entry.filename).resolve().is_relative_to(root) or ((entry.external_attr>>16)&0o170000)==0o120000:
                    raise ValueError('Unsafe archive member')
            z.extractall(root)
    for row in read(ROOT/'library.json')['sources']:
        if file_digest(root/row['file'])!=row['sha256']:raise ValueError('Texture checksum mismatch')
    print('Image library ready:',root/'library',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('fig6_data'))
    download(parser.parse_args().data)
