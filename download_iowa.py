"""Independent Steinway recordings at three dynamics, with provenance."""
import concurrent.futures
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE='https://theremin.music.uiowa.edu/sound%20files/MIS/Piano_Other/piano/'

def main():
    p=json.loads((ROOT/'study_protocol.json').read_text())
    dest=ROOT/'data/iowa';dest.mkdir(exist_ok=True)
    def fetch(pair):
        dynamic,note=pair;name=f'Piano.{dynamic}.{note}.aiff';path=dest/name;url=BASE+name
        prefix=1_048_576
        if path.exists() and path.stat().st_size>prefix:
            full=dest/'full_downloads';full.mkdir(exist_ok=True);path.rename(full/name)
        meta=path.with_suffix('.source.json')
        if not path.exists():
            request=urllib.request.Request(url,headers={'Range':f'bytes=0-{prefix-1}'})
            with urllib.request.urlopen(request,timeout=40) as response:
                data=response.read(prefix+1);content_range=response.headers.get('Content-Range','')
                if response.status!=206 or not content_range.startswith(f'bytes 0-{prefix-1}/'):
                    raise ValueError(f'Server did not honour the requested range: {name}')
            if len(data)!=prefix or data[:4]!=b'FORM':raise ValueError(f'Unexpected AIFF prefix: {name}')
            path.write_bytes(data)
            meta.write_text(json.dumps({'content_range':content_range,'source_total_bytes':int(content_range.split('/')[-1])}))
        metadata=json.loads(meta.read_text())
        return dict(file=name,note=note,dynamic=dynamic,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,**metadata,download_scope='first 1048576 bytes of original AIFF; approximately 5.94 seconds at 44.1kHz stereo 16bit; not full source-file hash')
    pairs=[(d,n) for d in p['dynamics'] for n in p['evaluation_notes']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,pairs))
    manifest=dest/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=rows:raise ValueError('Checksum drift; preserve old manifest and investigate.')
    manifest.write_text(json.dumps(rows,indent=2)+'\n')
    for name,url in [('SOURCE.html','https://theremin.music.uiowa.edu/MISpiano.html'),('USE_PERMISSION.html','https://theremin.music.uiowa.edu/MIS.html')]:
        with urllib.request.urlopen(url,timeout=30) as response:(dest/name).write_bytes(response.read())
    print(f'{len(rows)} real recordings; {sum(r["bytes"] for r in rows):,} bytes')

if __name__=='__main__':main()
