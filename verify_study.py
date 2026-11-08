"""Check persisted study evidence without rerunning the expensive fits."""
import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import soundfile as sf

from melody import match_events

ROOT=Path(__file__).resolve().parent


def main():
    result=ROOT/'results'
    for name,sha in json.loads((result/'study_frozen.json').read_text())['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    manifest=json.loads((ROOT/'data/iowa/manifest.json').read_text())
    assert len(manifest)==len({r['sha256'] for r in manifest})==39
    for row in manifest:
        path=ROOT/'data/iowa'/row['file']
        assert path.stat().st_size==row['bytes']==1048576
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        x,sr=sf.read(path)
        assert sr==44100 and np.isfinite(x).all() and len(x)/sr>5.9 and np.max(abs(x))>1e-6
    rows=list(csv.DictReader((result/'study_evaluation/metrics.csv').open()))
    summary=json.loads((result/'study_evaluation/summary.json').read_text())
    assert len(rows)==summary['n']==39
    for method in summary['methods']:
        errors=np.array([float(r[method+'_cents']) for r in rows])
        assert int((abs(errors)<=50).sum())==summary['methods'][method]['within50']
        assert np.isclose(abs(errors).mean(),summary['methods'][method]['mean_absolute_cents'])
    clips=list((result/'melody_evaluation').glob('clip*.json'))
    assert len(clips)==12
    aggregate=json.loads((result/'melody_evaluation/summary.json').read_text())
    for path in clips:
        clip=json.loads(path.read_text())
        for method,events in [('reference',clip['reference']),*clip['predictions'].items()]:
            raw=path.with_name(path.stem+'_'+method+'.mid').read_bytes()
            assert raw[:14]==b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0'
            assert raw[14:18]==b'MTrk' and int.from_bytes(raw[18:22],'big')==len(raw)-22
            assert raw[22:29]==b'\x00\xff\x51\x03\x07\xa1\x20' and raw[-4:]==b'\x00\xff\x2f\x00'
            index=29;tick=0;decoded=[]
            while index<len(raw)-4:
                delta=0
                while True:
                    byte=raw[index];index+=1;delta=(delta<<7)+(byte&127)
                    if not byte&128:break
                tick+=delta;status,note,velocity=raw[index:index+3];index+=3
                assert status in (0x80,0x90) and 0<=note<=127
                assert velocity==(80 if status==0x90 else 0)
                decoded.append((tick,1 if status==0x90 else 0,note))
            expected=sorted((round(e[time]*960),on,int(np.clip(round(e['midi']),0,127))) for e in events for time,on in [('onset',1),('offset',0)])
            assert decoded==expected,path
    for method in ['yin','synthesis']:
        for metric,kwargs in [('onset_only',dict(require_pitch=False)),('pitch_onset',{}),('pitch_onset_offset',dict(offset_tolerance=.05))]:
            metrics=[]
            for path in clips:
                clip=json.loads(path.read_text())
                m=match_events(clip['reference'],clip['predictions'][method],**kwargs)
                assert m==clip['metrics'][method][metric],path
                metrics.append(m)
            m=aggregate['methods'][method][metric]
            for field in ['matches','reference','predicted']:assert m[field]==sum(x[field] for x in metrics)
            assert np.isclose(m['f1'],2*m['matches']/(m['reference']+m['predicted']))
    wavs=list((result/'study_evaluation').glob('*.wav'))+list((result/'melody_evaluation').glob('*.wav'))
    assert len(wavs)==39*5+12
    for path in wavs:
        x,sr=sf.read(path)
        assert sr==8000 and x.ndim==1 and len(x)>0 and np.isfinite(x).all() and np.max(abs(x))<=1,path
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and not value.startswith(('http:','https:','#')):
                    assert (result/value).exists(),value
    Links().feed((result/'study.html').read_text())
    receipt=dict(source_recordings=39,verified_output_wavs=len(wavs),verified_melodies=12,verified_midi_files=36,frozen_hashes_match=True,metrics_recomputed=True,local_report_links_exist=True)
    (result/'study_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
