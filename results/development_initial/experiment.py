"""Run the declared split; save every result, audio, MIDI, and provenance."""
import argparse
import csv
import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
import torch

from model import fit, prepare, yin

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def note_midi(name):
    names={'C':0,'Ds':3,'Fs':6,'A':9}
    return (int(name[-1])+1)*12+names[name[:-1]]


def midi_bytes(frequency):
    # One isolated 1-second note, not an inferred onset/offset transcription.
    note=int(np.clip(round(69+12*np.log2(frequency/440)),0,127))
    track=b'\x00\xff\x51\x03\x07\xa1\x20'+bytes([0,0x90,note,80])+bytes([0x87,0x40,0x80,note,0])+b'\x00\xff\x2f\x00'
    return b'MThd'+struct.pack('>IHHH',6,0,1,480)+b'MTrk'+struct.pack('>I',len(track))+track


def run(split):
    config=json.loads((ROOT/'protocol.json').read_text())
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    dest=ROOT/'results'/split
    if dest.exists():
        raise FileExistsError(f'{dest} exists; preserve prior evidence before a new run.')
    if split=='evaluation':
        lock=json.loads((ROOT/'results/frozen.json').read_text())
        for name,digest in lock['files'].items():
            if sha(ROOT/name)!=digest:raise ValueError(f'Frozen file changed: {name}')
    dest.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows=[]; started=time.time()
    for item in [r for r in manifest if r['split']==split]:
        path=ROOT/'data'/item['file']
        if sha(path)!=item['sha256']:raise ValueError(f'Checksum mismatch: {path}')
        audio,sr=sf.read(path)
        x,pre=prepare(audio,sr,config['sample_rate'],config['duration_seconds'])
        # Reference label is read after inference; it never enters fit().
        waveforms={'original':x}
        results={}
        for variant in config['variants']:
            begin=time.time()
            reconstructed,result=fit(x,config,variant)
            result['elapsed_seconds']=time.time()-begin
            waveforms[variant]=reconstructed;results[variant]=result
        reference=note_midi(path.stem)
        baseline=yin(x,config['sample_rate'],*config['pitch_range_hz'])
        row=dict(note=path.stem,reference_midi=reference,split=split,yin_hz=baseline,
                 yin_cents=1200*np.log2(baseline/(440*2**((reference-69)/12))))
        for variant,result in results.items():
            row[variant+'_hz']=result['f0_hz']
            row[variant+'_cents']=1200*np.log2(result['f0_hz']/(440*2**((reference-69)/12)))
            row[variant+'_loss']=result['loss']
            row[variant+'_seconds']=result['elapsed_seconds']
        gain=min(1.,.98/max(float(np.max(np.abs(w))) for w in waveforms.values()))
        for variant,wave in waveforms.items():
            sf.write(dest/f'{path.stem}_{variant}.wav',wave*gain,config['sample_rate'],subtype='PCM_16')
        for variant,frequency in [('yin',baseline)]+[(v,r['f0_hz']) for v,r in results.items()]:
            (dest/f'{path.stem}_{variant}.mid').write_bytes(midi_bytes(frequency))
        (dest/f'{path.stem}.json').write_text(json.dumps(dict(preprocessing=pre,playback_common_gain=gain,results=results),indent=2)+'\n')
        rows.append(row)
        with (dest/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
        print(f"{split} {path.stem}: cents YIN {row['yin_cents']:.1f}, fixed {row['fixed_cents']:.1f}, learned {row['learned_cents']:.1f}; loss {row['fixed_loss']:.3f} -> {row['learned_loss']:.3f}",flush=True)
    summary={'split':split,'n':len(rows),'elapsed_seconds':time.time()-started,'methods':{}}
    for method in ('yin','fixed','learned'):
        errors=np.array([r[method+'_cents'] for r in rows])
        summary['methods'][method]={'median_absolute_cents':float(np.median(abs(errors))),
                                    'mean_absolute_cents':float(np.mean(abs(errors))),
                                    'within_50_cents':int((abs(errors)<=50).sum()),
                                    'octave_errors':int((np.abs(abs(errors)-1200)<50).sum())}
        if method!='yin':summary['methods'][method]['mean_spectral_loss']=float(np.mean([r[method+'_loss'] for r in rows]))
    improvements=np.array([r['fixed_loss']-r['learned_loss'] for r in rows])
    summary['learned_lower_loss_count']=int((improvements>0).sum())
    summary['mean_loss_reduction']=float(improvements.mean())
    summary['mean_relative_loss_reduction']=float(np.mean([1-r['learned_loss']/r['fixed_loss'] for r in rows]))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    provenance=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,soundfile=sf.__version__,threads=torch.get_num_threads(),config=config,
                    files={p.name:sha(p) for p in [ROOT/'model.py',ROOT/'experiment.py',ROOT/'protocol.json']},manifest_sha256=sha(ROOT/'data/manifest.json'))
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def freeze():
    if not (ROOT/'results/development/summary.json').exists():raise ValueError('Complete development first.')
    lock=ROOT/'results/frozen.json'
    if lock.exists():raise FileExistsError('Already frozen; do not silently overwrite the evaluation protocol.')
    lock.write_text(json.dumps({'files':{name:sha(ROOT/name) for name in ['model.py','experiment.py','protocol.json','data/manifest.json']}},indent=2)+'\n')
    print('Frozen model, runner, protocol and sample manifest.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['development','freeze','evaluation']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:run(args.action)
