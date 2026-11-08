"""Independent-instrument tests and systematic octave-selection ablations."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model import PianoSynth, magnitudes, spectral_loss, prepare, yin

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def midi_number(note):
    pitch={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[note[0]]
    if len(note)==3:pitch+=1 if note[1] in ('s','#') else -1
    return (int(note[-1])+1)*12+pitch


def fit_candidates(audio, config, variant='learned', multipliers=(.5,1,2)):
    """Optimise every candidate; return best controls per candidate, not only a winner."""
    initial=yin(audio,config['sample_rate'],*config['pitch_range_hz'])
    candidates=[initial*m for m in multipliers if config['pitch_range_hz'][0]<=initial*m<=config['pitch_range_hz'][1]]
    if not candidates:raise ValueError('No pitch candidate lies inside configured bounds.')
    synth=PianoSynth(candidates,config,variant)
    target=magnitudes(torch.from_numpy(audio)[None],config['fft_sizes'])
    optim=torch.optim.Adam([p for p in synth.parameters() if p.requires_grad],lr=config['learning_rate'])
    best=[float('inf')]*len(candidates);records=[None]*len(candidates);waves=[None]*len(candidates)
    for step in range(config['steps']+1):
        optim.zero_grad();wave=synth();losses=spectral_loss(magnitudes(wave,config['fft_sizes']),target)
        if not torch.isfinite(losses).all():raise FloatingPointError('Nonfinite fitting loss.')
        with torch.no_grad():
            f,a,d,b=synth.controls()
            for i,loss in enumerate(losses.tolist()):
                if loss<best[i]:
                    best[i]=loss;waves[i]=wave[i].detach().numpy().copy()
                    records[i]=dict(f0_hz=float(f[i]),loss=loss,yin_hz=initial,candidate_initial_hz=candidates[i],amplitudes=a[i].tolist(),decays=d[i].tolist(),inharmonicity=float(b[i]),best_step=step)
        if step<config['steps']:losses.sum().backward();optim.step()
    return records,waves


def choose_candidates(records, weight):
    if not np.isfinite(weight) or weight<0:raise ValueError('Prior weight must be nonnegative.')
    free=min(range(len(records)),key=lambda i:records[i]['loss'])
    anchor=min(range(len(records)),key=lambda i:abs(np.log2(records[i]['candidate_initial_hz']/records[i]['yin_hz'])))
    prior=min(range(len(records)),key=lambda i:records[i]['loss']+weight*abs(np.log2(records[i]['f0_hz']/records[i]['yin_hz'])))
    return dict(free=free,anchor=anchor,prior=prior)


def bootstrap_by_pitch(rows,method,seed=2028):
    # All dynamics of a pitch stay together; recordings are not independent people/instruments.
    groups=sorted({r['note'] for r in rows});rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(2000):
        sample=rng.choice(groups,len(groups),replace=True)
        vals=[abs(r[method+'_cents']) for key in sample for r in rows if r['note']==key]
        draws.append(np.mean(vals))
    return np.percentile(draws,[2.5,97.5]).tolist()


def run(split):
    protocol=json.loads((ROOT/'study_protocol.json').read_text());config=json.loads((ROOT/'protocol.json').read_text())
    config['steps']=protocol['fit_steps']
    out=ROOT/'results'/('study_'+split)
    if out.exists():raise FileExistsError(f'Preserve previous results first: {out}')
    if split=='evaluation':
        lock=json.loads((ROOT/'results/study_frozen.json').read_text())
        for name,sha in lock['files'].items():
            if digest(ROOT/name)!=sha:raise ValueError(f'Frozen input changed: {name}')
        manifest=json.loads((ROOT/'data/iowa/manifest.json').read_text());items=[{**r,'path':ROOT/'data/iowa'/r['file']} for r in manifest]
    else:
        manifest=json.loads((ROOT/'data/manifest.json').read_text());items=[{**r,'note':Path(r['file']).stem,'dynamic':'unspecified','path':ROOT/'data'/r['file']} for r in manifest if r['split']=='development']
    out.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows=[];start=time.time()
    for item in items:
        if digest(item['path'])!=item['sha256']:raise ValueError('Audio checksum mismatch.')
        audio,sr=sf.read(item['path']);x,pre=prepare(audio,sr)
        begin=time.time();records,waves=fit_candidates(x,config)
        choices=choose_candidates(records,protocol['octave_selection_prior'])
        fixed,fixed_waves=fit_candidates(x,config,'fixed',(1,))
        # Nominal note labels enter only after every fit/selection is complete.
        ref=440*2**((midi_number(item['note'])-69)/12)
        row=dict(note=item['note'],dynamic=item['dynamic'],file=item['file'],seconds=time.time()-begin)
        estimates={'yin':records[0]['yin_hz'],'fixed':fixed[0]['f0_hz'],**{k:records[i]['f0_hz'] for k,i in choices.items()}}
        for method,freq in estimates.items():row[method+'_cents']=float(1200*np.log2(freq/ref))
        row['fixed_loss']=fixed[0]['loss']
        for method,i in choices.items():row[method+'_loss']=records[i]['loss']
        row['prior_selected_octave']=float(np.log2(records[choices['prior']]['candidate_initial_hz']/records[0]['yin_hz']))
        name=Path(item['file']).stem
        info=dict(preprocessing=pre,candidates=records,fixed=fixed[0],selection=choices,metrics=row)
        (out/f'{name}.json').write_text(json.dumps(info,indent=2)+'\n')
        audio_outputs={'original':x,'fixed':fixed_waves[0],**{k:waves[i] for k,i in choices.items()}}
        gain=min(1.,.98/max(float(np.max(abs(w))) for w in audio_outputs.values()))
        for method,w in audio_outputs.items():sf.write(out/f'{name}_{method}.wav',w*gain,8000,subtype='PCM_16')
        rows.append(row)
        with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
        print(f'{split} {name}: '+', '.join(f'{m} {row[m+"_cents"]:.1f}c' for m in estimates),flush=True)
    summary=dict(n=len(rows),elapsed_seconds=time.time()-start,methods={},by_dynamic={})
    for method in ['yin','fixed','free','anchor','prior']:
        errors=np.array([r[method+'_cents'] for r in rows]);summary['methods'][method]=dict(mean_absolute_cents=float(abs(errors).mean()),median_absolute_cents=float(np.median(abs(errors))),within50=int((abs(errors)<=50).sum()),octave_errors=int((abs(abs(errors)-1200)<50).sum()),mean_error_pitch_cluster_bootstrap95=bootstrap_by_pitch(rows,method))
        if method!='yin':summary['methods'][method]['mean_loss']=float(np.mean([r[method+'_loss'] for r in rows]))
        for dynamic in sorted({r['dynamic'] for r in rows}):
            e=np.array([r[method+'_cents'] for r in rows if r['dynamic']==dynamic])
            summary['by_dynamic'].setdefault(dynamic,{})[method]=dict(n=len(e),within50=int((abs(e)<=50).sum()),mean_absolute_cents=float(abs(e).mean()))
    summary['anchor_loss_improved_count']=sum(r['anchor_loss']<r['fixed_loss'] for r in rows)
    summary['anchor_mean_relative_loss_reduction']=float(np.mean([1-r['anchor_loss']/r['fixed_loss'] for r in rows]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['methods'],indent=2),flush=True)


def freeze():
    if not (ROOT/'results/study_development/summary.json').exists():raise ValueError('Finish development first.')
    p=ROOT/'results/study_frozen.json'
    if p.exists():raise FileExistsError('Protocol already frozen.')
    files=['model.py','study.py','melody.py','study_protocol.json','protocol.json','data/manifest.json','data/iowa/manifest.json']
    p.write_text(json.dumps({'files':{f:digest(ROOT/f) for f in files}},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['development','freeze','evaluation']);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:run(a.action)
