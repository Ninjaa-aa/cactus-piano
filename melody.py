"""Controlled sample-assembled monophonic sequences, never labelled human performances."""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, stft

from model import fit, prepare, yin
from study import digest, midi_number

ROOT=Path(__file__).resolve().parent


def assemble(items, rng, settings):
    sr=8000;events=[];parts=[np.zeros(sr//4,dtype=np.float32)];clock=.25
    for item in items:
        audio,rate=sf.read(item['path']);x,_=prepare(audio,rate)
        duration=float(rng.choice(settings['durations']));gap=float(rng.choice(settings['gaps']))
        n=round(duration*sr);w=x[:n].copy()*float(rng.uniform(.5,1))
        fade=min(160,n//4);w[-fade:]*=np.linspace(1,0,fade)
        parts.extend([w,np.zeros(round(gap*sr),dtype=np.float32)])
        events.append(dict(onset=clock,offset=clock+duration,midi=midi_number(item['note']),source=item['file'],dynamic=item.get('dynamic','unspecified')))
        clock+=duration+gap
    parts.append(np.zeros(sr//4,dtype=np.float32))
    return np.concatenate(parts),events


def segment(audio, settings, sr=8000):
    x=np.asarray(audio)
    if x.ndim!=1 or len(x)<sr//3 or not np.isfinite(x).all():raise ValueError('Expected finite mono audio of at least 1/3 second.')
    if np.max(abs(x))<1e-6:return []
    _,times,z=stft(x,fs=sr,nperseg=settings['fft'],noverlap=settings['fft']-settings['hop'],boundary='zeros')
    mag=np.sqrt(abs(z));flux=np.maximum(np.diff(mag,axis=1,prepend=mag[:,:1]),0).sum(axis=0)
    flux=gaussian_filter1d(flux,1)
    peaks,_=find_peaks(flux,prominence=float(flux.max())*settings['flux_threshold_fraction'],distance=round(settings['minimum_peak_distance_seconds']*sr/settings['hop']))
    # Peak time is a smoothed frame centre; a fixed half-window correction is signal-derived.
    # Release transients can create positive spectral flux; require rising local energy.
    rising=[]
    for peak in peaks:
        centre=round(times[peak]*sr);window=round(.04*sr)
        before=np.mean(x[max(0,centre-window):max(1,centre)]**2)
        after=np.mean(x[centre:min(len(x),centre+window)]**2) if centre<len(x) else 0.
        if after>1.15*before:rising.append(peak)
    onsets=np.maximum(0,times[rising]-settings['fft']/sr/2)
    hop=settings['hop'];rms=np.sqrt(np.convolve(x*x,np.ones(160)/160,mode='same'))
    events=[]
    for i,on in enumerate(onsets):
        limit=float(onsets[i+1]) if i+1<len(onsets) else len(x)/sr
        begin=round(on*sr);stop=round(limit*sr)
        if stop-begin<sr*.15:continue
        peak=float(np.max(rms[begin:stop]));threshold=peak*settings['offset_rms_fraction']
        offset=limit
        for sample in range(begin+round(.15*sr),max(begin,stop-2*hop),hop):
            if np.max(rms[sample:sample+2*hop])<threshold:
                offset=sample/sr;break
        if offset-on>=.15:events.append(dict(onset=float(on),offset=float(offset)))
    return events


def transcribe(audio, settings, config, use_synth=False):
    events=segment(audio,settings);result=[];sr=config['sample_rate']
    for event in events:
        start=round(event['onset']*sr);stop=min(len(audio),round(event['offset']*sr))
        x=audio[start:stop]
        # YIN uses a fixed 0.28s prefix; pad only short detected segments, never use labels.
        length=max(round(.35*sr),min(len(x),sr));x=np.pad(x[:length],(0,max(0,length-len(x)))).astype(np.float32)
        peak=float(np.max(abs(x)))
        if peak<1e-6:continue
        x=x/peak*.9;frequency=yin(x,sr,*config['pitch_range_hz'])
        if use_synth:
            local={**config,'duration_seconds':len(x)/sr,'candidate_multipliers':[1.]}
            _,fitted=fit(x,local,'learned');frequency=fitted['f0_hz']
        result.append({**event,'midi':float(69+12*np.log2(frequency/440))})
    return result


def match_events(reference,prediction,onset_tolerance=.05,offset_tolerance=None,require_pitch=True):
    if not reference or not prediction:
        return dict(matches=0,reference=len(reference),predicted=len(prediction),precision=0.,recall=0.,f1=0.,onset_mae=None,offset_mae=None)
    cost=np.full((len(reference),len(prediction)),1000.)
    for i,r in enumerate(reference):
        for j,p in enumerate(prediction):
            if abs(r['onset']-p['onset'])>onset_tolerance:continue
            if require_pitch and abs(r['midi']-p['midi'])>.5:continue
            if offset_tolerance is not None and abs(r['offset']-p['offset'])>offset_tolerance:continue
            cost[i,j]=abs(r['onset']-p['onset'])
    a,b=linear_sum_assignment(cost);pairs=[(i,j) for i,j in zip(a,b) if cost[i,j]<1000]
    n=len(pairs);precision=n/len(prediction);recall=n/len(reference)
    return dict(matches=n,reference=len(reference),predicted=len(prediction),precision=precision,recall=recall,f1=2*n/(len(reference)+len(prediction)),onset_mae=float(np.mean([abs(reference[i]['onset']-prediction[j]['onset']) for i,j in pairs])) if n else None,offset_mae=float(np.mean([abs(reference[i]['offset']-prediction[j]['offset']) for i,j in pairs])) if n else None)


def varlen(value):
    if value<0:raise ValueError('Negative MIDI delta.')
    data=[value&127];value>>=7
    while value:data.insert(0,(value&127)|128);value>>=7
    return bytes(data)


def midi_events(events):
    timed=[]
    for e in events:
        if not all(math.isfinite(e[k]) for k in ['midi','onset','offset']) or e['onset']<0 or e['offset']<=e['onset']:raise ValueError('Invalid note event.')
        note=int(np.clip(round(e['midi']),0,127))
        timed.extend([(round(e['onset']*960),1,note),(round(e['offset']*960),0,note)])
    track=b'\x00\xff\x51\x03\x07\xa1\x20';previous=0
    for tick,on,note in sorted(timed):
        track+=varlen(tick-previous)+bytes([0x90 if on else 0x80,note,80 if on else 0]);previous=tick
    track+=b'\x00\xff\x2f\x00'
    return b'MThd'+struct.pack('>IHHH',6,0,1,480)+b'MTrk'+struct.pack('>I',len(track))+track


def run(split):
    protocol=json.loads((ROOT/'study_protocol.json').read_text());settings=protocol['melody'];config=json.loads((ROOT/'protocol.json').read_text())
    out=ROOT/'results'/('melody_'+split)
    if out.exists():raise FileExistsError('Preserve prior melody run before rerunning.')
    if split=='evaluation':
        lock=json.loads((ROOT/'results/study_frozen.json').read_text())
        for name,sha in lock['files'].items():
            if digest(ROOT/name)!=sha:raise ValueError(f'Frozen input changed: {name}')
        manifest=json.loads((ROOT/'data/iowa/manifest.json').read_text());pool=[{**r,'path':ROOT/'data/iowa'/r['file']} for r in manifest]
    else:
        manifest=json.loads((ROOT/'data/manifest.json').read_text());pool=[{**r,'note':Path(r['file']).stem,'path':ROOT/'data'/r['file']} for r in manifest if r['split']=='development']
    for item in pool:
        if digest(item['path'])!=item['sha256']:raise ValueError('Audio checksum mismatch.')
    out.mkdir();rng=np.random.default_rng(protocol['seed']);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows=[]
    for clip in range(settings[split+'_clips']):
        selected=[pool[int(i)] for i in rng.integers(0,len(pool),settings['notes_per_clip'])]
        # Include repeated notes in every third clip to challenge onset rather than pitch-change detection.
        if clip%3==0:selected[2]=selected[1]
        audio,ref=assemble(selected,rng,settings);name=f'clip{clip+1:02}'
        sf.write(out/f'{name}.wav',audio,8000,subtype='PCM_16');(out/f'{name}_reference.mid').write_bytes(midi_events(ref))
        info=dict(construction='assembled isolated recordings; not a human performance',reference=ref,predictions={},metrics={})
        for method,use in [('yin',False),('synthesis',True)]:
            pred=transcribe(audio,settings,config,use)
            info['predictions'][method]=pred
            info['metrics'][method]={
                'onset_only':match_events(ref,pred,require_pitch=False),
                'pitch_onset':match_events(ref,pred),
                'pitch_onset_offset':match_events(ref,pred,offset_tolerance=.05)}
            (out/f'{name}_{method}.mid').write_bytes(midi_events(pred))
        (out/f'{name}.json').write_text(json.dumps(info,indent=2)+'\n');rows.append(info)
        print(f"{split} {name}: YIN note F1={info['metrics']['yin']['pitch_onset']['f1']:.3f}, synth={info['metrics']['synthesis']['pitch_onset']['f1']:.3f}",flush=True)
    summary=dict(clips=len(rows),reference_events=sum(len(r['reference']) for r in rows),methods={})
    for method in ['yin','synthesis']:
        summary['methods'][method]={}
        for metric in ['onset_only','pitch_onset','pitch_onset_offset']:
            matches=sum(r['metrics'][method][metric]['matches'] for r in rows);nr=sum(r['metrics'][method][metric]['reference'] for r in rows);npred=sum(r['metrics'][method][metric]['predicted'] for r in rows)
            summary['methods'][method][metric]=dict(matches=matches,reference=nr,predicted=npred,precision=matches/npred if npred else 0,recall=matches/nr,f1=2*matches/(nr+npred))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['development','evaluation']);run(p.parse_args().split)
