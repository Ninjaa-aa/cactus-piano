"""Transcribe a real audio file using the frozen monophonic study pipeline."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from melody import transcribe, midi_events

ROOT=Path(__file__).resolve().parent


def read_audio(path):
    audio, rate=sf.read(path, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio must be nonempty and finite.')
    mono=audio.mean(axis=1)
    divisor=math.gcd(rate,8000)
    if rate!=8000:mono=resample_poly(mono,8000//divisor,rate//divisor)
    if len(mono)<8000/3:raise ValueError('Provide at least one third of a second of audio.')
    return mono.astype(np.float32)


def main():
    parser=argparse.ArgumentParser(description='Separated monophonic piano notes to MIDI; chords and pedal are unsupported.')
    parser.add_argument('audio',type=Path)
    parser.add_argument('output_prefix',type=Path)
    parser.add_argument('--refine',action='store_true',help='Fit the differentiable synthesizer per detected note (CPU, slower).')
    args=parser.parse_args()
    outputs=[Path(str(args.output_prefix)+suffix) for suffix in ('.mid','.json')]
    if any(p.exists() for p in outputs):raise FileExistsError('Output exists; choose a new prefix.')
    protocol=json.loads((ROOT/'study_protocol.json').read_text())
    config=json.loads((ROOT/'protocol.json').read_text())
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    audio=read_audio(args.audio)
    events=transcribe(audio,protocol['melody'],config,args.refine)
    midi=midi_events(events)
    evidence=dict(source=str(args.audio),duration_seconds=len(audio)/8000,method='differentiable refinement' if args.refine else 'YIN',events=events,limitations='Separated monophonic notes only; no inferred velocity, chords or pedal; no reference labels supplied.')
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    outputs[0].write_bytes(midi)
    outputs[1].write_text(json.dumps(evidence,indent=2)+'\n')
    print(f'{len(events)} detected notes; wrote {outputs[0]} and {outputs[1]}')


if __name__=='__main__':main()
