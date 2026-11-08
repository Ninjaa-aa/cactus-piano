"""Resynthesize saved controls after changing pitch or decay, without refitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model import PianoSynth

ROOT=Path(__file__).resolve().parent


def render_controls(controls, config, semitones=0., decay_multiplier=1.):
    if not np.isfinite([semitones,decay_multiplier]).all() or decay_multiplier<=0:
        raise ValueError('Interventions must be finite; decay multiplier must be positive.')
    f0=controls['f0_hz']*2**(semitones/12)
    if not 0<f0<config['sample_rate']/2:
        raise ValueError('Intervened pitch is outside the synthesizer range.')
    synth=PianoSynth([f0],config,'learned')
    with torch.no_grad():
        for attribute,values in [('amplitudes',controls['amplitudes']),('decays',np.array(controls['decays'])*decay_multiplier)]:
            v=torch.as_tensor(values,dtype=torch.float32)
            # Stable inverse softplus restores the fitted positive controls.
            getattr(synth,attribute).copy_((v+torch.log(-torch.expm1(-v)))[None,:])
        stretch=torch.tensor(controls['inharmonicity']/.005).clamp(1e-7,1-1e-7)
        synth.inharmonicity.copy_(torch.logit(stretch)[None])
        return synth()[0].numpy()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('fit_json',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--semitones',type=float,default=0)
    parser.add_argument('--decay-multiplier',type=float,default=1)
    args=parser.parse_args()
    config=json.loads((ROOT/'protocol.json').read_text())
    controls=json.loads(args.fit_json.read_text())['results']['learned']
    audio=render_controls(controls,config,args.semitones,args.decay_multiplier)
    gain=min(1.,.98/float(np.max(np.abs(audio))))
    sf.write(args.output,audio*gain,config['sample_rate'],subtype='PCM_16')
    args.output.with_suffix('.json').write_text(json.dumps(dict(source=str(args.fit_json),semitones=args.semitones,decay_multiplier=args.decay_multiplier,playback_gain=gain),indent=2)+'\n')
