"""Label-free isolated-note fitting with a differentiable additive synthesizer."""
import math

import numpy as np
import torch
from scipy.signal import find_peaks, resample_poly


def prepare(audio, original_rate, sample_rate=8000, duration=1.0):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) == 0:
        raise ValueError('Expected finite mono or stereo audio.')
    if original_rate <= 0 or sample_rate <= 0 or duration <= 0:
        raise ValueError('Sample rates and duration must be positive.')
    if np.max(np.abs(x)) < 1e-6:
        raise ValueError('Silent input.')
    divisor = math.gcd(int(original_rate), int(sample_rate))
    x = resample_poly(x, sample_rate // divisor, original_rate // divisor)
    # Isolated samples only: locate energy onset, not an annotated note boundary.
    envelope = np.convolve(x*x, np.ones(40)/40, mode='same')
    onset = max(0, int(np.flatnonzero(envelope > envelope.max() * .01)[0]) - 16)
    length = round(sample_rate * duration)
    if len(x) - onset < length:
        raise ValueError('Recording is too short after onset for the configured crop.')
    x = x[onset:onset+length]
    peak = np.max(np.abs(x))
    return (x / peak * .9).astype(np.float32), dict(onset_seconds=onset/sample_rate, peak_scale=.9/peak)


def yin(audio, sample_rate, fmin=55., fmax=1200.):
    """YIN cumulative mean normalised difference, fixed threshold 0.15."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or np.max(np.abs(x), initial=0) < 1e-6:
        raise ValueError('Pitch estimation requires finite non-silent mono audio.')
    if not 0 < fmin < fmax < sample_rate/2:
        raise ValueError('Invalid pitch search bounds.')
    maxlag = math.ceil(sample_rate/fmin)
    minlag = max(2, math.floor(sample_rate/fmax))
    # Skip the noisy hammer attack; take a fixed-size common window for every lag.
    x = x[int(.03*sample_rate):int(.28*sample_rate)]
    size = len(x)-maxlag
    if size < maxlag:
        raise ValueError('Insufficient audio for the pitch range.')
    diffs = np.array([np.sum((x[:size]-x[t:t+size])**2) for t in range(maxlag+1)])
    cmnd = np.ones_like(diffs)
    cmnd[1:] = diffs[1:]*np.arange(1,len(diffs))/np.maximum(np.cumsum(diffs[1:]),1e-15)
    troughs, _ = find_peaks(-cmnd)
    troughs = troughs[(troughs >= minlag) & (troughs < maxlag)]
    candidates = troughs[cmnd[troughs] < .15]
    lag = int(candidates[0]) if len(candidates) else int(np.argmin(cmnd[minlag:maxlag])+minlag)
    den = cmnd[lag-1]-2*cmnd[lag]+cmnd[lag+1]
    correction = .5*(cmnd[lag-1]-cmnd[lag+1])/den if abs(den)>1e-12 else 0.
    return float(sample_rate/(lag+np.clip(correction,-.5,.5)))


class PianoSynth(torch.nn.Module):
    """Independent controls for pitch, partial amplitudes, and partial decay rates."""
    def __init__(self, candidates, config, variant):
        super().__init__()
        if variant not in ('fixed', 'learned'):
            raise ValueError('Unknown envelope variant.')
        self.config, self.variant = config, variant
        f = torch.as_tensor(candidates, dtype=torch.float32)
        if not torch.isfinite(f).all() or (f <= 0).any():
            raise ValueError('Frequencies must be finite and positive.')
        self.register_buffer('initial_f0', f)
        self.register_buffer('t', torch.arange(round(config['sample_rate']*config['duration_seconds']))/config['sample_rate'])
        self.register_buffer('h', torch.arange(1,config['harmonics']+1, dtype=torch.float32))
        self.pitch = torch.nn.Parameter(torch.zeros(len(f)))
        self.amplitudes = torch.nn.Parameter((-2-torch.log(self.h))[None,:].repeat(len(f),1))
        self.inharmonicity = torch.nn.Parameter(torch.full((len(f),), -7.))
        self.decays = torch.nn.Parameter(torch.full((len(f),config['harmonics']), math.log(math.expm1(config['fixed_decay_per_second']))), requires_grad=variant=='learned')

    def controls(self):
        f0 = self.initial_f0 * 2**(self.config['pitch_refinement_semitones']*torch.tanh(self.pitch)/12)
        amplitude = torch.nn.functional.softplus(self.amplitudes)
        decay = torch.nn.functional.softplus(self.decays)
        stretch = .005*torch.sigmoid(self.inharmonicity)
        return f0, amplitude, decay, stretch

    def forward(self):
        f0, amplitude, decay, stretch = self.controls()
        frequencies = f0[:,None]*self.h[None,:]*torch.sqrt(1+stretch[:,None]*self.h[None,:]**2)
        mask = (frequencies < self.config['sample_rate']/2-40).to(amplitude.dtype)
        envelope = torch.exp(-decay[:,:,None]*self.t)*(1-torch.exp(-self.t/.005))
        partials = torch.sin(2*torch.pi*frequencies[:,:,None]*self.t)
        return (amplitude[:,:,None]*mask[:,:,None]*envelope*partials).sum(dim=1)


def magnitudes(x, fft_sizes):
    return [torch.stft(x, n_fft=n, hop_length=n//4, window=torch.hann_window(n), return_complex=True).abs() for n in fft_sizes]


def spectral_loss(predicted, target):
    terms=[]
    for p,t in zip(predicted,target):
        linear=(p-t).abs().mean(dim=(-2,-1))/t.mean().clamp_min(1e-5)
        logarithmic=(torch.log(p+1e-3)-torch.log(t+1e-3)).abs().mean(dim=(-2,-1))
        terms.append(linear+logarithmic)
    return torch.stack(terms).mean(dim=0)


def fit(audio, config, variant):
    """No note names or MIDI labels are accepted by the fitting boundary."""
    x=np.asarray(audio,dtype=np.float32)
    if x.ndim!=1 or len(x)!=round(config['sample_rate']*config['duration_seconds']) or not np.isfinite(x).all():
        raise ValueError('Incorrect waveform shape or nonfinite audio.')
    initial = yin(x,config['sample_rate'],*config['pitch_range_hz'])
    candidates = [v for v in (initial*m for m in config['candidate_multipliers']) if config['pitch_range_hz'][0] <= v <= config['pitch_range_hz'][1]]
    torch.manual_seed(config['seed'])
    synth=PianoSynth(candidates, config, variant)
    target=magnitudes(torch.from_numpy(x)[None,:],config['fft_sizes'])
    optimizer=torch.optim.Adam([p for p in synth.parameters() if p.requires_grad],lr=config['learning_rate'])
    best_loss=float('inf'); best_state=None; history=[]
    for step in range(config['steps']+1):
        optimizer.zero_grad()
        output=synth()
        losses=spectral_loss(magnitudes(output,config['fft_sizes']),target)
        if not torch.isfinite(losses).all():
            raise FloatingPointError('Nonfinite optimisation loss.')
        value,index=losses.detach().min(dim=0)
        if float(value)<best_loss:
            best_loss=float(value);best_index=int(index)
            best_state={k:v.detach().clone() for k,v in synth.state_dict().items()}
        history.append(float(value))
        if step<config['steps']:
            losses.sum().backward();optimizer.step()
    synth.load_state_dict(best_state)
    with torch.no_grad():
        f0,amp,decay,stretch=synth.controls()
        reconstructed=synth()[best_index].numpy()
    result=dict(f0_hz=float(f0[best_index]),yin_hz=initial,loss=best_loss,initial_loss=history[0],
                candidate_initial_hz=candidates[best_index],amplitudes=amp[best_index].detach().tolist(),
                decays=decay[best_index].detach().tolist(),inharmonicity=float(stretch[best_index]),history=history)
    return reconstructed,result
