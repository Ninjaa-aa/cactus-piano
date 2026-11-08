import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model import PianoSynth, fit, magnitudes, prepare, spectral_loss, yin

CONFIG=json.loads(Path(__file__).with_name('protocol.json').read_text())
torch.set_num_threads(2)

@pytest.mark.parametrize('frequency',[65.406,110.,220.,440.,880.,1046.502])
def test_yin_known_pitch(frequency):
    t=np.arange(8000)/8000
    x=np.sin(2*np.pi*frequency*t)+.3*np.sin(4*np.pi*frequency*t)
    estimate=yin(x,8000)
    assert abs(1200*np.log2(estimate/frequency))<12

@pytest.mark.parametrize('x',[np.zeros(8000),np.full(8000,np.nan),np.ones((2,2,2))])
def test_invalid_audio_rejected(x):
    with pytest.raises(ValueError):prepare(x,8000)


def test_preprocessing_and_duration():
    t=np.arange(24000)/16000
    wave=np.sin(2*np.pi*220*t)
    x,meta=prepare(np.stack([wave,wave],axis=1),16000)
    assert x.shape==(8000,) and np.isfinite(x).all()
    assert np.max(np.abs(x))==pytest.approx(.9)
    assert meta['onset_seconds']>=0
    with pytest.raises(ValueError):prepare(wave[:100],16000)


def test_differentiable_controls_and_ablation():
    for variant in ('fixed','learned'):
        synth=PianoSynth([220],CONFIG,variant)
        loss=synth().square().mean();loss.backward()
        for name in ('pitch','amplitudes','inharmonicity'):
            grad=getattr(synth,name).grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
        assert (synth.decays.grad is not None)==(variant=='learned')


def test_no_aliasing_above_nyquist():
    synth=PianoSynth([5000],CONFIG,'fixed')
    assert torch.count_nonzero(synth())==0


def test_spectral_identity_and_real_difference():
    t=torch.arange(8000)/8000
    a=magnitudes(torch.sin(2*torch.pi*220*t)[None],CONFIG['fft_sizes'])
    b=magnitudes(torch.sin(2*torch.pi*330*t)[None],CONFIG['fft_sizes'])
    assert spectral_loss(a,a).item()==0
    assert spectral_loss(a,b).item()>0


def test_fitting_reduces_loss_without_labels():
    c={**CONFIG,'steps':25}
    synth=PianoSynth([220],c,'learned')
    x=synth().detach().numpy()[0]
    reconstructed,result=fit(x,c,'learned')
    assert np.isfinite(reconstructed).all()
    assert result['loss']<result['initial_loss']
    assert abs(1200*np.log2(result['f0_hz']/220))<50


def test_protocol_has_disjoint_nonempty_splits():
    assert CONFIG['development'] and CONFIG['evaluation']
    assert not set(CONFIG['development']) & set(CONFIG['evaluation'])


def test_single_note_midi_encodes_pitch_and_duration():
    import struct
    from experiment import midi_bytes
    b=midi_bytes(440.)
    assert b[:4]==b'MThd' and b[14:18]==b'MTrk'
    assert struct.unpack('>I',b[18:22])[0]==len(b)-22
    assert bytes([0,0x90,69,80]) in b
    assert bytes([0x87,0x40,0x80,69,0]) in b


def test_interventions_preserve_controls_and_change_audio():
    from intervene import render_controls
    synth=PianoSynth([220],CONFIG,'learned')
    with torch.no_grad():
        f,a,d,b=synth.controls()
        controls=dict(f0_hz=float(f[0]),amplitudes=a[0].tolist(),decays=d[0].tolist(),inharmonicity=float(b[0]))
        original=synth()[0].numpy()
    restored=render_controls(controls,CONFIG)
    assert np.max(abs(restored-original))<1e-5
    shifted=render_controls(controls,CONFIG,semitones=12)
    assert abs(1200*np.log2(yin(shifted,8000)/440))<15
    fast=render_controls(controls,CONFIG,decay_multiplier=2)
    assert np.mean(fast[-2000:]**2)<np.mean(original[-2000:]**2)
    with pytest.raises(ValueError):render_controls(controls,CONFIG,decay_multiplier=-1)
