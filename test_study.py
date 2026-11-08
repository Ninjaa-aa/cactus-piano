import json
from pathlib import Path

import numpy as np
import pytest
import torch

from melody import match_events, midi_events, segment, varlen
from study import choose_candidates, fit_candidates, midi_number

ROOT=Path(__file__).parent
CONFIG=json.loads((ROOT/'protocol.json').read_text())
SETTINGS=json.loads((ROOT/'study_protocol.json').read_text())['melody']
torch.set_num_threads(2)


def test_pitch_names_equivalent():
    assert midi_number('Ds4')==midi_number('Eb4')==63
    assert midi_number('Gb3')==54


def test_octave_prior_selects_with_signal_prior_not_label():
    records=[dict(f0_hz=220,yin_hz=440,candidate_initial_hz=220,loss=1),dict(f0_hz=440,yin_hz=440,candidate_initial_hz=440,loss=1.1)]
    assert choose_candidates(records,.3)==dict(free=0,anchor=1,prior=1)
    assert choose_candidates(records,0)['prior']==0
    with pytest.raises(ValueError):choose_candidates(records,-1)


def test_multistart_retains_all_finite_candidates():
    t=np.arange(8000)/8000;x=(.5*np.sin(2*np.pi*220*t)*np.exp(-2*t)).astype(np.float32)
    results,waves=fit_candidates(x,{**CONFIG,'steps':3})
    assert len(results)==len(waves)==3
    for r,w in zip(results,waves):
        assert np.isfinite(w).all() and w.shape==(8000,)
        assert np.isfinite(r['loss']) and r['best_step']<=3


def test_event_matching_is_one_to_one_and_rejects_octaves():
    r=[dict(onset=.1,offset=.5,midi=60)]
    p=[dict(onset=.11,offset=.51,midi=60)]*2
    m=match_events(r,p);assert m['matches']==1 and m['precision']==.5 and m['recall']==1
    bad=[dict(onset=.1,offset=.5,midi=72)]
    assert match_events(r,bad)['matches']==0
    assert match_events(r,bad,require_pitch=False)['matches']==1


def test_offset_metric_rejects_overlong_note():
    r=[dict(onset=.1,offset=.5,midi=60)];p=[dict(onset=.1,offset=.8,midi=60)]
    assert match_events(r,p)['matches']==1
    assert match_events(r,p,offset_tolerance=.05)['matches']==0


def test_silence_and_nonfinite_audio():
    assert segment(np.zeros(8000),SETTINGS)==[]
    with pytest.raises(ValueError):segment(np.full(8000,np.nan),SETTINGS)


def test_repeated_note_onsets_detected_without_pitch_change():
    x=np.zeros(16000,dtype=np.float32)
    for onset in [.25,1.0]:
        t=np.arange(4000)/8000
        w=(1-np.exp(-t/.003))*np.exp(-2*t)*np.sin(2*np.pi*220*t)
        w[-160:]*=np.linspace(1,0,160)
        x[round(onset*8000):round(onset*8000)+len(w)]=w
    result=segment(x,SETTINGS)
    assert len(result)==2
    assert abs(result[0]['onset']-.25)<.05 and abs(result[1]['onset']-1)<.05


@pytest.mark.parametrize('value',[0,127,128,960,100000])
def test_midi_varlen_roundtrip(value):
    decoded=0
    for b in varlen(value):decoded=(decoded<<7)+(b&127)
    assert decoded==value


def test_midi_validates_boundaries():
    with pytest.raises(ValueError):midi_events([dict(onset=1,offset=.5,midi=60)])
    with pytest.raises(ValueError):midi_events([dict(onset=0,offset=1,midi=float('nan'))])
    assert midi_events([]).startswith(b'MThd')
