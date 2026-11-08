"""Read-only reporting of the frozen independent study and melody experiment."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
LABELS={'yin':'YIN baseline','fixed':'Fixed decay, anchored','free':'Learned decay, free octave selection','anchor':'Learned decay, anchored','prior':'Learned decay, soft prior'}


def main():
    s=json.loads((R/'study_evaluation/summary.json').read_text())
    melodies=json.loads((R/'melody_evaluation/summary.json').read_text())
    with (R/'study_evaluation/metrics.csv').open() as f:rows=list(csv.DictReader(f))
    mdtable='| Method | Correct notes / 39 | One-octave errors | Median absolute cents | Mean absolute cents | Mean spectral loss |\n|---|---:|---:|---:|---:|---:|\n'
    trs=''
    for key,label in LABELS.items():
        m=s['methods'][key];values=[label,f"{m['within50']}/{s['n']}",str(m['octave_errors']),f"{m['median_absolute_cents']:.2f}",f"{m['mean_absolute_cents']:.2f}",f"{m['mean_loss']:.3f}" if key!='yin' else 'Not synthesised']
        mdtable+='| '+' | '.join(values)+' |\n';trs+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>'
    mel_table='| Method | Detected / reference | Pitch + onset F1 | Pitch + onset + offset F1 |\n|---|---:|---:|---:|\n';meltrs=''
    for key,label in [('yin','YIN + onset detector'),('synthesis','Differentiable refinement + same detector')]:
        m=melodies['methods'][key];a=m['pitch_onset'];b=m['pitch_onset_offset'];vals=[label,f"{a['predicted']}/{a['reference']}",f"{a['f1']:.3f}",f"{b['f1']:.3f}"]
        mel_table+='| '+' | '.join(vals)+' |\n';meltrs+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>'
    free=s['methods']['free'];prior=s['methods']['prior'];anchor=s['methods']['anchor'];baseline=s['methods']['yin']
    finding=(f"On {s['n']} independent Steinway recordings, unrestricted spectral-loss selection produced {free['octave_errors']} one-octave errors. "
             f"The predeclared soft prior produced {prior['octave_errors']}; hard anchoring produced {anchor['octave_errors']}. "
             f"YIN recovered {baseline['within50']}/{s['n']} notes within 50 cents. "
             f"Anchored learned decay lowered reconstruction loss on {s['anchor_loss_improved_count']}/{s['n']} recordings, "
             f"with {100*s['anchor_mean_relative_loss_reduction']:.2f}% mean per-recording relative reduction.")
    y=melodies['methods']['yin']['pitch_onset'];d=melodies['methods']['synthesis']['pitch_onset'];off=melodies['methods']['synthesis']['pitch_onset_offset']
    melody_finding=(f"Across {melodies['clips']} assembled melodies containing {melodies['reference_events']} reference notes, pitch-plus-onset F1 was {y['f1']:.3f} for YIN and {d['f1']:.3f} for differentiable refinement. "
                    f"Requiring offset agreement as well gave {off['f1']:.3f} for refinement. These are controlled sequences assembled from real samples, not human performances.")
    limitation=('Only two piano sample sets were examined, with the original Yamaha set used for development and a Steinway set reserved for evaluation. '
                'The 39 Steinway recordings cover 13 pitches × three dynamics; they are not 39 independent instruments. '
                'Per-note peak normalisation removes absolute loudness but retains timbral differences. '
                'Melodies reuse these evaluation recordings, so their results are a separate task, not an independent third dataset. '
                'Notes are intentionally separated, release tails are truncated with a fade, and no chords, pedal effects or expressive live-performance timing are modelled. '
                'Evaluation waveforms are fitted at inference; this is not a trained encoder generalising without adaptation. '
                'The spectral objective is both optimised and reported. No perceptual listening study, statistical superiority claim, or causal disentanglement claim is made.')
    report=f'''# Piano Notes: independent-instrument and transcription study

## Relevance to CaCTüS Project 15

Project 15 asks for label-free piano transcription with differentiable synthesis and controllable representations. This study implements per-recording synthesis fitting, editable pitch/decay controls, an explicit octave-selection ablation, and bounded monophonic MIDI transcription. It is a preparatory prototype, not a reproduction of the internship's full research objective.

## What changed from the pilot

- Preserved the original 17-recording Yamaha pilot, source snapshots and its failed octave-search experiment.
- Added 39 independent University of Iowa Steinway recordings across 13 pitches and three dynamics (pp, mf, ff).
- Compared the same learned candidates using free spectral-loss selection, hard anchoring and a soft pitch prior, alongside fixed-decay synthesis and YIN.
- Added spectral-flux onset detection, local-energy release rejection, offset estimation and timed MIDI export.
- Evaluated 12 assembled melodies (72 notes), including repeated-note passages, with one-to-one event matching.

## Protocol and independence

Only the four original Yamaha development notes and three development melodies were used to check the upgraded method. The independent Steinway set was not used for tuning. `study_frozen.json` records hashes of the implementation, settings and both sample manifests before independent evaluation. Fitting and candidate selection receive audio, not filenames or reference pitches.

YIN proposes a pitch. Three hypotheses (half, equal, double) are optimised with 24 partials, 180 Adam steps and the original multi-resolution spectral objective. Free selection takes the lowest reconstruction loss. Hard anchoring keeps the candidate initialised at YIN. Soft selection minimises `spectral_loss + 0.3 * abs(log2(fitted_pitch / YIN_pitch))`. The prior is a selection penalty, not an extra gradient term. These methods reuse the same candidate fits, isolating selection behaviour.

The fixed-decay comparison fits only the YIN-initialised candidate with the same 180-step budget. The learned candidates are fitted together; floating-point trajectories can differ slightly from the pilot's single-candidate fit. CPU execution uses two threads.

## Independent isolated-note results

{mdtable}
{finding}

The JSON summary also reports per-dynamic results and descriptive 95% bootstrap intervals for mean absolute pitch error. Resampling groups all three dynamics of a pitch together. These intervals characterise this small pitch set; they do not establish performance across instruments or populations.

## Melody transcription results

{mel_table}
{melody_finding}

Both methods share the same onset/offset detector; only their pitch estimator differs. Matching is one-to-one, with onset tolerance 50 ms and pitch tolerance 50 cents. The stricter measure also requires offset tolerance 50 ms. MIDI note times are inferred from audio, not copied from reference events. Velocity is fixed to 80 and is not inferred.

Sequences are constructed from six source samples each, with durations 0.35/0.50/0.70 seconds and gaps 0.08/0.12/0.20 seconds. Every third clip contains a repeated note. The construction seed, sample identities and reference events are saved. The detector sees only the assembled waveform. Pitch labels and constructed timing enter only evaluation.

A development unit test caught release transients being counted as onsets. A local-energy-rise condition corrected the detector before freezing. All 18 development notes passed pitch/onset/offset matching; evaluation remains the basis for claims.

## Failure analysis

The octave-error column counts errors within 50 cents of exactly one octave. Other mistakes, including two-octave errors, remain included in total accuracy and mean/median error. On `Piano.pp.C6`, YIN was about one octave low and free spectral-loss selection was about two octaves low; the soft prior also selected the latter. On `Piano.pp.Eb4`, YIN and anchored fitting were about 30 semitones low. The prior therefore cannot be described as a general solution to pitch ambiguity.

In melody `clip02`, both methods recovered only four of six reference events within pitch/onset tolerance, while producing seven predictions. One onset was 54 ms late (outside the predeclared 50 ms tolerance), and the last reference note was split into two detections, with an incorrect pitch in the earlier segment. These saved events demonstrate failures beyond isolated-note tuning errors. Their acoustic causes need further investigation; no post-test tuning or discarded samples are hidden.

Review every candidate in `study_evaluation/*.json`, all rows in `study_evaluation/metrics.csv`, and every reference/prediction in `melody_evaluation/clip*.json`. The report does not discard incorrect pitches or false/missed notes. A low spectral loss can favour a lower octave because its higher partials explain the observed note. A pitch prior can suppress this ambiguity but inherits the limitations of its YIN estimate. It is not evidence that reconstruction alone identifies the correct fundamental.

## Limitations

{limitation}

## Provenance

Iowa source: https://theremin.music.uiowa.edu/MISpiano.html . University of Iowa Electronic Music Studios / Lawrence Fritts. The source's usage statement allows the recordings to be downloaded and used for projects without restrictions; no invented SPDX or public-domain designation is assigned. Local `data/iowa/USE_PERMISSION.html` preserves that statement.

To avoid downloading long stereo decays, each Iowa file is a verified HTTP byte-range prefix (first 1,048,576 bytes, approximately 5.94 seconds at 44.1 kHz stereo 16-bit). Manifest hashes refer to those prefixes, not the complete remote file. Original content ranges and total source sizes are recorded. Only the first energy-aligned second is used for isolated-note evaluation. Crops and sequences are explicitly derived data.

Salamander/Yamaha recordings: Alexander Holm, distributed by Tone.js, CC BY 3.0; original pinned revision and checksums remain in the pilot manifest and README.

## Authorship and application use

This project was substantially AI-assisted. The applicant should independently understand, rerun and verify it before presenting it in an interview. No publication, independent origination, or state-of-the-art claim is warranted. No application or updated CV was submitted.
'''
    (R/'STUDY_REPORT.md').write_text(report)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    keys=list(LABELS);short=['YIN','Fixed','Free','Anchor','Prior']
    axes[0].bar(short,[s['methods'][k]['within50']/s['n'] for k in keys],color=['#778995','#b4884d','#a85050','#2d7d79','#23495e'])
    axes[0].set(ylabel='Fraction within 50 cents',ylim=(0,1.08),title='Independent piano: correct nominal pitch')
    for i,(key,label) in enumerate([('yin','YIN'),('synthesis','Refinement')]):
        axes[1].bar(np.arange(2)+(i-.5)*.3,[melodies['methods'][key][m]['f1'] for m in ['pitch_onset','pitch_onset_offset']],.3,label=label)
    axes[1].set(xticks=[0,1],xticklabels=['Pitch + onset','Pitch + onset + offset'],ylabel='Micro F1',ylim=(0,1.08),title='12 assembled melodies, 72 reference notes');axes[1].legend(frameon=False)
    fig.savefig(R/'study_comparison.png',dpi=160);plt.close(fig)
    cards=''
    for file in sorted((R/'melody_evaluation').glob('clip*.json')):
        clip=json.loads(file.read_text());name=file.stem
        fig,ax=plt.subplots(figsize=(10,2.5),constrained_layout=True)
        for label,events,colour,shift in [('Reference',clip['reference'],'#263b4a',.16),('YIN',clip['predictions']['yin'],'#be8c49',0),('Refinement',clip['predictions']['synthesis'],'#16746d',-.16)]:
            for i,e in enumerate(events):ax.plot([e['onset'],e['offset']],[e['midi']+shift]*2,lw=3,color=colour,label=label if i==0 else None)
        ax.set(xlabel='Seconds',ylabel='MIDI pitch',title=name+' — note intervals');ax.legend(ncol=3,fontsize=8,frameon=False)
        fig.savefig(R/'melody_evaluation'/f'{name}.png',dpi=120);plt.close(fig)
        cards+=f'<article><h3>{name}</h3><audio controls preload="none" src="melody_evaluation/{name}.wav"></audio><img src="melody_evaluation/{name}.png" alt="Reference and predicted note intervals for {name}"><p><a href="melody_evaluation/{name}_reference.mid">Reference MIDI</a> · <a href="melody_evaluation/{name}_yin.mid">YIN MIDI</a> · <a href="melody_evaluation/{name}_synthesis.mid">Refinement MIDI</a> · <a href="melody_evaluation/{name}.json">All events and metrics</a></p></article>'
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Piano Notes — independent study</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#23343f;background:#f5f7f6;margin:0}}main{{max-width:1080px;margin:auto;padding:40px 24px}}h1{{font:46px Georgia,serif;color:#143f50}}h2{{margin-top:34px}}article,.finding{{background:white;border:1px solid #dbe3e3;border-radius:10px;padding:22px;margin:18px 0}}.finding{{border-left:5px solid #16746d}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce4e6}}th{{background:#e7eeef}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}audio{{width:100%;max-width:550px}}a{{color:#126d77}}.kicker{{text-transform:uppercase;letter-spacing:.1em;color:#526b75;font-size:13px}}</style><main><p class="kicker">CaCTüS Project 15 · independent piano evaluation</p><h1>Piano Notes: beyond the pilot</h1><div class="finding"><p>{finding}</p><p>{melody_finding}</p></div><p><a href="STUDY_REPORT.md">Full research report</a> · <a href="../README.md">Reproduce</a> · <a href="report.html">Preserved pilot</a> · <a href="study_frozen.json">Frozen protocol</a></p><img src="study_comparison.png" alt="Independent pitch accuracy and assembled melody F1"><h2>39 independent piano recordings</h2><p>13 pitches × soft, medium and loud dynamics. Correct means within 50 cents of the nominal filename pitch.</p><div class="scroll"><table><tr><th>Method</th><th>Correct</th><th>One-octave errors</th><th>Median cents</th><th>Mean cents</th><th>Spectral loss</th></tr>{trs}</table></div><p><a href="study_evaluation/metrics.csv">Every recording</a> · <a href="study_evaluation/summary.json">Dynamics and bootstrap intervals</a></p><h2>12 assembled melody tests</h2><p>Real sampled notes with constructed timing. These are not live human performances. Both estimators share an audio-derived onset/offset detector.</p><div class="scroll"><table><tr><th>Method</th><th>Detected / reference</th><th>Pitch + onset F1</th><th>With offset F1</th></tr>{meltrs}</table></div>{cards}<h2>What this does not establish</h2><p>{limitation}</p><p>Audio credit: Alexander Holm / Salamander (CC BY 3.0), and University of Iowa Electronic Music Studios / Lawrence Fritts (source use permission preserved). Substantial AI assistance was used. Review the source, reproduce the results and accurately describe your contribution before using this work in an application.</p></main></html>'''
    (R/'study.html').write_text(doc)
    bullet=(f"Built an AI-assisted differentiable piano-synthesis and MIDI-transcription prototype; evaluated 39 independent recordings across three dynamics and {melodies['clips']} controlled melodies ({melodies['reference_events']} notes), with pitch/onset F1 {d['f1']:.3f}; investigated octave ambiguity using reconstruction-only and pitch-prior selection.")
    (ROOT/'CV_EVIDENCE.txt').write_text('Suggested CV wording, after independently reviewing and rerunning the project:\n\n'+bullet+'\n\nLimits: controlled sample-assembled melodies, not human performances; no claim of outperforming YIN. See results/STUDY_REPORT.md.\n')
    print(R/'study.html')

if __name__=='__main__':main()
