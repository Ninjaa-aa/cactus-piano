"""Build an offline report from recorded evidence, without rerunning fitting."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from intervene import render_controls

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'


def main():
    summary=json.loads((R/'evaluation/summary.json').read_text())
    config=json.loads((ROOT/'protocol.json').read_text())
    with (R/'evaluation/metrics.csv').open() as f:rows=list(csv.DictReader(f))
    methods=[('yin','YIN baseline'),('fixed','Fixed decay'),('learned','Learned decay')]
    table='| Method | Notes within 50 cents | Median abs. cents | Mean abs. cents | Mean spectral loss |\n|---|---:|---:|---:|---:|\n'
    html_rows=''
    for key,label in methods:
        m=summary['methods'][key]
        loss=f"{m['mean_spectral_loss']:.3f}" if key!='yin' else 'Not synthesised'
        cells=[label,f"{m['within_50_cents']}/{summary['n']}",f"{m['median_absolute_cents']:.2f}",f"{m['mean_absolute_cents']:.2f}",loss]
        table+='| '+' | '.join(cells)+' |\n'
        html_rows+='<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>'
    improvement=100*summary['mean_relative_loss_reduction']
    finding=(f"Learned decay lowered the fitted spectral objective on {summary['learned_lower_loss_count']} of {summary['n']} evaluation notes. "
             f"The mean per-note relative reduction was {improvement:.2f}%. YIN had the lowest median and mean absolute pitch error. "
             'All methods were within 50 cents on every evaluation note. These results support improved reconstruction overall, not improved transcription over YIN.')
    failure=('Initial development allowed candidates at half, equal and double the YIN estimate. It selected an octave-too-low fundamental on 3 of 4 development notes: higher partials could explain the recording. '
             'The initial results and sources are preserved. Final refinement was bounded to ±0.75 semitones around YIN, then frozen before evaluation. This audio-derived prior prevents broad octave search; the final zero-octave-error count is not evidence of unrestricted octave disambiguation.')
    limits=('13 evaluation notes from one piano/sample set; one second per note at 8 kHz; no independent instrument or velocity-layer evaluation. '
            'Each evaluation waveform is fitted at inference. The same spectral objective is optimised and reported. Learned envelopes add parameters. '
            'No onset/offset detection, chords, sustain-pedal inference, learned encoder, variational inference or causal representation learning is implemented. '
            'Filename pitches are nominal labels; deviations of a few cents can include instrument tuning. No listening-quality claim or population-level significance test is made.')
    weakest=sorted(rows,key=lambda r:float(r['learned_loss'])-float(r['fixed_loss']),reverse=True)[:2]
    negatives='; '.join(f"{r['note']}: fixed {float(r['fixed_loss']):.3f}, learned {float(r['learned_loss']):.3f}" for r in weakest)
    report=f'''# Piano Notes: analysis by synthesis

## Question and internship relevance

Does learning a piano note's partial decay envelopes improve reconstruction, and does that also improve pitch recovery? This is a small preparation study for CaCTüS 2027 Project 15: Piano Transcription with Differentiable Synthesizers. It implements label-free per-recording fitting and directly editable synthesizer controls, not the advertised full research programme.

## Protocol

17 real Salamander Yamaha C5 recordings: 4 development and 13 evaluation notes. Ground-truth filename pitches are used only for split definition and post-fit evaluation. YIN initialises the pitch, followed by 180 Adam steps on a 24-partial differentiable synthesizer. Fixed and learned variants share every setting except partial decay optimisation. The experiment uses two CPU threads. The evaluation run took {summary['elapsed_seconds']:.1f} seconds on this machine.

## Evaluation results

{table}
{finding}

Reconstruction regressions: {negatives}. More parameters and the same step budget do not guarantee a better optimum. All sample-level outcomes are in `evaluation/metrics.csv`; no difficult notes were dropped.

## Development failure and correction

{failure}

## Interventions

The report includes resynthesis of development note C4 with pitch shifted by 12 semitones and with decay rates doubled. These controls are edited without refitting; the decoder makes their effect explicit. This demonstrates controllability, not causal disentanglement. MIDI exports contain one rounded-pitch note with fixed one-second duration and velocity 80; expressive velocity and note boundaries are not inferred.

## Limits

{limits}

## Reproduce and inspect

See `../README.md` for setup and commands, `frozen.json` for pre-evaluation hashes, `evaluation/provenance.json` for environment/configuration, and `../data/manifest.json` for source checksums. `development_initial/` retains the original failure and exact source files. `verification.json` records tests, audio checks and deterministic rerun validation when generated.

## Attribution and assistance

Audio: Alexander Holm, Salamander Grand Piano, Yamaha C5, distributed by Tone.js; CC BY 3.0. Source commit efd8296360f9526e379bfbe5c1698ff54d6a1d34. Original audio was decoded, downmixed, resampled, cropped and peak-normalised; reconstructed/intervened samples are model outputs.

This project was substantially AI-assisted. Hammad should independently review, understand and rerun it before using it as application evidence. It is not a publication, an accepted research result, or a claim of independently originated ideas.
'''
    (R/'REPORT.md').write_text(report)
    colors=['#657785','#a57642','#16746d']
    fig,axes=plt.subplots(2,1,figsize=(10,6.5),constrained_layout=True)
    x=np.arange(len(rows));width=.25
    for i,(key,label) in enumerate(methods):axes[0].bar(x+(i-1)*width,[abs(float(r[key+'_cents'])) for r in rows],width,label=label,color=colors[i])
    axes[0].set(ylabel='Absolute pitch error (cents)',xticks=x,xticklabels=[r['note'] for r in rows],title='Reserved evaluation notes: pitch and reconstruction are different objectives')
    axes[0].legend(frameon=False,ncol=3);axes[0].grid(axis='y',alpha=.2)
    for key,label,col in [('fixed','Fixed decay',colors[1]),('learned','Learned decay',colors[2])]:
        axes[1].plot(x,[float(r[key+'_loss']) for r in rows],marker='o',label=label,color=col)
    axes[1].set(ylabel='Fitted spectral objective (lower is better)',xticks=x,xticklabels=[r['note'] for r in rows]);axes[1].legend(frameon=False);axes[1].grid(alpha=.2)
    fig.savefig(R/'comparison.png',dpi=160);plt.close(fig)
    controls=json.loads((R/'development/C4.json').read_text())['results']['learned']
    outputs=[('pitchup',12,1),('fastdecay',0,2)]
    for name,semi,decay in outputs:
        audio=render_controls(controls,config,semi,decay)
        sf.write(R/f'C4_{name}.wav',audio*min(1,.98/np.max(np.abs(audio))),8000,subtype='PCM_16')
    cards=''
    for r in rows:
        note=html.escape(r['note'])
        audios=''.join(f'<div><label>{label}</label><audio controls preload="none" src="evaluation/{note}_{key}.wav"></audio></div>' for key,label in [('original','Original crop'),('fixed','Fixed decay'),('learned','Learned decay')])
        cards+=f'<article><h3>{note}</h3><div class="audio-grid">{audios}</div><p><a href="evaluation/{note}_learned.mid">One-note MIDI</a> · <a href="evaluation/{note}.json">Fitted controls and loss history</a></p></article>'
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Piano Notes — CaCTüS preparation study</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#23343f;background:#f5f7f6;margin:0}}main{{max-width:1040px;margin:auto;padding:42px 24px}}h1{{font:48px/1.1 Georgia,serif;margin:10px 0;color:#143f50}}h2{{margin-top:34px}}.kicker{{letter-spacing:.12em;text-transform:uppercase;font-size:13px;color:#526b75}}.finding,article{{background:white;padding:22px;border:1px solid #dbe3e3;border-radius:10px;margin:18px 0}}.finding{{border-left:5px solid #16746d}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;padding:11px;border-bottom:1px solid #dce4e6}}th{{background:#e7eeef}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}.audio-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}}audio{{width:100%}}label{{display:block;font-weight:600}}a{{color:#126d77}}footer{{border-top:1px solid #ccd9db;margin-top:36px;padding-top:18px;color:#53636c}}@media(max-width:700px){{.audio-grid{{grid-template-columns:1fr}}h1{{font-size:36px}}}}</style>
<main><div class="kicker">CaCTüS Project 15 · CPU research prototype</div><h1>Piano Notes</h1><p>Can a better reconstruction recover a better note?</p>
<div class="finding"><strong>Measured result</strong><p>{finding}</p></div>
<p>Real piano audio → YIN pitch estimate → differentiable synthesis → spectral fitting → pitch and reconstruction evaluation.</p>
<h2>Evaluation on 13 reserved notes</h2><div class="scroll"><table><thead><tr><th>Method</th><th>Within 50 cents</th><th>Median absolute cents</th><th>Mean absolute cents</th><th>Mean spectral loss</th></tr></thead><tbody>{html_rows}</tbody></table></div>
<p>The entire evaluation took {summary['elapsed_seconds']:.1f} seconds using two CPU threads. No GPU was used.</p><img src="comparison.png" alt="Per-note pitch errors and reconstruction loss for all thirteen evaluation notes">
<h2>The failure that changed the method</h2><p>{failure}</p><p><a href="development_initial/summary.json">Original development results</a> · <a href="frozen.json">Frozen protocol hashes</a></p>
<h2>Change controls without refitting</h2><p>Development note C4. A pitch change and a faster decay are decoder interventions, not evidence of learned causal factors.</p><div class="audio-grid"><div><label>Fitted C4</label><audio controls preload="none" src="development/C4_learned.wav"></audio></div><div><label>Pitch +12 semitones</label><audio controls preload="none" src="C4_pitchup.wav"></audio></div><div><label>Decay rates ×2</label><audio controls preload="none" src="C4_fastdecay.wav"></audio></div></div>
<h2>Listen to every evaluation note</h2><p>Playback uses one common gain across each original/fixed/learned group to avoid clipping. Crops were peak-normalised before fitting. MIDI uses a fixed one-second duration; it is not onset/offset transcription.</p>{cards}
<h2>Scope and limitations</h2><p>{limits}</p><p>Reconstruction regressions: {negatives}.</p>
<p><a href="REPORT.md">Full text report</a> · <a href="evaluation/metrics.csv">All metrics</a> · <a href="evaluation/provenance.json">Environment and provenance</a> · <a href="../README.md">Reproduction instructions</a></p>
<footer>Audio: Alexander Holm / Salamander Grand Piano, distributed by Tone.js, <a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>. Cropped/resampled originals and model reconstructions are labelled above. Substantial AI assistance was used; no independent-authorship or publication claim is made.</footer></main></html>'''
    (R/'report.html').write_text(doc)
    print(R/'report.html')

if __name__=='__main__':main()
