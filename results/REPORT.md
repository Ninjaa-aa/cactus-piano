# Piano Notes: analysis by synthesis

## Question and internship relevance

Does learning a piano note's partial decay envelopes improve reconstruction, and does that also improve pitch recovery? This is a small preparation study for CaCTüS 2027 Project 15: Piano Transcription with Differentiable Synthesizers. It implements label-free per-recording fitting and directly editable synthesizer controls, not the advertised full research programme.

## Protocol

17 real Salamander Yamaha C5 recordings: 4 development and 13 evaluation notes. Ground-truth filename pitches are used only for split definition and post-fit evaluation. YIN initialises the pitch, followed by 180 Adam steps on a 24-partial differentiable synthesizer. Fixed and learned variants share every setting except partial decay optimisation. The experiment uses two CPU threads. The evaluation run took 66.1 seconds on this machine.

## Evaluation results

| Method | Notes within 50 cents | Median abs. cents | Mean abs. cents | Mean spectral loss |
|---|---:|---:|---:|---:|
| YIN baseline | 13/13 | 2.66 | 6.16 | Not synthesised |
| Fixed decay | 13/13 | 7.78 | 6.89 | 2.115 |
| Learned decay | 13/13 | 3.81 | 7.08 | 2.021 |

Learned decay lowered the fitted spectral objective on 11 of 13 evaluation notes. The mean per-note relative reduction was 4.71%. YIN had the lowest median and mean absolute pitch error. All methods were within 50 cents on every evaluation note. These results support improved reconstruction overall, not improved transcription over YIN.

Reconstruction regressions: C2: fixed 2.494, learned 2.586; C5: fixed 2.461, learned 2.544. More parameters and the same step budget do not guarantee a better optimum. All sample-level outcomes are in `evaluation/metrics.csv`; no difficult notes were dropped.

## Development failure and correction

Initial development allowed candidates at half, equal and double the YIN estimate. It selected an octave-too-low fundamental on 3 of 4 development notes: higher partials could explain the recording. The initial results and sources are preserved. Final refinement was bounded to ±0.75 semitones around YIN, then frozen before evaluation. This audio-derived prior prevents broad octave search; the final zero-octave-error count is not evidence of unrestricted octave disambiguation.

## Interventions

The report includes resynthesis of development note C4 with pitch shifted by 12 semitones and with decay rates doubled. These controls are edited without refitting; the decoder makes their effect explicit. This demonstrates controllability, not causal disentanglement. MIDI exports contain one rounded-pitch note with fixed one-second duration and velocity 80; expressive velocity and note boundaries are not inferred.

## Limits

13 evaluation notes from one piano/sample set; one second per note at 8 kHz; no independent instrument or velocity-layer evaluation. Each evaluation waveform is fitted at inference. The same spectral objective is optimised and reported. Learned envelopes add parameters. No onset/offset detection, chords, sustain-pedal inference, learned encoder, variational inference or causal representation learning is implemented. Filename pitches are nominal labels; deviations of a few cents can include instrument tuning. No listening-quality claim or population-level significance test is made.

## Reproduce and inspect

See `../README.md` for setup and commands, `frozen.json` for pre-evaluation hashes, `evaluation/provenance.json` for environment/configuration, and `../data/manifest.json` for source checksums. `development_initial/` retains the original failure and exact source files. `verification.json` records tests, audio checks and deterministic rerun validation when generated.

## Attribution and assistance

Audio: Alexander Holm, Salamander Grand Piano, Yamaha C5, distributed by Tone.js; CC BY 3.0. Source commit efd8296360f9526e379bfbe5c1698ff54d6a1d34. Original audio was decoded, downmixed, resampled, cropped and peak-normalised; reconstructed/intervened samples are model outputs.

This project was substantially AI-assisted. Hammad should independently review, understand and rerun it before using it as application evidence. It is not a publication, an accepted research result, or a claim of independently originated ideas.
