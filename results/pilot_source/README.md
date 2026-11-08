# Piano Notes: an analysis-by-synthesis study

A small CPU experiment aligned with **CaCTüS 2027 Project 15, Piano Transcription with Differentiable Synthesizers**, in the Empirical Inference lab. It fits interpretable synthesizer controls to real isolated piano recordings without supplying reference pitches to the fitting function. It compares a conventional pitch estimate with fixed- and learned-decay differentiable synthesis.

## Why this matches the internship

The advertised project concerns label-free piano transcription, differentiable synthesis, and an intervenable latent space. This study implements a deliberately small part of that direction: reconstruct an isolated note, recover its pitch, and change fitted pitch/decay controls without refitting. Python, signal processing, automatic differentiation, controlled experiments, and error analysis are directly relevant preparation.

It is **not** a learned latent encoder, a causal-disentanglement result, Bayesian/variational inference, or polyphonic transcription. Each recording is fitted separately. A successful resynthesis is not proof that its inferred note is correct.

Official project catalogue, updated 3 November 2026:
https://engine.is.tue.mpg.de/uploads/ckeditor/attachments/2467/Website_CaCT%C3%BCS_Internship_Projects_2027-updated_NOV_3.pdf

## Read the results

Open `results/report.html` in a browser for the tables, plots, original/reconstructed audio, and parameter interventions. `results/REPORT.md` is the text report. Every sample also has a JSON record of controls and optimisation history, CSV metrics, and a one-note MIDI export. The MIDI duration is fixed to the one-second crop; onset/offset estimation is not claimed.

`results/development_initial/` preserves the unsuccessful initial octave-search experiment and its source code. `results/frozen.json` records the final model/protocol/data hashes, fixed before evaluation. Do not discard the initial failure when describing the project.

## Setup and run

Python 3.13 was used. No GPU or model download is needed. From this directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download.py
pytest -q
python experiment.py development
python experiment.py freeze
python experiment.py evaluation
python report.py
```

The download is approximately 1.25 MB. CPU runs use two PyTorch threads. The supplied archive already includes data and results; the runner refuses to overwrite an existing split. To rerun, first move the entire `results/` directory to a backup location. Run the commands above to create fresh evidence. These commands reproduce the final protocol; initial exploratory results are retained separately.

To modify fitted controls without refitting:

```bash
python intervene.py results/development/C4.json /tmp/pitchup.wav --semitones 12
python intervene.py results/development/C4.json /tmp/fastdecay.wav --decay-multiplier 2
```

The zero-intervention resynthesis is tested against the model output. Altering one control does not establish causal disentanglement: pitch changes also alter the set of partials below Nyquist.

## Method

1. Verify each source file's SHA-256. Decode MP3 to mono and resample to 8 kHz with polyphase filtering. Locate the isolated note's onset from signal energy, then take one second and peak-normalise it. Reject silent, corrupt and short inputs.
2. Estimate pitch with a conventional YIN-style cumulative mean normalised difference function, using a fixed 0.15 trough threshold and parabolic interpolation. The search range is 55–1,200 Hz. This is an in-project implementation, not a claim of bitwise equivalence to another YIN library.
3. Initialise 24 sinusoidal partials around that estimate. Fit partial amplitudes, a bounded inharmonicity coefficient and a pitch adjustment of at most 0.75 semitones using Adam. Partials above Nyquist are masked out. The fixed envelope has a 5 ms attack and exponential decay rate 2/s. The learned variant optimises a separate positive decay rate for each partial; everything else is identical.
4. Optimise 180 steps of a multi-resolution spectral objective (FFT sizes 256, 512, 1,024). Each term combines normalised linear-magnitude error and log-magnitude error. Use the same objective for reporting, so improvements are in-sample reconstruction improvements, not a separate perceptual quality test.
5. Only after fitting, read the filename's reference MIDI note to calculate signed and absolute cents error. Report all samples, note accuracy within 50 cents, octave errors and reconstruction loss. Do not equate better reconstruction with better transcription.

The additive synthesizer is an intentionally simple approximation: fixed zero oscillator phases, a fixed attack, limited bandwidth, no dedicated hammer-noise model, no pedal/soundboard model, and a short crop. Frequencies use `f_h = h*f0*sqrt(1+B*h*h)`, a simple stiff-string approximation; fitted `f0` is the nominal unstretched fundamental parameter. It is not a physical identification claim.

## Evaluation boundaries

- Four development notes and thirteen evaluation notes, with no overlapping note files, chosen in `protocol.json` before fitting.
- Development: C3, A3, C4, A4. Evaluation: C2, Ds2, Fs2, A2, Ds3, Fs3, Ds4, Fs4, C5, Ds5, Fs5, A5, C6.
- All recordings come from **one piano/sample set** and one available recording per selected pitch. Results cannot establish generalisation across pianos, dynamics, microphones or musical passages.
- These are notes withheld from method selection. Each evaluation waveform is still optimised at inference; there is no trained network generalising without adaptation.
- Ground-truth note names are never inputs to the optimiser. However, YIN supplies an audio-derived pitch prior and candidate refinement is deliberately bounded. This is not pitch recovery from random initialisation.
- Filename pitch labels identify nominal equal-tempered notes; actual instrument tuning can differ by a few cents.
- Learned decay has more free parameters and is optimised against the reporting loss. Lower loss is expected to be possible; accuracy, failure analysis and complexity are essential context.
- Small descriptive sample, no population-level significance or state-of-the-art claim.

## Data and attribution

Original recordings: **Alexander Holm, Salamander Grand Piano, Yamaha C5**. Audio is licensed **CC BY 3.0**. This experiment uses the MP3 derivatives distributed by Tone.js, not the complete original velocity-layer library. Preserve attribution when redistributing any originals or derivatives.

- Source revision: `efd8296360f9526e379bfbe5c1698ff54d6a1d34`
- Source: https://github.com/Tonejs/audio/tree/efd8296360f9526e379bfbe5c1698ff54d6a1d34/salamander
- Author/licence README: https://raw.githubusercontent.com/Tonejs/audio/efd8296360f9526e379bfbe5c1698ff54d6a1d34/salamander/README
- Licence: https://creativecommons.org/licenses/by/3.0/
- Modifications: MP3 decoding, downmixing, resampling, energy-based crop, peak normalisation and model-based resynthesis. No recording is described as collected by the applicant.

The source README header says V2 and includes a V3 changelog; provenance is therefore pinned by repository revision rather than asserting an unverified source release/velocity layer. `data/manifest.json` contains URLs, sizes and SHA-256 checksums.

## References and authorship

- CaCTüS Project 15: official catalogue linked above.
- Engel et al., *DDSP: Differentiable Digital Signal Processing*, ICLR 2020: https://arxiv.org/abs/2001.04643 ; reference implementation https://github.com/magenta/ddsp . This project is a small independent PyTorch implementation, not a reproduction of DDSP's published results.
- de Cheveigné and Kawahara, *YIN, a fundamental frequency estimator for speech and music*, JASA 2002, DOI: https://doi.org/10.1121/1.1458024 .

Built with substantial AI coding and research assistance at Hammad Zahid's request. Before presenting it as personal research, independently understand, rerun and review the implementation, experimental decisions and limitations. Do not represent AI-proposed research ideas as independently originated. CaCTüS's application policy permits language editing but requires applicants' substantive contributions to be their own.

## Files

- `model.py`: audio preparation, YIN-style baseline, differentiable synthesis and fitting.
- `experiment.py`: split execution, metrics, MIDI export and freeze checks.
- `intervene.py`: resynthesis with pitch/decay changes.
- `download.py`: pinned sample retrieval and checksums.
- `test_model.py`: numerical, input-validation, gradient, MIDI and intervention tests.
- `report.py`: regenerates the report and comparison plot from saved results.
- `protocol.json`: explicit experiment settings and split definitions.

No application was submitted, no CV was modified to claim this new work, and no external repository was published.
