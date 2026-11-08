# Piano Notes: independent-instrument and transcription study

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

| Method | Correct notes / 39 | One-octave errors | Median absolute cents | Mean absolute cents | Mean spectral loss |
|---|---:|---:|---:|---:|---:|
| YIN baseline | 37/39 | 1 | 8.35 | 116.51 | Not synthesised |
| Fixed decay, anchored | 37/39 | 1 | 10.76 | 117.13 | 2.119 |
| Learned decay, free octave selection | 16/39 | 21 | 1185.74 | 751.50 | 1.955 |
| Learned decay, anchored | 37/39 | 1 | 9.93 | 115.79 | 2.036 |
| Learned decay, soft prior | 36/39 | 1 | 9.93 | 176.73 | 2.020 |

On 39 independent Steinway recordings, unrestricted spectral-loss selection produced 21 one-octave errors. The predeclared soft prior produced 1; hard anchoring produced 1. YIN recovered 37/39 notes within 50 cents. Anchored learned decay lowered reconstruction loss on 37/39 recordings, with 4.27% mean per-recording relative reduction.

The JSON summary also reports per-dynamic results and descriptive 95% bootstrap intervals for mean absolute pitch error. Resampling groups all three dynamics of a pitch together. These intervals characterise this small pitch set; they do not establish performance across instruments or populations.

## Melody transcription results

| Method | Detected / reference | Pitch + onset F1 | Pitch + onset + offset F1 |
|---|---:|---:|---:|
| YIN + onset detector | 87/72 | 0.604 | 0.415 |
| Differentiable refinement + same detector | 87/72 | 0.604 | 0.415 |

Across 12 assembled melodies containing 72 reference notes, pitch-plus-onset F1 was 0.604 for YIN and 0.604 for differentiable refinement. Requiring offset agreement as well gave 0.415 for refinement. These are controlled sequences assembled from real samples, not human performances.

Both methods share the same onset/offset detector; only their pitch estimator differs. Matching is one-to-one, with onset tolerance 50 ms and pitch tolerance 50 cents. The stricter measure also requires offset tolerance 50 ms. MIDI note times are inferred from audio, not copied from reference events. Velocity is fixed to 80 and is not inferred.

Sequences are constructed from six source samples each, with durations 0.35/0.50/0.70 seconds and gaps 0.08/0.12/0.20 seconds. Every third clip contains a repeated note. The construction seed, sample identities and reference events are saved. The detector sees only the assembled waveform. Pitch labels and constructed timing enter only evaluation.

A development unit test caught release transients being counted as onsets. A local-energy-rise condition corrected the detector before freezing. All 18 development notes passed pitch/onset/offset matching; evaluation remains the basis for claims.

## Failure analysis

The octave-error column counts errors within 50 cents of exactly one octave. Other mistakes, including two-octave errors, remain included in total accuracy and mean/median error. On `Piano.pp.C6`, YIN was about one octave low and free spectral-loss selection was about two octaves low; the soft prior also selected the latter. On `Piano.pp.Eb4`, YIN and anchored fitting were about 30 semitones low. The prior therefore cannot be described as a general solution to pitch ambiguity.

In melody `clip02`, both methods recovered only four of six reference events within pitch/onset tolerance, while producing seven predictions. One onset was 54 ms late (outside the predeclared 50 ms tolerance), and the last reference note was split into two detections, with an incorrect pitch in the earlier segment. These saved events demonstrate failures beyond isolated-note tuning errors. Their acoustic causes need further investigation; no post-test tuning or discarded samples are hidden.

Review every candidate in `study_evaluation/*.json`, all rows in `study_evaluation/metrics.csv`, and every reference/prediction in `melody_evaluation/clip*.json`. The report does not discard incorrect pitches or false/missed notes. A low spectral loss can favour a lower octave because its higher partials explain the observed note. A pitch prior can suppress this ambiguity but inherits the limitations of its YIN estimate. It is not evidence that reconstruction alone identifies the correct fundamental.

## Limitations

Only two piano sample sets were examined, with the original Yamaha set used for development and a Steinway set reserved for evaluation. The 39 Steinway recordings cover 13 pitches × three dynamics; they are not 39 independent instruments. Per-note peak normalisation removes absolute loudness but retains timbral differences. Melodies reuse these evaluation recordings, so their results are a separate task, not an independent third dataset. Notes are intentionally separated, release tails are truncated with a fade, and no chords, pedal effects or expressive live-performance timing are modelled. Evaluation waveforms are fitted at inference; this is not a trained encoder generalising without adaptation. The spectral objective is both optimised and reported. No perceptual listening study, statistical superiority claim, or causal disentanglement claim is made.

## Provenance

Iowa source: https://theremin.music.uiowa.edu/MISpiano.html . University of Iowa Electronic Music Studios / Lawrence Fritts. The source's usage statement allows the recordings to be downloaded and used for projects without restrictions; no invented SPDX or public-domain designation is assigned. Local `data/iowa/USE_PERMISSION.html` preserves that statement.

To avoid downloading long stereo decays, each Iowa file is a verified HTTP byte-range prefix (first 1,048,576 bytes, approximately 5.94 seconds at 44.1 kHz stereo 16-bit). Manifest hashes refer to those prefixes, not the complete remote file. Original content ranges and total source sizes are recorded. Only the first energy-aligned second is used for isolated-note evaluation. Crops and sequences are explicitly derived data.

Salamander/Yamaha recordings: Alexander Holm, distributed by Tone.js, CC BY 3.0; original pinned revision and checksums remain in the pilot manifest and README.

## Authorship and application use

This project was substantially AI-assisted. The applicant should independently understand, rerun and verify it before presenting it in an interview. No publication, independent origination, or state-of-the-art claim is warranted. No application or updated CV was submitted.
