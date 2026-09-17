# Completed TDA comparison and proposed GATr follow-up

Analyzed September 17, 2026. TDA extraction/readouts and the separate H100 onset
repeats are complete. All comparisons use one training seed; source-bootstrap
intervals do not include training-seed uncertainty.

## Instantaneous topology retained by frozen physical snapshot states

These are current-position targets, not future topology or relaxed topology.
The frozen encoders were trained on physical packets, without TDA supervision.
There are 7,680 test windows from 30 held-out sources; 5,081 test centers belong
to the observed noncrystalline subset (PTM type not in {1,2,3}, including unknown).
Errors balance sources and H0/H1/H2, using training-only target scales.

| Test population/readout | MACE | GATr | GATr relative error reduction, source-bootstrap 95% interval |
| --- | ---: | ---: | --- |
| All states, ridge | 0.22878 | 0.22265 | 2.68% [−3.07%, 8.53%] |
| All states, nonlinear | 0.19837 | 0.19719 | 0.59% [−7.74%, 8.95%] |
| Noncrystalline, ridge | 0.24042 | 0.22173 | 7.78% [0.48%, 14.94%] |
| Noncrystalline, nonlinear | 0.21400 | 0.20734 | 3.11% [−7.54%, 13.52%] |

Both states substantially improve on the all-state training-mean error (0.94457)
and condition-only ridge (0.68413). Overall topology is approximately tied.
The noncrystalline linear readout favors GATr in this one-seed comparison; its
nonlinear advantage is inconclusive. This does not establish a general GATr win.

With nonlinear readouts, all-state H0/H1/H2 R2 are respectively
0.730/0.805/0.813 for MACE and 0.742/0.807/0.802 for GATr. GATr's H2 point
estimate is worse. On the noncrystalline subset, all three within-frame R2 values
remain negative for both encoders: reconstruction error exceeds the measured
variation across centers within a source/frame. Good aggregate reconstruction
therefore does not establish accurate fine local distinctions.

Source files: [metrics](technical/metrics.json),
[table and definitions](tables/topology.csv), [metric protocol](tables/METRICS.md),
[readout selection](technical/readout_selection.json). Completion checksums and
current metric contracts were verified before this interpretation.

## Separate physical prediction and onset results

The matched 2,048-update physical screen has present MSE 0.53652 versus 0.71389
(MACE versus GATr), and mean future MSE 0.79993 versus 0.85640. GATr's future
error is 7.06% higher. These errors have a different normalization from TDA.

| Current physical target block | MACE MSE | GATr MSE |
| --- | ---: | ---: |
| Radial structure | 0.49939 | 0.80246 |
| Pair distances | 0.64381 | 0.88107 |
| Angular structure | 0.71790 | 0.99442 |
| Speed | 0.18177 | 0.16661 |
| Radial velocity | 0.48282 | 0.44655 |
| Mixed geometric/motion moments | 0.62325 | 0.73648 |

The observed weakness is predominantly geometric, not uniformly weak motion
encoding. This diagnoses the encoder-plus-native-decoder combination; it does
not yet locate the failure inside the encoder.

The separate onset fits have these joint event NLLs (lower is better):

| Input | MACE | GATr |
| --- | ---: | ---: |
| Snapshot | 1.10252 | 1.10336 |
| Real 12 ps history | 1.10246 | 1.09750 |
| Repeated-current-frame control | 1.10253 | 1.09556 |

GATr history improves over its snapshot on this metric, but repeated frames do
better still. This comparison does not demonstrate a useful history benefit.
Nine-ps onset average precision remains lower for GATr (history: 0.03689 versus
MACE 0.04051); these are distinct metrics and are not a broad superiority claim.
The existing intervals for individual models are not paired intervals for the
architecture difference.

The explicit same-H100 resident-batch profile gives GATr 1.50x snapshot update
throughput. History updates take 0.516 s versus MACE 0.471 s: GATr is 9.65%
slower there and uses 23.45 versus 16.78 GiB peak allocated memory. This is not
whole-run throughput. [Comparison exports](../h100-backbone-comparison-20260917/tables/).

## Proposed follow-up, not yet implemented or launched

1. **Diagnose physical readout fitting cheaply.** Reuse the saved frozen states
   to fit matched fresh ridge/nonlinear readouts for present physical blocks and
   the six future horizons. Use only training/selection sources for fitting and
   selection. Compare against the native heads. Improvement would implicate
   optimization/readout accessibility; persistence of the geometry gap would
   motivate encoder changes. TDA alone cannot settle that distinction.
2. **Test a smooth multiscale graph readout.** The current GATr exports only the
   final center's scalar channels. Keep that path and add smooth summaries of
   current atom features at center/inner/context scales, weighted counts, then
   project once to z128. In particular the 5–7 Angstrom region matches the
   physical packet support. MACE already uses multiscale pooling. This is a
   concrete architectural asymmetry, not proof that pooling causes the gap.
3. **Test explicit local geometry if needed.** Add radial basis features and a
   smooth pair-distance locality bias/local attention block while retaining
   broader context. The upstream GATr attention already includes Euclidean
   distance features; the hypothesis concerns locality and optimization, not
   an absence of geometry. A dense local mask alone does not establish a speedup.
   [Pinned upstream attention implementation](https://raw.githubusercontent.com/Qualcomm-AI-research/geometric-algebra-transformer/6afc26f26b8fcf51136ae8c1d264a36e14b6e497/gatr/primitives/attention.py).
4. **Use an informative training budget before scaling width.** Both physical
   models selected the final 2,048-update checkpoint; that is only 16,384 sampled
   training examples against 23,040 available windows, with replacement. A
   one-seed 8,192-update screen of unchanged MACE, unchanged GATr, and GATr with
   multiscale pooling would separate longer training from a readout change.
   Use the same sampler, targets and maximum budget; select on validation and
   report quality versus both updates and measured wall time. Test local bias
   after this comparison, rather than combining changes without controls.

If retained topology becomes an explicit training requirement, a subsequent
small auxiliary TDA loss can use these cached training targets. Apply it as a
separate objective ablation to both architectures, and retain independent future
physical/onset evaluation. It would no longer be an unsupervised TDA-retention test.
