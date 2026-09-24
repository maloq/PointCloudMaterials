# Completed first wave: information additions to relaxed-MACE forecasts

All12 fits completed on the same7654 test windows from30 sources. There are226
positive12 ps windows; overlapping windows are not independent events. One seed,
historical test sources, frozen MACE encoders, validation-selected checkpoints.

| Input / change | AP12 ↑ | Integrated Brier .75–12 ps ↓ | Detected timing MAE (ps) ↓ | Missed / positive | Realized test FPR |
|---|---:|---:|---:|---:|---:|
| Original reference | .5738 | .014492 | 2.344 | 92 / 226 | .0317 |
| Relaxed reference | .6457 | .012571 | 2.413 | 75 / 226 | .0266 |
| Relaxed + actual-time rates | .6393 | .012490 | 2.408 | 73 / 226 | .0290 |
| Relaxed + repeated original current descriptors | .6659 | .012445 | 2.356 | 73 / 226 | .0299 |
| Relaxed + dense original history | .6227 | .013133 | 2.384 | 82 / 226 | .0293 |
| Relaxed + repeated relaxed branch | .6565 | .012661 | 2.527 | 72 / 226 | .0312 |
| Relaxed + original encoder branch | .6270 | .013194 | 2.520 | 74 / 226 | .0278 |
| Relaxed + original-minus-relaxed physical descriptors | .6647 | .012421 | 2.419 | 71 / 226 | .0274 |

Relaxed versus original reduces integrated Brier by0.001921 (13.3%), with a
paired source-bootstrap95% interval [0.000959,0.002933]. Its AP gain is0.0719.
The modest improvements from repeated current descriptors and quench differences
are **not established by the Brier intervals**, which include zero. In particular,
the quench-difference gain is0.000149, interval[-0.000201,0.000471].

Dense history worsens Brier relative to its repeated-current control by0.000688;
the improvement-oriented interval is[-0.001159,-0.000223]. The second original
encoder branch also loses to its repeated-relaxed capacity control. These are
results for the tested input processing and optimizer, not proof that history
or original observations contain no useful information.

Short-horizon emphasis on the combined model lowers detected-event timing MAE
from2.486 to2.299 ps, with69 versus70 misses, but AP falls from.6176 to.6113.
Its Brier improvement interval includes zero. Removing absolute simulation age
has little effect in that matched test. Neither provides a clear overall winner.

The original/relaxed distinction changes both the encoder checkpoint and input
domain/support. Timing is conditional on detection and uses different detected
subsets; read it alongside misses and realized false alarms. Bootstrap intervals
measure source uncertainty conditional on one seed; exploratory multiple contrasts
and historical test reuse limit confirmatory claims.

## Validation-only choices for the next wave

The quench-difference model has the lowest selection Brier(.016771), followed by
the rates-only model(.016802); cold control is.016993. Quench-difference physical
validation MSE is.7850 versus.7676 for control, within the declared10% tolerance.
This justifies prioritizing the simple quench and rate additions without selecting
on their test AP. New matched controls are included because the next experiment
budget and auxiliary input width change.

Full machine results, all12 rows and pairwise intervals:
[first-wave report](../../output/crystallization_transfer/literature-followup-20260922/RESULTS.md),
`technical/comparison.json` and `tables/comparison.csv` in that directory.
The [next wave](SECOND_WAVE.md) tests optimization and explicit observed ordered-region geometry.
