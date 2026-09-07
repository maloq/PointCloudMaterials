# Liquid short-range-order benchmark — September 5, 2026

Research experiment record. The goal is to test structure within liquids and
its association with subsequent dynamics, without treating PTM classes as truth.

The primary dynamical data comprise 40 Al parent configurations from 20 source
runs at 400/450/500 K, with eight float32 shooting futures per parent. The
pre-existing validation sources become final test sources; velocity seed 35863
is held out from the remaining sources for model selection. All descendants of
a source stay together. Initially noncoherent centers are evaluated separately.
“Order acquisition” is a finite-horizon, persistent bond-coherence assay, not a
thermodynamic committor or an assertion that a particular crystal polymorph exists.

Continuous q4/q6/w4/w6/coarse-grained q6 measurements, alpha-complex persistence
images (H0/H1/H2), and independent nonaffine mobility provide complementary
targets. TDA is an operational geometry descriptor, not ground truth. Its fixed
nearest-neighbor window can introduce boundary effects; these must be measured.
The main tests forecast *future* descriptors across independent sources, rather
than rewarding a descriptor for reconstructing itself at the current time.

Training also includes Al/Mg/Ta continuations and saved static Al environments.
Al/Mg supplemental source snapshots are held out by snapshot; Ta has only one
trajectory and therefore uses separate times and central IDs. These auxiliary
tests do not provide independent-source evidence for Ta. The primary dynamical
test uses float32 data; supplemental Al/Mg trajectories retain float16 storage.

The MACE candidate uses the reference MACE model with two message-passing
layers, learned radial functions, element attributes, correlation order three,
and a central-atom readout. Graphs include the full two-hop receptive field.
Learned candidates receive same-center jitter views with a separate VICReg
projector; PTM, TDA, and future outcomes do not supervise encoder pretraining.

Configuration: [config.json](config.json). Generated data, detached logs,
checkpoints, and the final table are stored physically in
[output/liquid_sro_benchmark_20260905](../../output/liquid_sro_benchmark_20260905/).

Methods informing the protocol:

- [Hiraoka et al., persistent homology of amorphous solids](https://arxiv.org/abs/1501.03611).
- [Adams et al., persistence images](https://arxiv.org/abs/1507.06217).
- [Russo and Tanaka, orientational ordering before crystallization](https://pmc.ncbi.nlm.nih.gov/articles/PMC3395031/).
- [MACE reference architecture](https://arxiv.org/abs/2206.07697).
- [SchNet continuous-filter convolutions](https://arxiv.org/abs/1706.08566).

Run from the repository root with `pointnet`; GUDHI 3.11.0 is installed in that
environment. The final report is [RESULTS.md](../../output/liquid_sro_benchmark_20260905/RESULTS.md),
with [machine-readable results](../../output/liquid_sro_benchmark_20260905/evaluation/comparison.csv).

The corrected accelerated model uses `ir_mul`: the fused cuEquivariance
convolution descriptor requires that layout. The initial `mul_ir` attempt failed
the final GPU rotation audit and is retained separately as invalid. All three
affected MACE seeds were retrained from scratch. An explicit fused-GPU
rotation/cutoff/gradient regression test now supplements the native controls.

Reproduce in a fresh output directory by changing `output` in `config.json`
(preparation and training deliberately refuse to overwrite existing runs):

```bash
conda activate pointnet
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m pytest tests/test_liquid_sro_benchmark.py -q
python experiments/liquid_sro_benchmark_20260905/prepare.py --config experiments/liquid_sro_benchmark_20260905/config.json
python experiments/liquid_sro_benchmark_20260905/features.py --config experiments/liquid_sro_benchmark_20260905/config.json
python experiments/liquid_sro_benchmark_20260905/null_control.py
python experiments/liquid_sro_benchmark_20260905/robustness.py --config experiments/liquid_sro_benchmark_20260905/config.json --stage prepare
python experiments/liquid_sro_benchmark_20260905/temporal.py --config experiments/liquid_sro_benchmark_20260905/config.json --stage prepare
python experiments/liquid_sro_benchmark_20260905/train.py --config experiments/liquid_sro_benchmark_20260905/config.json
python experiments/liquid_sro_benchmark_20260905/postprocess.py
python experiments/liquid_sro_benchmark_20260905/conditioned_probe.py
```

`postprocess.py` waits for detached training to complete, then selects retained
checkpoints on the full validation population, exports untrained-MACE controls,
fits physical probes, and evaluates perturbation and trajectory controls. Its
subprocesses fail loudly on errors. During this run both training and that queue
were detached with `subprocess.Popen(..., start_new_session=True)`; PIDs, logs and
commands are saved under the output directory. The corrected retraining used
`train.py ... --models MACE` after explicitly archiving the invalid MACE outputs;
the unaffected six SchNet/density runs were retained.

`generate_report.py` renders tables, source-bootstrap figures, a provenance
record and checkpoint hashes after all stages finish. It requires the passing
five-test `validation.log` and gates the final MACE rotation results. Save the
pytest output as that file before generating a report in a new output directory.

The initial ordered-batch validation score and the full-population checkpoint
selection are both retained. The 16-PC TDA row and the common covariance-probe
audit are exploratory follow-ups; neither uses test targets for hyperparameter
selection. Main results, all seed scores, negative controls and failed TDA
variants remain visible. This is a benchmark of label-free spatial denoising
representations against future dynamics, not yet training of a temporal
transition model.
