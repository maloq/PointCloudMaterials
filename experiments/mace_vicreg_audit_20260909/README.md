# MACE VICReg audit — 2026-09-09

Question: why does plain80 lose within-material effective rank before TDA starts,
and is its VICReg computation or gradient replay incorrect?

`audit.py` is an experiment-specific numerical/optimization diagnostic. It
imports the actual `VICRegLoss` used by `vicreg_module.py`, the actual MACE
encoder, cached gradient implementation and repository dataset. It does not
introduce another maintained trainer or change existing runs.

The audit checks loss values/gradients against the reference, producer-owned
atom IDs and physical lags, real-MACE cached/direct gradients and clipped AdamW
updates, and the absence of TDA gradients. It measures the initial and selected
plain80 embedding geometry on identical samples. Finally it fits linear and
original-style MLP readouts on frozen initial MACE features, with two seeds and
no TDA supervision, to isolate the cost of directly regularizing MLIP channels.
Readout BatchNorm sees the entire view batch; evaluation uses its running stats.

Configuration: `config.json`. Uniformly sample 16,384 training anchors and use
all 9,472 validation anchors. Frozen readouts use 600 steps, batch 1,536,
AdamW at 1e-3 with cosine decay. These are diagnostic budgets, not a matched
comparison with the completed 12-epoch run. Any result involving the saved
epoch-6 checkpoint includes its one TDA-active epoch; pre-TDA statements use
the recorded epoch-1–5 metrics separately.

```bash
PYTHONPATH=. conda run -n pointnet python experiments/mace_vicreg_audit_20260909/audit.py \
  --config experiments/mace_vicreg_audit_20260909/config.json
```

Outputs and findings: `output/mace_vicreg_audit_20260909/`, including
`math_audit.json`, `data_audit.json`, `gradient_audit.json`, `features_audit.json`,
`readout_results.json`, and the final `REPORT.md`. Generated sample indices,
features, diagnostic weights and logs are output artifacts. This directory
contains versioned experiment records and reproduction code.

The [official VICReg implementation](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py)
also uses a learned MLP, but sums the two covariance penalties where this
repository averages them. It is a secondary comparison; the user's repository
implementation is the primary numerical reference.

Additional controls use the same command:

```bash
PYTHONPATH=. conda run -n pointnet python experiments/mace_vicreg_audit_20260909/audit.py --config experiments/mace_vicreg_audit_20260909/conditional.json --stage readouts
PYTHONPATH=. conda run -n pointnet python experiments/mace_vicreg_audit_20260909/audit.py --config experiments/mace_vicreg_audit_20260909/joint.json --stage joint
PYTHONPATH=. conda run -n pointnet python experiments/mace_vicreg_audit_20260909/audit.py --config experiments/mace_vicreg_audit_20260909/config.json --stage gradients
```

The conditional control changes only the reference variance/covariance moments
to within-material moments, weighted by actual sample counts. It adds no
sampling quotas, equal-element loss averaging, TDA target, or balancing algorithm.
Pairwise attraction stays unchanged. The joint controls update the MACE backbone
and optional 256D projector for 300 steps with batch 512: 153,600 anchor exposures
per arm, the same initialization, sampled training subset and batch order.
They use a cosine schedule without warmup and are short diagnostics, not replicas
of the original 12-epoch schedule. Raw and projected coordinates are both
reported. The projector sees full view batches before encoder gradient replay,
so BatchNorm is not recomputed or updated during microbatch replay.

The scalar-shrinkage control chooses a single multiplier on training features
and evaluates it on held-out features. It preserves rank and all normalized
neighbor-distance ratios exactly; a large loss decrease here is not structure
learning. The precision audit compares actual full-MACE parameter gradients
under compensated BF16 and FP32 as well as cached versus direct backpropagation.

The completed [report](../../output/mace_vicreg_audit_20260909/REPORT.md) includes
the source-radius/species factorial intervention (`--stage normalization` with
`normalization.json`). Its findings motivated restoring the original normalized,
geometry-only pipeline. The exploratory `normalized_joint.json` attempt stopped
at an overly strict round-trip numerical assertion before training; no optimizer
result is claimed for it. The production replacement and successful direct
Lightning preflight are recorded in
[mace_original_vicreg_20260909](../mace_original_vicreg_20260909/README.md).
