# MACE pilot diagnosis — September 5, 2026

Research question: is the smooth pilot's weak static-Al result caused by the
MACE product block, representation training, numerical/domain shift, or the
transferred structural readout?

This is an experiment record, not a maintained command. It inspects the completed
[smooth pilot](../smooth_temporal_encoder_20260905/README.md) without replacing its
checkpoints or metrics. Configuration: [config.json](config.json). Generated
diagnostics and findings live in
[output/mace_diagnosis_20260905](../../output/mace_diagnosis_20260905/).

Use `pointnet` from the repository root:

```bash
python experiments/mace_diagnosis_20260905/readouts.py --config experiments/mace_diagnosis_20260905/config.json
python experiments/mace_diagnosis_20260905/features.py --config experiments/mace_diagnosis_20260905/config.json
python experiments/mace_diagnosis_20260905/views.py --config experiments/mace_diagnosis_20260905/config.json
python experiments/mace_diagnosis_20260905/objectives.py --config experiments/mace_diagnosis_20260905/config.json
python experiments/mace_diagnosis_20260905/report.py --config experiments/mace_diagnosis_20260905/config.json
```

The readout control freezes all encoders. It fits the existing class-balanced
ridge probe on static snapshots 166/174 ps, selects its coefficient on 170 ps,
and evaluates 175/177/240 ps. It measures information accessible after readout
adaptation; these snapshots share a simulation and are not independent-source
generalization. Another control restricts the original MD probe fit to Al.
All reported per-class arrays use `[Other, FCC, HCP, BCC]`; four-class macro F1
and the common three-class mean are both retained because BCC is rare in Al.

`features.py` fits MD-only probes at each location in the frozen trained model,
checks real trained-parameter gradients, and samples 2,048 static centers per
snapshot for matched perturbation and adaptation controls. Perturbations retain
the original target labels; they are sensitivity assays, not physical MD.

`objectives.py` performs nine matched continuation runs (three conditions × three
seeds) with fresh optimizers and identical starting checkpoints per seed. A new
output directory is required to repeat training without replacing artifacts.
Preparation must populate that directory first using the commands above.

Completed findings: [report and plots](../../output/mace_diagnosis_20260905/RESULTS.md).
The tested model is a local product-block pilot, not full MACE. Its learned head
reduces linear structural accessibility. Removing the spatial pair loss improves
static readout in all three adaptation seeds; adding a projector in this short
adaptation does not reliably help. Static readout adaptation recovers part of the
transfer deficit. A faithful MACE baseline remains untested.
