"""Assemble the completed MACE diagnosis from saved, matched measurements."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=ROOT/cfg['output']
    f=json.loads((out/'features.json').read_text())
    r=json.loads((out/'readouts.json').read_text())['models']
    runs=json.loads((out/'objectives.json').read_text())
    views=json.loads((out/'spatial_view_assay.json').read_text())
    assert len(runs)==9 and json.loads((out/'objectives_status.json').read_text())['state']=='complete'
    conditions=cfg['objective_audit']['conditions']
    names={'continued_control':'Continue original objective','no_spatial':'Remove spatial pair loss','separate_projector':'Add separate projector'}
    groups={c:[v for v in runs if v['condition']==c] for c in conditions}
    layer_names=['power','prehead','hidden1','hidden2','output']
    labels=['Density powers','Powers + products','Head layer 1','Head layer 2','Exported output']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    axes[0].bar(np.arange(5),[f['layers'][k]['static']['macro_f1'] for k in layer_names],color=['C0','C0','C1','C1','C3'])
    axes[0].set(xticks=np.arange(5),xticklabels=labels,ylabel='Transferred PTM macro F1',title='Linear accessibility through the head')
    axes[0].tick_params(axis='x',labelrotation=35)
    x=np.arange(3)
    for delta,name,label in [(-.18,'mace_product','Pilot MACE products'),(.18,'geoframe_vicreg','GeoFrame')]:
        axes[1].bar(x+delta,r[name]['original_full_static']['f1'][:3],width=.36,label=label)
    axes[1].set(xticks=x,xticklabels=['Other','FCC','HCP'],ylabel='Per-class F1',title='Errors on all 772,953 static centers')
    axes[1].legend(fontsize=8)
    for seed in cfg['objective_audit']['seeds']:
        y=[next(v for v in groups[c] if v['seed']==seed)['static']['common_012_macro_f1'] for c in conditions]
        axes[2].plot(x,y,'o-',label=f'Seed {seed}')
    axes[2].set(xticks=x,xticklabels=['Continue','No spatial pairs','Add projector'],ylabel='Mean F1: Other / FCC / HCP',title='Matched continuation controls')
    axes[2].tick_params(axis='x',labelrotation=25)
    axes[2].legend(fontsize=8)
    fig.suptitle('MACE pilot diagnosis: training and transfer, not a benchmark of full MACE')
    fig.tight_layout()
    fig.savefig(out/'diagnosis.png',dpi=180)
    plt.close(fig)
    lines=['# Why the MACE pilot transfers poorly to static Al','',
      'September 5, 2026. Completed code/gradient audit, frozen-feature/readout controls, input sensitivity assays, and nine matched objective-adaptation runs. The original pilot remains intact.','',
      '**Conclusion:** the evidence points to our simplified model and training/readout protocol. The data do not establish that the full MACE architecture is unsuitable for this task. My earlier shorthand “MACE” was too broad.','',
      '![Diagnosis](diagnosis.png)','',
      '## 1. What was actually evaluated','',
      'The pilot uses one central `EquivariantProductBasisBlock`, fixed Gaussian radial densities, 32 scalar product outputs, and a 260D power-spectrum branch followed by a 292→256→256→128 MLP. There is no atom-to-atom message passing, no learned radial MLP, no chemical-element conditioning, and no pretrained MACE weights. Relative distances are divided by material-specific radii. It was trained on VICReg view agreement rather than energies or forces.','',
      'The original [MACE paper](https://arxiv.org/html/2206.07697v2) combines learned radial functions, chemical node features, higher-order products, and two message-passing layers in its benchmarks. The [official foundation-model documentation](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html) describes pretrained materials models with extensive atomistic supervision. Their reported success cannot be transferred automatically to this small randomly initialized product-block pilot.','',
      'The repository already contains a fuller [MACE backbone adapter](../../src/models/encoders/mace_encoder.py). It was not the model used in this pilot. Its finite-cloud pooling and neighbor construction would still need matching to the intended central-atom, periodic-domain protocol.','',
      '## 2. The failure is concentrated in Other versus HCP','',
      'All rows in this table use the same 772,953 static-Al centers and the original MD-trained readouts. PTM labels are an independent geometric assay, not perfect physical ground truth.','',
      '| Representation | Other F1 | FCC F1 | HCP F1 | BCC F1 | Four-class macro F1 |','|---|---:|---:|---:|---:|---:|']
    for name,label in [('density_pca','Density power PCA'),('power_mlp','Power MLP'),('mace_product','MACE-product pilot'),('geoframe_vicreg','GeoFrame')]:
        row=r[name]['original_full_static']
        lines.append(f'| {label} | '+' | '.join(f'{v:.4f}' for v in row['f1'])+f" | {row['macro_f1']:.4f} |")
    row=r['mace_product']['original_full_static']
    lines += ['',f"The pilot predicts {row['predicted_counts'][2]:,} HCP centers, versus {row['counts'][2]:,} PTM-HCP centers. HCP precision is {row['precision'][2]:.3f}. FCC is already recognized well. The power MLP shows essentially the same error, so this is not specific to the MACE product operation.",'',
      'BCC occupies only 782 of these centers (0.10%) but contributes one quarter of four-class macro F1. The diagnosis also retains the mean of Other/FCC/HCP F1 to prevent a few BCC predictions from dominating small-sample controls. Both metrics preserve the HCP/Other gap.','',
      'On the ordinary MD test observations, the same MACE-product model scores 0.8888 overall macro F1 versus 0.9001 for GeoFrame. The severe deficit is therefore a transfer result, not a universal failure to recognize atomic structure.','',
      '## 3. The learned head reduces linear structural accessibility','',
      'Each row fits the same class-balanced linear probe on the same MD training labels, selects its ridge coefficient on MD validation labels, and evaluates the same 12,288 static centers (2,048 per snapshot). No static labels fit these probes.','',
      '| Feature location | Dimension | MD test macro F1 | Static sample macro F1 |','|---|---:|---:|---:|']
    for name,label,dim in zip(layer_names,labels,[260,292,256,256,128]):
        row=f['layers'][name]
        lines.append(f"| {label} | {dim} | {row['md_test']['macro_f1']:.4f} | {row['static']['macro_f1']:.4f} |")
    lines += ['',
      'The input geometry already supports a stronger transferable linear readout. The deterioration occurs through the learned head. This establishes weaker linear accessibility/transfer, not a proof that every nonlinear decoder would fail or that information is irreversibly erased. The head was trained directly as the VICReg loss space, with no separate projector.','',
      'The [VICReg paper](https://arxiv.org/html/2105.04906v3) distinguishes the downstream encoder representation from an expander where invariance and decorrelation are enforced. This makes the pilot choice a plausible concern, but the continuation test below does not show that adding a projector after training repairs it.','',
      'The product block is not dead: its parameters receive finite, nonzero gradients and changed by about 7.1% in L2 relative to initialization. The radial channel mixer changed by about 7.9%. The model has 175,616 parameters, of which 173,696 (98.9%) belong to the final MLP. At the first head layer, the product contribution has summed variance 6.82 versus 759.66 for the power contribution. The product also supplies a substantial mean offset; these variances are not independent explained-variance fractions. Simply zeroing the branch is therefore not a valid standalone capacity test.','',
      '## 4. A controlled training change helps, but does not solve transfer','',
      'For each original seed, all three conditions start from its identical best checkpoint. Each receives a fresh AdamW optimizer, the same minibatch ordering, 30 additional epochs, batch 4096, and cosine learning rate starting at 0.0003. The separate-projector condition adds a 128→256→256→128 MLP and applies VICReg there. Checkpoints are selected by the validation objective within each condition; raw loss values across different objectives are not comparable. PTM labels never enter encoder adaptation.','',
      '| Adaptation | Static macro F1, mean ± seed SD | Other/FCC/HCP mean F1 | MD macro F1 |','|---|---:|---:|---:|']
    for c in conditions:
        values=np.array([v['static']['macro_f1'] for v in groups[c]])
        common=np.mean([v['static']['common_012_macro_f1'] for v in groups[c]])
        md=np.mean([v['md_test']['macro_f1'] for v in groups[c]])
        lines.append(f'| {names[c]} | {values.mean():.4f} ± {values.std(ddof=1):.4f} | {common:.4f} | {md:.4f} |')
    lines += ['',
      'Removing the spatial pair loss improves both static metrics in all three paired seeds. Merely training longer on the original objective does not recover the gap. The projector adaptation provides no reliable benefit. This is a short adaptation test, not a complete from-scratch projector or hyperparameter study. Removing the spatial pair also removes its variance/covariance contributions, so the experiment identifies the composite spatial-pair objective rather than isolating its attraction term algebraically.','',
      'The actual spatial positives sometimes cross PTM classes:','',
      '| Material | Disagreeing spatial pairs | Sampled pairs |','|---|---:|---:|']
    for m,v in views.items():
        lines.append(f"| {m} | {100*v['different_ptm_fraction']:.2f}% | {v['count']} |")
    lines += ['',
      'For Al, 22 of the 29 sampled PTM-HCP anchors have a differently labeled spatial partner. The HCP subset is small, but it directly illustrates why unconditional neighbor agreement can work against minority local structures at boundaries. PTM disagreement alone does not prove that every such pair is an invalid representation target.','',
      '## 5. Readout transfer explains part of the gap','',
      'Freeze the representation and change only the probe training domain: fit on static snapshots 166/174 ps, select the ridge coefficient on 170 ps, and evaluate 175/177/240 ps. On these identical held-out rows, the MACE-product score improves from **0.5350 to 0.6022**. The original GeoFrame transfer score on those rows is 0.6456. Thus probe adaptation recovers about 61% of this particular gap, but does not close it.','',
      'Fitting only on Al MD data makes transfer worse for all candidates; simply separating materials is not a demonstrated fix. These readout controls use labels and do not constitute unsupervised encoder improvements. The static snapshots share a simulation, so their split is not independent-source validation.','',
      'Global float16 coordinate storage is also a domain difference. Quantizing the static positions before forming offsets introduces approximately 0.039 Å RMS offset error. On the fixed static sample this changes the MACE score only from 0.5210 to 0.5276, so it does not explain the large deficit by itself. Bounded 0.1 Å offset noise raises the common Other/FCC/HCP mean F1 from 0.6657 to about 0.7714, consistent with sensitivity to fluctuation statistics. This is an input-sensitivity assay evaluated against original labels, not a physical heating experiment or a recommended inference fix.','',
      'The MD metadata specifies Al NPT at 650 K. It points to `initial_configurations/*.pos`, whereas static analysis uses `.npy` files in `inherent_configurations_off`. The original `.pos` files are absent on this node. Whether this specifically represents thermal versus energy-minimized data has not been independently established from the local provenance; it should not be asserted from directory names alone.','',
      '## Decision','',
      'A fair next MACE experiment should use the reference message-passing architecture or frozen pretrained materials descriptors, with correct chemical species, physical-distance units, periodic neighborhoods, and sufficient receptive-field coverage. The [official descriptor interface](https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html) provides a direct starting point for a pretrained baseline. No such checkpoint was benchmarked in this diagnosis.','',
      'For representation training, include static and MD environments, measure structural readouts on both held-out domains, and keep same-center augmentations as the main positive pairs. Treat spatial-neighbor agreement as an ablation or a carefully weighted auxiliary term. Compare pre-head and exported features explicitly. A separate projector is worth a from-scratch control, but the present results do not establish it as the fix.','',
      'These findings reduce the case for inventing a replacement for MACE before a faithful baseline has been tested. They do not eliminate the independently demonstrated need to remove GeoFrame’s discontinuous canonicalization.','',
      '## Artifacts and checks','',
      '[Reproduction record](../../experiments/mace_diagnosis_20260905/README.md), [full readout controls](readouts.json), [layer and sensitivity results](features.json), [nine objective runs](objectives.json), and [spatial-pair assay](spatial_view_assay.json). All diagnostics and adapted checkpoints are physically in this repository. The original pilot checkpoints were not replaced.','',
      'Recomputed sample features agree with the previously saved full-static features to at most 1.15e-5 absolute difference. The original rotation/cutoff/gradient tests remain applicable to the unchanged spatial implementation. New diagnostics check the actual trained branch gradients and paired experimental inputs.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    sources=list((ROOT/'experiments/mace_diagnosis_20260905').glob('*.py'))+[args.config.resolve()]
    write_json(out/'provenance.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        pilot_provenance=str(ROOT/cfg['pilot']/'provenance.json')))
    write_json(out/'status.json',dict(state='complete',objective_runs=9,report=str(out/'RESULTS.md'),finished_at=datetime.now(timezone.utc).isoformat()))
    print(out/'RESULTS.md')


if __name__=='__main__':
    main()
