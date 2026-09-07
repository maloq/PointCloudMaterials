"""Render the completed liquid benchmark tables, confidence intervals and figures."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments.smooth_temporal_encoder_20260905.prepare import write_json

LABELS={'CoarseBOO_temperature':'Current bond order + temperature','MACE':'Reference MACE + VICReg',
    'MACE_untrained':'Reference MACE, untrained','SchNet':'SchNet-style + VICReg',
    'DensityMLP':'Smooth-density MLP + VICReg','DensityPCA':'Smooth-density PCA',
    'SOAP':'SOAP + PCA','TDA':'TDA, 128 PCs','TDA_16':'TDA, 16 PCs (exploratory)',
    'GeoFrame_pretrained':'GeoFrame v2, earlier checkpoint','ShuffledSOAP':'Shuffled SOAP control'}
ORDER=['CoarseBOO_temperature','GeoFrame_pretrained','MACE_untrained','MACE','SchNet','DensityMLP','DensityPCA','SOAP','TDA_16','TDA','ShuffledSOAP']


def group(frame):
    frame=frame.copy();frame['group']=frame['model'].str.replace(r'_seed\d+$','',regex=True)
    return frame


def markdown(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def pairwise(out):
    saved=np.load(out/'evaluation/split_and_targets.npz');sources=saved['source'][saved['split']=='test'];groups=np.unique(sources)
    draws=np.random.default_rng(20260908).integers(0,len(groups),size=(1000,len(groups)))
    def errors(name,family):
        names=[f'{name}_seed{s}' for s in (123,456,789)] if name in ('MACE','MACE_untrained','DensityMLP','SchNet') else [name]
        return np.mean([np.load(out/'evaluation'/f'{n}_test_errors.npz')[family] for n in names],axis=0)
    rows=[]
    for a,b in [('MACE','MACE_untrained'),('DensityMLP','MACE'),('DensityMLP','DensityPCA'),('DensityMLP','SOAP')]:
        for family in ('topology','order','mobility'):
            delta=errors(b,family)-errors(a,family);baseline=errors('CoarseBOO_temperature',family)
            sums=np.array([delta[sources==s].sum() for s in groups]);base=np.array([baseline[sources==s].sum() for s in groups])
            ci=np.quantile(100*sums[draws].sum(1)/base[draws].sum(1),[.025,.975])
            rows.append(dict(a=a,b=b,target=family,advantage_percentage_points=float(100*delta.mean()/baseline.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
    pd.DataFrame(rows).to_csv(out/'evaluation/pairwise_advantages.csv',index=False)


def main():
    cfg=json.loads((Path(__file__).parent/'config.json').read_text());out=ROOT/cfg['output']
    assert json.loads((out/'postprocess_status.json').read_text())['state']=='complete'
    assert json.loads((out/'conditioned_probe/protocol.json').read_text())['state']=='complete'
    validation=(out/'validation.log').read_text();assert '5 passed' in validation
    table=pd.read_csv(out/'evaluation/comparison.csv').set_index('model')
    uncertainty=json.loads((out/'evaluation/uncertainty.json').read_text())
    perturb=group(pd.read_csv(out/'robustness/perturbation_results.csv'))
    check=perturb[(perturb['group'].isin(['MACE','MACE_untrained']))&(perturb['case']=='rotation')]
    assert (check['p95_relative_change']<.001).all(),'A fused MACE rotation control failed; do not publish the comparison'
    rows=[]
    for name in ORDER:
        cells=[LABELS[name]]
        for family in ('topology','order','mobility','neighbors'):
            value=table.loc[name,family+'_skill_pct'];sd=table.loc[name,family+'_seed_sd_pct']
            cells.append(f'{value:+.2f} ± {sd:.2f}' if table.loc[name,'seeds']==3 else f'{value:+.2f}')
        rows.append(cells)
    primary=markdown(['Frozen representation','Future topology','Future bond order','Future mobility','Matched-neighbor future agreement'],rows)
    confidence=markdown(['Representation','Topology: mean [95% CI]','Bond order: mean [95% CI]','Mobility: mean [95% CI]'],[
        [LABELS[n]]+[f"{uncertainty[n][f]['skill_pct']:+.2f} [{uncertainty[n][f]['source_ci95_pct'][0]:+.2f}, {uncertainty[n][f]['source_ci95_pct'][1]:+.2f}]" for f in ('topology','order','mobility')] for n in ORDER[1:]])
    aux=group(pd.read_csv(out/'evaluation/auxiliary_fidelity.csv'))
    fidelity=aux.groupby(['group','material'])['order_r2'].mean()
    aux_table=markdown(['Representation','Al order R²','Mg order R²','Ta order R²'],[
        [LABELS[n]]+[f'{fidelity.loc[n,m]:.3f}' for m in ('Al','Mg','Ta')] for n in ORDER if n not in ('CoarseBOO_temperature','ShuffledSOAP')])
    temp=group(pd.read_csv(out/'temporal/continuity.csv')).groupby('group')['median_relative_step'].mean()
    perturb_mean=perturb.groupby(['group','material','case'])['median_relative_change'].mean()
    rank=group(pd.read_csv(out/'evaluation/individual_results.csv')).groupby('group')['effective_rank'].mean()
    smooth_table=markdown(['Representation','Al effective rank','Rotation Δ','0.0001 Å jitter Δ','0.02 Å jitter Δ','0.3 ps trajectory Δ'],[
        [LABELS[n],f'{rank[n]:.1f}',*[f'{perturb_mean.loc[n,"Al",c]:.3g}' for c in ('rotation','noise_0.0001A','noise_0.02A')],f'{temp[n]:.3f}'] for n in ORDER if n not in ('CoarseBOO_temperature','ShuffledSOAP')])
    conditioned=pd.read_csv(out/'conditioned_probe/comparison.csv').set_index('model')
    conditioned_table=markdown(['Representation','Future topology','Future bond order','Future mobility'],[
        [LABELS[n]]+[f'{conditioned.loc[n,f+"_skill_pct"]:+.2f}' for f in ('topology','order','mobility')] for n in ORDER[1:]])
    reliability=json.loads((out/'evaluation/target_reliability.json').read_text())
    reliability_table=markdown(['Target','Odd/even four-shot correlation','Estimated eight-shot-mean noise MSE'],[
        [f,f"{r['half_ensemble_correlation']:.3f}",f"{r['estimated_full_mean_noise_mse']:.3f}"] for f,r in reliability.items()])
    event_table=markdown(['Representation','48 ps Brier skill, threshold .65','.70','.75'],[
        [LABELS[n]]+[f'{table.loc[n,f"event_{t:.2f}_skill_pct"]:+.2f}%' for t in (.65,.7,.75)] for n in ORDER])
    training=json.loads((out/'training_summary.json').read_text());selection=json.loads((out/'checkpoint_selection.json').read_text())
    model_table=markdown(['Model','Encoder parameters','Mean run time','Selected epochs (of 60)'],[
        [LABELS[n],next(r['encoder_parameters'] for r in training if r['name']==n),f"{np.mean([r['seconds'] for r in training if r['name']==n]):.1f} s",', '.join(str(r['selected']['epoch']+1) for r in selection if r['model'].startswith(n+'_seed'))] for n in ('MACE','SchNet','DensityMLP')])
    (out/'COMPARISON.md').write_text('Scores are percentage reduction in error versus current bond order + temperature; higher is better. ± is training-seed SD, not a confidence interval.\n\n'+primary+'\n')
    (out/'SOURCE_CONFIDENCE.md').write_text('Paired bootstrap of six independent test sources (1000 resamples). Three training seeds are not independent simulation sources.\n\n'+confidence+'\n')
    pairwise(out)
    figure_names=['GeoFrame_pretrained','MACE_untrained','MACE','SchNet','DensityMLP','DensityPCA','SOAP','TDA_16','ShuffledSOAP']
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True,layout='constrained')
    colors=['#88929f','#bca878','#d36c38','#b782a0','#21776e','#57a994','#3989ad','#8665a8','#aaaaaa']
    for ax,family,title in zip(axes,('topology','order','mobility'),('Future topology','Future bond order','Future mobility')):
        for i,(name,color) in enumerate(zip(figure_names,colors)):
            r=uncertainty[name][family];lo,hi=r['source_ci95_pct'];v=r['skill_pct']
            ax.plot([lo,hi],[i,i],color=color,lw=2);ax.scatter([v],[i],color=color,s=35,zorder=3)
        ax.axvline(0,color='#999999',lw=1);ax.set_title(title);ax.set_xlabel('Error reduction over coarse baseline (%)');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(len(figure_names)),[LABELS[n] for n in figure_names]);axes[0].invert_yaxis()
    fig.suptitle('Liquid Al: held-out sources, 12/24/48 ps forecasts\n95% intervals resample sources; unstable TDA-128 is retained in the full table',fontsize=12)
    fig.savefig(out/'forecast_comparison.png',dpi=180);fig.savefig(out/'forecast_comparison.pdf');plt.close(fig)
    boundary=json.loads((out/'robustness/tda_boundary.json').read_text())
    text=f'''# Liquid short-range-order benchmark — completed {datetime.now(timezone.utc).isoformat()}

The benchmark evaluates continuous local structure and its association with future dynamics. No PTM class, including HCP, is used as ground truth, training label, or primary score. All numerical results below use the corrected invariant MACE backend.

## Main comparison

Primary population: **initially noncoherent Al environments**, fewer than seven coherent bonds at the operational q6-dot threshold 0.70. There are **5,600 train / 1,529 validation / 3,054 test centers**, from **11 / 3 / 6 disjoint source runs**. Each parent has eight independent shooting futures. All descendants of a source stay together; atoms are not treated as independent replicates for confidence intervals.

Every frozen representation is appended to the same eight current bond-order/density measurements and temperature indicators. Ridge probes are fitted on training sources; regularization is selected using validation sources. Future topology, six continuous order measurements, and two mobility measurements are averaged over eight shots and evaluated at 12, 24 and 48 ps. Future topology uses a 16-component target PCA fitted on training targets only (99.84% of raw future-image variance). Components and horizons have equal weight after training-target standardization.

The first three columns report **100 × (1 − model MSE / coarse-baseline MSE)**. Positive values mean additional predictive information. The last column compares future similarity of ten neighbors selected by an embedding from 64 temperature- and coarse-order-matched training candidates, against ten nearest coarse-order neighbors. Training seeds are averaged at the error level, without prediction ensembling. ± reports seed SD.

{primary}

![Source-held-out forecasting comparison](forecast_comparison.png)

{confidence}

The smooth-density MLP has the best mean future-topology and bond-order scores in this comparison; the untrained density PCA has the best mean mobility score. Its matched-neighbor gain is also positive with a source confidence interval above zero, supporting information beyond temperature and the coarse order measurements. This is evidence for useful liquid-structure information, not identification of a new phase.

Correct MACE improves mobility forecasting, but its three forecast scores remain close to untrained MACE. Within liquid Al its exported 128-channel representation has effective rank {rank['MACE']:.1f}, compared with {rank['DensityMLP']:.1f} for the density MLP. Covariance-conditioned probes do not remove the gap. The evidence points to limited useful representation learning under this jitter-only VICReg objective. It does not show that MACE is intrinsically unsuitable: these runs have neither force/energy supervision nor a temporal prediction loss, and the 4 Å, two-layer configuration is only one MACE setting.

The shuffled-SOAP control preserves split, material and temperature distributions while destroying the assignment of local structures to centers. It checks that adding dimensions alone does not explain the predictive gains. Pairwise source-bootstrap differences are in [evaluation/pairwise_advantages.csv](evaluation/pairwise_advantages.csv).

## What was actually trained

Nine valid runs: three seeds each of reference MACE, SchNet-style and the smooth-density MLP, each for 60 epochs on **9,216 Al/Mg/Ta training environments** (7,680 Al / 1,024 Mg / 512 Ta; the training population is Al-heavy). Input views are two independently jittered copies of the same center, with 0.02 Å Cartesian Gaussian noise clipped at ±0.06 Å. This is label-free spatial denoising pretraining; future trajectories are evaluation targets, and **this run does not yet train a temporal transition model**. A separate identical 128→512→512→256 projector receives VICReg (25/25/1). AdamW, learning rate 0.001, five-epoch warm-up, cosine decay, 512 paired samples per batch, full float32.

MACE is **mace.modules.models.MACE from mace-torch 0.3.16**, with two learned message-passing layers, 64 channels, ell≤2, correlation three, eight Bessel radial functions, learned radial MLPs, and Al/Mg/Ta element attributes. Exported 128D features concatenate central scalars from both interactions. Untrained energy readouts are frozen. Complete 193-atom patches cover the full two-hop 4 Å receptive field, including the jitter halo. The explicit cutoff is applied after the radial MLP. Acceleration uses cuEquivariance **ir_mul** layout. The comparison includes three untrained initializations through the same physical probes; this is not a pretrained energy/force foundation model.

SchNet-style is this repository's two-layer continuous-filter implementation, with the same graph support and central 128D export, rather than a downloaded pretrained SchNet. Smooth density uses an 8 Å tapered radial/spherical expansion and a learned 128D MLP. SOAP is DScribe, radial cutoff 6.5 Å, nmax=8, lmax=6, Gaussian width 0.3 Å. It uses a shared geometry species for pure-metal patches; known material identity is handled separately. SOAP, density PCA and the main TDA baseline have 128 coordinates. GeoFrame is the earlier frozen checkpoint and has a different pretraining budget/data history.

{model_table}

Checkpoint selection compares the two retained candidates (legacy best and epoch-60 final) using VICReg over the entire held-out validation population. This avoids computing the variance objective on ordered single-source batches. No physical target or test label enters encoder checkpoint selection. Both candidates and the original selection are retained.

## Backend error caught before accepting MACE results

The first accelerated attempt combined fused convolution with **mul_ir**. The installed fused tensor-product descriptor uses **ir_mul**. Shapes matched, but angular components were scrambled; trained scalar features failed the GPU rotation audit. Native e3nn tests alone did not catch this. The affected MACE models were archived under [invalid_mul_ir_backend](invalid_mul_ir_backend/INVALID.md) and retrained from scratch after correction. No score from those MACE runs appears as a valid result here.

Five scientific tests pass, including an explicit fused-GPU rotation/cutoff/gradient test, native rotation and radial-gradient checks, complete-halo equality against a larger graph, analytic FCC invariants, affine D²=0, and alpha-image rotation/permutation invariance. The corrected preflight rotation relative RMS error is 9.03e-7. Actual trained-model rotation results are reported below and gated before this report is generated.

## Structural fidelity across Al, Mg and Ta

Auxiliary linear probes reconstruct **continuous** q4/q6/w4/w6/qbar6/coherence on supplemental held-out snapshots or times. These include all centers, so high Al scores can still be easier than the primary liquid-only dynamical task. TDA self-reconstruction is excluded from the separately saved topology-fidelity results.

{aux_table}

Al/Mg supplemental data are split by source snapshot, with static Al copies kept with the corresponding continuation. Ta has one trajectory with separate times and center IDs, so its score is **not independent-source generalization**. Supplemental Al/Mg trajectories retain float16 coordinates; the primary shooting test uses float32. Static-source minimization history is not assumed or used as a label.

The supplemental test sets contain only 7 initially noncoherent Al and 10 Mg centers under the same operational cutoff, compared with 408 Ta centers. Their all-center fidelity table therefore cannot establish liquid-only performance for Mg or Ta; the substantive liquid dynamical benchmark here is Al. Additional independent liquid Mg/Ta data are needed.

## Perturbation and physical-time controls

Δ is L2 embedding change divided by the square root of summed training feature variances within the same material. Values here are medians (seed means for learned models). Effective rank is measured within the initially noncoherent Al test population. Rank and predictive quality must accompany smoothness, because collapsing representations also change little.

{smooth_table}

The trajectory assay follows 192 centers from six held-out float32 Al sources over eight frames, spaced 0.3 ps apart. These steps include real thermal motion and neighbor exchange; lower trajectory Δ alone is not an encoder-quality ranking. The small-jitter refinement is the more direct continuity control.

TDA uses GUDHI alpha complexes on the center plus 64 nearest neighbors: finite H0/H1/H2 bars, radius rather than squared-radius filtration values, and 144 image coordinates. At 0.02 Å jitter, **{100*boundary['noise_0.02A']['fraction_windows_changed']:.1f}%** of sampled windows changed membership. Mean squared PI change was **{boundary['noise_0.02A']['fixed_window_mean_squared_change']:.6g}** with membership fixed and **{boundary['noise_0.02A']['reranked_window_mean_squared_change']:.6g}** when neighbors were reselected. No membership changed at 0.0001 Å in this sample; that does not establish global continuity. The fixed-neighbor boundary and hard finite-death truncation remain descriptor limitations.

The 128-PC TDA probe produced a few extreme errors. The additional 16-PC row was introduced **after that baseline audit**, using the existing training-only PCA, and is explicitly exploratory. The original result remains visible. TDA is an additional geometric measurement, not a replacement ground-truth crystal label.

## Common probe-conditioning sensitivity

The initial comparison mixes PCA descriptors and correlated raw learned channels. This exploratory audit applies the same training-only PCA/covariance conditioning to every representation. Each target family/horizon selects an eigenvalue floor (0.01, 0.0001 or 0.000001 times the largest eigenvalue) and ridge penalty on validation sources only. Encoder weights, train/test sources and target scaling remain fixed. This tests sensitivity to the geometry of the linear readout; it does not retrain the encoders or replace the original table.

{conditioned_table}

## Rare order acquisition and finite-shot noise

Order acquisition means ≥7 coherent bonds throughout the last three saved frames at 48 ps, a 0.6 ps persistence interval. It is an operational assay, **not a committor, verified crystal polymorph, or proof of nucleation**. At threshold 0.70, only **1.007%** of test center/shot outcomes satisfy it. The three thresholds produce prevalences 1.220%, 1.007% and 0.872%. Brier scores below evaluate shot outcomes, including within-eight-shot variance; they are not classification accuracy.

{event_table}

{reliability_table}

The noise estimate is one quarter of the squared difference between odd/even four-shot means, in the standardized target coordinates. Eight futures give a noisy estimate of each configuration's topology propensity. Confidence intervals resample six sources; they do not remove selection bias from choosing pre-event parents. Three seeds increase confidence in optimization repeatability, not the number of physical source runs.

## Scope and next evidence needed

This benchmark tests information within initially noncoherent local environments and its association with subsequent dynamics. It does not demonstrate a newly discovered metastable phase. The source parents were selected before events, so generalization to arbitrary equilibrium liquid samples remains untested. Matched-temperature/coarse-order neighbor tests help isolate subtler structure, but observational forecasting alone cannot establish a causal precursor.

Independent Mg/Ta shooting ensembles, more independent Al source configurations, and more than eight futures per parent would strengthen the test. A temporal/predictive encoder should be evaluated against these same held-out future assays and the existing continuity controls. TDA can supply complementary measurements or auxiliary targets; it should not be made the sole truth criterion.

## Artifacts and reproduction

- [Exact protocol](evaluation/protocol.json), [configuration](config.json), [source manifest](parent_manifest.json), [data summary](data_summary.json).
- [Primary CSV](evaluation/comparison.csv), [individual seeds](evaluation/individual_results.csv), [source uncertainty](evaluation/uncertainty.json), [conditioned probes](conditioned_probe/comparison.csv).
- [Auxiliary fidelity](evaluation/auxiliary_fidelity.csv), [perturbations](robustness/perturbation_results.csv), [time continuity](temporal/continuity.csv), [event sensitivity](evaluation/event_sensitivity.csv).
- [Selected checkpoints](checkpoint_selection.json), [training records](training_summary.json), [validation log](validation.log), [environment and hashes](provenance.json).
- Experiment code and commands: [README](../../experiments/liquid_sro_benchmark_20260905/README.md).

Preparation saved its then-current configuration (30 epochs, batch 128). Before training, the unchanged data were used with the measured 60-epoch, batch-512 training configuration. Checkpoints and training logs record the actual training settings. The first 12/24 ps-only data preflight is separately retained under `output/liquid_sro_benchmark_20260905_12_24ps_preflight`; the main benchmark uses the regenerated 12/24/48 ps data.

Literature: [Hiraoka et al., persistent homology of amorphous solids](https://arxiv.org/abs/1501.03611), [Adams et al., persistence images](https://arxiv.org/abs/1507.06217), [Russo and Tanaka, orientational ordering before crystallization](https://pmc.ncbi.nlm.nih.gov/articles/PMC3395031/) (hard-sphere evidence, not direct evidence for these metals), [MACE](https://arxiv.org/abs/2206.07697), [SchNet](https://arxiv.org/abs/1706.08566).
'''
    (out/'RESULTS.md').write_text(text)
    files=sorted((ROOT/'experiments/liquid_sro_benchmark_20260905').glob('*.py'))+sorted((ROOT/'experiments/liquid_sro_benchmark_20260905').glob('*.json'))+[ROOT/'src/analysis/liquid_structure.py',ROOT/'src/models/encoders/atomic_graph.py',ROOT/'src/models/encoders/smooth_density.py',ROOT/'tests/test_liquid_sro_benchmark.py']
    write_json(out/'provenance.json',dict(host=platform.node(),python=sys.version,executable=sys.executable,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        versions={n:importlib.metadata.version(n) for n in ('torch','mace-torch','e3nn','cuequivariance','cuequivariance-torch','gudhi','dscribe','ase','numpy','scipy')},
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        checkpoints={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'models').glob('*/selected.pt'))},
        validation='5 tests passed; detailed output in validation.log; report gates the final fused-MACE rotation results'))
    print(primary)


if __name__=='__main__':main()
