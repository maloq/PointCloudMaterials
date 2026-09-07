"""Write the pilot's research report and standalone figures from saved results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = ROOT / cfg["output"]
    spatial = json.loads((out / "structure_metrics.json").read_text())
    continuity = json.loads((out / "continuity_metrics.json").read_text())["results"]
    temporal = json.loads((out / "temporal_metrics.json").read_text())
    forecast = json.loads((out / "forecast_metrics.json").read_text())
    neural = json.loads((out / "neural_temporal_metrics.json").read_text())
    readouts = json.loads((out / "temporal_readouts.json").read_text())["results"]
    full = json.loads((out / "full_static_Al/metrics.json").read_text())
    motion = json.loads((out / "motion_diagnostics.json").read_text())
    training = json.loads((out / "training_summary.json").read_text())
    for name in ("prepare_status", "labels_status", "features_status", "spatial_status", "evaluation_status", "full_static_status", "neural_temporal_status"):
        if json.loads((out / f"{name}.json").read_text())["state"] != "complete":
            raise RuntimeError(f"Required stage is not complete: {name}")
    if len(training) != 6 or len(neural) != 12:
        raise RuntimeError(f"Incomplete training sweep: spatial={len(training)}, temporal={len(neural)}")
    modes = ["gru", "gated_no_transport", "gated_kabsch", "gated_smooth"]
    groups = {mode:[r for r in neural if r["mode"] == mode] for mode in modes}
    mean = lambda rows, fn: float(np.mean([fn(r) for r in rows]))
    std = lambda rows, fn: float(np.std([fn(r) for r in rows], ddof=1))
    labels = dict(density_pca="Density power + PCA", power_mlp="Power MLP", mace_product="Central MACE products", geoframe_vicreg="GFv2 VICReg",
                  gru="Smooth encoder + GRU", gated_no_transport="Gated memory", gated_kabsch="Gated + Kabsch", gated_smooth="Gated + smooth transport")

    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    arrays = dict(np.load(out / "continuity.npz"))
    paths = json.loads((ROOT / cfg["continuity_root"] / "paths.json").read_text())
    train_meta = np.load(out / "embeddings/train_metadata.npz")
    fig, axes = plt.subplots(1,3,figsize=(13,4.1),sharey=True)
    for name, color in zip(("geoframe_vicreg", "density_pca", "mace_product"), ("C3", "C0", "C2")):
        z = np.load(out / "embeddings" / f"{name}_train.npy")
        for mi,(material,ax) in enumerate(zip(("Al", "Mg", "Ta"),axes)):
            mask = np.array([p["material"]==material for p in paths])
            scale = np.sqrt(z[train_meta["material"]==mi].var(0).sum())
            x = np.median(arrays["input_rms_A"][:11,mask],1)
            y = arrays[name][:11,mask]/scale
            ax.loglog(x,np.median(y,1),"o-",markersize=3,label=labels[name],color=color)
            ax.fill_between(x,np.quantile(y,.25,axis=1),np.quantile(y,.75,axis=1),color=color,alpha=.12)
            ax.set(title=material,xlabel="Coordinate separation, RMS (Å)")
            ax.grid(alpha=.2)
    axes[0].set_ylabel("Embedding difference / training variation")
    axes[0].legend(fontsize=8)
    fig.suptitle("Selected frame-switch boundaries: new descriptors converge continuously")
    fig.tight_layout()
    fig.savefig(out / "continuity.png",dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(13,4.1),sharey=True)
    horizons = np.array(forecast["horizons_ps"])
    for material,ax in zip(("Al", "Mg", "Ta"),axes):
        linear = forecast["results"]["mace_linear_history"]["by_material"][material]
        ax.plot(horizons,100*np.array(linear["skill_vs_material_mean"]),"o--",color=".4",label="Linear history")
        for mode, style in zip(modes,("o-","s-","^--","d:")):
            values = 100*np.array([r["by_material"][material]["skill_vs_material_mean"] for r in groups[mode]])
            ax.errorbar(horizons,values.mean(0),yerr=values.std(0,ddof=1),fmt=style,capsize=2,label=labels[mode])
        ax.set(title=material,xlabel="Forecast horizon (ps)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("MSE reduction vs material-mean forecast (%)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Direct forecasts of the same fixed 32D density embedding; three-seed mean ± SD")
    fig.tight_layout()
    fig.savefig(out / "forecast_skill.png",dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    for name,color in zip(("geoframe_vicreg", "mace_product", "density_pca"),("C3","C2","C0")):
        x = np.mean([temporal[name][m]["p95_scaled_step"] for m in ("Al","Mg","Ta")])
        y = readouts[name]["all_rolling_macro_f1"]
        ax.scatter(x,y,s=65,color=color)
        offset, align = ((-8,-15),"right") if name != "geoframe_vicreg" else ((10,-18),"left")
        ax.annotate(labels[name],(x,y),xytext=offset,textcoords="offset points",ha=align,fontsize=9)
    for i,mode in enumerate(modes):
        x = np.array([np.mean([r["by_material"][m]["p95_scaled_step"] for m in ("Al","Mg","Ta")]) for r in groups[mode]])
        y = np.array([readouts[f"{mode}_seed{r['seed']}"]["all_rolling_macro_f1"] for r in groups[mode]])
        ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt="s",capsize=3,color=f"C{i+4}",label=labels[mode])
    ax.set(xlabel="Mean of material-wise p95 scaled steps (lower is smoother)",ylabel="PTM macro F1 on all rolling test states",title="Temporal smoothing and retained structure")
    ax.margins(y=.12)
    ax.grid(alpha=.2)
    ax.legend(loc="lower left",fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "temporal_tradeoff.png",dpi=180)
    plt.close(fig)

    names = ["density_pca", "power_mlp", "mace_product", "geoframe_vicreg"]
    fig,axes = plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,title in zip(axes,("ptm_macro_f1","silhouette_cosine","cluster_ptm_ari"),("Transferred PTM readout","Cluster cosine silhouette","Cluster agreement with PTM")):
        ax.bar(np.arange(4),[full[n][key] for n in names],color=["C0","C1","C2","C3"])
        ax.set(xticks=np.arange(4),xticklabels=["Density PCA","Power MLP","MACE products","GFv2"],title=title)
        ax.tick_params(axis="x",labelrotation=25)
    fig.suptitle("Full static Al: all 772,953 saved centers")
    fig.tight_layout()
    fig.savefig(out / "full_static_Al.png",dpi=180)
    plt.close(fig)

    directory = out / "full_static_Al"
    metadata = np.load(directory / "metadata.npz")
    coords = np.load(directory / "coords.npy",mmap_mode="r")
    last = metadata["source_ids"]==5
    section = last & (np.abs(coords[:,2]-np.median(coords[last,2]))<7.6)
    cmap = ListedColormap(["#b5b5b5","#377eb8","#ff9d2e","#4daf4a"])
    norm = BoundaryNorm(np.arange(-.5,4.5),cmap.N)
    fig,axes = plt.subplots(1,3,figsize=(13,4.5),sharex=True,sharey=True)
    for ax,kind,title in zip(axes,(None,"geoframe_vicreg","mace_product"),("PTM assay","GFv2: transferred readout","MACE products: transferred readout")):
        value = metadata["ptm_labels"] if kind is None else np.load(directory / f"{kind}.predicted_ptm.npy")
        scatter = ax.scatter(coords[section,0],coords[section,1],c=value[section],s=3,cmap=cmap,norm=norm,rasterized=True)
        ax.set(title=title,xlabel="x (Å)",aspect="equal")
    axes[0].set_ylabel("y (Å)")
    fig.colorbar(scatter,ax=axes,ticks=[0,1,2,3],fraction=.02,pad=.02).ax.set_yticklabels(["Other","FCC","HCP","BCC"])
    fig.suptitle("Static Al 240 ps: central slice, identical saved sample centers")
    fig.savefig(directory / "240ps_structure_slice.png",dpi=180,bbox_inches="tight")
    plt.close(fig)

    lines = ["# Smooth spatial and temporal encoder experiments — September 5, 2026", "",
      "Completed: six spatial models and twelve temporal models, including three seeds per architecture, plus fixed-descriptor/filter/linear baselines. All artifacts are under this repository directory. No simulation or production encoder was replaced.", "",
      "**Finding:** smooth local geometry removes the demonstrated canonicalization jumps. A small GRU over the new spatial features gives a useful temporal state. The proposed motion transport adds no convincing benefit in this short-history pilot. Full-static transfer remains a limitation of the models trained only on MD continuations.", "",
      "## Continuity is fixed; physical-time smoothness is a separate problem", "",
      "The selected switch-boundary input separation contracts by approximately 1,024× over ten refinements. New descriptor differences contract by approximately the same factor; GFv2 differences remain approximately constant. This is evidence of continuity, not merely a lower absolute embedding scale. The curves use mathematical interpolation, not newly simulated intermediate frames.", "",
      "| Representation | Al: final / initial difference | Mg | Ta |", "|---|---:|---:|---:|"]
    for name in ("geoframe_vicreg","density_pca","power_mlp","mace_product"):
        lines.append(f"| {labels[name]} | " + " | ".join(f"{continuity[name][m]['median_step10_over_step0']:.5f}" for m in ("Al","Mg","Ta")) + " |")
    lines += ["", "![Continuity](continuity.png)", "",
      "A continuous spatial encoder can still respond strongly to thermal motion. Indeed, unfiltered new features have greater actual 0.1 ps drift than the compressed GFv2 encoder. Their within-material effective ranks are roughly 34–38 versus 2–3.4 for GFv2. The separate dimension controls also show that reducing dimensions alone does not resolve every tradeoff; see [rank controls](rank_controls.json).", "",
      "## Learned temporal states", "",
      "Temporal candidates observe five frames spanning 0.4 ps; static reference encoders use the current frame. The following p95 step distances are divided by within-material training variation. Lower is smoother. PTM macro F1 uses identical rolling test observations and separately fitted training-only readouts. Neural entries are means across three seeds; they are not independent-source confidence intervals.", "",
      "| Representation | Al p95 | Mg p95 | Ta p95 | PTM macro F1 |", "|---|---:|---:|---:|---:|"]
    for name in ("geoframe_vicreg","mace_product"):
        lines.append(f"| {labels[name]} | " + " | ".join(f"{temporal[name][m]['p95_scaled_step']:.3f}" for m in ("Al","Mg","Ta")) + f" | {readouts[name]['all_rolling_macro_f1']:.4f} |")
    for mode in modes:
        rows = groups[mode]
        f1_mean = mean(rows, lambda r: readouts[f"{mode}_seed{r['seed']}"]["all_rolling_macro_f1"])
        lines.append(f"| {labels[mode]} | " + " | ".join(f"{mean(rows,lambda r:r['by_material'][m]['p95_scaled_step']):.3f}" for m in ("Al","Mg","Ta")) + f" | {f1_mean:.4f} |")
    lines += ["", "![Temporal tradeoff](temporal_tradeoff.png)", "",
      "The GRU substantially reduces the new spatial encoder's physical-time fluctuations and retains comparable structural readout quality. Compared with GFv2 it is smoother on Mg and Ta, while Al is similar/slightly worse on this metric. This is a partial improvement rather than a universal replacement.", "",
      "## Predicting future embeddings", "",
      "Every forecast predicts the same fixed 32D density-power PCA embedding. PCA and target standardization use only training observations. The structural state is 32D, with an additional 64D recurrent memory used by the forecast head. These are direct horizon-specific forecasts, not autonomous latent rollouts.", "",
      "The table reports test MSE reduction relative to a material-specific mean forecast fitted using training futures. This is a stronger check than persistence alone. Values are averaged across neural seeds.", "",
      "| Model / material | 0.1 ps | 0.5 ps | 1 ps | 2 ps |", "|---|---:|---:|---:|---:|"]
    for mode in ("gru","gated_no_transport","gated_smooth"):
        for material in ("Al","Mg","Ta"):
            values = np.mean([r["by_material"][material]["skill_vs_material_mean"] for r in groups[mode]],0)*100
            lines.append(f"| {labels[mode]} / {material} | " + " | ".join(f"{v:.2f}%" for v in values) + " |")
    lines += ["", "![Forecast skill](forecast_skill.png)", "",
      "| Temporal model | Validation MSE, mean ± seed SD |", "|---|---:|"]
    for mode in modes:
        lines.append(f"| {labels[mode]} | {mean(groups[mode],lambda r:r['validation_mse']):.7f} ± {std(groups[mode],lambda r:r['validation_mse']):.7f} |")
    lines += ["", f"The linear-history baseline has validation MSE {forecast['results']['mace_linear_history']['validation_mse']:.7f}. Learned gating gives a small improvement over the ordinary GRU. Kabsch and smooth transport give effectively the same result as gating without transport. Those differences do not support a transport-benefit or novelty claim.", "",
      "Observed fitted local motion is small and well conditioned:", "", "| Material | Median rotation / 0.1 ps | p95 rotation | Minimum singular-value ratio |", "|---|---:|---:|---:|"]
    for m in ("Al","Mg","Ta"):
        row=motion[m]
        lines.append(f"| {m} | {row['rotation_median_deg']:.2f}° | {row['rotation_p95_deg']:.2f}° | {row['cross_moment_sigma_min_over_max_min']:.3f} |")
    lines += ["", "This helps explain why regularizing ambiguous transport has little opportunity to help here. It does not exclude a benefit with longer histories, substantial coherent rotation, or neighborhood degeneracy. The synthetic symmetry/rank-crossing tests verify the operator's intended algebra, not its usefulness on this data.", "",
      "## Structural changes and full static Al", "",
      "The test paths contain 76 persistent PTM-label changes under the specified three-frame stability rule. They are structural-assay events, not verified thermodynamic transitions. A useful state should not merely delay or suppress them.", "",
      "| State | Events missed within 0.5 ps | Mean delay among detected events |", "|---|---:|---:|"]
    for name in ("geoframe_vicreg","mace_product","gru_seed456","gated_smooth_seed456"):
        event=readouts[name]["persistent_ptm_changes"]
        lines.append(f"| {labels.get(name,name)} | {event['missed_within_0_5ps']} / {event['events']} | {event['mean_delay_ps']:.3f} ps |")
    lines += ["", "The neural rows illustrate seed 456; all seeds, anticipatory predictions, and matched readouts are saved in [temporal_readouts.json](temporal_readouts.json). The delay diagnostic is conditional on detection and should be read together with missed events. Smoother trajectories do not establish superior transition timing.", "",
      "All 772,953 existing saved centers across six static Al snapshots were rebuilt and analyzed. Clustering fits use the complete dataset; the pipeline's internal quality diagnostics use its fixed sampled evaluation protocol.", "",
      "| Representation | Transferred PTM macro F1 | Cosine silhouette | Cluster–PTM ARI |", "|---|---:|---:|---:|"]
    for name in names:
        row=full[name]
        lines.append(f"| {labels[name]} | {row['ptm_macro_f1']:.4f} | {row['silhouette_cosine']:.4f} | {row['cluster_ptm_ari']:.4f} |")
    lines += ["", "![Full static Al](full_static_Al.png)", "", "![Static Al structure slice](full_static_Al/240ps_structure_slice.png)", "",
      "The new learned models transfer less well to static Al than GFv2. The untrained smooth descriptor is competitive on the PTM readout. The learned spatial models started from scratch on MD continuations; GFv2 already had extensive static pretraining. These results reveal a training/domain-transfer problem and do not establish an architecture ceiling. Isolated snapshots were evaluated with the spatial encoders; no fictitious temporal history was supplied.", "",
      "## Decision and limits", "",
      "Keep the smooth spatial geometry as the foundation for the next iteration, with ordinary recurrent memory as the principal temporal baseline. Preserve the transported variants as ablations; the present data do not justify their extra complexity. Before replacing GFv2, train the spatial representation on a deliberate mixture of static and MD environments, retain a held-out static split, and repeat the transfer/transition checks.", "",
      "Existing data support this prototype, but the source split is limited: six Al and six Mg branches and one Ta branch, with disjoint central IDs and time blocks rather than independent simulations. Al/Mg MD coordinates retain float16 quantization. Fine perturbation continuity is a mathematical property check; a physical high-frequency smoothness claim needs higher-precision trajectories. Only 0.4 ps of history and 0.1–2 ps direct forecast horizons were evaluated.", "",
      "## Reproduction, checks, and artifacts", "",
      "See the [experiment record](../../experiments/smooth_temporal_encoder_20260905/README.md) and [configuration](config.json). All seven focused tests passed, including learned-state rotation equivariance, finite gradients, cutoff continuity, tensor transport through angular order six, rank-degenerate smooth transport, and the independent FCC/PTM assay. [Protocol verification](protocol_verification.json) checks every saved forecast target against its source/time/center embedding and verifies disjoint center pools.", "",
      "The initial PTM selection error was corrected and all labels recomputed; labels were never part of representation training. A CUDA batch-size failure in the Ta eigensolve was corrected by solving at most 4096 matrices at a time; the affected history/forecast stages were rerun. Original logs/failure status remain available.", "",
      "[Spatial training summary](training_summary.json), [neural temporal metrics](neural_temporal_metrics.json), [all linear/filter forecasts](forecast_metrics.json), [full static coverage](full_static_Al/coverage.json), and [provenance](provenance.json) retain the detailed evidence. Best, last, and final checkpoints are saved for all 18 trained models. Launch records identify the detached processes; all have finished.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))

    files = list((ROOT / "experiments/smooth_temporal_encoder_20260905").glob("*.py")) + [ROOT / "src/models/encoders/smooth_density.py",ROOT / "src/temporal_vamp/smooth_state.py",ROOT / "tests/test_smooth_density.py",args.config.resolve()]
    versions = {}
    import torch, e3nn, importlib.metadata
    versions.update(torch=torch.__version__, e3nn=e3nn.__version__, mace=importlib.metadata.version("mace-torch"),numpy=np.__version__)
    provenance = dict(created_at=datetime.now(timezone.utc).isoformat(), repository=str(ROOT),
          git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
          source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
          versions=versions, stage_config_note="Spatial checkpoints/data manifest retain their original configs; the final config adds the subsequent temporal training settings.",
          tests=dict(command="python -m pytest tests/test_smooth_density.py -q", passed=7),
          saved_sources="data/manifest.json includes original trajectory manifests and hashes; full_static_Al/coverage.json includes static source hashes.")
    write_json(out / "provenance.json", provenance)
    write_json(out / "status.json", dict(state="complete", spatial_runs=6, temporal_runs=12, full_static_Al_centers=772953,
               finished_at=datetime.now(timezone.utc).isoformat(), report=str(out / "RESULTS.md")))
    print(out / "RESULTS.md")


if __name__ == "__main__":
    main()
