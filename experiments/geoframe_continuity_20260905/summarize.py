"""Render the controlled continuity audit and its selected Ta counterexample."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import ROOT, encode_groups, gather_groups, load_model, signatures, transport_frames, write_json


def main():
    output = ROOT / sys.argv[1]
    cfg = json.loads((output / "config.json").read_text())
    records = json.loads((output / "paths.json").read_text())
    modes = ["normal", "fixed_outer", "fixed_groups", "transported_frames", "fixed_frames"]
    labels = ["Full\nselection", "Fixed\n80 atoms", "Fixed\npatches", "Transported\nframes", "Held\nframes"]
    colors = ["#808b96", "#6193bb", "#be514b", "#168576", "#9c7bc0"]
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for row, name in enumerate(cfg["checkpoints"]):
        data = np.load(output / f"{name}_paths.npz")
        refinement = np.load(output / f"{name}_refinement.npz")
        summary[name] = {}
        for col, material in enumerate(["Al", "Mg", "Ta"]):
            mask = np.array([r["material"] == material for r in records])
            result = dict(paths=int(mask.sum()), modes={})
            for mode in modes:
                steps = np.linalg.norm(np.diff(data[mode][mask].astype(np.float64), axis=1), axis=-1)
                result["modes"][mode] = dict(zip(["p50", "p95", "max"], np.quantile(steps, [.5, .95, 1], axis=(0, 1)).tolist()))
            k = 10  # comfortably above the final float32 refinement floor
            result["refinement"] = dict(
                iteration=k, median_input_rms_A=float(np.median(refinement["input_rms_A"][k, mask])),
                median_frame_angle_deg=float(np.median(refinement["angle_deg"][k, mask])),
                median_native_jump=np.median(refinement["native_jump"][k, mask], axis=0).tolist(),
                median_held_jump=np.median(refinement["held_jump"][k, mask], axis=0).tolist(),
                median_transported_jump=np.median(refinement["transported_jump"][k, mask], axis=0).tolist(),
                median_held_suppression_percent=(100 * (1 - np.median(refinement["held_jump"][k, mask] / refinement["native_jump"][k, mask], axis=0))).tolist(),
                minimum_frame_angle_deg=float(refinement["angle_deg"][k, mask].min()))
            summary[name][material] = result
            axes[row, col].bar(np.arange(5), [result["modes"][mode]["p95"][0] for mode in modes], color=colors)
            axes[row, col].set(xticks=np.arange(5), xticklabels=labels, title=f"{name.replace('_', ' ')} · {material}", yscale="log")
            axes[row, col].tick_params(axis="x", labelsize=8)
            axes[row, col].grid(axis="y", alpha=.2)
        axes[row, 0].set_ylabel("95th percentile encoder step distance")
    fig.suptitle("The same continuous coordinate paths with grouping and frame interventions\n24 paths per material; 256 equal interpolation steps between 21.0 and 21.1 ps snapshots", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "grouping_frame_ablation.png", dpi=180)
    plt.close(fig)
    write_json(output / "metrics.json", summary)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    data = np.load(output / "vicreg_best_paths.npz")
    refinement = np.load(output / "vicreg_best_refinement.npz")
    ta = np.flatnonzero([r["material"] == "Ta" for r in records])
    case = int(ta[np.argmax(refinement["native_jump"][10, ta, 0])])
    inputs = np.load(output / "inputs.npz")
    x = torch.from_numpy(inputs["endpoints"][case, :, :80]).cuda()
    ci = torch.from_numpy(data["base_ci"][case:case+1]).cuda()
    gi = torch.from_numpy(data["base_gi"][case:case+1]).cuda()
    model = load_model(ROOT / cfg["checkpoints"]["vicreg_best"])
    left, right = (float(refinement[key][10, case]) for key in ["left", "right"])
    with torch.inference_mode():
        pair = x[0] + torch.tensor([left, right], device="cuda")[:, None, None] * (x[1] - x[0])
        ng, centers = gather_groups(pair, ci.expand(2, -1), gi.expand(2, -1, -1))
        frames = model.encoder.token_encoder._frames_shape_and_chirality(ng, centers)[0]
        angles = torch.acos((((frames[0].transpose(-1, -2) @ frames[1]).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        patch = int(angles.argmax())
        native = encode_groups(model, ng, centers)
        direction = (native[1, 0] - native[0, 0])
        direction /= torch.linalg.vector_norm(direction)
        alpha = torch.linspace(left - .00025, right + .00025, 257, device="cuda")
        clouds = x[0] + alpha[:, None, None] * (x[1] - x[0])
        patches, pc = gather_groups(clouds, ci.expand(len(alpha), -1), gi.expand(len(alpha), -1, -1))
        raw = encode_groups(model, patches, pc)
        transported = encode_groups(model, patches, pc, transport_frames(ng[:1].expand_as(patches), patches, frames[:1].expand(len(alpha), -1, -1, -1)))
        component = torch.stack([(raw[:, 0] - native[0, 0]) @ direction,
                                 (transported[:, 0] - native[0, 0]) @ direction], dim=1).cpu().numpy()
        axes_indices = signatures(ng)[:, patch].cpu().numpy()
    atom_ids = inputs["atom_ids"][case]
    group_ids = atom_ids[data["base_gi"][case, patch]]
    case_report = dict(path_index=case, path=records[case], patch_index=patch,
                       patch_center_atom_id=int(atom_ids[data["base_ci"][case, patch]]),
                       patch_atom_ids=group_ids.tolist(),
                       left_axis_atom_ids=group_ids[axes_indices[0]].tolist(),
                       right_axis_atom_ids=group_ids[axes_indices[1]].tolist())
    for key in ["left", "right", "input_rms_A", "angle_deg", "native_jump", "held_jump", "transported_jump"]:
        case_report[key] = refinement[key][10, case].tolist()
    write_json(output / "Ta_counterexample.json", case_report)
    np.savez_compressed(output / "Ta_counterexample.npz", points=pair.cpu().numpy(),
                        patches=ng.cpu().numpy(), centers=centers.cpu().numpy(), frames=frames.cpu().numpy(),
                        alpha=alpha.cpu().numpy(), projected_encoder=component, axis_indices=axes_indices)
    fig = plt.figure(figsize=(13, 4))
    ax = fig.add_subplot(131, projection="3d")
    cloud = ng[0, patch].cpu().numpy() * records[case]["radius_A"]
    ax.scatter(*cloud.T, s=18, c="#858585", alpha=.6)
    for side, style in [(0, "-"), (1, "--")]:
        basis = frames[side, patch].cpu().numpy()
        for k, color in enumerate(["#c74747", "#498c62", "#4c75b0"]):
            vector = basis[:, k] * 3.5
            ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], style, color=color, linewidth=2)
    ax.set(title=f"Same patch: {case_report['angle_deg']:.1f}° frame switch\nSolid = left; dashed = right", xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)")
    ax.set_box_aspect((1, 1, 1))
    ax = fig.add_subplot(132)
    offset = (alpha.cpu().numpy() - (left + right) / 2) * 1e4
    ax.plot(offset, component[:, 0], color=colors[2], label="Rebuilt triad")
    ax.plot(offset, component[:, 1], color=colors[3], label="Transported frame")
    ax.set(xlabel="Interpolation fraction offset (×10⁻⁴)", ylabel="Encoder coordinate along jump", title="Fixed atoms and patch membership")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(133)
    for key, label, color in [("native_jump", "Rebuilt triad", colors[2]), ("held_jump", "Held frame", colors[4]), ("transported_jump", "Transported frame", colors[3])]:
        ax.loglog(refinement["input_rms_A"][:, case], refinement[key][:, case, 0], marker=".", color=color, label=label)
    ax.set(xlabel="Input RMS separation (Å)", ylabel="Encoder L2 difference", title="Refining the same switch boundary")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "Ta_frame_counterexample.png", dpi=180)
    plt.close(fig)

    lines = ["# GeoFrame continuity audit — September 5, 2026", "",
             "Canonical-frame switches cause large embedding discontinuities in both the original and temporal VICReg checkpoints. The evidence includes float32 Ta and a frame-only intervention with all atom and patch identities held fixed.", "",
             "## Controlled experiment", "",
             "72 tracked centers: four from each of six Al and six Mg branches, and 24 from the single Ta branch. Endpoints are saved MD frames at 21.0 and 21.1 ps. Each path linearly interpolates atom-matched local coordinates at 257 fractions. **Intermediate points are mathematical continuity probes, not newly simulated MD frames.** Al/Mg endpoints retain float16 quantization; Ta is float32.", "",
             "Each center has 256 tracked candidate atoms; the pool was checked to include the endpoint's nearest 80. Full selection rebuilds the nearest 80, FPS centers, patch neighbors and frames. Fixed-outer selection retains the initial 80 atoms. Fixed-patch selection additionally retains the initial 24 centers and ordered 16-atom patches (including the 8-atom subpatch). Frame interventions change only canonical frames; current positions, shape features, chirality, network weights and projector remain active.", "",
             "Transport uses an SO(3) Kabsch fit of matched patch vectors from the reference to the current configuration, applied to the reference frame. It is a local motion-based diagnostic with fixed patch identities, not a deployed streaming canonicalizer.", "",
             "Inference is uncompiled float32, TF32 disabled, no augmentation or training. Grouped reconstruction matches the direct encoder exactly; repeated inference is bitwise identical. Transported frames pass the rigid-rotation control. Exact errors are in the two controls JSON files.", "",
             "## Isolating a frame switch", "",
             "One large frame-switch interval per path was selected from the fixed-patch encoder curve, separately for each checkpoint. Bisection narrows a triad-axis selection boundary inside that interval. All 72 paths contain a >10° frame switch, and all 72 refined boundaries still exceed 10°. This is targeted event selection, not an unbiased estimate of jump frequency.", "",
             "The table uses bisection iteration 10, above the final float32 refinement floor. Distances are raw L2 in the same checkpoint's encoder space; reductions compare interventions on identical inputs.", "",
             "| VICReg best | Median input RMS separation (Å) | Median maximum patch rotation | Median native encoder jump | Median encoder difference with frame held | Median paired reduction |",
             "|---|---:|---:|---:|---:|---:|"]
    for material in ["Al", "Mg", "Ta"]:
        r = summary["vicreg_best"][material]["refinement"]
        lines.append(f"| {material} | {r['median_input_rms_A']:.3g} | {r['median_frame_angle_deg']:.1f}° | {r['median_native_jump'][0]:.4f} | {r['median_held_jump'][0]:.3g} | {r['median_held_suppression_percent'][0]:.4f}% |")
    lines.extend(["", "The projector also jumps, and holding/transporting the frame removes essentially all of its selected-boundary jump. Raw checkpoint scales differ and selected events may differ, so the original-versus-fine-tuned jump amplitudes are not a causal comparison of training objectives.", "",
                  f"A concrete Ta example is center atom {case_report['path']['center_atom_id']}, patch center {case_report['patch_center_atom_id']}. Inputs differ by {case_report['input_rms_A']:.3g} Å RMS; the patch frame rotates {case_report['angle_deg']:.2f}°. Encoder difference is {case_report['native_jump'][0]:.6f}, falling to {case_report['held_jump'][0]:.3g} with the left frame held and {case_report['transported_jump'][0]:.3g} with motion transport. Axis atom IDs change from {case_report['left_axis_atom_ids']} to {case_report['right_axis_atom_ids']}. All patch atom IDs stay fixed. See [exact counterexample](Ta_counterexample.json).", "",
                  "![Ta frame-switch counterexample](Ta_frame_counterexample.png)", "",
                  "## Whole interpolation paths", "",
                  "![Grouping and frame interventions](grouping_frame_ablation.png)", "",
                  "| VICReg encoder, 95th-percentile step L2 | Full selection | Fixed outer 80 | Fixed patch identities | Transported frames with fixed patches |",
                  "|---|---:|---:|---:|---:|"])
    for material in ["Al", "Mg", "Ta"]:
        s = summary["vicreg_best"][material]["modes"]
        lines.append("| " + material + " | " + " | ".join(f"{s[m]['p95'][0]:.4f}" for m in modes[:4]) + " |")
    lines.extend(["", "Transporting frames reduces fixed-patch encoder p95 step distance by approximately 86% / 86% / 89% on Al / Mg / Ta. The grouping interventions also strongly reduce the upper tail, so a frame-only replacement does not address every source of discontinuity. These interventions change the computation, and their reductions are not additive causal percentages of real MD drift.", "",
                  "## Implications for dynamic canonicalization", "",
                  "A causal frame should follow local motion: initialize once, retain tracked atom/patch identities, estimate a weighted proper rotation from the previous patch to the current patch, and transport the previous frame by that rotation. Use smooth radial weights to handle entering/leaving neighbors, and a confidence-controlled correction toward current structural orientation to limit accumulated drift. Re-anchoring and patch replacement must be blended over time instead of resetting abruptly. A small constrained residual rotation could later be learned, but the deterministic transport baseline should be tested first.", "",
                  "The model state would include frames and membership history. Initialization dependence, recovery after a trajectory gap, rigid-rotation equivariance, long-time frame drift, sensitivity to real transitions and retained structural discrimination all need evaluation. The current pretrained model was trained with rebuilt triads, so transported frames are a distribution change; smoother outputs alone do not establish equal structural quality. An end-to-end trained dynamic model remains future work.", "",
                  "This experiment establishes a canonicalization failure mechanism, including without quantized endpoints. It does not determine the fraction of observed 0.1 ps MD drift attributable to it, nor demonstrate long-horizon predictive accuracy. The source sample is small and Ta still has one trajectory.", "",
                  "## Reproduction and artifacts", "",
                  "See [the experiment record](../../experiments/geoframe_continuity_20260905/README.md). inputs.npz retains the exact interpolated endpoints and atom identities. Per-checkpoint paths.npz and refinement.npz retain embeddings, group indices, frames, selection indices and refinement measurements. metrics.json contains encoder/projector statistics. provenance.json records model and implementation hashes. All generated artifacts are under this repository output directory.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(f"Saved {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
