# Implementation Guide for Coding Agent — Training

## Goal

Implement a new end-to-end **Temporal Motif Field** training method for bulk-material MD local-neighborhoods that:

1. learns **geometrically distinct stable motifs**,
2. learns **transient / bridge motifs** that recur during transitions,
3. predicts **future motif evolution** at multiple lags,
4. optionally models **spatial propagation** of motifs across nearby center atoms.

This guide is written to fit the current `PointCloudMaterials` repo layout and training conventions.

---

## Grounding assumptions from the current repo

Use these as constraints when coding:

- Keep using the repo’s generic trainer path in `src/training_methods/trainer.py`.
- Keep using the existing temporal datamodule path selected when `data.kind == "temporal_lammps"`.
- Temporal batches are `dict` objects with required key `points`; current temporal code also reads metadata keys such as `center_atom_id`, `frame_indices`, `timesteps`, `source_path`, `instance_id`, `anchor_frame_index`, `anchor_timestep`.
- Current temporal SSL already has a helper that can encode all frames in a sequence, but its main forward path still collapses a 4D sequence to the center frame; do **not** repeat that limitation in the new method.
- Keep encoder integration compatible with the current encoder API: some encoders return only invariant latents, others return invariant + equivariant latents.

---

## Deliverables

Create these files:

```text
src/training_methods/temporal_motif_field/
    __init__.py
    temporal_motif_field_module.py
    losses.py
    prototype_heads.py
    temporal_heads.py
    bridge_mining.py
    utils.py

src/training_methods/temporal_motif_field/train_temporal_motif_field.py

configs/temporal_motif_field_lammps.yaml
```

Optional but recommended:

```text
src/training_methods/temporal_motif_field/callbacks.py
```

for bridge-candidate refresh and extra logging.

---

## Design choice

The method should be **differentiable end-to-end**, but **trained in stages**.

Why:

- stable motif prototypes are easier to form before bridge motifs are introduced,
- bridge mining is naturally semi-EM / refresh-based,
- full joint training from step 0 is likely to be unstable,
- staged optimization still allows joint fine-tuning once the heads are initialized.

Use this schedule:

1. **Stage 0 — spatial warm start**  
   Train encoder + projector + stable head with spatial VICReg only (or VICReg + weak stable loss).
2. **Stage 1 — stable temporal motifs**  
   Turn on multi-lag temporal prediction and stable prototype learning.
3. **Stage 2 — bridge motifs**  
   Start bridge candidate mining / gating and train bridge prototype bank.
4. **Stage 3 — optional motif field**  
   Add neighbor-conditioned future prediction and fine-tune end-to-end.

Do **not** create separate incompatible checkpoints per stage. Keep one module and activate losses by epoch schedule.

---

## High-level module layout

Implement one Lightning module:

```python
class TemporalMotifFieldModule(pl.LightningModule):
    ...
```

### Submodules inside it

1. `encoder`
   - Reuse existing encoder build path.
   - Must accept per-frame local neighborhoods.
   - Must support invariant-only and invariant+equivariant outputs.

2. `projector`
   - Small MLP mapping encoder output to motif latent space.
   - Output dimension = `latent_size`.

3. `ema_teacher`
   - EMA copy of encoder + projector.
   - Used only for targets in temporal prediction and prototype assignments.
   - Teacher update after every optimizer step.

4. `stable_head`
   - Prototype bank for stable motifs.
   - Inputs: anchor latent.
   - Outputs:
     - `stable_logits`
     - `stable_probs`
     - `stable_proto_recon` (weighted prototype reconstruction)
     - optional `stable_assign_hard` for logging only

5. `residual_head`
   - Predicts residual component not explained by stable prototypes.
   - Keeps prototypes from absorbing thermal noise / elastic variation.

6. `temporal_context`
   - Sequence model over short windows.
   - Default: causal transformer encoder over context frames.
   - Alternative config options: MLP / GRU / SSM.
   - Input: `z_{t-context+1 : t}`.
   - Output: context summary for future prediction.

7. `future_predictor`
   - Predicts future stable motif distributions and future residuals for each lag.
   - Can be one shared MLP with per-lag embeddings, or one head per lag.

8. `hazard_head`
   - Predict probability of change event within each future lag.
   - Used for duration regularization and bridge candidate scoring.

9. `bridge_head`
   - Separate prototype bank for bridge/transient motifs.
   - Trained only on bridge candidates.
   - Inputs: anchor latent and/or change-window latent.

10. `field_head` (optional in first PR, but keep interface ready)
    - Predict future motif using anchor context plus neighboring-center motif states.
    - Enable only after stable + bridge stages work.

---

## Data and batch contract

Assume current temporal dataloader returns:

```python
batch["points"]              # shape [B, T, N, 3]
batch["center_atom_id"]      # shape [B] or list-like
batch["frame_indices"]       # shape [B, T] or list-like
batch["timesteps"]           # shape [B, T] or list-like
batch["source_path"]         # optional grouping key
batch["instance_id"]         # optional
batch["anchor_frame_index"]  # optional
batch["anchor_timestep"]     # optional
```

### Required invariants for the new method

- `T == data.sequence_length`
- `T >= tmf.context_frames + max(tmf.lags)`
- Sequence order must be chronological.
- Anchor frame is the **last context frame** by default.

### New optional batch keys

If easy to add in the datamodule, support:

```python
batch["neighbor_center_ids"]     # IDs of nearby center atoms
batch["neighbor_points"]         # local neighborhoods for neighboring center atoms
batch["neighbor_offsets"]        # index offsets to reconstruct ragged neighbors
```

Do not make these mandatory for the first working version.

---

## Encoder handling

### Rule

Always convert encoder output to one invariant latent `z_frame`.

Pseudo:

```python
z_inv_model, eq_z = self._split_encoder_output(self.encoder(frame_pc))

if eq_z is not None and self.hparams.tmf.use_eq_norms_for_invariant:
    z_frame = self._shared_invariant(None, eq_z)   # same logic style as existing modules
elif z_inv_model is not None:
    z_frame = z_inv_model
else:
    raise RuntimeError("Encoder returned neither invariant nor equivariant latent.")
```

Then pass `z_frame` through `projector`.

### First implementation recommendation

Make the method encoder-agnostic and get it working with the current encoder first.  
After that, switch the default experiment to the repo’s atomistic equivariant encoder wrapper.

---

## Forward pass API

Implement two forward modes.

### 1. `forward_sequence(batch)`

Input:
- temporal batch dict

Output dict:
- `z_seq`: `[B, T, D]`
- `z_anchor`: `[B, D]`
- `stable_logits_anchor`: `[B, K_stable]`
- `stable_probs_anchor`: `[B, K_stable]`
- `stable_recon_anchor`: `[B, D]`
- `residual_anchor`: `[B, D_r]`
- `context_repr`: `[B, D_ctx]`
- `future_stable_logits`: dict `lag -> [B, K_stable]`
- `future_residual_pred`: dict `lag -> [B, D_r]`
- `hazard_logits`: dict `lag -> [B, 1]`
- `bridge_logits_anchor`: `[B, K_bridge]` or `None`
- `bridge_gate`: `[B, 1]` or `None`
- `teacher_targets`: nested dict with stop-grad target tensors

### 2. `forward(batch)`

Return anchor-frame outputs only for compatibility with inference utilities that expect one latent per sample.

Default:
- return anchor invariant latent plus optional extras needed by cache hooks.

---

## Sequence encoding

Implement a helper similar in spirit to the current temporal SSL helper, but keep all frames:

```python
def _encode_temporal_frame_sequence(self, points_4d) -> torch.Tensor:
    # points_4d: [B, T, N, 3]
    # flatten to [B*T, N, 3]
    # encode each frame independently
    # reshape back to [B, T, D]
```

No temporal mixing inside the encoder itself in v1.  
All temporal reasoning lives in `temporal_context`, `future_predictor`, and bridge logic.

---

## Stable motif head

Use soft assignments.

### Computation

For anchor latent `z` and prototype matrix `P_stable [K, D]`:

```python
stable_logits = -cdist_sq(z, P_stable) / temperature
stable_probs = softmax(stable_logits, dim=-1)
stable_recon = stable_probs @ P_stable
```

### Loss terms

1. **Prototype commitment / reconstruction**
   - Encourage `z` to be explained by stable prototypes plus residual.
   - Example:
     ```python
     L_stable_recon = mse(z.detach(), stable_recon + residual_proj)
     ```
   - Or project both to same dimension and use cosine loss.

2. **Usage balance**
   - Encourage non-degenerate prototype usage across the batch.
   - Use batch-mean assignment entropy or Sinkhorn-style balancing.
   - Start simple:
     ```python
     usage = stable_probs.mean(0)
     L_stable_balance = -entropy(usage)
     ```

3. **Confidence regularization**
   - Mild entropy penalty on per-sample stable assignments after warmup.
   - Do not make this too strong early on.

### Logging

Track:
- `stable/usage_entropy`
- `stable/num_active`
- `stable/max_prob_mean`
- `stable/dead_fraction`

---

## Residual branch

Purpose: let the model represent strain / thermal noise without inventing new motifs.

### Implementation

- `residual_head = MLP(D -> D_r)`
- `residual_decoder = MLP(D_r -> D)` or single linear layer

Loss:
```python
L_residual = mse(z_teacher_anchor, stable_recon + residual_decode(residual_anchor))
```

Keep `D_r` smaller than `D` so prototypes remain the main structure carrier.

Recommended:
- `D = 256`
- `D_r = 64`

---

## Temporal context and future prediction

### Context indexing

Given sequence length `T`, context frames `C`, and lags list `L`:

- anchor index = `C - 1`
- context = frames `[0, 1, ..., C-1]`
- target for lag `Δ` = frame index `anchor + Δ`

Validate:
```python
max(lags) <= T - C
```

### Temporal context model

Start with:
- positional embeddings over local frame offsets,
- transformer encoder over context latents,
- final token or pooled output = `context_repr`.

### Future prediction targets

Teacher targets for each lag:
- `stable_probs_target[lag]`
- `residual_target[lag]`
- `change_target[lag]` = 1 if stable argmax changes between anchor and target

Use stop-grad teacher outputs.

### Losses

For each lag:

1. **Future stable motif CE / KL**
   ```python
   L_future_stable = KL(softmax(pred_logits), stopgrad(stable_probs_target))
   ```

2. **Future residual regression**
   ```python
   L_future_residual = mse(pred_residual, stopgrad(residual_target))
   ```

3. **Hazard / change BCE**
   ```python
   change_target = (argmax(stable_probs_anchor_teacher) != argmax(stable_probs_target_teacher)).float()
   L_hazard = bce_with_logits(hazard_logits, change_target)
   ```

Weight later lags slightly lower if unstable.

---

## Bridge motifs

## Core idea

Bridge motifs are **not** just noisy frames between stable motifs.
They are short-lived but recurring geometric forms.

### Implementation strategy for v1

Use **refresh-based bridge mining**, not fully differentiable online hard mining.

Create `bridge_mining.py` with:

```python
def score_bridge_candidates(cache, lags, pred_errors, stable_ids, stable_confidence, hazard_probs):
    ...
```

### Refresh schedule

Every `tmf.bridge.refresh_epochs`:
1. Run fast inference on a train subset.
2. Compute a `bridge_score` per sample:
   - stable assignment change within any lag,
   - plus prediction error,
   - plus hazard probability,
   - plus optionally high stable entropy.
3. Select top `candidate_fraction`.
4. Save candidate keys:
   - `(source_path, center_atom_id, anchor_frame_index)` or equivalent
5. Datamodule / sampler exposes `is_bridge_candidate` in subsequent epochs.

### Bridge head inputs

Start simple:
- anchor latent `z_anchor`
- optionally concat temporal context repr

### Bridge losses

Apply only to candidate samples.

1. **Bridge prototype assignment**
   - same prototype machinery as stable bank

2. **Bridge distinctiveness**
   - Encourage bridge prototypes to sit away from stable prototypes:
   ```python
   L_bridge_sep = relu(margin - min_pairwise_distance(P_bridge, P_stable))
   ```

3. **Bridge recurrence / balance**
   - same anti-collapse usage term as stable bank

### Important rule

Never let bridge prototypes absorb a huge fraction of all samples.
Log `bridge/candidate_fraction` and `bridge/usage_entropy`.

---

## Hazard and duration modeling

Keep this light in v1.

### Hazard target

For lag `Δ`, target is whether stable motif changes by that lag.

### Metrics during training

Log:
- `hazard/auc_proxy` or `hazard/accuracy`
- `hazard/positive_fraction`
- mean predicted hazard by stage

### Optional duration regularization

Once v1 works, add:
- penalty for rapid stable-label flicker over consecutive small lags,
- but do **not** suppress bridge assignments.

Simple version:
- penalize anchor stable assignment changing at lag 1 if both anchor and target confidences are high and bridge gate is low.

---

## Optional motif field head

Do this in a second PR or after base model is stable.

### Goal

Use neighboring center atoms to predict future motif of the anchor atom.

### Minimal implementation

For each anchor sample, aggregate neighbor stable probabilities and residual summaries:
```python
neighbor_summary = mean(MLP([neighbor_stable_probs, neighbor_residual, relative_position]))
field_input = concat(context_repr, stable_probs_anchor, neighbor_summary)
field_logits = field_head(field_input)
```

Loss:
```python
L_field = KL(softmax(field_logits), stable_probs_target[lag_field])
```

Keep neighbor count small and fixed if possible.

---

## Loss composition

Implement a single loss function returning a dict.

### Suggested total loss

```python
L_total =
    w_vicreg * L_vicreg
  + w_stable_recon * L_stable_recon
  + w_stable_balance * L_stable_balance
  + w_future_stable * sum_lag L_future_stable[lag]
  + w_future_residual * sum_lag L_future_residual[lag]
  + w_hazard * sum_lag L_hazard[lag]
  + w_bridge * L_bridge
  + w_bridge_sep * L_bridge_sep
  + w_field * L_field
```

### Stage-aware activation

Implement helper:

```python
def _loss_weights_for_epoch(self, epoch) -> dict[str, float]:
    ...
```

Example:
- Stage 0: `vicreg`, maybe weak `stable_recon`
- Stage 1: + `stable_*`, `future_*`, `hazard`
- Stage 2: + `bridge_*`
- Stage 3: + `field`

---

## Lightning hooks

### `training_step`

1. unpack batch
2. encode full sequence
3. compute teacher targets (no grad)
4. compute losses
5. log all scalar components
6. return total loss

### `validation_step`

Same as training, no teacher EMA update, no bridge mining refresh.

### `test_step`

Return same metrics plus optional export payload for evaluation.

### `on_after_backward`

Optional:
- gradient norm logging

### `on_train_epoch_end`

If bridge refresh is due:
- trigger callback / helper to recompute candidate set.

### `optimizer_step` or callback

Update EMA teacher after optimizer step.

---

## Checkpoint compatibility

Support:
- `resume_from_checkpoint`
- `init_from_checkpoint`
- `init_from_checkpoint_encoder_only`

Mirror the current temporal SSL behavior:
- if encoder-only init is true, load encoder + projector only,
- do not require prototype shapes to match.

Prototype banks should reinitialize cleanly if absent or mismatched.

---

## Logging requirements

Log these every epoch at minimum.

### Global

- `loss/total`
- `loss/vicreg`
- `loss/stable_recon`
- `loss/stable_balance`
- `loss/future_stable`
- `loss/future_residual`
- `loss/hazard`
- `loss/bridge`
- `loss/bridge_sep`
- `loss/field`

### Stable motifs

- `stable/usage_entropy`
- `stable/active_count`
- `stable/max_prob_mean`
- `stable/dead_fraction`

### Bridge motifs

- `bridge/candidate_fraction`
- `bridge/usage_entropy`
- `bridge/active_count`

### Temporal

- `temporal/top1_future_acc_lag_{d}`
- `temporal/kl_future_lag_{d}`

### Hazard

- `hazard/pos_fraction_lag_{d}`
- `hazard/pred_mean_lag_{d}`

---

## Failure modes to guard against

1. **Prototype collapse**
   - Fix with stronger balance loss, lower confidence regularization, lower learning rate.

2. **Everything becomes bridge**
   - Lower bridge candidate fraction, increase bridge separation, delay bridge stage.

3. **Temporal over-smoothing**
   - Lower temporal loss weight; keep residual branch active; reduce context depth.

4. **Memory blow-up**
   - Do not backprop through all 500 frames.
   - Use short sequences and log-spaced lags.

5. **Teacher drift**
   - Keep EMA high, e.g. 0.996–0.999.
   - Use stop-grad targets only.

---

## Implementation order (strict)

### PR 1 — skeleton
- add module files
- add config
- add train entry
- reuse current datamodule
- verify forward on one batch

### PR 2 — stable motifs + multi-lag prediction
- full sequence encoding
- teacher targets
- stable bank
- residual branch
- future prediction
- hazard head
- train/val runs

### PR 3 — bridge mining
- inference refresh utility
- candidate export
- batch flag ingestion
- bridge head + losses
- logs

### PR 4 — optional field head
- neighbor-aware inputs
- field loss
- ablation flags

---

## Acceptance criteria

A PR is done only if all of these pass.

### Functional
- training runs on one GPU and multi-GPU
- checkpoint save/load works
- encoder-only init works
- batch with `points` only still works
- no collapse after 3–5 epochs on a small debug run

### Metrics sanity
- stable prototype usage > 30% of prototypes active after warmup
- future motif prediction beats random baseline at lag 1
- bridge candidate fraction stays below configured ceiling
- hazard positive fraction roughly matches empirical change frequency

### Outputs
- train/val metrics logged to WandB / Lightning logger
- inference returns anchor latent plus optional dynamic outputs for evaluation hooks

---

## Command-line target

Add a train entry mirroring the current temporal training script pattern:

```bash
python src/training_methods/temporal_motif_field/train_temporal_motif_field.py
```

It should call the generic trainer with the new module and config.

---

## Notes for the agent

- Prefer simple working losses over complex elegant losses.
- Keep all optional branches behind config flags.
- Use teacher stop-grad targets everywhere a target is derived from current embeddings.
- Do not add raw coordinate reconstruction.
- Keep bridge mining refresh-based in v1.
