# Initial Ta source branch

Question: provide the initial Ta branch and verified trajectory artifact consumed
by the Al/Mg/Ta spatiotemporal view builder. The recipe retains the existing Ta
potential, manifest and completion assumptions.

```bash
conda run -n pointnet python experiments/ta_source_20260905/ta_initial_branch.py --help
```

Use the original manifest/root arguments shown by this command; generated data
and status records are written to the selected run directory. Dataset selection
and findings are recorded in the
[spatiotemporal study](../../docs/geoframe_spatiotemporal_vicreg_20260905.md) and
[simulation context](../../docs/simulation_context_for_agents.md).
This relocation preserves the recipe; it does not generate a new branch.
