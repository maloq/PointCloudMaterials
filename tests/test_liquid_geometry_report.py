"""Forecast association reports preserve source weighting and tied-score AP."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logit

from src.research.liquid_geometry.study import MODES, report, sha, table


def test_report_ap_bootstrap_and_empty_domain(tmp_path):
    root = tmp_path / "result"
    expanded = tmp_path / "expanded"
    (root / "technical").mkdir(parents=True)
    (expanded / "technical/assay").mkdir(parents=True)
    source = np.repeat(np.arange(4), [2, 3, 4, 5])
    n = len(source)
    event = np.full(n, 5)
    event[[0, 2, 5, 9]] = 2
    role = np.full(n, "test")
    np.savez(expanded / "technical/assay/population.npz", role=role)
    np.savez(root / "technical/population.npz", role=role, source=source, event=event,
             indices=np.arange(n), temperature=np.where(source < 2, 400., 500.))
    models = [dict(name=f"cold-{i}", domain="cold") for i in range(4)]
    config = dict(expanded_root=str(expanded), models=models + [dict(name="pending-hot", domain="hot")],
                  seed=13, bootstrap_draws=20)
    inputs = {}
    source_metric_names = ("neighbor_order_mse", "future_neighbor_mse", "metric_untrained_topology_mse",
                           "knn_brier", "knn_logloss", "rank", "neighbor_order_gain", "physical_probe_mse", "future_probe_mse")
    for i, model in enumerate(models):
        name = model["name"]
        status = root / "technical/models" / name / "status.json"
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps(dict(state="complete")))
        rows = [dict(model=name, domain="cold", mode=mode, source=s,
                     **{m: 0.1 + (i + 1) * (s + 1) / 40 for m in source_metric_names})
                for mode in MODES for s in range(4)]
        table(root / "tables" / f"{name}-sources.csv", rows)
        summary = [dict(model=name, domain="cold", mode=mode, conditional_rank=i + 1,
                        neighbor_order_gain=.1 * (i + 1), future_neighbor_mse=.4,
                        physical_probe_gain=.2, future_probe_gain=.05,
                        metric_untrained_topology_mse=.3, knn_brier=.2) for mode in MODES]
        table(root / "tables" / f"{name}-summary.csv", summary)
        for head in ("linear", "mlp"):
            folder = expanded / "readouts/technical/fits" / name / "snapshot" / head
            folder.mkdir(parents=True)
            # Repeated probabilities exercise score-group endpoints; unequal
            # source sizes make uniform-row bootstrap weighting incorrect.
            probability = np.resize(np.array([.1, .4, .4, .7]), n)
            probability = np.roll(probability, i)
            prediction = np.full((n, 5), -100.)
            prediction[:, 0] = logit(probability)
            path = folder / "predictions.npz"
            np.savez(path, test=prediction, test_indices=np.arange(n))
            inputs[str(path)] = sha(path)
    (root / "technical/identity.json").write_text(json.dumps(dict(
        config=config, inputs=inputs, population_sha256=sha(root / "technical/population.npz"))))
    report(config, root)
    with (root / "tables/forecast-associations.csv").open() as stream:
        associations = list(csv.DictReader(stream))
    assert {r["forecast_metric"] for r in associations} == {
        "matched_12ps_ap", "matched_12ps_brier", "matched_12ps_log_loss"}
    assert {r["domain"] for r in associations} == {"all", "cold"}
    assert all(int(r["valid_draws"]) <= 20 for r in associations)
    assert json.loads((root / "technical/report-status.json").read_text())["missing"] == ["pending-hot"]
