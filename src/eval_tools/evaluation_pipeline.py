from __future__ import annotations

"""High level evaluation pipeline utilities.

This module provides a thin orchestrator that wires together the pieces
required for evaluating a trained model.  The :class:`EvaluationPipeline`
loads configuration files, restores the trained model, prepares a dataset
and finally runs prediction and optional experiment routines specified in
an evaluation configuration.

Example
-------
>>> EvaluationPipeline.run(
...     "checkpoint.ckpt",
...     "configs/autoencoder_80.yaml",
...     "configs/eval_configs/benchmark.yml",
... )
"""

import inspect
import os
import sys
from typing import Any, Dict, List

from omegaconf import OmegaConf

# Ensure that ``src`` is importable when this module is executed as a script.
sys.path.append(os.getcwd())

from .predict_functions import (  # noqa: E402  -- imported after sys.path tweak
    _get_latents_from_dataloader,
    load_model_for_inference,
)


class EvaluationPipeline:
    """Orchestrates loading configs, models, data and experiments."""

    @staticmethod
    def run(
        checkpoint_path: str,
        train_cfg_path: str,
        eval_cfg_path: str,
    ) -> Dict[str, Any]:
        """Run the evaluation pipeline.

        Parameters
        ----------
        checkpoint_path:
            Path to the model checkpoint.
        train_cfg_path:
            YAML file describing the training configuration.  This is used
            primarily to determine the model type.
        eval_cfg_path:
            YAML file describing the evaluation set‑up.  The configuration is
            intentionally lightweight and only needs to specify the data files
            and, optionally, experiment parameters.

        Returns
        -------
        dict
            A dictionary containing the raw predictions (latents,
            reconstructions, originals and coordinates) as well as a nested
            ``experiments`` dictionary holding results from any configured
            experiments.
        """

        # ------------------------------------------------------------------
        # 1. Load configuration files
        # ------------------------------------------------------------------
        train_cfg = OmegaConf.load(train_cfg_path)
        eval_cfg = OmegaConf.load(eval_cfg_path)

        model_type = train_cfg.get("model_type", eval_cfg.get("model_type", "autoencoder"))
        cuda_device = int(eval_cfg.get("cuda_device", 0))

        # ------------------------------------------------------------------
        # 2. Restore model and prepare dataloader
        # ------------------------------------------------------------------
        model, model_cfg, device, backend, _ = load_model_for_inference(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            cuda_device=cuda_device,
        )

        # Data files may be specified with different keys; try a few common ones
        data_files: List[str] | None = (
            eval_cfg.get("data_files")
            or eval_cfg.get("files")
            or eval_cfg.get("snapshot_files")
            or eval_cfg.get("dataset", {}).get("files")
        )
        if not data_files:
            raise ValueError("No data files specified in evaluation config")
        if isinstance(data_files, str):
            data_files = [data_files]

        loader_kwargs = {
            "shuffle": False,
            "max_samples": eval_cfg.get("max_samples"),
            "return_coords": True,
        }
        batch_size = eval_cfg.get("batch_size")
        if batch_size is not None and "batch_size" in inspect.signature(
            backend.create_dataloader
        ).parameters:
            loader_kwargs["batch_size"] = batch_size

        dataloader = backend.create_dataloader(model_cfg, data_files, **loader_kwargs)

        # ------------------------------------------------------------------
        # 3. Run prediction
        # ------------------------------------------------------------------
        latents, reconstructions, originals, coords = _get_latents_from_dataloader(
            model, dataloader, device=device, return_coords=True
        )

        # ------------------------------------------------------------------
        # 4. Run experiments (optional)
        # ------------------------------------------------------------------
        experiment_results: Dict[str, Any] = {}
        experiments_cfg = eval_cfg.get("experiments")
        if experiments_cfg:
            for name, cfg in experiments_cfg.items():
                if name == "rotational_stability":
                    from .rotational_stability import run as rotational_run

                    experiment_results[name] = rotational_run(cfg)
                else:
                    raise ValueError(f"Unknown experiment '{name}'")

        return {
            "latents": latents,
            "reconstructions": reconstructions,
            "originals": originals,
            "coords": coords,
            "experiments": experiment_results,
        }


__all__ = ["EvaluationPipeline"]
