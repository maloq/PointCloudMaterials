#!/usr/bin/env python3
"""Compare one full-size FP32 and BF16-autocast MACE force/stress evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from src.data_utils.synthetic.atomistic.config import potential_calculator_settings
from src.data_utils.synthetic.atomistic.generator import build_calculator
from src.data_utils.synthetic.atomistic.homogeneous_campaign_config import (
    load_homogeneous_campaign_config,
)
from src.data_utils.synthetic.atomistic.homogeneous_generator import (
    _load_source_liquid,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--maximum-energy-difference-meV-per-atom", type=float, default=10.0
    )
    parser.add_argument(
        "--maximum-force-rmse-eV-per-A", type=float, default=0.01
    )
    parser.add_argument(
        "--maximum-stress-difference-GPa", type=float, default=0.25
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-blocks", type=int, default=3)
    parser.add_argument("--steps-per-block", type=int, default=5)
    parser.add_argument(
        "--fcc-repetitions",
        type=int,
        default=None,
        help="Use a cubic FCC Al smoke-test cell instead of the full campaign source.",
    )
    return parser.parse_args()


def _cuda_memory(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _evaluate(
    initial_atoms,
    generator,
    *,
    label: str,
    warmup_steps: int,
    measurement_blocks: int,
    steps_per_block: int,
) -> dict[str, object]:
    atoms = initial_atoms.copy()
    initialization_start = time.perf_counter()
    calculator = build_calculator(generator)
    torch.cuda.synchronize(calculator.device)
    initialization_seconds = time.perf_counter() - initialization_start
    initialization_memory = _cuda_memory(calculator.device)

    torch.cuda.reset_peak_memory_stats(calculator.device)
    evaluation_start = time.perf_counter()
    calculator.calculate(
        atoms,
        properties=["forces", "stress"],
        system_changes=all_changes,
    )
    torch.cuda.synchronize(calculator.device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    evaluation_memory = _cuda_memory(calculator.device)

    energy_eV = float(calculator.results["energy"])
    forces_eV_per_A = np.asarray(calculator.results["forces"], dtype=np.float64)
    stress_eV_per_A3 = np.asarray(calculator.results["stress"], dtype=np.float64)
    if forces_eV_per_A.shape != (len(atoms), 3):
        raise RuntimeError(
            f"{label}: force result has shape={forces_eV_per_A.shape}, "
            f"expected={(len(atoms), 3)}."
        )
    if stress_eV_per_A3.shape != (6,):
        raise RuntimeError(
            f"{label}: stress result has shape={stress_eV_per_A3.shape}, expected=(6,)."
        )
    if (
        not math.isfinite(energy_eV)
        or not np.all(np.isfinite(forces_eV_per_A))
        or not np.all(np.isfinite(stress_eV_per_A3))
    ):
        raise RuntimeError(f"{label}: MACE returned non-finite energy, forces, or stress.")

    atoms.calc = calculator
    rng = np.random.default_rng(84621)
    MaxwellBoltzmannDistribution(atoms, temperature_K=500.0, rng=rng)
    Stationary(atoms, preserve_temperature=True)
    dynamics = IsotropicMTKNPT(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=500.0,
        pressure_au=0.0 * units.GPa,
        tdamp=100.0 * units.fs,
        pdamp=500.0 * units.fs,
    )
    warmup_start = time.perf_counter()
    dynamics.run(warmup_steps)
    torch.cuda.synchronize(calculator.device)
    warmup_seconds = time.perf_counter() - warmup_start
    torch.cuda.reset_peak_memory_stats(calculator.device)
    block_seconds: list[float] = []
    for _ in range(measurement_blocks):
        block_start = time.perf_counter()
        dynamics.run(steps_per_block)
        torch.cuda.synchronize(calculator.device)
        block_seconds.append(time.perf_counter() - block_start)
    benchmark_memory = _cuda_memory(calculator.device)
    measured_steps = measurement_blocks * steps_per_block
    measured_seconds = float(sum(block_seconds))
    benchmark = {
        "warmup_steps": warmup_steps,
        "warmup_seconds": warmup_seconds,
        "measurement_blocks": measurement_blocks,
        "steps_per_block": steps_per_block,
        "measured_steps": measured_steps,
        "block_seconds": block_seconds,
        "measured_seconds": measured_seconds,
        "steps_per_second": float(measured_steps / measured_seconds),
        "final_temperature_K": float(atoms.get_temperature()),
        "final_pressure_GPa": float(
            -np.trace(atoms.get_stress(voigt=False, include_ideal_gas=True))
            / 3.0
            / units.GPa
        ),
        "memory": benchmark_memory,
        "graph_cache_metrics": calculator.graph_cache_metrics(),
    }
    result = {
        "label": label,
        "settings": potential_calculator_settings(generator.potential),
        "initialization_seconds": initialization_seconds,
        "evaluation_seconds": evaluation_seconds,
        "initialization_memory": initialization_memory,
        "evaluation_memory": evaluation_memory,
        "energy_eV": energy_eV,
        "maximum_force_eV_per_A": float(
            np.max(np.linalg.norm(forces_eV_per_A, axis=1))
        ),
        "stress_GPa": (stress_eV_per_A3 / units.GPa).tolist(),
        "graph_cache_metrics": calculator.graph_cache_metrics(),
        "md_benchmark": benchmark,
        "_forces": forces_eV_per_A,
        "_stress": stress_eV_per_A3,
    }
    atoms.calc = None
    del calculator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = _arguments()
    for name in ("warmup_steps", "measurement_blocks", "steps_per_block"):
        value = getattr(args, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer.")
    if args.fcc_repetitions is not None and args.fcc_repetitions < 2:
        raise ValueError("--fcc-repetitions must be at least 2.")
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 validation requires an allocated CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The allocated CUDA GPU does not support native BF16.")

    campaign = load_homogeneous_campaign_config(args.campaign_config)
    runtime_generator = campaign.runtime_generator
    if runtime_generator is None:
        raise RuntimeError(
            f"{args.campaign_config}: BF16 validation requires runtime_generator_config."
        )
    if runtime_generator.potential.autocast_dtype != "bfloat16":
        raise RuntimeError(
            f"{runtime_generator.config_path}: expected autocast_dtype='bfloat16', got "
            f"{runtime_generator.potential.autocast_dtype!r}."
        )
    fp32_settings = potential_calculator_settings(
        campaign.homogeneous.generator.potential
    )
    bf16_settings = potential_calculator_settings(runtime_generator.potential)
    observed_autocast = bf16_settings.pop("autocast_dtype")
    observed_autocast_scope = bf16_settings.pop("autocast_scope")
    if (
        observed_autocast != "bfloat16"
        or observed_autocast_scope != "second_interaction"
        or bf16_settings != fp32_settings
    ):
        raise RuntimeError(
            "The parity evaluation must differ only by second-interaction BF16 "
            f"autocast: FP32={fp32_settings}, "
            f"BF16-without-autocast={bf16_settings}."
        )
    bf16_settings["autocast_dtype"] = observed_autocast
    bf16_settings["autocast_scope"] = observed_autocast_scope

    if args.fcc_repetitions is None:
        source = _load_source_liquid(campaign.homogeneous)
        atoms = source.atoms
        workload = {
            "kind": "campaign_liquid_source",
            "source_dataset": str(campaign.homogeneous.source_dataset),
        }
    else:
        atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat(
            (args.fcc_repetitions,) * 3
        )
        workload = {
            "kind": "fcc_smoke_test",
            "repetitions": [args.fcc_repetitions] * 3,
        }
    fp32 = _evaluate(
        atoms,
        campaign.homogeneous.generator,
        label="fp32",
        warmup_steps=args.warmup_steps,
        measurement_blocks=args.measurement_blocks,
        steps_per_block=args.steps_per_block,
    )
    bf16 = _evaluate(
        atoms,
        runtime_generator,
        label="bf16_autocast",
        warmup_steps=args.warmup_steps,
        measurement_blocks=args.measurement_blocks,
        steps_per_block=args.steps_per_block,
    )

    fp32_forces = fp32.pop("_forces")
    bf16_forces = bf16.pop("_forces")
    fp32_stress = fp32.pop("_stress")
    bf16_stress = bf16.pop("_stress")
    energy_difference_meV_per_atom = (
        abs(float(bf16["energy_eV"]) - float(fp32["energy_eV"]))
        * 1000.0
        / len(atoms)
    )
    force_difference = bf16_forces - fp32_forces
    force_rmse_eV_per_A = float(np.sqrt(np.mean(np.square(force_difference))))
    maximum_force_vector_difference_eV_per_A = float(
        np.max(np.linalg.norm(force_difference, axis=1))
    )
    maximum_stress_difference_GPa = float(
        np.max(np.abs(bf16_stress - fp32_stress)) / units.GPa
    )
    thresholds = {
        "maximum_energy_difference_meV_per_atom": (
            args.maximum_energy_difference_meV_per_atom
        ),
        "maximum_force_rmse_eV_per_A": args.maximum_force_rmse_eV_per_A,
        "maximum_stress_difference_GPa": args.maximum_stress_difference_GPa,
    }
    comparisons = {
        "energy_difference_meV_per_atom": energy_difference_meV_per_atom,
        "force_rmse_eV_per_A": force_rmse_eV_per_A,
        "maximum_force_vector_difference_eV_per_A": (
            maximum_force_vector_difference_eV_per_A
        ),
        "maximum_stress_difference_GPa": maximum_stress_difference_GPa,
        "evaluation_peak_reserved_reduction_fraction": (
            1.0
            - int(bf16["evaluation_memory"]["peak_reserved_bytes"])
            / int(fp32["evaluation_memory"]["peak_reserved_bytes"])
        ),
        "md_steps_per_second_fp32": float(
            fp32["md_benchmark"]["steps_per_second"]
        ),
        "md_steps_per_second_bf16": float(
            bf16["md_benchmark"]["steps_per_second"]
        ),
        "md_speedup": float(
            bf16["md_benchmark"]["steps_per_second"]
            / fp32["md_benchmark"]["steps_per_second"]
        ),
        "md_peak_reserved_reduction_fraction": (
            1.0
            - int(bf16["md_benchmark"]["memory"]["peak_reserved_bytes"])
            / int(fp32["md_benchmark"]["memory"]["peak_reserved_bytes"])
        ),
    }
    passed = (
        energy_difference_meV_per_atom
        <= thresholds["maximum_energy_difference_meV_per_atom"]
        and force_rmse_eV_per_A <= thresholds["maximum_force_rmse_eV_per_A"]
        and maximum_stress_difference_GPa
        <= thresholds["maximum_stress_difference_GPa"]
    )
    report = {
        "schema_version": 1,
        "campaign_config": str(args.campaign_config.resolve()),
        "atom_count": len(atoms),
        "workload": workload,
        "device": {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "thresholds": thresholds,
        "comparisons": comparisons,
        "fp32": fp32,
        "bf16_autocast": bf16,
        "passed": passed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError(
            "BF16 parity exceeded the explicit safety thresholds; full campaigns were "
            f"not authorized. See {args.output_json}."
        )


if __name__ == "__main__":
    main()
