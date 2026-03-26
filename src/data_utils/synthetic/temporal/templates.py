from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DomainConfig, RenderingConfig, StateConfig


@dataclass(frozen=True)
class StatePhaseRecipe:
    state_name: str
    recipe: dict[str, Any]


class TemplateLibrary:
    """
    Temporal rendering recipe library.

    The temporal benchmark keeps latent site/state dynamics as its core
    abstraction, but frame rendering should use the same style of full-box
    space-filling phase recipes as the static atomistic generator.
    """

    def __init__(
        self,
        domain: DomainConfig,
        rendering: RenderingConfig,
        states: list[StateConfig],
        seed: int,
    ) -> None:
        self.domain = domain
        self.rendering = rendering
        self.seed = int(seed)
        self.avg_nn_distance = float(domain.avg_nn_distance)
        self.target_density = float(rendering.target_density)
        self.default_crystal_structure = str(rendering.default_crystal_structure)
        self._recipes: dict[str, StatePhaseRecipe] = {}
        self._state_cfg_map: dict[str, StateConfig] = {}

        for state in states:
            if state.name in self._state_cfg_map:
                raise ValueError(f"Duplicate state definition for {state.name!r}.")
            self._state_cfg_map[state.name] = state
            self._recipes[state.name] = StatePhaseRecipe(
                state_name=state.name,
                recipe=self._build_phase_recipe(state),
            )

    def state_config(self, state_name: str) -> StateConfig:
        if state_name not in self._state_cfg_map:
            raise KeyError(f"Unknown state {state_name!r}.")
        return self._state_cfg_map[state_name]

    def phase_recipe(self, state_name: str) -> dict[str, Any]:
        if state_name not in self._recipes:
            raise KeyError(f"No rendering recipe registered for state {state_name!r}.")
        recipe = self._recipes[state_name].recipe
        copied: dict[str, Any] = {}
        for key, value in recipe.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, dict):
                copied[key] = dict(value)
            else:
                copied[key] = value
        return copied

    def crystal_variant_recipe(self, state_name: str, variant_id: int) -> dict[str, Any]:
        state_cfg = self.state_config(state_name)
        if state_cfg.template_kind != "crystal":
            raise ValueError(
                f"Crystal variant recipes are only defined for template_kind='crystal', got "
                f"template_kind={state_cfg.template_kind!r} for state {state_name!r}."
            )
        if variant_id == 0:
            return self.phase_recipe(state_name)
        if variant_id != 1:
            raise ValueError(f"Unsupported crystal variant_id={variant_id} for state {state_name!r}.")
        base_structure = state_cfg.crystal_structure or self.default_crystal_structure
        if base_structure == "fcc":
            return self._crystal_recipe(structure="hcp", lattice_scale=1.0)
        return self._crystal_recipe(structure=base_structure, lattice_scale=1.03)

    def _build_phase_recipe(self, state: StateConfig) -> dict[str, Any]:
        kind = state.template_kind
        if kind == "liquid":
            return self._liquid_recipe(state)
        if kind == "precursor":
            return self._precursor_recipe(state, interface_like=False)
        if kind == "interface":
            return self._interface_recipe(state)
        if kind == "crystal":
            return self._crystal_recipe_for_state(state)
        if kind == "defective_crystal":
            return self._crystal_recipe_for_state(state)
        if kind == "grain_boundary":
            return self._grain_boundary_recipe(state)
        raise ValueError(
            f"Unsupported template_kind={kind!r} for temporal full-box rendering of state {state.name!r}."
        )

    def _crystal_recipe_for_state(self, state: StateConfig) -> dict[str, Any]:
        structure = state.crystal_structure or self.default_crystal_structure
        return self._crystal_recipe(structure=structure, lattice_scale=1.0)

    def _crystal_recipe(self, *, structure: str, lattice_scale: float) -> dict[str, Any]:
        if structure == "fcc":
            lattice_constant = lattice_scale * self.avg_nn_distance * math.sqrt(2.0)
            motif = np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
                dtype=np.float32,
            )
            return {
                "phase_type": "crystal_fcc",
                "lattice_constant": lattice_constant,
                "lattice_vectors": np.eye(3, dtype=np.float32) * lattice_constant,
                "motif": motif,
            }
        if structure == "bcc":
            lattice_constant = lattice_scale * (2.0 / math.sqrt(3.0)) * self.avg_nn_distance
            motif = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
            return {
                "phase_type": "crystal_bcc",
                "lattice_constant": lattice_constant,
                "lattice_vectors": np.eye(3, dtype=np.float32) * lattice_constant,
                "motif": motif,
            }
        if structure == "hcp":
            a = lattice_scale * self.avg_nn_distance
            c = lattice_scale * self.avg_nn_distance * math.sqrt(8.0 / 3.0)
            lattice_vectors = np.array(
                [[a, 0.0, 0.0], [0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0], [0.0, 0.0, c]],
                dtype=np.float32,
            )
            motif = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]], dtype=np.float32)
            return {
                "phase_type": "crystal_hcp",
                "lattice_constant": a,
                "lattice_vectors": lattice_vectors,
                "motif": motif,
            }
        raise ValueError(
            f"Unsupported crystal structure {structure!r}. Expected one of 'fcc', 'bcc', or 'hcp'."
        )

    def _liquid_recipe(self, state: StateConfig) -> dict[str, Any]:
        min_pair_distance = float(state.template_params.get("min_pair_distance", 0.82 * self.avg_nn_distance))
        liquid_method = str(state.template_params.get("liquid_method", "simple"))
        liquid_config = {
            "method": liquid_method,
            "target_coordination": float(state.template_params.get("target_coordination", 12.0)),
            "rdf_iterations": int(state.template_params.get("rdf_iterations", 200)),
            "rdf_tolerance": float(state.template_params.get("rdf_tolerance", 0.08)),
        }
        return {
            "phase_type": "liquid_metal",
            "min_pair_dist": min_pair_distance,
            "liquid_config": liquid_config,
        }

    def _precursor_recipe(self, state: StateConfig, *, interface_like: bool) -> dict[str, Any]:
        structure = state.crystal_structure or self.default_crystal_structure
        min_pair_distance = float(state.template_params.get("min_pair_distance", 0.82 * self.avg_nn_distance))
        ordered_fraction = float(state.template_params.get("ordered_fraction", 0.60))
        if interface_like:
            default_embed = 0.16 + 0.18 * ordered_fraction
            default_radius = 2.4 * self.avg_nn_distance
        else:
            default_embed = 0.06 + 0.16 * ordered_fraction
            default_radius = 1.8 * self.avg_nn_distance
        embedded_probability = float(state.template_params.get("embedded_probability", default_embed))
        embedded_radius = float(state.template_params.get("embedded_radius", default_radius))
        liquid_method = str(state.template_params.get("liquid_method", "simple"))
        liquid_config = {
            "method": liquid_method,
            "target_coordination": float(state.template_params.get("target_coordination", 11.2)),
            "rdf_iterations": int(state.template_params.get("rdf_iterations", 150)),
            "rdf_tolerance": float(state.template_params.get("rdf_tolerance", 0.10)),
        }
        return {
            "phase_type": "amorphous_mixed",
            "min_pair_dist": min_pair_distance,
            "embedded_crystal": f"crystal_{structure}",
            "embedded_probability": embedded_probability,
            "embedded_radius": embedded_radius,
            "liquid_config": liquid_config,
        }

    def _interface_recipe(self, state: StateConfig) -> dict[str, Any]:
        structure = state.crystal_structure or self.default_crystal_structure
        min_pair_distance = float(state.template_params.get("min_pair_distance", 0.82 * self.avg_nn_distance))
        liquid_method = str(state.template_params.get("liquid_method", "simple"))
        liquid_config = {
            "method": liquid_method,
            "target_coordination": float(state.template_params.get("target_coordination", 11.6)),
            "rdf_iterations": int(state.template_params.get("rdf_iterations", 180)),
            "rdf_tolerance": float(state.template_params.get("rdf_tolerance", 0.09)),
        }
        return {
            "phase_type": "surface_interface",
            "liquid_recipe": {
                "phase_type": "liquid_metal",
                "min_pair_dist": min_pair_distance,
                "liquid_config": liquid_config,
            },
            "solid_recipe": self._crystal_recipe_for_state(
                StateConfig(
                    name=state.name,
                    template_kind="crystal",
                    crystal_structure=structure,
                    template_params=state.template_params,
                    base_thermal_jitter=state.base_thermal_jitter,
                    base_defect_amplitude=state.base_defect_amplitude,
                    base_strain_scale=state.base_strain_scale,
                    metastable=state.metastable,
                    grain_bearing=state.grain_bearing,
                    dwell=state.dwell,
                )
            ),
            "interface_width": float(state.template_params.get("interface_width", 1.8 * self.avg_nn_distance)),
            "solid_fraction_start": float(state.template_params.get("solid_fraction_start", 0.18)),
            "solid_fraction_end": float(state.template_params.get("solid_fraction_end", 0.86)),
            "plane_jitter": float(state.template_params.get("plane_jitter", 0.20 * self.avg_nn_distance)),
        }

    def _grain_boundary_recipe(self, state: StateConfig) -> dict[str, Any]:
        structure = state.crystal_structure or self.default_crystal_structure
        return {
            "phase_type": "grain_boundary_bicrystal",
            "base_solid_recipe": self._crystal_recipe(structure=structure, lattice_scale=1.0),
            "boundary_width": float(state.template_params.get("boundary_width", 1.4 * self.avg_nn_distance)),
            "plane_jitter": float(state.template_params.get("plane_jitter", 0.18 * self.avg_nn_distance)),
        }
