from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import StateConfig, TransitionConfig


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: float


class TransitionGraph:
    def __init__(
        self,
        states: list[StateConfig],
        transitions: list[TransitionConfig],
        primary_path: list[str] | None = None,
    ) -> None:
        self.states = list(states)
        self.state_names = [state.name for state in states]
        self.state_to_idx = {name: idx for idx, name in enumerate(self.state_names)}
        if len(self.state_to_idx) != len(self.state_names):
            raise ValueError(f"State names must be unique, got {self.state_names}.")

        self.state_map = {state.name: state for state in states}
        self.primary_path = list(primary_path or [])
        for state_name in self.primary_path:
            if state_name not in self.state_to_idx:
                raise ValueError(
                    f"Primary path references unknown state {state_name!r}; known states are {self.state_names}."
                )
        self.primary_position = {state_name: idx for idx, state_name in enumerate(self.primary_path)}

        self.outgoing: dict[str, list[GraphEdge]] = {state_name: [] for state_name in self.state_names}
        for transition in transitions:
            if transition.source not in self.state_to_idx:
                raise ValueError(
                    f"Transition source {transition.source!r} is unknown; known states are {self.state_names}."
                )
            if transition.target not in self.state_to_idx:
                raise ValueError(
                    f"Transition target {transition.target!r} is unknown; known states are {self.state_names}."
                )
            if transition.weight <= 0.0:
                raise ValueError(
                    f"Transition {transition.source}->{transition.target} must have positive weight, "
                    f"got {transition.weight}."
                )
            self.outgoing[transition.source].append(
                GraphEdge(source=transition.source, target=transition.target, weight=float(transition.weight))
            )

    def index(self, state_name: str) -> int:
        if state_name not in self.state_to_idx:
            raise KeyError(f"Unknown state {state_name!r}; known states are {self.state_names}.")
        return self.state_to_idx[state_name]

    def name(self, state_idx: int) -> str:
        if state_idx < 0 or state_idx >= len(self.state_names):
            raise IndexError(f"State index {state_idx} is out of range for {len(self.state_names)} states.")
        return self.state_names[state_idx]

    def state_config(self, state_name: str) -> StateConfig:
        if state_name not in self.state_map:
            raise KeyError(f"Unknown state {state_name!r}; known states are {self.state_names}.")
        return self.state_map[state_name]

    def outgoing_edges(self, state_name: str) -> list[GraphEdge]:
        if state_name not in self.outgoing:
            raise KeyError(f"Unknown state {state_name!r}; known states are {self.state_names}.")
        return self.outgoing[state_name]

    def has_state(self, state_name: str | None) -> bool:
        return state_name is not None and state_name in self.state_to_idx

    def progress_position_for_name(self, state_name: str) -> int | None:
        return self.primary_position.get(state_name)

    def progress_position_for_index(self, state_idx: int) -> int | None:
        return self.progress_position_for_name(self.name(state_idx))

    def is_valid_transition(self, source_idx: int, target_idx: int) -> bool:
        if source_idx == target_idx:
            return True
        source_name = self.name(source_idx)
        target_name = self.name(target_idx)
        return any(edge.target == target_name for edge in self.outgoing_edges(source_name))

    def choose_initial_state(self, initial_probs: dict[str, float], rng: np.random.Generator) -> int:
        if not initial_probs:
            return 0
        states = []
        probs = []
        for state_name, prob in initial_probs.items():
            if state_name not in self.state_to_idx:
                raise ValueError(
                    f"Initial state distribution references unknown state {state_name!r}; "
                    f"known states are {self.state_names}."
                )
            if prob < 0.0:
                raise ValueError(f"Initial probability for state {state_name!r} must be non-negative, got {prob}.")
            if prob > 0.0:
                states.append(self.index(state_name))
                probs.append(float(prob))
        if not states:
            raise ValueError("Initial state distribution has no positive probabilities.")
        probabilities = np.asarray(probs, dtype=np.float64)
        probabilities = probabilities / np.sum(probabilities)
        sampled = int(rng.choice(states, p=probabilities))
        return sampled

    def serializable(self) -> dict[str, object]:
        return {
            "state_names": self.state_names,
            "primary_path": self.primary_path,
            "transitions": [
                {"source": edge.source, "target": edge.target, "weight": edge.weight}
                for state_name in self.state_names
                for edge in self.outgoing[state_name]
            ],
        }
