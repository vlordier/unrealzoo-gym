from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoidsConfig:
    max_speed: float = 3.0
    max_force: float = 2.0
    neighbor_radius: float = 8.0
    separation_radius: float = 2.0
    alignment_weight: float = 0.8
    cohesion_weight: float = 0.5
    separation_weight: float = 1.4
    target_weight: float = 1.0
    dt: float = 0.1


def _limit_norm(vectors: np.ndarray, limit: float) -> np.ndarray:
    if limit <= 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.ones_like(norms)
    mask = norms > limit
    scale[mask] = limit / np.maximum(norms[mask], 1e-9)
    return vectors * scale


def boids_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    target_velocities: np.ndarray,
    config: BoidsConfig,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    target_velocities = np.asarray(target_velocities, dtype=np.float32)

    n_agents = positions.shape[0]
    if n_agents == 0:
        return velocities.copy()

    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)

    neighbor_mask = distances < config.neighbor_radius
    separation_mask = distances < config.separation_radius

    alignment = np.zeros_like(velocities)
    cohesion = np.zeros_like(velocities)
    separation = np.zeros_like(velocities)

    for agent_idx in range(n_agents):
        neighbors = np.where(neighbor_mask[agent_idx])[0]
        if neighbors.size > 0:
            alignment[agent_idx] = velocities[neighbors].mean(axis=0) - velocities[agent_idx]
            cohesion[agent_idx] = positions[neighbors].mean(axis=0) - positions[agent_idx]

        close = np.where(separation_mask[agent_idx])[0]
        if close.size > 0:
            away = positions[agent_idx] - positions[close]
            away_dist = np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-6)
            separation[agent_idx] = (away / (away_dist**2)).sum(axis=0)

    target_pull = target_velocities - velocities

    accel = (
        config.alignment_weight * alignment
        + config.cohesion_weight * cohesion
        + config.separation_weight * separation
        + config.target_weight * target_pull
    )

    accel = _limit_norm(accel, config.max_force)
    new_velocity = velocities + accel * config.dt
    new_velocity = _limit_norm(new_velocity, config.max_speed)
    return new_velocity
