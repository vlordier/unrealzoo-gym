from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlannedPathController:
    waypoints: np.ndarray
    reach_threshold: float = 2.0
    cruise_speed: float = 2.0
    loop: bool = True

    def __post_init__(self) -> None:
        points = np.asarray(self.waypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('waypoints must be an array with shape [N, 3]')
        if points.shape[0] < 2:
            raise ValueError('need at least two waypoints for planned path mode')
        self.waypoints = points
        self.current = 0

    def velocity_command(self, center: np.ndarray) -> np.ndarray:
        target = self.waypoints[self.current]
        error = target - center
        distance = float(np.linalg.norm(error))

        if distance < self.reach_threshold:
            if self.current < len(self.waypoints) - 1:
                self.current += 1
            elif self.loop:
                self.current = 0

            target = self.waypoints[self.current]
            error = target - center
            distance = float(np.linalg.norm(error))

        if distance < 1e-6:
            return np.zeros(3, dtype=np.float32)

        return (error / distance) * self.cruise_speed


@dataclass
class StreetCircleMission:
    center_xy: np.ndarray
    radius: float = 8.0
    angular_speed: float = 0.35
    low_height: float = 1.8
    altitude_gain: float = 1.2
    radial_gain: float = 0.6
    tangent_speed: float = 2.0

    def __post_init__(self) -> None:
        center_xy = np.asarray(self.center_xy, dtype=np.float32)
        if center_xy.shape != (2,):
            raise ValueError('center_xy must be shape (2,)')
        self.center_xy = center_xy
        self.phase = 0.0

    def step(self, dt: float) -> None:
        self.phase += self.angular_speed * dt

    def desired_velocities(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        n_agents = positions.shape[0]
        out = np.zeros_like(positions)

        for idx in range(n_agents):
            agent_phase = self.phase + 2 * np.pi * idx / max(n_agents, 1)
            ring_target = np.array(
                [
                    self.center_xy[0] + self.radius * np.cos(agent_phase),
                    self.center_xy[1] + self.radius * np.sin(agent_phase),
                ],
                dtype=np.float32,
            )
            radial_error = ring_target - positions[idx, :2]

            tangent = np.array([-np.sin(agent_phase), np.cos(agent_phase)], dtype=np.float32)
            horizontal = self.radial_gain * radial_error + self.tangent_speed * tangent
            vertical = self.altitude_gain * (self.low_height - positions[idx, 2])
            out[idx] = np.array([horizontal[0], horizontal[1], vertical], dtype=np.float32)

        return out
