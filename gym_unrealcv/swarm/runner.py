from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gym  # type: ignore[import-untyped]
import numpy as np

from gym_unrealcv.envs.wrappers.configUE import ConfigUEWrapper
from gym_unrealcv.swarm.boids import BoidsConfig, boids_step
from gym_unrealcv.swarm.input import GamepadTeleop
from gym_unrealcv.swarm.missions import PlannedPathController, StreetCircleMission


Mode = Literal['gamepad', 'planned_path', 'mission']


@dataclass
class SwarmRunnerConfig:
    env_id: str
    n_agents: int
    mode: Mode = 'mission'
    steps: int = 1000
    dt: float = 0.1
    offscreen: bool = True
    resolution: tuple[int, int] = (320, 240)
    seed: int = 0
    low_height: float = 1.8
    planned_waypoints: np.ndarray | None = None
    mission_center_xy: tuple[float, float] = (0.0, 0.0)
    mission_radius: float = 12.0
    mission_angular_speed: float = 0.35
    base_speed: float = 2.5


class DroneSwarmBoidsRunner:
    def __init__(self, config: SwarmRunnerConfig, boids_config: BoidsConfig | None = None) -> None:
        self.cfg = config
        self.boids = boids_config or BoidsConfig(dt=config.dt)
        self.gamepad = GamepadTeleop()

        self.env = gym.make(config.env_id)
        self.env = ConfigUEWrapper(
            self.env,
            offscreen=config.offscreen,
            resolution=config.resolution,
        )

        self.env.seed(config.seed)
        self.obs = self.env.reset()
        self._ensure_drone_population(config.n_agents)
        self.obs = self.env.reset()

        self.drone_indices = self._get_drone_indices()
        if len(self.drone_indices) < config.n_agents:
            raise RuntimeError(
                f'Only {len(self.drone_indices)} drones are available, expected {config.n_agents}. '
                'Please use a map/task config with enough drone agents.'
            )

        self.drone_indices = self.drone_indices[: config.n_agents]
        self.prev_positions = self._get_positions_from_env()
        self.velocities = np.zeros_like(self.prev_positions)

        self.path_controller = None
        if config.mode == 'planned_path':
            waypoints = config.planned_waypoints
            if waypoints is None:
                center = self.prev_positions[:, :2].mean(axis=0)
                z = config.low_height
                waypoints = np.array(
                    [
                        [center[0] + 20, center[1] + 0, z],
                        [center[0] + 20, center[1] + 20, z],
                        [center[0] + 0, center[1] + 20, z],
                        [center[0] + 0, center[1] + 0, z],
                    ],
                    dtype=np.float32,
                )
            self.path_controller = PlannedPathController(waypoints=waypoints)

        self.street_mission = StreetCircleMission(
            center_xy=np.asarray(config.mission_center_xy, dtype=np.float32),
            radius=config.mission_radius,
            angular_speed=config.mission_angular_speed,
            low_height=config.low_height,
        )

    def _get_drone_indices(self) -> list[int]:
        env = self.env.unwrapped
        out: list[int] = []
        for idx, obj in enumerate(env.player_list):
            agent_type = env.agents[obj]['agent_type']
            if agent_type == 'drone':
                out.append(idx)
        return out

    def _ensure_drone_population(self, n_agents: int) -> None:
        env = self.env.unwrapped
        drone_objs = [obj for obj in env.player_list if env.agents[obj]['agent_type'] == 'drone']
        if not drone_objs:
            raise RuntimeError('No drone agents found in the selected environment/task config.')

        base_drone = drone_objs[0]
        refer = env.agents[base_drone]

        while len([obj for obj in env.player_list if env.agents[obj]['agent_type'] == 'drone']) < n_agents:
            name = f'drone_swarm_{len(env.player_list)}'
            loc = env.sample_init_pose(use_reset_area=env.random_init, num_agents=1)[0]
            env.agents[name] = env.add_agent(name, loc, refer)

    def _get_positions_from_env(self) -> np.ndarray:
        env = self.env.unwrapped
        poses = np.asarray(env.obj_poses, dtype=np.float32)
        drone_poses = poses[self.drone_indices]
        return drone_poses[:, :3]

    def _target_velocities(self, positions: np.ndarray) -> np.ndarray:
        if self.cfg.mode == 'gamepad':
            gp = self.gamepad.poll()
            base = np.array([gp.forward, gp.right, gp.up], dtype=np.float32) * self.cfg.base_speed
            targets = np.repeat(base[None, :], len(self.drone_indices), axis=0)
            targets[:, 2] += (self.cfg.low_height - positions[:, 2])
            return targets

        if self.cfg.mode == 'planned_path':
            assert self.path_controller is not None
            center = positions.mean(axis=0)
            base = self.path_controller.velocity_command(center)
            targets = np.repeat(base[None, :], len(self.drone_indices), axis=0)
            targets[:, 2] += (self.cfg.low_height - positions[:, 2])
            return targets

        self.street_mission.step(self.cfg.dt)
        return self.street_mission.desired_velocities(positions)

    def _to_action(self, agent_idx: int, desired_velocity: np.ndarray) -> np.ndarray:
        action_space = self.env.action_space[agent_idx]
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        dim = int(low.shape[0])

        action = np.zeros(dim, dtype=np.float32)
        if dim >= 4:
            horizontal = desired_velocity[:2]
            yaw_rate = float(np.arctan2(horizontal[1], horizontal[0]))
            action[:4] = np.array([desired_velocity[0], desired_velocity[1], desired_velocity[2], yaw_rate], dtype=np.float32)
        elif dim == 3:
            action[:] = np.array([desired_velocity[0], desired_velocity[1], desired_velocity[2]], dtype=np.float32)
        elif dim == 2:
            yaw_rate = float(np.arctan2(desired_velocity[1], desired_velocity[0]))
            speed = float(np.linalg.norm(desired_velocity[:2]))
            action[:] = np.array([yaw_rate, speed], dtype=np.float32)
        elif dim == 1:
            action[0] = float(np.linalg.norm(desired_velocity[:2]))

        return np.clip(action, low, high)

    def run(self) -> dict[str, float]:
        total_reward = 0.0
        steps = 0

        for _ in range(self.cfg.steps):
            positions = self._get_positions_from_env()
            self.velocities = (positions - self.prev_positions) / max(self.cfg.dt, 1e-6)
            targets = self._target_velocities(positions)
            desired = boids_step(positions, self.velocities, targets, self.boids)

            actions: list[object] = [None] * len(self.env.unwrapped.player_list)
            for local_idx, env_idx in enumerate(self.drone_indices):
                actions[env_idx] = self._to_action(env_idx, desired[local_idx])

            obs, rewards, done, _info = self.env.step(actions)
            self.obs = obs
            self.prev_positions = positions
            total_reward += float(np.sum(rewards))
            steps += 1

            if done:
                break

        return {
            'steps': float(steps),
            'reward_sum': total_reward,
        }

    def close(self) -> None:
        self.env.close()
