import numpy as np

from gym_unrealcv.swarm.boids import BoidsConfig, boids_step


def test_boids_velocity_is_limited_by_max_speed():
    cfg = BoidsConfig(max_speed=1.0, max_force=10.0, dt=1.0)
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    target = np.zeros_like(velocities)

    updated = boids_step(positions, velocities, target, cfg)
    speeds = np.linalg.norm(updated, axis=1)
    assert np.all(speeds <= 1.00001)


def test_boids_moves_toward_target_velocity_when_isolated():
    cfg = BoidsConfig(
        max_speed=5.0,
        max_force=5.0,
        neighbor_radius=0.1,
        separation_radius=0.05,
        target_weight=2.0,
        alignment_weight=0.0,
        cohesion_weight=0.0,
        separation_weight=0.0,
        dt=0.2,
    )
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    updated = boids_step(positions, velocities, target, cfg)
    assert updated[0, 0] > 0.0
    assert abs(updated[0, 1]) < 1e-6
