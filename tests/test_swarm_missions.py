import numpy as np

from gym_unrealcv.swarm.missions import PlannedPathController, StreetCircleMission


def test_planned_path_controller_advances_waypoint_when_close():
    controller = PlannedPathController(
        waypoints=np.array([[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]], dtype=np.float32),
        reach_threshold=0.5,
        cruise_speed=2.0,
        loop=False,
    )

    center_near_first = np.array([0.1, 0.1, 2.0], dtype=np.float32)
    velocity = controller.velocity_command(center_near_first)

    assert controller.current == 1
    assert velocity[0] > 0.0


def test_street_circle_mission_enforces_low_height_recovery():
    mission = StreetCircleMission(
        center_xy=np.array([0.0, 0.0], dtype=np.float32),
        radius=5.0,
        angular_speed=0.3,
        low_height=1.0,
        altitude_gain=1.5,
        radial_gain=0.5,
        tangent_speed=1.0,
    )

    positions = np.array(
        [
            [4.0, 0.0, 4.0],
            [0.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )

    desired = mission.desired_velocities(positions)
    assert desired.shape == positions.shape
    assert np.all(desired[:, 2] < 0.0)
