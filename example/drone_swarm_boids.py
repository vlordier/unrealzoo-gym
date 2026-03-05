import argparse
import json
from pathlib import Path

from gym_unrealcv.swarm import BoidsConfig, DroneSwarmBoidsRunner, SwarmRunnerConfig


def parse_waypoints(raw: str | None) -> list[list[float]] | None:
    if not raw:
        return None

    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text())

    parsed: list[list[float]] = []
    for block in raw.split(';'):
        values = [float(x.strip()) for x in block.split(',') if x.strip()]
        if len(values) != 3:
            raise ValueError('Each waypoint must be x,y,z')
        parsed.append(values)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description='Run N-drone Boids swarms in UnrealZoo')
    parser.add_argument('--env-id', default='UnrealNavigationMulti-SuburbNeighborhood_Day-ContinuousColor-v0')
    parser.add_argument('--n', type=int, default=10, help='Number of drones (e.g., 5/10/30/100)')
    parser.add_argument('--mode', choices=['gamepad', 'planned_path', 'mission'], default='mission')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--offscreen', action='store_true')

    parser.add_argument('--low-height', type=float, default=1.8)
    parser.add_argument('--mission-center', default='0.0,0.0', help='x,y center for mission mode')
    parser.add_argument('--mission-radius', type=float, default=12.0)
    parser.add_argument('--mission-angular-speed', type=float, default=0.35)

    parser.add_argument('--waypoints', default=None, help='Path to JSON or inline x,y,z;x,y,z')

    parser.add_argument('--max-speed', type=float, default=3.0)
    parser.add_argument('--max-force', type=float, default=2.0)
    parser.add_argument('--neighbor-radius', type=float, default=8.0)
    parser.add_argument('--separation-radius', type=float, default=2.0)
    parser.add_argument('--alignment-weight', type=float, default=0.8)
    parser.add_argument('--cohesion-weight', type=float, default=0.5)
    parser.add_argument('--separation-weight', type=float, default=1.4)
    parser.add_argument('--target-weight', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.1)

    args = parser.parse_args()

    center_x, center_y = [float(v.strip()) for v in args.mission_center.split(',')]
    planned = parse_waypoints(args.waypoints)

    run_cfg = SwarmRunnerConfig(
        env_id=args.env_id,
        n_agents=args.n,
        mode=args.mode,
        steps=args.steps,
        dt=args.dt,
        offscreen=args.offscreen,
        seed=args.seed,
        low_height=args.low_height,
        planned_waypoints=None if planned is None else planned,
        mission_center_xy=(center_x, center_y),
        mission_radius=args.mission_radius,
        mission_angular_speed=args.mission_angular_speed,
    )

    boids_cfg = BoidsConfig(
        max_speed=args.max_speed,
        max_force=args.max_force,
        neighbor_radius=args.neighbor_radius,
        separation_radius=args.separation_radius,
        alignment_weight=args.alignment_weight,
        cohesion_weight=args.cohesion_weight,
        separation_weight=args.separation_weight,
        target_weight=args.target_weight,
        dt=args.dt,
    )

    runner = DroneSwarmBoidsRunner(config=run_cfg, boids_config=boids_cfg)
    try:
        result = runner.run()
        print('Swarm run completed:', result)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
