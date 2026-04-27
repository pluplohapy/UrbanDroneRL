"""
Preview scenario generation without drone flight.
Shows how obstacles are generated and positioned.

Usage:
    python visualization/preview_scenarios.py --stage pretrain --obstacle-type random --duration 5
    python visualization/preview_scenarios.py --stage 0 --duration 10
    python visualization/preview_scenarios.py --stage 1 --duration 5
"""

import argparse
import time
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import pybullet_data

from scenarios.stage0_empty import Stage0Scenario
from scenarios.stage1_static import Stage1Scenario
from scenarios.stage_pretrain import StagePretrainScenario
from config import load_config

PRETRAIN_OBSTACLE_CHOICES = (
    'random',
    'dynamic_mix',
    'cylinders',
    'spheres',
    'crossing_spheres',
    'walls',
    'beams',
    'boxes',
    'gates',
    'slalom',
    'city_blocks',
    'swinging_sticks',
)


def add_marker(position, color, size=0.2, client_id=0):
    """Add a visual marker at position."""
    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=size,
        rgbaColor=color,
        physicsClientId=client_id
    )

    marker_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        physicsClientId=client_id
    )

    return marker_id


def add_line(start, end, color, width=3, client_id=0):
    """Add a line between two points."""
    return p.addUserDebugLine(
        start, end, color, lineWidth=width,
        physicsClientId=client_id
    )


def preview_scenario(stage, obstacle_type, duration, continuous):
    """
    Preview scenario generation.

    Args:
        stage: Stage type ('0', '1', 'pretrain')
        obstacle_type: Obstacle type for pretrain
        duration: Seconds to show each scenario
        continuous: If True, regenerate continuously
    """
    # Connect to PyBullet
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Setup camera
    p.resetDebugVisualizerCamera(
        cameraDistance=20,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 2],
        physicsClientId=client
    )

    # Load config
    config = load_config(stage)

    # Add ground plane
    p.loadURDF("plane.urdf", physicsClientId=client)

    # Add arena boundaries (visual only)
    arena_x = config.ARENA_SIZE_X
    arena_y = config.ARENA_SIZE_Y
    arena_h = config.ARENA_HEIGHT

    # Draw arena box
    corners = [
        [-arena_x/2, -arena_y/2, 0],
        [arena_x/2, -arena_y/2, 0],
        [arena_x/2, arena_y/2, 0],
        [-arena_x/2, arena_y/2, 0],
    ]

    for i in range(4):
        # Bottom edges
        p.addUserDebugLine(corners[i], corners[(i+1)%4], [0.5, 0.5, 0.5], lineWidth=2, physicsClientId=client)
        # Vertical edges
        p.addUserDebugLine(corners[i], [corners[i][0], corners[i][1], arena_h], [0.5, 0.5, 0.5], lineWidth=2, physicsClientId=client)
        # Top edges
        top_i = [corners[i][0], corners[i][1], arena_h]
        top_next = [corners[(i+1)%4][0], corners[(i+1)%4][1], arena_h]
        p.addUserDebugLine(top_i, top_next, [0.5, 0.5, 0.5], lineWidth=2, physicsClientId=client)

    print("=" * 60)
    print(f"SCENARIO PREVIEW - STAGE {stage.upper()}")
    if stage == 'pretrain':
        print(f"Obstacle type: {obstacle_type}")
    print("=" * 60)
    print(f"\nArena: {arena_x}x{arena_y}x{arena_h}m")
    print(f"Duration per scenario: {duration}s")
    print(f"Continuous regeneration: {continuous}")
    print("\nPress Ctrl+C to exit\n")

    scenario_count = 0
    markers = []
    lines = []

    try:
        while True:
            scenario_count += 1
            print(f"\n[Scenario {scenario_count}]")

            # Create scenario
            if stage == '0':
                scenario = Stage0Scenario()
            elif stage == '1':
                scenario = Stage1Scenario()
            else:  # pretrain
                scenario = StagePretrainScenario(obstacle_type=obstacle_type)

            # Generate
            start_pos, goal_pos = scenario.reset(client)

            print(f"  Start: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
            print(f"  Goal:  [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")

            distance = np.linalg.norm(goal_pos - start_pos)
            print(f"  Distance: {distance:.2f}m")

            if stage != '0':
                print(f"  Obstacles: {len(scenario.obstacles)}")

                # Count dynamic obstacles
                dynamic_count = sum(1 for obs in scenario.obstacles
                                  if hasattr(obs, 'dynamic') and obs.dynamic)
                if dynamic_count > 0:
                    print(f"  Dynamic obstacles: {dynamic_count}")

            # Add start marker (green)
            start_marker = add_marker(start_pos, [0, 1, 0, 1], size=0.3, client_id=client)
            markers.append(start_marker)

            # Add goal marker (red)
            goal_marker = add_marker(goal_pos, [1, 0, 0, 1], size=0.3, client_id=client)
            markers.append(goal_marker)

            # Add line between start and goal (yellow)
            line = add_line(start_pos, goal_pos, [1, 1, 0], width=2, client_id=client)
            lines.append(line)

            # Show for duration seconds
            start_time = time.time()
            while time.time() - start_time < duration:
                # Update dynamic obstacles
                scenario.update_dynamic_obstacles(0.01)
                p.stepSimulation(physicsClientId=client)
                time.sleep(0.01)

            # Clean up for next scenario
            if continuous:
                # Remove markers and lines
                for marker in markers:
                    p.removeBody(marker, physicsClientId=client)
                markers.clear()

                for line in lines:
                    p.removeUserDebugItem(line, physicsClientId=client)
                lines.clear()

                # Clean up scenario
                scenario.cleanup()
            else:
                break

    except KeyboardInterrupt:
        print("\n\nExiting...")

    finally:
        p.disconnect(physicsClientId=client)


def main():
    parser = argparse.ArgumentParser(description='Preview scenario generation')
    parser.add_argument('--stage', type=str, required=True,
                       choices=['0', '1', 'pretrain'],
                       help='Stage to preview')
    parser.add_argument('--obstacle-type', type=str, default='random',
                       choices=PRETRAIN_OBSTACLE_CHOICES,
                       help='Obstacle type for pretrain stage')
    parser.add_argument('--duration', type=int, default=5,
                       help='Seconds to show each scenario (default: 5)')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuously regenerate scenarios')

    args = parser.parse_args()

    preview_scenario(args.stage, args.obstacle_type, args.duration, args.continuous)


if __name__ == "__main__":
    main()
