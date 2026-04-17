"""
Test script for RRT* planner.
Tests basic functionality and visualizes results.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planners.rrt_star import RRTStarPlanner
from planners.visualization import visualize_rrt_tree_3d, visualize_rrt_tree_2d
from scenarios.stage1_static import Stage1Scenario
import config
import pybullet as p


def test_rrt_star_empty():
    """Test RRT* in empty space (no obstacles)."""
    print("\n" + "="*60)
    print("TEST 1: RRT* in Empty Space")
    print("="*60)

    # Setup
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([5.0, 5.0, 3.0])

    planner = RRTStarPlanner(
        arena_bounds=(config.ARENA_SIZE_X, config.ARENA_SIZE_Y, config.ARENA_HEIGHT),
        max_iter=1000,
        step_size=1.0,
        goal_bias=0.1,
        goal_threshold=0.8,
        rewire_radius=3.0
    )

    # Plan
    path = planner.plan(start, goal, obstacles=[])

    # Results
    if path is not None:
        print(f"\n✓ Test passed!")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Path cost: {planner.path_cost:.2f}m")
        print(f"  Planning time: {planner.planning_time:.3f}s")

        # Visualize
        visualize_rrt_tree_2d(planner, path, start, goal, view='xy')
        visualize_rrt_tree_3d(planner, path, start, goal)
    else:
        print(f"\n✗ Test failed: No path found")


def test_rrt_star_with_obstacles():
    """Test RRT* with static obstacles."""
    print("\n" + "="*60)
    print("TEST 2: RRT* with Static Obstacles")
    print("="*60)

    # Create PyBullet client (headless)
    client = p.connect(p.DIRECT)

    # Create scenario
    scenario = Stage1Scenario(seed=42)
    start, goal = scenario.reset(client)

    print(f"\nStart: {start}")
    print(f"Goal: {goal}")
    print(f"Obstacles: {len(scenario.obstacles)}")

    # Create planner
    planner = RRTStarPlanner(
        arena_bounds=(config.ARENA_SIZE_X, config.ARENA_SIZE_Y, config.ARENA_HEIGHT),
        max_iter=2000,
        step_size=1.0,
        goal_bias=0.15,
        goal_threshold=0.8,
        rewire_radius=3.0
    )

    # Plan
    path = planner.plan(start, goal, scenario.obstacles)

    # Results
    if path is not None:
        print(f"\n✓ Test passed!")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Path cost: {planner.path_cost:.2f}m")
        print(f"  Planning time: {planner.planning_time:.3f}s")
        print(f"  Tree nodes: {len(planner.nodes)}")

        # Print waypoints
        print(f"\nWaypoints:")
        for i, wp in enumerate(path):
            print(f"  {i}: [{wp[0]:6.2f}, {wp[1]:6.2f}, {wp[2]:6.2f}]")

        # Visualize
        visualize_rrt_tree_2d(planner, path, start, goal, scenario.obstacles, view='xy')
        visualize_rrt_tree_3d(planner, path, start, goal, scenario.obstacles)
    else:
        print(f"\n✗ Test failed: No path found")

    # Cleanup
    scenario.cleanup()
    p.disconnect(client)


def test_rrt_star_multiple_runs():
    """Test RRT* consistency over multiple runs."""
    print("\n" + "="*60)
    print("TEST 3: RRT* Consistency (10 runs)")
    print("="*60)

    # Create PyBullet client (headless)
    client = p.connect(p.DIRECT)

    # Create scenario
    scenario = Stage1Scenario(seed=42)
    start, goal = scenario.reset(client)

    # Run multiple times
    n_runs = 10
    results = []

    for run in range(n_runs):
        planner = RRTStarPlanner(
            arena_bounds=(config.ARENA_SIZE_X, config.ARENA_SIZE_Y, config.ARENA_HEIGHT),
            max_iter=2000,
            step_size=1.0,
            goal_bias=0.15,
            goal_threshold=0.8,
            rewire_radius=3.0
        )

        path = planner.plan(start, goal, scenario.obstacles)

        if path is not None:
            results.append({
                'success': True,
                'cost': planner.path_cost,
                'time': planner.planning_time,
                'waypoints': len(path),
                'nodes': len(planner.nodes)
            })
        else:
            results.append({
                'success': False,
                'cost': float('inf'),
                'time': planner.planning_time,
                'waypoints': 0,
                'nodes': len(planner.nodes)
            })

        print(f"  Run {run+1}/{n_runs}: {'✓' if path else '✗'} "
              f"cost={results[-1]['cost']:.2f}m, "
              f"time={results[-1]['time']:.3f}s")

    # Statistics
    success_rate = sum(1 for r in results if r['success']) / n_runs
    successful_results = [r for r in results if r['success']]

    if successful_results:
        avg_cost = np.mean([r['cost'] for r in successful_results])
        std_cost = np.std([r['cost'] for r in successful_results])
        avg_time = np.mean([r['time'] for r in successful_results])
        avg_waypoints = np.mean([r['waypoints'] for r in successful_results])

        print(f"\n✓ Test completed!")
        print(f"  Success rate: {success_rate*100:.0f}%")
        print(f"  Avg cost: {avg_cost:.2f} ± {std_cost:.2f}m")
        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Avg waypoints: {avg_waypoints:.1f}")
    else:
        print(f"\n✗ Test failed: No successful runs")

    # Cleanup
    scenario.cleanup()
    p.disconnect(client)


def test_rrt_star_difficult_scenario():
    """Test RRT* with many obstacles (difficult scenario)."""
    print("\n" + "="*60)
    print("TEST 4: RRT* with Many Obstacles")
    print("="*60)

    # Create PyBullet client (headless)
    client = p.connect(p.DIRECT)

    # Create scenario with more obstacles
    original_n_obstacles = config.STAGE1_N_OBSTACLES
    config.STAGE1_N_OBSTACLES = (15, 20)  # More obstacles

    scenario = Stage1Scenario(seed=123)
    start, goal = scenario.reset(client)

    print(f"\nStart: {start}")
    print(f"Goal: {goal}")
    print(f"Obstacles: {len(scenario.obstacles)}")

    # Create planner with more iterations
    planner = RRTStarPlanner(
        arena_bounds=(config.ARENA_SIZE_X, config.ARENA_SIZE_Y, config.ARENA_HEIGHT),
        max_iter=5000,  # More iterations for difficult scenario
        step_size=0.8,  # Smaller steps
        goal_bias=0.2,  # Higher goal bias
        goal_threshold=0.8,
        rewire_radius=2.5
    )

    # Plan
    path = planner.plan(start, goal, scenario.obstacles)

    # Results
    if path is not None:
        print(f"\n✓ Test passed!")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Path cost: {planner.path_cost:.2f}m")
        print(f"  Planning time: {planner.planning_time:.3f}s")
        print(f"  Tree nodes: {len(planner.nodes)}")

        # Visualize
        visualize_rrt_tree_2d(planner, path, start, goal, scenario.obstacles, view='xy')
    else:
        print(f"\n✗ Test failed: No path found")

    # Restore config
    config.STAGE1_N_OBSTACLES = original_n_obstacles

    # Cleanup
    scenario.cleanup()
    p.disconnect(client)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RRT* PLANNER TEST SUITE")
    print("="*60)

    # Run tests
    test_rrt_star_empty()
    test_rrt_star_with_obstacles()
    test_rrt_star_multiple_runs()
    test_rrt_star_difficult_scenario()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
