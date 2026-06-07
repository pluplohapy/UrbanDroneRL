import numpy as np
import pybullet as p
from scenarios.base_scenario import BaseScenario
from envs.obstacles import StaticObstacle
from config import load_config


class Stage1Scenario(BaseScenario):

    def __init__(self, seed=None):
        super().__init__(seed)
        self.config = load_config('1')
        self.n_obstacles = 0
        self.client_id = None

    def generate(self, client_id):
        return self.reset(client_id)

    def reset(self, client_id):

        self.client_id = client_id


        self.obstacles = []


        start_pos, goal_pos = self._generate_start_goal()


        self.n_obstacles = self.rng.randint(
            self.config.STAGE1_N_OBSTACLES[0],
            self.config.STAGE1_N_OBSTACLES[1] + 1
        )



        for i in range(self.n_obstacles):

            radius = self.rng.uniform(
                self.config.STAGE1_RADIUS[0],
                self.config.STAGE1_RADIUS[1]
            )


            max_attempts = 50
            for attempt in range(max_attempts):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + 1, self.config.ARENA_SIZE_X/2 - 1)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + 1, self.config.ARENA_SIZE_Y/2 - 1)
                pos = np.array([x, y, 0.0])


                dist_to_start = np.linalg.norm(pos[:2] - start_pos[:2])
                dist_to_goal = np.linalg.norm(pos[:2] - goal_pos[:2])

                if (dist_to_start < self.config.MIN_CLEARANCE + radius or
                    dist_to_goal < self.config.MIN_CLEARANCE + radius):
                    continue


                line_vec = goal_pos[:2] - start_pos[:2]
                line_length = np.linalg.norm(line_vec)

                if line_length > 0:
                    line_dir = line_vec / line_length


                    start_to_obs = pos[:2] - start_pos[:2]


                    projection = np.dot(start_to_obs, line_dir)


                    if 0 < projection < line_length:

                        perpendicular = start_to_obs - projection * line_dir
                        dist_to_line = np.linalg.norm(perpendicular)


                        if dist_to_line < self.config.MIN_CLEARANCE + radius:
                            continue


                obstacle = StaticObstacle(
                    position=pos,
                    radius=radius,
                    height=self.config.ARENA_HEIGHT,
                    physics_client=client_id
                )
                self.obstacles.append(obstacle)

                break

        return start_pos, goal_pos

    def update_dynamic_obstacles(self, dt):
        pass

    def get_obstacles(self):
        return self.obstacles