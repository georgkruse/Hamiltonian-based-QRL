import gymnasium as gym

from games.gym_turtle.builders.default import default_builder


class CompactEnv(gym.Env):

    def __init__(self, mode):

        self.turtleEnv = default_builder()

        self.sim = self.turtleEnv.sim
        self.sim_steps = self.turtleEnv.sim_steps
        self.world = self.turtleEnv.world
        self.turtlebot = self.turtleEnv.turtlebot
        self.control = self.turtleEnv.control
        self.reward = self.turtleEnv.reward
        self.termination = self.turtleEnv.termination
        self.observation_space = self.turtleEnv.observation_space
        self.action_space = self.turtleEnv.action_space
        self.collisions = self.turtleEnv.collisions
        self.max_state = self.turtleEnv.max_state
        self.steps = self.turtleEnv.steps

    def step(self, action):
        return self.turtleEnv.step(action)

    def reset(self, *, seed=None, options=None):
        return self.turtleEnv.reset(seed=None, options=None)

    def render(self, *_):
        self.turtleEnv.render(*_)

    def close(self):
        self.turtleEnv.close()

    def seed(self, *_):
        self.turtleEnv.seed(*_)

    def _run_sim_steps(self, steps, check_collisions=True):
        self.turtleEnv._run_sim_steps(steps, check_collisions)

    def _check_collisions(self):
        self.turtleEnv._check_collisions()
