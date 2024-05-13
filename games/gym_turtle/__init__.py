from gymnasium.envs.registration import register

__all__ = []

from .core import make
__all__.append("make")

register(
    id='TurtleEnv-v0',
    entry_point='gym_turtle.envs:TurtleEnv'
)