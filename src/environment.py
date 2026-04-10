from gymnasium import ObservationWrapper as GObservationWrapper
from gymnasium.spaces import Box
from cv2 import resize, cvtColor, COLOR_RGB2GRAY
from numpy import expand_dims, uint8
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)

# Required to find vizoom environments
import vizdoom.gymnasium_wrapper


# Reduce size to something manageable
class ObservationWrapper(GObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(0, 255, shape=(80, 100, 1), dtype=uint8)

    def observation(self, observation):
        screen = resize(observation["screen"], (100, 80))
        gray = cvtColor(screen, COLOR_RGB2GRAY)
        return expand_dims(gray, axis=-1)


def make_environment(
    seed: int, logs_dir: str | None, total_environments: int = 1
) -> VecEnv:
    environment = make_vec_env(
        "VizdoomDefendCenter-v1",
        n_envs=total_environments,
        wrapper_class=ObservationWrapper,
        env_kwargs={"frame_skip": 4},
        monitor_dir=logs_dir,
        seed=seed,
    )
    # History for temporal context
    environment = VecFrameStack(environment, n_stack=4)
    # Rearrange to required format
    return VecTransposeImage(environment)
