from dotenv import load_dotenv
import os
import logging
from enum import Enum
from src.agent import Agent
from stable_baselines3 import A2C, PPO, DQN

load_dotenv()
logging.basicConfig(level=logging.INFO)

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "cache")
LOGGING_DIR = os.getenv("LOGGING_DIR", "logs")

ALGORITHM = os.getenv("ALGORITHM", "PPO")
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 1000))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.00083))
GAMMA = float(os.getenv("GAMMA", 0.995))


class Algorithms(Enum):
    A2C = A2C
    PPO = PPO
    DQN = DQN

    def __str__(self):
        return self.name


if __name__ == "__main__":
    try:
        algorithm = Algorithms[ALGORITHM]
    except KeyError:
        raise ValueError(f"{ALGORITHM} isnt available")

    agent = Agent(algorithm.value, "CnnPolicy", MODEL_CACHE_DIR, LOGGING_DIR)

    agent.train(TOTAL_TIMESTEPS, LEARNING_RATE, GAMMA)
    logging.info("Finished training agent")

    # Watch on screen the learnt policy
    logging.info("Rendering trained agent playing")
    agent.watch(1000)
