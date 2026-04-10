from dotenv import load_dotenv
import os
import logging
from itertools import product
from stable_baselines3 import DQN, PPO, A2C
from src.experimenter import BaseConfig, Experimenter, Experiment

load_dotenv()
logging.basicConfig(level=logging.INFO)

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "cache")
LOGGING_DIR = os.getenv("LOGGING_DIR", "logs")
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", "experiments")

if __name__ == "__main__":
    config = BaseConfig(
        experiments_dir=EXPERIMENTS_DIR,
        cache_dir=MODEL_CACHE_DIR,
        logging_dir=LOGGING_DIR,
    )
    experimenter = Experimenter(config)

    algorithms = [DQN, PPO, A2C]
    policies = ["CnnPolicy"]
    total_timesteps = [500_000, 1_000_000]
    learning_rates = [3e-4, 5e-5]
    gammas = [0.95, 0.99]
    seeds = [42, 7, 105]

    experiments: dict[str, Experiment] = {
        f"{i}": Experiment(
            algorithm=algorithm,
            policy=policy,
            timesteps=timesteps,
            learning_rate=learning_rate,
            gamma=gamma,
            random_seed=random_seed,
        )
        for i, (
            algorithm,
            policy,
            timesteps,
            learning_rate,
            gamma,
            random_seed,
        ) in enumerate(
            product(
                algorithms, policies, total_timesteps, learning_rates, gammas, seeds
            )
        )
    }

    logging.info(f"Generated {len(experiments)} experiments")

    try:
        results = experimenter.run(experiments)
        logging.info(f"Best experiment {results}")
    except KeyboardInterrupt:
        logging.info("Interrupted")
