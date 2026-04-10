from dotenv import load_dotenv
import os
import logging
from src.agent import Agent
from stable_baselines3 import A2C

load_dotenv()
logging.basicConfig(level=logging.INFO)

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "cache")
LOGGING_DIR = os.getenv("LOGGING_DIR", "logs")

TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 10))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.00083))
GAMMA = float(os.getenv("GAMMA", 0.995))

if __name__ == "__main__":
    agent = Agent(A2C, "CnnPolicy", MODEL_CACHE_DIR, LOGGING_DIR)

    agent.train(TOTAL_TIMESTEPS, LEARNING_RATE, GAMMA)
    agent.watch()
