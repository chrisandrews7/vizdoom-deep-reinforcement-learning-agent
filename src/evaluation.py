from dataclasses import dataclass
from numpy import mean, std


@dataclass
class Evaluation:
    """
    Evaluate an agent and return a set of common model evaluation metrics
    """

    mean_episode_reward: float
    std_episode_reward: float
    mean_steps_per_episode: float
    score: float

    train_time: float
    test_time: float

    def __init__(
        self,
        episode_rewards: list[float],
        episode_lengths: list[float],
        train_time: float,
        test_time: float,
    ):
        """
        Returns a set of common model evaluation metrics
        """
        self.mean_episode_reward = float(mean(episode_rewards))
        self.std_episode_reward = float(std(episode_rewards))
        self.mean_steps_per_episode = float(mean(episode_lengths))

        # This is a measure of overall performance
        self.score = self.mean_episode_reward

        self.train_time = train_time
        self.test_time = test_time
