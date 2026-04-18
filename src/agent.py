import logging
from time import time
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import make_environment
from src.evaluation import Evaluation


class Agent:
    """
    Deep reinforcement learning Agent
    """

    __model_cache_path: str
    __logging_dir: str
    __algorithm: type[OffPolicyAlgorithm] | type[OnPolicyAlgorithm]
    __policy: str
    __train_time: float
    __random_seed: int

    def __init__(
        self,
        algorithm: type[OffPolicyAlgorithm] | type[OnPolicyAlgorithm],
        policy: str,
        cache_dir: str,
        logging_dir: str,
        random_seed: int = 42,
    ):
        self.__model_cache_path = f"{cache_dir}/{algorithm.__name__}-{policy}"
        self.__logging_dir = logging_dir
        self.__algorithm = algorithm
        self.__policy = policy
        self.__random_seed = random_seed

    def train(self, total_timesteps: int, learning_rate: float, gamma: float):
        """
        Train the agent by playing
        """
        total_environments = 8
        training_environment = make_environment(
            self.__random_seed,
            logs_dir=f"{self.__logging_dir}/train",
            # Parallelise the learning
            total_environments=total_environments,
        )
        evaluation_environment = make_environment(self.__random_seed, logs_dir=None)

        evaluation_callback = EvalCallback(
            evaluation_environment,
            best_model_save_path=self.__model_cache_path,
            log_path=f"{self.__logging_dir}/evaluation",
            # Required so the model saves even if a few timesteps
            eval_freq=min(10_000, total_timesteps // total_environments),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        buffer_args = (
            # Something managable without running out of memory
            {"buffer_size": 50_000}
            if issubclass(self.__algorithm, OffPolicyAlgorithm)
            else {}
        )
        model = self.__algorithm(
            self.__policy,
            env=training_environment,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=1,
            seed=self.__random_seed,
            **buffer_args,
        )

        logging.info(
            f"Starting training using {self.__algorithm.__name__} for {total_timesteps} timesteps at a rate of {learning_rate} and gamma {gamma}"
        )

        start = time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=evaluation_callback,
        )
        self.__train_time = time() - start

    def watch(self, run_time: int):
        """
        View a agent playing a game using a pretrained model
        """
        environment = make_environment(self.__random_seed, logs_dir=None)

        model = self.__algorithm.load(
            f"{self.__model_cache_path}/best_model", environment
        )
        logging.info(f"Loaded model from {self.__model_cache_path}")

        environment = model.get_env()
        if environment is None:
            raise ValueError("environment is broken")

        observation = environment.reset()
        for i in range(run_time):
            action, _ = model.predict(observation, deterministic=True)

            environment.step(action)
            environment.render("human")

    def evaluate(self) -> Evaluation:
        """
        Evaluate the agents performance
        """
        environment = make_environment(
            self.__random_seed, logs_dir=None, total_environments=1
        )

        model = self.__algorithm.load(
            f"{self.__model_cache_path}/best_model", environment
        )
        logging.info(f"Loaded model from {self.__model_cache_path}")

        start = time()
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            environment,
            n_eval_episodes=10,
            deterministic=True,
            return_episode_rewards=True,
        )
        test_time = time() - start

        return Evaluation(
            episode_rewards,
            episode_lengths,
            train_time=self.__train_time,
            test_time=test_time,
        )
