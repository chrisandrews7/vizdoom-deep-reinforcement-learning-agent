import logging
from os import makedirs
from json import dump
from typing import Tuple
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from src.agent import Agent
from src.llm import LLM
from src.evaluation import Evaluation


@dataclass
class BaseConfig:
    experiments_dir: str
    cache_dir: str
    logging_dir: str
    max_experiments: int | None = None


@dataclass
class Experiment:
    algorithm: type[OffPolicyAlgorithm] | type[OnPolicyAlgorithm]
    policy: str
    timesteps: int
    learning_rate: float
    gamma: float
    random_seed: int


@dataclass
class State:
    # ID must be a number for the current agent model to be able to return a valid response
    # that only contains the id and not lots of other text
    current_experiment_id: str
    experiments: dict[str, Experiment]
    evaluations: dict[str, Evaluation]


class Experimenter:
    """
    Agentic AI for finding most performant hyperparameters
    """

    __config: BaseConfig

    def __init__(self, config: BaseConfig):
        graph = StateGraph(State)
        graph.add_node("plan", self.__plan)
        graph.add_node("run", self.__run)

        graph.add_edge("plan", "run")
        graph.set_entry_point("plan")
        # End edge
        graph.add_conditional_edges(
            "run", self.__should_continue, {"plan": "plan", "end": END}
        )

        self.__workflow = graph.compile()
        self.__config = config
        self.__llm = LLM()

    def run(self, experiments: dict[str, Experiment]) -> Tuple[Experiment, Evaluation]:
        """
        Start the agents journey
        """
        state = State(
            # State isnt returned typed but as a dict, must recast it
            **self.__workflow.invoke(
                State(evaluations={}, experiments=experiments, current_experiment_id="")
            )
        )

        # Return the best experiment and evaluation based on whatever the evaluation defined as score
        best_experiment_id = max(
            state.evaluations, key=lambda key: state.evaluations[key].score
        )
        return (
            state.experiments[best_experiment_id],
            state.evaluations[best_experiment_id],
        )

    def __plan(self, state: State) -> State:
        """
        Find the best experiment to run next
        """
        remaining_experiments = self.__remaining_experiments(state)
        logging.info(f"{len(remaining_experiments)} experiments remaining")

        logging.info("Prompting LLM for next experiment")
        next_experiment_id: str = self.__llm.prompt(
            """You are an RL experiment optimiser for ViZDoom.
            Past results: {evaluations}.
            Remaining experiments: {experiments}.
            Analyze and select the best next experiment. Output ONLY a single number (the experiment ID). No explanation, no other text.""",
            evaluations=state.evaluations,
            experiments=remaining_experiments,
        )

        # This would benefit from some better parsing logic if the agent returns "54" instead of 54
        if next_experiment_id not in remaining_experiments:
            logging.warning(f'LLM experiment "{next_experiment_id}" is not valid')
            state.current_experiment_id = next(iter(remaining_experiments))
            logging.info(f'Using experiment "{state.current_experiment_id}" instead')
            return state

        state.current_experiment_id = next_experiment_id
        logging.info(
            f"LLM selected experiment {state.current_experiment_id} as the next best experiment"
        )

        return state

    def __run(self, state: State) -> State:
        """
        Execute the experiment and report evaluation
        """
        if state.current_experiment_id not in state.experiments:
            raise ValueError(f"{state.current_experiment_id} not found in experiments")

        experiment: Experiment = state.experiments[state.current_experiment_id]
        logging.info(f"Running experiment {state.current_experiment_id}")

        agent = Agent(
            algorithm=experiment.algorithm,
            policy=experiment.policy,
            cache_dir=self.__config.cache_dir,
            logging_dir=f"{self.__config.logging_dir}/{state.current_experiment_id}",
            random_seed=experiment.random_seed,
        )

        agent.train(
            total_timesteps=experiment.timesteps,
            learning_rate=experiment.learning_rate,
            gamma=experiment.gamma,
        )
        evaluation = agent.evaluate()

        logging.info(
            f"Finished experiment {state.current_experiment_id} with evaluation {evaluation}"
        )
        state.evaluations[state.current_experiment_id] = evaluation

        if self.__config.experiments_dir != "":
            # Save as we go to avoid total loss
            makedirs(self.__config.experiments_dir, exist_ok=True)
            # JSONL so we can use newlines and append only
            file_name = f"{self.__config.experiments_dir}/experiments.jsonl"
            with open(file_name, "a") as file:
                dump(
                    {
                        "id": state.current_experiment_id,
                        "experiment": asdict(
                            state.experiments[state.current_experiment_id]
                        ),
                        "evaluation": asdict(
                            state.evaluations[state.current_experiment_id]
                        ),
                    },
                    file,
                    default=lambda value: getattr(value, "__name__", str(value)),
                )
                file.write("\n")
            logging.info(
                f"Wrote experiment {state.current_experiment_id} to {file_name}"
            )

        return state

    def __should_continue(self, state: State) -> str:
        """
        Breakout of the loop
        """
        if (
            self.__config.max_experiments is not None
            and len(state.evaluations) >= self.__config.max_experiments
        ):
            return "end"

        return "plan" if self.__remaining_experiments(state) else "end"

    def __remaining_experiments(self, state: State) -> dict[str, Experiment]:
        """
        Single source of truth for experiments remaining to be run
        """
        return {
            key: value
            for key, value in state.experiments.items()
            if key not in state.evaluations
        }
