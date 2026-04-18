from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import no_grad, cuda
import json
import logging
from enum import Enum
from dataclasses import is_dataclass, asdict


class LLM:
    """
    LLM based agent to ask questions to
    """

    def __init__(self):
        device = "cuda" if cuda.is_available() else "cpu"

        # Reasonable tradeoff on size vs ability to follow instructions
        # Smaller and it doesnt return what you ask it
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.__model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.__tokeniser = AutoTokenizer.from_pretrained(model_name)

    def prompt(self, prompt: str, **kwargs) -> str:
        """
        Ask the LLM a question
        """
        # Handle serialisation of all types to json
        context = {
            key: json.dumps(value, default=self.serialise)
            for key, value in kwargs.items()
        }
        encoded_prompt = prompt.format(**context)
        logging.debug(f"Prompting LLM with {encoded_prompt}")

        messages = [{"role": "user", "content": encoded_prompt}]
        chat_prompt = self.__tokeniser.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.__tokeniser(chat_prompt, return_tensors="pt")
        # Load to device
        inputs = {key: value.to(self.__model.device) for key, value in inputs.items()}

        with no_grad():
            outputs = self.__model.generate(**inputs, max_new_tokens=20)

            # Get the answer
            return self.__tokeniser.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

    def serialise(self, value: dict):
        """
        Wrangle data into string formats for the prompt
        """
        if is_dataclass(value):
            return asdict(value)

        if isinstance(value, Enum):
            return value.name

        if isinstance(value, type):
            return value.__name__

        return str(value)
