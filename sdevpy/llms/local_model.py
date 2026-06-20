from abc import ABC, abstractmethod


class LocalModel(ABC):
    def __init__(self, config: dict):
        self.model = None
        self.config = config

    @abstractmethod
    def respond_prompt(self, prompt: str, **kwargs) -> str: # pragma: no cov
        """ Respond to simple prompt """
        pass

    @abstractmethod
    def respond_instruction(self, system_prompt: str, user_prompt: str, **kwargs) -> str: # pragma: no cov
        """ Respond to instruction prompt """
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ]
        return self.chat(messages, **kwargs)

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str: # pragma: no cov
        """ Chat-oriented structured response to messages """
        pass

    def clean_response(self, response: str) -> str:
        """ Remove useless nodes in instruction responses """
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        if response.strip().startswith("```"):
            response = response.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return response

    def underlying_model(self):
        return self.model

    @abstractmethod
    def load(self, max_context_tokens: int=None) -> None: # pragma: no cov
        """ Load underlying model into memory """
        pass

    @abstractmethod
    def unload(self) -> None: # pragma: no cov
        """ Delete underlying model from memory """
        pass

    @abstractmethod
    def pretty_print(self) -> None: # pragma: no cov
        """ Display information about the model """
        pass
