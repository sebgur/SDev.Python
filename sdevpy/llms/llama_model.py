import warnings
import os
import gc
from llama_cpp import Llama
from sdevpy.llms.llama_model import LocalModel


class LlamaModel(LocalModel):
    def __init__(self, config: dict):
        super().__init__(config)

    def respond_prompt(self, prompt: str, **kwargs) -> str:
        """ Single prompt response from model """
        max_tokens = kwargs.get('max_tokens', 4096)
        if max_tokens is None:
            max_tokens = -1 # 'no limit' in Llama

        messages = [{"role": "user", "content": prompt}]

        # Remove thinking mode if specified
        if self.config.get("no_think", False):
            messages = messages.copy()
            messages[0] = {**messages[0], "content": messages[0]["content"] + "\n/no_think"}

        response = self.model.create_chat_completion(messages=messages, max_tokens=max_tokens)
        response = response["choices"][0]["message"]["content"]

        # Return clean response
        return self.clean_response(response)

    def chat(self, messages: list[dict], **kwargs) -> str:
        """ Chat-oriented response to messages """
        max_tokens = kwargs.get('max_tokens', 4096)
        if max_tokens is None:
            max_tokens = -1 # 'no limit' in Llama

        # Remove thinking mode if specified
        if self.config.get("no_think", False):
            messages = messages.copy()
            messages[0] = {**messages[0], "content": messages[0]["content"] + "\n/no_think"}

        response = self.model.create_chat_completion(messages=messages, max_tokens=max_tokens)
        response = response["choices"][0]["message"]["content"]

        # Return clean response
        return self.clean_response(response)

    def pretty_print(self) -> None: # pragma: no cov
        """ Display information about the model """
        metadata = self.model.metadata
        print(f"Model path: {self.model.model_path}")
        print(f"Model name: {metadata.get('general.name', 'Unknown')}")
        print(f"Architecture: {metadata.get('general.architecture', 'Unknown')}")
        print(f"Context length: {self.model.n_ctx()}")
        size_gb = os.path.getsize(self.model.model_path) / (1024**3)
        print(f"File size (GB): {size_gb:.3f}")

    def load(self) -> None:
        """ Load Llama model from Hugging Face into memory.
            n_ctx: restrict max context size, default=0 (no restriction, maximum available)
        """
        hf_repo_id = self.config.get("repo_id")
        filename = self.config.get("filename")
        n_ctx = self.config.get("max_context_tokens", 0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")
            self.model = Llama.from_pretrained(repo_id=hf_repo_id, filename=filename, n_ctx=n_ctx, verbose=False)

    def unload(self) -> None:
        """ Unload Llama model from memory """
        del self.model
        gc.collect()


def get_model(repo_id: str, filename: str="*.gguf", n_ctx: int=0) -> Llama:
    """ Retrieve Llama model """
    return Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=n_ctx, verbose=False)


if __name__ == "__main__":
    repo_id, filename, no_think = "unsloth/Qwen3.5-27B-GGUF", "Qwen3.5-27B-Q4_K_M.gguf", True
    config = {"repo_id": repo_id, "filename": filename, "no_think": no_think}

    # Load model into RAM
    model = LlamaModel(config)
    model.load()

    # Display information
    model.pretty_print()

    # Respond to simple prompt
    prompt = "Why is the sky blue?"
    response = model.respond_prompt(prompt, max_tokens=1200)
    print(response)

    model.unload()
