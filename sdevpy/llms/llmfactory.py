from pathlib import Path
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.llms.local_model import LocalModel
from sdevpy.llms.transformers_model import TransformersModel
from sdevpy.llms.llama_model import LlamaModel


def read_llm_config() -> dict:
    """ Read Local LLM config and return object """
    config_file = Path() / "local_llms.json"
    if config_file.exists():
        return jsm.deserialize(config_file)
    else:
        raise ValueError(f"LLM config file not found: {config_file}")

def get_llm_config(model_id: str) -> dict:
    """ Retrieve model's config given internal ID """
    all_model_config = read_llm_config()
    model_config = all_model_config.get(model_id, None)
    if model_config is not None:
        return model_config
    else:
        raise ValueError(f"No model config found for ID: {model_id}")


def from_pretrained(model_id: str) -> LocalModel:
    """ Load LocalModel knowing its internal ID. Unifies Transformers and Llama models under one interface. """
    config = get_llm_config(model_id)
    model_type = config['type'].lower()
    match model_type:
        case 'transformers':
            model = TransformersModel(config)
        case 'llama':
            model = LlamaModel(config)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    model.load()
    return model


if __name__ == "__main__":
    model_id = "qwen3.5-0.8B"
    print()
    print(f"Chatting with: {model_id}")

    # Load model
    model = from_pretrained(model_id)

    prompt = "Why is the sky blue?"
    print()
    print("Prompt:")
    print(prompt)

    print("\nBot:")
    response = model.respond_prompt(prompt)
    model.unload()

    print(response)
