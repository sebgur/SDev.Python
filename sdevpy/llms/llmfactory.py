from pathlib import Path
import pandas as pd
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.llms.local_model import LocalModel
from sdevpy.llms.transformers_model import TransformersModel
from sdevpy.llms.llama_model import LlamaModel
from sdevpy.llms import huggingface
from sdevpy.tests import conftest as tst


def run_instruction(model_id: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
    """ Return response from model """
    model = None
    try:
        max_context_tokens = kwargs.pop('max_context_tokens', None)
        model = from_pretrained(model_id, max_context_tokens=max_context_tokens)
        response = model.respond_instruction(system_prompt, user_prompt, **kwargs)
        return response
    finally:
        if model is not None:
            model.unload()


def list_models() -> list[str]:
    """ List configured models (downloaded or not) """
    config_df = list_model_info()
    return config_df['Name'].tolist()


def list_model_info() -> list[dict]:
    """ List information about the models that are configured for use """
    config = read_llm_config() # Configured models
    hf_list = huggingface.list_available_models() # Available models (downloaded)

    lib_ids, repo_ids, downloadeds, types = [], [], [], []
    filenames, size_gbs, size_strings = [], [], []
    for model_key, model_config in config.items():
        try:
            repo_id = model_config.get("repo_id")
            type_ = model_config.get("type")
            filename = model_config.get("filename", None)

            # Get information from Hugging Face if downloaded
            hf_model = next((m for m in hf_list if m['repo_id'] == repo_id), None)
            if hf_model is None:
                downloaded = False
                size_gb = 0
                size_string = '0'
            else:
                downloaded = True
                size_gb = hf_model['size_gb']
                size_string = hf_model['size_string']

            lib_ids.append(model_key)
            repo_ids.append(repo_id)
            size_gbs.append(size_gb)
            size_strings.append(size_string)
            downloadeds.append(downloaded)
            types.append(type_)
            filenames.append(filename)
        except Exception as e:
            print(f"Failed to retrieve model information for {model_key}: {str(e)}")

    df = pd.DataFrame({'Name': lib_ids, 'Repo': repo_ids, 'Size(GB)': size_strings, 'Size': size_gbs,
                       'Type': types, 'Downloaded': downloadeds, 'Filename': filenames})
    df.sort_values(by='Size', ascending=False, inplace=True)
    return df


def read_llm_config(folder: str=None) -> dict:
    """ Read Local LLM config and return object. Folder defaults to testing. """
    folder = (tst.staticdata_path() if folder is None else folder)
    config_file = Path(folder) / "llmconfig.json"
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


def from_pretrained(model_id: str, max_context_tokens: int=None) -> LocalModel:
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

    model.load(max_context_tokens)
    return model


if __name__ == "__main__":
    df = list_model_info()
    print(df)
    # model_id = "qwen3.5-0.8B"
    # print()
    # print(f"Chatting with: {model_id}")
    # # model_id = "tiny-gpt2"

    # # Load model
    # model = from_pretrained(model_id)

    # prompt = "Why is the sky blue?"
    # print()
    # print("Prompt:")
    # print(prompt)

    # print("\nBot:")
    # response = model.respond_prompt(prompt)

    # model.unload()

    # print()
    # print("Bot:")
    # print(response)
