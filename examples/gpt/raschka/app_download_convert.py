""" Download the GPT2 weights as released by Open AI. Then create a GPT2 model object in PyTorch and save its weights.
    This allows a conversion of the weight format from the format released by Open AI using tensorflow to
    the PyTorch format which we use in the rest of the library.
"""
import os, gc
from pathlib import Path
import urllib.request
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all messages except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
import tensorflow as tf
from tqdm import tqdm
import torch
from sdevpy.llms.gpt import gpt
from sdevpy.utilities import jsonmanager as jsm


def download_and_load_gpt2(model_size: str, model_dir: str):
    """ Download Open AI model weights and retrieve settings/parameters """
    download_gpt2(model_size, model_dir)
    model_dir = os.path.join(model_dir, model_size)
    return load_gpt2(model_dir)


def load_gpt2(model_dir: str):
    """ Retrieve settings/parameters given model save location """
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def download_gpt2(model_size: str, model_dir: str):
    """ Download Open AI model weights """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(model_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


if __name__ == "__main__":
    # Choose export path and model size
    project_path = Path(os.environ.get('SDEVPY_DATA', Path.home() / 'sdevpy'))
    project_path = project_path / "llms" / "gpt"
    model_size = "355M" # 124M, 355M, 774M, 1558M

    print(f"Project path: {project_path}")

    # Retrieve model topology and parameters
    settings, params = download_and_load_gpt2(model_size, project_path)
    print("Settings: ", settings)
    print("Param dict keys: ", params.keys())

    # Create model
    print("Creating model from its topology")
    GPT_CONFIG = {"vocab_size": settings['n_vocab'], "context_length": settings['n_ctx'],
                  "emb_dim": settings['n_embd'], "n_heads": settings['n_head'],
                  "n_layers": settings['n_layer'], "drop_rate": 0.1, "qkv_bias": True}
    model = gpt.GPTModel(GPT_CONFIG)
    model.eval() # Not sure we really need this

    # Load parameters into model
    print("Loading weights for GPT-2 model size: " + model_size)
    gpt.load_weights(model, params)
    print("Done loading weights!")

    # Save model into PyTorch format
    model_path = project_path / f"gpt2-{model_size}"
    model_path.mkdir(parents=True, exist_ok=True)
    weight_file = model_path / "weights.pth"
    torch.save({"model_state_dict": model.state_dict()}, weight_file)
    config_file = model_path / "config.json"
    jsm.serialize(GPT_CONFIG, config_file)

    # Delete model from memory
    del model
    gc.collect()
