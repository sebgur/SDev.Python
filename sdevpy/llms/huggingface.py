from huggingface_hub import scan_cache_dir, constants, HfApi


def list_available_models() -> list[dict]:
    """ List locally available models and sizes """
    cache_info = scan_cache_dir()
    models = []
    for repo in cache_info.repos:
        models.append({'repo_id': repo.repo_id, 'size_on_disk': repo.size_on_disk})

    # Sort by size
    models.sort(key=lambda x: x['size_on_disk'], reverse=True)

    # Add other size units
    for model in models:
        size_gb = model['size_on_disk'] / (1024**3)
        size_str = f"{size_gb:.2f}"
        model['size_gb'] = size_gb
        model['size_string'] = size_str

    return models


def print_available_models() -> None: # pragma: no cov
    """ Print list of available models and info to screen """
    models = list_available_models()
    print()
    print("Available models (size on disk in GB):")
    for model in models:
        print(f"  * {model['repo_id']} ({model['size_string']}GB)")


def model_location() -> str:
    """ Local path to where the models are stored """
    return constants.HF_HUB_CACHE


def list_model_files(repo_id: str) -> list[str]:
    """ List available files for given model (repo ID) """
    api = HfApi()
    files = list(api.list_repo_files(repo_id))
    return files


if __name__ == "__main__":
    repo_id = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
    model_files = list_model_files(repo_id)
    print(model_files)
