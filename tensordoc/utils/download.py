from pathlib import Path

import requests

TENSORLAKE_CACHE_DIR = ".tensorlake"


def get_cache_directory() -> Path:
    """
    Get the cache directory from the home directory.
    Creates the cache directory if it does not exist.
    """
    home_directory = Path.home()
    cache_directory = home_directory / TENSORLAKE_CACHE_DIR
    if not cache_directory.exists():
        cache_directory.mkdir()

    return cache_directory


def check_file_exists(file_path: str) -> bool:
    """
    Check if the file exists in the TensorLake cache directory.
    """

    cache_directory = get_cache_directory()
    return cache_directory / file_path


def download_file(url: str, file_name: str):
    """
    Download the file from the URL and save it to the file path.
    """

    cache_directory = get_cache_directory()

    file_path = cache_directory / file_name

    if file_path.exists():
        return file_path

    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path
