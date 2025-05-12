import shutil
import zipfile
from pathlib import Path

from ..common import Config, get_file_hash
from .core import logger


def unpack_dataset(dataset_path: Path, module: str) -> Path:
    """Unzip the dataset to the cache directory, verifying the hash."""

    # Verify if the dataset path is a valid zip file
    if not zipfile.is_zipfile(dataset_path):
        logger.error(f"Invalid zip file: {dataset_path}")
        raise ValueError("Dataset path must be a zip file.")

    zip_hash = get_file_hash(dataset_path)
    logger.debug(f"Dataset zip hash: {zip_hash}")

    # Define cache directory and hash store
    cache_dir = Path(Config.get("global_cache_dir"), module)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hash_store = Path(cache_dir, "dataset.cache")

    # Handle hash comparison and cache management
    if hash_store.is_file():
        with open(hash_store, "r") as f:
            stored_hash = f.read()

        if stored_hash == zip_hash:
            logger.debug("Dataset already unpacked and newest version. Using cache.")
            return cache_dir
        else:
            logger.debug(
                f"Hash mismatch! Recreating cache and unpacking new dataset: {zip_hash}"
            )
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(hash_store, "w") as f:
                f.write(zip_hash)
    else:
        logger.debug(f"Hash store not found: {hash_store}. Creating new one.")
        with open(hash_store, "w") as f:
            f.write(zip_hash)

    # Unzip the dataset to the cache directory
    try:
        logger.debug(f"Unzipping dataset to {cache_dir}")
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
    except (zipfile.BadZipFile, Exception) as e:
        logger.error(f"Error unzipping dataset: {e}")
        raise ValueError("Failed to unzip dataset.") from e

    logger.debug(f"Dataset unpacked to {cache_dir}")
    return cache_dir
