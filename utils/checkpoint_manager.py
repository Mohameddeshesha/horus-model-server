from pathlib import Path

from huggingface_hub import hf_hub_download

from config import HF_FILES, HF_REPOSITORY, HF_TOKEN


def ensure_checkpoint(model_id: str, destination: Path) -> Path:
    """
    Ensure a checkpoint exists locally.
    Downloads it from Hugging Face if necessary.
    """

    if destination.exists():
        print(f"[checkpoint] Found local checkpoint: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"[checkpoint] Downloading {model_id}...")

    downloaded = hf_hub_download(
        repo_id=HF_REPOSITORY,
        filename=HF_FILES[model_id],
        token=HF_TOKEN if HF_TOKEN else None,
        local_dir=destination.parent,
        local_dir_use_symlinks=False,
    )

    downloaded = Path(downloaded)

    if downloaded != destination:
        downloaded.rename(destination)

    print(f"[checkpoint] Download completed.")

    return destination