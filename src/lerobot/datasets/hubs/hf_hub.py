from pathlib import Path

from huggingface_hub import HfApi

from .abstract_hub import (
    DEFAULT_DOWNLOAD_ALLOW_PATTERNS,
    DEFAULT_DOWNLOAD_IGNORE_PATTERNS,
    AbstractDownloadHub,
)


class HuggingfaceDownloadHub(AbstractDownloadHub):
    """
    Implementation of AbstractDownloadHub for Hugging Face dataset downloads.

    This class provides functionality to download datasets from Hugging Face Hub
    with pattern-based file filtering.

    Attributes:
        hub (HfApi): Hugging Face API client instance.
    """

    def __init__(self) -> None:
        """
        Initialize the HuggingfaceDownloadHub.

        This method initializes the HuggingfaceDownloadHub by setting up the
        Hugging Face API client.
        """
        self.hub = HfApi()

    def repo_exists(self, repo_id: str) -> bool:
        """
        Check if a dataset repository exists on Hugging Face Hub.

        Args:
            repo_id (str): Identifier of the repository to check.

        Returns:
            bool: True if repository exists, False otherwise.
        """
        return self.hub.repo_exists(repo_id=repo_id, repo_type="dataset")

    def download_repo_with_patterns(
        self,
        repo_id: str,
        download_path: Path,
    ) -> None:
        """
        Download a repository from Hugging Face Hub with specific file patterns.

        Args:
            repo_id (str): Identifier of the repository to download.
            download_path (Path): Local path where the repository will be downloaded.

        Raises:
            Exception: If download fails for any reason.
        """
        download_path = download_path.expanduser().absolute()
        try:
            self.hub.snapshot_download(
                repo_id=repo_id,
                local_dir=download_path,
                allow_patterns=DEFAULT_DOWNLOAD_ALLOW_PATTERNS,
                ignore_patterns=DEFAULT_DOWNLOAD_IGNORE_PATTERNS,
                repo_type="dataset",
                token=False,
            )

        except Exception as e:
            raise e from e
