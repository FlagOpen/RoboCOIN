import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
from pathlib import Path  # noqa: E402

from .abstract_hub import (
    DEFAULT_DOWNLOAD_ALLOW_PATTERNS,
    DEFAULT_DOWNLOAD_IGNORE_PATTERNS,
    AbstractDownloadHub,
)


class ModelscopeDownloadHub(AbstractDownloadHub):
    """
    Implementation of AbstractDownloadHub for ModelScope dataset downloads.

    This class provides functionality to download datasets from ModelScope Hub
    with pattern-based file filtering.

    Attributes:
        hub (HubApi): ModelScope API client instance.
    """

    from modelscope.hub.api import HubApi

    def __init__(self) -> None:
        self.hub = HubApi()

    def repo_exists(self, repo_id: str) -> bool:
        """
        Check if a dataset repository exists on ModelScope Hub.

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
        Download a repository from ModelScope Hub with specific file patterns.

        Args:
            repo_id (str): Identifier of the repository to download.
            download_path (Path): Local path where the repository will be downloaded.

        Raises:
            Exception: If download fails for any reason.
        """
        from modelscope.hub.snapshot_download import snapshot_download

        download_path = download_path.expanduser().absolute()

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(download_path),
                allow_patterns=DEFAULT_DOWNLOAD_ALLOW_PATTERNS,
                ignore_patterns=DEFAULT_DOWNLOAD_IGNORE_PATTERNS,
                repo_type="dataset",
            )
        except Exception as e:
            raise e from e
