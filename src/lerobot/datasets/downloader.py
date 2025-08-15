import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import draccus
from tqdm import tqdm

from lerobot.constants import DS_PLATFORM_NAME, HF_LEROBOT_HOME

from .hubs.hf_hub import HuggingfaceDownloadHub
from .hubs.ms_hub import ModelscopeDownloadHub

DEFAULT_DOWNLOAD_PATH = HF_LEROBOT_HOME


class DatasetsHubEnum(str, Enum):
    """
    Enumeration of supported dataset hub platforms.

    Attributes:
        HUGGINGFACE: Huggingface platform identifier.
        MODELSCOPE: Modelscope platform identifier.
    """

    HUGGINGFACE = "Huggingface"
    MODELSCOPE = "Modelscope"


@dataclass()
class DownLogConfig:
    """
    Configuration class for logging settings.

    This class defines the configuration parameters for logging in the application,
    including log directory, console output options, and log level.

    Attributes:
        log_dir (str): Directory path where log files will be stored. Defaults to empty string.
        log_to_console (bool): Flag indicating whether logs should be output to console. Defaults to True.
        log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR"). Defaults to "INFO".
    """

    log_dir: str = ""
    log_to_console: bool = True
    log_level: str = "INFO"


@dataclass
class DsDownloadConfig:
    download_log: DownLogConfig = field(default_factory=DownLogConfig)
    hub: DatasetsHubEnum = DatasetsHubEnum.HUGGINGFACE
    download_root_path: Path = Path(f"{DEFAULT_DOWNLOAD_PATH}/{hub}")
    ds_names: str = field(default_factory=str)


class DsDownloadUtil:
    def __init__(self, config: DsDownloadConfig) -> None:
        """
        Initialize the downloader with configuration.

        Args:
            config (DsDownloadConfig): Configuration object for the downloader.

        Raises:
            Exception: If the specified hub platform is not supported.
        """
        self.config: DsDownloadConfig = config
        self.logger = self._setup_logger()

        if config.hub == DatasetsHubEnum.MODELSCOPE:
            self.hub = ModelscopeDownloadHub()
        elif config.hub == DatasetsHubEnum.HUGGINGFACE:
            self.hub = HuggingfaceDownloadHub()
        else:
            raise Exception(f"hub {config.hub} is not supported.")
        pass

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"DsDownloadUtil-{self.config.hub}")
        logger.setLevel(getattr(logging, self.config.download_log.log_level.upper()))

        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        log_dir = Path(self.config.download_log.log_dir).joinpath(f"{self.config.hub}")
        log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"ds_download_{timestamp}.log"

        log_filepath = log_dir / log_filename

        if self.config.download_log.log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, self.config.download_log.log_level.upper()))
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # Add file handler
        fh = logging.FileHandler(log_filepath, encoding="utf-8")
        fh.setLevel(getattr(logging, self.config.download_log.log_level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Optional: Log the file path
        logger.info(f"Logging enabled, output to: {log_filepath}")

        return logger

    def _get_dsname_list(self) -> list[str]:
        return [item for item in re.split(r"[,\s]+", self.config.ds_names.strip()) if item]

    def download_datasets(self) -> None:
        root_path = self.config.download_root_path.expanduser().absolute()
        if not root_path.exists():
            self.logger.info(f"download root path {root_path} does not exists, try to create it.")
            try:
                root_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"mkdir {root_path} success")
            except Exception as e:
                self.logger.error(f"failed to mkdir {root_path}: {e}")
                raise e from Exception(f"failed to mkdir {root_path}: {e}")

        ds_list = self._get_dsname_list()
        for ds_name in tqdm(
            ds_list,
            desc=f"Download {DS_PLATFORM_NAME} datasets",
            bar_format="\033[32m{l_bar}{bar}\033[0m{r_bar}",
        ):
            repo_id = f"{DS_PLATFORM_NAME}/{ds_name}"
            if not self.hub.repo_exists(repo_id):
                self.logger.warning(
                    f"repo {repo_id} does not exists in {self.config.hub} hub, please check error"
                )
                continue

            download_path = Path(f"{self.config.download_root_path}/{ds_name}")
            try:
                self.hub.download_repo_with_patterns(repo_id=repo_id, download_path=download_path)
                self.logger.info(f"dataset {ds_name} downloaded successfully to {download_path}")
            except KeyboardInterrupt:
                self.logger.info("Download interrupted by user.")
                break
            except Exception as e:
                self.logger.info(f"dataset {ds_name} download failed: {e}")


if __name__ == "__main__":
    config = draccus.parse(DsDownloadConfig)
    downloader = DsDownloadUtil(config)
    downloader.download_datasets()
    pass


"""
# usage:

# you can download datasets by specifing datasets names, which can be searched at https://robocoin.github.io/search.index
# the default download hub is huggingface
python -m robocoin.Datasets.download  --ds_names "aaa, bbb, ccc"

# you can also download datasets from modelscope
python -m robocoin.Datasets.download  --ds_names "aaa, bbb, ccc" --hub modelscope

# the default download saving path is ~/.cache/RoboCoin/huggingface/, or you can specify your download path
python -m robocoin.Datasets.download  --ds_names "aaa, bbb, ccc" --root_path ./datas/

# for more config info, please refer to ./configs/download.yaml
python -m robocoin.datasets.download --config configs/download.yaml

"""
