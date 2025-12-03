#!/usr/bin/env python3
"""
Primary CLI for the hub download helper.

This script owns argparse handling and implements the reusable download helpers
for HuggingFace and ModelScope so they can be imported from a single entry point.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

DEFAULT_NAMESPACE = "RoboCOIN"
DEFAULT_MAX_RETRIES = 5
DEFAULT_SLEEP_SECONDS = 5
MAX_SLEEP_SECONDS = 120
DEFAULT_OUTPUT_DIR = "~/.cache/huggingface/lerobot/"

# Setup logging with file output
_log_dir = Path("logs/download")
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_log_dir / "download.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), _file_handler],
)
LOGGER = logging.getLogger("hub-download")


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download datasets from HuggingFace or ModelScope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hub", required=True, choices=["huggingface", "modelscope"])
    parser.add_argument("--ds_lists", nargs="+", help="Dataset names provided on the CLI.")
    parser.add_argument("--ds_file", help="Optional text file with one dataset per line.")
    parser.add_argument("--namespace", help="Hub namespace/owner.", default=None)
    parser.add_argument(
        "--output_dir",
        "--target-dir",
        dest="output_dir",
        default=None,
        help=f"Where datasets should be stored. If not provided, uses the default directory: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--token", help="Authentication token (else env vars are used).")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers for downloading. Used for both HuggingFace and ModelScope.",
    )
    parser.add_argument(
        "--max_retry_time",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum number of retry attempts per dataset.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print plan and exit.")
    return parser


def _resolve_output_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_namespace(namespace: str | None) -> str:
    return namespace or DEFAULT_NAMESPACE


# --------------------------------------------------------------------------- #
# Dataset helper implementations
# --------------------------------------------------------------------------- #
def _read_dataset_names(cli_values: Iterable[str] | None, file_path: str | None) -> list[str]:
    names: list[str] = []

    if cli_values:
        names.extend(cli_values)

    if file_path:
        parsed_path = Path(file_path).expanduser().resolve()
        if not parsed_path.exists():
            raise FileNotFoundError(f"Dataset list not found: {parsed_path}")
        for line in parsed_path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                names.append(item)

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)
    return ordered_unique


def _retry_loop(label: str, max_retries: int, fn: Callable[[], Path]) -> Path:
    sleep_time = DEFAULT_SLEEP_SECONDS
    last_exc: Exception | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        try:
            LOGGER.info(f"{label}: attempt {attempt}")
            return fn()
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            remaining_attempts = max_retries - attempt
            if remaining_attempts <= 0:
                break
            wait = sleep_time
            LOGGER.warning(
                "%s: failed (%s); retrying in %ds (%d attempt(s) left)",
                label,
                exc,
                int(wait),
                remaining_attempts,
            )
            time.sleep(wait)
            sleep_time = min(sleep_time * 2, MAX_SLEEP_SECONDS)

    raise RuntimeError(f"{label}: download timeout after {max_retries} attempt(s)") from last_exc


def _resolve_token(hub: Literal["huggingface", "modelscope"], explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if hub == "huggingface":
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return os.environ.get("MODELSCOPE_TOKEN") or os.environ.get("MODELSCOPE_API_TOKEN")


def _prepare_dataset_list(dataset_names: Iterable[str]) -> list[str]:
    datasets = list(dataset_names)
    if not datasets:
        raise ValueError("No datasets provided.")
    return datasets


def _log_download_plan(
    hub: Literal["huggingface", "modelscope"],
    namespace: str,
    out_dir: Path,
    datasets: Iterable[str],
    max_retries: int,
    token_provided: bool,
) -> None:
    LOGGER.info("Hub: %s", hub)
    LOGGER.info("Namespace: %s", namespace)
    LOGGER.info("Output: %s", out_dir)
    LOGGER.info("Datasets: %s", ", ".join(datasets))
    LOGGER.info("Retry budget: %d attempt(s) per dataset", int(max_retries))
    LOGGER.info("Token: %s", "provided" if token_provided else "not provided")


def _download_requested_datasets(
    *,
    hub: Literal["huggingface", "modelscope"],
    datasets: list[str],
    out_dir: Path,
    namespace: str | None,
    token: str | None,
    max_workers: int,
    max_retries: int,
) -> list[str]:
    failures: list[str] = []
    for idx, name in enumerate(datasets, 1):
        LOGGER.info("[%d/%d] %s", idx, len(datasets), name)
        try:
            path = download_dataset(
                hub=hub,
                dataset_name=name,
                output_dir=out_dir,
                namespace=namespace,
                token=token,
                max_workers=max_workers,
                max_retries=max_retries,
            )
            LOGGER.info("Completed: %s --> %s", name, path)
        except Exception as exc:  # noqa: PERF203
            LOGGER.error("Failed: %s (%s)", name, exc)
            failures.append(name)
    return failures


def _ensure_gate_dataset(
    *,
    hub: Literal["huggingface", "modelscope"],
    namespace: str,
    out_dir: Path,
    token: str | None,
    max_workers: int,
    max_retries: int,
) -> None:
    gate_name = "gate"
    gate_repo_id = f"{namespace}/{gate_name}"
    gate_path = out_dir / namespace / gate_name


    # Check if gate dataset already exists
    if gate_path.exists() and any(gate_path.rglob("*")):
        LOGGER.info("Gate dataset already exists at: %s", gate_path)
        LOGGER.info("Verifying gate dataset access...")
    else:
        LOGGER.info("Gate dataset not found. Attempting to download mandatory dataset %s from %s", gate_repo_id, hub)

    try:
        gate_path = download_dataset(
            hub=hub,
            dataset_name=gate_name,
            output_dir=out_dir,
            namespace=namespace,
            token=token,
            max_workers=max_workers,
            max_retries=1,
            enable_retry=False,
        )
        LOGGER.error("============================================================")
        LOGGER.error(" GATE CHECK PASSED — THANK YOU FOR SUPPORTING ROBOCOIN")
        LOGGER.error("============================================================")
        LOGGER.error("Gate dataset is ready at: %s", gate_path)
        LOGGER.error("Your consent keeps RoboCOIN sustainable and region-aware.")
        LOGGER.error("Proceeding with the remaining dataset downloads...")
        LOGGER.error("------------------------------------------------------------")
    except Exception as exc:  # noqa: PERF203
        if hub == "huggingface":
            gate_url = f"https://huggingface.co/datasets/{gate_repo_id}"
        else:
            gate_url = f"https://modelscope.cn/datasets/{gate_repo_id}"

        LOGGER.error("============================================================")
        LOGGER.error(" GATE DATASET ACCESS REQUIRED — PLEASE COMPLETE STATISTICS FORM")
        LOGGER.error("============================================================")
        LOGGER.error("To improve RoboCOIN’s regional coverage and understand how the data")
        LOGGER.error("is used, we need a one-time, lightweight consent submission before")
        LOGGER.error("any other datasets can be downloaded. Please visit the following link")
        LOGGER.error("and fill out the brief form, then re-run this command:")
        LOGGER.error("  %s", gate_url)
        LOGGER.error("The information is collected solely via the official Hugging Face flow")
        LOGGER.error("and will never be used for unrelated purposes. Your response helps us")
        LOGGER.error("prioritize support and keep the project sustainable. Thank you!")
        LOGGER.error("------------------------------------------------------------")
        LOGGER.error("Technical tips:")
        LOGGER.error("  - Ensure you have granted access at the URL above")
        LOGGER.error("  - If the dataset is private, confirm your token and permissions")
        LOGGER.error("  - Verify network connectivity and try again")
        LOGGER.error("Original error: %s: %s", type(exc).__name__, exc)
        raise RuntimeError(f"Gate dataset '{gate_repo_id}' download failed") from exc


# --------------------------------------------------------------------------- #
# Hub specific downloaders
# --------------------------------------------------------------------------- #
def _download_from_hf(repo_id: str, target_dir: Path, token: str | None, max_workers: int) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("huggingface_hub is missing: pip install huggingface_hub") from exc

    def _run() -> Path:
        try:
            # 让 huggingface_hub 自己创建 / 管理目录，避免在仓库不存在时生成空目录
            download_kwargs = {
                "repo_id": repo_id,
                "repo_type": "dataset",
                "token": token,
                "resume_download": True,
                "max_workers": max_workers,
                "local_dir": str(target_dir),
            }
            path = snapshot_download(**download_kwargs)
            return Path(path)
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Repository not found: {repo_id}\n"
                f"  - Check the dataset name and namespace are correct\n"
                f"  - Verify the repo exists at https://huggingface.co/datasets/{repo_id}\n"
                f"  - If the repo is private, ensure you have access and a valid token"
            ) from exc
        except HfHubHTTPError as exc:
            if exc.response.status_code == 401:
                raise RuntimeError(
                    f"Authentication failed for {repo_id}\n"
                    f"  - The repo may be private and require authentication\n"
                    f"  - Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable\n"
                    f"  - Or pass --token with a valid HuggingFace token\n"
                    f"  - Get your token from: https://huggingface.co/settings/tokens"
                ) from exc
            if exc.response.status_code == 403:
                raise RuntimeError(
                    f"Access forbidden to {repo_id}\n"
                    f"  - You may not have permission to access this dataset\n"
                    f"  - If this is a private dataset, request access from the owner"
                ) from exc
            raise

    return _run()


def _download_from_ms(repo_id: str, target_dir: Path, token: str | None, max_workers: int) -> Path:
    try:
        from modelscope import dataset_snapshot_download
        from modelscope.hub.api import HubApi
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("modelscope is missing: pip install modelscope") from exc

    # Check if datasets module is available (ModelScope requires it internally)
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "datasets module is missing but required by ModelScope\n"
            "  - Install it with: pip install datasets\n"
            "  - Or install the full lerobot package: pip install -e ."
        )

    def _run() -> Path:
        LOGGER.info("ModelScope: attempting to download dataset_id=%s", repo_id)
        LOGGER.debug("  local_dir=%s", target_dir)

        try:
            if token:
                LOGGER.info("Logging in to ModelScope with provided token")
                HubApi().login(token)

            # Use dataset_snapshot_download for downloading dataset files
            # This downloads all raw files from the dataset repository
            LOGGER.info("Downloading dataset using dataset_snapshot_download...")
            download_kwargs = {
                "dataset_id": repo_id,
                "local_dir": str(target_dir),
            }
            # ModelScope may support max_workers parameter for parallel downloads
            # If the API doesn't support it, it will be silently ignored
            if max_workers > 1:
                download_kwargs["max_workers"] = max_workers
                LOGGER.debug("Using max_workers=%d for ModelScope download", max_workers)
            path = dataset_snapshot_download(**download_kwargs)

            # The dataset files are now downloaded to target_dir (or default cache)
            LOGGER.info("Dataset downloaded successfully to %s", path)
            return Path(path)

        except Exception as exc:
            # Log the full exception details for debugging
            LOGGER.error("ModelScope exception type: %s", type(exc).__name__)
            LOGGER.error("ModelScope exception details: %s", exc)

            # ModelScope exceptions are less standardized, provide helpful context
            # But only when we're confident about the error type to avoid false positives
            error_msg = str(exc).lower()

            # Only treat as "not found" if it's clearly a repo/model not found error
            # Be more specific to avoid false positives from file path errors
            if ("not found" in error_msg and ("repository" in error_msg or "model" in error_msg or "dataset" in error_msg)) or \
               ("404" in error_msg and "http" in error_msg):
                raise RuntimeError(
                    f"Dataset not found: {repo_id}\n"
                    f"  - Check the dataset name and namespace are correct\n"
                    f"  - Verify the dataset exists at https://modelscope.cn/datasets/{repo_id}\n"
                    f"  - If the dataset is private, ensure you have access and a valid token\n"
                    f"  - Original error: {type(exc).__name__}: {exc}"
                ) from exc
            if ("unauthorized" in error_msg or "401" in error_msg) or \
                 ("forbidden" in error_msg or "403" in error_msg and "http" in error_msg):
                raise RuntimeError(
                    f"Authentication/authorization failed for {repo_id}\n"
                    f"  - The dataset may be private and require authentication\n"
                    f"  - Set MODELSCOPE_TOKEN or MODELSCOPE_API_TOKEN environment variable\n"
                    f"  - Or pass --token with a valid ModelScope token\n"
                    f"  - You can get your token from: https://modelscope.cn/my/account\n"
                    f"  - Original error: {type(exc).__name__}: {exc}"
                ) from exc
            # Check for missing datasets module error (common ModelScope issue)
            if "no module named 'datasets'" in error_msg or "No module named 'datasets'" in str(exc):
                raise RuntimeError(
                    f"ModelScope requires the 'datasets' module but it's not available\n"
                    f"  - Install it with: pip install datasets\n"
                    f"  - Or install the full lerobot package: pip install -e .\n"
                    f"  - Original error: {type(exc).__name__}: {exc}"
                ) from exc

            # For all other errors, preserve the original exception with context
            raise RuntimeError(
                f"ModelScope dataset download failed for {repo_id}\n"
                f"  - Exception type: {type(exc).__name__}\n"
                f"  - Error details: {exc}\n"
                f"  - This may be a network issue, file system error, or other problem\n"
                f"  - Verify the dataset exists at: https://modelscope.cn/datasets/{repo_id}"
            ) from exc

    return _run()


def download_dataset(
    hub: Literal["huggingface", "modelscope"],
    dataset_name: str,
    output_dir: Path,
    namespace: str | None,
    token: str | None,
    max_workers: int,
    max_retries: int,
    enable_retry: bool = True,
) -> Path:
    namespace = _resolve_namespace(namespace)
    repo_id = f"{namespace}/{dataset_name}"
    
    # Create a subdirectory for this dataset, preserving namespace structure.
    # This ensures consistent directory structure across HuggingFace and ModelScope:
    # - HuggingFace default: ~/.cache/huggingface/hub/datasets--RoboCOIN--dataset/ (uses -- separator)
    # - ModelScope default: ~/.cache/modelscope/hub/datasets/RoboCOIN/dataset/ (preserves / structure)
    # By explicitly specifying local_dir with namespace/dataset structure, both platforms
    # will use the same consistent path: output_dir/namespace/dataset_name/
    dataset_path: Path = output_dir / namespace / dataset_name

    LOGGER.info("Downloading repo_id: %s from %s", repo_id, hub)
    LOGGER.debug("Target path: %s", dataset_path)
    LOGGER.debug("Token provided: %s", bool(token))

    def _perform_download() -> Path:
        if hub == "huggingface":
            return _download_from_hf(repo_id, dataset_path, token, max_workers)
        if hub == "modelscope":
            return _download_from_ms(repo_id, dataset_path, token, max_workers)
        raise ValueError(f"Unsupported hub: {hub}")

    if enable_retry:
        return _retry_loop(f"{hub}:{repo_id}", max_retries, _perform_download)
    return _perform_download()


def download_datasets(
    hub: Literal["huggingface", "modelscope"],
    dataset_names: Iterable[str],
    output_dir: Path | str | None,
    namespace: str | None = None,
    token: str | None = None,
    max_workers: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[str]:
    """
    Download multiple datasets, returning a list of failures (if any).

    Args:
        hub: Target hub name.
        dataset_names: Iterable of dataset identifiers (unique entries recommended).
        output_dir: Directory where dataset folders will be stored. If None, uses the default directory: ~/.cache/huggingface/lerobot/
        namespace: Optional namespace override.
        token: Optional authentication token, falling back to env vars when None.
        max_workers: Maximum number of parallel workers for downloading (used for both HuggingFace and ModelScope).
        max_retries: Maximum attempts per dataset (including the first try).
    """
    datasets = _prepare_dataset_list(dataset_names)
    out_dir = _resolve_output_dir(output_dir or DEFAULT_OUTPUT_DIR)
    resolved_namespace = _resolve_namespace(namespace)
    resolved_token = _resolve_token(hub, token)
    safe_workers = max(1, max_workers)
    safe_retries = int(max_retries)

    _log_download_plan(
        hub=hub,
        namespace=resolved_namespace,
        out_dir=out_dir,
        datasets=datasets,
        max_retries=safe_retries,
        token_provided=bool(resolved_token),
    )

    failures = _download_requested_datasets(
        hub=hub,
        datasets=datasets,
        out_dir=out_dir,
        namespace=namespace,
        token=resolved_token,
        max_workers=safe_workers,
        max_retries=safe_retries,
    )

    if failures:
        LOGGER.error("Failed datasets: %s", ", ".join(failures))
    else:
        LOGGER.info("All datasets downloaded successfully.")

    return failures


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset_names = _read_dataset_names(args.ds_lists, args.ds_file)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if not dataset_names:
        parser.error("No datasets supplied. Use --ds_lists and/or --ds_file.")

    # Use default output directory if not provided
    if args.output_dir is None:
        output_dir = _resolve_output_dir(DEFAULT_OUTPUT_DIR)
    else:
        output_dir = _resolve_output_dir(args.output_dir)

    if args.dry_run:
        LOGGER.info("Dry run")
        LOGGER.info("  Hub: %s", args.hub)
        LOGGER.info("  Namespace: %s", args.namespace or DEFAULT_NAMESPACE)
        LOGGER.info("  Output: %s", output_dir)
        LOGGER.info("  Datasets (%d): %s", len(dataset_names), ", ".join(dataset_names))
        LOGGER.info("  Max retries: %d", args.max_retry_time)
        LOGGER.info("  Token: %s", "provided" if args.token else "not provided")
        return 0

    # Perform gate check before actual download (HuggingFace only)
    resolved_namespace = _resolve_namespace(args.namespace)
    resolved_token = _resolve_token(args.hub, args.token)
    if args.hub == "huggingface":
        try:
            _ensure_gate_dataset(
                hub=args.hub,
                namespace=resolved_namespace,
                out_dir=output_dir,
                token=resolved_token,
                max_workers=max(1, args.max_workers),
                max_retries=int(args.max_retry_time),
            )
            LOGGER.error("Gate check completed successfully. Proceeding with dataset downloads...")
        except RuntimeError as exc:
            # Gate dataset failure – abort cleanly before downloading other datasets
            LOGGER.error("Download aborted due to gate check failure: %s", exc)
            return 1

    try:
        failures = download_datasets(
            hub=args.hub,
            dataset_names=dataset_names,
            output_dir=output_dir,
            namespace=args.namespace,
            token=args.token,
            max_workers=max(1, args.max_workers),
            max_retries=int(args.max_retry_time),
        )
    except RuntimeError as exc:
        # Gate dataset failure (or other fatal download error) – abort cleanly.
        LOGGER.error("Download aborted: %s", exc)
        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
