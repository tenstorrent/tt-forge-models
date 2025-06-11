# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import urllib.parse
import hashlib
import requests
from pathlib import Path


def get_file(path):
    """Get a file from local filesystem, cache, or URL.

    This function handles both local files and URLs, retrieving from cache when available
    or downloading/fetching as needed. For URLs, it creates a unique cached filename using
    a hash of the URL to prevent collisions.

    Args:
        path: Path to a local file or URL to download

    Returns:
        Path to the file in the cache
    """
    # Check if path is a URL - handle URLs and files differently
    is_url = path.startswith(("http://", "https://"))

    if is_url:
        url = path
        # Create a hash from the URL to ensure uniqueness and use it to
        # prevent collisions in downloaded files with same filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]

        # Get filename from URL, or create one if not available
        file_name = os.path.basename(urllib.parse.urlparse(url).path)
        if not file_name:
            file_name = f"downloaded_file_{url_hash}"
        else:
            file_name = f"{url_hash}_{file_name}"

        # For URLs use a separate cache directory structure
        if (
            "DOCKER_CACHE_ROOT" in os.environ
            and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
        ):
            cache_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / "url_cache"
        elif "LOCAL_LF_CACHE" in os.environ:
            cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / "url_cache"
        else:
            cache_dir = Path.home() / ".cache/url_cache"
    else:
        rel_dir, file_name = os.path.split(path)
        if (
            "DOCKER_CACHE_ROOT" in os.environ
            and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
        ):
            cache_dir = (
                Path(os.environ["DOCKER_CACHE_ROOT"])
                / "models/tt-ci-models-private"
                / rel_dir
            )
        elif "LOCAL_LF_CACHE" in os.environ:
            cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_dir
        else:
            cache_dir = Path.home() / ".cache/lfcache" / rel_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / file_name

    # If file is not found in cache, download URL from web, or get file from IRD_LF_CACHE web server.
    if not file_path.exists():
        if is_url:
            try:
                print(f"Downloading {url} to {file_path}")
                response = requests.get(url, stream=True, timeout=(15, 60))
                response.raise_for_status()  # Raise exception for HTTP errors

                with open(file_path, "wb") as f:
                    f.write(response.content)

            except Exception as e:
                raise RuntimeError(f"Failed to download {url}: {str(e)}")
        elif "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            exit_code = os.system(
                f"wget -nH -np -R \"indexg.html*\" -P {cache_dir} {os.environ['IRD_LF_CACHE']}/{path} --connect-timeout=15 --read-timeout=60 --tries=3"
            )
            # Check for wget failure
            if exit_code != 0:
                raise RuntimeError(
                    f"wget failed with exit code {exit_code} when downloading {os.environ['IRD_LF_CACHE']}/{path}"
                )

            # Ensure file_path exists after wget command
            if not file_path.exists():
                raise RuntimeError(
                    f"Download appears to have failed: File {file_name} not found in {cache_dir} after wget command"
                )

    return file_path
