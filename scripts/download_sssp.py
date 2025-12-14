#!/usr/bin/env python3
"""
Task 1: SSSP Downloader
Downloads SSSP Precision (PBE) version 1.3.0 from Materials Cloud.
"""
import tarfile
from pathlib import Path

import requests
from loguru import logger

SSSP_URL = "https://archive.materialscloud.org/record/file?filename=SSSP_1.3.0_PBE_precision.tar.gz&record_id=63"
SSSP_FILENAME = "SSSP_1.3.0_PBE_precision.tar.gz"
SSSP_JSON_NAME = "SSSP_1.3.0_PBE_precision.json"
TARGET_DIR = Path("data/sssp")

def download_file(url: str, dest_path: Path):
    """Download a file with progress indication."""
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def setup_sssp():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    tarball_path = TARGET_DIR / SSSP_FILENAME
    json_path = TARGET_DIR / SSSP_JSON_NAME

    # Check if already installed (heuristic: json exists and some upf files exist)
    if json_path.exists() and len(list(TARGET_DIR.glob("*.upf"))) > 10:
        logger.info(f"SSSP appears to be installed in {TARGET_DIR}. Skipping download.")
        return

    # Download
    if not tarball_path.exists():
        try:
            download_file(SSSP_URL, tarball_path)
        except Exception:
            logger.error("Failed to download SSSP. The URL might be outdated or blocked.")
            logger.error(f"Please manually download '{SSSP_FILENAME}' from https://sssp.materialscloud.org/ or https://archive.materialscloud.org/")
            logger.error(f"and place it in {TARGET_DIR}.")
            return

    # Extract
    logger.info(f"Extracting {tarball_path}...")
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=TARGET_DIR)
        logger.info("Extraction complete.")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

    # Cleanup tarball to save space
    # tarball_path.unlink()

    logger.success(f"SSSP installed successfully in {TARGET_DIR}")
    logger.info(f"Metadata file: {json_path}")

if __name__ == "__main__":
    setup_sssp()
