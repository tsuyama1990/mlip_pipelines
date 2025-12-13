import json
import os
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

def load_sssp_db(path: str) -> Dict[str, Any]:
    """
    Load SSSP database from JSON file.

    Parameters
    ----------
    path : str
        Path to the SSSP JSON file.

    Returns
    -------
    Dict[str, Any]
        The SSSP database dictionary (e.g., mapping element -> info).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SSSP database file not found: {path}")

    try:
        with open(path_obj, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse SSSP JSON: {e}")

def validate_pseudopotentials(pseudo_dir: str, elements: List[str], sssp_db: Dict[str, Any]) -> None:
    """
    Validate that pseudopotential files exist for the given elements.

    Parameters
    ----------
    pseudo_dir : str
        Directory containing pseudopotential files.
    elements : List[str]
        List of chemical symbols to validate.
    sssp_db : Dict[str, Any]
        SSSP database dictionary. Expects keys to be elements and values to contain 'filename'.

    Raises
    ------
    FileNotFoundError
        If a pseudopotential file is missing.
    KeyError
        If an element is missing from the SSSP database.
    """
    pseudo_path = Path(pseudo_dir)
    if not pseudo_path.exists():
         raise FileNotFoundError(f"Pseudopotential directory not found: {pseudo_dir}")

    missing_files = []

    for elem in elements:
        if elem not in sssp_db:
            raise KeyError(f"Element {elem} not found in SSSP database.")

        # SSSP JSON structure assumption:
        # { "Si": { "filename": "Si.pbe-n-kjpaw_psl.1.0.0.UPF", ... }, ... }
        # Adapt based on actual SSSP format if different.
        # Standard SSSP/QE json usually has filename.

        info = sssp_db[elem]
        if isinstance(info, dict) and "filename" in info:
            filename = info["filename"]
        elif isinstance(info, str):
            # Maybe simple mapping?
            filename = info
        else:
             # Fallback or error
             logger.warning(f"Unexpected SSSP format for {elem}. Assuming it's a dict with 'filename'.")
             filename = info.get("filename", f"{elem}.upf")

        file_path = pseudo_path / filename
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing pseudopotential files: {', '.join(missing_files)}")

    logger.info("All pseudopotentials validated.")
