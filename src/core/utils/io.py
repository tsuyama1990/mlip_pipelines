import json
from pathlib import Path
from typing import Dict, Any
from ase import Atoms
from ase.io import write
from loguru import logger
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def save_results(result_dict: Dict[str, Any], settings: Any, output_dir: Path):
    """
    Save optimization results and configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Final Structure
    if "final_structure" in result_dict and isinstance(result_dict["final_structure"], Atoms):
        atoms = result_dict["final_structure"]
        # XYZ with energy/forces
        write(output_dir / "final_structure.xyz", atoms, format="extxyz")
        # CIF for visualization
        write(output_dir / "final_structure.cif", atoms, format="cif")
        logger.info(f"Saved structures to {output_dir}")

    # Save Trajectory if present (handled by Relaxer usually, but good to ensure)
    if "trajectory" in result_dict:
        # If trajectory is list of Atoms
        traj = result_dict["trajectory"]
        if traj and isinstance(traj, list) and isinstance(traj[0], Atoms):
             write(output_dir / "trajectory.xyz", traj, format="extxyz")

    # Save Config
    # Check if settings is pydantic model
    if hasattr(settings, "model_dump"):
        config_dict = settings.model_dump()
    elif hasattr(settings, "dict"):
        config_dict = settings.dict()
    else:
        config_dict = str(settings)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4, cls=NumpyEncoder)

    # Save Results Summary
    summary = {k: v for k, v in result_dict.items() if k not in ["final_structure", "trajectory"]}
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)

    logger.info("Saved config and results summary.")
