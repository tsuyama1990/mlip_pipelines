import re

with open("src/generators/extraction.py") as f:
    data = f.read()

# Add bounds validation for spherical extraction
extraction_replacement = """def _extract_spherical_zones(
    atoms: Atoms, center_idx: int, core_r: float, buffer_r: float
) -> Atoms:
    if not (0 <= center_idx < len(atoms)):
        msg = f"center_idx {center_idx} is out of bounds for atoms of length {len(atoms)}."
        raise ValueError(msg)
    if core_r <= 0 or buffer_r <= 0:
        msg = "core_r and buffer_r must be strictly positive."
        raise ValueError(msg)
    if core_r >= buffer_r:
        msg = "core_r must be strictly smaller than buffer_r."
        raise ValueError(msg)
    if buffer_r > 50.0:
        msg = "buffer_r exceeds maximum allowed cutoff limit of 50.0 A."
        raise ValueError(msg)

    i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", atoms, buffer_r)

    mask = i_arr == center_idx
    neighbors_j = j_arr[mask]
    distances = d_arr[mask]
    vectors = D_arr[mask]

    valid_mask = neighbors_j != center_idx
    neighbors_j = neighbors_j[valid_mask]
    distances = distances[valid_mask]
    vectors = vectors[valid_mask]

    cluster_indices: np.ndarray = np.append(neighbors_j, center_idx)

    if len(cluster_indices) > 5000:
        msg = f"Extraction resulted in an excessively large cluster ({len(cluster_indices)} atoms)."
        raise ValueError(msg)

    cluster_distances: np.ndarray = np.append(distances, 0.0)
    cluster_vectors = np.vstack([vectors, np.zeros(3)]) if len(vectors) > 0 else np.zeros((1, 3))"""

data = re.sub(
    r"def _extract_spherical_zones.*?cluster_vectors = np\.vstack\(\[vectors, np\.zeros\(3\)\]\) if len\(vectors\) > 0 else np\.zeros\(\(1, 3\)\)",
    extraction_replacement,
    data,
    flags=re.DOTALL,
)

with open("src/generators/extraction.py", "w") as f:
    f.write(data)
