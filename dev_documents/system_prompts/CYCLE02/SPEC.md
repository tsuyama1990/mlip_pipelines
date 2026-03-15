# CYCLE02 Specification

## 1. Summary

CYCLE02 addresses one of the most critical and difficult physical challenges in Active Learning for molecular dynamics: the intelligent extraction and robust passivation of local atomic clusters. In Phase 01, naive cluster extraction often resulted in highly unphysical dangling bonds at the vacuum interface. When these damaged, highly reactive clusters were sent to the DFT Oracle, they routinely caused severe Self-Consistent Field (SCF) convergence failures, or worse, resulted in "garbage" forces being learned by the ACE potential, polluting the entire training set and degrading the model's accuracy.

To definitively solve this, we implement the Intelligent Cutout and Passivation module within `src/generators/extraction.py`. When the Dynamics Engine halts due to encountering high uncertainty, this module identifies the exact "epicenter" atoms responsible. It then geometrically extracts a perfect spherical core and an outer buffer layer. Crucially, it applies an automated passivation algorithm to terminate dangling bonds (e.g., using fractional hydrogen atoms) on the buffer's surface, and utilizes the MACE foundation model to rapidly pre-relax the buffer atoms while strictly freezing the core. This complex sequence of geometric and quantum-mechanical operations guarantees that the final cluster passed to the DFT oracle is physically stable, electrically neutral, and yields pristine, high-fidelity ground truth forces specifically for the core region.

## 2. System Architecture

The architecture for this cycle introduces a dedicated extraction module acting as a highly sophisticated pre-processor positioned directly between the Dynamics Engine's halt event and the DFT Oracle's execution phase. By isolating this logic into its own module, we keep the orchestrator clean and ensure the physics logic is easily testable.

The `extraction.py` script encapsulates the complex `extract_intelligent_cluster` logic. It is designed to accept a massive `ase.Atoms` object (representing the entire halted MD snapshot, potentially containing thousands of atoms), a list of target atom indices indicating the uncertainty epicenter, and the strongly-typed `CutoutConfig` developed in CYCLE01. It delegates the initial spatial filtering to highly optimized neighbor-list utilities, optionally calls out to a mocked MACE calculator for the pre-relaxation step, and finally executes the chemical passivation logic before returning a pristine, drastically reduced-size `Atoms` object ready for DFT.

### File Structure Modification

We will introduce a new file to house this complex geometric and chemical logic, while potentially utilizing existing utilities if appropriate.

```text
.
├── src/
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── **extraction.py**       (NEW: Intelligent cluster cutout, pre-relaxation, and passivation logic)
│   ├── dynamics/
│   │   ├── security_utils.py       (Utilize for generic, secure ASE manipulation if needed)
```

The introduction of `extraction.py` centralizes all geometric transformations. This prevents the Dynamics Engine from having to understand cluster geometries, maintaining strict separation of concerns. The module will rely purely on the standard `ase` library for atomic manipulations, ensuring broad compatibility.

## 3. Design Architecture

The design of the `extraction.py` module is heavily focused on precise geometric manipulations of the `ase.Atoms` object and the careful management of physical metadata via ASE's internal array mechanisms. It must handle periodic boundary conditions correctly and manage atomic forces explicitly.

### Domain Concepts and Constraints

1. **Two-Zone Spherical Extraction:** The core conceptual design is dividing the extracted region into two distinct spatial zones based on the `CutoutConfig`. The 'Core' (defined by radius $R_{core}$) contains the exact atoms that triggered the uncertainty halt. The 'Buffer' (defined by the space between $R_{core}$ and $R_{buffer}$) provides a chemically realistic environment shielding the core from the vacuum. The system must accurately compute distances taking into account the minimum image convention if periodic boundaries are present.
2. **Force Weighting:** A critical physical invariant is that atoms in the core receive a `force_weight = 1.0` (meaning their DFT forces will be actively used by the Pacemaker algorithm for learning), while atoms in the buffer receive a `force_weight = 0.0` (their forces are polluted by surface effects and must be ignored). Passivating atoms also strictly receive `0.0`. This array must be seamlessly and immutably attached to the `Atoms.arrays['force_weights']` dictionary.
3. **MACE Pre-Relaxation:** The buffer and passivating atoms, being geometrically abrupt, are far from their local energy minima. Sending this raw structure to DFT guarantees SCF divergence. We use `ase.constraints.FixAtoms` to rigidly freeze the core, and an external LBFGS optimizer driven by the fast MACE surrogate model to gently relax the buffer into a physically sensible state.
4. **Auto-Passivation:** Dangling bonds are identified by calculating local coordination numbers and comparing them against expected chemical valences (based on electronegativity rules). For example, an under-coordinated oxygen atom at the buffer surface will automatically be bonded to a dummy atom (e.g., Hydrogen) along the vector of the missing bond to neutralize charge dipoles and stabilize the electronic structure.

## 4. Implementation Approach

The implementation leverages the `ase.neighborlist` module heavily for efficient spatial queries, ensuring the extraction process is fast even on massive supercells.

1. **Extraction Geometry:** Implement `_extract_spherical_zones(atoms, center_idx, core_r, buffer_r)`. Use `ase.neighborlist.neighbor_list('i', atoms, buffer_r)` centered on the target atom to quickly grab all atomic indices within the maximum radius. Sort these indices into `core_indices` and `buffer_indices` arrays strictly based on their distance from the epicenter.
2. **Weight Assignment:** Create the new sliced `Atoms` object containing only the extracted atoms. Immediately initialize a numpy array of zeros for the `force_weights`. Set the specific indices corresponding to the core atoms to `1.0`. Append this critical array to the `atoms.arrays` dictionary to ensure it travels with the object.
3. **Passivation Logic:** Implement `_passivate_surface(cluster, passivating_element="H")`. Compute the local coordination of the buffer atoms using a distance matrix. If an atom (e.g., O or Mg) has fewer neighbors than expected for the bulk material, calculate the vector pointing outwards from the cluster center and meticulously place a Hydrogen atom at a typical bond distance (e.g., 1.0 Å) along that vector. Append the new atom, ensuring its `force_weight` is strictly 0.0.
4. **Pre-Relaxation:** Implement `_pre_relax_buffer(cluster, mock_calculator)`. Apply `ase.constraints.FixAtoms(indices=core_indices)` to the cluster to protect the high-uncertainty geometry. Attach the provided MACE calculator. Run an `ase.optimize.LBFGS` optimization for a maximum of 50 steps or until forces converge below a generous threshold of 0.1 eV/Å.
5. **Main API:** Implement the public function `extract_intelligent_cluster` which sequentially calls the extraction, passivation, and pre-relaxation steps based exactly on the boolean flags provided in the `CutoutConfig`. It must handle exceptions gracefully, returning a safely fallback structure if pre-relaxation fails.

## 5. Test Strategy

Testing this geometric logic requires precise synthetic systems and robust mocking of the surrogate optimizer to avoid massive CI execution times.

### Unit Testing Approach
We will construct simple, mathematically verifiable test cases within `tests/generators/test_extraction.py`.
- **Zonal Extraction Test:** Create a large 10x10x10 simple cubic lattice. Call the extraction function targeting the central atom with a core radius of 1.1a and a buffer radius of 2.1a. Assert that exactly the correct number of nearest and next-nearest neighbors are selected. Assert that the `force_weights` array is correctly attached, with exactly 7 atoms (center + 6 nearest) having a weight of `1.0`, and all others strictly `0.0`.
- **Passivation Vector Test:** Create a simple diatomic molecule (e.g., NaCl). Run the passivation logic on the Na atom, treating it as the buffer surface. Assert that a dummy Cl (or H) atom is added along the exact correct geometric vector extending outward from the center of mass, and verify the bond distance is accurate.

### Integration Testing Approach
We must test the full pipeline including the pre-relaxation step without requiring a real, gigabyte-sized MACE PyTorch model.
- **Mocked Optimization Test:** We will construct a synthetic cluster and configure the extraction pipeline to enable pre-relaxation. We will inject a custom, highly simplified `ase.calculators.calculator.Calculator` mock. This mock will implement `get_forces` by returning simple Hookean spring forces pulling the buffer atoms slightly inwards toward the origin. We will run the extraction and assert that the coordinates of the core atoms are exactly identical to the input (verifying `FixAtoms` worked perfectly), while the coordinates of the buffer atoms have moved closer to the origin (verifying the `LBFGS` optimizer functioned correctly with the mocked forces).
- **Side-Effect Management:** The test environment will strictly avoid writing any intermediate trajectory or `.traj` files to disk during the mocked LBFGS optimization. We will override ASE's default logging behaviors to ensure standard output remains clean and CI logs do not blow up in size.
