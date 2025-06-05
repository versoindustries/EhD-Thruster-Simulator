# EHD Thruster Design Optimization

This Python script simulates and optimizes the design of an ElectroHydroDynamic (EHD) thruster, a propulsion system that uses electric fields to ionize and accelerate air. It leverages physics simulations, Bayesian optimization, and 3D model generation to determine optimal design parameters for a given payload weight. The script outputs design specifications and STL files for 3D printing the thruster components.

## Overview

The script integrates several components:
- **Physics Simulations**: Imported from `physix_single.py` to model thruster behavior.
- **Material and Battery Data**: Loaded from JSON files to inform simulations and weight calculations.
- **Bayesian Optimization**: Finds optimal design parameters minimizing system weight.
- **STL Generation**: Creates 3D models of thruster components using `trimesh`.

## Requirements

- Python 3.8+
- Libraries: `numpy`, `scipy`, `tensorflow`, `sklearn`, `trimesh`, `shapely`, `tqdm`, `json`, `os`, `sys`, `logging`, `hashlib`, `typing`
- External module: `physix_single.py` (not included, must provide simulation functions)

## Code Structure

### Main Components

#### 1. Simulation Functions
Imported from `physix_single.py`:
- `couple_physics_tf`: Simulates coupled physics (electric fields, ion flows, fluid dynamics).
- `calculate_thrust_tf`: Computes thrust from simulation outputs.
- `setup_grid_tf`: Initializes the computational grid.
- `get_material_properties`: Fetches material properties for simulations.
- `air_density_tf`: Calculates air density based on environmental conditions.

Used within `cached_simulate_thrust` to simulate thrust with caching.

#### 2. Data Loading
- **`load_json_file`**: Loads JSON files (`materials.json`, `samsung_21700_batteries.json`) with fallback to defaults if files are missing or invalid.
  - Expected inputs: Material properties (density, conductivity) and battery specs (capacity, voltage, weight).

#### 3. Optimization
- **`bayesian_optimization`**: Implements Bayesian optimization to minimize total system weight.
  - Parameters optimized: `r_e` (emitter radius), `r_c` (collector radius), `d` (gap distance), `l` (collector length), `V` (voltage).
  - Uses Latin Hypercube Sampling for initial points and Gaussian Process Regression with Expected Improvement for iteration.
- **`objective`**: Evaluates system weight, including thruster, generator, and battery weights, with penalties for aspect ratio deviations.

#### 4. STL Generation
Functions generate 3D models saved as STL files:
- `create_collector_tube`: Hollow cylinder for the collector.
- `create_emitter`: Emitter in cylindrical, pointed, or hexagonal shapes.
- `create_emitter_mount`: Mount to secure the emitter.
- `create_collector_cap`: Cap for the collector tube.
- `create_rigid_ring`: Structural support ring.
- `create_wire_mesh`: Mesh for airflow or electrical purposes.
- `create_copper_wire_ring`: Copper wire ring component.
- `create_assembly`: Combines components into a full model.

#### 5. Helper Functions
- `calculate_thruster_weight`: Computes thruster weight from dimensions and material densities.
- `cached_simulate_thrust`: Simulates thrust with caching to avoid redundant computations.
- `simulate_single_candidate`: Runs simulation for a single parameter set.
- `calculate_battery_temp_profile`: Estimates battery temperature during operation.
- `find_min_battery_weight`: Finds minimal battery configuration for power/energy needs.
- `create_design_details`: Compiles design data into a dictionary.
- `save_design_files`: Saves design details and STL files.

### Key Constants
- `PARAM_BOUNDS`: Defines parameter ranges (e.g., `r_e`: 0.5-7.5 mm).
- `SHAPES`: Supported emitter shapes: `cylindrical`, `pointed`, `hexagonal`.
- `SAFETY_MARGIN`: 1.3x required thrust.
- `FLIGHT_TIME`: 1800 seconds (30 minutes).
- `EFFICIENCY`: 10% power efficiency.
- `GENERATOR_WEIGHT`: 0.057 kg.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install numpy scipy tensorflow scikit-learn trimesh shapely tqdm
   ```

2. **Provide `physix_single.py`**:
   - Ensure this module is in the same directory with required simulation functions.

3. **Prepare Input Files**:
   - `materials.json`: Material properties.
   - `samsung_21700_batteries.json`: Battery specifications.
   - Defaults are used if files are absent.

4. **Run the Script**:
   ```bash
   python script_name.py
   ```
   - Prompt: "Enter payload weight (kg):"
   - Example: Enter `0.5` for a 0.5 kg payload.

5. **Output**:
   - Directory: `optimization_results/payload_<weight>kg/`
   - Files per design: `design_details.json`, `design_summary.txt`, STL files (e.g., `collector_tube.stl`), and `assembly.stl`.
   - Summary file: `optimal_design_<weight>kg.txt`.

## Example Run
```bash
Enter payload weight (kg): 0.5
```
- Optimizes for 0.5 kg payload.
- Outputs feasible designs and the optimal design with minimal weight.

## Customization
- Modify `PARAM_BOUNDS` for different parameter ranges.
- Adjust `SHAPES` or `MATERIALS` in `optimize_design` for new configurations.
- Update constants (e.g., `SAFETY_MARGIN`, `EFFICIENCY`) as needed.

## License
Apache License, Version 2.0. See `LICENSE` file for details.

## Contributing
- Fork the repository on GitHub.
- Submit pull requests with improvements or bug fixes.
- Report issues via GitHub Issues.

## Notes
- GPU support is optional; the script configures TensorFlow for GPU if available.
- Logging is enabled to `ehd_simulator.log` for debugging.