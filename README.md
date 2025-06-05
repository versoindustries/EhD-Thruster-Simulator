# EhD-Thruster-Simulator

EhD-Thruster-Simulator is an open-source simulation tool for modeling Electrohydrodynamic (EHD) thrusters. These devices leverage electric fields, ion flows, and fluid dynamics to generate thrust, with potential applications in propulsion systems such as drones or aerial vehicles. The simulator integrates physics-based modeling, optimization algorithms, and GPU-accelerated computation to predict thrust performance and optimize thruster designs. It also generates 3D STL models for visualization and manufacturing.

## Features

- **Physics Simulation**: Models electrostatic and fluid dynamic interactions within EHD thrusters, incorporating real-world material properties and environmental conditions.
- **Optimization**: Employs Bayesian optimization to identify efficient thruster designs based on user-defined payload weights.
- **Material and Battery Data**: Integrates detailed material properties (e.g., copper, aluminum) and battery specifications (e.g., Samsung 21700 series) for realistic simulations.
- **STL File Generation**: Produces 3D models of thruster components (e.g., emitter, collector, mounts) for visualization or 3D printing.
- **GPU Acceleration**: Utilizes TensorFlow with DirectML for hardware acceleration on a variety of GPUs, including AMD, Intel, and NVIDIA. For NVIDIA users, an option to use the standard TensorFlow GPU version is available for potentially better performance.

## Installation Instructions

To set up and run the EhD-Thruster-Simulator, ensure you have Python 3.10.11 or later installed. Follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/versoindustries/EhD-Thruster-Simulator.git
   cd EhD-Thruster-Simulator
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file is configured for the DirectML pipeline and includes:
   - tensorflow-cpu==2.10.1
   - tensorflow-directml-plugin
   - numpy==1.26.4
   - scipy
   - scikit-learn
   - numba
   - numpy-stl
   - tqdm
   - tensorboard
   - matplotlib

## TensorFlow Configuration

This project is built for the TensorFlow DirectML pipeline by default, enabling hardware acceleration on a wide range of GPUs, including AMD, Intel, and NVIDIA. The `requirements.txt` file includes `tensorflow-cpu==2.10.1` and `tensorflow-directml-plugin` to support this configuration.

### What is TensorFlow DirectML?

TensorFlow DirectML is a plugin that allows TensorFlow to utilize DirectML, a hardware-accelerated machine learning API for Windows. It broadens TensorFlow's compatibility, enabling GPU acceleration on systems without NVIDIA hardware.

### For NVIDIA GPU Users

If you have an NVIDIA GPU and prefer to use the standard TensorFlow GPU version for potentially better performance, modify the `requirements.txt` file as follows:

- Remove the line containing `tensorflow-directml-plugin`.
- Replace `tensorflow-cpu==2.10.1` with `tensorflow==2.10.1`.

After making these changes, reinstall the dependencies using:
```bash
pip install -r requirements.txt
```

Ensure you have the necessary NVIDIA drivers and CUDA Toolkit installed. Refer to the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu) for detailed setup instructions.

**Note on GPU Usage**: This project uses TensorFlow with DirectML by default, supporting hardware acceleration on various GPUs, including AMD, Intel, and NVIDIA. No additional setup is required beyond installing the dependencies. For NVIDIA users who prefer the standard TensorFlow GPU version, follow the instructions in the "TensorFlow Configuration" section above.

## Usage Guide

The primary script is `ehd_simulator.py`. Run it from the command line to optimize thruster designs for a specified payload weight.

### Basic Usage

```bash
python ehd_simulator.py
```

When prompted, enter the payload weight in kilograms (e.g., `1.0`). The simulator will perform optimization and save results in the `optimization_results` directory.

### Command-Line Options

View all options with:
```bash
python ehd_simulator.py --help
```

### Configuration

Customize the simulation by editing:
- `materials.json`: Material properties (e.g., density, conductivity).
- `samsung_21700_batteries.json`: Battery specifications (e.g., capacity, voltage).

## Examples

1. **Optimize for a 0.5 kg Payload**:
   ```bash
   python ehd_simulator.py
   # Enter "0.5" when prompted
   ```

   Results, including STL files, are saved in `optimization_results/payload_0.500kg`.

2. **View Generated STL Files**:
   Find STL files (e.g., `collector_tube.stl`, `emitter.stl`) in the design subdirectories. Use software like Blender or MeshLab to visualize them.

3. **Modify Material Properties**:
   Edit `materials.json` to adjust properties, then rerun the simulation:
   ```json
   "copper": {
       "density_kg_m3": 8960,
       "elec_conductivity_S_m": 5.96e7,
       ...
   }
   ```

## Performance Considerations

- **Memory Usage**: High-resolution grids can be memory-intensive. Ensure sufficient RAM and GPU VRAM (e.g., 8GB+ recommended).
- **Computation Time**: Optimization duration varies with grid size and iterations. Reduce grid resolution for faster initial tests.
- **Caching**: Intermediate results are cached in the `simulation_cache` directory. Ensure adequate disk space.

## Documentation

Detailed information on simulation models, optimization algorithms, and code structure is available in the documentation.

## Minimum Requirements

specs code was built on, could be ran cpu on less with much longer runtimes. 

- **Ryzen 7 2700x**
- **64 GB RAM**
- **AMD rx6600**

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and report issues via [GitHub Issues](https://github.com/yourusername/EhD-Thruster-Simulator/issues).

## Notes

There are currently several variations of our end cap we are prototyping with right now, will update the code with the most efficient model.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

# EhD-Thruster-Simulator Sponsorship Tiers

## Supporter Tier - $10/month
- **Benefits**:
  - Recognition on the project's website and documentation.
  - Access to project updates and newsletters.
- **Description**:
  - Ideal for individuals or small organizations who want to support the project without needing additional perks.

## Contributor Tier - $100/month
- **Benefits**:
  - All Supporter Tier benefits.
  - Early access to new features and releases (e.g., beta versions or private repository access).
  - Standard support for issues and feature requests.
- **Description**:
  - Suitable for users or companies who actively use the simulator and want to stay ahead of updates.

## Patron Tier - $500/month
- **Benefits**:
  - All Contributor Tier benefits.
  - Priority support for issues and bug fixes.
  - Feature request prioritization (requests considered in the next development cycle).
  - Logo displayed on the project's homepage.
- **Description**:
  - Designed for companies or organizations that rely heavily on the simulator and want their needs addressed promptly.

## Director Tier - $5,000/year
- **Benefits**:
  - All Patron Tier benefits.
  - Guaranteed inclusion of one feature request per year (subject to feasibility and project maintainers' approval).
  - Participation in quarterly strategy meetings with project maintainers to discuss roadmap and strategic decisions.
  - Invitation to an Advisory Board to provide input on major project directions.
- **Description**:
  - For organizations that want a significant say in the project's future while ensuring their use cases are prioritized.

## Founder Tier - $50,000/year
- **Benefits**:
  - All Director Tier benefits.
  - Opportunity to sponsor a major project milestone or release (e.g., funding a specific version or feature set).
  - Custom development work on a specific module or feature (within the scope of the project and open-source principles).
  - Co-branding opportunities (e.g., company name/logo prominently featured in project materials).
- **Description**:
  - For major stakeholders who want to deeply integrate the project into their operations and ensure it aligns with their long-term goals.


## Citation

If you use this simulator in your research, please cite it as:
> Verso Industries, Michael B. Zimmerman. (2025). EhD-Thruster-Simulator. GitHub repository, https://github.com/versoindustries/EhD-Thruster-Simulator

## Acknowledgments

- Thanks to the developers of TensorFlow, NumPy, SciPy, scikit-learn, trimesh, and shapely for their foundational libraries.
- Special appreciation to Verso Industries and Michael B. Zimmerman for initial development efforts.