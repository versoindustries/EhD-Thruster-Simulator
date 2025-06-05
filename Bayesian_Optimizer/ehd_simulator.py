# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import json
import numpy as np
from scipy.stats import qmc
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
import trimesh
from shapely.geometry import Polygon, Point  # Updated import statement
import traceback
from tqdm import tqdm
import logging
import hashlib
from typing import Dict, Union, Optional, List

# Enable eager execution for compatibility
tf.config.run_functions_eagerly(True)

# Import simulation functions
try:
    from physix_single import (
        couple_physics_tf, calculate_thrust_tf, setup_grid_tf,
        get_material_properties, air_density_tf, ELEM_CHARGE, MU_AIR
    )
except ImportError as e:
    print(f"Error: Could not import physix_single.py - {e}")
    sys.exit(1)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected and configured: {gpus}")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU detected. Running on CPU.")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='ehd_simulator.log')
logger = logging.getLogger(__name__)

# Constants
G = 9.8
SAFETY_MARGIN = 1.3
FLIGHT_TIME = 1800
EFFICIENCY = 0.1
GENERATOR_WEIGHT = 0.057
T_COLLECTOR = 0.0001  # Collector shell thickness in meters (used in simulation)
MAX_BATTERIES = 48
PI = np.pi
SQRT3 = np.sqrt(3)
AWG_10_DIAMETER_MM = 2.588  # Fixed diameter of AWG 10 wire in mm
PENALTY_FACTOR = 0.1  # kg per meter, adjust as needed

# Parameter bounds
PARAM_BOUNDS = {
    'r_e': (0.0005, 0.0075), # Emitter radius: 0.5 mm to 7.5 mm (diameter 1 mm to 15 mm)
    'r_c': (0.008, 0.022),   # Collector radius: 8 mm to 22 mm (max outer diameter ~51 mm)
    'd': (0.005, 0.05),      # Gap distance: 5 mm to 50 mm
    'l': (0.01, 0.19),       # Collector length: 10 mm to 190 mm
    'V': (10e3, 500e3)       # Voltage: 10 kV to 500 kV
}

SHAPES = ['cylindrical', 'pointed', 'hexagonal']
DEFAULT_MATERIALS = {'steel': {'density': 7800}, 'aluminum': {'density': 2700}, 'tungsten': {'density': 19250}, 'copper': {'density': 8960}}
DEFAULT_BATTERIES = {'Samsung_50E': {'capacity_mAh': 5000, 'nominal_voltage_V': 3.7, 'weight_g': 70, 'max_continuous_discharge_A': 10, 'internal_resistance_ohm': 0.020, 'specific_heat_J_kgK': 900}}
PRIOR_DATA = {'params': {'r_e': 1e-3, 'r_c': 20e-3, 'd': 30e-3, 'l': 0.5, 'V': 100e3}, 'shape': 'cylindrical', 'mat_emitter': 'tungsten', 'mat_collector': 'copper', 'thrust': 0.15}

# STL Constants (in mm)
COLLECTOR_WALL_THICKNESS_MM = 3.5  # Fixed for 3D printing feasibility

# Helper Functions
def load_json_file(filename: str, default_data: Dict, desc: str) -> Dict:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load {filename} ({e}). Using defaults.")
        return default_data

def calculate_thruster_weight(r_e: float, r_c: float, l: float, shape_emitter: str, rho_emitter: float, rho_collector: float) -> Optional[float]:
    if r_c <= r_e or T_COLLECTOR >= r_c:
        return None
    if shape_emitter == 'cylindrical' or shape_emitter == 'pointed':
        emitter_volume = PI * r_e**2 * 0.01  # Fixed emitter length for simulation (0.01 m)
    elif shape_emitter == 'hexagonal':
        # Area of a regular hexagon: (3 * sqrt(3) / 2) * r_e^2
        emitter_volume = (3 * SQRT3 / 2) * r_e**2 * 0.01  # Fixed emitter length (0.01 m)
    else:
        raise ValueError(f"Unknown emitter shape: {shape_emitter}")
    collector_volume = PI * l * (2 * r_c * T_COLLECTOR - T_COLLECTOR**2)
    thruster_weight = emitter_volume * rho_emitter + collector_volume * rho_collector
    return thruster_weight if np.isfinite(thruster_weight) and thruster_weight > 0 else None

def get_cache_filename(params_tuple, material_props_tuple, grid_settings_tuple, env_conditions_tuple) -> str:
    param_str = json.dumps({
        'params': params_tuple,
        'material_props': material_props_tuple,
        'grid_settings': grid_settings_tuple,
        'env_conditions': env_conditions_tuple
    }, sort_keys=True)
    hash_id = hashlib.md5(param_str.encode()).hexdigest()
    return f"cache_{hash_id}.json"

def cached_simulate_thrust(params_tuple, material_props_tuple, grid_settings_tuple, env_conditions_tuple):
    cache_dir = "simulation_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = get_cache_filename(params_tuple, material_props_tuple, grid_settings_tuple, env_conditions_tuple)
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return tf.constant(data['thrust'], dtype=tf.float32), tf.constant(data['status'], dtype=tf.int32)

    r_e, r_c, d, l, V, shape = params_tuple
    townsend_A, townsend_B, E_onset = material_props_tuple
    nx_base, ny_base, nz_base = grid_settings_tuple
    pressure_atm, temperature_k, altitude_m = env_conditions_tuple

    if not all(PARAM_BOUNDS[p][0] <= v <= PARAM_BOUNDS[p][1] for p, v in zip(['r_e', 'r_c', 'd', 'l', 'V'], [r_e, r_c, d, l, V])) or r_e >= r_c or shape not in SHAPES:
        return tf.constant(np.nan, dtype=tf.float32), tf.constant(1, dtype=tf.int32)

    r_e_tf, r_c_tf, d_tf, l_tf, V_tf = [tf.constant(x, dtype=tf.float32) for x in [r_e, r_c, d, l, V]]
    nx_tf, ny_tf, nz_tf = [tf.constant(x, dtype=tf.int32) for x in [nx_base, ny_base, nz_base]]
    env_tf = [tf.constant(x, dtype=tf.float32) for x in [pressure_atm, temperature_k, altitude_m]]
    mat_tf = [tf.constant(x, dtype=tf.float32) for x in [townsend_A, townsend_B, E_onset]]

    current_rho_air_tf, density_status = air_density_tf(env_tf[2])
    if density_status.numpy() != 0:
        return tf.constant(np.nan, dtype=tf.float32), density_status

    try:
        x_1d_tf, y_1d_tf, z_1d_tf, nx, ny, nz = setup_grid_tf(r_e_tf, r_c_tf, d_tf, l_tf, nx_tf, ny_tf, nz_tf)
    except tf.errors.InvalidArgumentError:
        return tf.constant(np.nan, dtype=tf.float32), tf.constant(5, dtype=tf.int32)

    params_for_coupling = (r_e_tf, r_c_tf, d_tf, l_tf, V_tf, shape, None, None)
    grid_res_for_coupling = (nx_tf, ny_tf, nz_tf)
    result = couple_physics_tf(params_for_coupling, grid_res_for_coupling, 10, env_tf[0], env_tf[1], mat_tf[0], mat_tf[1], mat_tf[2], current_rho_air_tf, tf.constant(MU_AIR, dtype=tf.float32))
    status = result[11].numpy()
    if status != 0:
        return tf.constant(np.nan, dtype=tf.float32), result[11]

    rho_charge_final = ELEM_CHARGE * (result[4] - result[3])
    thrust_result = calculate_thrust_tf(rho_charge_final, result[6], result[7], result[8], result[12], result[13], result[14], env_tf[2])
    thrust, thrust_status = thrust_result[3], thrust_result[4]
    status = max(thrust_status.numpy(), status)

    with open(cache_path, 'w') as f:
        json.dump({'thrust': float(thrust.numpy()), 'status': int(status)}, f)
    return thrust, thrust_status

def simulate_single_candidate(params, shape_emitter, mat_emitter, mat_collector, env_conditions, grid_settings, materials_data):
    r_e, r_c, d, l, V = params
    params_tuple = (float(r_e), float(r_c), float(d), float(l), float(V), shape_emitter)
    material_props = get_material_properties(mat_emitter, materials_data)
    material_props_tuple = (
        float(material_props.get('townsend_A', 15.0)),
        float(material_props.get('townsend_B', 365.0)),
        float(material_props.get('corona_onset_V_m', 3e6))
    )
    thrust_tensor, status_tensor = cached_simulate_thrust(params_tuple, material_props_tuple, grid_settings, env_conditions)
    thrust = float(thrust_tensor.numpy()) if np.isfinite(thrust_tensor) else 0.0
    status = int(status_tensor.numpy())
    return thrust, status

def calculate_battery_temp_profile(power_draw_W: float, battery_config: Dict) -> Optional[List[float]]:
    spec = battery_config['model_spec']
    n_series, n_parallel = battery_config['n_series'], battery_config['n_parallel']
    V_bat_nominal = spec['nominal_voltage_V']
    weight_g = spec['weight_g']
    R_internal_cell_ohm = spec.get('internal_resistance_ohm', 0.020)
    T_ambient_K, Cp_cell_J_per_kgK = 300, spec.get('specific_heat_J_kgK', 900)
    h_conv_W_m2K = 10
    cell_diameter_m, cell_length_m = 0.021, 0.070
    A_cell_m2 = 2 * PI * (cell_diameter_m/2)**2 + PI * cell_diameter_m * cell_length_m
    if V_bat_nominal <= 0 or n_parallel <= 0 or n_series <= 0 or weight_g <= 0 or Cp_cell_J_per_kgK <= 0:
        return None
    I_total_A = power_draw_W / V_bat_nominal
    I_per_cell_A = I_total_A / n_parallel
    P_heat_per_cell_W = I_per_cell_A**2 * R_internal_cell_ohm
    mass_cell_kg = weight_g / 1000.0
    C_thermal_cell_J_K = Cp_cell_J_per_kgK * mass_cell_kg
    if C_thermal_cell_J_K <= 0:
        return None
    dt_sim_s = 60
    n_steps = int(3600 / dt_sim_s) + 1
    T_bat_profile_K = np.zeros(n_steps)
    T_bat_profile_K[0] = T_ambient_K
    for t in range(n_steps - 1):
        T_current_K = T_bat_profile_K[t]
        P_cool_per_cell_W = h_conv_W_m2K * A_cell_m2 * (T_current_K - T_ambient_K)
        P_net_cell_W = P_heat_per_cell_W - P_cool_per_cell_W
        delta_T_K = (P_net_cell_W / C_thermal_cell_J_K) * dt_sim_s
        T_bat_profile_K[t + 1] = T_current_K + delta_T_K
    return T_bat_profile_K.tolist()

def find_min_battery_weight(power_W: float, energy_Wh: float, battery_specs: Dict) -> tuple[float, Optional[Dict]]:
    best_weight = np.inf
    best_config = None
    for model, spec in battery_specs.items():
        capacity_mAh = spec.get('capacity_mAh')
        voltage_V = spec.get('nominal_voltage_V')
        weight_g = spec.get('weight_g')
        max_discharge_A = spec.get('max_continuous_discharge_A')
        if None in [capacity_mAh, voltage_V, weight_g, max_discharge_A] or any(v <= 0 for v in [capacity_mAh, voltage_V, weight_g, max_discharge_A]):
            continue
        capacity_Wh = (capacity_mAh / 1000) * voltage_V
        max_power_W = max_discharge_A * voltage_V
        for n_series in range(1, 5):
            V_pack_nominal = n_series * voltage_V
            min_p_for_discharge = int(np.ceil((power_W / V_pack_nominal) / max_discharge_A))
            min_p_for_energy = int(np.ceil(energy_Wh / (n_series * capacity_Wh)))
            n_parallel = max(1, min_p_for_discharge, min_p_for_energy)
            total_batteries = n_series * n_parallel
            if total_batteries > MAX_BATTERIES:
                continue
            total_weight = (total_batteries * weight_g) / 1000
            if total_weight < best_weight:
                best_weight = total_weight
                best_config = {
                    'model': model,
                    'model_spec': spec,
                    'n_series': n_series,
                    'n_parallel': n_parallel,
                    'n_total': total_batteries,
                    'voltage_V': V_pack_nominal,
                    'weight_kg': total_weight
                }
    return best_weight, best_config if best_weight < np.inf else None

def objective(params, shape, mat_emitter, mat_collector, payload_weight, materials_data, battery_specs):
    env_conditions = (1.0, 300.0, 0.0)
    grid_settings = (64, 64, 128)
    if any(x <= 0 for x in params) or params[0] >= params[1]:
        return 1e6, None, 0.0, 1
    thrust, status = simulate_single_candidate(params, shape, mat_emitter, mat_collector, env_conditions, grid_settings, materials_data)
    if status != 0 or thrust < 1e-9:
        return 1e6, None, thrust, status
    rho_emitter = materials_data['materials'].get(mat_emitter, DEFAULT_MATERIALS.get(mat_emitter, {})).get('density_kg_m3', 8000)
    rho_collector = materials_data['materials'].get(mat_collector, DEFAULT_MATERIALS.get(mat_collector, {})).get('density_kg_m3', 8000)
    thruster_weight = calculate_thruster_weight(params[0], params[1], params[3], shape, rho_emitter, rho_collector)
    if thruster_weight is None:
        return 1e6, None, thrust, status
    power_W = thrust / EFFICIENCY
    energy_Wh = (power_W * FLIGHT_TIME) / 3600
    battery_weight, battery_config = find_min_battery_weight(power_W, energy_Wh, battery_specs)
    if battery_config is None:
        return 1e6, None, thrust, status
    total_system_weight = thruster_weight + GENERATOR_WEIGHT + battery_weight
    required_thrust = (payload_weight + total_system_weight) * G * SAFETY_MARGIN
    if thrust >= required_thrust:
        # Calculate aspect ratio penalty
        aspect_ratio = params[3] / (2 * params[1])  # l / (2 * r_c)
        target_min_ar = 0.5
        target_max_ar = 2.0
        PENALTY_FACTOR = 1.0  # kg per unit deviation
        if aspect_ratio < target_min_ar:
            ar_penalty = PENALTY_FACTOR * (target_min_ar - aspect_ratio)
        elif aspect_ratio > target_max_ar:
            ar_penalty = PENALTY_FACTOR * (aspect_ratio - target_max_ar)
        else:
            ar_penalty = 0.0
        battery_config['system_weight_kg'] = total_system_weight
        battery_config['required_thrust_N'] = required_thrust
        return total_system_weight + ar_penalty, battery_config, thrust, status
    return 1e6, None, thrust, status

def create_design_details(params, shape, mat_emitter, mat_collector, thrust, thruster_weight, battery_config, payload_weight):
    r_e, r_c, d, l, V = params
    battery_weight = battery_config.get('weight_kg', 0) if battery_config else 0
    total_weight = thruster_weight + GENERATOR_WEIGHT + battery_weight
    thrust_to_weight = thrust / (total_weight * G) if total_weight > 0 else 0
    power_W = thrust / EFFICIENCY
    temp_profile = calculate_battery_temp_profile(power_W, battery_config) if battery_config else None
    return {
        'r_e_m': r_e, 'r_c_m': r_c, 'd_m': d, 'l_m': l, 'V_V': V,
        'shape_emitter': shape, 'mat_emitter': mat_emitter, 'mat_collector': mat_collector,
        'T_collector_m': T_COLLECTOR,
        'simulated_thrust_N': thrust,
        'required_thrust_N': battery_config.get('required_thrust_N', 0) if battery_config else 0,
        'power_draw_W': power_W,
        'flight_time_s': FLIGHT_TIME,
        'thrust_to_weight_ratio': thrust_to_weight,
        'payload_weight_kg': payload_weight,
        'thruster_weight_kg': thruster_weight,
        'battery_weight_kg': battery_weight,
        'generator_weight_kg': GENERATOR_WEIGHT,
        'system_weight_kg': total_weight,
        'battery_model': battery_config.get('model', 'N/A') if battery_config else 'N/A',
        'n_batteries': battery_config.get('n_total', 0) if battery_config else 0,
        'n_series': battery_config.get('n_series', 0) if battery_config else 0,
        'n_parallel': battery_config.get('n_parallel', 0) if battery_config else 0,
        'battery_voltage_nominal_V': battery_config.get('voltage_V', 0) if battery_config else 0,
        'battery_temp_profile_K': temp_profile,
        'battery_max_temp_K': max(temp_profile) if temp_profile else None
    }

# STL Generation Functions from stl_test.py
def create_hollow_cylinder_poly(outer_diameter: float, inner_diameter: float, segments: int = 64) -> 'Polygon':
    outer_radius = outer_diameter / 2.0
    inner_radius = inner_diameter / 2.0
    if inner_radius >= outer_radius:
        raise ValueError("Inner diameter must be less than outer diameter.")
    if inner_radius <= 0 or outer_radius <= 0:
        raise ValueError("Diameters must be positive.")
    outer_poly = Point(0, 0).buffer(outer_radius, resolution=segments)
    inner_poly = Point(0, 0).buffer(inner_radius, resolution=segments)
    hollow_poly = outer_poly.difference(inner_poly)
    if not hollow_poly.is_valid or not isinstance(hollow_poly, Polygon):
        outer_poly_buf = outer_poly.buffer(0)
        inner_poly_buf = inner_poly.buffer(0)
        if outer_poly_buf.is_valid and inner_poly_buf.is_valid:
            hollow_poly = outer_poly_buf.difference(inner_poly_buf)
            if not hollow_poly.is_valid or not isinstance(hollow_poly, Polygon):
                raise ValueError("Generated hollow polygon is invalid even after buffering.")
        else:
            raise ValueError("Generated hollow polygon is invalid, and buffering failed.")
    return hollow_poly

def create_collector_tube(inner_diameter_mm: float, wall_thickness_mm: float, length_mm: float) -> trimesh.Trimesh:
    outer_diameter_mm = inner_diameter_mm + 2 * wall_thickness_mm
    tube_cross_section = create_hollow_cylinder_poly(outer_diameter_mm, inner_diameter_mm)
    tube = trimesh.creation.extrude_polygon(tube_cross_section, height=length_mm)
    tube.process(validate=True)
    return tube

def create_emitter(r_e_mm: float, height_mm: float, shape_emitter: str = "cylindrical", sections: int = 64) -> trimesh.Trimesh:
    if shape_emitter.lower() == "cylindrical":
        emitter = trimesh.creation.cylinder(radius=r_e_mm, height=height_mm, sections=sections)
    elif shape_emitter.lower() == "pointed":
        emitter = trimesh.creation.cone(radius=r_e_mm, height=height_mm, sections=sections)
    elif shape_emitter.lower() == "hexagonal":
        # Create a hexagonal cross-section
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices for a hexagon
        vertices = np.array([[r_e_mm * np.cos(a), r_e_mm * np.sin(a)] for a in angles])
        hex_polygon = Polygon(vertices)
        emitter = trimesh.creation.extrude_polygon(hex_polygon, height=height_mm)
    else:
        raise ValueError(f"Unsupported emitter shape: {shape_emitter}")
    emitter.process(validate=True)
    return emitter

def create_emitter_mount(
    r_e_mm: float,
    collector_inner_diameter_mm: float,
    emitter_height_mm: float,
    exposure_percentage: float = 0.08,
    sections: int = 64,
    engine: str = 'blender'
) -> trimesh.Trimesh:
    # Define a minimum mount height to ensure sliding and securing capability
    MIN_MOUNT_HEIGHT_MM = 5.0  # Minimum height in mm for structural integrity
    exposed_height_mm = exposure_percentage * emitter_height_mm
    mount_height_mm = emitter_height_mm - exposed_height_mm
    # Ensure mount height is at least the minimum required
    if mount_height_mm < MIN_MOUNT_HEIGHT_MM:
        mount_height_mm = MIN_MOUNT_HEIGHT_MM
        exposed_height_mm = emitter_height_mm - mount_height_mm
    
    if mount_height_mm <= 0:
        raise ValueError("Mount height must be positive after accounting for exposure.")
    
    emitter_base_diameter = 2 * r_e_mm
    emitter_mount_hole_diameter = emitter_base_diameter + 0.2
    base_outer_diameter = collector_inner_diameter_mm - 0.2
    top_outer_diameter = min(emitter_mount_hole_diameter + 2.0, base_outer_diameter)
    base_r = base_outer_diameter / 2.0
    top_r = top_outer_diameter / 2.0
    hole_r = emitter_mount_hole_diameter / 2.0
    
    if not (base_r > hole_r and top_r > hole_r):
        raise ValueError("Mount radii are invalid (hole larger than outer?)")
    
    profile_points = np.array([
        [hole_r, 0],
        [base_r, 0],
        [top_r, mount_height_mm],
        [hole_r, mount_height_mm],
        [hole_r, 0]
    ])
    mount = trimesh.creation.revolve(profile_points, sections=sections)
    
    airflow_hole_diameter_mm = max(2.0, 0.05 * collector_inner_diameter_mm)
    airflow_hole_radius_mm = airflow_hole_diameter_mm / 2.0
    radial_position = (base_r + top_r) / 2.0
    N_airflow_holes = max(3, int(2 * np.pi * radial_position / (airflow_hole_diameter_mm * 1.5)))
    airflow_hole_height = mount_height_mm * 2.0 + 20.0
    hole_meshes = []
    
    for i in range(N_airflow_holes):
        angle = i * 2 * np.pi / N_airflow_holes
        tx = radial_position * np.cos(angle)
        ty = radial_position * np.sin(angle)
        tz = -10.0
        hole = trimesh.creation.cylinder(radius=airflow_hole_radius_mm, height=airflow_hole_height, sections=32)
        translation = trimesh.transformations.translation_matrix([tx, ty, tz])
        hole.apply_transform(translation)
        hole_meshes.append(hole)
    
    if hole_meshes:
        combined_holes = trimesh.util.concatenate(hole_meshes)
        if not mount.is_watertight:
            mount.fill_holes()
            mount.fix_normals()
        if not combined_holes.is_watertight:
            combined_holes.fill_holes()
            combined_holes.fix_normals()
        try:
            mount = mount.difference(combined_holes, engine=engine)
        except:
            mount = mount.difference(combined_holes, engine=None)
    
    mount.process(validate=True)
    return mount

def create_collector_cap(collector_inner_diameter_mm: float, collector_outer_diameter_mm: float, plug_height_mm: float, flange_height_mm: float, sections: int = 512, engine: str = 'blender') -> trimesh.Trimesh:
    cap_plug_diameter = collector_inner_diameter_mm - 0.2
    cap_flange_diameter = collector_outer_diameter_mm + 1.0
    cap_hole_diameter = collector_inner_diameter_mm - 2.0
    hole_r = cap_hole_diameter / 2.0
    plug_r = cap_plug_diameter / 2.0
    flange_r = cap_flange_diameter / 2.0
    total_h = plug_height_mm + flange_height_mm
    if not (flange_r > plug_r > hole_r > 0):
        raise ValueError("Cap radii are invalid")
    profile_points = np.array([
        [hole_r, 0],
        [plug_r, 0],
        [plug_r, plug_height_mm],
        [flange_r, plug_height_mm],
        [flange_r, total_h],
        [hole_r, total_h],
        [hole_r, 0]
    ])
    cap = trimesh.creation.revolve(profile_points, sections=sections)
    flange_circumference = 2 * np.pi * flange_r
    notch_width = 1.0
    min_spacing = 1.0
    max_notch_count = int(flange_circumference / (notch_width + min_spacing))
    notch_count = max(4, min(36, max_notch_count))
    notch_depth = 2.0
    notch_meshes = []
    angle_step_rad = 2 * np.pi / notch_count
    notch_box_extents = [notch_depth + 0.1, notch_width + 0.1, flange_height_mm + 0.1]
    for i in range(notch_count):
        angle_rad = i * angle_step_rad
        notch_box = trimesh.creation.box(extents=notch_box_extents)
        translation_radius = flange_r - notch_depth / 2.0
        tx = translation_radius * np.cos(angle_rad)
        ty = translation_radius * np.sin(angle_rad)
        tz = plug_height_mm + flange_height_mm / 2.0
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        translation_matrix = trimesh.transformations.translation_matrix([tx, ty, tz])
        transform_matrix = translation_matrix @ rotation_matrix
        notch_box.apply_transform(transform_matrix)
        notch_meshes.append(notch_box)
    if notch_meshes:
        combined_notches = trimesh.util.concatenate(notch_meshes)
        if not cap.is_watertight: cap.fill_holes(); cap.fix_normals()
        if not combined_notches.is_watertight: combined_notches.fill_holes(); combined_notches.fix_normals()
        try:
            cap = cap.difference(combined_notches, engine=engine)
        except:
            cap = cap.difference(combined_notches, engine=None)
    cap.process(validate=True)
    return cap

def create_rigid_ring(inner_diameter_mm=85.2, wall_thickness_mm=6.0, height_mm=5.0, groove_diameter_mm=2.588, hole_radius_mm=2.0, engine='manifold'):
    inner_radius = inner_diameter_mm / 2.0
    outer_radius = (inner_diameter_mm + 2 * wall_thickness_mm) / 2.0
    y_bottom_groove = 1.5
    y_top_groove = 3.5
    z_mid = (y_bottom_groove + y_top_groove) / 2.0
    groove_radius = (y_top_groove - y_bottom_groove) / 2.0
    r_center = inner_radius
    arc_segments = 20
    theta = np.linspace(np.pi / 2, -np.pi / 2, num=arc_segments)
    arc_points = [
        [r_center + groove_radius * np.cos(t), z_mid + groove_radius * np.sin(t)]
        for t in theta
    ]
    profile_points = [
        [outer_radius, 0],
        [outer_radius, height_mm],
        [inner_radius, height_mm],
    ] + arc_points + [
        [inner_radius, 0],
        [outer_radius, 0]
    ]
    profile_points = np.array(profile_points)
    ring = trimesh.creation.revolve(profile_points, sections=256)
    if not ring.is_watertight:
        ring.fill_holes()
        ring.fix_normals()
    hole_height = wall_thickness_mm + 1.0
    hole = trimesh.creation.cylinder(radius=hole_radius_mm, height=hole_height, sections=256)
    if not hole.is_watertight:
        hole.fill_holes()
        ring.fix_normals()
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    hole.apply_transform(rotation)
    x_pos = (inner_radius + outer_radius) / 2
    translation = trimesh.transformations.translation_matrix([x_pos, 0, height_mm / 2])
    hole.apply_transform(translation)
    final_ring = ring.difference(hole, engine=engine)
    if not final_ring.is_watertight:
        final_ring.fill_holes()
        final_ring.fix_normals()
    final_ring.process(validate=True)
    return final_ring

def create_wire_mesh(r_max: float, z_position: float, wire_diameter: float = 0.2, mesh_spacing: float = 2.0, segments: int = 16) -> trimesh.Trimesh:
    if wire_diameter <= 0 or mesh_spacing <= 0 or r_max <= 0:
        raise ValueError("Wire mesh dimensions must be positive.")
    wire_radius = wire_diameter / 2.0
    positions = np.arange(-r_max, r_max + mesh_spacing, mesh_spacing)
    wires = []
    for y in positions:
        if abs(y) > r_max:
            continue
        x_half_length = np.sqrt(max(0, r_max**2 - y**2))
        wire_length = 2 * x_half_length
        if wire_length < wire_diameter: continue
        wire = trimesh.creation.cylinder(radius=wire_radius, height=wire_length, sections=segments)
        rotation = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        wire.apply_transform(rotation)
        wire.apply_translation([0, y, z_position])
        wire.fill_holes()
        wires.append(wire)
    for x in positions:
        if abs(x) > r_max:
            continue
        y_half_length = np.sqrt(max(0, r_max**2 - x**2))
        wire_length = 2 * y_half_length
        if wire_length < wire_diameter: continue
        wire = trimesh.creation.cylinder(radius=wire_radius, height=wire_length, sections=segments)
        rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        wire.apply_transform(rotation)
        wire.apply_translation([x, 0, z_position])
        wire.fill_holes()
        wires.append(wire)
    if not wires:
        return trimesh.Trimesh()
    wire_mesh = trimesh.util.concatenate(wires)
    if not wire_mesh.is_watertight: wire_mesh.fill_holes(); wire_mesh.fix_normals()
    wire_mesh.process(validate=True)
    return wire_mesh

def create_copper_wire_ring(major_radius: float, minor_radius: float, sections: int = 64) -> trimesh.Trimesh:
    if major_radius <= 0 or minor_radius <= 0:
        raise ValueError("Wire ring radii must be positive.")
    wire_ring = trimesh.creation.torus(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_sections=sections,
        minor_sections=sections // 2
    )
    wire_ring.process(validate=True)
    return wire_ring

def compute_stl_dimensions(design):
    r_e_m = design['r_e_m']
    r_c_m = design['r_c_m']
    l_m = design['l_m']
    shape_emitter = design['shape_emitter']
    
    r_e_mm = r_e_m * 1000
    r_c_mm = r_c_m * 1000
    l_mm = l_m * 1000
    
    collector_inner_diameter_mm = 2 * r_c_mm
    collector_outer_diameter_mm = collector_inner_diameter_mm + 2 * COLLECTOR_WALL_THICKNESS_MM
    collector_length_mm = l_mm
    
    emitter_height_mm = 10.0  # Fixed as per simulation
    exposed_height_mm = 2.0   # Fixed exposure height for consistency
    emitter_mount_height_mm = emitter_height_mm - exposed_height_mm
    
    cap_plug_height_mm = max(5.0, 0.1 * r_c_mm)
    cap_flange_height_mm = max(5.0, 0.1 * r_c_mm)
    
    rigid_ring_inner_diameter_mm = collector_outer_diameter_mm + 1.0 + 0.2
    rigid_ring_wall_thickness_mm = max(6.0, 0.2 * r_c_mm)
    rigid_ring_height_mm = max(5.0, 0.1 * r_c_mm)
    
    cap_hole_diameter_mm = collector_inner_diameter_mm - 2.0
    mesh_spacing_mm = max(2.0, 0.02 * collector_inner_diameter_mm)
    
    copper_wire_major_radius_mm = (rigid_ring_inner_diameter_mm / 2.0) - (AWG_10_DIAMETER_MM / 2.0)
    
    return {
        'collector_inner_diameter_mm': collector_inner_diameter_mm,
        'collector_outer_diameter_mm': collector_outer_diameter_mm,
        'collector_length_mm': collector_length_mm,
        'emitter_radius_mm': r_e_mm,
        'emitter_height_mm': emitter_height_mm,
        'exposed_height_mm': exposed_height_mm,
        'emitter_mount_height_mm': emitter_mount_height_mm,
        'cap_plug_height_mm': cap_plug_height_mm,
        'cap_flange_height_mm': cap_flange_height_mm,
        'rigid_ring_inner_diameter_mm': rigid_ring_inner_diameter_mm,
        'rigid_ring_wall_thickness_mm': rigid_ring_wall_thickness_mm,
        'rigid_ring_height_mm': rigid_ring_height_mm,
        'cap_hole_diameter_mm': cap_hole_diameter_mm,
        'mesh_spacing_mm': mesh_spacing_mm,
        'copper_wire_major_radius_mm': copper_wire_major_radius_mm,
        'shape_emitter': shape_emitter
    }

def generate_stl_files(design, design_id, design_subdir):
    dims = compute_stl_dimensions(design)
    components = {}
    
    tube = create_collector_tube(
        inner_diameter_mm=dims['collector_inner_diameter_mm'],
        wall_thickness_mm=COLLECTOR_WALL_THICKNESS_MM,
        length_mm=dims['collector_length_mm']
    )
    save_stl(tube, "collector_tube.stl", design_subdir)
    components['collector_tube'] = tube
    
    emitter = create_emitter(
        r_e_mm=dims['emitter_radius_mm'],
        height_mm=dims['emitter_height_mm'],
        shape_emitter=dims['shape_emitter']
    )
    save_stl(emitter, "emitter.stl", design_subdir)
    components['emitter'] = emitter
    
    mount = create_emitter_mount(
        r_e_mm=dims['emitter_radius_mm'],
        collector_inner_diameter_mm=dims['collector_inner_diameter_mm'],
        emitter_height_mm=dims['emitter_height_mm'],
        exposure_percentage=0.2,  # Adjusted for reasonable exposure
        sections=64,
        engine='blender'
    )
    save_stl(mount, "emitter_mount.stl", design_subdir)
    components['emitter_mount'] = mount
    
    cap = create_collector_cap(
        collector_inner_diameter_mm=dims['collector_inner_diameter_mm'],
        collector_outer_diameter_mm=dims['collector_outer_diameter_mm'],
        plug_height_mm=dims['cap_plug_height_mm'],
        flange_height_mm=dims['cap_flange_height_mm'],
        sections=512,
        engine='blender'
    )
    save_stl(cap, "collector_cap.stl", design_subdir)
    components['collector_cap'] = cap
    
    rigid_ring = create_rigid_ring(
        inner_diameter_mm=dims['rigid_ring_inner_diameter_mm'],
        wall_thickness_mm=dims['rigid_ring_wall_thickness_mm'],
        height_mm=dims['rigid_ring_height_mm'],
        groove_diameter_mm=AWG_10_DIAMETER_MM,
        hole_radius_mm=AWG_10_DIAMETER_MM / 2 + 0.2,
        engine='manifold'
    )
    save_stl(rigid_ring, "rigid_ring.stl", design_subdir)
    components['rigid_ring'] = rigid_ring
    
    wire_mesh = create_wire_mesh(
        r_max=dims['cap_hole_diameter_mm'] / 2.0 - 1.0,
        z_position=0.0,  # Will be translated in assembly
        wire_diameter=0.2,
        mesh_spacing=dims['mesh_spacing_mm']
    )
    save_stl(wire_mesh, "wire_mesh.stl", design_subdir)
    components['wire_mesh'] = wire_mesh
    
    copper_wire = create_copper_wire_ring(
        major_radius=dims['copper_wire_major_radius_mm'],
        minor_radius=AWG_10_DIAMETER_MM / 2.0
    )
    save_stl(copper_wire, "copper_wire_ring.stl", design_subdir)
    components['copper_wire'] = copper_wire
    
    return components

def create_assembly(components, design):
    dims = compute_stl_dimensions(design)
    emitter_mount_height_mm = dims['emitter_mount_height_mm']
    collector_length_mm = dims['collector_length_mm']
    cap_plug_height_mm = dims['cap_plug_height_mm']
    cap_flange_height_mm = dims['cap_flange_height_mm']
    rigid_ring_height_mm = dims['rigid_ring_height_mm']
    
    tube_base_z = 0.0
    mount_base_z = 0.0  # Align bottom of emitter mount with collector tube base
    emitter_base_z = 0.0  # Position emitter base at mount base, inside the mount
    cap_base_z = tube_base_z + collector_length_mm - cap_plug_height_mm
    rigid_ring_base_z = tube_base_z + collector_length_mm
    mesh_z_position = cap_base_z + cap_plug_height_mm + cap_flange_height_mm
    copper_wire_z_position = rigid_ring_base_z + cap_flange_height_mm / 2.0
    
    components['collector_tube'].apply_translation([0, 0, tube_base_z])
    components['emitter_mount'].apply_translation([0, 0, mount_base_z])
    components['emitter'].apply_translation([0, 0, emitter_base_z])
    components['collector_cap'].apply_translation([0, 0, cap_base_z])
    components['rigid_ring'].apply_translation([0, 0, rigid_ring_base_z])
    components['wire_mesh'].apply_translation([0, 0, mesh_z_position])
    components['copper_wire'].apply_translation([0, 0, copper_wire_z_position])
    
    valid_parts = [part for part in components.values() if part and len(part.faces) > 0]
    if not valid_parts:
        raise ValueError("No valid parts to assemble.")
    assembly = trimesh.util.concatenate(valid_parts)
    assembly.process(validate=True)
    if not assembly.is_watertight:
        assembly.fill_holes()
        assembly.fix_normals()
    return assembly

def save_stl(trimesh_obj: trimesh.Trimesh, filename: str, output_dir: str) -> None:
    filepath = os.path.join(output_dir, filename)
    print(f"Saving {filename} to {filepath}...")
    try:
        if not trimesh_obj.is_watertight:
            print(f"Warning: Mesh '{filename}' is not watertight before saving. Attempting repair...")
            trimesh_obj.fill_holes()
            trimesh_obj.fix_normals()
        trimesh_obj.export(filepath)
        print(f"Successfully saved {filepath}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def save_design_files(design: Dict, design_id: str, payload_dir: str) -> None:
    design_subdir = os.path.join(payload_dir, f"design_{design_id}")
    os.makedirs(design_subdir, exist_ok=True)
    with open(os.path.join(design_subdir, "design_details.json"), 'w') as f:
        json.dump(design, f, indent=4)
    txt_output = (
        f"Design ID: {design_id}\n"
        f"Emitter Radius: {design['r_e_m'] * 1000:.3f} mm\n"
        f"Collector Radius: {design['r_c_m'] * 1000:.3f} mm\n"
        f"Gap Distance: {design['d_m'] * 1000:.3f} mm\n"
        f"Collector Length: {design['l_m']:.3f} m\n"
        f"Voltage: {design['V_V'] / 1000:.1f} kV\n"
        f"Shape: {design['shape_emitter']}\n"
        f"Emitter Material: {design['mat_emitter']}\n"
        f"Collector Material: {design['mat_collector']}\n"
        f"Thrust: {design['simulated_thrust_N']:.6f} N\n"
        f"Total System Weight: {design['system_weight_kg']:.4f} kg\n"
    )
    with open(os.path.join(design_subdir, "design_summary.txt"), 'w') as f:
        f.write(txt_output)
    components = generate_stl_files(design, design_id, design_subdir)
    assembly = create_assembly(components, design)
    save_stl(assembly, "assembly.stl", design_subdir)

def bayesian_optimization(shape, mat_emitter, mat_collector, payload_weight, materials_data, battery_specs, payload_dir, pbar=None):
    bounds = np.array([PARAM_BOUNDS[p] for p in ['r_e', 'r_c', 'd', 'l', 'V']])
    n_params = len(bounds)
    n_initial = 4
    n_iterations = 10
    max_no_improvement = 3

    sampler = qmc.LatinHypercube(d=n_params)
    X_initial = sampler.random(n=n_initial - 1)
    # Bias r_e toward lower end (~0.5-2 mm) and r_c toward upper end (~18-22 mm)
    X_initial[:, 0] = bounds[0, 0] + 0.2 * (bounds[0, 1] - bounds[0, 0])  # r_e
    X_initial[:, 1] = bounds[1, 0] + 0.8 * (bounds[1, 1] - bounds[1, 0])  # r_c
    X_initial[:, 2:] = bounds[2:, 0] + X_initial[:, 2:] * (bounds[2:, 1] - bounds[2:, 0])  # d, l, V
    prior_params = np.array([PRIOR_DATA['params'][p] for p in ['r_e', 'r_c', 'd', 'l', 'V']])
    if (PRIOR_DATA['shape'] == shape and PRIOR_DATA['mat_emitter'] == mat_emitter and PRIOR_DATA['mat_collector'] == mat_collector):
        X_initial = np.vstack([X_initial, prior_params])
    else:
        X_initial = X_initial[:n_initial - 1]

    # Normalize X_initial
    X_initial_normalized = (X_initial - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    y_initial = []
    configs_initial = []
    thrusts_initial = []
    statuses_initial = []
    design_counter = 0
    feasible_designs = []

    for x in X_initial:
        y, config, thrust, status = objective(x, shape, mat_emitter, mat_collector, payload_weight, materials_data, battery_specs)
        y_initial.append(y)
        configs_initial.append(config)
        thrusts_initial.append(thrust)
        statuses_initial.append(status)
        if pbar:
            pbar.update(1)
        if status == 0:
            rho_emitter = materials_data['materials'].get(mat_emitter, DEFAULT_MATERIALS.get(mat_emitter, {})).get('density_kg_m3', 8000)
            rho_collector = materials_data['materials'].get(mat_collector, DEFAULT_MATERIALS.get(mat_collector, {})).get('density_kg_m3', 8000)
            thruster_weight = calculate_thruster_weight(x[0], x[1], x[3], shape, rho_emitter, rho_collector)
            if thruster_weight:
                design = create_design_details(x, shape, mat_emitter, mat_collector, thrust, thruster_weight, config or {}, payload_weight)
                design_id = f"{shape}_{mat_emitter}_{mat_collector}_{design_counter}"
                save_design_files(design, design_id, payload_dir)
                design_counter += 1
                if y < 1e6 and config:
                    feasible_designs.append(design)

    y_initial = np.array(y_initial)
    X_normalized = X_initial_normalized.copy()
    y = y_initial.copy()
    best_y = min(y_initial)
    no_improvement_count = 0

    # Define kernel with anisotropic length scales and increased upper bound
    kernel = Matern(nu=2.5, length_scale=[1.0] * n_params, length_scale_bounds=[(1e-2, 1e5)] * n_params)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)

    for i in range(n_iterations):
        gp.fit(X_normalized, y)
        next_point_normalized = propose_next_point(X_normalized, y, gp, [(0, 1)] * n_params)
        next_point = bounds[:, 0] + next_point_normalized * (bounds[:, 1] - bounds[:, 0])
        next_y, config, thrust, status = objective(next_point, shape, mat_emitter, mat_collector, payload_weight, materials_data, battery_specs)
        X_normalized = np.vstack([X_normalized, next_point_normalized.reshape(1, -1)])
        y = np.append(y, next_y)
        if pbar:
            pbar.update(1)
        if status == 0:
            rho_emitter = materials_data['materials'].get(mat_emitter, DEFAULT_MATERIALS.get(mat_emitter, {})).get('density_kg_m3', 8000)
            rho_collector = materials_data['materials'].get(mat_collector, DEFAULT_MATERIALS.get(mat_collector, {})).get('density_kg_m3', 8000)
            thruster_weight = calculate_thruster_weight(next_point[0], next_point[1], next_point[3], shape, rho_emitter, rho_collector)
            if thruster_weight:
                design = create_design_details(next_point, shape, mat_emitter, mat_collector, thrust, thruster_weight, config or {}, payload_weight)
                design_id = f"{shape}_{mat_emitter}_{mat_collector}_{design_counter}"
                save_design_files(design, design_id, payload_dir)
                design_counter += 1
                if next_y < 1e6 and config:
                    feasible_designs.append(design)
        if next_y < best_y:
            best_y = next_y
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= max_no_improvement:
            break
    return feasible_designs

def propose_next_point(X_normalized, y, gp, normalized_bounds, n_restarts=10):
    n_params = X_normalized.shape[1]
    best_y = np.min(y)
    def ei(x_normalized):
        x_normalized = x_normalized.reshape(1, -1)
        mu, sigma = gp.predict(x_normalized, return_std=True)
        sigma = np.maximum(sigma, 1e-10)
        z = (best_y - mu) / sigma
        return -(mu - best_y) * norm.cdf(z) - sigma * norm.pdf(z)
    best_x_normalized, best_ei = None, float('inf')
    for _ in range(n_restarts):
        x0_normalized = np.random.uniform(0, 1, size=n_params)
        res = minimize(ei, x0_normalized, bounds=normalized_bounds, method='L-BFGS-B')
        if res.fun < best_ei:
            best_ei = res.fun
            best_x_normalized = res.x
    return best_x_normalized

def optimize_design(payload_weight: float):
    materials_data = load_json_file('materials.json', {'materials': DEFAULT_MATERIALS}, 'material properties')
    battery_specs = load_json_file('samsung_21700_batteries.json', DEFAULT_BATTERIES, 'battery specifications')
    MATERIALS = list(materials_data['materials'].keys())
    payload_dir = f"optimization_results/payload_{payload_weight:.3f}kg"
    os.makedirs(payload_dir, exist_ok=True)
    combinations = [(s, me, mc) for s in ['pointed', 'cylindrical', 'hexagonal'] for me in MATERIALS for mc in MATERIALS]
    n_initial = 4
    n_iterations = 10
    total_simulations = len(combinations) * (n_initial + n_iterations)
    pbar = tqdm(total=total_simulations, desc="Overall simulations")
    all_feasible_designs = []

    for shape, mat_emitter, mat_collector in combinations:
        feasible_designs = bayesian_optimization(shape, mat_emitter, mat_collector, payload_weight, materials_data, battery_specs, payload_dir, pbar)
        all_feasible_designs.extend(feasible_designs)
    pbar.close()

    if not all_feasible_designs:
        print("No feasible designs found.")
        return None

    best_design = min(all_feasible_designs, key=lambda x: x['system_weight_kg'])
    txt_output = (
        f"Optimal Design\n"
        f"Payload Weight: {payload_weight:.4f} kg\n"
        f"Emitter Radius: {best_design['r_e_m'] * 1000:.3f} mm\n"
        f"Collector Radius: {best_design['r_c_m'] * 1000:.3f} mm\n"
        f"Thrust: {best_design['simulated_thrust_N']:.6f} N\n"
        f"Total Weight: {best_design['system_weight_kg']:.4f} kg\n"
    )
    with open(os.path.join(payload_dir, f"optimal_design_{payload_weight:.3f}kg.txt"), 'w') as f:
        f.write(txt_output)

    print(f"\nSaved {len(all_feasible_designs)} feasible designs in {payload_dir}.")
    print("\nOptimal Design:")
    print(f"Emitter Radius: {best_design['r_e_m']*1000:.2f} mm")
    print(f"Collector Radius: {best_design['r_c_m']*1000:.2f} mm")
    print(f"Gap Distance: {best_design['d_m']*1000:.2f} mm")
    print(f"Length: {best_design['l_m']:.2f} m")
    print(f"Voltage: {best_design['V_V']/1000:.1f} kV")
    print(f"Shape: {best_design['shape_emitter']}")
    print(f"Emitter Material: {best_design['mat_emitter']}")
    print(f"Collector Material: {best_design['mat_collector']}")
    print(f"Thrust: {best_design['simulated_thrust_N']:.3f} N")
    print(f"Total System Weight: {best_design['system_weight_kg']:.3f} kg")
    return best_design

if __name__ == "__main__":
    try:
        payload_weight = float(input("Enter payload weight (kg): "))
        if payload_weight <= 0:
            raise ValueError("Payload weight must be positive.")
        optimize_design(payload_weight)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        print(f"Unexpected error: {e}")
    finally:
        tf.keras.backend.clear_session()