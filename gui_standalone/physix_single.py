# Enhanced EHD Physics Simulation Module (physix.py)
# Incorporates Pressure Correction, Townsend Ionization, and Material Property Hooks
import sys
import os
import numpy as np
import numba
from numba import njit, prange
import tkinter as tk  # <-- ADD THIS
from tkinter import scrolledtext # <-- ADD THIS
import traceback
import tensorflow as tf

# Ensure GPU is used
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected and configured:", gpus)
    except RuntimeError as e:
        print("Error configuring GPU:", e)
else:
    print("No GPU detected. Running on CPU.")

DEBUG_FLAG = tf.Variable(True, dtype=tf.bool, trainable=False, name="DEBUG_FLAG")

DEBUG_FLAG.assign(False)

@tf.function
def conditional_tf_print(condition: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    """
    Conditionally prints arguments based on a boolean condition using tf.print.

    Args:
        condition: A scalar tf.Tensor (tf.bool) indicating whether to print.
        *args: Positional arguments to print.
        **kwargs: Keyword arguments for tf.print (e.g., output_stream).

    Returns:
        A dummy tensor (tf.int32) for graph compatibility.
    """
    def print_op() -> tf.Tensor:
        tf.print(*args, **kwargs)
        return tf.constant(0, dtype=tf.int32)
    
    def no_op() -> tf.Tensor:
        return tf.constant(0, dtype=tf.int32)
    
    return tf.cond(condition, print_op, no_op)

# Define a log file path for tf.print output
LOG_FILE_PATH = 'file://physix_log.txt'

# Function to initialize logging
def initialize_logging():
    """Initialize logging to ensure the log file is accessible and writable."""
    try:
        # Extract the file path without the 'file://' prefix for local operations
        local_log_path = LOG_FILE_PATH.replace('file://', '')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_log_path) or '.', exist_ok=True)
        # Test write to ensure the file is writable
        with open(local_log_path, 'a') as f:
            f.write('')
        print(f"Logging initialized to {LOG_FILE_PATH}")
    except IOError as e:
        print(f"Warning: Could not initialize log file '{local_log_path}': {e}. Falling back to stdout.")
        return sys.stdout
    return LOG_FILE_PATH

# Initialize logging and set the output stream
_output_stream = initialize_logging()

# Override tf.print to write to the log file
_original_tf_print = tf.print
def _custom_tf_print(*args, **kwargs):
    """Custom tf.print that writes to a dedicated log file or stdout."""
    kwargs['output_stream'] = _output_stream
    _original_tf_print(*args, **kwargs)

tf.print = _custom_tf_print

# ==================================
# Physical Constants and Parameters
# ==================================
# Physical Constants
EPSILON_0 = tf.constant(8.854e-12, dtype=tf.float32)  # Permittivity of free space (F/m)
RHO_AIR = tf.constant(1.225, dtype=tf.float32)        # Air density (kg/m^3) at standard conditions
MU_AIR = tf.constant(1.8e-5, dtype=tf.float32)        # Dynamic viscosity of air (Pa·s)
MU_ION = tf.constant(2e-4, dtype=tf.float32)          # Approximate ion mobility in air (m^2/V·s)
E_ONSET_DEFAULT = tf.constant(3e6, dtype=tf.float32)  # Default Corona onset field strength (V/m)
P_ATM = tf.constant(101325.0, dtype=tf.float32)       # Standard atmospheric pressure (Pa)
RHO_AIR_SEA_LEVEL = tf.constant(1.225, dtype=tf.float32)  # kg/m^3 at sea level

# Townsend Coefficients for air
TOWNSEND_A_DEFAULT = tf.constant(15.0, dtype=tf.float32)  # 1/(Pa·m)
TOWNSEND_B_DEFAULT = tf.constant(365.0, dtype=tf.float32)  # V/(Pa·m)

# Simulation Control Parameters
DEFAULT_TOL = tf.constant(1e-3, dtype=tf.float32)     # Default tolerance for iterative solvers
DEFAULT_DT = tf.constant(1e-9, dtype=tf.float32)      # Default time step (s)
DEFAULT_STEPS = tf.constant(15, dtype=tf.int32)       # Default number of steps per physics solve
COUPLING_ITERATIONS = tf.constant(10, dtype=tf.int32) # Number of iterations for physics coupling loop

# Diffusion and Mobility Coefficients
D_ION = tf.constant(5e-5, dtype=tf.float32)           # Ion diffusion coefficient (m^2/s)
MU_E = tf.constant(0.04, dtype=tf.float32)            # Electron mobility (m^2/V·s)
DEFAULT_MU_I = tf.constant(2e-4, dtype=tf.float32)    # Default ion mobility (m^2/V·s)
DEFAULT_D_I = tf.constant(5e-5, dtype=tf.float32)     # Default ion diffusion coefficient (m^2/s)
MU_I = tf.constant(2e-4, dtype=tf.float32)            # Ion mobility (m^2/V·s)
D_E = tf.constant(5e-3, dtype=tf.float32)             # Electron diffusion coefficient (m^2/s)
D_I = tf.constant(5e-5, dtype=tf.float32)             # Ion diffusion coefficient (m^2/s)
BETA_RECOMB = tf.constant(2e-14, dtype=tf.float32)    # Recombination coefficient (m^3/s)
ELEM_CHARGE = tf.constant(1.602e-19, dtype=tf.float32)  # Elementary charge (C)
ETA_ATTACH = tf.constant(1.0e-15, dtype=tf.float32)   # Attachment coefficient (m^3/s)
SOR_OMEGA = tf.constant(1.0, dtype=tf.float32)        # SOR relaxation parameter

# Predefined TensorFlow Constants
TF_FLOAT32_ZERO = tf.constant(0.0, dtype=tf.float32)
TF_FLOAT32_ONE = tf.constant(1.0, dtype=tf.float32)
TF_FLOAT32_TWO = tf.constant(2.0, dtype=tf.float32)
TF_FLOAT32_HALF = tf.constant(0.5, dtype=tf.float32)
TF_EPSILON = tf.constant(1e-30, dtype=tf.float32)     # Small value for stability

TF_FLOAT32_1_OVER_3 = tf.constant(1.0/3.0, dtype=tf.float32)
TF_FLOAT32_7_OVER_6 = tf.constant(7.0/6.0, dtype=tf.float32)
TF_FLOAT32_11_OVER_6 = tf.constant(11.0/6.0, dtype=tf.float32)
TF_FLOAT32_NEG_1_OVER_6 = tf.constant(-1.0/6.0, dtype=tf.float32)
TF_FLOAT32_5_OVER_6 = tf.constant(5.0/6.0, dtype=tf.float32)
TF_FLOAT32_13_OVER_12 = tf.constant(13.0/12.0, dtype=tf.float32)
TF_FLOAT32_1_OVER_4 = tf.constant(1.0/4.0, dtype=tf.float32)
TF_FLOAT32_1_0 = tf.constant(1.0, dtype=tf.float32)
TF_FLOAT32_2_0 = tf.constant(2.0, dtype=tf.float32)
TF_FLOAT32_3_0 = tf.constant(3.0, dtype=tf.float32)
TF_FLOAT32_4_0 = tf.constant(4.0, dtype=tf.float32)
TF_FLOAT32_WENO_STABILITY = tf.constant(1e-40, dtype=tf.float32)  # Stability for normalization
TF_FLOAT32_NEG_7_OVER_6 = tf.constant(-7.0/6.0, dtype=tf.float32)
TF_FLOAT32_THREE = tf.constant(3.0, dtype=tf.float32)
TF_FLOAT32_FOUR = tf.constant(4.0, dtype=tf.float32)
TF_FLOAT32_NEG_2_0 = tf.constant(-2.0, dtype=tf.float32)

TF_ELEM_CHARGE = tf.constant(1.602e-19, dtype=tf.float32)
TF_EPSILON_0 = tf.constant(8.854e-12, dtype=tf.float32)
TF_P_ATM_STD = tf.constant(101325.0, dtype=tf.float32)
TF_R_SPECIFIC_AIR = tf.constant(287.058, dtype=tf.float32)
TF_ZERO = tf.constant(0.0, dtype=tf.float32)
TF_ONE = tf.constant(1.0, dtype=tf.float32)
TF_TWO = tf.constant(2.0, dtype=tf.float32)
TF_HALF = tf.constant(0.5, dtype=tf.float32)
TF_THREE = tf.constant(3.0, dtype=tf.float32)
TF_FOUR = tf.constant(4.0, dtype=tf.float32)
TF_SIX = tf.constant(6.0, dtype=tf.float32)
TF_SQRT3 = tf.constant(1.7320508075688772, dtype=tf.float32)  # sqrt(3) computed directly
TF_PI = tf.constant(3.141592653589793, dtype=tf.float32)      # pi value directly
TF_SMALL_VEL = tf.constant(1e-9, dtype=tf.float32)            # For CFL stability
TF_SMALL_DENSITY = tf.constant(1e-6, dtype=tf.float32)        # Threshold for electron seeding
conditional_tf_print(DEBUG_FLAG,"INFO: Using TF_SMALL_DENSITY threshold:", TF_SMALL_DENSITY)
TF_SMALL_SPACING = tf.constant(1e-12, dtype=tf.float32)       # Threshold for grid spacing
TF_SMALL_THERMAL = tf.constant(1e-9, dtype=tf.float32)        # For thermal mass stability

TF_MU_E = tf.constant(40.0, dtype=tf.float32)                 # Example electron mobility
TF_MU_ION = tf.constant(1.4e-4, dtype=tf.float32)             # Example ion mobility
TF_D_E = tf.constant(0.1, dtype=tf.float32)                   # Example electron diffusion
TF_D_ION = tf.constant(3e-6, dtype=tf.float32)                # Example ion diffusion
TF_BETA_RECOMB = tf.constant(1.6e-13, dtype=tf.float32)       # Example recombination value
TF_DEFAULT_DT = tf.constant(1e-7, dtype=tf.float32)           # Example default timestep

# Status Codes
SUCCESS_CODE = tf.constant(0, dtype=tf.int32)
NAN_INF_STEP_CODE = tf.constant(2, dtype=tf.int32)
NAN_INF_FINAL_CODE = tf.constant(3, dtype=tf.int32)
GRID_ERROR_CODE = tf.constant(5, dtype=tf.int32)
INIT_ERROR_CODE = tf.constant(6, dtype=tf.int32)
UNKNOWN_EXCEPTION_CODE = tf.constant(99, dtype=tf.int32)

_WENO_EPSILON = tf.constant(1e-6, dtype=tf.float32)
_WENO_D0 = tf.constant(0.1, dtype=tf.float32)
_WENO_D1 = tf.constant(0.6, dtype=tf.float32)
_WENO_D2 = tf.constant(0.3, dtype=tf.float32)
NAN_INF_STEP_TENSOR = tf.constant(2, dtype=tf.int32)  # Matches NAN_INF_STEP_CODE
TF_EPSILON_MIN_SPACING = tf.constant(1e-9, dtype=tf.float32)

def air_density_tf(altitude_m):
    """
    Calculates air density using a simplified exponential decay model,
    implemented in TensorFlow for GPU acceleration with float32 precision.

    Formula: rho = rho_0 * exp(-h/H), where H ~ 8000m scale height.

    Args:
        altitude_m (tf.Tensor or compatible): Altitude(s) in meters.
                                              Can be a scalar, list, NumPy array,
                                              or TensorFlow tensor. It will be
                                              converted to tf.float32.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - density_tf (tf.Tensor): Calculated air density (kg/m^3) as a
                                      tf.float32 tensor with the same shape
                                      as the input altitude_m.
            - error_code (tf.Tensor): Status code indicating success or failure:
                                      0 = Success
                                      3 = NaN/Inf detected in the final result
                                      4 = Unexpected exception during execution
                                      dtype=tf.int32.
    """
    try:
        # --- Constants ---
        # Define necessary constants as tf.constant with tf.float32 dtype
        H_tf = tf.constant(8000.0, dtype=tf.float32, name="ScaleHeight")
        RHO_AIR_SEA_LEVEL_tf = tf.constant(RHO_AIR_SEA_LEVEL, dtype=tf.float32, name="SeaLevelDensity")

        # --- Input Processing ---
        # Convert the input altitude to a tf.float32 tensor.
        altitude_tf = tf.convert_to_tensor(altitude_m, dtype=tf.float32, name="Altitude")

        # --- Core Calculation ---
        exponent = tf.negative(altitude_tf / H_tf, name="Exponent")
        density_tf = tf.multiply(RHO_AIR_SEA_LEVEL_tf, tf.exp(exponent), name="Density")

        # --- Error Checking ---
        # Check if ALL elements in the final result tensor are finite (not NaN or Inf).
        is_finite_result = tf.reduce_all(tf.math.is_finite(density_tf), name="IsFiniteCheck")

        # Determine the error code using tf.cond for graph compatibility.
        error_code = tf.cond(
            is_finite_result,
            lambda: SUCCESS_CODE,        # Return 0 if finite
            lambda: NAN_INF_FINAL_CODE   # Return 3 if NaN/Inf found
        )

        # --- Corrected Conditional Printing ---
        # Define functions that perform the action and return a compatible type (tf.int32 scalar)
        def print_warning_and_return_zero():
            conditional_tf_print(DEBUG_FLAG,"WARNING (air_density_tf): NaN or Inf detected in output density.", output_stream=sys.stderr)
            # Return a tensor matching the type and shape of the other branch
            return tf.constant(0, dtype=tf.int32)

        def no_op_return_zero():
            # Return a tensor matching the type and shape of the other branch
            return tf.constant(0, dtype=tf.int32)

        # Conditionally execute the print, ensuring both branches return compatible tensors
        _ = tf.cond(
              tf.equal(error_code, NAN_INF_FINAL_CODE),
              print_warning_and_return_zero, # Calls the function that prints and returns int32
              no_op_return_zero              # Calls the function that just returns int32
        )
        # Assigning to '_' clarifies that we don't use the return value of this tf.cond

    except Exception as e:
        # Basic exception handling during graph construction/tracing.
        conditional_tf_print(DEBUG_FLAG,"ERROR (air_density_tf): Unexpected exception during execution:", e, output_stream=sys.stderr)
        # Return tensors indicating failure. Shape might be unknown if input failed.
        try:
            out_shape = tf.shape(altitude_m)
        except: # If input shape is not available
             out_shape = [] # Scalar shape
        density_tf = tf.fill(out_shape, tf.constant(np.nan, dtype=tf.float32))
        # Use a distinct error code for unexpected exceptions
        error_code = tf.constant(4, dtype=tf.int32, name="ExceptionErrorCode")

    return density_tf, error_code

# ==================================
# Grid and Electrode Setup
# ==================================

def custom_linspace(start, stop, num):
    """
    Generates a linearly spaced tensor using GPU-compatible operations.
    
    Args:
        start (tf.Tensor): Starting value (tf.float32).
        stop (tf.Tensor): Ending value (tf.float32).
        num (tf.Tensor or int): Number of points (tf.int32 or convertible).
    
    Returns:
        tf.Tensor: Linearly spaced tensor of shape [num], dtype=tf.float32.
    """
    num = tf.cast(num, tf.int32)  # Ensure num is int32
    step = (stop - start) / tf.cast(num - 1, tf.float32)  # Compute step size as float32
    indices = tf.range(num, dtype=tf.float32)  # Generate indices as float32
    return start + step * indices  # Linearly spaced values, all float32 operations

@tf.function
def stretched_linspace(physical_min, physical_max, num_points, stretching_factor, cluster_point=None):
    """
    Generates non-uniformly spaced coordinates using hyperbolic tangent stretching.
    If cluster_point is specified, stretches toward that point with symmetric stretching
    around the domain midpoint if cluster_point is near it; otherwise, uses endpoint
    stretching toward the cluster_point.

    Args:
        physical_min (tf.Tensor): Minimum physical coordinate (tf.float32).
        physical_max (tf.Tensor): Maximum physical coordinate (tf.float32).
        num_points (tf.Tensor or int): Number of points (tf.int32).
        stretching_factor (tf.Tensor or float): Stretching factor (tf.float32).
        cluster_point (tf.Tensor or float, optional): Point to cluster points toward (tf.float32).

    Returns:
        tf.Tensor: Non-uniform coordinates [num_points], dtype=tf.float32.
    """
    num_points = tf.cast(num_points, tf.int32)
    stretching_factor = tf.cast(stretching_factor, tf.float32)
    physical_min = tf.cast(physical_min, tf.float32)
    physical_max = tf.cast(physical_max, tf.float32)

    TF_FLOAT32_ZERO = tf.constant(0.0, dtype=tf.float32)
    TF_FLOAT32_ONE = tf.constant(1.0, dtype=tf.float32)
    TF_FLOAT32_TWO = tf.constant(2.0, dtype=tf.float32)
    TF_FLOAT32_HALF = tf.constant(0.5, dtype=tf.float32)
    TF_EPSILON = tf.constant(1e-30, dtype=tf.float32)

    if cluster_point is not None:
        cluster_point = tf.cast(cluster_point, tf.float32)
        midpoint = (physical_min + physical_max) / 2
        is_midpoint = tf.abs(cluster_point - midpoint) < 1e-6
        indices = tf.range(num_points, dtype=tf.float32)
        denom = tf.cast(tf.maximum(num_points - 1, 1), tf.float32)

        def symmetric_stretching():
            xi = -TF_FLOAT32_ONE + (TF_FLOAT32_TWO / denom) * indices
            tanh_stretch = tf.tanh(stretching_factor)
            safe_tanh_stretch = tf.where(tf.equal(tanh_stretch, 0.0), TF_FLOAT32_ONE, tanh_stretch)
            mapped_xi = tf.tanh(stretching_factor * xi) / safe_tanh_stretch
            coords = midpoint + ((physical_max - physical_min) / 2) * mapped_xi
            return coords

        def endpoint_stretching():
            xi = TF_FLOAT32_ZERO + (TF_FLOAT32_ONE / denom) * indices
            tanh_stretch = tf.tanh(stretching_factor)
            safe_tanh_stretch = tf.where(tf.equal(tanh_stretch, 0.0), TF_FLOAT32_ONE, tanh_stretch)
            mapped_xi = tf.tanh(stretching_factor * xi) / safe_tanh_stretch
            squash_epsilon = tf.constant(1e-7, dtype=tf.float32)
            mapped_xi_squashed = mapped_xi * (TF_FLOAT32_ONE - squash_epsilon)
            domain_length = physical_max - physical_min
            coords = tf.cond(
                cluster_point < midpoint,
                lambda: physical_min + domain_length * mapped_xi_squashed,
                lambda: physical_max - domain_length * (TF_FLOAT32_ONE - mapped_xi_squashed)
            )
            return tf.clip_by_value(coords, physical_min, physical_max)

        coords = tf.cond(is_midpoint, symmetric_stretching, endpoint_stretching)
    else:
        indices = tf.range(num_points, dtype=tf.float32)
        step = (physical_max - physical_min) / tf.cast(tf.maximum(num_points - 1, 1), tf.float32)
        coords = physical_min + step * indices

    return coords

@tf.function
def setup_grid_tf(
    r_e, r_c, d, l,
    nx_base=16, ny_base=16, nz_base=32,  # Reduced default base sizes
    min_points_per_feature=10,  # Reduced from 50
    max_elements=5e6,  # Reduced from 1e7
    stretching_factor_x=3.0,  # Increased from 2.0 to reduce min dx
    stretching_factor_y=3.0,  # Increased from 2.0 to reduce min dy
    stretching_factor_z=2.0   # Kept at 2.0 as dz is sufficient
):
    """
    Sets up the 3D simulation grid with dynamic resolution and non-uniform stretching
    in X, Y, and Z directions using TensorFlow. Ensures grid sizes are powers of 2 and
    total elements do not exceed max_elements after adjustment.

    Args:
        r_e (tf.Tensor or float): Emitter radius (m), expected as tf.float32.
        r_c (tf.Tensor or float): Collector radius (m), expected as tf.float32.
        d (tf.Tensor or float): Gap distance between electrodes (m), expected as tf.float32.
        l (tf.Tensor or float): Emitter length (m), expected as tf.float32.
        nx_base, ny_base, nz_base (tf.Tensor or int): Base grid resolutions, expected as tf.int32.
        min_points_per_feature (int): Minimum grid points across the smallest feature, converted to tf.float32.
        max_elements (float): Maximum total grid elements, converted to tf.float32.
        stretching_factor_x (float): Factor controlling X-axis stretching.
        stretching_factor_y (float): Factor controlling Y-axis stretching.
        stretching_factor_z (float): Factor controlling Z-axis stretching.

    Returns:
        Tuple: x, y, z (1D coordinate tensors, tf.float32), nx, ny, nz (scalar tensors, tf.int32).
    """
    # --- Input Handling ---
    r_e_tf = tf.cast(r_e, dtype=tf.float32)
    r_c_tf = tf.cast(r_c, dtype=tf.float32)
    d_tf = tf.cast(d, dtype=tf.float32)
    l_tf = tf.cast(l, dtype=tf.float32)
    nx_base_tf = tf.cast(nx_base, dtype=tf.int32)
    ny_base_tf = tf.cast(ny_base, dtype=tf.int32)
    nz_base_tf = tf.cast(nz_base, dtype=tf.int32)
    min_points_tf = tf.cast(min_points_per_feature, dtype=tf.float32)
    max_elements_tf = tf.cast(max_elements, dtype=tf.float32)
    stretch_x = tf.constant(stretching_factor_x, dtype=tf.float32)
    stretch_y = tf.constant(stretching_factor_y, dtype=tf.float32)
    stretch_z = tf.constant(stretching_factor_z, dtype=tf.float32)

    # Define float32 constants
    tf_0_1 = tf.constant(0.1, dtype=tf.float32)
    tf_0_5 = tf.constant(0.5, dtype=tf.float32)
    tf_1_0 = tf.constant(1.0, dtype=tf.float32)
    tf_2_0 = tf.constant(2.0, dtype=tf.float32)
    tf_3_0 = tf.constant(3.0, dtype=tf.float32)
    tf_zero = tf.constant(0.0, dtype=tf.float32)
    tf_epsilon = tf.constant(1e-9, dtype=tf.float32)

    # --- Domain Size Calculation ---
    buffer = tf.maximum(r_c_tf * tf_0_5, tf.maximum(d_tf * tf_0_1, tf.constant(0.01, dtype=tf.float32)))
    x_max = r_c_tf + buffer
    y_max = r_c_tf + buffer
    z_min = -l_tf / tf_2_0 - buffer
    z_max = d_tf + l_tf / tf_2_0 + buffer

    x_length = tf_2_0 * x_max
    y_length = tf_2_0 * y_max
    z_length = z_max - z_min

    # --- Target Spacings ---
    dx_target = r_e_tf / (min_points_tf + tf_epsilon)
    dy_target = dx_target
    dz_target = tf.minimum(dx_target, tf.maximum(d_tf, l_tf / tf_2_0) / (min_points_tf + tf_epsilon))

    # --- Initial Grid Sizes ---
    nx_init = tf.maximum(nx_base_tf, tf.cast(tf.math.ceil(x_length / (dx_target + tf_epsilon)), dtype=tf.int32))
    ny_init = tf.maximum(ny_base_tf, tf.cast(tf.math.ceil(y_length / (dy_target + tf_epsilon)), dtype=tf.int32))
    nz_init = tf.maximum(nz_base_tf, tf.cast(tf.math.ceil(z_length / (dz_target + tf_epsilon)), dtype=tf.int32))

    # Ensure minimum grid size to resolve r_e
    min_nx_ny = tf.cast(tf.math.ceil(r_e_tf / dx_target * 10.0), tf.int32)  # At least 10 points across r_e
    nx_init = tf.maximum(nx_init, min_nx_ny)
    ny_init = tf.maximum(ny_init, min_nx_ny)

    # --- Enforce Memory Constraint Early ---
    total_elements_init = tf.cast(nx_init, tf.float32) * \
                         tf.cast(ny_init, tf.float32) * \
                         tf.cast(nz_init, tf.float32)

    def scale_grid():
        conditional_tf_print(DEBUG_FLAG,"Grid requires scaling to fit memory constraint.")
        scale_factor = tf.minimum(tf_1_0, tf.pow(max_elements_tf / (total_elements_init + tf_epsilon), tf_1_0 / tf_3_0))
        nx_scaled = tf.maximum(nx_base_tf, tf.cast(tf.cast(nx_init, tf.float32) * scale_factor, dtype=tf.int32))
        ny_scaled = tf.maximum(ny_base_tf, tf.cast(tf.cast(ny_init, tf.float32) * scale_factor, dtype=tf.int32))
        nz_scaled = tf.maximum(nz_base_tf, tf.cast(tf.cast(nz_init, tf.float32) * scale_factor, dtype=tf.int32))
        return nx_scaled, ny_scaled, nz_scaled

    def dont_scale_grid():
        conditional_tf_print(DEBUG_FLAG,"Grid size within memory constraints. No scaling needed.")
        return nx_init, ny_init, nz_init

    nx_scaled, ny_scaled, nz_scaled = tf.cond(total_elements_init > max_elements_tf, scale_grid, dont_scale_grid)

    # --- Adjust to Power of 2 ---
    def next_power_of_2(n):
        n_float = tf.cast(n, dtype=tf.float32)
        return tf.cast(tf.pow(2.0, tf.math.ceil(tf.math.log(n_float + tf_epsilon) / tf.math.log(2.0))), dtype=tf.int32)

    nx_p2 = next_power_of_2(nx_scaled)
    ny_p2 = next_power_of_2(ny_scaled)
    nz_p2 = next_power_of_2(nz_scaled)

    # --- Reduction Mechanism to Ensure Total Elements <= max_elements ---
    min_size = tf.constant(16, dtype=tf.int32)  # Minimum grid size per dimension

    def condition(nx, ny, nz):
        total = tf.cast(nx, tf.float32) * tf.cast(ny, tf.float32) * tf.cast(nz, tf.float32)
        return tf.logical_and(total > max_elements_tf,
                              tf.logical_and(nx > min_size,
                                             tf.logical_and(ny > min_size, nz > min_size)))

    def body(nx, ny, nz):
        # Find the largest dimension
        dims = tf.stack([nx, ny, nz])
        max_dim_idx = tf.argmax(dims)
        # Reduce the largest dimension by half
        nx_new = tf.cond(tf.equal(max_dim_idx, 0), lambda: nx // 2, lambda: nx)
        ny_new = tf.cond(tf.equal(max_dim_idx, 1), lambda: ny // 2, lambda: ny)
        nz_new = tf.cond(tf.equal(max_dim_idx, 2), lambda: nz // 2, lambda: nz)
        return [nx_new, ny_new, nz_new]

    nx_final, ny_final, nz_final = tf.while_loop(
        condition,
        body,
        [nx_p2, ny_p2, nz_p2],
        maximum_iterations=10
    )

    # Ensure final grid sizes are at least the base sizes
    nx = tf.maximum(nx_final, nx_base_tf)
    ny = tf.maximum(ny_final, ny_base_tf)
    nz = tf.maximum(nz_final, nz_base_tf)

    # Final total elements check
    total_elements_final = tf.cast(nx, tf.float32) * tf.cast(ny, tf.float32) * tf.cast(nz, tf.float32)
    tf.debugging.assert_less_equal(total_elements_final, max_elements_tf,
                                   message="Final grid size still exceeds max_elements after reduction.")
    conditional_tf_print(DEBUG_FLAG,"Final grid sizes after reduction: nx=", nx, "ny=", ny, "nz=", nz, "Total Elements=", total_elements_final)

    # --- Generate Coordinate Arrays ---
    # X-grid: Split into left and right halves, clustering towards x=0
    nx_left = (nx + 1) // 2
    nx_right = nx - nx_left + 1
    conditional_tf_print(DEBUG_FLAG,"Generating non-uniform X-grid with stretching factor:", stretch_x, "nx_left=", nx_left, "nx_right=", nx_right)
    x_left = stretched_linspace(-x_max, tf_zero, nx_left, stretch_x, cluster_point=tf.constant(0.0, dtype=tf.float32))
    x_right = stretched_linspace(tf_zero, x_max, nx_right, stretch_x, cluster_point=tf.constant(0.0, dtype=tf.float32))
    x = tf.concat([x_left[:-1], x_right], axis=0)

    # Y-grid
    ny_left = (ny + 1) // 2
    ny_right = ny - ny_left + 1
    conditional_tf_print(DEBUG_FLAG,"Generating non-uniform Y-grid with stretching factor:", stretch_y, "ny_left=", ny_left, "ny_right=", ny_right)
    y_left = stretched_linspace(-y_max, tf_zero, ny_left, stretch_y, cluster_point=tf.constant(0.0, dtype=tf.float32))
    y_right = stretched_linspace(tf_zero, y_max, ny_right, stretch_y, cluster_point=tf.constant(0.0, dtype=tf.float32))
    y = tf.concat([y_left[:-1], y_right], axis=0)

    # Z-grid: Split into lower and upper halves, clustering towards z=0
    nz_lower = (nz + 1) // 2
    nz_upper = nz - nz_lower + 1
    conditional_tf_print(DEBUG_FLAG,"Generating non-uniform Z-grid with stretching factor:", stretch_z, "nz_lower=", nz_lower, "nz_upper=", nz_upper)
    z_lower = stretched_linspace(z_min, tf_zero, nz_lower, stretch_z, cluster_point=tf.constant(0.0, dtype=tf.float32))
    z_upper = stretched_linspace(tf_zero, z_max, nz_upper, stretch_z, cluster_point=tf.constant(0.0, dtype=tf.float32))
    z = tf.concat([z_lower[:-1], z_upper], axis=0)

    # --- Verify Final Grid Spacings ---
    conditional_tf_print(DEBUG_FLAG,"--- Grid Generation Final Check ---")
    min_dx_final = tf.reduce_min(tf.abs(x[1:] - x[:-1]))
    min_dy_final = tf.reduce_min(tf.abs(y[1:] - y[:-1]))
    min_dz_final = tf.reduce_min(tf.abs(z[1:] - z[:-1]))
    conditional_tf_print(DEBUG_FLAG,"Final min spacings: dx=", min_dx_final, "dy=", min_dy_final, "dz=", min_dz_final)

    # Check for origin coverage
    min_abs_x = tf.reduce_min(tf.abs(x))
    min_abs_y = tf.reduce_min(tf.abs(y))
    min_abs_z = tf.reduce_min(tf.abs(z))
    conditional_tf_print(DEBUG_FLAG,"Minimum absolute grid coords: x=", min_abs_x, "y=", min_abs_y, "z=", min_abs_z)

    # Ensure points are close to origin
    tf.debugging.assert_less(min_abs_x, r_e_tf / 2.0, message="Grid does not have point sufficiently close to x=0 for emitter resolution.")
    tf.debugging.assert_less(min_abs_y, r_e_tf / 2.0, message="Grid does not have point sufficiently close to y=0 for emitter resolution.")
    tf.debugging.assert_less(min_abs_z, r_e_tf / 2.0, message="Grid does not have point sufficiently close to z=0 for emitter resolution.")

    # --- Spacing Target Assertions ---
    min_spacing_target_xy = r_e_tf / 10.0  # Target: 1/10th of emitter radius for x and y
    min_spacing_target_z = tf.constant(4e-4, dtype=tf.float32)  # Target: 400 micrometers for z

    tf.debugging.Assert(
        tf.less_equal(min_dx_final, min_spacing_target_xy),
        [tf.strings.format("Minimum dx ({}) exceeds target ({}). Increase resolution or stretching.",
                           (min_dx_final, min_spacing_target_xy))]
    )
    tf.debugging.Assert(
        tf.less_equal(min_dy_final, min_spacing_target_xy),
        [tf.strings.format("Minimum dy ({}) exceeds target ({}). Increase resolution or stretching.",
                           (min_dy_final, min_spacing_target_xy))]
    )
    tf.debugging.Assert(
        tf.less_equal(min_dz_final, min_spacing_target_z),
        [tf.strings.format("Minimum dz ({}) exceeds target ({}). Increase resolution or stretching.",
                           (min_dz_final, min_spacing_target_z))]
    )

    conditional_tf_print(DEBUG_FLAG,"--- End Grid Generation Final Check ---")

    # --- Compute Spacing Statistics ---
    def compute_spacing_stats(coord_array, dim):
        dim_int32 = tf.cast(dim, dtype=tf.int32)
        spacings = tf.cond(dim_int32 > 1,
                           lambda: tf.abs(coord_array[1:] - coord_array[:-1]),
                           lambda: tf.constant([0.0], dtype=tf.float32))
        min_spacing = tf.cond(dim_int32 > 1, lambda: tf.reduce_min(spacings), lambda: tf_zero)
        max_spacing = tf.cond(dim_int32 > 1, lambda: tf.reduce_max(spacings), lambda: tf_zero)
        avg_spacing = tf.cond(dim_int32 > 1, lambda: tf.reduce_mean(spacings), lambda: tf_zero)
        return min_spacing, max_spacing, avg_spacing

    min_dx, max_dx, avg_dx = compute_spacing_stats(x, nx)
    min_dy, max_dy, avg_dy = compute_spacing_stats(y, ny)
    min_dz, max_dz, avg_dz = compute_spacing_stats(z, nz)

    # --- Log Spacing Near Collector ---
    collector_z_coord = d_tf
    idx_closest_z = tf.argmin(tf.abs(z - collector_z_coord))
    idx_closest_z_int32 = tf.cast(idx_closest_z, tf.int32)
    is_valid_z = tf.logical_and(idx_closest_z_int32 > 0, idx_closest_z_int32 < nz - 1)
    spacing_at_collector_z = tf.where(
        is_valid_z,
        (z[idx_closest_z_int32 + 1] - z[idx_closest_z_int32 - 1]) / 2.0,
        tf.constant(0.0, dtype=tf.float32)
    )
    conditional_tf_print(DEBUG_FLAG,"Spacing near collector (Z=d):", spacing_at_collector_z)

    # --- Diagnostic Logging ---
    conditional_tf_print(DEBUG_FLAG,"Adjusted min spacings: dx=", min_dx, "dy=", min_dy, "dz=", min_dz)
    tf.debugging.assert_greater(min_dx, 1e-9, message="Minimum dx too small after adjustment")
    tf.debugging.assert_greater(min_dy, 1e-9, message="Minimum dy too small after adjustment")
    tf.debugging.assert_greater(min_dz, 1e-9, message="Minimum dz too small after adjustment")

    conditional_tf_print(DEBUG_FLAG,"--- TensorFlow Stretched Grid Setup ---")
    conditional_tf_print(DEBUG_FLAG,"Final Grid Dimensions: nx=", nx, ", ny=", ny, ", nz=", nz, ", Total Elements=", total_elements_final)
    conditional_tf_print(DEBUG_FLAG,"X-Spacing (Non-Uniform): min dx=", min_dx, ", max dx=", max_dx, ", avg dx=", avg_dx)
    conditional_tf_print(DEBUG_FLAG,"Y-Spacing (Non-Uniform): min dy=", min_dy, ", max dy=", max_dy, ", avg dy=", avg_dy)
    conditional_tf_print(DEBUG_FLAG,"Z-Spacing (Non-Uniform): min dz=", min_dz, ", max dz=", max_dz, ", avg dz=", avg_dz)
    conditional_tf_print(DEBUG_FLAG,"Target minimum spacing: xy=<", min_spacing_target_xy, "m, z=<", min_spacing_target_z, "m")
    tf.cond(
        tf.logical_and(tf.logical_and(min_dx <= min_spacing_target_xy, min_dy <= min_spacing_target_xy), min_dz <= min_spacing_target_z),
        lambda: conditional_tf_print(DEBUG_FLAG,"INFO: Minimum spacing targets achieved."),
        lambda: conditional_tf_print(DEBUG_FLAG,"WARNING: Minimum spacing targets not achieved in all dimensions. Check min_dx/dy/dz logs vs targets.")
    )

    conditional_tf_print(DEBUG_FLAG,"Domain: X=[", tf.reduce_min(x), ",", tf.reduce_max(x),
             "], Y=[", tf.reduce_min(y), ",", tf.reduce_max(y),
             "], Z=[", tf.reduce_min(z), ",", tf.reduce_max(z), "]")

    # --- Memory Estimation ---
    bytes_per_float = 4.0
    gb_divisor = 1e9
    array_mem_1d = (tf.cast(nx + ny + nz, dtype=tf.float32) * bytes_per_float) / gb_divisor
    potential_3d_mem_gb = tf.cast(nx * ny * nz, dtype=tf.float32) * bytes_per_float * 10.0 / gb_divisor
    total_mem_gb_est = array_mem_1d + potential_3d_mem_gb
    conditional_tf_print(DEBUG_FLAG,"Memory (Estimate): 1D Coords=", array_mem_1d, "GB, Potential 10 3D Arrays=", potential_3d_mem_gb, "GB")
    conditional_tf_print(DEBUG_FLAG,"Memory (Estimate): Total =", total_mem_gb_est, "GB")

    return x, y, z, nx, ny, nz

@tf.function
def red_black_gauss_seidel_smoother_tf(phi_init, f, level_data, boundary_mask, max_iter=10, omega=1.0):
    """
    Red-Black Gauss-Seidel smoother for the Poisson equation on a non-uniform grid.
    
    Args:
        phi_init (tf.Tensor): Initial potential [nx, ny, nz], dtype=tf.float32.
        f (tf.Tensor): Right-hand side (-rho/epsilon_0) [nx, ny, nz], dtype=tf.float32.
        level_data (dict): Grid data with 'x', 'y', 'z', 'nx', 'ny', 'nz', 'coeffs'.
        boundary_mask (tf.Tensor): Boolean mask for fixed boundary nodes [nx, ny, nz].
        max_iter (int): Number of RBGS iterations (default 10).
        omega (float): Relaxation factor (default 1.0).
    
    Returns:
        tf.Tensor: Smoothed potential [nx, ny, nz], dtype=tf.float32.
    """
    nx, ny, nz = level_data['nx'], level_data['ny'], level_data['nz']
    if nx <= 2 or ny <= 2 or nz <= 2:
        return phi_init
    
    phi = tf.cast(phi_init, tf.float32)
    f = tf.cast(f, tf.float32)
    
    # Extract Laplacian coefficients
    coeff_x_ip1, coeff_x_im1, coeff_x_i = level_data['coeffs'][0:3]
    coeff_y_jp1, coeff_y_jm1, coeff_y_j = level_data['coeffs'][3:6]
    coeff_z_kp1, coeff_z_km1, coeff_z_k = level_data['coeffs'][6:9]
    
    # Compute diagonal coefficient
    diagonal = coeff_x_i[:, None, None] + coeff_y_j[None, :, None] + coeff_z_k[None, None, :]
    
    # Create red-black masks (checkerboard pattern)
    indices = tf.stack(tf.meshgrid(tf.range(nx), tf.range(ny), tf.range(nz), indexing='ij'), axis=-1)
    red_mask = tf.reduce_sum(tf.cast(indices, tf.int32), axis=-1) % 2 == 0
    black_mask = tf.logical_not(red_mask)
    
    # Ensure boundary nodes are excluded from updates
    red_mask = tf.logical_and(red_mask, tf.logical_not(boundary_mask))
    black_mask = tf.logical_and(black_mask, tf.logical_not(boundary_mask))
    
    V_tf = tf.constant(20000.0, dtype=tf.float32)
    
    def update_color(phi_current, color_mask):
        """Update one color (red or black) in parallel."""
        phi_clipped = tf.clip_by_value(phi_current, -V_tf, V_tf)
        laplacian = compute_laplacian_tf(
            phi_clipped, level_data['x'], level_data['y'], level_data['z'], *level_data['coeffs']
        )
        residual = f - laplacian
        # Extract interior residual to match diagonal's shape
        residual_interior = residual[1:-1, 1:-1, 1:-1]
        # Compute update for interior points
        update_interior = omega * residual_interior / (diagonal + tf.constant(1e-6, tf.float32))
        # Pad update_interior to full grid shape with zeros
        update_full = tf.pad(update_interior, [[1,1],[1,1],[1,1]], mode='CONSTANT', constant_values=0.0)
        # Apply update only where color_mask is True
        phi_new = phi_current + tf.where(color_mask, update_full, tf.zeros_like(phi_current))
        return tf.where(boundary_mask, phi_init, phi_new)
    
    # Iteration loop
    for _ in range(max_iter):
        # Update red points
        phi = update_color(phi, red_mask)
        tf.debugging.check_numerics(phi, "Phi after red update contains NaN/Inf")
        
        # Update black points
        phi = update_color(phi, black_mask)
        tf.debugging.check_numerics(phi, "Phi after black update contains NaN/Inf")
    
    # Final boundary enforcement
    phi = tf.where(boundary_mask, phi_init, phi)
    conditional_tf_print(DEBUG_FLAG,"RBGS Smoother: Max |phi| =", tf.reduce_max(tf.abs(phi)))
    
    return phi

@tf.function
def define_electrodes_tf(
    X, Y, Z, r_e, r_c, d_gap, l, V,
    x_1d_tf, y_1d_tf, z_1d_tf,
    shape_emitter='cylindrical', x_offset=0.0 # Added x_offset parameter back, though it's overwritten later
):
    """
    Defines electrode boundaries and potential based on geometry using TensorFlow.

    **Modifications (based on Step 2):**
    - Relaxed tolerance calculations significantly to better capture near-emitter points.
    - Added detailed diagnostics for tolerances and mask counts (initial and fallback).
    - Kept previous fixes (casting, shape enforcement, fallback logic improvements).

    Args:
        X, Y, Z (tf.Tensor): 3D coordinate tensors [nx, ny, nz], dtype=tf.float32.
        r_e, r_c, d_gap, l, V (float or tf.Tensor): Emitter radius, collector radius, gap distance, emitter length, voltage.
        x_1d_tf, y_1d_tf, z_1d_tf (tf.Tensor): 1D coordinate arrays [nx], [ny], [nz], dtype=tf.float32.
        shape_emitter (str): Shape of the emitter ('cylindrical', 'pointed', 'hexagonal').
        x_offset (float): Lateral offset of collector (default 0.0 - currently hardcoded below).

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            - phi: Potential field [nx, ny, nz], dtype=tf.float32.
            - boundary_mask: Combined boundary mask [nx, ny, nz], dtype=tf.bool.
            - collector_mask: Collector mask [nx, ny, nz], dtype=tf.bool.
            - error_code: Status code (0=success, 3=NaN/Inf), dtype=tf.int32.
    """
    conditional_tf_print(DEBUG_FLAG,"Starting define_electrodes_tf (Robust Emitter Mask with Extended Radius Fallback & Relaxed Tolerance)...") # Added Relaxed Tolerance to message
    conditional_tf_print(DEBUG_FLAG,"Input shapes: X=", tf.shape(X), "Y=", tf.shape(Y), "Z=", tf.shape(Z))
    conditional_tf_print(DEBUG_FLAG,"Parameters: r_e=", r_e, "r_c=", r_c, "d=", d_gap, "l=", l, "V=", V,
             "shape=", shape_emitter)

    # **Explicit Casting and Shape Enforcement**
    r_e_tf = tf.ensure_shape(tf.cast(r_e, dtype=tf.float32), [])
    r_c_tf = tf.ensure_shape(tf.cast(r_c, dtype=tf.float32), [])
    d_gap_tf = tf.ensure_shape(tf.cast(d_gap, dtype=tf.float32), [])
    l_tf = tf.ensure_shape(tf.cast(l, dtype=tf.float32), [])
    V_tf = tf.ensure_shape(tf.cast(V, dtype=tf.float32), [])
    # NOTE: x_offset is currently hardcoded below, overriding the input parameter
    # x_offset_tf = tf.constant(x_offset, dtype=tf.float32) # Use input parameter if needed
    x_offset_tf = tf.constant(0.005, dtype=tf.float32) # Hardcoded value from previous version
    conditional_tf_print(DEBUG_FLAG,"Hardcoded collector x_offset=", x_offset_tf)

    X_tf = tf.cast(X, dtype=tf.float32)
    Y_tf = tf.cast(Y, dtype=tf.float32)
    Z_tf = tf.cast(Z, dtype=tf.float32)
    x_1d = tf.cast(x_1d_tf, dtype=tf.float32)
    y_1d = tf.cast(y_1d_tf, dtype=tf.float32)
    z_1d = tf.cast(z_1d_tf, dtype=tf.float32)
    conditional_tf_print(DEBUG_FLAG,"1D array shapes: x_1d=", tf.shape(x_1d), "y_1d=", tf.shape(y_1d), "z_1d=", tf.shape(z_1d))

    # **Constants**
    tf_zero = tf.constant(0.0, dtype=tf.float32)
    tf_half = tf.constant(0.5, dtype=tf.float32)
    tf_two = tf.constant(2.0, dtype=tf.float32)
    tf_sqrt3_2 = tf.constant(np.sqrt(3.0)/2.0, dtype=tf.float32)
    # TF_EPSILON_MIN_SPACING = tf.constant(1e-9, dtype=tf.float32) # Moved to global scope for clarity

    # **Tolerance Calculation**
    conditional_tf_print(DEBUG_FLAG,"Calculating tolerances based on emitter radius and grid spacings...")
    min_dx = tf.cond(tf.shape(x_1d)[0] > 1,
                     lambda: tf.reduce_min(tf.abs(x_1d[1:] - x_1d[:-1])),
                     lambda: tf.constant(1.0, dtype=tf.float32)) # Default large value if only 1 point
    min_dy = tf.cond(tf.shape(y_1d)[0] > 1,
                     lambda: tf.reduce_min(tf.abs(y_1d[1:] - y_1d[:-1])),
                     lambda: tf.constant(1.0, dtype=tf.float32)) # Default large value if only 1 point
    min_dz = tf.cond(tf.shape(z_1d)[0] > 1,
                     lambda: tf.reduce_min(tf.abs(z_1d[1:] - z_1d[:-1])),
                     lambda: tf.constant(1.0, dtype=tf.float32)) # Default large value if only 1 point

    # Ensure minimums are positive
    min_spacing_xy = tf.maximum(tf.minimum(min_dx, min_dy), TF_EPSILON_MIN_SPACING)
    min_spacing_z = tf.maximum(min_dz, TF_EPSILON_MIN_SPACING)

    # Calculate average dz
    avg_dz = tf.cond(tf.shape(z_1d)[0] > 1,
                     lambda: tf.reduce_mean(tf.abs(z_1d[1:] - z_1d[:-1])),
                     lambda: tf.constant(0.0, dtype=tf.float32)) # Default 0 if only 1 point

    # -------- MODIFICATION START (Step 2: Relax Tolerance) --------
    # Relax tolerances significantly to better capture near-emitter points
    # Base the tolerance more on the emitter radius itself, ensuring it's
    # at least a few times the minimum grid spacing found.
    xy_tolerance = tf.maximum(min_spacing_xy * 3.0, r_e_tf * 0.2) # Increased multiplier and percentage of r_e
    z_tolerance_tf = tf.maximum(min_spacing_z * 3.0, avg_dz * 1.0) # Increased multiplier and using avg_dz
    conditional_tf_print(DEBUG_FLAG,"DEBUG define_electrodes: Min spacings (dx, dy, dz):", min_dx, min_dy, min_dz) # Added DEBUG
    conditional_tf_print(DEBUG_FLAG,"DEBUG define_electrodes: Avg dz:", avg_dz) # Added DEBUG
    conditional_tf_print(DEBUG_FLAG,"RELAXED tolerances: xy_tolerance=", xy_tolerance, "z_tolerance_tf=", z_tolerance_tf) # Changed ADJUSTED to RELAXED
    # -------- MODIFICATION END --------


    # **Emitter Mask Calculation Utilities** (Moved into main scope for clarity before use)
    z_min_emitter = -l_tf / tf_two
    z_max_emitter = l_tf / tf_two
    R_sq_emitter = tf.square(X_tf) + tf.square(Y_tf) # Calculate once

    # **Emitter Shape Logic**
    conditional_tf_print(DEBUG_FLAG,"Determining emitter shape logic...")
    shape_emitter_tf_str = tf.strings.lower(tf.convert_to_tensor(shape_emitter, dtype=tf.string))
    conditional_tf_print(DEBUG_FLAG,"Shape emitter normalized:", shape_emitter_tf_str)

    def cylindrical_case(xy_tol, z_tol): # Pass z_tol too
        conditional_tf_print(DEBUG_FLAG,"Processing cylindrical emitter shape with xy_tol=", xy_tol, "z_tol=", z_tol)
        emitter_z_mask = tf.logical_and(
            tf.greater_equal(Z_tf, z_min_emitter - z_tol),
            tf.less_equal(Z_tf, z_max_emitter + z_tol)
        )
        r_squared_threshold = tf.square(r_e_tf + xy_tol)
        emitter_xy_mask = tf.less_equal(R_sq_emitter, r_squared_threshold)
        return tf.logical_and(emitter_xy_mask, emitter_z_mask)

    def pointed_case(xy_tol, z_tol): # Pass z_tol too
        conditional_tf_print(DEBUG_FLAG,"Processing pointed emitter shape with xy_tol=", xy_tol, "z_tol=", z_tol)
        emitter_z_mask = tf.logical_and(
            tf.greater_equal(Z_tf, z_min_emitter - z_tol),
            tf.less_equal(Z_tf, z_max_emitter + z_tol)
        )
        r_squared_threshold = tf.square(r_e_tf + xy_tol)
        emitter_xy_mask = tf.less_equal(R_sq_emitter, r_squared_threshold)
        conditional_tf_print(DEBUG_FLAG," INFO: 'pointed' emitter shape uses a cylindrical mask for BCs.")
        return tf.logical_and(emitter_xy_mask, emitter_z_mask)

    def hexagonal_case(xy_tol, z_tol): # Pass z_tol too
        conditional_tf_print(DEBUG_FLAG,"Processing hexagonal emitter shape with xy_tol=", xy_tol, "z_tol=", z_tol)
        emitter_z_mask = tf.logical_and(
            tf.greater_equal(Z_tf, z_min_emitter - z_tol),
            tf.less_equal(Z_tf, z_max_emitter + z_tol)
        )
        r_e_buffered = r_e_tf + xy_tol
        hex_mask_1 = tf.less_equal(tf.abs(X_tf), r_e_buffered)
        hex_mask_2 = tf.less_equal(tf.abs(tf_sqrt3_2 * Y_tf + tf_half * X_tf), r_e_buffered)
        hex_mask_3 = tf.less_equal(tf.abs(tf_sqrt3_2 * Y_tf - tf_half * X_tf), r_e_buffered)
        emitter_xy_mask = tf.logical_and(tf.logical_and(hex_mask_1, hex_mask_2), hex_mask_3)
        return tf.logical_and(emitter_xy_mask, emitter_z_mask)

    def unknown_shape_case(xy_tol, z_tol): # Pass z_tol too
        conditional_tf_print(DEBUG_FLAG,"Unknown emitter shape detected: '", shape_emitter_tf_str, "'")
        message_tensor = tf.strings.format("Unknown emitter shape: '{}'", shape_emitter_tf_str)
        tf.debugging.Assert(tf.constant(False), [message_tensor])
        # Return a mask of the correct shape but all False
        return tf.zeros_like(X_tf, dtype=tf.bool)

    valid_shapes = [b'cylindrical', b'pointed', b'hexagonal']
    shape_indices = [tf.cast(tf.equal(shape_emitter_tf_str, shape), tf.int32) * i
                     for i, shape in enumerate(valid_shapes)]
    shape_index = tf.reduce_sum(shape_indices)
    is_known_shape = tf.reduce_any([tf.equal(shape_emitter_tf_str, shape) for shape in valid_shapes])
    # Use index 3 (out of bounds for branch_fns) to trigger default if shape is unknown
    shape_index = tf.cond(is_known_shape, lambda: shape_index, lambda: tf.constant(len(valid_shapes), dtype=tf.int32))

    # **Compute Initial Emitter Mask**
    conditional_tf_print(DEBUG_FLAG,"Calculating emitter mask...")
    def compute_emitter_mask(xy_tol, z_tol):
        branch_fns = {
            0: lambda: cylindrical_case(xy_tol, z_tol),
            1: lambda: pointed_case(xy_tol, z_tol),
            2: lambda: hexagonal_case(xy_tol, z_tol)
        }
        mask = tf.switch_case(
            shape_index,
            branch_fns=branch_fns,
            # Pass dummy tolerances to default, it won't use them anyway
            default=lambda: unknown_shape_case(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        )
        node_count = tf.reduce_sum(tf.cast(mask, tf.int32))
        # Don't print here, print *after* calling compute_emitter_mask before fallback logic
        # conditional_tf_print(DEBUG_FLAG,"Emitter mask nodes:", node_count)
        return mask, node_count

    emitter_mask, emitter_node_count = compute_emitter_mask(xy_tolerance, z_tolerance_tf)
    # --- ADD DIAGNOSTICS (Step 2) ---
    conditional_tf_print(DEBUG_FLAG,"DEBUG define_electrodes: Initial emitter mask nodes (before fallback):", emitter_node_count) # Added DEBUG


    # **Fallback Logic with Diagnostics**
    def nearest_points_fallback():
        conditional_tf_print(DEBUG_FLAG,"WARNING: Emitter mask empty. Entering extended radius fallback...")
        # Define a fallback radius (e.g., 5 * r_e)
        fallback_radius = r_e_tf * 5.0 # Consider making this larger if needed
        fallback_r_sq_threshold = tf.square(fallback_radius)
        fallback_xy_mask = tf.less_equal(R_sq_emitter, fallback_r_sq_threshold)

        # Fallback z-tolerance (e.g., 4 * z_tolerance_tf or 10% of emitter length)
        # Make sure z_tolerance_tf has a minimal value if avg_dz was 0 (e.g., single z-layer)
        effective_z_tol = tf.maximum(z_tolerance_tf, TF_EPSILON_MIN_SPACING * 10.0) # Use a small value if z_tolerance_tf is zero
        fallback_z_tolerance = tf.maximum(effective_z_tol * 4.0, l_tf * 0.1) # Adjusted logic for z_tolerance_tf possibly being zero
        fallback_z_mask = tf.logical_and(
            tf.greater_equal(Z_tf, z_min_emitter - fallback_z_tolerance),
            tf.less_equal(Z_tf, z_max_emitter + fallback_z_tolerance)
        )
        fallback_mask = tf.logical_and(fallback_xy_mask, fallback_z_mask)
        node_count = tf.reduce_sum(tf.cast(fallback_mask, tf.int32))

        # --- ADD DIAGNOSTICS (Step 2) ---
        # Detailed Diagnostics
        xy_count_fb = tf.reduce_sum(tf.cast(fallback_xy_mask, tf.int32))
        z_count_fb = tf.reduce_sum(tf.cast(fallback_z_mask, tf.int32))
        conditional_tf_print(DEBUG_FLAG,"Fallback Diagnostics: Radius=", fallback_radius, "Z Tol=", fallback_z_tolerance)
        conditional_tf_print(DEBUG_FLAG,"Fallback Counts: XY Mask=", xy_count_fb, "Z Mask=", z_count_fb, "Combined=", node_count)

        # Check grid points near emitter
        distances = tf.sqrt(R_sq_emitter + tf.square(Z_tf)) # R_sq_emitter = X^2+Y^2
        min_distance = tf.reduce_min(distances)
        conditional_tf_print(DEBUG_FLAG,"DEBUG define_electrodes: Minimum distance from grid to emitter center (in fallback):", min_distance) # Added DEBUG label
        # --- END DIAGNOSTICS (Step 2) ---


        # Conditional logic for using multi-point or single-point fallback
        def use_multi_points():
            conditional_tf_print(DEBUG_FLAG,"Using multiple points from extended radius fallback mask:", node_count)
            return fallback_mask, node_count

        def use_single_point():
            conditional_tf_print(DEBUG_FLAG,"ERROR: Extended radius fallback mask STILL empty. Resorting to single nearest point (likely indicates grid issue near emitter).")
            # Calculate distance squared to origin (0,0,0) - assuming emitter centered there
            R_sq_center = R_sq_emitter + tf.square(Z_tf) # Use R_sq_emitter calculated earlier
            # Find the index of the minimum distance squared
            min_idx_flat = tf.argmin(tf.reshape(R_sq_center, [-1]))
            min_idx_flat = tf.cast(min_idx_flat, tf.int64) # Cast index to int64 for unravel_index
            # Get the shape as int64
            shape_int64 = tf.cast(tf.shape(X_tf), tf.int64)
            # Convert the flat index to 3D coordinates
            min_idx_3d = tf.unravel_index(min_idx_flat, shape_int64)
            conditional_tf_print(DEBUG_FLAG,"Nearest point index (fallback):", min_idx_3d)

            # Create a mask with only this single point set to True
            # Need to reshape the index for scatter_nd
            single_point_mask = tf.scatter_nd(
                indices=tf.expand_dims(min_idx_3d, axis=0), # Shape [1, 3]
                updates=[True],                            # Value to place
                shape=tf.shape(X_tf, out_type=tf.int64)    # Shape of the output tensor
            )
            return single_point_mask, tf.constant(1, dtype=tf.int32)

        final_mask, final_count = tf.cond(
            tf.greater(node_count, 0),
            use_multi_points,
            use_single_point
        )
        return final_mask, final_count

    # Decide whether to use the initial mask or the fallback result
    final_emitter_mask, final_node_count = tf.cond(
        tf.equal(emitter_node_count, 0),
        nearest_points_fallback, # Call the fallback function
        lambda: (emitter_mask, emitter_node_count) # Return the original mask and count
    )

    conditional_tf_print(DEBUG_FLAG,"Final emitter mask: nodes=", final_node_count)
    # Add an assertion to ensure at least one emitter node is found
    tf.debugging.assert_positive(
        final_node_count,
        message="Emitter mask is empty even after fallback. Check grid resolution near emitter or tolerance logic."
    )


    # **Collector Mask**
    conditional_tf_print(DEBUG_FLAG,"Calculating collector mask...")
    nx_tf, ny_tf, nz_tf = tf.shape(X_tf)[0], tf.shape(X_tf)[1], tf.shape(X_tf)[2]
    conditional_tf_print(DEBUG_FLAG,"Grid dimensions (nx, ny, nz):", nx_tf, ny_tf, nz_tf)

    # Find the z-index closest to the collector position d_gap_tf
    diff_z = tf.abs(z_1d - d_gap_tf)
    k_closest = tf.argmin(diff_z)
    conditional_tf_print(DEBUG_FLAG,"Closest z index to d_gap:", k_closest, "z[k_closest]:", z_1d[k_closest], "d_gap_tf:", d_gap_tf)

    # Initial collector mask based on the closest z-plane and XY radius
    # Create a 3D boolean mask where only the k_closest slice along z is True
    collector_z_mask_init = tf.equal(tf.range(nz_tf, dtype=tf.int32)[None, None, :], tf.cast(k_closest, tf.int32))
    collector_z_mask_init = tf.broadcast_to(collector_z_mask_init, tf.shape(X_tf)) # Ensure full 3D shape

    # Calculate squared distance in XY plane from collector center (x_offset_tf, 0)
    R_sq_collector = tf.square(X_tf - x_offset_tf) + tf.square(Y_tf)

    # Tolerance for collector XY boundary (more generous than emitter initially)
    xy_tolerance_collector = tf.maximum(min_spacing_xy * 2.0, r_c_tf * 0.1) # Base tolerance
    conditional_tf_print(DEBUG_FLAG,"Collector initial xy_tolerance=", xy_tolerance_collector)
    r_squared_threshold_collector = tf.square(r_c_tf + xy_tolerance_collector)
    collector_xy_mask_init = tf.less_equal(R_sq_collector, r_squared_threshold_collector)

    collector_mask_init = tf.logical_and(collector_xy_mask_init, collector_z_mask_init)

    collector_node_count_init = tf.reduce_sum(tf.cast(collector_mask_init, tf.int32))
    conditional_tf_print(DEBUG_FLAG,"Initial collector nodes:", collector_node_count_init)

    # Fallback/Adjustment logic for collector mask if initial is empty
    def adjust_collector_mask():
        conditional_tf_print(DEBUG_FLAG,"WARNING: Initial collector mask empty. Adjusting collector mask logic...")
        # Option 1: Expand Z range slightly
        k_min = tf.maximum(tf.cast(k_closest, tf.int32) - 1, 0)
        k_max = tf.minimum(tf.cast(k_closest, tf.int32) + 1, nz_tf - 1)
        collector_z_mask_adj = tf.logical_and(
            tf.range(nz_tf, dtype=tf.int32)[None, None, :] >= k_min,
            tf.range(nz_tf, dtype=tf.int32)[None, None, :] <= k_max
        )
        collector_z_mask_adj = tf.broadcast_to(collector_z_mask_adj, tf.shape(X_tf)) # Ensure full 3D shape
        conditional_tf_print(DEBUG_FLAG,"Adjusted collector z-range indices:", k_min, "to", k_max)

        # Option 2: Increase XY tolerance significantly
        xy_tolerance_adj = tf.maximum(min_spacing_xy * 8.0, r_c_tf * 0.5) # Much larger tolerance
        conditional_tf_print(DEBUG_FLAG,"Adjusted collector xy_tolerance=", xy_tolerance_adj)
        r_squared_threshold_adj = tf.square(r_c_tf + xy_tolerance_adj)
        collector_xy_mask_adj = tf.less_equal(R_sq_collector, r_squared_threshold_adj)

        # Combine adjusted Z and XY masks
        collector_mask_adj = tf.logical_and(collector_xy_mask_adj, collector_z_mask_adj)
        adj_node_count = tf.reduce_sum(tf.cast(collector_mask_adj, tf.int32))
        conditional_tf_print(DEBUG_FLAG,"Adjusted collector nodes:", adj_node_count)
        return collector_mask_adj

    # Use initial mask if it has nodes, otherwise use the adjusted mask
    collector_mask = tf.cond(
        tf.equal(collector_node_count_init, 0),
        adjust_collector_mask,
        lambda: collector_mask_init
    )
    final_collector_node_count = tf.reduce_sum(tf.cast(collector_mask, tf.int32))
    conditional_tf_print(DEBUG_FLAG,"Final collector mask: nodes=", final_collector_node_count)
    # Assert that the collector mask is not empty after adjustment
    tf.debugging.assert_positive(
        final_collector_node_count,
        message="Collector mask empty even after adjustment. Check grid resolution near collector or d_gap value."
    )


    # **Potential Array**
    conditional_tf_print(DEBUG_FLAG,"Initializing potential array...")
    phi = tf.zeros_like(X_tf, dtype=tf.float32)
    # Apply collector potential (0 V)
    phi = tf.where(collector_mask, tf_zero, phi)
    # Apply emitter potential (V_tf) - use the FINAL emitter mask
    phi = tf.where(final_emitter_mask, V_tf, phi)
    conditional_tf_print(DEBUG_FLAG,"Boundary conditions applied: phi shape=", tf.shape(phi))
    conditional_tf_print(DEBUG_FLAG,"Phi max:", tf.reduce_max(phi), "min:", tf.reduce_min(phi))


    # **Combined Boundary Mask**
    # Combine the FINAL emitter mask and the final collector mask
    boundary_mask = tf.logical_or(final_emitter_mask, collector_mask)
    conditional_tf_print(DEBUG_FLAG,"Combined boundary mask computed: shape=", tf.shape(boundary_mask),
             "Total boundary nodes:", tf.reduce_sum(tf.cast(boundary_mask, tf.int32)))


    # **Error Checking**
    phi_is_finite = tf.reduce_all(tf.math.is_finite(phi))
    error_code = tf.cond(phi_is_finite,
                         lambda: tf.constant(0, dtype=tf.int32), # 0 = success
                         lambda: tf.constant(3, dtype=tf.int32)) # 3 = NaN/Inf error

    tf.cond(tf.equal(error_code, 3),
            lambda: conditional_tf_print(DEBUG_FLAG," ERROR: NaN or Inf detected in the final potential field 'phi'."),
            lambda: tf.no_op()) # No operation if finite
    conditional_tf_print(DEBUG_FLAG,"Returning results from define_electrodes_tf. Final error_code=", error_code)
    return phi, boundary_mask, collector_mask, error_code

@tf.function
def setup_multigrid_hierarchy_tf(x, y, z, nx, ny, nz):
    """Sets up a multigrid hierarchy with padded coordinate and coefficient arrays."""
    max_levels = tf.constant(10, dtype=tf.int32)
    initial_nx = nx
    initial_ny = ny
    initial_nz = nz

    # Compute max_size for padding coefficients based on the finest grid level
    max_size = 3 * (initial_nx - 2) + 3 * (initial_ny - 2) + 3 * (initial_nz - 2)

    # Initialize TensorArrays
    x_levels = tf.TensorArray(tf.float32, size=max_levels, dynamic_size=False, clear_after_read=False, infer_shape=False)
    y_levels = tf.TensorArray(tf.float32, size=max_levels, dynamic_size=False, clear_after_read=False, infer_shape=False)
    z_levels = tf.TensorArray(tf.float32, size=max_levels, dynamic_size=False, clear_after_read=False, infer_shape=False)
    nx_levels = tf.TensorArray(tf.int32, size=max_levels, dynamic_size=False, clear_after_read=False, element_shape=[])
    ny_levels = tf.TensorArray(tf.int32, size=max_levels, dynamic_size=False, clear_after_read=False, element_shape=[])
    nz_levels = tf.TensorArray(tf.int32, size=max_levels, dynamic_size=False, clear_after_read=False, element_shape=[])
    coeffs_levels = tf.TensorArray(tf.float32, size=max_levels, dynamic_size=False, clear_after_read=False, element_shape=[None])

    # Adjust initial grid sizes
    nx = tf.cast(tf.maximum(nx, 4), tf.int32)
    ny = tf.cast(tf.maximum(ny, 4), tf.int32)
    nz = tf.cast(tf.maximum(nz, 4), tf.int32)
    nx = tf.where(tf.math.mod(nx, 2) != 0, nx - 1, nx)
    ny = tf.where(tf.math.mod(ny, 2) != 0, ny - 1, ny)
    nz = tf.where(tf.math.mod(nz, 2) != 0, nz - 1, nz)
    conditional_tf_print(DEBUG_FLAG,"Initial grid sizes (adjusted): nx=", nx, "ny=", ny, "nz=", nz)
    conditional_tf_print(DEBUG_FLAG,"Initial coordinate shapes: x=", tf.shape(x), "y=", tf.shape(y), "z=", tf.shape(z))

    # Initialize coordinates
    cx = tf.cast(x[:nx], tf.float32)
    cy = tf.cast(y[:ny], tf.float32)
    cz = tf.cast(z[:nz], tf.float32)
    cnx = nx
    cny = ny
    cnz = nz
    level = tf.constant(0, dtype=tf.int32)

    def cond(level, cnx, cny, cnz, cx, cy, cz, x_arr, y_arr, z_arr, nx_arr, ny_arr, nz_arr, coeffs_arr):
        return tf.logical_and(
            tf.logical_and(cnx >= 4, cny >= 4),
            tf.logical_and(cnz >= 4, level < max_levels)
        )

    def body(level, cnx, cny, cnz, cx, cy, cz, x_arr, y_arr, z_arr, nx_arr, ny_arr, nz_arr, coeffs_arr):
        conditional_tf_print(DEBUG_FLAG,"Level", level, ": Before write - cnx=", cnx, "x_shape=", tf.shape(cx))
        coeffs = precompute_laplacian_coefficients(cx, cy, cz)

        # Concatenate coefficients
        coeffs_concat = tf.concat([
            coeffs[0], coeffs[1], coeffs[2],  # x-direction coefficients
            coeffs[3], coeffs[4], coeffs[5],  # y-direction coefficients
            coeffs[6], coeffs[7], coeffs[8]   # z-direction coefficients
        ], axis=0)

        # Pad coeffs_concat to max_size
        current_size = tf.shape(coeffs_concat)[0]
        pad_size = tf.maximum(max_size - current_size, 0)  # Ensure non-negative padding
        coeffs_concat_padded = tf.pad(coeffs_concat, [[0, pad_size]], mode='CONSTANT', constant_values=0.0)

        # Pad coordinates to initial sizes
        cx_padded = tf.pad(cx, [[0, initial_nx - cnx]], mode='CONSTANT', constant_values=0.0)
        cy_padded = tf.pad(cy, [[0, initial_ny - cny]], mode='CONSTANT', constant_values=0.0)
        cz_padded = tf.pad(cz, [[0, initial_nz - cnz]], mode='CONSTANT', constant_values=0.0)

        # Write padded arrays to TensorArrays
        x_arr = x_arr.write(level, cx_padded)
        y_arr = y_arr.write(level, cy_padded)
        z_arr = z_arr.write(level, cz_padded)
        nx_arr = nx_arr.write(level, cnx)
        ny_arr = ny_arr.write(level, cny)
        nz_arr = nz_arr.write(level, cnz)
        coeffs_arr = coeffs_arr.write(level, coeffs_concat_padded)
        conditional_tf_print(DEBUG_FLAG,"Level", level, ": nx=", cnx, "ny=", cny, "nz=", cnz,
                 "x_padded_shape=", tf.shape(cx_padded))

        # Coarsen grid
        cnx_new = tf.cast(tf.math.floordiv(cnx, 2), tf.int32)
        cny_new = tf.cast(tf.math.floordiv(cny, 2), tf.int32)
        cnz_new = tf.cast(tf.math.floordiv(cnz, 2), tf.int32)
        cnx_new = tf.where(tf.math.mod(cnx_new, 2) != 0, cnx_new - 1, cnx_new)
        cny_new = tf.where(tf.math.mod(cny_new, 2) != 0, cny_new - 1, cny_new)
        cnz_new = tf.where(tf.math.mod(cnz_new, 2) != 0, cnz_new - 1, cnz_new)
        cx_new = cx[:cnx_new * 2:2]
        cy_new = cy[:cny_new * 2:2]
        cz_new = cz[:cnz_new * 2:2]

        return [level + 1, cnx_new, cny_new, cnz_new, cx_new, cy_new, cz_new,
                x_arr, y_arr, z_arr, nx_arr, ny_arr, nz_arr, coeffs_arr]

    # Define loop variables and shape invariants
    loop_vars = [level, cnx, cny, cnz, cx, cy, cz,
                 x_levels, y_levels, z_levels, nx_levels, ny_levels, nz_levels, coeffs_levels]
    shape_invariants = [
        tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
        tf.TensorSpec([None], dtype=tf.float32),
        tf.TensorSpec([None], dtype=tf.float32),
        tf.TensorSpec([None], dtype=tf.float32),
        tf.TensorSpec(None, dtype=tf.float32),
        tf.TensorSpec(None, dtype=tf.float32),
        tf.TensorSpec(None, dtype=tf.float32),
        nx_levels, ny_levels, nz_levels, coeffs_levels
    ]

    # Execute while loop
    final_level, _, _, _, _, _, _, x_levels_final, y_levels_final, z_levels_final, \
    nx_levels_final, ny_levels_final, nz_levels_final, coeffs_levels_final = tf.while_loop(
        cond, body, loop_vars, shape_invariants=shape_invariants, maximum_iterations=max_levels
    )

    num_levels_tf = final_level
    conditional_tf_print(DEBUG_FLAG,"Multigrid hierarchy created with", num_levels_tf, "levels (graph mode).")
    return (x_levels_final, y_levels_final, z_levels_final,
            nx_levels_final, ny_levels_final, nz_levels_final,
            coeffs_levels_final, num_levels_tf)

# Precompute coefficients assuming static grid (outside the function)
@tf.function(reduce_retracing=True)
def precompute_laplacian_coefficients(x, y, z):
    """
    Precomputes finite difference coefficients for non-uniform grid Laplacian.

    Args:
        x (tf.Tensor): 1D tensor of x-coordinates [nx], dtype=tf.float32.
        y (tf.Tensor): 1D tensor of y-coordinates [ny], dtype=tf.float32.
        z (tf.Tensor): 1D tensor of z-coordinates [nz], dtype=tf.float32.

    Returns:
        Tuple: Coefficient tensors for x, y, z directions.
    """
    tf_2_0 = tf.constant(2.0, dtype=tf.float32)
    tf_neg_2_0 = tf.constant(-2.0, dtype=tf.float32)
    epsilon = tf.constant(1e-5, dtype=tf.float32)  # Increased for stability
    conditional_tf_print(DEBUG_FLAG,"INFO (precompute_laplacian_coefficients): Using epsilon =", epsilon)

    # X-direction coefficients
    dx = x[1:] - x[:-1]
    dx_left = dx[:-1]
    dx_right = dx[1:]
    denom_left_sum_x = dx_left * (dx_left + dx_right) + epsilon
    denom_right_sum_x = dx_right * (dx_left + dx_right) + epsilon
    denom_prod_x = dx_left * dx_right + epsilon
    tf.debugging.assert_greater(tf.abs(denom_left_sum_x), epsilon / 10.0, message="Denominator left_sum_x too small")
    tf.debugging.assert_greater(tf.abs(denom_right_sum_x), epsilon / 10.0, message="Denominator right_sum_x too small")
    tf.debugging.assert_greater(tf.abs(denom_prod_x), epsilon / 10.0, message="Denominator prod_x too small")
    coeff_x_ip1 = tf_2_0 / denom_right_sum_x
    coeff_x_im1 = tf_2_0 / denom_left_sum_x
    coeff_x_i = tf_neg_2_0 / denom_prod_x

    # Y-direction coefficients
    dy = y[1:] - y[:-1]
    dy_left = dy[:-1]
    dy_right = dy[1:]
    denom_left_sum_y = dy_left * (dy_left + dy_right) + epsilon
    denom_right_sum_y = dy_right * (dy_left + dy_right) + epsilon
    denom_prod_y = dy_left * dy_right + epsilon
    tf.debugging.assert_greater(tf.abs(denom_left_sum_y), epsilon / 10.0, message="Denominator left_sum_y too small")
    tf.debugging.assert_greater(tf.abs(denom_right_sum_y), epsilon / 10.0, message="Denominator right_sum_y too small")
    tf.debugging.assert_greater(tf.abs(denom_prod_y), epsilon / 10.0, message="Denominator prod_y too small")
    coeff_y_jp1 = tf_2_0 / denom_right_sum_y
    coeff_y_jm1 = tf_2_0 / denom_left_sum_y
    coeff_y_j = tf_neg_2_0 / denom_prod_y

    # Z-direction coefficients
    dz = z[1:] - z[:-1]
    dz_left = dz[:-1]
    dz_right = dz[1:]
    denom_left_sum_z = dz_left * (dz_left + dz_right) + epsilon
    denom_right_sum_z = dz_right * (dz_left + dz_right) + epsilon
    denom_prod_z = dz_left * dz_right + epsilon
    tf.debugging.assert_greater(tf.abs(denom_left_sum_z), epsilon / 10.0, message="Denominator left_sum_z too small")
    tf.debugging.assert_greater(tf.abs(denom_right_sum_z), epsilon / 10.0, message="Denominator right_sum_z too small")
    tf.debugging.assert_greater(tf.abs(denom_prod_z), epsilon / 10.0, message="Denominator prod_z too small")
    coeff_z_kp1 = tf_2_0 / denom_right_sum_z
    coeff_z_km1 = tf_2_0 / denom_left_sum_z
    coeff_z_k = tf_neg_2_0 / denom_prod_z

    # Check coefficients for NaN/Inf
    tf.debugging.check_numerics(coeff_x_i, "coeff_x_i contains NaN/Inf")
    tf.debugging.check_numerics(coeff_y_j, "coeff_y_j contains NaN/Inf")
    tf.debugging.check_numerics(coeff_z_k, "coeff_z_k contains NaN/Inf")

    return (coeff_x_ip1, coeff_x_im1, coeff_x_i,
            coeff_y_jp1, coeff_y_jm1, coeff_y_j,
            coeff_z_kp1, coeff_z_km1, coeff_z_k)

@tf.function
def restrict_tf(fine, level_data, next_level_data):
    """
    Restricts a fine grid tensor to the next coarser level using averaging.

    Args:
        fine (tf.Tensor): Fine grid tensor to restrict, dtype=tf.float32.
        level_data (dict): Current (fine) level's grid info with 'nx', 'ny', 'nz'.
        next_level_data (dict): Next (coarse) level's grid info with 'nx', 'ny', 'nz'.

    Returns:
        tf.Tensor: Restricted tensor on the coarse grid, dtype=tf.float32.
    """
    nx_f, ny_f, nz_f = level_data['nx'], level_data['ny'], level_data['nz']
    nx_c, ny_c, nz_c = next_level_data['nx'], next_level_data['ny'], next_level_data['nz']

    # Simple averaging over 2x2x2 blocks
    coarse = fine[::2, ::2, ::2]

    # Ensure correct shape using provided coarse grid dimensions
    coarse = coarse[:nx_c, :ny_c, :nz_c]
    return coarse

@tf.function
def prolongate_tf(coarse, level_data, prev_level_data):
    """
    Prolongates a coarse grid tensor to the finer level using linear interpolation.

    Args:
        coarse (tf.Tensor): Coarse grid tensor, dtype=tf.float32.
        level_data (dict): Current (coarse) level's grid info with 'nx', 'ny', 'nz'.
        prev_level_data (dict): Finer level's grid info with 'nx', 'ny', 'nz'.

    Returns:
        tf.Tensor: Prolongated tensor on the finer grid, dtype=tf.float32.
    """
    nx_c, ny_c, nz_c = level_data['nx'], level_data['ny'], level_data['nz']
    nx_f, ny_f, nz_f = prev_level_data['nx'], prev_level_data['ny'], prev_level_data['nz']

    # Compute even indices for the fine grid
    x_even_indices = tf.range(0, tf.minimum(nx_f, 2 * nx_c), 2, dtype=tf.int32)
    y_even_indices = tf.range(0, tf.minimum(ny_f, 2 * ny_c), 2, dtype=tf.int32)
    z_even_indices = tf.range(0, tf.minimum(nz_f, 2 * nz_c), 2, dtype=tf.int32)

    # Initialize fine grid
    fine = tf.zeros([nx_f, ny_f, nz_f], dtype=tf.float32)

    # Assign coarse values to even indices
    # Ensure coarse is sliced to match available indices
    num_x = tf.minimum(tf.size(x_even_indices), nx_c)
    num_y = tf.minimum(tf.size(y_even_indices), ny_c)
    num_z = tf.minimum(tf.size(z_even_indices), nz_c)
    coarse_sliced = coarse[:num_x, :num_y, :num_z]
    # Create meshgrid for even indices
    indices = tf.stack(tf.meshgrid(x_even_indices[:num_x], y_even_indices[:num_y], z_even_indices[:num_z], indexing='ij'), axis=-1)
    indices = tf.reshape(indices, [-1, 3])
    updates = tf.reshape(coarse_sliced, [-1])
    fine = tf.tensor_scatter_nd_update(fine, indices, updates)

    # Linear interpolation for odd indices
    # X-direction
    if nx_f > 1:
        x_odd_indices = tf.range(1, nx_f, 2, dtype=tf.int32)
        if tf.size(x_odd_indices) > 0:
            # Compute interpolated values
            fine_padded = tf.pad(fine, [[1, 1], [0, 0], [0, 0]], mode='REFLECT')
            interp_x = (fine_padded[1:-1, :, :] + fine_padded[2:, :, :]) / 2.0  # Shape [nx_f, ny_f, nz_f]
            # Create indices for odd x
            x_odd, y_idx, z_idx = tf.meshgrid(x_odd_indices, tf.range(ny_f), tf.range(nz_f), indexing='ij')
            odd_indices_x = tf.stack([x_odd, y_idx, z_idx], axis=-1)
            odd_indices_x = tf.reshape(odd_indices_x, [-1, 3])
            updates_x = tf.reshape(interp_x[1::2, :, :], [-1])
            fine = tf.tensor_scatter_nd_update(fine, odd_indices_x, updates_x)

    # Y-direction
    if ny_f > 1:
        y_odd_indices = tf.range(1, ny_f, 2, dtype=tf.int32)
        if tf.size(y_odd_indices) > 0:
            # Compute interpolated values
            fine_padded = tf.pad(fine, [[0, 0], [1, 1], [0, 0]], mode='REFLECT')
            interp_y = (fine_padded[:, 1:-1, :] + fine_padded[:, 2:, :]) / 2.0  # Shape [nx_f, ny_f, nz_f]
            # Create indices for odd y
            x_idx, y_odd, z_idx = tf.meshgrid(tf.range(nx_f), y_odd_indices, tf.range(nz_f), indexing='ij')
            odd_indices_y = tf.stack([x_idx, y_odd, z_idx], axis=-1)
            odd_indices_y = tf.reshape(odd_indices_y, [-1, 3])
            updates_y = tf.reshape(interp_y[:, 1::2, :], [-1])
            fine = tf.tensor_scatter_nd_update(fine, odd_indices_y, updates_y)

    # Z-direction
    if nz_f > 1:
        z_odd_indices = tf.range(1, nz_f, 2, dtype=tf.int32)
        if tf.size(z_odd_indices) > 0:
            # Compute interpolated values
            fine_padded = tf.pad(fine, [[0, 0], [0, 0], [1, 1]], mode='REFLECT')
            interp_z = (fine_padded[:, :, 1:-1] + fine_padded[:, :, 2:]) / 2.0  # Shape [nx_f, ny_f, nz_f]
            # Create indices for odd z
            x_idx, y_idx, z_odd = tf.meshgrid(tf.range(nx_f), tf.range(ny_f), z_odd_indices, indexing='ij')
            odd_indices_z = tf.stack([x_idx, y_idx, z_odd], axis=-1)
            odd_indices_z = tf.reshape(odd_indices_z, [-1, 3])
            updates_z = tf.reshape(interp_z[:, :, 1::2], [-1])
            fine = tf.tensor_scatter_nd_update(fine, odd_indices_z, updates_z)

    return fine

@tf.function
def cg_smoother_tf(phi_init, f, level_data, max_iter=3, tol=1e-6):
    """
    Applies Conjugate Gradient as a smoother for a given grid level.

    Args:
        phi_init (tf.Tensor): Initial guess [nx, ny, nz], dtype=tf.float32.
        f (tf.Tensor): Right-hand side [nx, ny, nz], dtype=tf.float32.
        level_data (dict): Grid level data with coords and coeffs.
        max_iter (int): Number of iterations (fixed for smoothing).
        tol (float): Tolerance (not used for fixed iterations).

    Returns:
        tf.Tensor: Smoothed solution [nx, ny, nz], dtype=tf.float32.
    """
    x = tf.cast(phi_init, tf.float32)
    r = f - compute_laplacian_tf(
        x, level_data['x'], level_data['y'], level_data['z'], *level_data['coeffs']
    )
    p = tf.identity(r)
    rsold = tf.reduce_sum(r * r)

    def cg_body(i, x, r, p, rsold):
        tf.debugging.check_numerics(p, "Input p to cg_body contains NaN/Inf")
        p_clipped = tf.clip_by_value(p, -1e10, 1e10)  # Prevent extreme values
        conditional_tf_print(DEBUG_FLAG,"CG Iter", i, "Max |p|:", tf.reduce_max(tf.abs(p_clipped)), output_stream=_output_stream)

        Ap = compute_laplacian_tf(p_clipped, level_data['x'], level_data['y'], level_data['z'], *level_data['coeffs'])
        tf.debugging.check_numerics(Ap, "Ap (Laplacian of p) in cg_body contains NaN/Inf")

        pAp_sum = tf.reduce_sum(p_clipped * Ap)
        alpha_denom = pAp_sum + tf.constant(1e-12, dtype=tf.float32)  # Increased epsilon
        tf.debugging.assert_greater(tf.abs(alpha_denom), 1e-15, message="CG alpha denominator too small")
        alpha = rsold / alpha_denom
        tf.debugging.check_numerics(alpha, "alpha in cg_body contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG,"CG Iter", i, "alpha:", alpha, output_stream=_output_stream)

        x_new = x + alpha * p_clipped
        r_new = r - alpha * Ap
        rsnew = tf.reduce_sum(r_new * r_new)

        beta_denom = rsold + tf.constant(1e-20, dtype=tf.float32)  # Increased epsilon
        tf.debugging.assert_greater(tf.abs(beta_denom), 1e-25, message="CG beta denominator too small")
        beta = rsnew / beta_denom
        tf.debugging.check_numerics(beta, "beta in cg_body contains NaN/Inf")

        p_new = r_new + beta * p_clipped

        tf.debugging.check_numerics(x_new, "x_new in cg_body contains NaN/Inf")
        tf.debugging.check_numerics(r_new, "r_new in cg_body contains NaN/Inf")
        tf.debugging.check_numerics(p_new, "p_new in cg_body contains NaN/Inf")

        return [i + 1, x_new, r_new, p_new, rsnew]

    def cg_cond(i, x, r, p, rsold):
        return i < max_iter

    _, x_final, _, _, _ = tf.while_loop(
        cg_cond,
        cg_body,
        [tf.constant(0, tf.int32), x, r, p, rsold],
        maximum_iterations=max_iter
    )
    return x_final

@tf.function
def compute_laplacian_tf(phi, x, y, z,
                         coeff_x_ip1, coeff_x_im1, coeff_x_i,
                         coeff_y_jp1, coeff_y_jm1, coeff_y_j,
                         coeff_z_kp1, coeff_z_km1, coeff_z_k):
    """
    Computes the Laplacian using TensorFlow operations, handling non-uniform
    spacing in x, y, and z directions. Uses 'REFLECT' padding for Neumann BCs.

    Relies on precomputed coefficients calculated with appropriate stability
    (e.g., using an increased epsilon in precompute_laplacian_coefficients).

    Args:
        phi (tf.Tensor): Potential field tensor (shape: [nx, ny, nz]).
        x (tf.Tensor): 1D tensor of x-coordinates (shape: [nx]).
        y (tf.Tensor): 1D tensor of y-coordinates (shape: [ny]).
        z (tf.Tensor): 1D tensor of z-coordinates (shape: [nz]).
        coeff_x_ip1, coeff_x_im1, coeff_x_i (tf.Tensor): Precomputed x-direction coefficients.
        coeff_y_jp1, coeff_y_jm1, coeff_y_j (tf.Tensor): Precomputed y-direction coefficients.
        coeff_z_kp1, coeff_z_km1, coeff_z_k (tf.Tensor): Precomputed z-direction coefficients.

    Returns:
        tf.Tensor: Laplacian of phi (shape: [nx, ny, nz]), dtype=tf.float32.
    """
    # Ensure inputs are float32
    phi = tf.cast(phi, dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)

    # Get dimensions as tensors
    nx, ny, nz = tf.shape(phi)[0], tf.shape(phi)[1], tf.shape(phi)[2]

    # Validate coordinate sizes
    tf.debugging.assert_equal(tf.shape(x)[0], nx, message="x size must match phi's first dimension")
    tf.debugging.assert_equal(tf.shape(y)[0], ny, message="y size must match phi's second dimension")
    tf.debugging.assert_equal(tf.shape(z)[0], nz, message="z size must match phi's third dimension")

    # Pad with 'REFLECT' mode (2 points on each side for central differencing up to boundaries)
    phi_padded_x = tf.pad(phi, [[2, 2], [0, 0], [0, 0]], mode='REFLECT')
    phi_padded_y = tf.pad(phi, [[0, 0], [2, 2], [0, 0]], mode='REFLECT')
    phi_padded_z = tf.pad(phi, [[0, 0], [0, 0], [2, 2]], mode='REFLECT')

    # X second derivative
    coeff_x_ip1_r = tf.reshape(coeff_x_ip1, [-1, 1, 1])
    coeff_x_im1_r = tf.reshape(coeff_x_im1, [-1, 1, 1])
    coeff_x_i_r = tf.reshape(coeff_x_i, [-1, 1, 1])
    phi_im1_x = tf.slice(phi_padded_x, [2, 0, 0], [nx - 2, ny, nz])
    phi_i_x = tf.slice(phi_padded_x, [3, 0, 0], [nx - 2, ny, nz])
    phi_ip1_x = tf.slice(phi_padded_x, [4, 0, 0], [nx - 2, ny, nz])
    d2phi_dx2_inner = phi_ip1_x * coeff_x_ip1_r + phi_i_x * coeff_x_i_r + phi_im1_x * coeff_x_im1_r
    d2phi_dx2 = tf.pad(d2phi_dx2_inner, [[1, 1], [0, 0], [0, 0]], mode='CONSTANT')

    # Y second derivative
    coeff_y_jp1_r = tf.reshape(coeff_y_jp1, [1, -1, 1])
    coeff_y_jm1_r = tf.reshape(coeff_y_jm1, [1, -1, 1])
    coeff_y_j_r = tf.reshape(coeff_y_j, [1, -1, 1])
    phi_jm1_y = tf.slice(phi_padded_y, [0, 2, 0], [nx, ny - 2, nz])
    phi_j_y = tf.slice(phi_padded_y, [0, 3, 0], [nx, ny - 2, nz])
    phi_jp1_y = tf.slice(phi_padded_y, [0, 4, 0], [nx, ny - 2, nz])
    d2phi_dy2_inner = phi_jp1_y * coeff_y_jp1_r + phi_j_y * coeff_y_j_r + phi_jm1_y * coeff_y_jm1_r
    d2phi_dy2 = tf.pad(d2phi_dy2_inner, [[0, 0], [1, 1], [0, 0]], mode='CONSTANT')

    # Z second derivative
    coeff_z_kp1_r = tf.reshape(coeff_z_kp1, [1, 1, -1])
    coeff_z_km1_r = tf.reshape(coeff_z_km1, [1, 1, -1])
    coeff_z_k_r = tf.reshape(coeff_z_k, [1, 1, -1])
    phi_km1_z = tf.slice(phi_padded_z, [0, 0, 2], [nx, ny, nz - 2])
    phi_k_z = tf.slice(phi_padded_z, [0, 0, 3], [nx, ny, nz - 2])
    phi_kp1_z = tf.slice(phi_padded_z, [0, 0, 4], [nx, ny, nz - 2])
    d2phi_dz2_inner = phi_kp1_z * coeff_z_kp1_r + phi_k_z * coeff_z_k_r + phi_km1_z * coeff_z_km1_r
    d2phi_dz2 = tf.pad(d2phi_dz2_inner, [[0, 0], [0, 0], [1, 1]], mode='CONSTANT')

    # Combine all terms
    laplacian = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

    # Debugging prints to trace NaN/Inf source
    conditional_tf_print(DEBUG_FLAG,"Max |phi_clipped|:", tf.reduce_max(tf.abs(phi)))
    conditional_tf_print(DEBUG_FLAG,"Max |d2phi_dx2|:", tf.reduce_max(tf.abs(d2phi_dx2)))
    conditional_tf_print(DEBUG_FLAG,"Max |d2phi_dy2|:", tf.reduce_max(tf.abs(d2phi_dy2)))
    conditional_tf_print(DEBUG_FLAG,"Max |d2phi_dz2|:", tf.reduce_max(tf.abs(d2phi_dz2)))
    conditional_tf_print(DEBUG_FLAG,"Max |laplacian|:", tf.reduce_max(tf.abs(laplacian)))

    # Check for NaNs/Infs
    tf.debugging.check_numerics(laplacian, "Laplacian computation resulted in NaN/Inf")

    return laplacian

@tf.function
def multigrid_v_cycle_tf(phi, f, x_levels, y_levels, z_levels, nx_levels, ny_levels, nz_levels, coeffs_levels, num_levels, boundary_mask, num_cycles=10, cg_max_iter=3):
    """
    Multigrid V-cycle solver for the Poisson equation using TensorFlow.
    Solves ∇²φ = f with boundary conditions specified by boundary_mask.
    Uses Red-Black Gauss-Seidel smoother for improved robustness and GPU compatibility.
    
    Args:
        phi (tf.Tensor): Initial potential [nx, ny, nz], dtype=tf.float32.
        f (tf.Tensor): Right-hand side (-rho/epsilon_0) [nx, ny, nz], dtype=tf.float32.
        x_levels, y_levels, z_levels (tf.TensorArray): Grid coordinates for each level.
        nx_levels, ny_levels, nz_levels (tf.TensorArray): Grid sizes for each level.
        coeffs_levels (tf.TensorArray): Laplacian coefficients for each level.
        num_levels (tf.Tensor): Number of multigrid levels.
        boundary_mask (tf.Tensor): Boolean mask for fixed boundary nodes [nx, ny, nz] at finest level.
        num_cycles (int): Number of V-cycles (default 10).
        cg_max_iter (int): Unused; kept for compatibility (default 3).
    
    Returns:
        tf.Tensor: Final potential [nx, ny, nz], dtype=tf.float32.
    """
    conditional_tf_print(DEBUG_FLAG,"Starting Multigrid V-cycle solver (TensorArray-based)...")
    phi_current = tf.cast(phi, dtype=tf.float32)
    f_current = tf.cast(f, dtype=tf.float32)
    num_levels_int = tf.cast(num_levels, tf.int32)
    max_levels = 10  # Must match max_levels in setup_multigrid_hierarchy

    # Helper function to get level data
    def get_level_data(level):
        safe_level = tf.minimum(level, num_levels_int - 1)
        x_l = x_levels.read(safe_level)
        y_l = y_levels.read(safe_level)
        z_l = z_levels.read(safe_level)
        nx_l = nx_levels.read(safe_level)
        ny_l = ny_levels.read(safe_level)
        nz_l = nz_levels.read(safe_level)
        coeffs_concat = coeffs_levels.read(safe_level)

        # Slice coordinate arrays to match the level's grid size
        x_l_actual = x_l[:nx_l]
        y_l_actual = y_l[:ny_l]
        z_l_actual = z_l[:nz_l]

        # Coefficient slicing
        len_x = tf.maximum(nx_l - 2, 0)
        len_y = tf.maximum(ny_l - 2, 0)
        len_z = tf.maximum(nz_l - 2, 0)
        total_size_needed = 3 * len_x + 3 * len_y + 3 * len_z
        coeffs_concat_actual = coeffs_concat[:total_size_needed]
        sizes = [len_x, len_x, len_x, len_y, len_y, len_y, len_z, len_z, len_z]
        coeffs_list = tf.split(coeffs_concat_actual, sizes)
        coeff_x_ip1, coeff_x_im1, coeff_x_i, coeff_y_jp1, coeff_y_jm1, coeff_y_j, coeff_z_kp1, coeff_z_km1, coeff_z_k = coeffs_list

        coeffs = (coeff_x_ip1, coeff_x_im1, coeff_x_i,
                  coeff_y_jp1, coeff_y_jm1, coeff_y_j,
                  coeff_z_kp1, coeff_z_km1, coeff_z_k)

        return {
            'x': x_l_actual,
            'y': y_l_actual,
            'z': z_l_actual,
            'nx': nx_l,
            'ny': ny_l,
            'nz': nz_l,
            'coeffs': coeffs
        }

    # Single V-cycle function
    def single_v_cycle(phi_v, f_v):
        phi_levels = []
        residual_levels = []
        for lvl in range(max_levels):
            if lvl < num_levels_int:
                nx_l = nx_levels.read(lvl)
                ny_l = ny_levels.read(lvl)
                nz_l = nz_levels.read(lvl)
                shape = [nx_l, ny_l, nz_l]
                phi_levels.append(tf.zeros(shape, dtype=tf.float32))
                residual_levels.append(tf.zeros(shape, dtype=tf.float32))
            else:
                phi_levels.append(tf.zeros([1], dtype=tf.float32))
                residual_levels.append(tf.zeros([1], dtype=tf.float32))

        phi_levels[0] = phi_v
        residual_levels[0] = f_v

        def down_cond(level, *loop_vars):
            return level < num_levels_int - 1

        def down_body(level, *loop_vars):
            phi_list = list(loop_vars[:max_levels])
            res_list = list(loop_vars[max_levels:2*max_levels])
            level_data = get_level_data(level)
            # Define boundary_mask for this level
            boundary_mask_level = tf.cond(
                tf.equal(level, 0),
                lambda: boundary_mask,
                lambda: tf.zeros([level_data['nx'], level_data['ny'], level_data['nz']], dtype=tf.bool)
            )
            phi_smoothed = red_black_gauss_seidel_smoother_tf(
                phi_list[level], res_list[level], level_data, boundary_mask_level, max_iter=10, omega=1.0
            )
            residual = res_list[level] - compute_laplacian_tf(
                phi_smoothed, level_data['x'], level_data['y'], level_data['z'], *level_data['coeffs']
            )
            next_level_data = get_level_data(level + 1)
            residual_coarse = restrict_tf(residual, level_data, next_level_data)
            phi_list[level] = phi_smoothed
            res_list[level + 1] = residual_coarse
            phi_list[level + 1] = tf.zeros_like(residual_coarse)
            return [level + 1] + phi_list + res_list

        initial_vars = [tf.constant(0, tf.int32)] + phi_levels + residual_levels
        final_vars = tf.while_loop(
            down_cond, down_body,
            initial_vars,
            maximum_iterations=num_levels_int - 1
        )
        level = final_vars[0]
        phi_levels = final_vars[1:max_levels + 1]
        residual_levels = final_vars[max_levels + 1:]

        coarsest_level = num_levels_int - 1
        coarsest_data = get_level_data(coarsest_level)
        boundary_mask_coarsest = tf.zeros([coarsest_data['nx'], coarsest_data['ny'], coarsest_data['nz']], dtype=tf.bool)
        phi_coarsest = red_black_gauss_seidel_smoother_tf(
            tf.zeros_like(residual_levels[coarsest_level]), residual_levels[coarsest_level],
            coarsest_data, boundary_mask_coarsest, max_iter=10, omega=1.0
        )
        phi_levels[coarsest_level] = phi_coarsest

        def up_cond(level, *loop_vars):
            return level >= 0

        def up_body(level, *loop_vars):
            phi_list = list(loop_vars[:max_levels])
            res_list = list(loop_vars[max_levels:2*max_levels])
            curr_level_data = get_level_data(level)
            # Define boundary_mask for this level
            boundary_mask_level = tf.cond(
                tf.equal(level, 0),
                lambda: boundary_mask,
                lambda: tf.zeros([curr_level_data['nx'], curr_level_data['ny'], curr_level_data['nz']], dtype=tf.bool)
            )
            next_level_data = get_level_data(level + 1)
            error_correction_coarse = phi_list[level + 1]
            error_fine = prolongate_tf(error_correction_coarse, next_level_data, curr_level_data)
            phi_corrected = phi_list[level] + error_fine
            phi_final = red_black_gauss_seidel_smoother_tf(
                phi_corrected, f_v if level == 0 else residual_levels[level],
                curr_level_data, boundary_mask_level, max_iter=10, omega=1.0
            )
            phi_final = tf.cond(
                tf.equal(level, 0),
                lambda: tf.where(boundary_mask, phi_v, phi_final),
                lambda: phi_final
            )
            phi_list[level] = phi_final
            return [level - 1] + phi_list + res_list

        initial_up_vars = [num_levels_int - 2] + phi_levels + residual_levels
        final_up_vars = tf.while_loop(
            up_cond, up_body,
            initial_up_vars,
            maximum_iterations=num_levels_int - 1
        )
        phi_levels = final_up_vars[1:max_levels + 1]
        return phi_levels[0]

    # Define clipping bound
    max_initial_phi_abs = tf.reduce_max(tf.abs(phi)) + tf.constant(1.0, dtype=tf.float32)

    def cycle_body(cycle, phi_c):
        conditional_tf_print(DEBUG_FLAG,"Starting V-cycle", cycle + 1)
        phi_next = single_v_cycle(phi_c, f_current)
        phi_next_clipped = tf.clip_by_value(phi_next, -max_initial_phi_abs, max_initial_phi_abs)
        return [cycle + 1, phi_next_clipped]

    _, phi_final = tf.while_loop(
        lambda c, p: c < num_cycles,
        cycle_body,
        [tf.constant(0, tf.int32), phi_current],
        maximum_iterations=num_cycles
    )

    # Residual Check
    conditional_tf_print(DEBUG_FLAG,"Computing final Poisson residual...")
    finest_level_data = get_level_data(0)
    residual = f_current - compute_laplacian_tf(
        phi_final,
        finest_level_data['x'],
        finest_level_data['y'],
        finest_level_data['z'],
        *finest_level_data['coeffs']
    )
    max_residual = tf.reduce_max(tf.abs(residual))
    conditional_tf_print(DEBUG_FLAG,"Poisson residual: Max |residual| =", max_residual)

    conditional_tf_print(DEBUG_FLAG,"Multigrid solver completed.")
    return phi_final

@tf.function
def solve_pressure_poisson_tf(
    p_init, div_u_star, rho_fluid, dt, x, y, z,
    nx, ny, nz,
    tol=1e-3, max_iter=50000, omega=1.0
):
    """
    Solves the pressure Poisson equation using a multigrid method with TensorFlow.
    
    Args:
        p_init (tf.Tensor): Initial pressure field [nx, ny, nz], dtype=tf.float32.
        div_u_star (tf.Tensor): Divergence of the intermediate velocity [nx, ny, nz], dtype=tf.float32.
        rho_fluid (float or tf.Tensor): Fluid density, scalar.
        dt (float or tf.Tensor): Time step, scalar.
        x, y, z (tf.Tensor): 1D coordinate arrays [nx], [ny], [nz], dtype=tf.float32.
        nx, ny, nz (int or tf.Tensor): Grid dimensions, scalar.
        tol (float): Convergence tolerance, default 1e-3.
        max_iter (int): Maximum iterations (not used in multigrid, kept for compatibility).
        omega (float): Relaxation parameter (not used in multigrid, kept for compatibility).
    
    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            - p_final: Final pressure field [nx, ny, nz], dtype=tf.float32.
            - converged: Boolean indicating convergence, dtype=tf.bool.
            - iter_count: Number of iterations (approximate), dtype=tf.int32.
            - residual: Final residual norm, dtype=tf.float32.
            - error_code: Status code (0=success, 1=not converged, 3=NaN/Inf), dtype=tf.int32.
    """
    # Convert inputs to TensorFlow tensors with appropriate types
    p_tf = tf.convert_to_tensor(p_init, dtype=tf.float32)
    div_u_star_tf = tf.convert_to_tensor(div_u_star, dtype=tf.float32)
    rho_fluid_tf = tf.cast(rho_fluid, tf.float32)
    dt_tf = tf.cast(dt, tf.float32)
    x_tf = tf.cast(x, tf.float32)
    y_tf = tf.cast(y, tf.float32)
    z_tf = tf.cast(z, tf.float32)
    nx_tf = tf.cast(nx, tf.int32)
    ny_tf = tf.cast(ny, tf.int32)
    nz_tf = tf.cast(nz, tf.int32)
    tol_tf = tf.cast(tol, tf.float32)
    
    # Compute right-hand side
    rhs = (rho_fluid_tf / dt_tf) * div_u_star_tf
    boundary_mask = tf.zeros([nx_tf, ny_tf, nz_tf], dtype=tf.bool)  # All Neumann BCs
    
    # Precompute multigrid hierarchy using updated function
    (x_array, y_array, z_array, nx_array, ny_array, nz_array, coeffs_array, num_levels) = setup_multigrid_hierarchy_tf(x_tf, y_tf, z_tf, nx_tf, ny_tf, nz_tf)
    p_final = multigrid_v_cycle_tf(
        p_tf, rhs,
        x_array, y_array, z_array,
        nx_array, ny_array, nz_array,
        coeffs_array, num_levels,
        boundary_mask,
        num_cycles=2
    )
    
    # Compute residual for convergence check
    coeffs = precompute_laplacian_coefficients(x_tf, y_tf, z_tf)
    residual = tf.reduce_max(tf.abs(compute_laplacian_tf(
        p_final, x_tf, y_tf, z_tf, *coeffs
    ) - rhs))
    converged = tf.cast(residual < tol_tf, tf.bool)
    iter_count = tf.constant(2, tf.int32)  # Approximate, based on MG cycles
    
    # Determine error code
    error_code = tf.cond(
        tf.reduce_all(tf.math.is_finite(p_final)),
        lambda: tf.constant(0 if converged else 1, tf.int32),
        lambda: tf.constant(3, tf.int32)
    )
    
    return p_final, converged, iter_count, residual, error_code

def get_material_properties(mat_name, materials_data, log_func=None):
    """
    Retrieves physics properties based on material name.
    Placeholder function - Needs actual data linking materials to E_onset.

    Args:
        mat_name (str): The name of the material to look up.
        materials_data (dict): Dictionary containing material properties, typically loaded from a JSON file.
                               Expected structure: {'materials': {'material_name': {'property': value, ...}}}
        log_func (callable, optional): Function to use for logging messages. Defaults to None (uses print).

    Returns:
        dict: A dictionary containing the retrieved property ('E_onset').
              Returns default values if the material is not found or data is missing.
    """
    # Default properties
    props = {'E_onset': E_ONSET_DEFAULT}

    if materials_data and isinstance(materials_data, dict) and 'materials' in materials_data and mat_name in materials_data['materials']:
        mat_props = materials_data['materials'][mat_name]
        if isinstance(mat_props, dict):  # Check if the specific material entry is a dictionary
            # Retrieve properties, using defaults if a specific property is missing in the data
            props['E_onset'] = mat_props.get('corona_onset_V_m', E_ONSET_DEFAULT)
            # Add other material-specific properties like work function, secondary emission coeff if needed
        else:
            # Log or print a warning if the material entry isn't a dictionary
            msg = f"Warning: Data for material '{mat_name}' is not in the expected format. Using default EHD properties."
            if log_func:
                log_func(msg)
            else:
                print(msg)
    else:
        # Use conditional logging for the "material not found" warning
        msg = f"Warning: Material '{mat_name}' not found in provided data. Using default EHD properties."
        if log_func:
            log_func(msg)
        else:
            print(msg)

    return props

# --- Parameter Preparation Function (Example - adapt as needed) ---
def prepare_corona_params(emitter_props=None):
    """
    Prepares the parameter dictionary for the corona simulation,
    using defaults and allowing overrides from emitter_props.

    Args:
        emitter_props (dict, optional): Dictionary with emitter-specific
                                        properties that might override defaults.
                                        Defaults to None.

    Returns:
        dict: Dictionary containing parameters for the simulation.
    """
    if emitter_props is None:
        emitter_props = {}

    # Updated corona_params dictionary using corrected defaults and keys
    # Fetching 'townsend_A' and 'townsend_B' allows overriding defaults
    corona_params = {
        # Use standard A, B parameters directly
        'townsend_A': emitter_props.get('townsend_A', TOWNSEND_A_DEFAULT),
        'townsend_B': emitter_props.get('townsend_B', TOWNSEND_B_DEFAULT),

        # Include other necessary physical parameters
        'electron_mobility': MU_E,         # Assumes MU_E is globally defined/imported
        'recombination_coeff': BETA_RECOMB, # Assumes BETA_RECOMB is globally defined/imported
        'attachment_coeff': ETA_ATTACH,    # Assumes ETA_ATTACH is globally defined/imported

        # Add other parameters your simulation might need
        # 'pressure': P_ATM # Example if pressure is constant and part of params
    }
    return corona_params

# --- Corrected Ionization Calculation Function ---
@tf.function
def corona_source_tf(E_mag, p, townsend_A, townsend_B):
    """
    Calculates the effective Townsend ionization coefficient (alpha_eff) with proper unit conversion.
    
    Args:
        E_mag (tf.Tensor): Electric field magnitude (V/m).
        p (tf.Tensor): Pressure (Pa).
        townsend_A (tf.Tensor or float): Townsend A coefficient (cm⁻¹ torr⁻¹).
        townsend_B (tf.Tensor or float): Townsend B coefficient (V cm⁻¹ torr⁻¹).
    
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: (alpha_eff_tf in m⁻¹, error_code)
    """
    try:
        # Conversion constants
        PA_PER_TORR = tf.constant(133.322, dtype=tf.float32)  # Pa to torr
        CM_PER_M = tf.constant(100.0, dtype=tf.float32)       # m⁻¹ to cm⁻¹ conversion factor
        ZERO_TF = tf.constant(0.0, dtype=tf.float32)
        TF_EPSILON = tf.constant(1e-30, dtype=tf.float32)

        # Cast inputs
        A_tf = tf.cast(townsend_A, dtype=tf.float32)  # e.g., 15 cm⁻¹ torr⁻¹
        B_tf = tf.cast(townsend_B, dtype=tf.float32)  # e.g., 365 V cm⁻¹ torr⁻¹
        E_mag_tf = tf.cast(E_mag, dtype=tf.float32)   # V/m
        p_tf = tf.cast(p, dtype=tf.float32)           # Pa

        # Convert units
        p_torr = p_tf / PA_PER_TORR                  # Convert Pa to torr
        E_cm = E_mag_tf * CM_PER_M                   # Convert V/m to V/cm

        # Compute alpha in cm⁻¹ using standard Townsend formula
        alpha_cm = p_torr * A_tf * tf.exp(-B_tf / (E_cm / p_torr + TF_EPSILON))

        # Convert alpha to m⁻¹
        alpha_eff_tf = alpha_cm * CM_PER_M
        alpha_eff_tf = tf.maximum(alpha_eff_tf, ZERO_TF)

        # Error checking
        is_finite = tf.reduce_all(tf.math.is_finite(alpha_eff_tf))
        error_code = tf.cond(
            is_finite,
            lambda: tf.constant(0, dtype=tf.int32),
            lambda: tf.constant(3, dtype=tf.int32)
        )

        # Debugging output
        conditional_tf_print(DEBUG_FLAG,"Max alpha_eff_tf:", tf.reduce_max(alpha_eff_tf))

    except Exception as e:
        conditional_tf_print(DEBUG_FLAG,"ERROR (corona_source_tf):", e, output_stream=sys.stderr)
        alpha_eff_tf = tf.fill(tf.shape(E_mag), tf.constant(np.nan, dtype=tf.float32))
        error_code = tf.constant(4, dtype=tf.int32)

    return alpha_eff_tf, error_code

@tf.function
def solve_electron_transport_tf(
    n_e_in, n_i_in, Ex_in, Ey_in, Ez_in, S_e_in, dx_in, dy_in, z_in,
    nx, ny, nz, collector_mask_in, dt_in, steps,
    BETA_RECOMB_ARG_in, MU_E_ARG_in, D_E_ARG_in,
    x_1d_tf, y_1d_tf,  # 1D coordinate arrays for non-uniform grid
    current_seeding_mask_in,  # Mask for applying seed density cap
    initial_seed_density_in  # Value for the seed density cap
):
    """
    Solves the electron transport equation using a semi-implicit (IMEX) scheme
    with TensorFlow. The diffusion term is treated implicitly using Jacobi iterations,
    while advection, source, and recombination are treated explicitly.

    Incorporates clipping and increased Jacobi iterations for stability based on recommendations.
    Adds extensive debugging checks (tf.debugging.check_numerics and tf.print).

    Args:
        n_e_in (tf.Tensor): Initial electron density [nx, ny, nz].
        n_i_in (tf.Tensor): Ion density (for recombination) [nx, ny, nz].
        Ex_in, Ey_in, Ez_in (tf.Tensor): Electric field components [nx, ny, nz].
        S_e_in (tf.Tensor): Electron source term [nx, ny, nz].
        dx_in, dy_in (tf.Tensor): Grid spacings in x, y (scalar, float32).
        z_in (tf.Tensor): 1D z-coordinate array [nz], float32.
        nx, ny, nz (tf.Tensor): Grid dimensions (scalar, int32).
        collector_mask_in (tf.Tensor): Boolean mask for collector nodes [nx, ny, nz].
        dt_in (tf.Tensor): Time step (scalar, float32).
        steps (tf.Tensor): Number of time steps (scalar, int32).
        BETA_RECOMB_ARG_in (tf.Tensor): Recombination coefficient (scalar, float32).
        MU_E_ARG_in (tf.Tensor): Electron mobility (scalar, float32).
        D_E_ARG_in (tf.Tensor): Electron diffusion coefficient (scalar, float32).
        x_1d_tf (tf.Tensor): 1D x-coordinate array [nx], float32.
        y_1d_tf (tf.Tensor): 1D y-coordinate array [ny], float32.
        current_seeding_mask_in (tf.Tensor): Boolean mask for applying seed density cap [nx, ny, nz].
        initial_seed_density_in (tf.Tensor): Value for the seed density cap (scalar, float32).

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - n_e_final (tf.Tensor): Final electron density [nx, ny, nz].
            - final_error_code (tf.Tensor): Final status code (scalar, int32).
    """
    # --- Ensure Inputs are TensorFlow Tensors and float32 Precision ---
    n_e_init = tf.cast(tf.convert_to_tensor(n_e_in), dtype=tf.float32)
    n_i = tf.cast(tf.convert_to_tensor(n_i_in), dtype=tf.float32)
    Ex = tf.cast(tf.convert_to_tensor(Ex_in), dtype=tf.float32)
    Ey = tf.cast(tf.convert_to_tensor(Ey_in), dtype=tf.float32)
    Ez = tf.cast(tf.convert_to_tensor(Ez_in), dtype=tf.float32)
    S_e = tf.cast(tf.convert_to_tensor(S_e_in), dtype=tf.float32)
    z = tf.cast(tf.convert_to_tensor(z_in), dtype=tf.float32)  # 1D z-coords
    collector_mask = tf.cast(tf.convert_to_tensor(collector_mask_in), dtype=tf.bool)
    x_1d = tf.cast(x_1d_tf, dtype=tf.float32)  # Use directly
    y_1d = tf.cast(y_1d_tf, dtype=tf.float32)  # Use directly
    current_seeding_mask = tf.cast(current_seeding_mask_in, dtype=tf.bool)
    initial_seed_density = tf.cast(initial_seed_density_in, dtype=tf.float32)

    # Cast scalar parameters to tf.float32/tf.int32
    dx = tf.cast(dx_in, dtype=tf.float32)
    dy = tf.cast(dy_in, dtype=tf.float32)
    dt = tf.cast(dt_in, dtype=tf.float32)
    BETA_RECOMB_ARG = tf.cast(BETA_RECOMB_ARG_in, dtype=tf.float32)
    MU_E_ARG = tf.cast(MU_E_ARG_in, dtype=tf.float32)
    D_E_ARG = tf.cast(D_E_ARG_in, dtype=tf.float32)
    steps_tf = tf.cast(steps, dtype=tf.int32)

    # Compute dz_scalar_tf from z_in for correct z-direction spacing
    dz_scalar_tf = tf.cond(
        nz > 1,
        lambda: tf.reduce_mean(z[1:] - z[:-1]),
        lambda: tf.constant(1.0, dtype=tf.float32)
    )

    # Define maximum density for clipping
    MAX_DENSITY_CLIP = tf.constant(1e17, dtype=tf.float32)

    # --- Extensive Debugging: Check initial inputs ---
    conditional_tf_print(DEBUG_FLAG,"--- Starting solve_electron_transport_tf (Semi-Implicit with Debugging) ---")
    tf.debugging.check_numerics(n_e_init, "Initial n_e_init contains NaN/Inf")
    tf.debugging.check_numerics(n_i, "Initial n_i contains NaN/Inf")
    tf.debugging.check_numerics(Ex, "Initial Ex contains NaN/Inf")
    tf.debugging.check_numerics(Ey, "Initial Ey contains NaN/Inf")
    tf.debugging.check_numerics(Ez, "Initial Ez contains NaN/Inf")
    tf.debugging.check_numerics(S_e, "Initial S_e contains NaN/Inf")  # Initial check on source term
    conditional_tf_print(DEBUG_FLAG,"Input Max Values: n_e=", tf.reduce_max(n_e_init), "n_i=", tf.reduce_max(n_i),
             "|Ex|=", tf.reduce_max(tf.abs(Ex)), "|Ey|=", tf.reduce_max(tf.abs(Ey)), "|Ez|=", tf.reduce_max(tf.abs(Ez)),
             "S_e=", tf.reduce_max(S_e))
    conditional_tf_print(DEBUG_FLAG,"Simulation Params: dt=", dt, "steps=", steps_tf, "D_E=", D_E_ARG, "Mu_E=", MU_E_ARG, "Beta_R=", BETA_RECOMB_ARG)
    conditional_tf_print(DEBUG_FLAG,"Max Density Clip=", MAX_DENSITY_CLIP)
    # --- End Extensive Debugging ---

    # --- Precompute Laplacian Coefficients ---
    (coeff_x_ip1, coeff_x_im1, coeff_x_i,
     coeff_y_jp1, coeff_y_jm1, coeff_y_j,
     coeff_z_kp1, coeff_z_km1, coeff_z_k) = precompute_laplacian_coefficients(x_1d, y_1d, z)
    conditional_tf_print(DEBUG_FLAG,"Laplacian coefficients precomputed.")

    # --- Precompute Drift Velocities ---
    v_drift_x = -MU_E_ARG * Ex
    v_drift_y = -MU_E_ARG * Ey
    v_drift_z = -MU_E_ARG * Ez
    # --- Extensive Debugging: Check drift velocities ---
    tf.debugging.check_numerics(v_drift_x, "v_drift_x contains NaN/Inf")
    tf.debugging.check_numerics(v_drift_y, "v_drift_y contains NaN/Inf")
    tf.debugging.check_numerics(v_drift_z, "v_drift_z contains NaN/Inf")
    conditional_tf_print(DEBUG_FLAG,"Drift velocities computed. Max |vx|=", tf.reduce_max(tf.abs(v_drift_x)),
             "|vy|=", tf.reduce_max(tf.abs(v_drift_y)), "|vz|=", tf.reduce_max(tf.abs(v_drift_z)))
    # --- End Extensive Debugging ---

    # --- Initialize Loop Variables ---
    step = tf.constant(0, dtype=tf.int32)
    n_e_current = n_e_init
    error_code = SUCCESS_CODE
    delta_n_e = tf.constant(np.inf, dtype=tf.float32)  # Initialize to ensure loop entry
    convergence_tolerance = tf.constant(1e-3, dtype=tf.float32)

    conditional_tf_print(DEBUG_FLAG,"Initial Loop Variables: n_e max=", tf.reduce_max(n_e_current), "delta_n_e=", delta_n_e, "Tolerance=", convergence_tolerance)

    # --- Define Loop Condition ---
    def condition(step, n_e_curr, err_code, delta_n_e):
        conditional_tf_print(DEBUG_FLAG,"Loop Condition Check: step=", step, "steps_tf=", steps_tf, "err_code=", err_code, "delta_n_e=", delta_n_e)
        return tf.logical_and(
            tf.logical_and(step < steps_tf, tf.equal(err_code, SUCCESS_CODE)),
            delta_n_e > convergence_tolerance
        )

    # --- Define Loop Body ---
    def body(step, n_e_curr, err_code, delta_n_e):
        conditional_tf_print(DEBUG_FLAG,"\n--- Electron Transport Step:", step + 1, "/", steps_tf, "---")
        n_e_old = n_e_curr
        conditional_tf_print(DEBUG_FLAG," Start of step", step + 1, ": Max n_e_old=", tf.reduce_max(n_e_old))

        # --- Calculate Explicit Terms ---
        conditional_tf_print(DEBUG_FLAG," Calculating explicit terms...")
        recomb_term_inner = -BETA_RECOMB_ARG * n_e_old[1:-1, 1:-1, 1:-1] * n_i[1:-1, 1:-1, 1:-1]
        source_term_inner = S_e[1:-1, 1:-1, 1:-1]  # Use S_e passed into the function

        v_drift_tuple = (v_drift_x, v_drift_y, v_drift_z)
        advection_term_full = calculate_weno_advection_term_tf(
            v_drift_tuple, n_e_old, dx, dy, dz_scalar_tf, nx, ny, nz, boundary_mode=1
        )
        # Clip advection term to prevent extreme values leading to NaN/Inf
        advection_term_full = tf.clip_by_value(advection_term_full, -1e15, 1e15)
        advection_term_inner = advection_term_full[1:-1, 1:-1, 1:-1]

        # --- Extensive Debugging: Check explicit terms ---
        tf.debugging.check_numerics(recomb_term_inner, "Recombination term contains NaN/Inf")
        tf.debugging.check_numerics(source_term_inner, "Source term contains NaN/Inf")
        tf.debugging.check_numerics(advection_term_full, "Advection term (full) contains NaN/Inf")
        tf.debugging.check_numerics(advection_term_inner, "Advection term (inner) contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG," Explicit terms calculated.")
        conditional_tf_print(DEBUG_FLAG," Max |recomb|:", tf.reduce_max(tf.abs(recomb_term_inner)),
                 "Max |source|:", tf.reduce_max(tf.abs(source_term_inner)),
                 "Max |adv|:", tf.reduce_max(tf.abs(advection_term_inner)))
        # --- End Extensive Debugging ---

        # --- Calculate RHS for Implicit Diffusion ---
        conditional_tf_print(DEBUG_FLAG," Calculating RHS...")
        rhs_inner = n_e_old[1:-1, 1:-1, 1:-1] + dt * (source_term_inner + recomb_term_inner - advection_term_inner)
        rhs = tf.pad(rhs_inner, [[1, 1], [1, 1], [1, 1]], mode='CONSTANT', constant_values=0.0)

        # --- Extensive Debugging: Check RHS ---
        tf.debugging.check_numerics(rhs, "RHS contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG," RHS computed. Max |RHS|=", tf.reduce_max(tf.abs(rhs)))
        # --- End Extensive Debugging ---

        # --- Solve Implicit Diffusion using Jacobi Iterations ---
        n_e_jacobi_iter = n_e_old
        num_jacobi_iters = tf.constant(20, dtype=tf.int32)  # Increased from 5
        conditional_tf_print(DEBUG_FLAG," Starting Jacobi iterations (", num_jacobi_iters, " iters)...")

        def jacobi_condition(j, n_e_iter):
            return j < num_jacobi_iters

        def jacobi_body(j, n_e_iter):
            conditional_tf_print(DEBUG_FLAG," Jacobi iter", j + 1)  # Log iteration number
            lap_n_e_iter = compute_laplacian_tf(
                n_e_iter, x_1d, y_1d, z,
                coeff_x_ip1, coeff_x_im1, coeff_x_i,
                coeff_y_jp1, coeff_y_jm1, coeff_y_j,
                coeff_z_kp1, coeff_z_km1, coeff_z_k
            )
            # --- Extensive Debugging: Check Laplacian inside Jacobi ---
            tf.debugging.check_numerics(lap_n_e_iter, "Laplacian in Jacobi contains NaN/Inf")
            conditional_tf_print(DEBUG_FLAG," Max Lap(n_e_iter)=", tf.reduce_max(tf.abs(lap_n_e_iter)))
            # --- End Extensive Debugging ---

            n_e_next_jacobi_raw = rhs + dt * D_E_ARG * lap_n_e_iter

            # Apply clipping inside Jacobi loop
            n_e_next_jacobi = tf.maximum(n_e_next_jacobi_raw, TF_ZERO)
            n_e_next_jacobi = tf.minimum(n_e_next_jacobi, MAX_DENSITY_CLIP)  # Clip to 1e17 m^-3

            # --- Extensive Debugging: Check Jacobi update ---
            tf.debugging.check_numerics(n_e_next_jacobi, "n_e_next_jacobi contains NaN/Inf")
            conditional_tf_print(DEBUG_FLAG," Max n_e_next_jacobi (clipped)=", tf.reduce_max(n_e_next_jacobi))
            # --- End Extensive Debugging ---

            return [j + 1, n_e_next_jacobi]  # Ensure list return

        # Execute Jacobi loop
        jacobi_loop_vars = [tf.constant(0, dtype=tf.int32), n_e_jacobi_iter]
        jacobi_final_step, n_e_implicit = tf.while_loop(
            jacobi_condition,
            jacobi_body,
            jacobi_loop_vars,
            maximum_iterations=num_jacobi_iters,
            shape_invariants=[tf.TensorSpec([], tf.int32), tf.TensorSpec([nx, ny, nz], tf.float32)]
        )
        # --- Extensive Debugging: Check result after Jacobi ---
        tf.debugging.check_numerics(n_e_implicit, "n_e_implicit contains NaN/Inf after Jacobi loop")
        conditional_tf_print(DEBUG_FLAG," Jacobi iterations finished. Max n_e_implicit=", tf.reduce_max(n_e_implicit))
        # --- End Extensive Debugging ---

        # --- Post-Processing ---
        conditional_tf_print(DEBUG_FLAG," Applying post-processing (BCs, seeding clamp, clipping)...")
        # Apply Neumann BC first, then collector BC
        n_e_implicit_neumann = apply_neumann_bc_tf(n_e_implicit)
        n_e_implicit_bc = tf.where(collector_mask, TF_ZERO, n_e_implicit_neumann)

        # Apply seeding clamp
        n_e_final_step_before_clip = tf.where(
            current_seeding_mask,
            tf.minimum(n_e_implicit_bc, initial_seed_density),
            n_e_implicit_bc
        )

        # Ensure non-negativity and apply final clip
        n_e_final_step = tf.maximum(n_e_final_step_before_clip, TF_ZERO)
        n_e_final_step = tf.clip_by_value(n_e_final_step, 0.0, MAX_DENSITY_CLIP)

        # --- Extensive Debugging: Check final step result ---
        tf.debugging.check_numerics(n_e_final_step, "n_e_final_step contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG," Post-processing applied. Max n_e_final_step=", tf.reduce_max(n_e_final_step))
        # --- End Extensive Debugging ---

        delta_n_e_new = tf.reduce_max(tf.abs(n_e_final_step - n_e_old))
        conditional_tf_print(DEBUG_FLAG," Change delta_n_e =", delta_n_e_new)

        is_finite_step = tf.reduce_all(tf.math.is_finite(n_e_final_step))
        new_err_code = tf.cond(
            is_finite_step,
            lambda: err_code,
            lambda: tf.constant(NAN_INF_STEP_CODE, dtype=tf.int32)
        )
        conditional_tf_print(DEBUG_FLAG," Finiteness check: is_finite=", is_finite_step, "new_err_code=", new_err_code)

        n_e_next = tf.cond(
            is_finite_step,
            lambda: n_e_final_step,
            lambda: n_e_old  # Revert to old state if NaN/Inf detected
        )

        return [step + 1, n_e_next, new_err_code, delta_n_e_new]  # Ensure list return

    # --- Execute Outer Loop ---
    conditional_tf_print(DEBUG_FLAG,"\nStarting outer time-stepping loop...")
    outer_loop_vars = [step, n_e_current, error_code, delta_n_e]
    final_step, n_e_final, final_error_code_loop, final_delta_n_e = tf.while_loop(
        condition,
        body,
        loop_vars=outer_loop_vars,
        maximum_iterations=steps_tf,
        shape_invariants=[
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([nx, ny, nz], tf.float32),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.float32)
        ]
    )
    conditional_tf_print(DEBUG_FLAG,"\nOuter loop finished at step:", final_step, "with status:", final_error_code_loop, "final_delta_n_e:", final_delta_n_e)

    # --- Final Checks ---
    converged_check = final_delta_n_e <= convergence_tolerance
    tf.cond(
        tf.logical_and(tf.equal(final_step, steps_tf), tf.logical_not(converged_check)),
        lambda: conditional_tf_print(DEBUG_FLAG,"WARNING: Reached max steps but delta_n_e (", final_delta_n_e, ") > tolerance (", convergence_tolerance, ")."),
        lambda: tf.no_op()
    )

    is_finite_final = tf.reduce_all(tf.math.is_finite(n_e_final))
    final_error_code_out = tf.cond(
        tf.equal(final_error_code_loop, SUCCESS_CODE),
        lambda: tf.cond(
            is_finite_final,
            lambda: SUCCESS_CODE,
            lambda: NAN_INF_FINAL_CODE
        ),
        lambda: final_error_code_loop  # Propagate earlier error (e.g., NAN_INF_STEP_CODE)
    )

    tf.cond(
        tf.equal(final_error_code_out, NAN_INF_STEP_CODE),
        lambda: conditional_tf_print(DEBUG_FLAG,"WARNING: Loop stopped due to NaN/Inf during a step."),
        lambda: tf.no_op()
    )
    tf.cond(
        tf.equal(final_error_code_out, NAN_INF_FINAL_CODE),
        lambda: conditional_tf_print(DEBUG_FLAG,"ERROR: NaN/Inf in final electron density."),
        lambda: tf.no_op()
    )

    conditional_tf_print(DEBUG_FLAG,"Final state: Max n_e=", tf.reduce_max(n_e_final), "Status Code:", final_error_code_out)
    conditional_tf_print(DEBUG_FLAG,"--- Exiting solve_electron_transport_tf ---")
    return n_e_final, final_error_code_out

@tf.function
def solve_ion_transport_tf(
    n_i_old, Ex, Ey, Ez, ux, uy, uz, S_i,
    nx, ny, nz, dx, dy, z, # nx, ny, nz, dx, dy are tf.Tensor scalars; z is 1D tf.Tensor
    collector_mask, dt, steps, # collector_mask is bool Tensor; dt, steps are tf.Tensor scalars
    MU_I_ARG, D_I_ARG, # Scalar tf.Tensor arguments
    x_1d_tf, y_1d_tf # 1D tf.Tensor coordinate arrays
):
    """
    Solves the ion transport equation using a semi-implicit (IMEX) scheme
    with TensorFlow. Diffusion is implicit (Jacobi), others explicit.

    Incorporates clipping and increased Jacobi iterations for stability based on recommendations.

    Args:
        n_i_old (tf.Tensor): Initial ion density [nx, ny, nz].
        Ex, Ey, Ez (tf.Tensor): Electric field components [nx, ny, nz].
        ux, uy, uz (tf.Tensor): Fluid velocity components [nx, ny, nz].
        S_i (tf.Tensor): Ion source term [nx, ny, nz].
        nx, ny, nz (tf.Tensor): Grid dimensions (scalar, int32).
        dx, dy (tf.Tensor): Average grid spacings (scalar, float32).
        z (tf.Tensor): 1D z-coordinate array [nz], float32.
        collector_mask (tf.Tensor): Boolean mask for collector nodes [nx, ny, nz].
        dt (tf.Tensor): Time step (scalar, float32).
        steps (tf.Tensor): Number of time steps (scalar, int32).
        MU_I_ARG (tf.Tensor): Ion mobility (scalar, float32).
        D_I_ARG (tf.Tensor): Ion diffusion coefficient (scalar, float32).
        x_1d_tf (tf.Tensor): 1D x-coordinate array [nx], float32.
        y_1d_tf (tf.Tensor): 1D y-coordinate array [ny], float32.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - n_i_final (tf.Tensor): Final ion density [nx, ny, nz].
            - final_error_code (tf.Tensor): Final status code (scalar, int32).
    """
    # --- Input Casting --- Ensure float32/int32/bool as needed
    n_i_tf = tf.cast(tf.convert_to_tensor(n_i_old), dtype=tf.float32)
    Ex_tf = tf.cast(tf.convert_to_tensor(Ex), dtype=tf.float32)
    Ey_tf = tf.cast(tf.convert_to_tensor(Ey), dtype=tf.float32)
    Ez_tf = tf.cast(tf.convert_to_tensor(Ez), dtype=tf.float32)
    ux_tf = tf.cast(tf.convert_to_tensor(ux), dtype=tf.float32)
    uy_tf = tf.cast(tf.convert_to_tensor(uy), dtype=tf.float32)
    uz_tf = tf.cast(tf.convert_to_tensor(uz), dtype=tf.float32)
    S_i_tf = tf.cast(tf.convert_to_tensor(S_i), dtype=tf.float32)
    z_tf = tf.cast(tf.convert_to_tensor(z), dtype=tf.float32) # 1D z-coords
    collector_mask_tf = tf.cast(tf.convert_to_tensor(collector_mask), dtype=tf.bool)
    x_1d = tf.cast(x_1d_tf, dtype=tf.float32) # Use directly
    y_1d = tf.cast(y_1d_tf, dtype=tf.float32) # Use directly

    # Cast scalar parameters
    dt_tf = tf.cast(dt, dtype=tf.float32)
    dx_tf = tf.cast(dx, dtype=tf.float32) # Assumed average dx
    dy_tf = tf.cast(dy, dtype=tf.float32) # Assumed average dy
    steps_tf = tf.cast(steps, dtype=tf.int32)
    MU_I_tf = tf.cast(MU_I_ARG, dtype=tf.float32)
    D_I_tf = tf.cast(D_I_ARG, dtype=tf.float32)

    # Define maximum density for clipping
    MAX_DENSITY_CLIP = tf.constant(1e17, dtype=tf.float32)

    conditional_tf_print(DEBUG_FLAG,"Starting solve_ion_transport_tf (Semi-Implicit)")
    conditional_tf_print(DEBUG_FLAG,"dt=", dt_tf, "steps=", steps_tf, "D_I=", D_I_tf, "Max Density Clip=", MAX_DENSITY_CLIP)

    # --- Precompute Laplacian Coefficients ---
    (coeff_x_ip1, coeff_x_im1, coeff_x_i,
     coeff_y_jp1, coeff_y_jm1, coeff_y_j,
     coeff_z_kp1, coeff_z_km1, coeff_z_k) = precompute_laplacian_coefficients(x_1d, y_1d, z_tf)
    conditional_tf_print(DEBUG_FLAG,"Laplacian coefficients precomputed for ions.")

    # --- Compute dz_scalar_tf from z_tf ---
    dz_scalar_tf = tf.cond(
        nz > 1,
        lambda: tf.reduce_mean(z_tf[1:] - z_tf[:-1]),
        lambda: tf.constant(1.0, dtype=tf.float32)
    )

    # --- Precompute Effective Velocities ---
    vx_eff = ux_tf + MU_I_tf * Ex_tf
    vy_eff = uy_tf + MU_I_tf * Ey_tf
    vz_eff = uz_tf + MU_I_tf * Ez_tf
    conditional_tf_print(DEBUG_FLAG,"Effective ion velocities computed.")
    conditional_tf_print(DEBUG_FLAG,"Max |vx_eff|:", tf.reduce_max(tf.abs(vx_eff)))
    conditional_tf_print(DEBUG_FLAG,"Max |vy_eff|:", tf.reduce_max(tf.abs(vy_eff)))
    conditional_tf_print(DEBUG_FLAG,"Max |vz_eff|:", tf.reduce_max(tf.abs(vz_eff)))

    # --- Initialize Loop Variables ---
    step = tf.constant(0, dtype=tf.int32)
    n_i_current = n_i_tf
    error_code = tf.constant(SUCCESS_CODE, dtype=tf.int32) # Use name from context
    conditional_tf_print(DEBUG_FLAG,"Initial max n_i:", tf.reduce_max(n_i_current))

    # --- Define Loop Condition ---
    def condition(step, n_i_curr, err_code):
        # conditional_tf_print(DEBUG_FLAG,"Ion Loop Condition Check: step=", step, "steps_tf=", steps_tf, "err_code=", err_code) # Verbose
        return tf.logical_and(step < steps_tf, tf.equal(err_code, SUCCESS_CODE))

    # --- Define Loop Body ---
    def body(step, n_i_curr, err_code):
        conditional_tf_print(DEBUG_FLAG,"\n--- Ion Transport Step:", step + 1, "/", steps_tf, "---")
        n_i_old_step = n_i_curr # Keep track of state at start of step

        conditional_tf_print(DEBUG_FLAG," Step", step, "Max n_i_curr:", tf.reduce_max(n_i_curr))

        # --- Calculate Explicit Terms ---
        conditional_tf_print(DEBUG_FLAG," Calculating explicit terms for ions (Advection, Source)...")
        advection_drift_term_full = calculate_weno_advection_term_tf(
            (vx_eff, vy_eff, vz_eff), n_i_old_step, # Use state at start of step
            dx_tf, dy_tf, dz_scalar_tf, # Use correct dz_scalar_tf
            nx, ny, nz, # Pass Tensor dimensions
            boundary_mode=1 # Assume outflow/Neumann-like BC for WENO padding
        )
        # Clip advection term to prevent extreme values
        advection_drift_term_full = tf.clip_by_value(advection_drift_term_full, -1e15, 1e15)
        # Optional: Check numerics of advection term
        tf.debugging.check_numerics(advection_drift_term_full, "Ion Advection term contains NaN/Inf")

        source_term_full = S_i_tf # Source term is assumed constant for this step
        conditional_tf_print(DEBUG_FLAG," Max advection_drift_term_full:", tf.reduce_max(tf.abs(advection_drift_term_full)))
        conditional_tf_print(DEBUG_FLAG," Max source_term_full:", tf.reduce_max(source_term_full))

        # --- Calculate RHS for Implicit Diffusion ---
        rhs = n_i_old_step + dt_tf * (source_term_full - advection_drift_term_full)
        # Optional: Check numerics of RHS
        tf.debugging.check_numerics(rhs, "Ion RHS contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG," RHS for implicit ion solve computed. Max RHS=", tf.reduce_max(tf.abs(rhs)))

        # --- Solve Implicit Diffusion using Jacobi Iterations ---
        n_i_jacobi_iter = n_i_old_step # Start iteration from value at beginning of step
        # Increase number of Jacobi iterations for stability
        num_jacobi_iters = tf.constant(20, dtype=tf.int32)
        conditional_tf_print(DEBUG_FLAG," Starting Jacobi iterations for ions (", num_jacobi_iters, " iters)...")

        def jacobi_condition_ion(j, n_i_iter):
            return j < num_jacobi_iters

        def jacobi_body_ion(j, n_i_iter):
            # conditional_tf_print(DEBUG_FLAG,"  Ion Jacobi iter", j + 1) # Verbose
            lap_n_i_iter = compute_laplacian_tf(
                n_i_iter, x_1d, y_1d, z_tf,
                coeff_x_ip1, coeff_x_im1, coeff_x_i,
                coeff_y_jp1, coeff_y_jm1, coeff_y_j,
                coeff_z_kp1, coeff_z_km1, coeff_z_k
            )
            # Optional: Check numerics of Laplacian
            tf.debugging.check_numerics(lap_n_i_iter, "Ion Laplacian in Jacobi contains NaN/Inf")
            # conditional_tf_print(DEBUG_FLAG,"  Max Lap(n_i_iter):", tf.reduce_max(tf.abs(lap_n_i_iter))) # Verbose

            n_i_next_jacobi_raw = rhs + dt_tf * D_I_tf * lap_n_i_iter

            # Apply clipping inside Jacobi loop
            n_i_next_jacobi = tf.maximum(n_i_next_jacobi_raw, TF_ZERO) # Non-negativity
            n_i_next_jacobi = tf.minimum(n_i_next_jacobi, MAX_DENSITY_CLIP) # Added upper clip

            # Optional: Check numerics of intermediate Jacobi result
            tf.debugging.check_numerics(n_i_next_jacobi, "n_i_next_jacobi contains NaN/Inf")
            # conditional_tf_print(DEBUG_FLAG,"  Max n_i_next_jacobi (clipped):", tf.reduce_max(n_i_next_jacobi)) # Verbose

            return [j + 1, n_i_next_jacobi] # Ensure list return

        # Execute Jacobi loop
        jacobi_loop_vars = [tf.constant(0, dtype=tf.int32), n_i_jacobi_iter]
        _, n_i_implicit = tf.while_loop(
            jacobi_condition_ion,
            jacobi_body_ion,
            jacobi_loop_vars,
            maximum_iterations=num_jacobi_iters,
            # Specify shape invariants if needed, matching the output shapes
            shape_invariants=[tf.TensorSpec([], tf.int32),
                              tf.TensorSpec([None, None, None], tf.float32)] # Allow dynamic shapes
        )
        # Optional: Check numerics after Jacobi loop
        tf.debugging.check_numerics(n_i_implicit, "n_i_implicit contains NaN/Inf after Jacobi")
        conditional_tf_print(DEBUG_FLAG," Ion Jacobi iterations finished. Max n_i_implicit=", tf.reduce_max(n_i_implicit))

        # --- Post-Processing ---
        # Apply Neumann BC first, then collector BC
        n_i_implicit_neumann = apply_neumann_bc_tf(n_i_implicit)
        n_i_final_step_before_clip = tf.where(collector_mask_tf, TF_ZERO, n_i_implicit_neumann)

        # Apply final clipping and ensure non-negativity again
        n_i_final_step = tf.clip_by_value(n_i_final_step_before_clip, 0.0, MAX_DENSITY_CLIP)

        # Optional: Check numerics of final step result
        tf.debugging.check_numerics(n_i_final_step, "n_i_final_step contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG," Final BCs and clipping applied to ions. Max n_i_final_step=", tf.reduce_max(n_i_final_step))

        # Finiteness Check for this step
        is_finite_step = tf.reduce_all(tf.math.is_finite(n_i_final_step))
        new_err_code = tf.cond(
            is_finite_step,
            lambda: err_code, # Keep existing error code if current step is fine
            lambda: tf.constant(NAN_INF_STEP_CODE, dtype=tf.int32) # Set error if NaN/Inf found
        )
        conditional_tf_print(DEBUG_FLAG," Ion finiteness check: is_finite=", is_finite_step, "new_err_code=", new_err_code)

        # Update n_i for the next step OR keep old value if error occurred
        n_i_next = tf.cond(
            is_finite_step,
            lambda: n_i_final_step,
            lambda: n_i_old_step # Revert to state at start of step if NaN/Inf detected
        )

        return [step + 1, n_i_next, new_err_code] # Ensure list return

    # --- Execute Outer Time Stepping Loop ---
    conditional_tf_print(DEBUG_FLAG,"\nStarting outer time-stepping loop for ions...")
    outer_loop_vars = [step, n_i_current, error_code]
    final_step, n_i_final, final_error_code_loop = tf.while_loop(
        condition,
        body,
        loop_vars=outer_loop_vars,
        maximum_iterations=steps_tf,
        shape_invariants=[
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([None, None, None], tf.float32), # Allow dynamic shapes
            tf.TensorSpec([], tf.int32)
        ]
    )
    conditional_tf_print(DEBUG_FLAG,"\nIon outer loop finished at step:", final_step, "with status:", final_error_code_loop)

    # --- Final Checks ---
    is_finite_final = tf.reduce_all(tf.math.is_finite(n_i_final))
    final_error_code_out = tf.cond(
        tf.equal(final_error_code_loop, SUCCESS_CODE), # If loop finished without step error
        lambda: tf.cond(
            is_finite_final, # Check final result
            lambda: SUCCESS_CODE,
            lambda: NAN_INF_FINAL_CODE # Error if final result is bad
        ),
        lambda: final_error_code_loop # Propagate earlier error (e.g., NAN_INF_STEP_CODE)
    )

    # Print warnings based on final error code
    tf.cond(
        tf.equal(final_error_code_out, NAN_INF_STEP_CODE),
        lambda: conditional_tf_print(DEBUG_FLAG,"WARNING (solve_ion_transport): Loop stopped due to NaN/Inf during a step."),
        lambda: tf.no_op()
    )
    tf.cond(
        tf.equal(final_error_code_out, NAN_INF_FINAL_CODE),
        lambda: conditional_tf_print(DEBUG_FLAG,"ERROR (solve_ion_transport): NaN/Inf in final ion density."),
        lambda: tf.no_op()
    )

    conditional_tf_print(DEBUG_FLAG,"Final ion transport state: Max n_i=", tf.reduce_max(n_i_final), "Final Status Code:", final_error_code_out)
    return n_i_final, final_error_code_out

# --- WENO Advection Helper Functions (Simplified 1D WENO-JS 3rd order for demonstration) ---

# Helper function for flux splitting (e.g., Lax-Friedrichs)
@tf.function
def lax_friedrichs_split_tf(u_comp_in, field_val_in, max_speed_in):
    """
    Calculates Lax-Friedrichs flux splitting for F = u*q using TensorFlow,
    optimized for GPU execution with tf.float32 precision. Includes NaN/Inf checks.

    Args:
        u_comp_in (tf.Tensor or convertible): Velocity component relevant to the
                                              flux direction. Converted to tf.float32.
        field_val_in (tf.Tensor or convertible): Value of the scalar field being
                                                 transported. Converted to tf.float32.
        max_speed_in (tf.Tensor or convertible): Local maximum wave speed
                                                 (e.g., |u| for linear advection).
                                                 Converted to tf.float32.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            - flux_plus_tf (tf.Tensor): Positive split flux (F+), dtype=tf.float32.
            - flux_minus_tf (tf.Tensor): Negative split flux (F-), dtype=tf.float32.
            - error_code (tf.Tensor): Status code (dtype=tf.int32):
                                        0 = Success
                                        3 = NaN/Inf detected in final flux calculations
                                        4 = Unexpected exception during execution
    """
    try:
        # --- Input Conversion (Ensure tf.float32) ---
        # Convert inputs to tf.float32 tensors. Handles scalars, NumPy arrays, lists, etc.
        u_comp = tf.convert_to_tensor(u_comp_in, dtype=tf.float32, name="u_component")
        field_val = tf.convert_to_tensor(field_val_in, dtype=tf.float32, name="field_value")
        max_speed = tf.convert_to_tensor(max_speed_in, dtype=tf.float32, name="max_speed")

        # --- Core Calculation (Using TensorFlow Ops) ---
        # Calculate the physical flux: F = u * q
        # TensorFlow's '*' operator performs element-wise multiplication on tensors.
        flux = tf.multiply(u_comp, field_val, name="physical_flux")

        # Calculate the positive and negative split fluxes using Lax-Friedrichs formula
        # F+ = 0.5 * (F + alpha * q)
        # F- = 0.5 * (F - alpha * q)
        # All operations involve tf.float32 tensors/constants, ensuring tf.float32 output.
        term_alpha_q = tf.multiply(max_speed, field_val, name="alpha_times_q")

        flux_plus_tf = tf.multiply(TF_FLOAT32_HALF,
                                   tf.add(flux, term_alpha_q),
                                   name="flux_plus")
        flux_minus_tf = tf.multiply(TF_FLOAT32_HALF,
                                    tf.subtract(flux, term_alpha_q),
                                    name="flux_minus")

        # --- Error Checking (NaN/Inf in final results) ---
        # Check if ALL elements in BOTH final result tensors are finite.
        is_finite_plus = tf.reduce_all(tf.math.is_finite(flux_plus_tf))
        is_finite_minus = tf.reduce_all(tf.math.is_finite(flux_minus_tf))
        is_finite_result = tf.logical_and(is_finite_plus, is_finite_minus, name="is_finite_check")

        # Determine the error code using tf.cond for graph compatibility.
        error_code = tf.cond(
            is_finite_result,
            lambda: tf.constant(0, dtype=tf.int32, name="SuccessCode"),     # Return 0 if finite
            lambda: tf.constant(3, dtype=tf.int32, name="NaNFinalCode")     # Return 3 if NaN/Inf found
        )

        # --- Conditional Printing with Consistent Outputs ---
        def print_warning():
            conditional_tf_print(DEBUG_FLAG,"WARNING (lax_friedrichs_split_tf): NaN or Inf detected in output fluxes.")
            return tf.constant(0, dtype=tf.int32)

        def no_op():
            return tf.constant(0, dtype=tf.int32)

        _ = tf.cond(
            tf.equal(error_code, 3),
            print_warning,
            no_op
        )

    except Exception as e:
        # Basic exception handling during graph construction/tracing or execution.
        conditional_tf_print(DEBUG_FLAG,"ERROR (lax_friedrichs_split_tf): Unexpected exception during execution:", e, output_stream=sys.stderr)
        # Return tensors indicating failure. Shape might be unknown if inputs failed early.
        # Try to create NaN tensors matching input shape if possible, otherwise scalar NaN.
        try:
            # Infer shape from one of the inputs (assuming they are compatible)
            out_shape = tf.shape(u_comp_in)
        except: # If input shape is not available (e.g., scalar or error during conversion)
             out_shape = [] # Use scalar shape as fallback

        # Create NaN tensors of the determined shape (or scalar if shape unknown)
        nan_tensor = tf.fill(out_shape, tf.constant(np.nan, dtype=tf.float32))
        flux_plus_tf = nan_tensor
        flux_minus_tf = nan_tensor
        error_code = tf.constant(4, dtype=tf.int32, name="ExceptionErrorCode") # Use code 4 for unexpected exceptions

    # Return the computed TensorFlow tensors and the error code
    return flux_plus_tf, flux_minus_tf, error_code

@tf.function
def weno5_reconstruction_tf(f_minus2_in, f_minus1_in, f_0_in, f_plus1_in, f_plus2_in):
    """
    Computes the 5th order WENO reconstruction (WENO-JS) at the interface
    i+1/2 using TensorFlow for GPU acceleration. Operates on tensors.
    This reconstructs the *left* state at the interface using tf.float32 precision.

    Args:
        f_minus2_in (tf.Tensor or convertible): Field values at stencil point i-2.
        f_minus1_in (tf.Tensor or convertible): Field values at stencil point i-1.
        f_0_in (tf.Tensor or convertible): Field values at stencil point i.
        f_plus1_in (tf.Tensor or convertible): Field values at stencil point i+1.
        f_plus2_in (tf.Tensor or convertible): Field values at stencil point i+2.
        Inputs will be cast to tf.float32.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - f_recon_left_tf (tf.Tensor): Reconstructed field values at the left
              side of the i+1/2 interface, dtype=tf.float32. Shape matches input tensors.
            - error_code (tf.Tensor): Status code (dtype=tf.int32):
                                       0 = Success
                                       3 = NaN/Inf detected in final result
    """
    try:
        # --- Input Conversion (Ensure tf.float32) ---
        # Convert inputs to tf.float32 tensors. Handles scalars, NumPy arrays, lists, etc.
        f_minus2 = tf.cast(f_minus2_in, dtype=tf.float32)
        f_minus1 = tf.cast(f_minus1_in, dtype=tf.float32)
        f_0 = tf.cast(f_0_in, dtype=tf.float32)
        f_plus1 = tf.cast(f_plus1_in, dtype=tf.float32)
        f_plus2 = tf.cast(f_plus2_in, dtype=tf.float32)

        # --- Candidate Polynomials (Jiang & Shu (1996) Eq. (2.4)) ---
        # Using TensorFlow operations and pre-defined tf.float32 constants
        p0 = TF_FLOAT32_1_OVER_3*f_minus2 - TF_FLOAT32_7_OVER_6*f_minus1 + TF_FLOAT32_11_OVER_6*f_0
        p1 = TF_FLOAT32_NEG_1_OVER_6*f_minus1 + TF_FLOAT32_5_OVER_6*f_0 + TF_FLOAT32_1_OVER_3*f_plus1
        p2 = TF_FLOAT32_1_OVER_3*f_0 + TF_FLOAT32_5_OVER_6*f_plus1 + TF_FLOAT32_NEG_1_OVER_6*f_plus2 # -(1/6)f_{i+2}

        # --- Smoothness Indicators (Beta factors - Jiang & Shu (1996) Eq. (2.7)) ---
        # Using tf.square which is equivalent to tf.pow(..., 2.0) for float32
        beta0 = TF_FLOAT32_13_OVER_12 * tf.square(f_minus2 - TF_FLOAT32_2_0*f_minus1 + f_0) + \
                TF_FLOAT32_1_OVER_4 * tf.square(f_minus2 - TF_FLOAT32_4_0*f_minus1 + TF_FLOAT32_3_0*f_0)
        beta1 = TF_FLOAT32_13_OVER_12 * tf.square(f_minus1 - TF_FLOAT32_2_0*f_0 + f_plus1) + \
                TF_FLOAT32_1_OVER_4 * tf.square(f_minus1 - f_plus1)
        beta2 = TF_FLOAT32_13_OVER_12 * tf.square(f_0 - TF_FLOAT32_2_0*f_plus1 + f_plus2) + \
                TF_FLOAT32_1_OVER_4 * tf.square(TF_FLOAT32_3_0*f_0 - TF_FLOAT32_4_0*f_plus1 + f_plus2)

        # --- Alpha Weights (Unnormalized nonlinear weights - Jiang & Shu (1996) Eq. (2.8)) ---
        # Relies on global _WENO_D0, _WENO_D1, _WENO_D2, _WENO_EPSILON (tf.float32)
        alpha0_den = tf.square(_WENO_EPSILON + beta0)
        alpha1_den = tf.square(_WENO_EPSILON + beta1)
        alpha2_den = tf.square(_WENO_EPSILON + beta2)

        # Add small stability constant to denominator to prevent 0/0 -> NaN if Dk=0 and den=0
        alpha0 = _WENO_D0 / (alpha0_den + TF_FLOAT32_WENO_STABILITY)
        alpha1 = _WENO_D1 / (alpha1_den + TF_FLOAT32_WENO_STABILITY)
        alpha2 = _WENO_D2 / (alpha2_den + TF_FLOAT32_WENO_STABILITY)

        # --- Nonlinear WENO Weights (Normalized - Jiang & Shu (1996) Eq. (2.9)) ---
        sum_alpha = alpha0 + alpha1 + alpha2

        # Use tf.where for robust normalization, avoiding division by near-zero sum_alpha
        # If sum_alpha is very small, use the linear weights directly.
        inv_sum_alpha_safe = TF_FLOAT32_1_0 / (sum_alpha + TF_FLOAT32_WENO_STABILITY) # Add stability constant

        # Calculate nonlinear weights (will be used unless sum_alpha is too small)
        w0_nonlin = alpha0 * inv_sum_alpha_safe
        w1_nonlin = alpha1 * inv_sum_alpha_safe
        w2_nonlin = alpha2 * inv_sum_alpha_safe

        # Condition to check if sum_alpha is close to zero
        use_linear_weights_cond = tf.abs(sum_alpha) < TF_FLOAT32_WENO_STABILITY

        # Choose weights: linear if sum_alpha is small, nonlinear otherwise
        w0 = tf.where(use_linear_weights_cond, _WENO_D0, w0_nonlin)
        w1 = tf.where(use_linear_weights_cond, _WENO_D1, w1_nonlin)
        w2 = tf.where(use_linear_weights_cond, _WENO_D2, w2_nonlin)

        # --- Reconstructed Value (Jiang & Shu (1996) Eq. (2.6)) ---
        # All inputs are tf.float32, so the result is tf.float32
        f_recon_left_tf = w0 * p0 + w1 * p1 + w2 * p2

        # --- Error Checking (Final Result) ---
        # Check if ALL elements in the final result tensor are finite (not NaN or Inf).
        is_finite_result = tf.reduce_all(tf.math.is_finite(f_recon_left_tf))

        # Determine the error code using tf.cond for graph compatibility.
        error_code = tf.cond(
            is_finite_result,
            lambda: tf.constant(0, dtype=tf.int32, name="SuccessCode"),  # Return 0 if finite
            lambda: tf.constant(3, dtype=tf.int32, name="NaNFinalCode")   # Return 3 if NaN/Inf found
        )

        # Optional: Add a tf.print for debugging within the graph execution if NaN/Inf occurs.
        def print_error():
            conditional_tf_print(DEBUG_FLAG,"ERROR (weno5_reconstruction_tf): NaN or Inf detected in final reconstructed value.")
            return tf.constant(0, dtype=tf.int32)

        def no_op():
            return tf.constant(0, dtype=tf.int32)

        _ = tf.cond(
            tf.equal(error_code, 3),
            print_error,
            no_op
        )

    except Exception as e:
        # Basic exception handling during graph construction/tracing or execution.
        conditional_tf_print(DEBUG_FLAG,"ERROR (weno5_reconstruction_tf): Unexpected exception during execution:", e, output_stream=sys.stderr)
        # Attempt to return NaN tensor of appropriate shape (if inputs allow shape inference)
        try:
            out_shape = tf.shape(f_0_in) # Infer shape from one input
        except:
            out_shape = [] # Fallback to scalar shape
        f_recon_left_tf = tf.fill(out_shape, tf.constant(np.nan, dtype=tf.float32))
        # Use a different error code for unexpected exceptions if desired, e.g., 4
        error_code = tf.constant(4, dtype=tf.int32, name="ExceptionErrorCode")

    # Return the reconstructed tensor and the error code tensor
    return f_recon_left_tf, error_code

@tf.function
def weno5_reconstruction_right_tf(f_minus1_in, f_0_in, f_plus1_in, f_plus2_in, f_plus3_in):
    """
    Computes the 5th order WENO reconstruction (WENO-JS) for the *right* state
    at the interface i+1/2 using values from cell centers i-1, i, i+1, i+2, i+3.
    Implemented using TensorFlow for GPU acceleration with tf.float32 precision.

    Uses the standard Jiang-Shu (1996) formulation and weights, adapted for the
    right interface reconstruction. Relies on global TF constants for WENO parameters.

    Args:
        f_minus1_in (tf.Tensor or convertible): Field values at stencil point i-1.
        f_0_in (tf.Tensor or convertible): Field values at stencil point i.
        f_plus1_in (tf.Tensor or convertible): Field values at stencil point i+1.
        f_plus2_in (tf.Tensor or convertible): Field values at stencil point i+2.
        f_plus3_in (tf.Tensor or convertible): Field values at stencil point i+3.
                                                Inputs will be cast to tf.float32.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - f_recon_right_tf (tf.Tensor): Reconstructed field values at the right
                                           side of the i+1/2 interface, dtype=tf.float32.
                                           Shape matches input tensors.
            - error_code (tf.Tensor): Status code (dtype=tf.int32):
                                        0 = Success
                                        3 = NaN/Inf detected in final result
                                        4 = Unexpected exception during execution
    """
    try:
        # --- Input Conversion (Ensure tf.float32) ---
        f_minus1 = tf.cast(f_minus1_in, dtype=tf.float32)
        f_0      = tf.cast(f_0_in,      dtype=tf.float32)
        f_plus1  = tf.cast(f_plus1_in,  dtype=tf.float32)
        f_plus2  = tf.cast(f_plus2_in,  dtype=tf.float32)
        f_plus3  = tf.cast(f_plus3_in,  dtype=tf.float32)

        # --- Candidate Polynomials (Right Interface Reconstruction) ---
        # Stencil i-1, i, i+1 -> p0_r
        p0_r = TF_FLOAT32_NEG_1_OVER_6 * f_minus1 + TF_FLOAT32_5_OVER_6 * f_0   + TF_FLOAT32_1_OVER_3 * f_plus1
        # Stencil i, i+1, i+2 -> p1_r
        p1_r = TF_FLOAT32_1_OVER_3 * f_0      + TF_FLOAT32_5_OVER_6 * f_plus1 - TF_FLOAT32_NEG_1_OVER_6 * f_plus2
        # Stencil i+1, i+2, i+3 -> p2_r
        p2_r = TF_FLOAT32_11_OVER_6 * f_plus1 + TF_FLOAT32_NEG_7_OVER_6 * f_plus2 + TF_FLOAT32_1_OVER_3 * f_plus3

        # --- Smoothness Indicators (Beta factors - Right Interface) ---
        # Using tf.square for element-wise squaring.
        # beta0_r (stencil i-1, i, i+1)
        beta0_r = TF_FLOAT32_13_OVER_12 * tf.square(f_minus1 - TF_FLOAT32_TWO * f_0 + f_plus1) + \
                  TF_FLOAT32_1_OVER_4 * tf.square(f_minus1 - f_plus1)
        # beta1_r (stencil i, i+1, i+2)
        beta1_r = TF_FLOAT32_13_OVER_12 * tf.square(f_0 - TF_FLOAT32_TWO * f_plus1 + f_plus2) + \
                  TF_FLOAT32_1_OVER_4 * tf.square(f_0 - TF_FLOAT32_FOUR * f_plus1 + TF_FLOAT32_THREE * f_plus2)
        # beta2_r (stencil i+1, i+2, i+3)
        beta2_r = TF_FLOAT32_13_OVER_12 * tf.square(f_plus1 - TF_FLOAT32_TWO * f_plus2 + f_plus3) + \
                  TF_FLOAT32_1_OVER_4 * tf.square(TF_FLOAT32_THREE * f_plus1 - TF_FLOAT32_FOUR * f_plus2 + f_plus3)

        # --- Alpha Weights (Unnormalized Nonlinear Weights - Right Interface) ---
        # Using symmetric linear weights: d0_r=0.3 (_WENO_D2), d1_r=0.6 (_WENO_D1), d2_r=0.1 (_WENO_D0)
        alpha0_den_r = tf.square(_WENO_EPSILON + beta0_r) + TF_FLOAT32_WENO_STABILITY
        alpha1_den_r = tf.square(_WENO_EPSILON + beta1_r) + TF_FLOAT32_WENO_STABILITY
        alpha2_den_r = tf.square(_WENO_EPSILON + beta2_r) + TF_FLOAT32_WENO_STABILITY

        alpha0_r = _WENO_D2 / alpha0_den_r
        alpha1_r = _WENO_D1 / alpha1_den_r
        alpha2_r = _WENO_D0 / alpha2_den_r

        # --- Nonlinear WENO Weights (Normalized - Right Interface) ---
        sum_alpha_r = alpha0_r + alpha1_r + alpha2_r

        # Use tf.where for robust normalization, avoiding division by near-zero sum_alpha_r
        inv_sum_alpha_r_safe = TF_FLOAT32_ONE / (sum_alpha_r + TF_FLOAT32_WENO_STABILITY)

        # Calculate nonlinear weights
        w0_nonlin_r = alpha0_r * inv_sum_alpha_r_safe
        w1_nonlin_r = alpha1_r * inv_sum_alpha_r_safe
        w2_nonlin_r = alpha2_r * inv_sum_alpha_r_safe

        # Condition to check if sum_alpha_r is close to zero
        use_linear_weights_cond_r = tf.abs(sum_alpha_r) < TF_FLOAT32_WENO_STABILITY

        # Choose weights: linear if sum_alpha_r is small, nonlinear otherwise
        w0_r = tf.where(use_linear_weights_cond_r, _WENO_D2, w0_nonlin_r)
        w1_r = tf.where(use_linear_weights_cond_r, _WENO_D1, w1_nonlin_r)
        w2_r = tf.where(use_linear_weights_cond_r, _WENO_D0, w2_nonlin_r)

        # --- Reconstructed Value ---
        # Combine the candidate polynomials using the chosen nonlinear weights.
        f_recon_right_tf = w0_r * p0_r + w1_r * p1_r + w2_r * p2_r

        # --- Error Checking (Final Result) ---
        is_finite_result = tf.reduce_all(tf.math.is_finite(f_recon_right_tf))

        error_code = tf.cond(
            is_finite_result,
            lambda: tf.constant(0, dtype=tf.int32, name="SuccessCode"),
            lambda: tf.constant(3, dtype=tf.int32, name="NaNFinalCode")
        )

        # Optional: Print warning if NaN/Inf occurs
        def print_error():
            conditional_tf_print(DEBUG_FLAG,"ERROR (weno5_reconstruction_right_tf): NaN or Inf detected in final reconstructed value.")
            return tf.constant(0, dtype=tf.int32)

        def no_op():
            return tf.constant(0, dtype=tf.int32)

        _ = tf.cond(
            tf.equal(error_code, 3),
            print_error,
            no_op
        )

    except Exception as e:
        # Handle exceptions during graph construction or execution
        conditional_tf_print(DEBUG_FLAG,"ERROR (weno5_reconstruction_right_tf): Unexpected exception:", e, output_stream=sys.stderr)
        try:
            out_shape = tf.shape(f_0_in)
        except:
            out_shape = []
        f_recon_right_tf = tf.fill(out_shape, tf.constant(np.nan, dtype=tf.float32))
        error_code = tf.constant(4, dtype=tf.int32, name="ExceptionErrorCode")

    return f_recon_right_tf, error_code

# Apply tf.function decorator for potential graph optimization
@tf.function
def calculate_weno_advection_term_tf(u_vel_tuple, field, dx, dy, dz, nx, ny, nz, boundary_mode=1):
    """
    TensorFlow-based WENO5 advection term computation (∇·(u * field)) using
    Lax-Friedrichs flux splitting and Jiang-Shu reconstruction. Optimized for GPU
    and ensures float32 precision for TensorFlow operations.

    Args:
        u_vel_tuple: Tuple of velocity components (ux, uy, uz) as numpy arrays
                     (will be converted to tf.float32).
        field: Scalar field to advect (numpy array, will be converted to tf.float32).
        dx, dy, dz: Grid spacings (tf.Tensor with dtype=tf.float32).
        nx, ny, nz: Grid dimensions (int). Not directly used in TF computations here,
                    but potentially useful for context or slicing elsewhere.
        boundary_mode: Integer specifying boundary conditions.
                       1: Outflow ('SYMMETRIC' padding - Neumann-like).
                       0: Periodic ('CONSTANT' padding - requires careful setup or manual wrap).
                          Note: TF 'CONSTANT' pads with 0, not wrap. Manual periodic
                          padding before calling might be needed for true periodicity.

    Returns:
        Advection term (tf.Tensor) with dtype float32 and the same shape
        as the input field (excluding padding).
    """
    # Constants (defined explicitly as tf.float32)
    TF_FLOAT32_ZERO = tf.constant(0.0, dtype=tf.float32)
    TF_FLOAT32_0_5 = tf.constant(0.5, dtype=tf.float32)
    TF_FLOAT32_1 = tf.constant(1.0, dtype=tf.float32)
    TF_FLOAT32_2 = tf.constant(2.0, dtype=tf.float32)
    TF_FLOAT32_3 = tf.constant(3.0, dtype=tf.float32)
    TF_FLOAT32_4 = tf.constant(4.0, dtype=tf.float32)
    TF_FLOAT32_1_OVER_3 = tf.constant(1.0/3.0, dtype=tf.float32)
    TF_FLOAT32_7_OVER_6 = tf.constant(7.0/6.0, dtype=tf.float32)
    TF_FLOAT32_11_OVER_6 = tf.constant(11.0/6.0, dtype=tf.float32)
    TF_FLOAT32_NEG_1_OVER_6 = tf.constant(-1.0/6.0, dtype=tf.float32)
    TF_FLOAT32_5_OVER_6 = tf.constant(5.0/6.0, dtype=tf.float32)
    TF_FLOAT32_13_OVER_12 = tf.constant(13.0/12.0, dtype=tf.float32)
    TF_FLOAT32_1_OVER_4 = tf.constant(1.0/4.0, dtype=tf.float32)
    TF_FLOAT32_WENO_STABILITY = tf.constant(1e-40, dtype=tf.float32)

    # Convert inputs to TensorFlow tensors with float32 dtype
    ux_tf = tf.convert_to_tensor(u_vel_tuple[0], dtype=tf.float32)
    uy_tf = tf.convert_to_tensor(u_vel_tuple[1], dtype=tf.float32)
    uz_tf = tf.convert_to_tensor(u_vel_tuple[2], dtype=tf.float32)
    field_tf = tf.convert_to_tensor(field, dtype=tf.float32)
    deltas_tf = [dx, dy, dz]

    # Padding for WENO5 (3 cells on each side)
    pad_width = 3
    paddings = [[pad_width, pad_width], [pad_width, pad_width], [pad_width, pad_width]]
    pad_mode = 'SYMMETRIC' if boundary_mode == 1 else 'CONSTANT'

    # Apply padding
    field_padded = tf.pad(field_tf, paddings, mode=pad_mode)
    ux_padded = tf.pad(ux_tf, paddings, mode=pad_mode)
    uy_padded = tf.pad(uy_tf, paddings, mode=pad_mode)
    uz_padded = tf.pad(uz_tf, paddings, mode=pad_mode)

    # Initialize advection term
    adv_term_tf = tf.zeros_like(field_tf, dtype=tf.float32)

    # Compute fluxes for each dimension
    velocity_components_padded = [ux_padded, uy_padded, uz_padded]
    for axis, delta_tf in enumerate(deltas_tf):
        u_padded_comp = velocity_components_padded[axis]

        # WENO5 reconstruction (fused left state at i+1/2)
        q_m2 = tf.roll(field_padded, shift=2, axis=axis)
        q_m1 = tf.roll(field_padded, shift=1, axis=axis)
        q_0 = field_padded
        q_p1 = tf.roll(field_padded, shift=-1, axis=axis)
        q_p2 = tf.roll(field_padded, shift=-2, axis=axis)

        # Fused candidate polynomials and smoothness indicators
        p0 = TF_FLOAT32_1_OVER_3 * q_m2 - TF_FLOAT32_7_OVER_6 * q_m1 + TF_FLOAT32_11_OVER_6 * q_0
        p1 = TF_FLOAT32_NEG_1_OVER_6 * q_m1 + TF_FLOAT32_5_OVER_6 * q_0 + TF_FLOAT32_1_OVER_3 * q_p1
        p2 = TF_FLOAT32_1_OVER_3 * q_0 + TF_FLOAT32_5_OVER_6 * q_p1 + TF_FLOAT32_NEG_1_OVER_6 * q_p2

        diff0 = q_m2 - TF_FLOAT32_2 * q_m1 + q_0
        diff1 = q_m1 - TF_FLOAT32_2 * q_0 + q_p1
        diff2 = q_0 - TF_FLOAT32_2 * q_p1 + q_p2
        beta0 = TF_FLOAT32_13_OVER_12 * tf.square(diff0) + TF_FLOAT32_1_OVER_4 * tf.square(q_m2 - TF_FLOAT32_4 * q_m1 + TF_FLOAT32_3 * q_0)
        beta1 = TF_FLOAT32_13_OVER_12 * tf.square(diff1) + TF_FLOAT32_1_OVER_4 * tf.square(q_m1 - q_p1)
        beta2 = TF_FLOAT32_13_OVER_12 * tf.square(diff2) + TF_FLOAT32_1_OVER_4 * tf.square(TF_FLOAT32_3 * q_0 - TF_FLOAT32_4 * q_p1 + q_p2)

        # Fused weight calculation
        alpha0 = _WENO_D0 / (tf.square(_WENO_EPSILON + beta0) + TF_FLOAT32_WENO_STABILITY)
        alpha1 = _WENO_D1 / (tf.square(_WENO_EPSILON + beta1) + TF_FLOAT32_WENO_STABILITY)
        alpha2 = _WENO_D2 / (tf.square(_WENO_EPSILON + beta2) + TF_FLOAT32_WENO_STABILITY)
        sum_alpha = alpha0 + alpha1 + alpha2
        inv_sum_alpha = TF_FLOAT32_1 / (sum_alpha + TF_FLOAT32_WENO_STABILITY)
        q_left_iph = (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) * inv_sum_alpha

        # Right state by rolling
        q_right_iph = tf.roll(q_left_iph, shift=-1, axis=axis)

        # Fused Lax-Friedrichs flux calculation
        u_left_iph = u_padded_comp
        u_right_iph = tf.roll(u_padded_comp, shift=-1, axis=axis)
        flux_phys = u_left_iph * q_left_iph + u_right_iph * q_right_iph
        alpha = tf.maximum(tf.abs(u_left_iph), tf.abs(u_right_iph))
        flux_iph = TF_FLOAT32_0_5 * (flux_phys - alpha * (q_right_iph - q_left_iph))

        # Compute divergence
        flux_imh = tf.roll(flux_iph, shift=1, axis=axis)
        flux_diff_padded = (flux_iph - flux_imh) / delta_tf
        adv_term_tf += flux_diff_padded[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

    return adv_term_tf

# ==================================
# TensorFlow Helper Functions for NS Solver
# ==================================

@tf.function
def apply_neumann_bc_tf(field_tf):
    """Applies zero-gradient (Neumann) boundary conditions using padding and slicing."""
    # Ensure field_tf is float32
    field_tf = tf.cast(field_tf, dtype=tf.float32)

    # Apply symmetric padding to enforce Neumann BCs (zero gradient) at boundaries
    # Pad one layer on each side, reflecting the adjacent interior values
    field_tf = tf.pad(field_tf[1:-1, :, :], [[1, 1], [0, 0], [0, 0]], mode='SYMMETRIC')  # x boundaries
    field_tf = tf.pad(field_tf[:, 1:-1, :], [[0, 0], [1, 1], [0, 0]], mode='SYMMETRIC')  # y boundaries
    field_tf = tf.pad(field_tf[:, :, 1:-1], [[0, 0], [0, 0], [1, 1]], mode='SYMMETRIC')  # z boundaries

    return field_tf

@tf.function
def compute_gradient_tf(scalar_field, x, y, z, field_name="Scalar"):
    """
    Computes the gradient of a scalar field on a non-uniform grid using
    second-order finite differences. Handles variable spacing in x, y, and z.
    Uses second-order central differences for the interior and second-order
    forward/backward differences at the boundaries.

    Modified to remove smoothing to preserve peak gradients and added logging
    for raw gradient values before clipping to aid debugging. Clipping limits
    remain at ±1e10 V/m to handle high voltages up to 500,000 V.

    Args:
        scalar_field (tf.Tensor): The 3D scalar field tensor [nx, ny, nz], dtype=tf.float32.
        x (tf.Tensor): 1D tensor of x-coordinates [nx], dtype=tf.float32.
        y (tf.Tensor): 1D tensor of y-coordinates [ny], dtype=tf.float32.
        z (tf.Tensor): 1D tensor of z-coordinates [nz], dtype=tf.float32.
        field_name (str): Name of the field for debugging purposes (default: "Scalar").

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Gradient components (grad_x, grad_y, grad_z),
                                                each with shape [nx, ny, nz] and dtype=tf.float32.
    """
    conditional_tf_print(DEBUG_FLAG,"Starting non-uniform gradient computation for:", field_name, ". Field shape:", tf.shape(scalar_field))

    # --- Input Validation and Casting ---
    scalar_field = tf.cast(scalar_field, dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)

    nx, ny, nz = tf.shape(scalar_field)[0], tf.shape(scalar_field)[1], tf.shape(scalar_field)[2]
    conditional_tf_print(DEBUG_FLAG,"Grid dimensions: nx=", nx, "ny=", ny, "nz=", nz)

    # Stability constant
    TF_EPSILON_GRAD = tf.constant(1e-6, dtype=tf.float32)

    # --- Initialize Gradient Tensors ---
    grad_x = tf.zeros_like(scalar_field, dtype=tf.float32)
    grad_y = tf.zeros_like(scalar_field, dtype=tf.float32)
    grad_z = tf.zeros_like(scalar_field, dtype=tf.float32)

    # --- X Gradient ---
    if nx > 1:
        conditional_tf_print(DEBUG_FLAG,"Computing grad_x...")
        dx = x[1:] - x[:-1]  # Shape [nx-1]
        safe_dx = tf.maximum(dx, TF_EPSILON_GRAD)  # Shape [nx-1]

        # Pad scalar_field for full-grid computation (accessing i-2 to i+2)
        phi_padded = tf.pad(scalar_field, [[2, 2], [0, 0], [0, 0]], mode='SYMMETRIC')
        phi_im2 = phi_padded[:-4, :, :]  # Shape [nx, ny, nz], aligned with node i
        phi_im1 = phi_padded[1:-3, :, :]  # Shape [nx, ny, nz], aligned with node i
        phi_i   = phi_padded[2:-2, :, :]  # Shape [nx, ny, nz], aligned with node i
        phi_ip1 = phi_padded[3:-1, :, :]  # Shape [nx, ny, nz], aligned with node i
        phi_ip2 = phi_padded[4:, :, :]    # Shape [nx, ny, nz], aligned with node i

        # Spacing arrays h_L (left) and h_R (right) of shape [nx] aligned with nodes i=0..nx-1
        h_L = tf.pad(safe_dx, [[1, 0]], mode='SYMMETRIC')  # Shape [nx]
        h_R = tf.pad(safe_dx, [[0, 1]], mode='SYMMETRIC')  # Shape [nx]
        h_S = h_L + h_R  # Shape [nx]

        # Reshape to [nx, 1, 1] for proper broadcasting with 3D tensors
        h_L_3d = h_L[:, None, None]  # Shape [nx, 1, 1]
        h_R_3d = h_R[:, None, None]  # Shape [nx, 1, 1]
        h_S_3d = h_S[:, None, None]  # Shape [nx, 1, 1]

        # Compute gradient across the full grid (interior formula) with broadcasting
        numerator = (phi_ip1 * tf.square(h_L_3d) -
                     phi_im1 * tf.square(h_R_3d) +
                     phi_i * (tf.square(h_R_3d) - tf.square(h_L_3d)))
        denominator = h_L_3d * h_R_3d * h_S_3d + TF_EPSILON_GRAD
        grad_x_full = numerator / denominator  # Shape [nx, ny, nz]

        # Enhanced logging for raw gradient
        conditional_tf_print(DEBUG_FLAG,"DEBUG: Raw grad_x for", field_name, "- min:", tf.reduce_min(grad_x_full),
                 "max:", tf.reduce_max(grad_x_full), "finite:", tf.reduce_all(tf.math.is_finite(grad_x_full)))
        tf.debugging.check_numerics(grad_x_full, "Raw grad_x contains NaN/Inf")

        # Clip raw gradient to avoid extreme values before boundary conditions
        clip_min_raw = tf.constant(-1e10, dtype=tf.float32)
        clip_max_raw = tf.constant(1e10, dtype=tf.float32)
        grad_x_full = tf.clip_by_value(grad_x_full, clip_min_raw, clip_max_raw)

        # Define boundary conditions
        is_i0 = tf.range(nx)[:, None, None] == 0  # Shape [nx, 1, 1]
        is_iN = tf.range(nx)[:, None, None] == nx - 1  # Shape [nx, 1, 1]

        # Extract scalar values for boundary conditions
        dx0 = h_R[0]   # Scalar, dx[0] = x[1]-x[0]
        dx1 = h_R[1]   # Scalar, dx[1] = x[2]-x[1]
        dxn1 = h_L[nx-1]  # Scalar, dx[nx-2] = x[nx-1]-x[nx-2]
        dxn2 = h_L[nx-2]  # Scalar, dx[nx-3] = x[nx-2]-x[nx-3]

        # Compute boundary formulas (2nd order forward/backward)
        if nx > 2:
            # Forward difference at i=0
            grad_x_i0 = (-phi_ip2 * dx0**2 +
                         phi_ip1 * (dx0 + dx1)**2 -
                         phi_i * ((dx0 + dx1)**2 - dx0**2)) / (dx0 * dx1 * (dx0 + dx1) + TF_EPSILON_GRAD)
            # Backward difference at i=N
            grad_x_iN = (phi_im2 * dxn1**2 -
                         phi_im1 * (dxn1 + dxn2)**2 +
                         phi_i * ((dxn1 + dxn2)**2 - dxn1**2)) / (dxn1 * dxn2 * (dxn1 + dxn2) + TF_EPSILON_GRAD)
            grad_x_i0 = tf.clip_by_value(grad_x_i0, clip_min_raw, clip_max_raw)
            grad_x_iN = tf.clip_by_value(grad_x_iN, clip_min_raw, clip_max_raw)
        elif nx == 2:
            # First order difference if only 2 points
            grad_x_i0 = (phi_ip1 - phi_i) / (h_R[0] + TF_EPSILON_GRAD)
            grad_x_iN = (phi_i - phi_im1) / (h_L[1] + TF_EPSILON_GRAD)
            grad_x_i0 = tf.clip_by_value(grad_x_i0, clip_min_raw, clip_max_raw)
            grad_x_iN = tf.clip_by_value(grad_x_iN, clip_min_raw, clip_max_raw)
        else:  # nx == 1
            grad_x_i0 = tf.zeros_like(phi_i)
            grad_x_iN = tf.zeros_like(phi_i)

        # Apply boundary formulas with tf.where
        grad_x = tf.where(
            is_i0,
            grad_x_i0,
            tf.where(
                is_iN,
                grad_x_iN,
                grad_x_full
            )
        )
        conditional_tf_print(DEBUG_FLAG,"grad_x computed (after BCs).")
        conditional_tf_print(DEBUG_FLAG,"DEBUG: grad_x after BCs - min:", tf.reduce_min(grad_x),
                 "max:", tf.reduce_max(grad_x), "finite:", tf.reduce_all(tf.math.is_finite(grad_x)))
    else:
        conditional_tf_print(DEBUG_FLAG,"Skipping grad_x (nx=1).")

    # --- Y Gradient ---
    if ny > 1:
        conditional_tf_print(DEBUG_FLAG,"Computing grad_y...")
        dy = y[1:] - y[:-1]
        safe_dy = tf.maximum(dy, TF_EPSILON_GRAD)  # Shape [ny-1]

        phi_padded = tf.pad(scalar_field, [[0, 0], [2, 2], [0, 0]], mode='SYMMETRIC')
        phi_jm2 = phi_padded[:, :-4, :]
        phi_jm1 = phi_padded[:, 1:-3, :]
        phi_j   = phi_padded[:, 2:-2, :]
        phi_jp1 = phi_padded[:, 3:-1, :]
        phi_jp2 = phi_padded[:, 4:, :]

        # Spacing arrays
        h_L_y = tf.pad(safe_dy, [[1, 0]], mode='SYMMETRIC')  # Shape [ny]
        h_R_y = tf.pad(safe_dy, [[0, 1]], mode='SYMMETRIC')  # Shape [ny]
        h_S_y = h_L_y + h_R_y  # Shape [ny]

        h_L_y_3d = h_L_y[None, :, None]  # Shape [1, ny, 1]
        h_R_y_3d = h_R_y[None, :, None]  # Shape [1, ny, 1]
        h_S_y_3d = h_S_y[None, :, None]  # Shape [1, ny, 1]

        numerator = (phi_jp1 * tf.square(h_L_y_3d) -
                     phi_jm1 * tf.square(h_R_y_3d) +
                     phi_j * (tf.square(h_R_y_3d) - tf.square(h_L_y_3d)))
        denominator = h_L_y_3d * h_R_y_3d * h_S_y_3d + TF_EPSILON_GRAD
        grad_y_full = numerator / denominator

        conditional_tf_print(DEBUG_FLAG,"DEBUG: Raw grad_y for", field_name, "- min:", tf.reduce_min(grad_y_full),
                 "max:", tf.reduce_max(grad_y_full), "finite:", tf.reduce_all(tf.math.is_finite(grad_y_full)))
        tf.debugging.check_numerics(grad_y_full, "Raw grad_y contains NaN/Inf")

        grad_y_full = tf.clip_by_value(grad_y_full, clip_min_raw, clip_max_raw)

        is_j0 = tf.range(ny)[None, :, None] == 0
        is_jN = tf.range(ny)[None, :, None] == ny - 1

        dy0 = h_R_y[0]
        dy1 = h_R_y[1]
        dyn1 = h_L_y[ny-1]
        dyn2 = h_L_y[ny-2]

        if ny > 2:
            grad_y_j0 = (-phi_jp2 * dy0**2 +
                         phi_jp1 * (dy0 + dy1)**2 -
                         phi_j * ((dy0 + dy1)**2 - dy0**2)) / (dy0 * dy1 * (dy0 + dy1) + TF_EPSILON_GRAD)
            grad_y_jN = (phi_jm2 * dyn1**2 -
                         phi_jm1 * (dyn1 + dyn2)**2 +
                         phi_j * ((dyn1 + dyn2)**2 - dyn1**2)) / (dyn1 * dy1 * (dyn1 + dyn2) + TF_EPSILON_GRAD)
            grad_y_j0 = tf.clip_by_value(grad_y_j0, clip_min_raw, clip_max_raw)
            grad_y_jN = tf.clip_by_value(grad_y_jN, clip_min_raw, clip_max_raw)
        elif ny == 2:
            grad_y_j0 = (phi_jp1 - phi_j) / (h_R_y[0] + TF_EPSILON_GRAD)
            grad_y_jN = (phi_j - phi_jm1) / (h_L_y[1] + TF_EPSILON_GRAD)
            grad_y_j0 = tf.clip_by_value(grad_y_j0, clip_min_raw, clip_max_raw)
            grad_y_jN = tf.clip_by_value(grad_y_jN, clip_min_raw, clip_max_raw)
        else:  # ny == 1
            grad_y_j0 = tf.zeros_like(phi_j)
            grad_y_jN = tf.zeros_like(phi_j)

        grad_y = tf.where(is_j0, grad_y_j0, tf.where(is_jN, grad_y_jN, grad_y_full))
        conditional_tf_print(DEBUG_FLAG,"grad_y computed (after BCs).")
        conditional_tf_print(DEBUG_FLAG,"DEBUG: grad_y after BCs - min:", tf.reduce_min(grad_y),
                 "max:", tf.reduce_max(grad_y), "finite:", tf.reduce_all(tf.math.is_finite(grad_y)))
    else:
        conditional_tf_print(DEBUG_FLAG,"Skipping grad_y (ny=1).")

    # --- Z Gradient ---
    if nz > 1:
        conditional_tf_print(DEBUG_FLAG,"Computing grad_z...")
        dz = z[1:] - z[:-1]
        safe_dz = tf.maximum(dz, TF_EPSILON_GRAD)  # Shape [nz-1]

        phi_padded = tf.pad(scalar_field, [[0, 0], [0, 0], [2, 2]], mode='SYMMETRIC')
        phi_km2 = phi_padded[:, :, :-4]
        phi_km1 = phi_padded[:, :, 1:-3]
        phi_k   = phi_padded[:, :, 2:-2]
        phi_kp1 = phi_padded[:, :, 3:-1]
        phi_kp2 = phi_padded[:, :, 4:]

        # Spacing arrays
        h_L_z = tf.pad(safe_dz, [[1, 0]], mode='SYMMETRIC')  # Shape [nz]
        h_R_z = tf.pad(safe_dz, [[0, 1]], mode='SYMMETRIC')  # Shape [nz]
        h_S_z = h_L_z + h_R_z  # Shape [nz]

        h_L_z_3d = h_L_z[None, None, :]  # Shape [1, 1, nz]
        h_R_z_3d = h_R_z[None, None, :]  # Shape [1, 1, nz]
        h_S_z_3d = h_S_z[None, None, :]  # Shape [1, 1, nz]

        numerator = (phi_kp1 * tf.square(h_L_z_3d) -
                     phi_km1 * tf.square(h_R_z_3d) +
                     phi_k * (tf.square(h_R_z_3d) - tf.square(h_L_z_3d)))
        denominator = h_L_z_3d * h_R_z_3d * h_S_z_3d + TF_EPSILON_GRAD
        grad_z_full = numerator / denominator

        conditional_tf_print(DEBUG_FLAG,"DEBUG: Raw grad_z for", field_name, "- min:", tf.reduce_min(grad_z_full),
                 "max:", tf.reduce_max(grad_z_full), "finite:", tf.reduce_all(tf.math.is_finite(grad_z_full)))
        tf.debugging.check_numerics(grad_z_full, "Raw grad_z contains NaN/Inf")

        grad_z_full = tf.clip_by_value(grad_z_full, clip_min_raw, clip_max_raw)

        is_k0 = tf.range(nz)[None, None, :] == 0
        is_kN = tf.range(nz)[None, None, :] == nz - 1

        dz0 = h_R_z[0]
        dz1 = h_R_z[1]
        dzn1 = h_L_z[nz-1]
        dzn2 = h_L_z[nz-2]

        if nz > 2:
            grad_z_k0 = (-phi_kp2 * dz0**2 +
                         phi_kp1 * (dz0 + dz1)**2 -
                         phi_k * ((dz0 + dz1)**2 - dz0**2)) / (dz0 * dz1 * (dz0 + dz1) + TF_EPSILON_GRAD)
            grad_z_kN = (phi_km2 * dzn1**2 -
                         phi_km1 * (dzn1 + dzn2)**2 +
                         phi_k * ((dzn1 + dzn2)**2 - dzn1**2)) / (dzn1 * dz1 * (dzn1 + dzn2) + TF_EPSILON_GRAD)
            grad_z_k0 = tf.clip_by_value(grad_z_k0, clip_min_raw, clip_max_raw)
            grad_z_kN = tf.clip_by_value(grad_z_kN, clip_min_raw, clip_max_raw)
        elif nz == 2:
            grad_z_k0 = (phi_kp1 - phi_k) / (h_R_z[0] + TF_EPSILON_GRAD)
            grad_z_kN = (phi_k - phi_km1) / (h_L_z[1] + TF_EPSILON_GRAD)
            grad_z_k0 = tf.clip_by_value(grad_z_k0, clip_min_raw, clip_max_raw)
            grad_z_kN = tf.clip_by_value(grad_z_kN, clip_min_raw, clip_max_raw)
        else:  # nz == 1
            grad_z_k0 = tf.zeros_like(phi_k)
            grad_z_kN = tf.zeros_like(phi_k)

        grad_z = tf.where(is_k0, grad_z_k0, tf.where(is_kN, grad_z_kN, grad_z_full))
        conditional_tf_print(DEBUG_FLAG,"grad_z computed (after BCs).")
        conditional_tf_print(DEBUG_FLAG,"DEBUG: grad_z after BCs - min:", tf.reduce_min(grad_z),
                 "max:", tf.reduce_max(grad_z), "finite:", tf.reduce_all(tf.math.is_finite(grad_z)))
    else:
        conditional_tf_print(DEBUG_FLAG,"Skipping grad_z (nz=1).")

    # --- Final Clipping ---
    conditional_tf_print(DEBUG_FLAG,"Applying final clipping...")
    clip_min_final = tf.constant(-1e10, dtype=tf.float32)
    clip_max_final = tf.constant(1e10, dtype=tf.float32)

    grad_x = tf.clip_by_value(grad_x, clip_min_final, clip_max_final, name="clip_grad_x_final")
    grad_y = tf.clip_by_value(grad_y, clip_min_final, clip_max_final, name="clip_grad_y_final")
    grad_z = tf.clip_by_value(grad_z, clip_min_final, clip_max_final, name="clip_grad_z_final")
    conditional_tf_print(DEBUG_FLAG,"Final clipping applied.")

    # Final check for NaNs with debug prints
    conditional_tf_print(DEBUG_FLAG,"Final grad_x: min=", tf.reduce_min(grad_x), "max=", tf.reduce_max(grad_x),
             "finite=", tf.reduce_all(tf.math.is_finite(grad_x)))
    conditional_tf_print(DEBUG_FLAG,"Final grad_y: min=", tf.reduce_min(grad_y), "max=", tf.reduce_max(grad_y),
             "finite=", tf.reduce_all(tf.math.is_finite(grad_y)))
    conditional_tf_print(DEBUG_FLAG,"Final grad_z: min=", tf.reduce_min(grad_z), "max=", tf.reduce_max(grad_z),
             "finite=", tf.reduce_all(tf.math.is_finite(grad_z)))

    # Numerical checks
    tf.debugging.check_numerics(grad_x, "Grad_x contains NaN/Inf after final processing")
    tf.debugging.check_numerics(grad_y, "Grad_y contains NaN/Inf after final processing")
    tf.debugging.check_numerics(grad_z, "Grad_z contains NaN/Inf after final processing")

    conditional_tf_print(DEBUG_FLAG,"Gradient computation finished.")
    return grad_x, grad_y, grad_z

@tf.function
def compute_divergence_nonuniform_tf(ux, uy, uz, x, y, z):
    """
    Computes the divergence of a vector field on a non-uniform grid using TensorFlow.
    
    Args:
        ux (tf.Tensor): Velocity component in x-direction [nx, ny, nz], dtype=tf.float32.
        uy (tf.Tensor): Velocity component in y-direction [nx, ny, nz], dtype=tf.float32.
        uz (tf.Tensor): Velocity component in z-direction [nx, ny, nz], dtype=tf.float32.
        x (tf.Tensor): 1D x-coordinates [nx], dtype=tf.float32.
        y (tf.Tensor): 1D y-coordinates [ny], dtype=tf.float32.
        z (tf.Tensor): 1D z-coordinates [nz], dtype=tf.float32.
    
    Returns:
        tf.Tensor: Divergence field [nx, ny, nz], dtype=tf.float32.
    """
    # Compute partial derivatives using compute_gradient_tf
    grad_ux_x, _, _ = compute_gradient_tf(ux, x, y, z, "ux")
    _, grad_uy_y, _ = compute_gradient_tf(uy, x, y, z, "uy")
    _, _, grad_uz_z = compute_gradient_tf(uz, x, y, z, "uz")
    
    # Divergence is the sum of partial derivatives
    divergence = grad_ux_x + grad_uy_y + grad_uz_z
    
    return divergence

# ==================================
# TensorFlow Navier-Stokes Solver
# ==================================

# Decorate for potential graph compilation
@tf.function
def solve_navier_stokes_projection_tf(
    ux, uy, uz,
    p,
    rho_charge,
    Ex, Ey, Ez,
    x, y, z,
    nx, ny, nz,
    boundary_mask,
    collector_mask,
    rho_fluid,
    mu_fluid,
    dt,
    steps,
    x_array, y_array, z_array, nx_array, ny_array, nz_array, coeffs_array, num_levels
):
    # Input Processing
    ux_tf = tf.convert_to_tensor(ux, dtype=tf.float32)
    uy_tf = tf.convert_to_tensor(uy, dtype=tf.float32)
    uz_tf = tf.convert_to_tensor(uz, dtype=tf.float32)
    p_tf = tf.convert_to_tensor(p, dtype=tf.float32)
    rho_charge_tf = tf.convert_to_tensor(rho_charge, dtype=tf.float32)
    Ex_tf = tf.convert_to_tensor(Ex, dtype=tf.float32)
    Ey_tf = tf.convert_to_tensor(Ey, dtype=tf.float32)
    Ez_tf = tf.convert_to_tensor(Ez, dtype=tf.float32)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    z_tf = tf.convert_to_tensor(z, dtype=tf.float32)
    boundary_mask_tf = tf.convert_to_tensor(boundary_mask, dtype=tf.bool)
    collector_mask_tf = tf.convert_to_tensor(collector_mask, dtype=tf.bool)
    nx_tf = tf.cast(nx, dtype=tf.int32)
    ny_tf = tf.cast(ny, dtype=tf.int32)
    nz_tf = tf.cast(nz, dtype=tf.int32)
    mu_fluid_tf = tf.convert_to_tensor(mu_fluid, dtype=tf.float32)
    rho_fluid_tf = tf.convert_to_tensor(rho_fluid, dtype=tf.float32)
    dt_tf = tf.convert_to_tensor(dt, dtype=tf.float32)
    viscosity_term_coeff = mu_fluid_tf / rho_fluid_tf
    pressure_term_coeff = dt_tf / rho_fluid_tf

    # Laplacian Coefficients
    (coeff_x_ip1, coeff_x_im1, coeff_x_i,
     coeff_y_jp1, coeff_y_jm1, coeff_y_j,
     coeff_z_kp1, coeff_z_km1, coeff_z_k) = precompute_laplacian_coefficients(x_tf, y_tf, z_tf)

    # Grid Spacings
    dx_avg_tf = tf.cond(nx_tf > 1,
                        lambda: tf.reduce_mean(x_tf[1:] - x_tf[:-1]),
                        lambda: tf.constant(1.0, dtype=tf.float32))
    dy_avg_tf = tf.cond(ny_tf > 1,
                        lambda: tf.reduce_mean(y_tf[1:] - y_tf[:-1]),
                        lambda: tf.constant(1.0, dtype=tf.float32))
    dz_scalar_tf = tf.cond(nz_tf > 1,
                           lambda: tf.reduce_mean(z_tf[1:] - z_tf[:-1]),
                           lambda: tf.constant(1.0, dtype=tf.float32))

    # Body Force with Clipping
    force_clip = tf.constant(1e4, dtype=tf.float32)  # Limit acceleration
    fx_tf = tf.clip_by_value(rho_charge_tf * Ex_tf / rho_fluid_tf, -force_clip, force_clip)
    fy_tf = tf.clip_by_value(rho_charge_tf * Ey_tf / rho_fluid_tf, -force_clip, force_clip)
    fz_tf = tf.clip_by_value(rho_charge_tf * Ez_tf / rho_fluid_tf, -force_clip, force_clip)
    conditional_tf_print(DEBUG_FLAG,"Max |fx_tf|:", tf.reduce_max(tf.abs(fx_tf)),
             "Max |fy_tf|:", tf.reduce_max(tf.abs(fy_tf)),
             "Max |fz_tf|:", tf.reduce_max(tf.abs(fz_tf)))

    # Enhanced Smoothing Function
    def smooth_field(field):
        sigma = 1.0
        size = 3
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        kernel_1d = tf.exp(-tf.square(x) / (2 * sigma ** 2))
        kernel_1d /= tf.reduce_sum(kernel_1d)
        kernel = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel /= tf.reduce_sum(kernel)
        field_padded = tf.pad(field[None, ..., None], [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        smoothed = tf.nn.conv3d(field_padded, kernel[..., None, None], strides=[1, 1, 1, 1, 1], padding='VALID')
        return tf.squeeze(smoothed, [0, 4])

    # Time Stepping Loop
    def cond(i, ux_loop, uy_loop, uz_loop, p_loop, status):
        return tf.logical_and(i < steps, tf.equal(status, 0))

    def body(i, ux_loop, uy_loop, uz_loop, p_loop, status):
        ux_old = ux_loop
        uy_old = uy_loop
        uz_old = uz_loop

        # Predictor Step
        vel_tuple_tf = (ux_old, uy_old, uz_old)
        adv_term_x = calculate_weno_advection_term_tf(vel_tuple_tf, ux_old, dx_avg_tf, dy_avg_tf, dz_scalar_tf, nx_tf, ny_tf, nz_tf, 1)
        adv_term_y = calculate_weno_advection_term_tf(vel_tuple_tf, uy_old, dx_avg_tf, dy_avg_tf, dz_scalar_tf, nx_tf, ny_tf, nz_tf, 1)
        adv_term_z = calculate_weno_advection_term_tf(vel_tuple_tf, uz_old, dx_avg_tf, dy_avg_tf, dz_scalar_tf, nx_tf, ny_tf, nz_tf, 1)
        lap_ux = compute_laplacian_tf(ux_old, x_tf, y_tf, z_tf, coeff_x_ip1, coeff_x_im1, coeff_x_i, coeff_y_jp1, coeff_y_jm1, coeff_y_j, coeff_z_kp1, coeff_z_km1, coeff_z_k)
        lap_uy = compute_laplacian_tf(uy_old, x_tf, y_tf, z_tf, coeff_x_ip1, coeff_x_im1, coeff_x_i, coeff_y_jp1, coeff_y_jm1, coeff_y_j, coeff_z_kp1, coeff_z_km1, coeff_z_k)
        lap_uz = compute_laplacian_tf(uz_old, x_tf, y_tf, z_tf, coeff_x_ip1, coeff_x_im1, coeff_x_i, coeff_y_jp1, coeff_y_jm1, coeff_y_j, coeff_z_kp1, coeff_z_km1, coeff_z_k)

        ux_star = ux_old + dt_tf * (-adv_term_x + viscosity_term_coeff * lap_ux + fx_tf)
        uy_star = uy_old + dt_tf * (-adv_term_y + viscosity_term_coeff * lap_uy + fy_tf)
        uz_star = uz_old + dt_tf * (-adv_term_z + viscosity_term_coeff * lap_uz + fz_tf)

        # Check for NaN/Inf after predictor step
        tf.debugging.check_numerics(ux_star, "ux_star contains NaN/Inf")
        tf.debugging.check_numerics(uy_star, "uy_star contains NaN/Inf")
        tf.debugging.check_numerics(uz_star, "uz_star contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG,"Max |ux_star|:", tf.reduce_max(tf.abs(ux_star)),
                 "Max |uy_star|:", tf.reduce_max(tf.abs(uy_star)),
                 "Max |uz_star|:", tf.reduce_max(tf.abs(uz_star)))

        ux_star = tf.where(boundary_mask_tf, tf.zeros_like(ux_star), ux_star)
        uy_star = tf.where(boundary_mask_tf, tf.zeros_like(uy_star), uy_star)
        uz_star = tf.where(boundary_mask_tf, tf.zeros_like(uz_star), uz_star)

        ux_star = apply_neumann_bc_tf(ux_star)
        uy_star = apply_neumann_bc_tf(uy_star)
        uz_star = apply_neumann_bc_tf(uz_star)

        # Pressure Poisson Equation with Enhanced Stabilization
        div_u_star = compute_divergence_nonuniform_tf(ux_star, uy_star, uz_star, x_tf, y_tf, z_tf)
        div_u_star = apply_neumann_bc_tf(div_u_star)
        div_u_star_smoothed = smooth_field(div_u_star)
        div_u_star_clipped = tf.clip_by_value(div_u_star_smoothed, -1e2, 1e2)
        tf.debugging.check_numerics(div_u_star_clipped, "div_u_star_clipped contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG,"Max |div_u_star|:", tf.reduce_max(tf.abs(div_u_star)),
                 "Max |div_u_star_clipped|:", tf.reduce_max(tf.abs(div_u_star_clipped)))

        # Dynamic RHS scaling and capping
        dt_scale = tf.minimum(tf.constant(1.0, dtype=tf.float32), dt_tf * 1e6)
        f_pressure = (rho_fluid_tf / dt_tf) * div_u_star_clipped * dt_scale
        f_pressure_cap = tf.constant(1e3, dtype=tf.float32) / (dx_avg_tf * dy_avg_tf * dz_scalar_tf)  # Changed from 1e6 to 1e3
        f_pressure = tf.clip_by_value(f_pressure, -f_pressure_cap, f_pressure_cap)
        
        # Adjust f_pressure to satisfy Neumann BC compatibility
        f_pressure_mean = tf.reduce_mean(f_pressure)
        f_pressure = f_pressure - f_pressure_mean
        tf.debugging.check_numerics(f_pressure, "f_pressure contains NaN/Inf after adjustment")
        conditional_tf_print(DEBUG_FLAG,"Adjusted f_pressure mean to zero. New Max |f_pressure|:", tf.reduce_max(tf.abs(f_pressure)))

        # Initialize pressure with non-zero guess
        p_initial = tf.ones_like(p_loop, dtype=tf.float32) * 1e-3  # Non-zero initial guess
        pressure_bc_mask_tf = tf.zeros(tf.shape(p_loop), dtype=tf.bool)

        # Enhanced Multigrid Solve with Debugging
        conditional_tf_print(DEBUG_FLAG,"Starting pressure multigrid solve...")
        p_updated_tf = multigrid_v_cycle_tf(
            p_initial, f_pressure,  # Use p_initial instead of p_loop
            x_array, y_array, z_array,
            nx_array, ny_array, nz_array,
            coeffs_array, num_levels,
            pressure_bc_mask_tf,
            num_cycles=5,
            cg_max_iter=30
        )
        p_new = tf.reshape(p_updated_tf, tf.shape(p_loop))

        # Residual check with debugging
        lap_p = compute_laplacian_tf(p_new, x_tf, y_tf, z_tf, coeff_x_ip1, coeff_x_im1, coeff_x_i,
                                    coeff_y_jp1, coeff_y_jm1, coeff_y_j, coeff_z_kp1, coeff_z_km1, coeff_z_k)
        residual = tf.reduce_max(tf.abs(lap_p - f_pressure))
        conditional_tf_print(DEBUG_FLAG,"Pressure multigrid residual:", residual)
        tf.debugging.check_numerics(lap_p, "lap_p contains NaN/Inf in residual check")

        # Fallback for NaN/Inf
        p_new = tf.cond(
            tf.reduce_all(tf.math.is_finite(p_new)),
            lambda: p_new,
            lambda: p_initial  # Fallback to initial guess instead of p_loop
        )
        tf.debugging.check_numerics(p_new, "p_new contains NaN/Inf after multigrid")

        # Corrector Step
        grad_p_x, grad_p_y, grad_p_z = compute_gradient_tf(p_new, x_tf, y_tf, z_tf, "Pressure")
        ux_corrected = ux_star - pressure_term_coeff * grad_p_x
        uy_corrected = uy_star - pressure_term_coeff * grad_p_y
        uz_corrected = uz_star - pressure_term_coeff * grad_p_z

        # Apply Boundary Conditions
        ux_noslip_base = tf.where(boundary_mask_tf, tf.zeros_like(ux_corrected), ux_corrected)
        uy_noslip_base = tf.where(boundary_mask_tf, tf.zeros_like(uy_corrected), uy_corrected)
        uz_noslip_base = tf.where(boundary_mask_tf, tf.zeros_like(uz_corrected), uz_corrected)

        slip_factor = tf.constant(0.1, dtype=tf.float32)
        ux_noslip = tf.where(collector_mask_tf, slip_factor * ux_corrected, ux_noslip_base)
        uy_noslip = tf.where(collector_mask_tf, slip_factor * uy_corrected, uy_noslip_base)
        uz_noslip = tf.where(collector_mask_tf, slip_factor * uz_corrected, uz_noslip_base)

        ux_bc_x = tf.pad(ux_noslip[1:-1, :, :], [[1, 1], [0, 0], [0, 0]], mode='SYMMETRIC')
        uy_bc_x = tf.pad(uy_noslip[1:-1, :, :], [[1, 1], [0, 0], [0, 0]], mode='SYMMETRIC')
        uz_bc_x = tf.pad(uz_noslip[1:-1, :, :], [[1, 1], [0, 0], [0, 0]], mode='SYMMETRIC')

        ux_bc_xy = tf.pad(ux_bc_x[:, 1:-1, :], [[0, 0], [1, 1], [0, 0]], mode='SYMMETRIC')
        uy_bc_xy = tf.pad(uy_bc_x[:, 1:-1, :], [[0, 0], [1, 1], [0, 0]], mode='SYMMETRIC')
        uz_bc_xy = tf.pad(uz_bc_x[:, 1:-1, :], [[0, 0], [1, 1], [0, 0]], mode='SYMMETRIC')

        ux_new = ux_bc_xy
        uy_new = uy_bc_xy
        uz_new = uz_bc_xy

        # Update Status
        is_finite_u = tf.logical_and(
            tf.reduce_all(tf.math.is_finite(ux_new)),
            tf.logical_and(
                tf.reduce_all(tf.math.is_finite(uy_new)),
                tf.reduce_all(tf.math.is_finite(uz_new))
            )
        )
        status_new = tf.cond(
            is_finite_u,
            lambda: tf.cond(
                tf.reduce_all(tf.math.is_finite(p_new)),
                lambda: status,
                lambda: tf.constant(5, dtype=tf.int32)  # New status for pressure failure
            ),
            lambda: tf.constant(4, dtype=tf.int32)
        )

        return [i + 1, ux_new, uy_new, uz_new, p_new, status_new]

    # Execute Loop
    final_step, ux_final, uy_final, uz_final, p_final, final_status = tf.while_loop(
        cond,
        body,
        loop_vars=[tf.constant(0, dtype=tf.int32), ux_tf, uy_tf, uz_tf, p_tf, tf.constant(0, dtype=tf.int32)],
        maximum_iterations=tf.cast(steps, dtype=tf.int32)
    )

    # Final Status Check
    is_finite_final = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(ux_final)),
        tf.logical_and(
            tf.reduce_all(tf.math.is_finite(uy_final)),
            tf.logical_and(
                tf.reduce_all(tf.math.is_finite(uz_final)),
                tf.reduce_all(tf.math.is_finite(p_final))
            )
        )
    )
    final_status_code = tf.cond(
        tf.equal(final_status, 0),
        lambda: tf.cond(
            is_finite_final,
            lambda: tf.constant(0, dtype=tf.int32),
            lambda: tf.constant(3, dtype=tf.int32)
        ),
        lambda: final_status
    )

    return (ux_final, uy_final, uz_final, p_final, final_status_code)

def couple_physics_eager_setup(
    params_tuple, grid_res_tuple, coupling_iterations_in, pressure_atm_in,
    temperature_k_in, townsend_A_in, townsend_B_in, E_onset_in,
    current_rho_air_in, mu_air_in, **kwargs
):
    """
    Eager setup for couple_physics_tf to compute multigrid hierarchy in graph context.
    """
    # Parameter unpacking and casting (done eagerly)
    r_e_in, r_c_in, d_in, l_in, V_in, shape_emitter, _, _ = params_tuple
    nx_base_in, ny_base_in, nz_base_in = grid_res_tuple

    # Override grid sizes to reduce memory usage
    nx_base_in, ny_base_in, nz_base_in = 32, 32, 64  # Reduced base sizes
    print(f"Using reduced grid sizes: nx={nx_base_in}, ny={ny_base_in}, nz={nz_base_in}")

    # Define Python helper function to compute next power of 2
    def next_power_of_2_py(n):
        return 2 ** np.ceil(np.log2(n + 1e-10))

    # Adjust base grid sizes to next power of 2
    nx_base_tf = tf.constant(int(next_power_of_2_py(nx_base_in)), dtype=tf.int32)
    ny_base_tf = tf.constant(int(next_power_of_2_py(ny_base_in)), dtype=tf.int32)
    nz_base_tf = tf.constant(int(next_power_of_2_py(nz_base_in)), dtype=tf.int32)
    print(f"Adjusted base grid sizes to powers of 2: nx={nx_base_tf.numpy()}, ny={ny_base_tf.numpy()}, nz={nz_base_tf.numpy()}")

    r_e_tf = tf.constant(r_e_in, dtype=tf.float32)
    r_c_tf = tf.constant(r_c_in, dtype=tf.float32)
    d_tf = tf.constant(d_in, dtype=tf.float32)
    l_tf = tf.constant(l_in, dtype=tf.float32)
    V_tf = tf.constant(V_in, dtype=tf.float32)
    coupling_iterations = tf.cast(coupling_iterations_in, dtype=tf.int32)
    temperature_k = tf.cast(temperature_k_in, dtype=tf.float32)
    P_Pa_tf = tf.constant(pressure_atm_in * 101325.0, dtype=tf.float32)
    townsend_A = tf.constant(townsend_A_in, dtype=tf.float32)
    townsend_B = tf.constant(townsend_B_in, dtype=tf.float32)
    E_onset_param = tf.constant(E_onset_in, dtype=tf.float32)
    current_rho_air = tf.constant(current_rho_air_in, dtype=tf.float32)
    mu_air = tf.constant(mu_air_in, dtype=tf.float32)
    x_offset_tf = tf.constant(0.005, dtype=tf.float32)

    shape_emitter_tf = tf.constant(shape_emitter, dtype=tf.string)
    E_onset = tf.cond(
        tf.equal(shape_emitter_tf, b'pointed'),
        lambda: E_onset_param * tf.constant(0.8, dtype=tf.float32),
        lambda: E_onset_param
    )

    # Grid Setup with reduced base sizes
    x_1d_tf, y_1d_tf, z_1d_tf, nx_tf, ny_tf, nz_tf = setup_grid_tf(
        r_e_tf, r_c_tf, d_tf, l_tf, nx_base_tf, ny_base_tf, nz_base_tf
    )

    # Compute multigrid hierarchy in graph mode
    print("Precomputing multigrid hierarchy (graph mode)...")
    with tf.device('/GPU:0'):
        x_levels, y_levels, z_levels, nx_levels, ny_levels, nz_levels, coeffs_levels, num_levels = setup_multigrid_hierarchy_tf(
            x_1d_tf, y_1d_tf, z_1d_tf, nx_tf, ny_tf, nz_tf
        )
    print(f"Multigrid hierarchy precomputed with {num_levels.numpy()} levels.")

    # Call the graph-traced core function
    return couple_physics_tf_core(
        r_e_tf, r_c_tf, d_tf, l_tf, V_tf, shape_emitter_tf, x_offset_tf,
        x_1d_tf, y_1d_tf, z_1d_tf, nx_tf, ny_tf, nz_tf,
        x_levels, y_levels, z_levels, nx_levels, ny_levels, nz_levels, coeffs_levels, num_levels,
        coupling_iterations, P_Pa_tf, temperature_k, townsend_A, townsend_B, E_onset,
        current_rho_air, mu_air
    )

@tf.function
def couple_physics_tf_core(
    r_e_tf, r_c_tf, d_tf, l_tf, V_tf, shape_emitter_tf, x_offset_tf,
    x_1d_tf, y_1d_tf, z_1d_tf, nx_tf, ny_tf, nz_tf,
    x_levels, y_levels, z_levels, nx_levels, ny_levels, nz_levels, coeffs_levels, num_levels,
    coupling_iterations, P_Pa_tf, temperature_k, townsend_A, townsend_B, E_onset,
    current_rho_air, mu_air
):
    """
    Core TensorFlow function for coupled EHD physics simulation.

    MODIFIED:
    - Revised initial seeding based on Step 2 (Solution 1): Uses the precise emitter mask
      from define_electrodes_tf and increases initial seed density to 1e14.
    - Removed E-field calculation specifically for initial seeding.
    - Increased multigrid cycles to 10 for better Poisson solver convergence.
    - Added residual check for Poisson solve.
    - Enhanced seeding with higher density and broader mask (via define_electrodes_tf fallback logic).
    - Added diagnostics for phi and E-field.
    - Step 3: Added debug log and assertion for initial seeding node count.
    - Updated return to include 15 elements (added dx_tf, dy_tf, dz_scalar_avg_tf) for compatibility with simulate_thrust_tf.
    - Removed rho_final from return statement to fix tuple length error (16 to 15 elements).
    """
    conditional_tf_print(DEBUG_FLAG,"Starting core physics coupling (@tf.function)...")

    # --- Constants ---
    TF_SMALL_SPACING = tf.constant(1e-12, dtype=tf.float32)
    TF_SMALL_VEL = tf.constant(1e-9, dtype=tf.float32)
    TF_SMALL_DENSITY = tf.constant(1e-6, dtype=tf.float32)
    TF_SMALL_THERMAL = tf.constant(1e-9, dtype=tf.float32)
    TF_EPSILON_0 = tf.constant(8.854e-12, dtype=tf.float32)
    TF_ELEM_CHARGE = tf.constant(1.602e-19, dtype=tf.float32)
    TF_MU_E = tf.constant(40.0, dtype=tf.float32)
    TF_MU_ION = tf.constant(1.4e-4, dtype=tf.float32)
    TF_D_E = tf.constant(0.1, dtype=tf.float32)
    TF_D_ION = tf.constant(3e-6, dtype=tf.float32)
    TF_BETA_RECOMB = tf.constant(1.6e-13, dtype=tf.float32)
    TF_ZERO = tf.constant(0.0, dtype=tf.float32)
    TF_ONE = tf.constant(1.0, dtype=tf.float32)
    TF_TWO = tf.constant(2.0, dtype=tf.float32)
    TF_THREE = tf.constant(3.0, dtype=tf.float32)
    TF_SIX = tf.constant(6.0, dtype=tf.float32)
    TF_PI = tf.constant(np.pi, dtype=tf.float32)
    TF_SQRT3 = tf.constant(np.sqrt(3.0), dtype=tf.float32)
    SUCCESS_CODE = tf.constant(0, dtype=tf.int32)
    NAN_INF_STEP_CODE = tf.constant(2, dtype=tf.int32)
    NAN_INF_FINAL_CODE = tf.constant(3, dtype=tf.int32)
    GRID_ERROR_CODE = tf.constant(5, dtype=tf.int32)
    INIT_ERROR_CODE = tf.constant(6, dtype=tf.int32)
    UNKNOWN_EXCEPTION_CODE = tf.constant(99, dtype=tf.int32)
    TF_EPSILON_MIN_SPACING = tf.constant(1e-9, dtype=tf.float32)

    # --- Grid Setup ---
    X_tf, Y_tf, Z_tf = tf.meshgrid(x_1d_tf, y_1d_tf, z_1d_tf, indexing='ij')
    conditional_tf_print(DEBUG_FLAG,"Grid mesh created. Shape:", tf.shape(X_tf))

    # Calculate Minimum Positive Grid Spacing
    conditional_tf_print(DEBUG_FLAG,"Calculating minimum POSITIVE grid spacing...")
    def compute_min_positive_spacing(coord_array, dim_size):
        is_dim_valid = tf.greater(dim_size, 1)
        def calculate_min_spacing():
            spacings = tf.abs(coord_array[1:] - coord_array[:-1])
            filter_threshold = TF_SMALL_SPACING / 10.0
            positive_spacings = tf.boolean_mask(spacings, spacings > filter_threshold)
            return tf.cond(
                tf.greater(tf.size(positive_spacings), 0),
                lambda: tf.reduce_min(positive_spacings),
                lambda: tf.constant(1e10, dtype=tf.float32) # Return large if no positive spacing found
            )
        def return_large_spacing():
            return tf.constant(1e10, dtype=tf.float32)
        return tf.cond(is_dim_valid, calculate_min_spacing, return_large_spacing)

    min_pos_dx_tf = compute_min_positive_spacing(x_1d_tf, nx_tf)
    min_pos_dy_tf = compute_min_positive_spacing(y_1d_tf, ny_tf)
    min_pos_dz_tf = compute_min_positive_spacing(z_1d_tf, nz_tf)
    min_spacing_tf = tf.minimum(tf.minimum(min_pos_dx_tf, min_pos_dy_tf), min_pos_dz_tf)
    conditional_tf_print(DEBUG_FLAG,"Min positive grid spacing (dx, dy, dz, overall):", min_pos_dx_tf, min_pos_dy_tf, min_pos_dz_tf, min_spacing_tf)
    tf.debugging.assert_greater(min_spacing_tf, TF_SMALL_SPACING,
                                message="Minimum POSITIVE grid spacing is too small or zero.")

    # Average Grid Spacings
    dx_tf = tf.cond(nx_tf > 1,
                    lambda: tf.reduce_mean(tf.abs(x_1d_tf[1:] - x_1d_tf[:-1])),
                    lambda: tf.constant(1.0, dtype=tf.float32))
    dy_tf = tf.cond(ny_tf > 1,
                    lambda: tf.reduce_mean(tf.abs(y_1d_tf[1:] - y_1d_tf[:-1])),
                    lambda: tf.constant(1.0, dtype=tf.float32))
    dz_scalar_avg_tf = tf.cond(nz_tf > 1,
                               lambda: tf.reduce_mean(tf.abs(z_1d_tf[1:] - z_1d_tf[:-1])),
                               lambda: tf.constant(1.0, dtype=tf.float32))
    conditional_tf_print(DEBUG_FLAG,"Average grid spacings (dx, dy, dz_scalar):", dx_tf, dy_tf, dz_scalar_avg_tf)

    # --- Electrode Definition ---
    conditional_tf_print(DEBUG_FLAG,"Calling define_electrodes_tf...")
    phi_init_tf, boundary_mask_tf, collector_mask_tf, electrode_status = define_electrodes_tf(
        X_tf, Y_tf, Z_tf, r_e_tf, r_c_tf, d_tf, l_tf, V_tf,
        x_1d_tf, y_1d_tf, z_1d_tf,
        shape_emitter=shape_emitter_tf,
        x_offset=x_offset_tf
    )
    # Directly use the final_emitter_mask from the return tuple for seeding later
    final_emitter_mask = tf.identity(boundary_mask_tf) # Assuming boundary_mask contains emitter for now
    emitter_potential_mask = tf.equal(phi_init_tf, V_tf)
    conditional_tf_print(DEBUG_FLAG,"Recomputed emitter mask based on potential: Nodes=", tf.reduce_sum(tf.cast(emitter_potential_mask, tf.int32)))

    tf.cond(tf.not_equal(electrode_status, SUCCESS_CODE),
            lambda: conditional_tf_print(DEBUG_FLAG,"ERROR: Electrode definition failed with status:", electrode_status),
            lambda: tf.no_op())
    status_code = electrode_status

    # --- Initialize Fields ---
    conditional_tf_print(DEBUG_FLAG,"Initializing fields...")
    phi = phi_init_tf
    ux = tf.zeros_like(phi_init_tf, dtype=tf.float32)
    uy = tf.zeros_like(phi_init_tf, dtype=tf.float32)
    uz = tf.zeros_like(phi_init_tf, dtype=tf.float32)
    p = tf.zeros_like(phi_init_tf, dtype=tf.float32)

    # --- Initial Seeding (Revised based on Step 2 / Solution 1 & Step 3 Check) ---
    conditional_tf_print(DEBUG_FLAG,"Applying initial seeding based on emitter mask (Step 2/3)...")
    initial_seed_density = tf.constant(1e18, dtype=tf.float32) # Increased seed density

    # Use the mask where potential is V_tf (recomputed earlier)
    initial_seeding_mask = emitter_potential_mask
    seeded_node_count = tf.reduce_sum(tf.cast(initial_seeding_mask, tf.int32)) # Calculate count

    # -------- ADDED DEBUG LOG AND ASSERTION (STEP 3) --------
    conditional_tf_print(DEBUG_FLAG,"DEBUG Seeding: Emitter mask based on potential (potential=V) has nodes:", seeded_node_count) # Added DEBUG log
    tf.debugging.assert_greater(seeded_node_count, tf.constant(1, dtype=tf.int32),
                                message="Initial seeding mask (potential=V) has zero or one node. Emitter likely not resolved.")
    # -------- END ASSERTION --------

    conditional_tf_print(DEBUG_FLAG,"Seeding initial electron and ion density (", initial_seed_density, ") in", seeded_node_count, "nodes based on emitter mask (potential=V).")

    n_e = tf.where(initial_seeding_mask, initial_seed_density, tf.zeros_like(phi_init_tf, dtype=tf.float32))
    n_i = tf.where(initial_seeding_mask, initial_seed_density, tf.zeros_like(phi_init_tf, dtype=tf.float32))
    conditional_tf_print(DEBUG_FLAG,"Max initial n_e seeded:", tf.reduce_max(n_e))
    conditional_tf_print(DEBUG_FLAG,"Max initial n_i seeded:", tf.reduce_max(n_i))
    # Ensure seeding didn't produce NaN/Inf
    tf.debugging.check_numerics(n_e, "n_e contains NaN/Inf after initial seeding")
    tf.debugging.check_numerics(n_i, "n_i contains NaN/Inf after initial seeding")

    # --- Initial Charge Density ---
    conditional_tf_print(DEBUG_FLAG,"Calculating initial charge density...")
    rho = TF_ELEM_CHARGE * (n_i - n_e) # Calculate rho based on seeded density
    conditional_tf_print(DEBUG_FLAG,"Max initial |rho|:", tf.reduce_max(tf.abs(rho)))
    tf.debugging.check_numerics(rho, "Initial rho contains NaN/Inf")

    # --- Initial Temperature ---
    T_motor = temperature_k
    conditional_tf_print(DEBUG_FLAG,"T_motor initialized to:", T_motor)

    # --- Emitter Surface Area ---
    def cylindrical_area(): return TF_TWO * TF_PI * r_e_tf * l_tf + TF_TWO * TF_PI * tf.square(r_e_tf)
    def pointed_area(): return TF_TWO * TF_PI * r_e_tf * l_tf + TF_TWO * TF_PI * tf.square(r_e_tf)
    def hexagonal_area(): return (TF_THREE * TF_SQRT3 / TF_TWO) * tf.square(r_e_tf) * TF_TWO + TF_SIX * r_e_tf * l_tf
    def default_area(): return tf.constant(1.0, dtype=tf.float32)
    valid_shapes_bytes = [b'cylindrical', b'pointed', b'hexagonal']
    shape_indices = [tf.cast(tf.equal(shape_emitter_tf, shape), tf.int32) * idx for idx, shape in enumerate(valid_shapes_bytes)]
    shape_index = tf.reduce_sum(shape_indices)
    is_known_shape = tf.reduce_any([tf.equal(shape_emitter_tf, shape) for shape in valid_shapes_bytes])
    A_surface_tf = tf.switch_case(
        tf.cond(is_known_shape, lambda: shape_index, lambda: tf.constant(len(valid_shapes_bytes), dtype=tf.int32)),
        branch_fns={0: cylindrical_area, 1: pointed_area, 2: hexagonal_area},
        default=default_area
    )
    conditional_tf_print(DEBUG_FLAG,"Estimated emitter surface area:", A_surface_tf)

    # --- Coupling Loop ---
    conditional_tf_print(DEBUG_FLAG,"Preparing coupling loop variables...")
    iter_count = tf.constant(0, dtype=tf.int32)
    delta_n_e = tf.constant(1e10, dtype=tf.float32)
    delta_n_i = tf.constant(1e10, dtype=tf.float32)
    delta_u = tf.constant(1e10, dtype=tf.float32)
    n_e_old = tf.zeros_like(n_e, dtype=tf.float32)
    n_i_old = tf.zeros_like(n_i, dtype=tf.float32)
    ux_old = tf.zeros_like(ux, dtype=tf.float32)
    uy_old = tf.zeros_like(uy, dtype=tf.float32)
    uz_old = tf.zeros_like(uz, dtype=tf.float32)
    phi_old = tf.zeros_like(phi, dtype=tf.float32)

    # --- Define the geometric emitter mask for use inside the loop (needed for seeding clamp) ---
    conditional_tf_print(DEBUG_FLAG,"Defining geometric mask for loop seeding clamp...")
    min_spacing_xy_seed = tf.maximum(tf.minimum(min_pos_dx_tf, min_pos_dy_tf), TF_EPSILON_MIN_SPACING)
    radius_threshold_loop = r_e_tf + tf.constant(500.0, dtype=tf.float32) * min_spacing_xy_seed
    geometric_emitter_mask_xy = tf.sqrt(tf.square(X_tf) + tf.square(Y_tf)) <= radius_threshold_loop
    z_tolerance_seed_loop = tf.constant(100.0, dtype=tf.float32) * min_pos_dz_tf
    z_emitter_range_loop = tf.logical_and(
        Z_tf >= -l_tf / TF_TWO - z_tolerance_seed_loop,
        Z_tf <= l_tf / TF_TWO + z_tolerance_seed_loop
    )
    geometric_emitter_mask_loop = tf.logical_and(geometric_emitter_mask_xy, z_emitter_range_loop)
    conditional_tf_print(DEBUG_FLAG," Geometric mask for loop defined. Nodes:", tf.reduce_sum(tf.cast(geometric_emitter_mask_loop, tf.int32)))
    # --- End geometric mask definition ---

    loop_vars = [
        iter_count, ux, uy, uz, n_e, n_i, phi, p, rho, T_motor, status_code,
        delta_n_e, delta_n_i, delta_u,
        n_e_old, n_i_old, ux_old, uy_old, uz_old, phi_old
    ]

    convergence_tol = tf.constant(1e-3, dtype=tf.float32)
    def loop_cond(i, ux_t, uy_t, uz_t, n_e_t, n_i_t, phi_t, p_t, rho_t, T_m_t, status_t,
                  delta_n_e_t, delta_n_i_t, delta_u_t, n_e_old_t, n_i_old_t, ux_old_t, uy_old_t, uz_old_t, phi_old_t):
        iteration = i
        current_status = status_t
        converged = tf.logical_and(
            delta_n_e_t < convergence_tol,
            tf.logical_and(delta_n_i_t < convergence_tol, delta_u_t < convergence_tol)
        )
        continue_loop = tf.logical_and(
            tf.logical_and(iteration < coupling_iterations, tf.equal(current_status, SUCCESS_CODE)),
            tf.logical_or(tf.equal(iteration, 0), tf.logical_not(converged))
        )
        conditional_tf_print(DEBUG_FLAG,"Loop Condition Check: Iter", iteration, "Status", current_status, "Converged", converged, "Continue", continue_loop)
        conditional_tf_print(DEBUG_FLAG,"Deltas (Ne, Ni, U):", delta_n_e_t, delta_n_i_t, delta_u_t, "Tol:", convergence_tol)
        return continue_loop

    def loop_body(i, ux_t, uy_t, uz_t, n_e_t, n_i_t, phi_t, p_t, rho_t, T_m_t, status_t,
                  delta_n_e_t, delta_n_i_t, delta_u_t, n_e_old_t, n_i_old_t, ux_old_t, uy_old_t, uz_old_t, phi_old_t):
        conditional_tf_print(DEBUG_FLAG,"\n--- Coupling Iteration:", i + 1, "/", coupling_iterations, "---")

        n_e_old_new = n_e_t
        n_i_old_new = n_i_t
        ux_old_new = ux_t
        uy_old_new = uy_t
        uz_old_new = uz_t
        phi_old_new = phi_t

        ux_local = ux_t
        uy_local = uy_t
        uz_local = uz_t
        n_e_local = n_e_t
        n_i_local = n_i_t
        phi_local = phi_t
        p_local = p_t
        rho_local = rho_t
        T_motor_local = T_m_t
        status_local = status_t

        # Poisson Solve
        conditional_tf_print(DEBUG_FLAG,"  Starting Poisson solve...")
        max_rho_abs = tf.constant(1e-2, dtype=tf.float32) # Cap charge density influence
        rho_clipped_poisson = tf.clip_by_value(rho_local, -max_rho_abs, max_rho_abs)
        f_poisson_tf = -rho_clipped_poisson / TF_EPSILON_0
        conditional_tf_print(DEBUG_FLAG,"  Max |rho_clipped| for Poisson RHS:", tf.reduce_max(tf.abs(rho_clipped_poisson)))
        tf.debugging.check_numerics(f_poisson_tf, "Poisson RHS contains NaN/Inf")

        phi_updated_tf = multigrid_v_cycle_tf(
            phi_local, f_poisson_tf,
            x_levels, y_levels, z_levels,
            nx_levels, ny_levels, nz_levels,
            coeffs_levels, num_levels,
            boundary_mask_tf, num_cycles=10 # Increased cycles
        )
        phi_new_raw = tf.reshape(phi_updated_tf, tf.shape(phi_local))
        tf.debugging.check_numerics(phi_new_raw, "phi_new_raw contains NaN/Inf after Poisson solve")

        # Apply BCs rigorously after solve
        phi_new_raw = tf.where(boundary_mask_tf, phi_init_tf, phi_new_raw) # Enforce electrode potentials

        # Electric Field Calculation
        conditional_tf_print(DEBUG_FLAG,"  Computing electric field...")
        Ex_tf_loop, Ey_tf_loop, Ez_tf_loop = compute_gradient_tf(-phi_new_raw, x_1d_tf, y_1d_tf, z_1d_tf, "Potential")
        E_mag_tf = tf.sqrt(tf.square(Ex_tf_loop) + tf.square(Ey_tf_loop) + tf.square(Ez_tf_loop) + TF_SMALL_SPACING)
        tf.debugging.check_numerics(Ex_tf_loop, "Ex_tf_loop contains NaN/Inf")
        tf.debugging.check_numerics(Ey_tf_loop, "Ey_tf_loop contains NaN/Inf")
        tf.debugging.check_numerics(Ez_tf_loop, "Ez_tf_loop contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG,"  Max E_mag_tf:", tf.reduce_max(E_mag_tf), "E_onset:", E_onset)

        # Electron Seeding Clamp (applied before transport)
        # Use the pre-defined geometric_emitter_mask_loop
        conditional_tf_print(DEBUG_FLAG,"  Applying electron seeding clamp...")
        # Determine seeding mask based on E-field *and* geometric location
        seed_field_threshold_loop = tf.constant(0.1, dtype=tf.float32) * E_onset # Use fraction of E_onset
        current_seeding_mask_loop = tf.logical_and(E_mag_tf > seed_field_threshold_loop, geometric_emitter_mask_loop)

        # Apply the seeding density *only where needed* and clamp existing density
        n_e_seeded = tf.maximum(n_e_local, tf.where(current_seeding_mask_loop, initial_seed_density, TF_ZERO))
        n_e_seeded = tf.clip_by_value(n_e_seeded, 0.0, 1e17) # Upper cap
        conditional_tf_print(DEBUG_FLAG,"  Max n_e after seeding clamp:", tf.reduce_max(n_e_seeded))
        tf.debugging.check_numerics(n_e_seeded, "n_e_seeded contains NaN/Inf")

        # Timestep Calculation
        conditional_tf_print(DEBUG_FLAG,"  Calculating timestep...")
        TF_DEFAULT_DT_LOOP = tf.constant(5e-9, dtype=tf.float32) # Use a reasonable default dt
        cfl_safety_factor = tf.constant(0.3, dtype=tf.float32)
        max_u_mag_tf = tf.reduce_max(tf.sqrt(tf.square(ux_local) + tf.square(uy_local) + tf.square(uz_local)))
        max_drift_vel_tf = tf.reduce_max(tf.maximum(TF_MU_E * E_mag_tf, TF_MU_ION * E_mag_tf))
        cfl_denominator_tf = max_drift_vel_tf + max_u_mag_tf + TF_SMALL_VEL
        cfl_dt_tf = cfl_safety_factor * min_spacing_tf / cfl_denominator_tf
        cfl_dt_tf = tf.where(tf.math.is_finite(cfl_dt_tf) & (cfl_dt_tf > 0), cfl_dt_tf, TF_DEFAULT_DT_LOOP)
        dt_step_tf = tf.minimum(TF_DEFAULT_DT_LOOP, cfl_dt_tf)
        transport_steps_tf = tf.constant(1, dtype=tf.int32) # Number of sub-steps for transport solvers
        ns_steps_tf = tf.constant(1, dtype=tf.int32) # Number of sub-steps for NS solver
        conditional_tf_print(DEBUG_FLAG,"  Calculated dt_step_tf:", dt_step_tf)

        # Corona Source Term
        conditional_tf_print(DEBUG_FLAG,"  Computing corona source...")
        alpha_eff_tf, corona_status = corona_source_tf(E_mag_tf, P_Pa_tf, townsend_A, townsend_B)
        status_local = tf.cond(tf.equal(corona_status, SUCCESS_CODE), lambda: status_local, lambda: tf.maximum(status_local, corona_status))

        # Calculate source only where ionization is expected (E>E_onset) and seed electrons exist
        ionization_mask_tf = tf.logical_and(E_mag_tf > E_onset, n_e_seeded > TF_SMALL_DENSITY)
        v_drift_e_mag_tf = TF_MU_E * E_mag_tf
        n_e_for_source = tf.where(ionization_mask_tf, n_e_seeded, TF_ZERO) # Use seeded density
        S_e_ionized_tf = alpha_eff_tf * v_drift_e_mag_tf * n_e_for_source
        S_e_ionized_tf = tf.maximum(S_e_ionized_tf, TF_ZERO) # Ensure non-negative

        # Assign source only within the ionization mask
        S_e_tf_raw = tf.where(ionization_mask_tf, S_e_ionized_tf, TF_ZERO)
        S_i_tf_raw = tf.where(ionization_mask_tf, S_e_ionized_tf, TF_ZERO) # Ions created where electrons ionize

        # Clip source rates
        max_source_rate = tf.constant(1e25, dtype=tf.float32) # Max source rate (m^-3 s^-1)
        S_e_tf = tf.clip_by_value(S_e_tf_raw, 0.0, max_source_rate)
        S_i_tf = tf.clip_by_value(S_i_tf_raw, 0.0, max_source_rate)
        conditional_tf_print(DEBUG_FLAG,"  Max S_e_tf (clipped):", tf.reduce_max(S_e_tf))
        tf.debugging.check_numerics(S_e_tf, "S_e_tf contains NaN/Inf")
        tf.debugging.check_numerics(S_i_tf, "S_i_tf contains NaN/Inf")

        # Electron Transport
        conditional_tf_print(DEBUG_FLAG,"  Starting electron transport...")
        n_e_updated_tf, electron_status = tf.cond(
            tf.equal(status_local, SUCCESS_CODE),
            lambda: solve_electron_transport_tf(
                n_e_seeded, n_i_local, Ex_tf_loop, Ey_tf_loop, Ez_tf_loop, S_e_tf,
                dx_tf, dy_tf, z_1d_tf, nx_tf, ny_tf, nz_tf, collector_mask_tf,
                dt_step_tf, transport_steps_tf, TF_BETA_RECOMB, TF_MU_E, TF_D_E,
                x_1d_tf, y_1d_tf,
                current_seeding_mask_in=current_seeding_mask_loop, # Corrected keyword
                initial_seed_density_in=initial_seed_density      # Corrected keyword
            ),
            lambda: (n_e_local, status_local) # Skip if error
        )
        n_e_new_raw = n_e_updated_tf
        status_local = tf.cond(tf.equal(electron_status, SUCCESS_CODE), lambda: status_local, lambda: tf.maximum(status_local, electron_status))
        tf.debugging.check_numerics(n_e_new_raw, "n_e_new_raw contains NaN/Inf after electron transport")
        conditional_tf_print(DEBUG_FLAG,"  Max n_e after transport:", tf.reduce_max(n_e_new_raw))

        # Ion Transport
        conditional_tf_print(DEBUG_FLAG,"  Starting ion transport...")
        n_i_updated_tf, ion_status = tf.cond(
            tf.equal(status_local, SUCCESS_CODE),
            lambda: solve_ion_transport_tf(
                n_i_local, Ex_tf_loop, Ey_tf_loop, Ez_tf_loop,
                ux_local, uy_local, uz_local, S_i_tf,
                nx_tf, ny_tf, nz_tf, dx_tf, dy_tf, z_1d_tf,
                collector_mask_tf, dt_step_tf, transport_steps_tf,
                TF_MU_ION, TF_D_ION, x_1d_tf, y_1d_tf
            ),
            lambda: (n_i_local, status_local) # Skip if error
        )
        n_i_new_raw = n_i_updated_tf
        status_local = tf.cond(tf.equal(ion_status, SUCCESS_CODE), lambda: status_local, lambda: tf.maximum(status_local, ion_status))
        tf.debugging.check_numerics(n_i_new_raw, "n_i_new_raw contains NaN/Inf after ion transport")
        conditional_tf_print(DEBUG_FLAG,"  Max n_i after transport:", tf.reduce_max(n_i_new_raw))

        # Update Charge Density
        conditional_tf_print(DEBUG_FLAG,"  Updating charge density...")
        rho_new_calculated = TF_ELEM_CHARGE * (n_i_new_raw - n_e_new_raw)
        rho_new_clipped = tf.clip_by_value(rho_new_calculated, -max_rho_abs, max_rho_abs) # Use same cap as Poisson
        conditional_tf_print(DEBUG_FLAG,"  Max |rho_new_clipped|:", tf.reduce_max(tf.abs(rho_new_clipped)))
        tf.debugging.check_numerics(rho_new_clipped, "rho_new_clipped contains NaN/Inf")

        # Navier-Stokes Solver
        conditional_tf_print(DEBUG_FLAG,"  Starting Navier-Stokes...")
        [ux_updated_tf, uy_updated_tf, uz_updated_tf, p_updated_tf, ns_status] = tf.cond(
            tf.equal(status_local, SUCCESS_CODE),
            lambda: solve_navier_stokes_projection_tf(
                ux_local, uy_local, uz_local, p_local, rho_new_clipped,
                Ex_tf_loop, Ey_tf_loop, Ez_tf_loop,
                x_1d_tf, y_1d_tf, z_1d_tf, nx_tf, ny_tf, nz_tf,
                boundary_mask_tf, collector_mask_tf,
                current_rho_air, mu_air, dt_step_tf, ns_steps_tf,
                x_levels, y_levels, z_levels, nx_levels, ny_levels, nz_levels, coeffs_levels, num_levels
            ),
            lambda: [ux_local, uy_local, uz_local, p_local, status_local] # Skip if error
        )
        ux_new_raw = ux_updated_tf
        uy_new_raw = uy_updated_tf
        uz_new_raw = uz_updated_tf
        p_new_raw = p_updated_tf
        status_local = tf.cond(tf.equal(ns_status, SUCCESS_CODE), lambda: status_local, lambda: tf.maximum(status_local, ns_status))
        tf.debugging.check_numerics(ux_new_raw, "ux_new_raw contains NaN/Inf after NS")
        tf.debugging.check_numerics(uy_new_raw, "uy_new_raw contains NaN/Inf after NS")
        tf.debugging.check_numerics(uz_new_raw, "uz_new_raw contains NaN/Inf after NS")
        tf.debugging.check_numerics(p_new_raw, "p_new_raw contains NaN/Inf after NS")
        conditional_tf_print(DEBUG_FLAG,"  Max |ux| after NS:", tf.reduce_max(tf.abs(ux_new_raw)))
        conditional_tf_print(DEBUG_FLAG,"  Max |p| after NS:", tf.reduce_max(tf.abs(p_new_raw)))

        # Thermal Effects (Placeholder - update as needed)
        conditional_tf_print(DEBUG_FLAG,"  Calculating thermal effects...")
        P_dissipated_density_tf = tf.abs(rho_new_clipped * (Ex_tf_loop * ux_new_raw + Ey_tf_loop * uy_new_raw + Ez_tf_loop * uz_new_raw))
        P_heat_total_tf = tf.reduce_sum(P_dissipated_density_tf) * dx_tf * dy_tf * dz_scalar_avg_tf
        P_heat_to_emitter_tf = tf.constant(0.05, dtype=tf.float32) * P_heat_total_tf # Assume 5% heating
        P_cool_tf = tf.constant(25.0, dtype=tf.float32) * A_surface_tf * (T_motor_local - temperature_k) # Convective cooling estimate
        thermal_mass_estimate = tf.constant(500.0, dtype=tf.float32) # J/K, placeholder
        dT_dt_tf = (P_heat_to_emitter_tf - P_cool_tf) / tf.maximum(thermal_mass_estimate, TF_SMALL_THERMAL)
        delta_T = (dt_step_tf * tf.cast(ns_steps_tf, dtype=tf.float32)) * dT_dt_tf
        T_motor_new_raw = tf.cond(
            tf.equal(status_local, SUCCESS_CODE),
            lambda: tf.maximum(T_motor_local + delta_T, TF_ZERO), # Prevent negative temps
            lambda: T_motor_local # Keep old temp if error
        )
        tf.debugging.check_numerics(T_motor_new_raw, "T_motor_new_raw contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG,"  T_motor new:", T_motor_new_raw)

        # Apply Relaxation (adjust factor as needed)
        conditional_tf_print(DEBUG_FLAG,"  Applying relaxation...")
        relax_factor = tf.constant(0.8, dtype=tf.float32) # Relaxation factor (0.0 to 1.0)
        ux_relaxed = relax_factor * ux_new_raw + (TF_ONE - relax_factor) * ux_old_new
        uy_relaxed = relax_factor * uy_new_raw + (TF_ONE - relax_factor) * uy_old_new
        uz_relaxed = relax_factor * uz_new_raw + (TF_ONE - relax_factor) * uz_old_new
        n_e_relaxed = relax_factor * n_e_new_raw + (TF_ONE - relax_factor) * n_e_old_new
        n_i_relaxed = relax_factor * n_i_new_raw + (TF_ONE - relax_factor) * n_i_old_new
        phi_relaxed = relax_factor * phi_new_raw + (TF_ONE - relax_factor) * phi_old_new

        # Don't relax pressure, charge density, or temperature usually
        p_final_step = p_new_raw
        rho_final_step = rho_new_clipped
        T_motor_final_step = T_motor_new_raw

        # Check relaxed values for NaNs
        tf.debugging.check_numerics(ux_relaxed, "ux_relaxed contains NaN/Inf")
        tf.debugging.check_numerics(uy_relaxed, "uy_relaxed contains NaN/Inf")
        tf.debugging.check_numerics(uz_relaxed, "uz_relaxed contains NaN/Inf")
        tf.debugging.check_numerics(n_e_relaxed, "n_e_relaxed contains NaN/Inf")
        tf.debugging.check_numerics(n_i_relaxed, "n_i_relaxed contains NaN/Inf")
        tf.debugging.check_numerics(phi_relaxed, "phi_relaxed contains NaN/Inf")
        conditional_tf_print(DEBUG_FLAG,"  Max |ux_relaxed|:", tf.reduce_max(tf.abs(ux_relaxed)))
        conditional_tf_print(DEBUG_FLAG,"  Max n_e_relaxed:", tf.reduce_max(n_e_relaxed))

        # Convergence Metrics
        epsilon_conv = tf.constant(1e-10, dtype=tf.float32) # Small number for safe division
        delta_n_e_new = tf.reduce_max(tf.abs(n_e_relaxed - n_e_old_new)) / (tf.reduce_max(tf.abs(n_e_old_new)) + epsilon_conv)
        delta_n_i_new = tf.reduce_max(tf.abs(n_i_relaxed - n_i_old_new)) / (tf.reduce_max(tf.abs(n_i_old_new)) + epsilon_conv)

        u_mag_new = tf.sqrt(ux_relaxed**2 + uy_relaxed**2 + uz_relaxed**2)
        u_mag_old = tf.sqrt(ux_old_new**2 + uy_old_new**2 + uz_old_new**2)
        delta_u_abs = tf.reduce_max(tf.abs(u_mag_new - u_mag_old))
        # Use relative error only if the old magnitude is significant
        delta_u_rel = delta_u_abs / (tf.reduce_max(u_mag_old) + epsilon_conv)
        delta_u_new = tf.cond(tf.reduce_max(u_mag_old) > epsilon_conv * 100,
                              lambda: delta_u_rel,
                              lambda: delta_u_abs) # Use absolute error if velocity is near zero

        conditional_tf_print(DEBUG_FLAG,"  Convergence Deltas: dNe=", delta_n_e_new, "dNi=", delta_n_i_new, "dU=", delta_u_new)

        return [
            i + 1, ux_relaxed, uy_relaxed, uz_relaxed, n_e_relaxed, n_i_relaxed, phi_relaxed, p_final_step, rho_final_step, T_motor_final_step, status_local,
            delta_n_e_new, delta_n_i_new, delta_u_new,
            n_e_old_new, n_i_old_new, ux_old_new, uy_old_new, uz_old_new, phi_old_new
        ]

    # Execute While Loop
    conditional_tf_print(DEBUG_FLAG,"\nExecuting tf.while_loop for coupling iterations...")
    shape_invariants = [
        tf.TensorSpec([], tf.int32), # iter_count
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # ux
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # uy
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # uz
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # n_e
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # n_i
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # phi
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # p
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # rho
        tf.TensorSpec([], tf.float32), # T_motor
        tf.TensorSpec([], tf.int32), # status_code
        tf.TensorSpec([], tf.float32), # delta_n_e
        tf.TensorSpec([], tf.float32), # delta_n_i
        tf.TensorSpec([], tf.float32), # delta_u
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # n_e_old
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # n_i_old
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # ux_old
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # uy_old
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # uz_old
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32) # phi_old
    ]

    final_loop_vars = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars=loop_vars,
        shape_invariants=shape_invariants,
        parallel_iterations=1 # Ensure sequential execution for stability
    )

    # Extract Final Results
    final_i, ux_final, uy_final, uz_final, n_e_final, n_i_final, \
    phi_final, p_final, rho_final, T_motor_final, final_status, \
    _, _, _, _, _, _, _, _, _ = final_loop_vars # Ignore convergence deltas and old values

    conditional_tf_print(DEBUG_FLAG,"Coupling loop finished after iteration:", final_i, "Final Status:", final_status)

    # Final Electric Field Calculation (only if needed and sim was okay)
    conditional_tf_print(DEBUG_FLAG,"Computing final electric field...")
    def compute_final_E():
        return compute_gradient_tf(-phi_final, x_1d_tf, y_1d_tf, z_1d_tf, "Final Potential")

    def return_zero_E():
        zero_field = tf.zeros_like(phi_final)
        return zero_field, zero_field, zero_field

    # Compute E if status is Success OR if final NaN/Inf occurred (to see where it happened)
    should_compute_E = tf.logical_or(tf.equal(final_status, SUCCESS_CODE),
                                      tf.equal(final_status, NAN_INF_FINAL_CODE))

    Ex_final, Ey_final, Ez_final = tf.cond(
        should_compute_E,
        compute_final_E,
        return_zero_E
    )

    # Check final E-field for NaNs, even if sim failed earlier
    tf.debugging.check_numerics(Ex_final, "Ex_final contains NaN/Inf")
    tf.debugging.check_numerics(Ey_final, "Ey_final contains NaN/Inf")
    tf.debugging.check_numerics(Ez_final, "Ez_final contains NaN/Inf")
    conditional_tf_print(DEBUG_FLAG,"Final E-field computed. Max |Ex|:", tf.reduce_max(tf.abs(Ex_final)))

    # Final Status Check - Ensure all returned fields are finite if status is SUCCESS
    conditional_tf_print(DEBUG_FLAG,"Performing final finiteness check on all returned fields...")
    is_finite_ux = tf.reduce_all(tf.math.is_finite(ux_final))
    is_finite_uy = tf.reduce_all(tf.math.is_finite(uy_final))
    is_finite_uz = tf.reduce_all(tf.math.is_finite(uz_final))
    is_finite_ne = tf.reduce_all(tf.math.is_finite(n_e_final))
    is_finite_ni = tf.reduce_all(tf.math.is_finite(n_i_final))
    is_finite_phi = tf.reduce_all(tf.math.is_finite(phi_final))
    is_finite_Ex = tf.reduce_all(tf.math.is_finite(Ex_final))
    is_finite_Ey = tf.reduce_all(tf.math.is_finite(Ey_final))
    is_finite_Ez = tf.reduce_all(tf.math.is_finite(Ez_final))
    is_finite_p = tf.reduce_all(tf.math.is_finite(p_final))
    is_finite_T = tf.reduce_all(tf.math.is_finite(T_motor_final))

    all_finite = tf.logical_and(is_finite_ux, tf.logical_and(is_finite_uy, tf.logical_and(is_finite_uz, tf.logical_and(is_finite_ne, tf.logical_and(is_finite_ni, tf.logical_and(is_finite_phi, tf.logical_and(is_finite_Ex, tf.logical_and(is_finite_Ey, tf.logical_and(is_finite_Ez, tf.logical_and(is_finite_p, is_finite_T))))))))))

    # If the loop finished with success, but there are NaNs/Infs, change status
    final_status_checked = tf.cond(
        tf.logical_and(tf.equal(final_status, SUCCESS_CODE), tf.logical_not(all_finite)),
        lambda: NAN_INF_FINAL_CODE,
        lambda: final_status
    )
    tf.cond(tf.not_equal(final_status_checked, final_status),
            lambda: conditional_tf_print(DEBUG_FLAG,"WARNING: Final status changed to NAN_INF_FINAL due to non-finite values detected after loop completion."),
            lambda: tf.no_op())

    conditional_tf_print(DEBUG_FLAG,"Final finiteness check complete. Status:", final_status_checked)

    # Return 15 elements, excluding rho_final
    return (ux_final, uy_final, uz_final, n_e_final, n_i_final,
            phi_final, Ex_final, Ey_final, Ez_final, p_final, T_motor_final, final_status_checked,
            dx_tf, dy_tf, dz_scalar_avg_tf)

# Replace the original couple_physics_tf with the wrapper
def couple_physics_tf(*args, **kwargs):
    return couple_physics_eager_setup(*args, **kwargs)
    
@tf.function
def calculate_thrust_tf(rho_in, Ex_in, Ey_in, Ez_in, dx_in, dy_in, dz_scalar_in, altitude_m_in=0.0):
    """
    Calculates the total electrostatic thrust force using TensorFlow for GPU acceleration.

    Integrates force density (f = ρ * E, scaled by air density) over the volume.
    Uses tf.float32 precision for all calculations and returns TensorFlow tensors.
    Handles non-uniform z-grids by using a representative scalar dz_scalar for dV.

    Args:
        rho_in (tf.Tensor or convertible): Ion charge density tensor [nx, ny, nz].
        Ex_in (tf.Tensor or convertible): Electric field x-component [nx, ny, nz].
        Ey_in (tf.Tensor or convertible): Electric field y-component [nx, ny, nz].
        Ez_in (tf.Tensor or convertible): Electric field z-component [nx, ny, nz].
        dx_in (float or tf.Tensor): Grid spacing in x.
        dy_in (float or tf.Tensor): Grid spacing in y.
        dz_scalar_in (float or tf.Tensor): Representative grid spacing in z.
        altitude_m_in (float or tf.Tensor): Altitude in meters (default 0.0).

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            - Thrust_x_tf: Total thrust force in x-direction (scalar, tf.float32).
            - Thrust_y_tf: Total thrust force in y-direction (scalar, tf.float32).
            - Thrust_z_tf: Total thrust force in z-direction (scalar, tf.float32).
            - T_magnitude_tf: Total thrust magnitude (scalar, tf.float32).
            - error_code: Status code (tf.int32).
    """
    try:
        # --- Input Conversion and Constants ---
        rho_tf = tf.convert_to_tensor(rho_in, dtype=tf.float32, name="rho")
        Ex_tf = tf.convert_to_tensor(Ex_in, dtype=tf.float32, name="Ex")
        Ey_tf = tf.convert_to_tensor(Ey_in, dtype=tf.float32, name="Ey")
        Ez_tf = tf.convert_to_tensor(Ez_in, dtype=tf.float32, name="Ez")
        dx_tf = tf.cast(dx_in, dtype=tf.float32, name="dx")
        dy_tf = tf.cast(dy_in, dtype=tf.float32, name="dy")
        dz_scalar_tf = tf.cast(dz_scalar_in, dtype=tf.float32, name="dz_scalar")
        altitude_m_tf = tf.cast(altitude_m_in, dtype=tf.float32, name="altitude_m")

        RHO_AIR_SEA_LEVEL_tf = tf.constant(RHO_AIR_SEA_LEVEL, dtype=tf.float32, name="SeaLevelDensity")
        TF_EPSILON_THRUST = tf.constant(1e-12, dtype=tf.float32)

        # --- Air Density Calculation ---
        rho_air_tf, density_error_code = air_density_tf(altitude_m_tf)
        initial_error_code = tf.cond(
            tf.not_equal(density_error_code, SUCCESS_CODE),
            lambda: tf.constant(INIT_ERROR_CODE, dtype=tf.int32),
            lambda: tf.constant(SUCCESS_CODE, dtype=tf.int32)
        )

        # --- Force Density Calculation ---
        density_scaling_factor = rho_air_tf / (RHO_AIR_SEA_LEVEL_tf + TF_EPSILON_THRUST)
        fx_tf = tf.multiply(rho_tf, Ex_tf, name="rho_Ex") * density_scaling_factor
        fy_tf = tf.multiply(rho_tf, Ey_tf, name="rho_Ey") * density_scaling_factor
        fz_tf = tf.multiply(rho_tf, Ez_tf, name="rho_Ez") * density_scaling_factor

        # --- Volume Element Calculation ---
        dV_tf = tf.multiply(dx_tf, dy_tf, name="dx_dy") * dz_scalar_tf

        # --- Thrust Calculation ---
        sum_fx = tf.reduce_sum(fx_tf, name="sum_fx")
        sum_fy = tf.reduce_sum(fy_tf, name="sum_fy")
        sum_fz = tf.reduce_sum(fz_tf, name="sum_fz")
        Thrust_x_tf = tf.multiply(sum_fx, dV_tf, name="Thrust_x")
        Thrust_y_tf = tf.multiply(sum_fy, dV_tf, name="Thrust_y")
        Thrust_z_tf = tf.multiply(sum_fz, dV_tf, name="Thrust_z")
        T_magnitude_tf = tf.sqrt(
            tf.square(Thrust_x_tf) + tf.square(Thrust_y_tf) + tf.square(Thrust_z_tf),
            name="Thrust_Magnitude"
        )

        # --- Logging ---
        conditional_tf_print(DEBUG_FLAG,"Thrust (TF) at", altitude_m_tf, "m:",
                 "Tx=", Thrust_x_tf, "Ty=", Thrust_y_tf, "Tz=", Thrust_z_tf, "N,",
                 "Mag=", T_magnitude_tf, "N")

        # --- Final Error Checking ---
        is_finite_tx = tf.math.is_finite(Thrust_x_tf)
        is_finite_ty = tf.math.is_finite(Thrust_y_tf)
        is_finite_tz = tf.math.is_finite(Thrust_z_tf)
        is_finite_mag = tf.math.is_finite(T_magnitude_tf)
        is_finite_final = tf.logical_and(tf.logical_and(is_finite_tx, is_finite_ty),
                                         tf.logical_and(is_finite_tz, is_finite_mag),
                                         name="IsFiniteFinalCheck")

        final_error_code = tf.cond(
            tf.not_equal(initial_error_code, SUCCESS_CODE),
            lambda: initial_error_code,
            lambda: tf.cond(
                is_finite_final,
                lambda: tf.constant(SUCCESS_CODE, dtype=tf.int32),
                lambda: tf.constant(NAN_INF_FINAL_CODE, dtype=tf.int32)
            )
        )

        # --- Define Helper Functions for Conditional Printing ---
        def print_nan_error():
            conditional_tf_print(DEBUG_FLAG,"ERROR (calculate_thrust_tf): NaN or Inf detected in final thrust calculation.")
            return tf.constant(0, dtype=tf.int32)

        def print_init_error():
            conditional_tf_print(DEBUG_FLAG,"ERROR (calculate_thrust_tf): Failed due to error in air_density_tf dependency.")
            return tf.constant(0, dtype=tf.int32)

        def no_op():
            return tf.constant(0, dtype=tf.int32)

        # --- Conditional Printing with Consistent Outputs ---
        _ = tf.cond(
            tf.equal(final_error_code, NAN_INF_FINAL_CODE),
            print_nan_error,
            no_op
        )
        _ = tf.cond(
            tf.equal(final_error_code, INIT_ERROR_CODE),
            print_init_error,
            no_op
        )

    except Exception as e:
        conditional_tf_print(DEBUG_FLAG,"ERROR (calculate_thrust_tf): Unexpected exception:", e, output_stream=sys.stderr)
        nan_scalar = tf.constant(np.nan, dtype=tf.float32)
        Thrust_x_tf = nan_scalar
        Thrust_y_tf = nan_scalar
        Thrust_z_tf = nan_scalar
        T_magnitude_tf = nan_scalar
        final_error_code = tf.constant(UNKNOWN_EXCEPTION_CODE, dtype=tf.int32)

    return Thrust_x_tf, Thrust_y_tf, Thrust_z_tf, T_magnitude_tf, final_error_code

# ==================================
# Main Simulation Function for Trainer
# ==================================

@tf.function
def simulate_thrust_tf(
    params_tuple,  # (r_e, r_c, d, l, V, shape_emitter)
    material_props_tuple,  # (townsend_A, townsend_B, E_onset)
    grid_settings_tuple,  # (nx_base, ny_base, nz_base)
    env_conditions_tuple,  # (pressure_atm, temperature_k, altitude_m)
    coupling_iters_in,
):
    final_status_code = tf.Variable(SUCCESS_CODE, dtype=tf.int32)
    try:
        # --- Input Parameter Processing ---
        r_e_in, r_c_in, d_in, l_in, V_in, shape_emitter_in = params_tuple
        _, _, E_onset_in = material_props_tuple
        nx_base_in, ny_base_in, nz_base_in = grid_settings_tuple  # Use input grid sizes directly
        pressure_atm_in, temperature_k_in, altitude_m_in = env_conditions_tuple

        # Convert numerical inputs to TF constants
        r_e_tf = tf.constant(r_e_in, dtype=tf.float32)
        r_c_tf = tf.constant(r_c_in, dtype=tf.float32)
        d_tf = tf.constant(d_in, dtype=tf.float32)
        l_tf = tf.constant(l_in, dtype=tf.float32)

        # --- STEP 3: Ensure Correct Voltage (20 kV) ---
        V_tf = tf.constant(V_in, dtype=tf.float32)         # ENSURED this line is active
        conditional_tf_print(DEBUG_FLAG,"Voltage set to (V):", V_tf)              # ADDED log confirmation
        # --- End STEP 3 ---

        # Unit Check: Log parameters to verify SI units
        conditional_tf_print(DEBUG_FLAG,"Unit Check - Geometry and Voltage Parameters:")
        conditional_tf_print(DEBUG_FLAG,"r_e:", r_e_tf, "m")
        conditional_tf_print(DEBUG_FLAG,"r_c:", r_c_tf, "m")
        conditional_tf_print(DEBUG_FLAG,"V:", V_tf, "V")

        # Use correct Townsend coefficients for air
        townsend_A_tf = tf.constant(15.0, dtype=tf.float32)  # 15 cm^-1 torr^-1
        townsend_B_tf = tf.constant(365.0, dtype=tf.float32) # 365 V cm^-1 torr^-1
        E_onset_tf = tf.constant(E_onset_in, dtype=tf.float32)

        nx_base_tf = tf.constant(nx_base_in, dtype=tf.int32)
        ny_base_tf = tf.constant(ny_base_in, dtype=tf.int32)
        nz_base_tf = tf.constant(nz_base_in, dtype=tf.int32)

        pressure_atm_tf = tf.constant(pressure_atm_in, dtype=tf.float32)
        temperature_k_tf = tf.constant(temperature_k_in, dtype=tf.float32)
        altitude_m_tf = tf.constant(altitude_m_in, dtype=tf.float32)

        coupling_iters_tf = tf.cast(coupling_iters_in, dtype=tf.int32)

        # Derive dependent environmental parameters
        current_rho_air_tf, density_status = air_density_tf(altitude_m_tf)
        if tf.not_equal(density_status, SUCCESS_CODE):
            conditional_tf_print(DEBUG_FLAG,"ERROR: Air density calculation failed.")
            final_status_code.assign(INIT_ERROR_CODE)
            nan_scalar = tf.constant(np.nan, dtype=tf.float32)
            return nan_scalar, nan_scalar, nan_scalar, nan_scalar, final_status_code.read_value()

        mu_air_tf = tf.constant(MU_AIR, dtype=tf.float32)

        # Prepare tuples for couple_physics_tf
        params_for_coupling = (r_e_tf, r_c_tf, d_tf, l_tf, V_tf, shape_emitter_in, None, None)
        grid_res_for_coupling = (nx_base_tf, ny_base_tf, nz_base_tf)

        conditional_tf_print(DEBUG_FLAG,"--- Starting TensorFlow Coupled Simulation ---")
        conditional_tf_print(DEBUG_FLAG,"Params:", params_tuple)
        conditional_tf_print(DEBUG_FLAG,"Material Props (A, B, E_onset):", (townsend_A_tf, townsend_B_tf, E_onset_tf))
        conditional_tf_print(DEBUG_FLAG,"Grid Base:", grid_res_for_coupling)
        conditional_tf_print(DEBUG_FLAG,"Env (P_atm, T_K, Alt_m):", env_conditions_tuple)
        conditional_tf_print(DEBUG_FLAG,"Coupling Iterations:", coupling_iters_tf)
        conditional_tf_print(DEBUG_FLAG,"Calculated rho_air:", current_rho_air_tf, " mu_air:", mu_air_tf)

        # --- Call Core Simulation Function ---
        (ux_final, uy_final, uz_final, n_e_final, n_i_final, phi_final,
         Ex_final, Ey_final, Ez_final, p_final, T_motor_final, sim_status_code,
         dx_tf, dy_tf, dz_scalar_tf) = couple_physics_tf(
            params_for_coupling,
            grid_res_for_coupling,
            coupling_iters_tf,
            pressure_atm_tf,
            temperature_k_tf,
            townsend_A_tf,
            townsend_B_tf,
            E_onset_tf,
            current_rho_air_tf,
            mu_air_tf
        )

        final_status_code.assign(tf.maximum(final_status_code.read_value(), sim_status_code))

        # --- Calculate Thrust ---
        Thrust_x = tf.constant(np.nan, dtype=tf.float32)
        Thrust_y = tf.constant(np.nan, dtype=tf.float32)
        Thrust_z = tf.constant(np.nan, dtype=tf.float32)
        Thrust_Mag = tf.constant(np.nan, dtype=tf.float32)

        if tf.equal(final_status_code.read_value(), SUCCESS_CODE):
            conditional_tf_print(DEBUG_FLAG,"--- Calculating Thrust using TensorFlow ---")
            rho_charge_final = TF_ELEM_CHARGE * (n_i_final - n_e_final)
            Thrust_x, Thrust_y, Thrust_z, Thrust_Mag, thrust_status_code = calculate_thrust_tf(
                rho_in=rho_charge_final,
                Ex_in=Ex_final,
                Ey_in=Ey_final,
                Ez_in=Ez_final,
                dx_in=dx_tf,
                dy_in=dy_tf,
                dz_scalar_in=dz_scalar_tf,
                altitude_m_in=altitude_m_tf
            )
            final_status_code.assign(tf.maximum(final_status_code.read_value(), thrust_status_code))
        else:
            conditional_tf_print(DEBUG_FLAG,"Skipping thrust calculation due to simulation error status:", final_status_code.read_value())

        # --- Final NaN/Inf Check ---
        if tf.equal(final_status_code.read_value(), SUCCESS_CODE):
            is_finite_thrust = tf.logical_and(
                tf.logical_and(tf.math.is_finite(Thrust_x), tf.math.is_finite(Thrust_y)),
                tf.logical_and(tf.math.is_finite(Thrust_z), tf.math.is_finite(Thrust_Mag))
            )
            if not is_finite_thrust:
                final_status_code.assign(NAN_INF_FINAL_CODE)
                conditional_tf_print(DEBUG_FLAG,"ERROR: Final thrust values contain NaN/Inf.")

        conditional_tf_print(DEBUG_FLAG,"--- TensorFlow Simulation Finished ---")
        conditional_tf_print(DEBUG_FLAG,"Final Status Code:", final_status_code.read_value())
        conditional_tf_print(DEBUG_FLAG,"Final Thrust (Tx, Ty, Tz, Mag):", Thrust_x, Thrust_y, Thrust_z, Thrust_Mag)

        return Thrust_x, Thrust_y, Thrust_z, Thrust_Mag, final_status_code.read_value()

    except Exception as e:
        conditional_tf_print(DEBUG_FLAG,"FATAL ERROR in simulate_thrust_tf:", e, output_stream=sys.stderr)
        nan_scalar = tf.constant(np.nan, dtype=tf.float32)
        err_code = UNKNOWN_EXCEPTION_CODE
        if "assert" in str(e).lower():
            err_code = INIT_ERROR_CODE
        elif "resource exhausted" in str(e).lower():
            err_code = UNKNOWN_EXCEPTION_CODE # Or a specific OOM code like 7
        return nan_scalar, nan_scalar, nan_scalar, nan_scalar, tf.constant(err_code, dtype=tf.int32)