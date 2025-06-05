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

# || START MODIFIED GUI CODE WITH THREADING (TF COMPATIBLE) ||
import sys
import tkinter as tk
from tkinter import scrolledtext, messagebox
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle, Circle
import traceback
import threading
import queue
import time
import tensorflow as tf # <-- ADDED TF import
from scipy.interpolate import griddata

# --- IMPORTANT ---
# Ensure 'physix_single.py' is in the same directory.
try:
    # Import necessary TF functions and constants
    from physix_single import (
        couple_physics_tf, calculate_thrust_tf, air_density_tf,
        get_material_properties, setup_grid_tf, ELEM_CHARGE, MU_AIR,
        SUCCESS_CODE, NAN_INF_STEP_CODE, NAN_INF_FINAL_CODE,
        GRID_ERROR_CODE, INIT_ERROR_CODE, UNKNOWN_EXCEPTION_CODE,
        TOWNSEND_A_DEFAULT, TOWNSEND_B_DEFAULT
    )
    # Ensure GPU is configured (optional but good practice)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU detected and configured by GUI:", gpus)
        except RuntimeError as e:
            print("Error configuring GPU from GUI:", e)
    else:
        print("No GPU detected by GUI. Running on CPU.")
    physics_code_loaded = True
except ImportError as import_err:
    print(f"Import Error: {import_err}")
    traceback.print_exc() # Print detailed import error
    physics_code_loaded = False
    # Define dummy status codes if import fails
    SUCCESS_CODE = 0
    NAN_INF_STEP_CODE = 2
    NAN_INF_FINAL_CODE = 3
    GRID_ERROR_CODE = 5
    INIT_ERROR_CODE = 6
    UNKNOWN_EXCEPTION_CODE = 99
    ELEM_CHARGE = 1.602e-19
    MU_AIR = 1.8e-5
    TOWNSEND_A_DEFAULT = 15.0  # Define dummy constants
    TOWNSEND_B_DEFAULT = 365.0

    # --- Define Dummy TF-like functions ---
    def _generate_dummy_fields(nx, ny, nz):
        shape = (nx, ny, nz)
        ux = np.random.rand(*shape) * 0.1
        uy = np.random.rand(*shape) * 0.1
        uz = np.random.rand(*shape) * 0.5 # More Z velocity
        n_e = np.random.rand(*shape) * 1e7 + 1e6
        n_i = np.random.rand(*shape) * 1e9 + 1e8
        phi = np.random.rand(*shape) * -10000 + 5000
        Ex = np.zeros(shape)
        Ey = np.zeros(shape)
        Ez = np.ones(shape) * -1e5 # Simplified E field
        p = np.random.rand(*shape) * 10 - 5
        T_motor = 310.0
        # Simulate a plausible grid matching the shape
        X, Y, Z = np.meshgrid(np.linspace(-0.02, 0.02, nx),
                              np.linspace(-0.02, 0.02, ny),
                              np.linspace(-0.01, 0.02, nz), indexing='ij')
        return ux, uy, uz, n_e, n_i, phi, Ex, Ey, Ez, p, T_motor, X, Y, Z # Include Grid

    # Dummy couple_physics returning tuple like couple_physics_tf
    def dummy_couple_physics(*args, **kwargs):
        params_tuple, grid_res_tuple, *_ = args # Get some inputs
        nx_base, ny_base, nz_base = grid_res_tuple

        # Use a slightly larger grid for dummy to avoid trivial cases
        nx, ny, nz = max(nx_base, 10), max(ny_base, 10), max(nz_base, 10)

        # Use _log_from_thread if available in kwargs (passed by thread function)
        log_func = kwargs.get('log_func_for_dummy') # Use a distinct kwarg name
        if log_func:
             log_func("WARNING: Using dummy physics simulation!")
             for i in range(3): # Shortened dummy wait
                 log_func(f"Dummy simulating step {i+1}/3...")
                 time.sleep(0.3)

        ux, uy, uz, n_e, n_i, phi, Ex, Ey, Ez, p, T_motor, _, _, _ = _generate_dummy_fields(nx, ny, nz)

        # Return tuple matching couple_physics_tf signature
        # Convert NumPy arrays to TF tensors (as the real function would return)
        # For the dummy, we'll return NumPy arrays, conversion happens in plotting
        return (ux, uy, uz, n_e, n_i, phi, Ex, Ey, Ez, p, T_motor, SUCCESS_CODE)

    # Dummy calculate_thrust returning tuple like calculate_thrust_tf
    def dummy_calculate_thrust(*args, **kwargs):
         log_func = kwargs.get('log_func') # For compatibility if used
         if log_func: log_func("Dummy Thrust: Calculating...")
         Tx = np.random.rand() * 1e-4
         Ty = np.random.rand() * 1e-5
         Tz = np.random.rand() * 1e-3 + 1e-4 # Mostly Z thrust
         Tmag = np.sqrt(Tx**2 + Ty**2 + Tz**2)
         # For dummy TF version, return Tensors (or NumPy arrays that plotting code will handle)
         return Tx, Ty, Tz, Tmag, SUCCESS_CODE

    # Dummy air_density_tf
    def dummy_air_density_tf(altitude_m):
        # Simple exponential decay approximation
        H = 8000.0
        rho0 = 1.225
        rho = rho0 * np.exp(-altitude_m / H)
        # Return NumPy float and success code
        return float(rho), SUCCESS_CODE # Return float, not tensor

    # Dummy get_material_properties (already handles missing file)
    def get_material_properties(mat_name, materials_data, log_func=None):
        props = {'E_onset': 3e6, 'A': 15.0, 'B': 365.0} # Defaults
        if log_func: log_func(f"Dummy: Getting props for {mat_name}")
        # Simplified lookup for dummy
        if materials_data and 'materials' in materials_data and mat_name in materials_data['materials']:
             mat_props = materials_data['materials'][mat_name]
             if isinstance(mat_props, dict):
                 props['E_onset'] = mat_props.get('corona_onset_V_m', props['E_onset'])
                 props['A'] = mat_props.get('townsend_A', props['A'])
                 props['B'] = mat_props.get('townsend_B', props['B'])
        return props

    # Dummy setup_grid_tf returning NumPy arrays like TF version
    def dummy_setup_grid_tf(r_e, r_c, d, l, nx_base=100, ny_base=100, nz_base=100, **kwargs):
        nx, ny, nz = max(nx_base, 10), max(ny_base, 10), max(nz_base, 10)
        x = np.linspace(-r_c*1.1, r_c*1.1, nx)
        y = np.linspace(-r_c*1.1, r_c*1.1, ny)
        # Simple non-uniform z
        theta = np.linspace(0.01 * np.pi, 0.99 * np.pi, nz)
        z_coords_1d = 0.5 * (d) - 0.5 * (d + 0.02) * np.cos(theta) # Centered around d/2

        X, Y, Z = np.meshgrid(x, y, z_coords_1d, indexing='ij')
        dx = x[1] - x[0] if nx > 1 else 0.01
        dy = y[1] - y[0] if ny > 1 else 0.01
        # Return NumPy arrays and ints matching TF version signature
        return X, Y, Z, dx, dy, z_coords_1d, nx, ny, nz

    # Assign dummy functions if real ones weren't loaded
    if not physics_code_loaded:
        couple_physics_tf = dummy_couple_physics
        calculate_thrust_tf = dummy_calculate_thrust
        air_density_tf = dummy_air_density_tf
        # get_material_properties is defined above
        setup_grid_tf = dummy_setup_grid_tf


# --- GUI Class ---
class EHDSimulationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EHD Thruster Simulation Visualizer (TF Updated)")
        self.geometry("1000x800")

        # Threading and Communication Setup
        self.simulation_thread = None
        self.results_queue = queue.Queue() # Queue for simulation results/errors
        self.log_queue = queue.Queue()     # Queue for log messages from thread

        # Load materials info
        try:
            with open('materials.json', 'r') as f:
                self.materials_info = json.load(f)
        except FileNotFoundError:
            self.materials_info = {'materials': {'Default': {}}}
            messagebox.showwarning("File Not Found", "materials.json not found. Using default material properties.")
        except json.JSONDecodeError:
            self.materials_info = {'materials': {'Default': {}}}
            messagebox.showerror("File Error", "Could not parse materials.json. Using default.")

        # --- GUI Layout Frames ---
        input_frame = tk.Frame(self)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        plot_frame = tk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        log_frame = tk.Frame(self)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Create GUI components
        self.create_input_fields(input_frame)

        # Log Area
        self.log_text = scrolledtext.ScrolledText(log_frame, width=80, height=10, wrap=tk.WORD, state=tk.DISABLED) # Start disabled
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Initialize Plot Canvas
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3) # Adjust layout
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.initialize_plots() # Set initial state

        # Start periodic check for logs and results
        self.after(100, self.process_queues)

    def initialize_plots(self):
        """Sets initial titles and labels for the plots."""
        titles = ['Electric Field (V/m) & Lines', 'Ion Density (m⁻³, log scale)', 'Fluid Velocity (m/s) & Streamlines']
        for i, ax in enumerate(self.axs):
            # Clear any potential leftover elements if re-initialized
            while ax.patches: ax.patches.pop()
            while ax.collections: ax.collections.pop() # Includes contours, streamplot lines
            while ax.lines: ax.lines.pop() # Includes plot lines, electrode lines
            while ax.images: ax.images.pop()
            if hasattr(ax, 'legend_') and ax.legend_:
                try: ax.legend_.remove()
                except: pass
            # Remove colorbars associated with this specific axis
            for cb_ax in self.fig.axes:
                if hasattr(cb_ax, '_matplotlib_colorbar_origin') and cb_ax._matplotlib_colorbar_origin() is ax:
                    try: cb_ax.remove()
                    except: pass
            ax.clear() # Final clear
            ax.set_title(titles[i])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, linestyle=':', alpha=0.6)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    class QueueFileLogger:
        def __init__(self, log_queue, log_file):
            self.log_queue = log_queue
            self.log_file = log_file
        def write(self, message):
            # Put message in queue for GUI display (no stripping to preserve newlines)
            self.log_queue.put(message)
            # Write to file, preserving exact output including newlines
            self.log_file.write(message)
            self.log_file.flush()  # Ensure immediate write to file
        def flush(self):
            self.log_file.flush()

    def create_input_fields(self, parent_frame):
        parent_frame.columnconfigure(1, weight=1)
        parent_frame.columnconfigure(3, weight=1)

        self.entries = {}
        defaults = [0.001, 0.01, 0.01, 0.1, 10000]
        labels = [
            "Emitter Radius (m):", "Collector Radius (m):", "Gap Distance (m):",
            "Length (m):", "Voltage (V):"
        ]
        row_num = 0
        for i, label in enumerate(labels):
            tk.Label(parent_frame, text=label).grid(row=row_num, column=0, padx=5, pady=2, sticky="w")
            entry = tk.Entry(parent_frame)
            entry.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
            entry.insert(0, str(defaults[i]))
            self.entries[label] = entry
            row_num += 1

        tk.Label(parent_frame, text="Emitter Shape:").grid(row=row_num, column=0, padx=5, pady=2, sticky="w")
        self.shape_var = tk.StringVar(self)
        self.shape_var.set('cylindrical')
        shape_menu = tk.OptionMenu(parent_frame, self.shape_var, 'cylindrical', 'pointed', 'hexagonal')
        shape_menu.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
        row_num += 1

        materials_list = list(self.materials_info.get('materials', {'Default':{}}).keys())
        if not materials_list: materials_list = ['Default']

        tk.Label(parent_frame, text="Emitter Material:").grid(row=row_num, column=0, padx=5, pady=2, sticky="w")
        self.mat_emitter_var = tk.StringVar(self)
        self.mat_emitter_var.set(materials_list[0])
        mat_emitter_menu = tk.OptionMenu(parent_frame, self.mat_emitter_var, *materials_list)
        mat_emitter_menu.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
        row_num += 1

        tk.Label(parent_frame, text="Collector Material:").grid(row=row_num, column=0, padx=5, pady=2, sticky="w")
        self.mat_collector_var = tk.StringVar(self)
        self.mat_collector_var.set(materials_list[0])
        mat_collector_menu = tk.OptionMenu(parent_frame, self.mat_collector_var, *materials_list)
        mat_collector_menu.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
        row_num += 1

        self.grid_res_entry = tk.Entry(parent_frame)
        self.coupling_iters_entry = tk.Entry(parent_frame)

        tk.Label(parent_frame, text="Grid Resolution (Nx,Ny,Nz):").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.grid_res_entry.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
        self.grid_res_entry.insert(0, "64,64,128")

        tk.Label(parent_frame, text="Coupling Iterations:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self.coupling_iters_entry.grid(row=1, column=3, padx=5, pady=2, sticky="ew")
        self.coupling_iters_entry.insert(0, "10")

        self.run_button = tk.Button(parent_frame, text="Run Simulation & Visualize", command=self.start_simulation, width=25)
        self.run_button.grid(row=row_num, column=0, columnspan=4, pady=10, padx=5)

    def _log_to_widget(self, message):
        """Safely logs a message to the ScrolledText widget."""
        try:
            if self.log_text.winfo_exists(): # Check if widget exists
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, str(message) + '\n')
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except tk.TclError:
            pass # Widget likely destroyed

    def _log_from_thread(self, message):
        """Puts a log message onto the queue from the simulation thread."""
        self.log_queue.put(message)

    def draw_electrodes_on_ax(self, ax, X_slice, Z_slice, r_e, r_c, d, l, shape_emitter, x_offset=0.0):
        """Helper to draw electrodes, accounting for collector X-offset."""
        try:
            if X_slice.ndim < 2 or Z_slice.ndim < 2 or X_slice.shape[0] <= 1 or Z_slice.shape[1] <= 1:
                self._log_to_widget("Warning: Cannot draw electrodes - slice dimensions invalid.")
                return
            x_coords = np.unique(X_slice[:, 0]); z_coords = np.unique(Z_slice[0, :])
            if len(x_coords) <= 1 or len(z_coords) <= 1:
                self._log_to_widget("Warning: Cannot draw electrodes - not enough unique coordinates in slice.")
                return
            x_min, x_max = x_coords.min(), x_coords.max(); z_min, z_max = z_coords.min(), z_coords.max()
            dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0.01
            dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else 0.01

            # Draw emitter (still centered at X=0)
            emitter_patch = None
            if shape_emitter == 'cylindrical' or shape_emitter == 'hexagonal':
                emitter_patch = Circle((0, 0), r_e, color='red', fill=False, lw=1.5, label='Emitter', zorder=10)
            elif shape_emitter == 'pointed':
                emitter_patch = Rectangle((-r_e/2, -dz/2), r_e, dz, color='red', fill=False, lw=1.5, label='Emitter', zorder=10)
            else:
                emitter_patch = Circle((0, 0), r_e, color='red', fill=False, lw=1.5, label='Emitter', zorder=10)
            if emitter_patch: ax.add_patch(emitter_patch)

            # Draw collector at offset position
            collector_z_pos = d
            line_thickness = max(dz, dx, 0.0001) * 2
            collector_line_kwargs = {'color': 'blue', 'lw': 1.5, 'zorder': 10}
            # Collector lines at x = x_offset ± r_c
            ax.plot([x_offset - r_c, x_offset - r_c], 
                    [collector_z_pos - line_thickness/2, collector_z_pos + line_thickness/2], 
                    label='Collector', **collector_line_kwargs)
            ax.plot([x_offset + r_c, x_offset + r_c], 
                    [collector_z_pos - line_thickness/2, collector_z_pos + line_thickness/2], 
                    **collector_line_kwargs)

            # Adjust plot limits considering the offset
            plot_buffer_x = r_c * 0.3
            plot_buffer_z_low = max(r_e * 3, l * 0.5, 0.005)
            plot_buffer_z_high = max((z_max - d)*0.3, 0.005) if z_max > d else 0.005
            z_plot_min = min(z_min, 0 - plot_buffer_z_low)
            z_plot_max = max(z_max, d + plot_buffer_z_high)
            # Adjust X limits to account for offset collector
            x_center = x_offset
            ax.set_xlim(x_center - r_c - plot_buffer_x, x_center + r_c + plot_buffer_x)
            ax.set_ylim(z_plot_min, z_plot_max)
            ax.set_aspect('equal', adjustable='box')

        except Exception as e:
            self._log_to_widget(f"Error drawing electrodes: {e}")

    def _run_simulation_thread(self, params_tuple, materials_info, grid_res_tuple, coupling_iters):
        tf.config.run_functions_eagerly(True)  # Enable eager execution for debugging
        self._log_from_thread("Simulation thread started.")
        
        # Set up output redirection with line buffering
        try:
            log_file = open('simulation_log.txt', 'w', buffering=1)  # Line buffering (buffering=1)
        except Exception as e:
            self._log_from_thread(f"Error opening log file: {e}")
            self.results_queue.put(Exception(f"Failed to open log file: {e}"))
            return

        # Create logger instance
        queue_logger = self.QueueFileLogger(self.log_queue, log_file)
        
        # Save and redirect stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = queue_logger
        sys.stderr = queue_logger

        try:
            r_e, r_c, d, l, V, shape_emitter, mat_emitter, mat_collector = params_tuple
            nx_base, ny_base, nz_base = grid_res_tuple

            altitude_m = 0.0
            pressure_atm = 1.0
            temperature_k = 300.0
            self._log_from_thread(f"Using standard env: Alt={altitude_m}m, P={pressure_atm}atm, T={temperature_k}K")

            emitter_props = get_material_properties(mat_emitter, materials_info, log_func=self._log_from_thread)
            E_onset = emitter_props['E_onset']
            townsend_A = TOWNSEND_A_DEFAULT
            townsend_B = TOWNSEND_B_DEFAULT
            self._log_from_thread(f"Using Material Props: A={townsend_A:.2f}, B={townsend_B:.2f}, E_onset={E_onset:.2e}")

            try:
                current_rho_air_tf, density_status_tf = air_density_tf(tf.constant(altitude_m, dtype=tf.float32))
                density_status = density_status_tf.numpy()
                if density_status != SUCCESS_CODE:
                    raise ValueError(f"Air density calculation failed with status {density_status}")
                current_rho_air = current_rho_air_tf.numpy()
                self._log_from_thread(f"Calculated Air Density: {current_rho_air:.4f} kg/m^3")
            except Exception as density_err:
                self._log_from_thread(f"ERROR during air density calculation: {density_err}")
                self.results_queue.put(density_err)
                return

            mu_air = MU_AIR

            tf_params_tuple = (float(r_e), float(r_c), float(d), float(l), float(V), str(shape_emitter), None, None)
            tf_grid_res_tuple = (int(nx_base), int(ny_base), int(nz_base))

            self._log_from_thread(f"Calling couple_physics_tf with {coupling_iters} iterations...")
            start_time = time.time()

            # Hardcode x_offset (matching your code)
            x_offset_hardcoded = 0.001
            # Conditionally pass log_func_for_dummy based on physics_code_loaded
            if physics_code_loaded:
                results_tuple = couple_physics_tf(
                    tf_params_tuple,
                    tf_grid_res_tuple,
                    int(coupling_iters),
                    float(pressure_atm),
                    float(temperature_k),
                    float(townsend_A),
                    float(townsend_B),
                    float(E_onset),
                    float(current_rho_air),
                    float(mu_air)
                )
            else:
                results_tuple = couple_physics_tf(
                    tf_params_tuple,
                    tf_grid_res_tuple,
                    int(coupling_iters),
                    float(pressure_atm),
                    float(temperature_k),
                    float(townsend_A),
                    float(townsend_B),
                    float(E_onset),
                    float(current_rho_air),
                    float(mu_air),
                    log_func_for_dummy=self._log_from_thread
                )

            end_time = time.time()
            self._log_from_thread(f"couple_physics_tf completed in {end_time - start_time:.2f} seconds.")
            self.results_queue.put(results_tuple)

        except Exception as e:
            tb_str = traceback.format_exc()
            self._log_from_thread(f"Error in simulation thread: {e}")
            self._log_from_thread(tb_str)
            self.results_queue.put(e)
        finally:
            # Restore stdout and stderr
            self._log_from_thread("Simulation thread finished.")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
            tf.config.run_functions_eagerly(False)  # Restore graph mode

    def start_simulation(self):
        if not physics_code_loaded:
            messagebox.showerror("Missing Code", "Physics simulation code (physix_single.py) failed to load. Cannot run.")
            return

        if self.simulation_thread and self.simulation_thread.is_alive():
            messagebox.showwarning("Busy", "A simulation is already running.")
            return

        self.run_button.config(state=tk.DISABLED)
        self._log_to_widget("="*20)
        self._log_to_widget("Starting simulation...")
        self.initialize_plots()

        try:
            params_tuple = self.get_current_params_tuple()
            if params_tuple is None: raise ValueError("Failed to read parameters.")
            grid_res_tuple = self.get_current_grid_res()
            if grid_res_tuple is None: raise ValueError("Failed to read grid resolution.")

            r_e, r_c, d, l, V, shape_emitter, mat_emitter, mat_collector = params_tuple
            nx_base, ny_base, nz_base = grid_res_tuple
            coupling_iters = int(self.coupling_iters_entry.get())

            if any(g <= 5 for g in grid_res_tuple): raise ValueError("Grid dimensions must be > 5.")
            if coupling_iters <= 0: raise ValueError("Coupling iterations must be positive.")
            if r_e <= 0 or r_c <= 0 or d <= 0 or l <= 0: raise ValueError("Geometric parameters must be positive.")
            if r_e >= r_c: raise ValueError("Emitter radius must be smaller than collector radius.")

        except ValueError as e:
            self._log_to_widget(f"Input Error: {e}")
            messagebox.showerror("Input Error", f"Invalid input parameter: {e}")
            self.run_button.config(state=tk.NORMAL)
            return
        except Exception as e:
            self._log_to_widget(f"Input Error: {e}")
            messagebox.showerror("Input Error", f"Error reading parameters: {e}")
            self.run_button.config(state=tk.NORMAL)
            return

        self._log_to_widget(f"Parameters: r_e={r_e:.2e}, r_c={r_c:.2e}, d={d:.2e}, l={l:.2f}, V={V:.1f}")
        self._log_to_widget(f"Shape={shape_emitter}, Emitter={mat_emitter}, Collector={mat_collector}")
        self._log_to_widget(f"Grid Base={grid_res_tuple}, Iterations={coupling_iters}")
        self._log_to_widget("Simulation running in background...")

        self.simulation_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(params_tuple, self.materials_info, grid_res_tuple, coupling_iters),
            daemon=True
        )
        self.simulation_thread.start()

    def process_queues(self):
        """ Checks queues periodically. Processes results in the main GUI thread. """
        # Process log messages
        try:
            while True:
                log_msg = self.log_queue.get_nowait()
                self._log_to_widget(log_msg)
        except queue.Empty:
            pass
        except Exception as e:
             print(f"Error processing log queue: {e}") # Log to console if GUI fails

        # Process simulation results/errors
        try:
            result = self.results_queue.get_nowait() # Check for simulation completion

            if isinstance(result, Exception):
                self._log_to_widget(f"Simulation failed: {result}")
                messagebox.showerror("Simulation Error", f"Simulation thread failed:\n{result}")
            elif isinstance(result, tuple): # <--- UPDATED: Expect tuple
                self._log_to_widget("Simulation thread finished, processing results tuple...")
                self.update_plots_with_results(result) # Pass the tuple
            else:
                self._log_to_widget(f"Unknown result type from simulation thread: {type(result)}")
                messagebox.showerror("Result Error", f"Received unexpected result type: {type(result)}")

            # Re-enable button once finished (success or failure)
            self.run_button.config(state=tk.NORMAL)
            self.simulation_thread = None

        except queue.Empty:
            # Simulation still running or queue empty
            pass
        except Exception as e:
             # Catch errors during queue processing/plotting itself
             self._log_to_widget(f"Error processing result queue or plotting: {e}")
             self._log_to_widget(traceback.format_exc())
             messagebox.showerror("GUI Error", f"Error processing results or plotting:\n{e}")
             self.run_button.config(state=tk.NORMAL) # Re-enable on error
             self.simulation_thread = None

        # Schedule the next check only if the window still exists
        if self.winfo_exists():
            self.after(100, self.process_queues) # Check every 100ms

    def interpolate_with_fallback(self, points, values, xi, yi):
        """Interpolates data with fallback from 'nearest' to 'linear' and handles NaNs."""
        try:
            interpolated = griddata(points, values.ravel(), (xi, yi), method='nearest', fill_value=0.0)
            if np.any(np.isnan(interpolated)):
                self._log_to_widget("Warning: NaN detected in 'nearest' interpolation, falling back to 'linear'.")
                interpolated = griddata(points, values.ravel(), (xi, yi), method='linear', fill_value=0.0)
            return np.nan_to_num(interpolated, nan=0.0)
        except Exception as e:
            self._log_to_widget(f"Interpolation error: {e}. Returning zero array.")
            return np.zeros_like(xi)
        
    def update_plots_with_results(self, results_tuple):
        """
        Updates the plots using the simulation results tuple. Called from the main thread.
        Converts TF Tensors to NumPy arrays.
        """
        self._log_to_widget("Processing results tuple and updating plots...")
        # Clear plots immediately in case of error later
        self.initialize_plots()
        try:
            # --- Unpack results tuple ---
            # Ensure this matches the return signature of couple_physics_tf EXACTLY
            # Expected: (ux, uy, uz, n_e, n_i, phi, Ex, Ey, Ez, p, T_motor, status, dx, dy, dz_scalar)
            # Check length first
            if not isinstance(results_tuple, tuple):
                self._log_to_widget(f"ERROR: Invalid results_tuple received. Type: {type(results_tuple)}")
                print(f"ERROR: Invalid results_tuple received. Type: {type(results_tuple)}")
                return
            if len(results_tuple) != 15:
                self._log_to_widget(f"ERROR: Invalid results_tuple length. Expected 15, Got: {len(results_tuple)}")
                print(f"ERROR: Invalid results_tuple length. Expected 15, Got: {len(results_tuple)}")
                return

            (ux_tf, uy_tf, uz_tf, n_e_tf, n_i_tf, phi_tf,
            Ex_tf, Ey_tf, Ez_tf, p_tf, T_motor_tf, status_code_tf,
            dx_tf, dy_tf, dz_scalar_avg_tf) = results_tuple

            # --- Convert Status Code ---
            if isinstance(status_code_tf, tf.Tensor):
                status_code = int(status_code_tf.numpy())  # Get Python int value
            else:
                try:
                    status_code = int(status_code_tf)
                except (ValueError, TypeError) as conv_err:
                    self._log_to_widget(f"ERROR: Could not convert status code '{status_code_tf}' to int. Error: {conv_err}")
                    print(f"ERROR: Could not convert status code '{status_code_tf}' to int. Error: {conv_err}")
                    return  # Stop processing if status code is invalid

            self._log_to_widget(f"Simulation finished with Status Code: {status_code}")

            # --- Handle Non-Success Status Codes ---
            error_map = {
                0: "Success",
                2: "NaN/Inf detected during simulation step.",
                3: "NaN/Inf detected in final simulation output.",
                4: "Numerical instability (NaN/Inf) in Navier-Stokes solver.",
                5: "Grid generation error.",
                6: "Initialization error (e.g., electrodes, air density).",
                99: "Unknown exception during simulation."
            }
            if status_code != 0:
                error_msg = error_map.get(status_code, f"Unknown error code {status_code}.")
                self._log_to_widget(f"ERROR: {error_msg}")
                print(f"ERROR: Simulation failed: {error_msg} (Code: {status_code})")
                messagebox.showerror("Simulation Error", f"Simulation failed (Code: {status_code}):\n{error_msg}")
                return

            # --- Convert Tensors to NumPy Arrays (only if status_code == 0) ---
            self._log_to_widget("Converting TF Tensors to NumPy arrays...")
            start_conv = time.time()
            # Helper to safely convert tensor to numpy array
            def safe_to_numpy(tensor):
                if tensor is None: return None
                try:
                    # Check if it's already a NumPy array (useful for dummy data)
                    if isinstance(tensor, np.ndarray):
                        return tensor
                    # Convert TF Tensor
                    if isinstance(tensor, tf.Tensor):
                        return tensor.numpy()
                    # Handle potential scalar values returned directly
                    if np.isscalar(tensor):
                        # Ensure correct dtype (e.g., float for T_motor)
                        if isinstance(tensor, (tf.Variable, tf.Tensor)):
                            return tensor.numpy()
                        return tensor  # Assume correct type if scalar
                    # If not recognized, return None and log warning
                    self._log_to_widget(f"Warning: Unrecognized type for conversion: {type(tensor)}")
                    print(f"Warning: Unrecognized type for conversion: {type(tensor)}")
                    return None
                except Exception as conv_e:
                    self._log_to_widget(f"Error during tensor conversion: {conv_e}")
                    print(f"Error during tensor conversion: {conv_e}")
                    return None

            ux = np.asarray(safe_to_numpy(ux_tf)) if ux_tf is not None else None
            uy = np.asarray(safe_to_numpy(uy_tf)) if uy_tf is not None else None
            uz = np.asarray(safe_to_numpy(uz_tf)) if uz_tf is not None else None
            n_e = np.asarray(safe_to_numpy(n_e_tf)) if n_e_tf is not None else None
            n_i = np.asarray(safe_to_numpy(n_i_tf)) if n_i_tf is not None else None
            phi = np.asarray(safe_to_numpy(phi_tf)) if phi_tf is not None else None
            Ex = np.asarray(safe_to_numpy(Ex_tf)) if Ex_tf is not None else None
            Ey = np.asarray(safe_to_numpy(Ey_tf)) if Ey_tf is not None else None
            Ez = np.asarray(safe_to_numpy(Ez_tf)) if Ez_tf is not None else None
            p = np.asarray(safe_to_numpy(p_tf)) if p_tf is not None else None
            T_motor_final = float(safe_to_numpy(T_motor_tf)) if T_motor_tf is not None and safe_to_numpy(T_motor_tf) is not None else np.nan
            dx = safe_to_numpy(dx_tf) if dx_tf is not None else None
            dy = safe_to_numpy(dy_tf) if dy_tf is not None else None
            dz_scalar = safe_to_numpy(dz_scalar_avg_tf) if dz_scalar_avg_tf is not None else None

            # Check if any necessary array is None after conversion attempt
            required_arrays = [ux, uy, uz, n_e, n_i, Ex, Ey, Ez]
            if any(arr is None for arr in required_arrays):
                missing = [name for name, arr in zip(['ux', 'uy', 'uz', 'n_e', 'n_i', 'Ex', 'Ey', 'Ez'], required_arrays) if arr is None]
                self._log_to_widget(f"ERROR: Required data arrays are None after conversion: {', '.join(missing)}. Cannot plot.")
                print(f"ERROR: Required data arrays are None after conversion: {', '.join(missing)}. Cannot plot.")
                return

            end_conv = time.time()
            self._log_to_widget(f"Tensor conversion took {end_conv - start_conv:.2f} seconds.")

            # --- Get Grid Coordinates (Recalculate and Convert) ---
            self._log_to_widget("Recalculating grid coordinates for plotting...")
            params_tuple = self.get_current_params_tuple()
            grid_res_tuple = self.get_current_grid_res()
            if params_tuple is None or grid_res_tuple is None:
                self._log_to_widget("ERROR: Failed to get parameters or grid resolution for plotting.")
                return  # Error logged in helpers

            r_e, r_c, d, l, V, shape_emitter, mat_emitter, mat_collector = params_tuple
            nx_base, ny_base, nz_base = grid_res_tuple

            # Call setup_grid_tf to get grid tensors
            try:
                # Assuming setup_grid_tf returns: x_1d_tf, y_1d_tf, z_1d_tf, nx_tf, ny_tf, nz_tf
                x_1d_tf, y_1d_tf, z_1d_tf, nx_tf, ny_tf, nz_tf = setup_grid_tf(
                    tf.constant(r_e, dtype=tf.float32), tf.constant(r_c, dtype=tf.float32),
                    tf.constant(d, dtype=tf.float32), tf.constant(l, dtype=tf.float32),
                    tf.constant(nx_base, dtype=tf.int32),
                    tf.constant(ny_base, dtype=tf.int32),
                    tf.constant(nz_base, dtype=tf.int32)
                )
                # Convert 1D coordinate tensors to NumPy arrays
                x_1d = safe_to_numpy(x_1d_tf)
                y_1d = safe_to_numpy(y_1d_tf)
                z_1d = safe_to_numpy(z_1d_tf)
                nx_val = int(safe_to_numpy(nx_tf)) if nx_tf is not None else 0
                ny_val = int(safe_to_numpy(ny_tf)) if ny_tf is not None else 0
                nz_val = int(safe_to_numpy(nz_tf)) if nz_tf is not None else 0

                if any(c is None for c in [x_1d, y_1d, z_1d]) or any(v == 0 for v in [nx_val, ny_val, nz_val]):
                    raise ValueError("Grid coordinate generation failed or resulted in zero dimensions.")

                # Generate 3D meshgrid for plotting
                X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

            except Exception as grid_err:
                self._log_to_widget(f"ERROR during grid coordinate generation for plotting: {grid_err}")
                print(f"ERROR during grid coordinate generation for plotting: {grid_err}")
                return

            # --- Validate Shapes ---
            try:
                expected_shape = (nx_val, ny_val, nz_val)
                if X.shape != expected_shape: raise ValueError(f"Grid X shape {X.shape} != expected {expected_shape}")
                if n_i.shape != expected_shape: raise ValueError(f"Ion density n_i shape {n_i.shape} != expected {expected_shape}")
                if Ex.shape != expected_shape: raise ValueError(f"Field Ex shape {Ex.shape} != expected {expected_shape}")
                if ux.shape != expected_shape: raise ValueError(f"Velocity ux shape {ux.shape} != expected {expected_shape}")
            except ValueError as shape_err:
                self._log_to_widget(f"ERROR: Shape mismatch after tensor conversion. {shape_err}")
                print(f"ERROR: Shape mismatch after tensor conversion. {shape_err}")
                return
            except Exception as E:
                self._log_to_widget(f"ERROR during shape validation: {E}")
                print(f"ERROR during shape validation: {E}")
                return

            # --- Calculate Thrust (using TF function with returned grid spacings) ---
            self._log_to_widget("Calculating thrust using TF function with returned grid spacings...")
            Tx, Ty, Tz, Tmag = np.nan, np.nan, np.nan, np.nan
            try:
                if n_e is None:
                    self._log_to_widget("Warning: Electron density (n_e) is None, cannot calculate charge density for thrust.")
                else:
                    rho_charge_np = ELEM_CHARGE * (n_i - n_e)
                    rho_charge_tf_thrust = tf.constant(rho_charge_np, dtype=tf.float32)
                    Ex_tf_thrust = tf.constant(Ex, dtype=tf.float32)
                    Ey_tf_thrust = tf.constant(Ey, dtype=tf.float32)
                    Ez_tf_thrust = tf.constant(Ez, dtype=tf.float32)
                    dx_thrust = tf.constant(dx, dtype=tf.float32) if dx is not None else tf.constant(0.01, dtype=tf.float32)
                    dy_thrust = tf.constant(dy, dtype=tf.float32) if dy is not None else tf.constant(0.01, dtype=tf.float32)
                    dz_scalar_thrust = tf.constant(dz_scalar, dtype=tf.float32) if dz_scalar is not None else tf.constant(0.01, dtype=tf.float32)

                    altitude_m = 0.0

                    Tx_tf, Ty_tf, Tz_tf, Tmag_tf, thrust_status_tf = calculate_thrust_tf(
                        rho_in=rho_charge_tf_thrust,
                        Ex_in=Ex_tf_thrust,
                        Ey_in=Ey_tf_thrust,
                        Ez_in=Ez_tf_thrust,
                        dx_in=dx_thrust,
                        dy_in=dy_thrust,
                        dz_scalar_in=dz_scalar_thrust,
                        altitude_m_in=tf.constant(altitude_m, dtype=tf.float32)
                    )
                    thrust_status = int(thrust_status_tf.numpy())
                    if thrust_status == 0:
                        Tx = Tx_tf.numpy()
                        Ty = Ty_tf.numpy()
                        Tz = Tz_tf.numpy()
                        Tmag = Tmag_tf.numpy()
                        self._log_to_widget(f"Calculated Thrust (TF, Sea Level): Tx={Tx:.3e}, Ty={Ty:.3e}, Tz={Tz:.3e}, Mag={Tmag:.3e} N")
                    else:
                        self._log_to_widget(f"Warning: Thrust calculation failed with status {thrust_status}.")

            except Exception as thrust_err:
                self._log_to_widget(f"Error during thrust calculation: {thrust_err}")
                self._log_to_widget(traceback.format_exc())

            self._log_to_widget(f"Final Estimated Motor Temperature: {T_motor_final:.2f} K")

            # --- Visualization: 2D slice at Y=mid ---
            nx, ny, nz = X.shape
            if ny == 0:
                self._log_to_widget("ERROR: Grid Y dimension (ny) is zero after conversion. Cannot plot.")
                print("ERROR: Grid Y dimension (ny) is zero after conversion. Cannot plot.")
                return
            y_mid_index = ny // 2

            try:
                X_slice = X[:, y_mid_index, :]
                Z_slice = Z[:, y_mid_index, :]
                if X_slice.size == 0 or Z_slice.size == 0:
                    raise ValueError("Slice creation resulted in an empty array.")
                if X_slice.ndim != 2 or Z_slice.ndim != 2:
                    raise ValueError(f"Slice dimensions are not 2D. X_slice: {X_slice.ndim}D, Z_slice: {Z_slice.ndim}D")

                Ex_slice = Ex[:, y_mid_index, :]
                Ey_slice = Ey[:, y_mid_index, :]
                Ez_slice = Ez[:, y_mid_index, :]
                E_mag_slice = np.sqrt(Ex_slice**2 + Ey_slice**2 + Ez_slice**2 + 1e-12)

                if n_i is None:
                    self._log_to_widget("Error: Ion density (n_i) is None, cannot create slice for plotting.")
                    return
                n_i_slice = n_i[:, y_mid_index, :]
                n_i_plot = np.maximum(n_i_slice, 1e1)

                ux_slice = ux[:, y_mid_index, :]
                uy_slice = uy[:, y_mid_index, :]
                uz_slice = uz[:, y_mid_index, :]
                u_mag_slice = np.sqrt(ux_slice**2 + uy_slice**2 + uz_slice**2 + 1e-12)

                # Get original non-uniform coordinates from the slice itself
                x_coords = X_slice[:, 0]
                z_coords = Z_slice[0, :]

                if x_coords.size == 0 or z_coords.size == 0:
                    raise ValueError("Coordinate extraction from slice failed (empty coords).")

                # Generate uniform coordinates for streamplot interpolation target
                x_uniform = np.linspace(x_coords.min(), x_coords.max(), len(x_coords))
                z_uniform = np.linspace(z_coords.min(), z_coords.max(), len(z_coords))
                X_uniform, Z_uniform = np.meshgrid(x_uniform, z_uniform, indexing='ij')

                # Prepare points for interpolation (from the non-uniform slice)
                points = np.column_stack((X_slice.ravel(), Z_slice.ravel()))

                # Interpolate electric field components onto the uniform grid
                Ex_uniform = griddata(points, Ex_slice.ravel(), (X_uniform, Z_uniform), method='linear')
                Ez_uniform = griddata(points, Ez_slice.ravel(), (X_uniform, Z_uniform), method='linear')
                Ex_uniform = np.nan_to_num(Ex_uniform, nan=0.0)
                Ez_uniform = np.nan_to_num(Ez_uniform, nan=0.0)

                # Interpolate velocity components onto the uniform grid with fallback
                def interpolate_with_fallback(points, values, xi, yi):
                    try:
                        interpolated = griddata(points, values.ravel(), (xi, yi), method='nearest', fill_value=0.0)
                        if np.any(np.isnan(interpolated)):
                            interpolated = griddata(points, values.ravel(), (xi, yi), method='linear', fill_value=0.0)
                        return np.nan_to_num(interpolated, nan=0.0)
                    except Exception:
                        return np.zeros_like(xi)

                ux_uniform = interpolate_with_fallback(points, ux_slice, X_uniform, Z_uniform)
                uz_uniform = interpolate_with_fallback(points, uz_slice, X_uniform, Z_uniform)

            except IndexError as slice_err:
                self._log_to_widget(f"ERROR creating slices at y_mid_index={y_mid_index} (ny={ny}): {slice_err}")
                print(f"ERROR creating slices at y_mid_index={y_mid_index} (ny={ny}): {slice_err}")
                return
            except ValueError as slice_val_err:
                self._log_to_widget(f"ERROR during slice validation or coordinate extraction: {slice_val_err}")
                print(f"ERROR during slice validation or coordinate extraction: {slice_val_err}")
                return
            except Exception as interp_err:
                self._log_to_widget(f"ERROR during interpolation for plotting: {interp_err}")
                print(f"ERROR during interpolation for plotting: {interp_err}")
                return

            # --- Update Plots ---
            self._log_to_widget("Updating Matplotlib plots...")

            # --- Plot 1: Electric Field ---
            try:
                ax1 = self.axs[0]
                ax1.clear()
                ax1.set_title('Electric Field (V/m) & Lines')
                E_max_val = np.nanmax(E_mag_slice)
                E_min_val = np.nanmin(E_mag_slice)
                if not np.isfinite(E_min_val) or not np.isfinite(E_max_val) or E_max_val <= E_min_val:
                    E_min_val = 0.0
                    E_max_val = 1.0
                    self._log_to_widget("Warning: Invalid E_mag range, using default [0, 1].")
                norm_E = Normalize(vmin=E_min_val, vmax=E_max_val)
                cf1 = ax1.contourf(X_slice, Z_slice, E_mag_slice, cmap='magma', levels=100, norm=norm_E, extend='both')

                if hasattr(self, '_cbar1') and self._cbar1:
                    try: self.fig.delaxes(self._cbar1.ax)
                    except (KeyError, ValueError, AttributeError): pass
                self._cbar1 = self.fig.colorbar(cf1, ax=ax1, label='E Field Mag (V/m)')

                speed_E_xz = np.sqrt(Ex_uniform**2 + Ez_uniform**2)
                max_speed_E_xz = np.nanmax(speed_E_xz)
                if np.isfinite(max_speed_E_xz) and max_speed_E_xz > 1e-9 and len(x_uniform) > 1 and len(z_uniform) > 1:
                    lw = np.clip(2 * speed_E_xz / max_speed_E_xz + 0.5, 0.5, 2.5)
                    lw = np.nan_to_num(lw, nan=0.5)
                    ax1.streamplot(x_uniform, z_uniform, Ex_uniform.T, Ez_uniform.T, density=1.0, color='white',
                                linewidth=lw.T, arrowstyle='->', arrowsize=0.8)
                else:
                    self._log_to_widget("Skipping E streamplot due to low field magnitude, insufficient grid points, or NaN values.")
                self.draw_electrodes_on_ax(ax1, X_slice, Z_slice, r_e, r_c, d, l, shape_emitter, x_offset=0.001)
                ax1.grid(True, linestyle=':', alpha=0.6)
                ax1.set_xlabel("X (m)")
                ax1.set_ylabel("Z (m)")
                ax1.set_aspect('equal', adjustable='box')
            except Exception as ax1_err:
                self._log_to_widget(f"Error updating Plot 1 (E-Field): {ax1_err}")
                self._log_to_widget(traceback.format_exc())

            # --- Plot 2: Ion Density ---
            try:
                ax2 = self.axs[1]
                ax2.clear()
                ax2.set_title('Ion Density (m⁻³, log scale)')
                min_log_val = np.nanmin(n_i_plot[n_i_plot > 0]) if np.any(n_i_plot > 0) else 1e1
                max_log_val = np.nanmax(n_i_plot)

                use_log = True
                if not np.isfinite(min_log_val) or not np.isfinite(max_log_val) or max_log_val <= min_log_val:
                    min_log_val = 1e1
                    max_log_val = 1e6
                    use_log = False
                    self._log_to_widget(f"Warning: Invalid range for n_i log plot. Using linear [{min_log_val:.1e}, {max_log_val:.1e}].")

                if hasattr(self, '_cbar2') and self._cbar2:
                    try: self.fig.delaxes(self._cbar2.ax)
                    except (KeyError, ValueError, AttributeError): pass

                if use_log:
                    try:
                        norm_ni = LogNorm(vmin=min_log_val, vmax=max_log_val)
                        levels = np.logspace(np.log10(min_log_val), np.log10(max_log_val), 50)
                        cf2 = ax2.contourf(X_slice, Z_slice, n_i_plot, cmap='cividis', levels=levels, norm=norm_ni, extend='max')
                        self._cbar2 = self.fig.colorbar(cf2, ax=ax2, label='Ion Density (m⁻³)')
                    except ValueError as log_err:
                        use_log = False
                        self._log_to_widget(f"Log plot failed for n_i ({log_err}), trying linear scale.")

                if not use_log:
                    lin_min = np.nanmin(n_i_slice)
                    lin_max = np.nanmax(n_i_slice)
                    if not np.isfinite(lin_min) or not np.isfinite(lin_max) or lin_max <= lin_min:
                        lin_min = 0
                        lin_max = max(1e6, lin_min + 1.0)
                    norm_ni_lin = Normalize(vmin=lin_min, vmax=lin_max)
                    cf2 = ax2.contourf(X_slice, Z_slice, n_i_slice, cmap='cividis', levels=50, norm=norm_ni_lin, extend='max')
                    self._cbar2 = self.fig.colorbar(cf2, ax=ax2, label='Ion Density (m⁻³, linear)')

                self.draw_electrodes_on_ax(ax2, X_slice, Z_slice, r_e, r_c, d, l, shape_emitter, x_offset=0.001)
                ax2.grid(True, linestyle=':', alpha=0.6)
                ax2.set_xlabel("X (m)")
                ax2.set_ylabel("Z (m)")
                ax2.set_aspect('equal', adjustable='box')
            except Exception as ax2_err:
                self._log_to_widget(f"Error updating Plot 2 (Ion Density): {ax2_err}")
                self._log_to_widget(traceback.format_exc())

            # --- Plot 3: Velocity ---
            try:
                ax3 = self.axs[2]
                ax3.clear()
                ax3.set_title('Fluid Velocity (m/s) & Streamlines')
                u_max_val = np.nanmax(u_mag_slice)
                u_min_val = np.nanmin(u_mag_slice)
                if not np.isfinite(u_min_val) or not np.isfinite(u_max_val) or u_max_val <= u_min_val:
                    u_min_val = 0.0
                    u_max_val = 1.0
                    self._log_to_widget("Warning: Invalid u_mag range, using default [0, 1].")

                norm_u = Normalize(vmin=u_min_val, vmax=u_max_val)
                cf3 = ax3.contourf(X_slice, Z_slice, u_mag_slice, cmap='jet', levels=150, norm=norm_u, extend='both')

                if hasattr(self, '_cbar3') and self._cbar3:
                    try: self.fig.delaxes(self._cbar3.ax)
                    except (KeyError, ValueError, AttributeError): pass
                self._cbar3 = self.fig.colorbar(cf3, ax=ax3, label='Velocity Mag (m/s)')

                speed_u_xz = np.sqrt(ux_uniform**2 + uz_uniform**2)
                max_speed_u_xz = np.nanmax(speed_u_xz)
                if len(x_uniform) < 5 or len(z_uniform) < 5 or np.any(np.isnan(ux_uniform)) or np.any(np.isnan(uz_uniform)):
                    self._log_to_widget("Warning: Insufficient grid points or NaN values detected. Skipping streamplot.")
                else:
                    if np.isfinite(max_speed_u_xz) and max_speed_u_xz > 1e-3 and len(x_uniform) > 1 and len(z_uniform) > 1:
                        lw_u = np.clip(2 * speed_u_xz / max_speed_u_xz + 0.5, 0.5, 2.5)
                        lw_u = np.nan_to_num(lw_u, nan=0.5)
                        ax3.streamplot(x_uniform, z_uniform, ux_uniform.T, uz_uniform.T, density=1.2, color='black',
                                    linewidth=lw_u.T, arrowstyle='->', arrowsize=0.8)
                    else:
                        self._log_to_widget("Skipping U streamplot due to low velocity magnitude, insufficient grid points, or NaN values.")
                self.draw_electrodes_on_ax(ax3, X_slice, Z_slice, r_e, r_c, d, l, shape_emitter, x_offset=0.001)
                ax3.grid(True, linestyle=':', alpha=0.6)
                ax3.set_xlabel("X (m)")
                ax3.set_ylabel("Z (m)")
                ax3.set_aspect('equal', adjustable='box')
            except Exception as ax3_err:
                self._log_to_widget(f"Error updating Plot 3 (Velocity): {ax3_err}")
                self._log_to_widget(traceback.format_exc())

            # Final Draw and Log
            try:
                try:
                    self.fig.tight_layout(pad=2.5, rect=[0, 0.03, 1, 0.95])
                except ValueError as tl_err:
                    self._log_to_widget(f"Warning: tight_layout failed ({tl_err}). Plot spacing might be suboptimal.")
                self.canvas.draw()
                self._log_to_widget("Visualization updated.")
                self._log_to_widget("="*20)
            except Exception as draw_err:
                self._log_to_widget(f"Error during final draw/tight_layout: {draw_err}")

        except Exception as e:
            self._log_to_widget(f"Error during plotting/results processing: {str(e)}")
            self._log_to_widget(traceback.format_exc())
            print(f"Error during plotting/results processing: {str(e)}\n{traceback.format_exc()}")
            try:
                self.initialize_plots()
                self.canvas.draw()
            except Exception as reset_err:
                self._log_to_widget(f"Error trying to reset plots after failure: {reset_err}")

    def get_current_params_tuple(self):
        """Helper to get current parameters from GUI fields."""
        try:
            r_e = float(self.entries["Emitter Radius (m):"].get())
            r_c = float(self.entries["Collector Radius (m):"].get())
            d = float(self.entries["Gap Distance (m):"].get())
            l = float(self.entries["Length (m):"].get())
            V = float(self.entries["Voltage (V):"].get())
            shape_emitter = self.shape_var.get()
            mat_emitter = self.mat_emitter_var.get()
            mat_collector = self.mat_collector_var.get()
            return (r_e, r_c, d, l, V, shape_emitter, mat_emitter, mat_collector)
        except Exception as e:
            self._log_to_widget(f"Error reading parameters: {e}")
            messagebox.showerror("Input Error", f"Could not read parameters: {e}")
            return None

    def get_current_grid_res(self):
        """Helper to get current grid resolution from GUI."""
        try:
            grid_str = self.grid_res_entry.get().split(',')
            if len(grid_str) != 3: raise ValueError("Grid resolution must be 3 comma-separated integers.")
            grid_res = tuple(map(int, grid_str))
            if any(g <= 0 for g in grid_res): raise ValueError("Grid dimensions must be positive.")
            return grid_res
        except Exception as e:
            self._log_to_widget(f"Error reading grid resolution: {e}")
            messagebox.showerror("Input Error", f"Invalid grid resolution: {e}")
            return None


if __name__ == "__main__":
    # Check if physics code loaded properly before starting GUI
    if not physics_code_loaded:
         print("\n*** FAILED TO LOAD physics_single.py ***")
         print("GUI will run with DUMMY physics functions.\n")
         # Optionally prevent GUI start:
         # messagebox.showerror("Load Error", "Failed to load physics_single.py. Cannot start GUI.")
         # exit() # Or sys.exit()

    app = EHDSimulationGUI()
    app.mainloop()

# || END MODIFIED GUI CODE WITH THREADING (TF COMPATIBLE) ||