from pathlib import Path
import threading
from core.iad_model import IADModel
import core.iad_core as iad_core
import dearpygui.dearpygui as dpg
import time
import sys
import os
import subprocess
import shutil
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from threading import Thread
import itertools
import queue


def __init__(self):
    self.model = IADModel()
    self.g_value = 0.8
    self.reference_wavelength = 600
    self.fit_min = 600
    self.fit_max = 750
    self.use_dual_beam = False
    self.run_dir = self.get_run_directory()
    self.iad_exe = os.path.join(self.run_dir, "iad.exe")
    self.input_dir = os.path.join(self.run_dir, "iad_inputs")
    self.output_dir = os.path.join(self.run_dir, "iad_outputs")
    self.current_step = "setup"
    self.fit_results = []
    self.processing = False
    
    # Ensure directories exist
    os.makedirs(self.output_dir, exist_ok=True)
    
    # Initialize DearPyGUI
    dpg.create_context()
    dpg.create_viewport(title="IAD Optical Analysis Suite", width=900, height=700)
    
    # Create the GUI
    self.create_gui()
    
    # In __init__:
    self.gui_queue = queue.Queue()
    
    
def get_run_directory(self):
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def log_message(self, message, level="INFO"):
    """Add message to the log window with color coding and auto-scroll"""
    timestamp = time.strftime("%H:%M:%S")
    color_map = {
        "INFO": (120, 120, 120),
        "WARNING": (255, 165, 0),
        "ERROR": (255, 0, 0),
        "SUCCESS": (0, 180, 0)
    }
    color = color_map.get(level, (120, 120, 120))
    log_text = f"[{timestamp}] [{level}] {message}"
    if dpg.does_item_exist("log_scroll"):
        dpg.add_text(log_text, color=color, parent="log_scroll")
        # Auto-scroll to bottom
        dpg.set_y_scroll("log_scroll", -1.0)

def check_iad_executable(self):
    """Check if iad.exe exists"""
    exists = os.path.exists(self.iad_exe)
    if exists:
        self.log_message(f"iad.exe found at: {self.iad_exe}")
    else:
        self.log_message(f"iad.exe NOT found at: {self.iad_exe}", "ERROR")
    return exists

def count_input_files(self):
    """Count .rxt files in input directory"""
    if not os.path.exists(self.input_dir):
        return 0
    return len([f for f in os.listdir(self.input_dir) if f.endswith(".rxt")])

def update_status(self, force_reset=False):
    """Update the status indicators with more detailed information"""
    # Always check iad.exe and input files
    iad_exists = self.check_iad_executable()
    dpg.set_value("iad_status", "Found" if iad_exists else "Missing")
    dpg.configure_item("iad_status", color=(0, 255, 0) if iad_exists else (255, 0, 0))
    file_count = self.count_input_files()
    dpg.set_value("files_status", f"{file_count} files" if file_count > 0 else "No files found")
    dpg.configure_item("files_status", color=(0, 255, 0) if file_count > 0 else (255, 165, 0))
    if force_reset:
        dpg.set_value("progress_bar", 0.0)
        dpg.set_value("progress_text", "Step 0/3: Ready to begin")
        dpg.set_value("status_label", "Ready")
        return
    
    # Update progress based on completed steps
    steps_completed = 0
    step_details = []
    
    # Check each step individually, only advance if previous step is complete
    scatter_file = os.path.join(self.output_dir, "scattering_values_multi.csv")
    powerlaw_file = os.path.join(self.output_dir, "powerlaw_summary.csv")
    final_file = os.path.join(self.output_dir, "combined_output.txt")
    
    # Step 1: Initial analysis
    if os.path.exists(scatter_file):
        steps_completed = 1
        step_details.append("Initial analysis complete")
        # Step 2: Power law fitting (only if step 1 is complete)
        if os.path.exists(powerlaw_file):
            steps_completed = 2
            step_details.append("Power law fitting complete")
            # Step 3: Final analysis (only if step 2 is complete)
            if os.path.exists(final_file):
                steps_completed = 3
                step_details.append("Final analysis complete")
    progress = steps_completed / 3.0
    dpg.set_value("progress_bar", progress)
    # Update progress text with detailed status
    if steps_completed == 0:
        status_text = "Step 0/3: Ready to begin"
        if file_count > 0:
            status_text += f" ({file_count} files ready for analysis)"
    else:
        status_text = f"Step {steps_completed}/3: {' -> '.join(step_details)}"
    dpg.set_value("progress_text", status_text)
    # Update status label with current state
    if self.processing:
        current_step = steps_completed + 1
        if current_step == 1:
            dpg.set_value("status_label", "Running initial analysis...")
        elif current_step == 2:
            dpg.set_value("status_label", "Fitting power law...")
        elif current_step == 3:
            dpg.set_value("status_label", "Running final analysis...")
    else:
        if steps_completed == 0:
            dpg.set_value("status_label", "Ready to begin analysis")
        elif steps_completed == 1:
            dpg.set_value("status_label", "Ready for power law fitting")
        elif steps_completed == 2:
            dpg.set_value("status_label", "Ready for final analysis")
        elif steps_completed == 3:
            dpg.set_value("status_label", "Analysis complete")

def on_g_value_change(self, sender, value):
    self.g_value = value
    self.log_message(f"Anisotropy value (g) set to: {value:.3f}")
    # Keep slider/input in sync if needed (for future compatibility)
    if dpg.does_item_exist("g_slider"):
        dpg.set_value("g_slider", value)
    if dpg.does_item_exist("g_input"):
        dpg.set_value("g_input", value)

def on_dual_beam_change(self, sender, value):
    self.use_dual_beam = value
    status = "ENABLED" if value else "DISABLED"
    self.log_message(f"Dual beam correction {status}")

def on_lambda0_change(self, sender, value):
    self.reference_wavelength = value
    self.log_message(f"Reference wavelength (lambda0) set to: {value} nm")

def on_fit_min_change(self, sender, value):
    self.fit_min = value
    self.log_message(f"Fit window min set to: {value} nm")

def on_fit_max_change(self, sender, value):
    self.fit_max = value
    self.log_message(f"Fit window max set to: {value} nm")

def run_initial_iad(self, sender=None, data=None):
    """Step 1: Run initial IAD analysis"""
    if self.processing:
        return
    
    self.processing = True
    dpg.set_value("btn_step1", "Processing...")
    dpg.configure_item("btn_step1", enabled=False)
    dpg.set_value("status_label", "Running Initial Analysis")
    dpg.set_value("progress_text", "Step 1/3: Running initial analysis...")
    dpg.set_value("progress_bar", 0.0)  # Reset progress bar
    
    def process():
        try:
            self.log_message("Starting initial IAD analysis...")
            
            if not self.check_iad_executable():
                self.log_message("Cannot proceed without iad.exe", "ERROR")
                return
            
            dual_beam_flag = "-X" if self.use_dual_beam else ""
            g_flag = f"-g {self.g_value}"
            
            processed_files = 0
            for fname in os.listdir(self.input_dir):
                if fname.endswith(".rxt"):
                    input_path = os.path.join(self.input_dir, fname)
                    output_txt_path = input_path.replace(".rxt", ".txt")
                    
                    cmd_parts = [self.iad_exe]
                    if dual_beam_flag:
                        cmd_parts.append(dual_beam_flag)
                    cmd_parts.extend([g_flag, input_path])
                    
                    self.log_message(f"Processing: {fname}")
                    subprocess.run(cmd_parts, capture_output=True)
                    
                    if os.path.exists(output_txt_path):
                        dest_path = os.path.join(self.output_dir, os.path.basename(output_txt_path))
                        shutil.copy(output_txt_path, dest_path)
                        os.remove(output_txt_path)
                        processed_files += 1
                    else:
                        self.log_message(f"No output generated for: {fname}", "WARNING")
            
            self.log_message(f"Initial IAD analysis complete. Processed {processed_files} files.")
            
            # Extract scattering data
            iad_core.extract_scattering_data(self.model)
            
            # Update progress to show completion of step 1
            dpg.set_value("progress_bar", 1/3.0)
            dpg.set_value("progress_text", "Step 1/3: Initial analysis complete")
            dpg.set_value("status_label", "Ready for power law fitting")
            
        except Exception as e:
            self.log_message(f"Error in initial IAD analysis: {str(e)}", "ERROR")
        finally:
            self.processing = False
            dpg.set_value("btn_step1", "Run Initial Analysis")
            dpg.configure_item("btn_step1", enabled=True)
            # Don't call update_status here as it might trigger step 3
    
    Thread(target=process, daemon=True).start()

def fit_powerlaw(self, sender=None, data=None):
    """Step 2: Fit power law to scattering data"""
    if self.processing:
        return
        
    self.processing = True
    dpg.set_value("btn_step2", "Processing...")
    dpg.configure_item("btn_step2", enabled=False)
    dpg.set_value("status_label", "Fitting Power Law")
    dpg.set_value("progress_text", "Step 2/3: Fitting power law...")
    dpg.set_value("progress_bar", 1/3.0)  # Start from step 1 completion
    
    def process():
        try:
            self.log_message("Starting power law fitting...")
            
            csv_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
            if not os.path.exists(csv_path):
                self.log_message("Scattering data not found. Run initial analysis first.", "ERROR")
                return
            
            df = pd.read_csv(csv_path)
            wavelengths = np.asarray(df["lambda"].values, dtype=float)
            samples = df.drop(columns=["lambda"])
            
            # Fitting parameters
            lambda0 = self.reference_wavelength
            fit_min = self.fit_min
            fit_max = self.fit_max
            fit_mask = (wavelengths >= fit_min) & (wavelengths <= fit_max)
            wavelength_fit = wavelengths[fit_mask]
            
            fit_results = []
            fit_curves = pd.DataFrame({"lambda": wavelengths})
            
            num_samples = len(samples.columns)
            
            def mie_powerlaw(lambda_, b_mie, a0, lambda0):
                return a0 * (lambda_ / lambda0) ** (-b_mie)
            
            for i, sample_name in enumerate(samples.columns):
                # Update progress for each sample
                progress = (i + 1) / num_samples
                base_progress = 1/3.0  # Progress from step 1
                total_progress = base_progress + (progress / 3.0) # Scale this step's progress to be 1/3 of total
                
                dpg.set_value("progress_bar", total_progress)
                dpg.set_value("progress_text", f"Step 2/3: Fitting sample {i+1}/{num_samples} ({sample_name})")
                us = samples[sample_name].values
                us_fit = us[fit_mask]
                
                # Get a0 from data at lambda0
                try:
                    index_600 = np.where(wavelengths == lambda0)[0][0]
                    a0 = us[index_600]
                except IndexError:
                    self.log_message(f"lambda0 = {lambda0} nm not found in {sample_name}, skipping.", "WARNING")
                    continue
                
                try:
                    # Fit only b_mie
                    def fit_model(lambda_, b):
                        return mie_powerlaw(lambda_, b, a0, lambda0)
                    us_fit = np.asarray(us_fit, dtype=float)
                    popt, _ = curve_fit(fit_model, wavelength_fit, us_fit, p0=[1.0])
                    b_mie = abs(popt[0])
                    us_smoothed = fit_model(wavelengths, b_mie)
                    # Calculate R^2
                    residuals = us_fit - fit_model(wavelength_fit, b_mie)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((us_fit - np.mean(us_fit))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
                    fit_results.append({"sample": sample_name, "a0": a0, "b_mie": b_mie, "r2": r2})
                    fit_curves[sample_name] = us_smoothed
                    self.log_message(f"Fit {sample_name}: a0 = {a0:.4f}, b_mie = {b_mie:.4f}, R^2 = {r2:.4f}")
                except Exception as e:
                    self.log_message(f"Fitting failed for {sample_name}: {str(e)}", "ERROR")
            
            # Save results
            summary_csv = os.path.join(self.output_dir, "powerlaw_summary.csv")
            smoothed_csv = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
            if fit_results:
                pd.DataFrame(fit_results).to_csv(summary_csv, index=False)
            fit_curves.to_csv(smoothed_csv, index=False)
            
            self.fit_results = fit_results
            self.log_message(f"Power law fitting complete. Fitted {len(fit_results)} samples.")
            self.update_results_table()
            
            # Update progress to show completion of step 2
            dpg.set_value("progress_bar", 2/3.0)
            dpg.set_value("progress_text", "Step 2/3: Power law fitting complete")
            dpg.set_value("status_label", "Ready for final analysis")
            
        except Exception as e:
            self.log_message(f"Error in power law fitting: {str(e)}", "ERROR")
        finally:
            self.processing = False
            dpg.set_value("btn_step2", "Fit Power Law")
            dpg.configure_item("btn_step2", enabled=True)
            # Don't call update_status here as it might trigger step 3
    
    Thread(target=process, daemon=True).start()

def run_final_iad(self, sender=None, data=None):
    """Step 3: Run final IAD analysis"""
    if self.processing:
        return
        
    self.processing = True
    dpg.set_value("btn_step3", "Processing...")
    dpg.configure_item("btn_step3", enabled=False)
    dpg.set_value("status_label", "Running Final Analysis")
    dpg.set_value("progress_text", "Step 3/3: Running final analysis...")
    
    def process():
        try:
            self.log_message("Starting final IAD analysis...")
            
            mode = dpg.get_value("analysis_mode")
            model_type = dpg.get_value("model_type")
            
            if mode == "Power Law":
                self.run_powerlaw_mode(model_type)
            else:
                self.run_fixed_mode()
            
            self.log_message("Final IAD analysis complete.")
            
        except Exception as e:
            self.log_message(f"Error in final IAD analysis: {str(e)}", "ERROR")
        finally:
            self.processing = False
            dpg.set_value("btn_step3", "Run Final Analysis")
            dpg.configure_item("btn_step3", enabled=True)
            self.update_status()
    
    Thread(target=process, daemon=True).start()

def run_powerlaw_mode(self, model_type):
    """Run final analysis using power law parameters"""
    summary_path = os.path.join(self.output_dir, "powerlaw_summary.csv")
    if not os.path.exists(summary_path):
        self.log_message("Power law summary not found. Run fitting first.", "ERROR")
        return
    
    summary_df = pd.read_csv(summary_path)
    lambda0 = self.reference_wavelength
    g = self.g_value
    if abs(1.0 - g) < 1e-9: # Check for g being close to 1
        self.log_message("Anisotropy (g) cannot be 1, this would cause a division by zero.", "ERROR")
        return
    num_files = len([f for f in os.listdir(self.input_dir) if f.endswith(".rxt")])
    
    for i, fname in enumerate(os.listdir(self.input_dir)):
        if fname.endswith(".rxt"):
            progress = (i + 1) / num_files
            base_progress = 2/3.0
            total_progress = base_progress + (progress / 3.0)
            dpg.set_value("progress_bar", total_progress)
            dpg.set_value("progress_text", f"Step 3/3: Final analysis on {fname} ({i+1}/{num_files})")
            
            sample_name = os.path.splitext(fname)[0]
            input_path = os.path.join(self.input_dir, fname)
            expected_output_txt = input_path.replace(".rxt", ".txt")
            
            row = summary_df[summary_df["sample"] == sample_name]
            if row.empty:
                self.log_message(f"No fit data found for {sample_name}, skipping.", "WARNING")
                continue
            
            a0_reduced = row["a0"].item()
            b_mie = row["b_mie"].item()
            
            # Convert a0 from reduced scattering (us') to scattering (us) for IAD
            # us = us' / (1 - g)
            a0_scattering = a0_reduced / (1 - g)
            dual_beam_flag = '-X' if self.use_dual_beam else ''
            cmd = [self.iad_exe]
            if dual_beam_flag:
                cmd.append(dual_beam_flag)
            cmd.extend(["-F", f"{model_type} {lambda0} {a0_scattering:.6f} {b_mie:.6f}", input_path])
            
            self.log_message(f"Processing final analysis for: {sample_name}")
            subprocess.run(cmd, capture_output=True)
            
            if os.path.exists(expected_output_txt):
                dest_path = os.path.join(self.output_dir, f"final_{os.path.basename(expected_output_txt)}")
                shutil.copy(expected_output_txt, dest_path)
                os.remove(expected_output_txt)
            else:
                self.log_message(f"No output generated for: {sample_name}", "WARNING")

def run_fixed_mode(self):
    """Run final analysis using fixed scattering values"""
    scatter_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
    if not os.path.exists(scatter_path):
        self.log_message("Scattering data not found. Run initial analysis first.", "ERROR")
        return
    
    scatter_df = pd.read_csv(scatter_path)
    lambda_values = scatter_df["lambda"].values
    temp_rxt = os.path.join(self.run_dir, "temp_single.rxt")
    combined_output = os.path.join(self.output_dir, "combined_output.txt")
    
    # Clear previous combined output
    if os.path.exists(combined_output):
        os.remove(combined_output)
    
    # Count total number of data lines to process
    total_tasks = 0
    for fname in os.listdir(self.input_dir):
        if fname.endswith(".rxt"):
            try:
                data_lines = self.get_all_data_lines(os.path.join(self.input_dir, fname))
                total_tasks += len(data_lines)
            except Exception:
                continue
    if total_tasks == 0:
        self.log_message("No data lines found for fixed scattering analysis.", "ERROR")
        return
    current_task = 0
    for fname in os.listdir(self.input_dir):
        if fname.endswith(".rxt"):
            sample_name = os.path.splitext(fname)[0]
            input_path = os.path.join(self.input_dir, fname)
            
            if sample_name not in scatter_df.columns:
                self.log_message(f"No scattering column for {sample_name}, skipping.", "WARNING")
                continue
            
            try:
                header = self.get_rxt_header(input_path)
                data_lines = self.get_all_data_lines(input_path)
                for wave, refl, trans in data_lines:
                    lambda_idx = np.argmin(np.abs(lambda_values - wave))
                    us_prime = scatter_df.loc[lambda_idx, sample_name]
                    us_val = us_prime / (1 - self.g_value)
                    self.append_data_to_header(header, wave, refl, trans, temp_rxt)
                    dual_beam_flag = '-X' if self.use_dual_beam else ''
                    cmd = [self.iad_exe]
                    if dual_beam_flag:
                        cmd.append(dual_beam_flag)
                    cmd.extend(["-F", f"{us_val:.6f}", temp_rxt])
                    subprocess.run(cmd, capture_output=True)
                    temp_out = temp_rxt.replace(".rxt", ".txt")
                    result = self.extract_last_data_line(temp_out)
                    if result:
                        with open(combined_output, "a") as fout:
                            fout.write(f"{sample_name}\t{result}\n")
                    # Update progress bar and text
                    current_task += 1
                    progress = current_task / total_tasks
                    dpg.set_value("progress_bar", progress)
                    dpg.set_value("progress_text", f"Step 3/3: Final analysis... ({current_task}/{total_tasks})")
            except Exception as e:
                self.log_message(f"Could not process {sample_name}: {str(e)}", "ERROR")
                continue

def get_rxt_header(self, file_path, header_lines=30):
    with open(file_path, 'r') as file:
        return [next(file) for _ in range(header_lines)]

def get_all_data_lines(self, file_path, header_lines=30):
    lines = []
    with open(file_path, 'r') as f:
        content = f.readlines()[header_lines:]
        for line in content:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    wave, refl, trans = map(float, parts)
                    lines.append((wave, refl, trans))
                except ValueError:
                    continue
    return lines

def append_data_to_header(self, header, wave, refl, trans, output_path):
    with open(output_path, 'w') as f:
        f.writelines(header)
        f.write("#wave\trefl\ttrans\n")
        f.write(f"{wave:.1f}\t{refl:.6f}\t{trans:.6f}\n")

def extract_last_data_line(self, file_path, min_floats=6):
    float_pattern = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in reversed(lines):
        floats = float_pattern.findall(line)
        if len(floats) >= min_floats:
            return line.strip()
    return None

def toggle_plots(self, sender=None, data=None):
    """Toggle the plot area and update the button label accordingly"""
    if dpg.does_item_exist("main_plot_area"):
        is_visible = dpg.is_item_shown("main_plot_area")
        if is_visible:
            dpg.configure_item("main_plot_area", show=False)
            dpg.set_value("toggle_plots_btn", "Show Plots")
        else:
            dpg.configure_item("main_plot_area", show=True)
            self.show_plots()
            dpg.set_value("toggle_plots_btn", "Hide Plots")

def show_plots(self, sender=None, data=None):
    """Show plot selection and plot area in the main window (not a popup)"""
    # Clear and show the main plot area
    if dpg.does_item_exist("main_plot_area"):
        dpg.delete_item("main_plot_area", children_only=True)
        dpg.configure_item("main_plot_area", show=True)
        dpg.set_value("toggle_plots_btn", "Hide Plots")
        dpg.add_text("Choose what you want to plot:", parent="main_plot_area")
        with dpg.group(horizontal=True, parent="main_plot_area"):
            dpg.add_radio_button(
                items=["Raw Scattering", "Power Law Smoothed", "Absorbance", "Raw + Fitted Scattering (with Fit Window)"],
                tag="plot_type_radio",
                default_value="Power Law Smoothed",
                callback=self.show_gui_plot
            )
            dpg.add_button(label="Refresh Plot", callback=self.show_gui_plot, width=120)
            dpg.add_button(label="Open in Matplotlib", callback=self.show_matplotlib_plot, width=160)
        # Add a dedicated child window for the plot itself
        dpg.add_child_window(tag="plot_canvas", parent="main_plot_area", width=-1, height=-1)
        self.show_gui_plot()

def show_gui_plot(self, sender=None, data=None):
    """Render the selected plot type inside the plot_canvas (live update)"""
    try:
        # Only clear the plot_canvas, not the controls
        if dpg.does_item_exist("plot_canvas"):
            dpg.delete_item("plot_canvas", children_only=True)
        plot_type = dpg.get_value("plot_type_radio")
        with dpg.plot(label=f"{plot_type}", height=-1, width=-1, parent="plot_canvas") as plot_id:
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Wavelength (nm)")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Value")
            if plot_type == "Raw Scattering":
                path = os.path.join(self.output_dir, "scattering_values_multi.csv")
                if not os.path.exists(path):
                    self.log_message("Raw scattering file missing", "ERROR")
                    return
                df = pd.read_csv(path)
                wavelengths = np.asarray(df["lambda"].values, dtype=float)
                for col in df.columns:
                    if col != "lambda":
                        x_data = wavelengths.tolist()
                        y_data = df[col].values.tolist()
                        dpg.add_line_series(x_data, y_data, label=col, parent=y_axis)
            elif plot_type == "Power Law Smoothed":
                path = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
                if not os.path.exists(path):
                    self.log_message("Smoothed scattering file missing", "ERROR")
                    return
                df = pd.read_csv(path)
                wavelengths = np.asarray(df["lambda"].values, dtype=float)
                for col in df.columns:
                    if col != "lambda":
                        x_data = wavelengths.tolist()
                        y_data = df[col].values.tolist()
                        dpg.add_line_series(x_data, y_data, label=col, parent=y_axis)
            elif plot_type == "Absorbance":
                path = os.path.join(self.output_dir, "combined_output.txt")
                if not os.path.exists(path):
                    self.log_message("Combined output missing", "ERROR")
                    return
                data = []
                with open(path) as f:
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            parts = line.strip().split()
                            if len(parts) >= 7:
                                try:
                                    data.append([parts[0], float(parts[1]), float(parts[6])])
                                except Exception as e:
                                    self.log_message(f"Absorbance parse error: {e}", "ERROR")
                                    continue
                if not data:
                    self.log_message("No absorbance data found", "ERROR")
                    return
                self.log_message(f"Absorbance data sample: {data[:3]}")
                COLUMNS = pd.Index(["Sample", "Wavelength", "Absorbance"])
                if data and isinstance(data[0], (list, tuple)) and len(data[0]) == len(COLUMNS):
                    df = pd.DataFrame(data, columns=COLUMNS)
                else:
                    df = pd.DataFrame(data)
                for sample, group in df.groupby("Sample"):
                    x_data = group["Wavelength"].astype(float).values.tolist()
                    y_data = group["Absorbance"].astype(float).values.tolist()
                    dpg.add_line_series(x_data, y_data, label=str(sample), parent=y_axis)
            elif plot_type == "Raw + Fitted Scattering (with Fit Window)":
                raw_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
                fit_path = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
                if not os.path.exists(raw_path) or not os.path.exists(fit_path):
                    self.log_message("Raw or fitted scattering file missing", "ERROR")
                    return
                raw_df = pd.read_csv(raw_path)
                fit_df = pd.read_csv(fit_path)
                wavelengths = np.asarray(raw_df["lambda"].values, dtype=float)
                for col in raw_df.columns:
                    if col != "lambda":
                        x_data = wavelengths.tolist()
                        y_data = raw_df[col].values.tolist()
                        dpg.add_line_series(x_data, y_data, label=f"Raw {col}", parent=y_axis)
                for col in fit_df.columns:
                    if col != "lambda":
                        x_data = wavelengths.tolist()
                        y_data = fit_df[col].values.tolist()
                        dpg.add_line_series(x_data, y_data, label=f"Fitted {col}", parent=y_axis)
                dpg.add_inf_line_series([self.fit_min, self.fit_max], label="Fit Window", parent=x_axis)
            dpg.add_plot_legend(outside=True)
    except Exception as e:
        self.log_message(f"Error in GUI plotting: {str(e)}", "ERROR")
def show_matplotlib_plot(self, sender=None, data=None):
    """Show the selected plot type in a matplotlib window (always on main thread)"""
    try:
        plot_type = dpg.get_value("plot_type_radio")
        # All plotting is synchronous and on the main thread
        if plot_type == "Raw Scattering":
            path = os.path.join(self.output_dir, "scattering_values_multi.csv")
            if not os.path.exists(path):
                self.log_message("Raw scattering file missing", "ERROR")
                return
            self.plot_raw_scattering(path)
        elif plot_type == "Power Law Smoothed":
            path = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
            if not os.path.exists(path):
                self.log_message("Smoothed scattering file missing", "ERROR")
                return
            self.plot_powerlaw_fits(path)
        elif plot_type == "Absorbance":
            path = os.path.join(self.output_dir, "combined_output.txt")
            if not os.path.exists(path):
                self.log_message("Combined output missing", "ERROR")
                return
            powerlaw_path = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
            self.plot_final_results(path, powerlaw_path)
        elif plot_type == "Raw + Fitted Scattering (with Fit Window)":
            raw_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
            fit_path = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
            if not os.path.exists(raw_path) or not os.path.exists(fit_path):
                self.log_message("Raw or fitted scattering file missing", "ERROR")
                return
            self.plot_raw_and_fitted_with_mask(raw_path, fit_path, fit_min=self.fit_min, fit_max=self.fit_max)
    except Exception as e:
        self.log_message(f"Error in matplotlib plotting: {str(e)}", "ERROR")
def open_output_folder(self, sender=None, data=None):
    """Open the output folder in file explorer"""
    try:
        if sys.platform.startswith('win'):
            os.startfile(self.output_dir)
        elif sys.platform.startswith('darwin'):
            subprocess.run(['open', self.output_dir])
        else:
            subprocess.run(['xdg-open', self.output_dir])
    except Exception as e:
        self.log_message(f"Could not open output folder: {str(e)}", "ERROR")

def create_gui(self):
    """Create the main GUI with enhanced status display"""
    # Load Lato font from docs folder and set as default
    with dpg.font_registry():
        lato_font = dpg.add_font("docs/Lato/Lato-Regular.ttf", 18)
    dpg.bind_font(lato_font)
    # Main window
    with dpg.window(label="IAD Optical Analysis", tag="main_window", width=1200, height=800):
        # Add logo image if available
        logo_path = os.path.join(self.run_dir, "docs", "logo.jpg")
        if os.path.exists(logo_path):
            with dpg.texture_registry():
                width, height, channels, data = dpg.load_image(logo_path)
                dpg.add_static_texture(width, height, data, tag="logo_texture")
            dpg.add_image("logo_texture", width=width, height=height)
        # Header section
        with dpg.group(horizontal=True):
            dpg.add_text("IAD Optical Analysis Suite", color=(200, 200, 200))
            dpg.add_spacer(width=300)
            dpg.add_button(label="Refresh Status", callback=lambda: self.update_status(force_reset=True), width=150, height=30)
            dpg.add_button(label="Open Output Folder", callback=self.open_output_folder, width=150, height=30)
            dpg.add_button(label="Open Documentation", callback=self.open_documentation, width=180, height=30)
        dpg.add_separator()
        # Status section with enhanced display
        with dpg.collapsing_header(label="System Status", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("IAD Executable:", color=(200, 200, 200))
                dpg.add_text("Checking...", tag="iad_status")
                dpg.add_spacer(width=50)
                dpg.add_text("Input Files:", color=(200, 200, 200))
                dpg.add_text("Checking...", tag="files_status")
            dpg.add_spacer(height=15)
            # Progress section
            with dpg.group(horizontal=True):
                dpg.add_text("Overall Progress:", color=(200, 200, 200))
                dpg.add_progress_bar(tag="progress_bar", width=400, height=25)
            dpg.add_text("", tag="progress_text", color=(200, 200, 200))
            # Status section
            with dpg.group(horizontal=True):
                dpg.add_text("Status:", color=(200, 200, 200))
                dpg.add_text("Initializing...", tag="status_label", color=(200, 200, 200))
        dpg.add_separator()
        # Settings section
        with dpg.collapsing_header(label="Analysis Settings", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Anisotropy factor (g):", color=(200, 200, 200))
                dpg.add_input_float(
                    tag="g_input",
                    default_value=0.8,
                    min_value=0.0,
                    max_value=1.0,
                    width=300,
                    step=0.01,
                    format="%.3f",
                    callback=self.on_g_value_change
                )
            dpg.add_text("Reference wavelength (lambda0) [nm]:", color=(200, 200, 200))
            dpg.add_input_int(
                tag="lambda0_input",
                default_value=600,
                min_value=300,
                max_value=1000,
                width=150,
                callback=self.on_lambda0_change
            )
            dpg.add_text("Fit window min [nm]:", color=(200, 200, 200))
            dpg.add_input_int(
                tag="fit_min_input",
                default_value=600,
                min_value=300,
                max_value=1000,
                width=100,
                callback=self.on_fit_min_change
            )
            dpg.add_text("Fit window max [nm]:", color=(200, 200, 200))
            dpg.add_input_int(
                tag="fit_max_input",
                default_value=750,
                min_value=300,
                max_value=1000,
                width=100,
                callback=self.on_fit_max_change
            )
            dpg.add_checkbox(
                label="Use dual beam correction",
                tag="dual_beam_check",
                default_value=False,
                callback=self.on_dual_beam_change
            )
        dpg.add_separator()
        # Analysis steps
        with dpg.collapsing_header(label="Analysis Steps", default_open=True):
            # Step 1
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="1. Run Initial Analysis",
                    tag="btn_step1",
                    callback=self.run_initial_iad,
                    width=250,
                    height=35
                )
                dpg.add_text("Extract scattering coefficients from raw data", color=(200, 200, 200))
            # Step 2
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="2. Fit Power Law",
                    tag="btn_step2",
                    callback=self.fit_powerlaw,
                    width=250,
                    height=35
                )
                dpg.add_text("Fit Mie scattering power law to wavelength dependence", color=(200, 200, 200))
            # Step 3 settings
            with dpg.group(horizontal=True):
                dpg.add_text("Final analysis mode:", color=(200, 200, 200))
                dpg.add_combo(
                    items=["Power Law", "Fixed Scattering"],
                    default_value="Power Law",
                    tag="analysis_mode",
                    width=200
                )
                dpg.add_text("Model type:", color=(200, 200, 200))
                dpg.add_combo(
                    items=["R", "P"],
                    default_value="R",
                    tag="model_type",
                    width=80
                )
            # Step 3
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="3. Run Final Analysis",
                    tag="btn_step3",
                    callback=self.run_final_iad,
                    width=250,
                    height=35
                )
                dpg.add_text("Generate final optical properties with chosen model", color=(200, 200, 200))
        dpg.add_separator()
        # Results section
        with dpg.collapsing_header(label="Results & Visualization", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Show Plots",
                    tag="toggle_plots_btn",
                    callback=self.toggle_plots,
                    width=150,
                    height=35
                )
            # Add a persistent plot area, initially hidden
            dpg.add_child_window(tag="main_plot_area", width=-1, height=600, show=False)
            # Add the powerlaw fit summary table below the plot area
            with dpg.child_window(tag="results_table_window", width=-1, height=220):
                dpg.add_text("Power Law Fit Summary:", color=(200, 200, 200))
                dpg.add_table(tag="results_table", header_row=True, resizable=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True)
        with dpg.child_window(tag="log_scroll", height=250, width=-1):
            pass  # Log lines will be added here as colored text
    
    # Set main window as primary
    dpg.set_primary_window("main_window", True)
    # Update status on launch (force reset to 0/3)
    self.update_status(force_reset=True)
    # Populate the results table on launch
    self.update_results_table()

def update_results_table(self):
    """Update the results table with power law fit data"""
    if not dpg.does_item_exist("results_table"):
        return
    # Clear previous table columns and rows
    dpg.delete_item("results_table", children_only=True)
    powerlaw_csv = os.path.join(self.output_dir, "powerlaw_summary.csv")
    if not os.path.exists(powerlaw_csv):
        return
    df = pd.read_csv(powerlaw_csv)
    # Add columns
    columns = ["Sample", "a0", "b_mie"]
    if "r2" in df.columns:
        columns.append("r2")
    for col in columns:
        dpg.add_table_column(label=col, parent="results_table")
    # Add rows
    for _, row in df.iterrows():
        row_data = [str(row.get("sample", row.get("Sample", ""))), f"{row['a0']:.4f}", f"{row['b_mie']:.4f}"]
        if "r2" in df.columns:
            row_data.append(f"{row['r2']:.4f}")
        with dpg.table_row(parent="results_table"):
            for cell in row_data:
                dpg.add_text(cell)

def export_results(self, sender=None, data=None):
    """Export all results to Excel file"""
    try:
        self.log_message("Exporting results to Excel...")
        
        output_file = os.path.join(self.output_dir, "iad_analysis_results.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Export scattering data
            scatter_csv = os.path.join(self.output_dir, "scattering_values_multi.csv")
            if os.path.exists(scatter_csv):
                df_scatter = pd.read_csv(scatter_csv)
                df_scatter.to_excel(writer, sheet_name='Raw_Scattering', index=False)
            
            # Export power law fits
            powerlaw_csv = os.path.join(self.output_dir, "powerlaw_summary.csv")
            if os.path.exists(powerlaw_csv):
                df_powerlaw = pd.read_csv(powerlaw_csv)
                df_powerlaw.to_excel(writer, sheet_name='PowerLaw_Fits', index=False)
            
            # Export smoothed data
            smoothed_csv = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
            if os.path.exists(smoothed_csv):
                df_smoothed = pd.read_csv(smoothed_csv)
                df_smoothed.to_excel(writer, sheet_name='Smoothed_Scattering', index=False)
            
            # Export final results if available
            combined_txt = os.path.join(self.output_dir, "combined_output.txt")
            if os.path.exists(combined_txt):
                # Parse combined output into DataFrame
                data = []
                with open(combined_txt, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = re.findall(r"[^\s]+", line)
                        if len(parts) >= 9:
                            try:
                                sample = parts[0]
                                wavelength = float(parts[1])
                                refl = float(parts[2])
                                trans = float(parts[3])
                                absorbance = float(parts[6])
                                data.append([sample, wavelength, refl, trans, absorbance])
                            except ValueError:
                                continue
                
                if data:
                    df_final = pd.DataFrame(data)
                    if not df_final.empty and df_final.shape[1] == 5:
                        df_final.columns = ['Sample', 'Wavelength', 'Reflectance', 'Transmittance', 'Absorbance']
                    df_final.to_excel(writer, sheet_name='Final_Results', index=False)
        
        self.log_message(f"Results exported to: {os.path.basename(output_file)}")
        
    except Exception as e:
        self.log_message(f"Error exporting results: {str(e)}", "ERROR")

def generate_report(self, sender=None, data=None):
    """Generate a comprehensive analysis report"""
    try:
        self.log_message("Generating analysis report...")
        
        report_path = os.path.join(self.output_dir, "analysis_report.html")
        
        html_content = self.create_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.log_message(f"Analysis report generated: {os.path.basename(report_path)}")
        
        # Try to open the report
        try:
            if sys.platform.startswith('win'):
                os.startfile(report_path)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', report_path])
            else:
                subprocess.run(['xdg-open', report_path])
        except:
            pass
            
    except Exception as e:
        self.log_message(f"Error generating report: {str(e)}", "ERROR")

def create_html_report(self):
    """Create HTML report content"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IAD Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .parameter { font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>IAD Optical Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Analysis Parameters</h2>
            <p><span class="parameter">Anisotropy factor (g):</span> {g_value}</p>
            <p><span class="parameter">Dual beam correction:</span> {dual_beam}</p>
            <p><span class="parameter">Input directory:</span> {input_dir}</p>
            <p><span class="parameter">Output directory:</span> {output_dir}</p>
        </div>
        
        <div class="section">
            <h2>Processing Summary</h2>
            <p><span class="parameter">Files processed:</span> {file_count}</p>
            <p><span class="parameter">Samples analyzed:</span> {sample_count}</p>
        </div>
        
        {powerlaw_section}
        
        <div class="section">
            <h2>Output Files</h2>
            <ul>
                {output_files}
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Fill in the template
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    file_count = self.count_input_files()
    
    # Power law section
    powerlaw_section = ""
    powerlaw_csv = os.path.join(self.output_dir, "powerlaw_summary.csv")
    if os.path.exists(powerlaw_csv):
        df_powerlaw = pd.read_csv(powerlaw_csv)
        
        powerlaw_section = """
        <div class="section">
            <h2>Power Law Fit Results</h2>
            <table>
                <tr><th>Sample</th><th>a0</th><th>b_mie</th></tr>
        """
        
        for _, row in df_powerlaw.iterrows():
            powerlaw_section += f"<tr><td>{row['sample']}</td><td>{row['a0']:.4f}</td><td>{row['b_mie']:.4f}</td></tr>"
        
        powerlaw_section += "</table></div>"
    
    # Output files list
    output_files = ""
    for file in os.listdir(self.output_dir):
        if file.endswith(('.csv', '.txt', '.xlsx')):
            output_files += f"<li>{file}</li>"
    
    return html.format(
        timestamp=timestamp,
        g_value=self.g_value,
        dual_beam="Enabled" if self.use_dual_beam else "Disabled",
        input_dir=self.input_dir,
        output_dir=self.output_dir,
        file_count=file_count,
        sample_count=len(self.fit_results) if self.fit_results else "N/A",
        powerlaw_section=powerlaw_section,
        output_files=output_files
    )

def clear_log(self, sender=None, data=None):
    """Clear the log window"""
    if dpg.does_item_exist("log_scroll"):
        dpg.delete_item("log_scroll", children_only=True)

def save_log(self, sender=None, data=None):
    """Save the log to a file"""
    try:
        log_content = dpg.get_value("log_text")
        log_path = os.path.join(self.output_dir, "analysis_log.txt")
        
        with open(log_path, 'w') as f:
            f.write(log_content)
        
        self.log_message(f"Log saved to: {os.path.basename(log_path)}")
        
    except Exception as e:
        self.log_message(f"Error saving log: {str(e)}", "ERROR")

def copy_log_to_clipboard(self):
    """Copy the current log content to clipboard"""
    try:
        import pyperclip
        log_content = dpg.get_value("log_text")
        pyperclip.copy(log_content)
        self.log_message("Log content copied to clipboard", "SUCCESS")
    except Exception as e:
        self.log_message(f"Failed to copy log: {str(e)}", "ERROR")

def open_documentation(self, sender=None, data=None):
    # Show a popup to select which documentation to open
    if dpg.does_item_exist("doc_select_popup"):
        dpg.delete_item("doc_select_popup")
    with dpg.window(label="Select Documentation", modal=True, tag="doc_select_popup", no_resize=True, width=350, height=180):
        dpg.add_text("Choose documentation to open:")
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Suite Documentation", width=150, callback=lambda: self._open_pdf("suite"))
            dpg.add_button(label="IAD Documentation", width=180, callback=lambda: self._open_pdf("iad"))
        dpg.add_spacer(height=10)
        dpg.add_button(label="Cancel", width=100, callback=lambda: dpg.delete_item("doc_select_popup"))
def _open_pdf(self, doc_type):
    if dpg.does_item_exist("doc_select_popup"):
        dpg.delete_item("doc_select_popup")
    if doc_type == "suite":
        pdf_path = os.path.join(self.run_dir, "docs", "iad_app_documentation_placeholder.pdf")
    elif doc_type == "iad":
        pdf_path = os.path.join(self.run_dir, "docs", "manual.pdf")
    else:
        self.log_message("Unknown documentation type selected.", "ERROR")
        return
    try:
        if sys.platform.startswith('win'):
            os.startfile(pdf_path)
        elif sys.platform.startswith('darwin'):
            subprocess.run(['open', pdf_path])
        else:
            subprocess.run(['xdg-open', pdf_path])
    except Exception as e:
        self.log_message(f"Could not open documentation: {str(e)}", "ERROR")

def run(self):
    """Start the GUI application"""
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Main loop
    while dpg.is_dearpygui_running():
        # Update results table periodically
        if hasattr(self, '_last_table_update'):
            if time.time() - self._last_table_update > 2.0:  # Update every 2 seconds
                self.update_results_table()
                self._last_table_update = time.time()
        else:
            self._last_table_update = time.time()
        
        # Poll the GUI queue and apply updates
        while not self.gui_queue.empty():
            action, *args = self.gui_queue.get()
            if action == 'set_value':
                getattr(dpg, action)(*args)
            elif action == 'log':
                self.log_message(*args)
            elif action == 'configure_item':
                getattr(dpg, action)(*args)
        
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()
def plot_raw_scattering(self, csv_path):
    """Plot raw scattering data in a Matplotlib window"""
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 8))
    for col in df.columns:
        if col != "lambda":
            plt.plot(df["lambda"], df[col], label=col)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reduced Scattering Coefficient us' (1/mm)")
    plt.title("Raw us' Spectra from IAD Output")
    plt.grid(True)
    plt.legend(fontsize="small", loc="upper right")
    plt.tight_layout()
    plt.show()
def plot_powerlaw_fits(self, csv_path):
    """Plot power law smoothed scattering data in a Matplotlib window"""
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 8))
    for col in df.columns:
        if col != "lambda":
            plt.plot(df["lambda"], df[col], label=col)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Smoothed us' (1/mm)")
    plt.title("Power Law Smoothed us' Spectra")
    plt.grid(True)
    plt.legend(fontsize="small", loc="upper right")
    plt.tight_layout()
    plt.show()
def plot_final_results(self, absorbance_path, powerlaw_path):
    """Plot absorbance and Mie fit us' in a Matplotlib window"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    data = []
    with open(absorbance_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 7:
                    try:
                        data.append([parts[0], float(parts[1]), float(parts[6])])
                    except Exception:
                        continue
    if not data:
        self.log_message("No absorbance data found for plotting", "ERROR")
        return
    if data:
        if isinstance(data[0], (list, tuple)):
            df = pd.DataFrame(data)
            if not df.empty and df.shape[1] == 3:
                df.columns = ["Sample", "Wavelength", "Absorbance"]
        elif isinstance(data[0], str):
            df = pd.DataFrame([[item] for item in data])
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    df_mie = pd.read_csv(powerlaw_path) if os.path.exists(powerlaw_path) else None
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for sample, group in df.groupby("Sample"):
        axs[0].plot(group["Wavelength"], group["Absorbance"], label=sample)
    axs[0].set_ylabel("Absorbance")
    axs[0].set_title("Absorbance and Mie Fit us'")
    axs[0].grid(True)
    axs[0].legend(fontsize="small")
    if df_mie is not None:
        for col in df_mie.columns:
            if col.lower() != "lambda":
                axs[1].plot(df_mie["lambda"], df_mie[col], label=col)
        axs[1].set_ylabel("us' (1/mm)")
        axs[1].legend(fontsize="small", loc="upper right")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()
def plot_raw_and_fitted_with_mask(self, raw_path, fit_path, fit_min, fit_max):
    """Plot raw and fitted scattering with fit window in a Matplotlib window"""
    import matplotlib.pyplot as plt
    import pandas as pd
    raw_df = pd.read_csv(raw_path)
    fit_df = pd.read_csv(fit_path)
    plt.figure(figsize=(12, 8))
    for col in raw_df.columns:
        if col != "lambda":
            plt.plot(raw_df["lambda"], raw_df[col], label=f"Raw {col}")
    for col in fit_df.columns:
        if col != "lambda":
            plt.plot(fit_df["lambda"], fit_df[col], label=f"Fitted {col}")
    plt.axvspan(fit_min, fit_max, color='gray', alpha=0.2, label="Fit Window")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("us' (1/mm)")
    plt.title("Raw and Fitted Scattering with Fit Window")
    plt.legend(fontsize="small", loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def on_done(self, result):
    """Callback function to update GUI after background task completes"""
    self.log_message(result)
    self.update_status()