# gui/gui_manager.py

import dearpygui.dearpygui as dpg
import os
import sys
import time
import queue
from core.iad_model import IADModel  # adjust path if needed
import subprocess
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add resource_path utility for PyInstaller compatibility
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', None)
    if base_path:
        return os.path.join(base_path, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class GUIManager:
    def __init__(self, workflow):
        self.workflow = workflow  # Reference to IADWorkflow instance
        self.run_dir = workflow.run_dir
        self.output_dir = workflow.output_dir
        self.input_dir = workflow.input_dir
        self.g_value = workflow.g_value
        self.fit_min = workflow.fit_min
        self.fit_max = workflow.fit_max
        self.use_dual_beam = workflow.use_dual_beam
        self.reference_wavelength = workflow.reference_wavelength
        self.fit_results = workflow.fit_results
        self.processing = workflow.processing
        self.gui_queue = queue.Queue()
        self.font_size = getattr(workflow, 'font_size', 18)

        os.makedirs(self.output_dir, exist_ok=True)

        dpg.create_context()
        dpg.create_viewport(title="IAD Optical Analysis Suite", width=900, height=700)

    def get_run_directory(self):
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def log_message(self, message, level="INFO"):
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
            dpg.set_y_scroll("log_scroll", -1.0)

    def check_iad_executable(self):
        exists = os.path.exists(self.workflow.iad_exe)
        if exists:
            self.log_message(f"iad.exe found at: {self.workflow.iad_exe}")
        else:
            self.log_message(f"iad.exe NOT found at: {self.workflow.iad_exe}", "ERROR")
        return exists

    def count_input_files(self):
        if not os.path.exists(self.input_dir):
            return 0
        return len([f for f in os.listdir(self.input_dir) if f.endswith(".rxt")])

    def update_status(self, force_reset=False):
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

        steps_completed = 0
        step_details = []
        scatter_file = os.path.join(self.output_dir, "scattering_values_multi.csv")
        powerlaw_file = os.path.join(self.output_dir, "powerlaw_summary.csv")
        final_file = os.path.join(self.output_dir, "combined_output.txt")

        if os.path.exists(scatter_file):
            steps_completed = 1
            step_details.append("Initial analysis complete")
            if os.path.exists(powerlaw_file):
                steps_completed = 2
                step_details.append("Power law fitting complete")
                if os.path.exists(final_file):
                    steps_completed = 3
                    step_details.append("Final analysis complete")

        progress = steps_completed / 3.0
        dpg.set_value("progress_bar", progress)

        if steps_completed == 0:
            status_text = "Step 0/3: Ready to begin"
            if file_count > 0:
                status_text += f" ({file_count} files ready for analysis)"
        else:
            status_text = f"Step {steps_completed}/3: {' -> '.join(step_details)}"

        dpg.set_value("progress_text", status_text)

        if self.processing:
            current_step = steps_completed + 1
            if current_step == 1:
                dpg.set_value("status_label", "Running initial analysis...")
            elif current_step == 2:
                dpg.set_value("status_label", "Fitting power law...")
            elif current_step == 3:
                dpg.set_value("status_label", "Running final analysis...")
        else:
            status_map = [
                "Ready to begin analysis",
                "Ready for power law fitting",
                "Ready for final analysis",
                "Analysis complete"
            ]
            dpg.set_value("status_label", status_map[steps_completed])

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

    def create_gui(self):
        # Load Lato font from _internal/docs folder and set as default
        font_path = resource_path(os.path.join(self.run_dir, "_internal", "docs", "Lato", "Lato-Regular.ttf"))
        with dpg.font_registry():
            lato_font = dpg.add_font(font_path, self.font_size)
        dpg.bind_font(lato_font)
        # Main window
        with dpg.window(label="IAD Optical Analysis", tag="main_window", width=1200, height=800):
            # Add logo images if available (side by side)
            with dpg.group(horizontal=True):
                # First logo
                logo_path = os.path.join(self.run_dir, "_internal", "docs", "logo.jpg")
                if os.path.exists(logo_path):
                    with dpg.texture_registry():
                        width, height, channels, data = dpg.load_image(logo_path)
                        dpg.add_static_texture(width, height, data, tag="logo_texture")
                    dpg.add_image("logo_texture", width=width, height=height)
                    # Second logo
                    logo2_path = os.path.join(self.run_dir, "_internal", "docs", "logo2.png")
                    if os.path.exists(logo2_path):
                        with dpg.texture_registry():
                            width2, height2, channels2, data2 = dpg.load_image(logo2_path)
                            dpg.add_static_texture(width2, height2, data2, tag="logo2_texture")
                        # Scale logo2 to match the height of the first logo, preserving aspect ratio
                        if 'height' in locals() and height2 != 0:
                            scaled_width2 = int(width2 * (height / height2))
                            scaled_height2 = height
                        else:
                            scaled_width2 = width2
                            scaled_height2 = height2
                        dpg.add_image("logo2_texture", width=scaled_width2, height=scaled_height2)
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
                        default_value=self.g_value,
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
                    default_value=self.reference_wavelength,
                    min_value=300,
                    max_value=1000,
                    width=150,
                    callback=self.on_lambda0_change
                )
                dpg.add_text("Fit window min [nm]:", color=(200, 200, 200))
                dpg.add_input_int(
                    tag="fit_min_input",
                    default_value=self.fit_min,
                    min_value=300,
                    max_value=1000,
                    width=100,
                    callback=self.on_fit_min_change
                )
                dpg.add_text("Fit window max [nm]:", color=(200, 200, 200))
                dpg.add_input_int(
                    tag="fit_max_input",
                    default_value=self.fit_max,
                    min_value=300,
                    max_value=1000,
                    width=100,
                    callback=self.on_fit_max_change
                )
                dpg.add_checkbox(
                    label="Use dual beam correction",
                    tag="dual_beam_check",
                    default_value=self.use_dual_beam,
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
                        callback=self.workflow.run_initial_iad,
                        width=250,
                        height=35
                    )
                    dpg.add_text("Extract scattering coefficients from raw data", color=(200, 200, 200))
                # Step 2
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="2. Fit Power Law",
                        tag="btn_step2",
                        callback=self.workflow.fit_powerlaw,
                        width=250,
                        height=35
                    )
                    dpg.add_text("Fit Mie scattering power law to wavelength dependence", color=(200, 200, 200))
                # Step 3 settings
                with dpg.group(horizontal=True):
                    dpg.add_text("Final analysis mode:", color=(200, 200, 200))
                    # Map display names to internal values
                    self.analysis_mode_display_map = {"Power Law": "P", "Fixed Scattering": "F"}
                    self.analysis_mode_reverse_map = {v: k for k, v in self.analysis_mode_display_map.items()}
                    dpg.add_combo(
                        items=["Power Law", "Fixed Scattering"],
                        default_value=self.analysis_mode_reverse_map.get(self.workflow.analysis_mode, "Power Law"),
                        tag="analysis_mode",
                        width=200,
                        callback=self._on_analysis_mode_change
                    )
                    dpg.add_text("Model type:", color=(200, 200, 200), tag="model_type_label", show=(self.workflow.analysis_mode == "P"))
                    dpg.add_combo(
                        items=["R", "P"],
                        default_value="R",
                        tag="model_type",
                        width=80,
                        show=(self.workflow.analysis_mode == "P")
                    )
                # Step 3
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="3. Run Final Analysis",
                        tag="btn_step3",
                        callback=lambda: self._run_final_analysis_with_mode(),
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
                    path = os.path.join(self.output_dir, "final_absorption_scattering.csv")
                    if not os.path.exists(path):
                        self.log_message("final_absorption_scattering.csv missing", "ERROR")
                        return
                    df = pd.read_csv(path)
                    if df.empty or not all(col in df.columns for col in ["Sample", "Wavelength", "Absorption"]):
                        self.log_message("Absorption CSV missing required columns", "ERROR")
                        return
                    for sample, group in df.groupby("Sample"):
                        x_data = group["Wavelength"].astype(float).values.tolist()
                        y_data = group["Absorption"].astype(float).values.tolist()
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
            pass

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
                self._run_on_main_thread(self.plot_raw_scattering, path)
            elif plot_type == "Power Law Smoothed":
                path = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
                if not os.path.exists(path):
                    self.log_message("Smoothed scattering file missing", "ERROR")
                    return
                self._run_on_main_thread(self.plot_powerlaw_fits, path)
            elif plot_type == "Absorbance":
                path = os.path.join(self.output_dir, "final_absorption_scattering.csv")
                if not os.path.exists(path):
                    self.log_message("final_absorption_scattering.csv missing", "ERROR")
                    return
                self._run_on_main_thread(self.plot_final_results, path, path)
            elif plot_type == "Raw + Fitted Scattering (with Fit Window)":
                raw_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
                fit_path = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
                if not os.path.exists(raw_path) or not os.path.exists(fit_path):
                    self.log_message("Raw or fitted scattering file missing", "ERROR")
                    return
                self._run_on_main_thread(self.plot_raw_and_fitted_with_mask, raw_path, fit_path, self.fit_min, self.fit_max)
        except Exception as e:
            self.log_message(f"Error in matplotlib plotting: {str(e)}", "ERROR")
            pass

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
        pass

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
        pass

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
        pass

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
            pass

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
        pass

    def save_log(self, sender=None, data=None):
        """Copy the current log content to clipboard"""
        try:
            import pyperclip
            log_content = dpg.get_value("log_text")
            pyperclip.copy(log_content)
            self.log_message("Log content copied to clipboard", "SUCCESS")
        except Exception as e:
            self.log_message(f"Failed to copy log: {str(e)}", "ERROR")
            pass

    def copy_log_to_clipboard(self):
        """Copy the current log content to clipboard"""
        try:
            import pyperclip
            log_content = dpg.get_value("log_text")
            pyperclip.copy(log_content)
            self.log_message("Log content copied to clipboard", "SUCCESS")
        except Exception as e:
            self.log_message(f"Failed to copy log: {str(e)}", "ERROR")
            pass

    def open_documentation(self, sender=None, data=None):
        """Show a popup to select which documentation to open"""
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
        pass

    def _open_pdf(self, doc_type):
        if dpg.does_item_exist("doc_select_popup"):
            dpg.delete_item("doc_select_popup")
        if doc_type == "suite":
            pdf_path = os.path.join(self.run_dir, "_internal", "docs", "IAD_Shell.pdf")
        elif doc_type == "iad":
            pdf_path = os.path.join(self.run_dir, "_internal", "docs", "manual.pdf")
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
            pass

    def on_done(self, result):
        """Callback function to update GUI after background task completes"""
        self.log_message(result)
        self.update_status()

    def plot_raw_scattering(self, csv_path):
        """Plot raw scattering data in a Matplotlib window"""
        import matplotlib
        matplotlib.use('TkAgg')
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
        import matplotlib
        matplotlib.use('TkAgg')
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
        """Plot absorbance and reduced scattering from final_absorption_scattering.csv in a Matplotlib window"""
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import pandas as pd
        path = os.path.join(self.output_dir, "final_absorption_scattering.csv")
        if not os.path.exists(path):
            self.log_message("final_absorption_scattering.csv missing", "ERROR")
            return
        df = pd.read_csv(path)
        if df.empty or not all(col in df.columns for col in ["Sample", "Wavelength", "Absorption", "ReducedScattering"]):
            self.log_message("Absorption CSV missing required columns", "ERROR")
            return
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for sample, group in df.groupby("Sample"):
            axs[0].plot(group["Wavelength"], group["Absorption"], label=sample)
        axs[0].set_ylabel("Absorption")
        axs[0].set_title("Absorption and Reduced Scattering")
        axs[0].grid(True)
        axs[0].legend(fontsize="small")
        for sample, group in df.groupby("Sample"):
            axs[1].plot(group["Wavelength"], group["ReducedScattering"], label=sample)
        axs[1].set_ylabel("Reduced Scattering (1/mm)")
        axs[1].legend(fontsize="small", loc="upper right")
        axs[1].set_xlabel("Wavelength (nm)")
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()

    def plot_raw_and_fitted_with_mask(self, raw_path, fit_path, fit_min, fit_max):
        """Plot raw and fitted scattering with fit window in a Matplotlib window"""
        import matplotlib
        matplotlib.use('TkAgg')
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

    def _run_final_analysis_with_mode(self):
        # Set analysis_mode and model_type from GUI before running final analysis
        self.workflow.analysis_mode = dpg.get_value("analysis_mode")
        self.workflow.model_type = dpg.get_value("model_type")
        self.workflow.run_final_iad()

    def run(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            while not self.gui_queue.empty():
                action, *args = self.gui_queue.get()
                if action == 'log':
                    self.log_message(*args)
                elif action == 'set_value':
                    getattr(dpg, action)(*args)
                elif action == 'configure_item':
                    getattr(dpg, action)(*args)
                elif action == 'matplotlib_plot':
                    plot_func, plot_args = args
                    plot_func(*plot_args)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def _run_on_main_thread(self, plot_func, *args):
        import threading
        if threading.current_thread() is threading.main_thread():
            plot_func(*args)
        else:
            self.gui_queue.put(('matplotlib_plot', plot_func, args))

    def _on_analysis_mode_change(self, sender, value):
        # Map display name to internal value
        mode_internal = self.analysis_mode_display_map.get(value, "P")
        self.workflow.analysis_mode = mode_internal
        # Show/hide model_type combo depending on analysis mode
        if mode_internal == "P":
            dpg.configure_item("model_type", show=True)
            dpg.configure_item("model_type_label", show=True)
        else:
            dpg.configure_item("model_type", show=False)
            dpg.configure_item("model_type_label", show=False)