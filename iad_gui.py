import os
import subprocess
import sys
import shutil
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from threading import Thread
import time


class IADAnalysisGUI:
    """Interactive GUI for inverse adding-doubling analysis."""

    def __init__(self):
        self.g_value = 0.8
        self.use_dual_beam = False
        self.run_dir = self.get_run_directory()
        self.iad_exe = os.path.join(self.run_dir, "iad.exe")
        self.input_dir = os.path.join(self.run_dir, "iad_inputs")
        self.output_dir = os.path.join(self.run_dir, "iad_outputs")
        self.fit_results = []
        self.processing = False

        os.makedirs(self.output_dir, exist_ok=True)

        dpg.create_context()
        dpg.create_viewport(title="IAD Optical Analysis Suite", width=900, height=700)

    def get_run_directory(self):
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))

    def log_message(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        log_text = f"[{timestamp}] [{level}] {message}\n"

        if dpg.does_item_exist("log_text"):
            current_text = dpg.get_value("log_text")
            dpg.set_value("log_text", current_text + log_text)
            dpg.set_y_scroll("log_scroll", -1.0)

    def check_iad_executable(self):
        exists = os.path.exists(self.iad_exe)
        if exists:
            self.log_message(f"iad.exe found at: {self.iad_exe}")
        else:
            self.log_message(f"iad.exe NOT found at: {self.iad_exe}", "ERROR")
        return exists

    def count_input_files(self):
        if not os.path.exists(self.input_dir):
            return 0
        return len([f for f in os.listdir(self.input_dir) if f.endswith('.rxt')])

    def update_status(self):
        iad_exists = self.check_iad_executable()
        theme = self.success_theme if iad_exists else self.error_theme
        dpg.set_value("iad_status", "\u2713 Found" if iad_exists else "\u2717 Missing")
        dpg.bind_item_theme("iad_status", theme)

        file_count = self.count_input_files()
        file_theme = self.success_theme if file_count > 0 else self.error_theme
        dpg.set_value("files_status", f"\u2713 {file_count} files" if file_count > 0 else "\u2717 No files")
        dpg.bind_item_theme("files_status", file_theme)

        steps_completed = 0
        if os.path.exists(os.path.join(self.output_dir, "scattering_values_multi.csv")):
            steps_completed = 1
        if os.path.exists(os.path.join(self.output_dir, "powerlaw_summary.csv")):
            steps_completed = 2
        if os.path.exists(os.path.join(self.output_dir, "combined_output.txt")):
            steps_completed = 3

        progress = steps_completed / 3.0
        dpg.set_value("progress_bar", progress)
        dpg.set_value("progress_text", f"Step {steps_completed}/3 Complete")

    def on_g_value_change(self, sender, value):
        self.g_value = value
        self.log_message(f"Anisotropy value (g) set to: {value:.3f}")

    def on_dual_beam_change(self, sender, value):
        self.use_dual_beam = value
        status = "ENABLED" if value else "DISABLED"
        self.log_message(f"Dual beam correction {status}")

    def run_initial_iad(self, sender=None, data=None):
        if self.processing:
            return
        self.processing = True
        dpg.set_value("btn_step1", "Processing...")
        dpg.configure_item("btn_step1", enabled=False)

        def process():
            try:
                self.log_message("Starting initial IAD analysis...")
                if not self.check_iad_executable():
                    self.log_message("Cannot proceed without iad.exe", "ERROR")
                    return
                dual_flag = "-X" if self.use_dual_beam else ""
                g_flag = f"-g {self.g_value}"
                processed = 0
                for fname in os.listdir(self.input_dir):
                    if fname.endswith('.rxt'):
                        input_path = os.path.join(self.input_dir, fname)
                        output_txt = input_path.replace('.rxt', '.txt')
                        cmd = [self.iad_exe]
                        if dual_flag:
                            cmd.append(dual_flag)
                        cmd.extend([g_flag, input_path])
                        self.log_message(f"Processing: {fname}")
                        subprocess.run(cmd, capture_output=True)
                        if os.path.exists(output_txt):
                            dest = os.path.join(self.output_dir, os.path.basename(output_txt))
                            shutil.copy(output_txt, dest)
                            os.remove(output_txt)
                            processed += 1
                        else:
                            self.log_message(f"No output generated for: {fname}", "WARNING")
                self.log_message(f"Initial IAD analysis complete. Processed {processed} files.")
                self.extract_scattering_data()
            except Exception as exc:
                self.log_message(f"Error in initial IAD analysis: {exc}", "ERROR")
            finally:
                self.processing = False
                dpg.set_value("btn_step1", "Run Initial Analysis")
                dpg.configure_item("btn_step1", enabled=True)
                self.update_status()

        Thread(target=process, daemon=True).start()

    def extract_scattering_data(self):
        try:
            self.log_message("Extracting scattering coefficients...")
            wavelength_to_us = defaultdict(dict)
            for fname in os.listdir(self.output_dir):
                if fname.endswith('.txt') and not fname.startswith('combined'):
                    sample = os.path.splitext(fname)[0]
                    full = os.path.join(self.output_dir, fname)
                    parsed = self.extract_us_from_output(full)
                    for wl, us in parsed:
                        wavelength_to_us[wl][sample] = us
            df = pd.DataFrame.from_dict(wavelength_to_us, orient='index').sort_index()
            df.index.name = 'lambda'
            out_csv = os.path.join(self.output_dir, 'scattering_values_multi.csv')
            df.to_csv(out_csv)
            self.log_message(f"Scattering data extracted and saved to: {os.path.basename(out_csv)}")
        except Exception as exc:
            self.log_message(f"Error extracting scattering data: {exc}", "ERROR")

    def extract_us_from_output(self, file_path):
        data = []
        in_data = False
        with open(file_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not in_data:
                    if 'mu_s' in stripped and 'wave' in stripped:
                        in_data = True
                        continue
                elif stripped and not stripped.startswith('#'):
                    parts = stripped.split()
                    if len(parts) >= 7:
                        try:
                            wl = float(parts[0])
                            mu_s_prime = float(parts[6])
                            data.append((wl, mu_s_prime))
                        except ValueError:
                            continue
        return data

    def fit_powerlaw(self, sender=None, data=None):
        if self.processing:
            return
        self.processing = True
        dpg.set_value("btn_step2", "Processing...")
        dpg.configure_item("btn_step2", enabled=False)

        def process():
            try:
                self.log_message("Starting power law fitting...")
                csv_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
                if not os.path.exists(csv_path):
                    self.log_message("Scattering data not found. Run initial analysis first.", "ERROR")
                    return
                df = pd.read_csv(csv_path)
                wavelengths = df['lambda'].values
                samples = df.drop(columns=['lambda'])
                lambda0 = 600
                fit_min, fit_max = 600, 750
                mask = (wavelengths >= fit_min) & (wavelengths <= fit_max)
                wl_fit = wavelengths[mask]
                fit_results = []
                fit_curves = pd.DataFrame({'lambda': wavelengths})

                def mie_powerlaw(lmbda, b, a0, l0):
                    return a0 * (lmbda / l0) ** (-b)

                for sample in samples.columns:
                    us = samples[sample].values
                    us_fit = us[mask]
                    try:
                        idx = np.where(wavelengths == lambda0)[0][0]
                        a0 = us[idx]
                    except IndexError:
                        self.log_message(f"\u03bb0 = {lambda0} nm not found in {sample}, skipping.", "WARNING")
                        continue

                    def fit_model(lmbda, b):
                        return mie_powerlaw(lmbda, b, a0, lambda0)

                    popt, _ = curve_fit(fit_model, wl_fit, us_fit, p0=[1.0])
                    b_mie = abs(popt[0])
                    fit_curves[sample] = fit_model(wavelengths, b_mie)
                    fit_results.append({"sample": sample, "a0": a0, "b_mie": b_mie})
                    self.log_message(f"Fit {sample}: a0 = {a0:.4f}, b_mie = {b_mie:.4f}")

                pd.DataFrame(fit_results).to_csv(os.path.join(self.output_dir, 'powerlaw_summary.csv'), index=False)
                fit_curves.to_csv(os.path.join(self.output_dir, 'powerlaw_smoothed.csv'), index=False)
                self.fit_results = fit_results
                self.log_message(f"Power law fitting complete. Fitted {len(fit_results)} samples.")
            except Exception as exc:
                self.log_message(f"Error in power law fitting: {exc}", "ERROR")
            finally:
                self.processing = False
                dpg.set_value("btn_step2", "Fit Power Law")
                dpg.configure_item("btn_step2", enabled=True)
                self.update_status()

        Thread(target=process, daemon=True).start()

    def create_gui(self):
        with dpg.theme(tag="success_theme") as self.success_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0))

        with dpg.theme(tag="error_theme") as self.error_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0))

        # main window
        with dpg.window(label="IAD Optical Analysis", tag="main_window"):
            with dpg.group(horizontal=True):
                dpg.add_text("IAD Optical Analysis Suite", color=(100, 200, 255))
                dpg.add_spacer(width=200)
                dpg.add_button(label="Refresh Status", callback=self.update_status)
            dpg.add_separator()
            with dpg.collapsing_header(label="System Status", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("IAD Executable:")
                    dpg.add_text("Checking...", tag="iad_status")
                    dpg.add_spacer(width=50)
                    dpg.add_text("Input Files:")
                    dpg.add_text("Checking...", tag="files_status")
                dpg.add_spacer(height=10)
                dpg.add_text("Overall Progress:")
                dpg.add_progress_bar(tag="progress_bar", width=400)
                dpg.add_text("Not Started", tag="progress_text")
                with dpg.group(horizontal=True):
                    dpg.add_text("Status:", color=(180, 180, 180))
                    dpg.add_text("Idle", tag="status_label", color=(200, 200, 0))
            dpg.add_separator()
            with dpg.collapsing_header(label="Analysis Settings", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Anisotropy factor (g):")
                    dpg.add_slider_float(tag="g_slider", default_value=0.8, min_value=0.0, max_value=1.0, width=200,
                                         callback=self.on_g_value_change, format="%.3f")
                dpg.add_checkbox(label="Use dual beam correction", tag="dual_beam_check", default_value=False,
                                  callback=self.on_dual_beam_change)
            dpg.add_separator()
            with dpg.collapsing_header(label="Analysis Steps", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="1. Run Initial Analysis", tag="btn_step1", callback=self.run_initial_iad, width=200)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="2. Fit Power Law", tag="btn_step2", callback=self.fit_powerlaw, width=200)
        dpg.set_primary_window("main_window", True)

    def run(self):
        self.create_gui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.update_status()
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


if __name__ == "__main__":
    IADAnalysisGUI().run()
