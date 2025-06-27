import os
import sys
import time
import queue
import pandas as pd
import numpy as np
import re
import subprocess
import shutil
from threading import Thread
from core.iad_model import IADModel
import core.iad_core as iad_core
from scipy.optimize import curve_fit
import csv

class IADWorkflow:
    def __init__(self, config=None):
        config = config or {}
        # Set run_dir to the directory containing the executable if frozen, or the script location otherwise
        if getattr(sys, 'frozen', False):
            run_dir = os.path.dirname(sys.executable)
        else:
            run_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.run_dir = run_dir
        # Always use _internal for all internal resources
        internal_dir = os.path.join(self.run_dir, "_internal")
        self.output_dir = os.path.join(internal_dir, "iad_outputs")
        self.input_dir = os.path.join(internal_dir, "iad_inputs")
        self.iad_exe = os.path.join(internal_dir, config.get('iad_exe', 'iad.exe'))
        self.g_value = config.get('g_value', 0.8)
        self.reference_wavelength = config.get('reference_wavelength', 600)
        self.fit_min = config.get('fit_min', 600)
        self.fit_max = config.get('fit_max', 750)
        self.use_dual_beam = config.get('use_dual_beam', False)
        self.current_step = "setup"
        self.fit_results = []
        self.processing = False
        self.analysis_mode = config.get('analysis_mode', 'Power Law')
        self.font_size = config.get('font_size', 18)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        self.gui_queue = queue.Queue()
        # Pass the correct run_dir to IADModel
        self.model = IADModel(run_dir=self.run_dir)

    def get_run_directory(self):
        # Always use the project root (parent of this file's directory)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def log_message(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        # Send to GUI if possible
        if hasattr(self, 'gui_queue') and self.gui_queue is not None:
            self.gui_queue.put(('log', message, level))
        else:
            print(log_entry)

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
        return len([f for f in os.listdir(self.input_dir) if f.endswith(".rxt")])

    def update_status(self, force_reset=False):
        # This should be connected to your GUI
        iad_exists = self.check_iad_executable()
        file_count = self.count_input_files()
        if force_reset:
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
        # You can add more status logic here as needed

    def run_initial_iad(self, sender=None, data=None):
        if self.processing:
            return
        self.processing = True
        def process():
            try:
                self.log_message("Starting initial IAD analysis...")
                if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                    self.gui_queue.put(('set_value', 'progress_bar', 0.0))
                    self.gui_queue.put(('set_value', 'status_label', 'Running initial analysis...'))
                    self.gui_queue.put(('set_value', 'progress_text', 'Step 1/3: Initial analysis in progress'))
                if not self.check_iad_executable():
                    self.log_message("Cannot proceed without iad.exe", "ERROR")
                    return
                dual_beam_flag = "-X" if self.use_dual_beam else ""
                g_flag = f"-g {self.g_value}"
                processed_files = 0
                rxt_files = [f for f in os.listdir(self.input_dir) if f.endswith(".rxt")]
                total_files = len(rxt_files)
                for idx, fname in enumerate(rxt_files):
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
                    # Update progress bar for each file
                    if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                        progress = (idx + 1) / total_files if total_files else 1.0
                        self.gui_queue.put(('set_value', 'progress_bar', progress * (1/3)))
                        self.gui_queue.put(('set_value', 'progress_text', f'Step 1/3: Initial analysis ({idx+1}/{total_files})'))
                self.log_message(f"Initial IAD analysis complete. Processed {processed_files} files.")
                iad_core.extract_scattering_data(self.model)
                if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                    self.gui_queue.put(('set_value', 'progress_bar', 1/3))
                    self.gui_queue.put(('set_value', 'status_label', 'Ready for power law fitting'))
                    self.gui_queue.put(('set_value', 'progress_text', 'Step 1/3: Initial analysis complete'))
            except Exception as e:
                self.log_message(f"Error in initial IAD analysis: {str(e)}", "ERROR")
            finally:
                self.processing = False
        Thread(target=process, daemon=True).start()

    def fit_powerlaw(self, sender=None, data=None):
        if self.processing:
            return
        self.processing = True
        def process():
            try:
                self.log_message("Starting power law fitting...")
                if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                    self.gui_queue.put(('set_value', 'progress_bar', 1/3))
                    self.gui_queue.put(('set_value', 'status_label', 'Fitting power law...'))
                    self.gui_queue.put(('set_value', 'progress_text', 'Step 2/3: Power law fitting in progress'))
                csv_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
                if not os.path.exists(csv_path):
                    self.log_message("Scattering data not found. Run initial analysis first.", "ERROR")
                    return
                df = pd.read_csv(csv_path)
                wavelengths = np.asarray(df["lambda"].values, dtype=float)
                samples = df.drop(columns=["lambda"])
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
                    us = samples[sample_name].values
                    us_fit = us[fit_mask]
                    try:
                        index_600 = np.where(wavelengths == lambda0)[0][0]
                        a0 = us[index_600]
                    except IndexError:
                        self.log_message(f"lambda0 = {lambda0} nm not found in {sample_name}, skipping.", "WARNING")
                        continue
                    try:
                        def fit_model(lambda_, b):
                            return mie_powerlaw(lambda_, b, a0, lambda0)
                        us_fit = np.asarray(us_fit, dtype=float)
                        popt, _ = curve_fit(fit_model, wavelength_fit, us_fit, p0=[1.0])
                        b_mie = abs(popt[0])
                        us_smoothed = fit_model(wavelengths, b_mie)
                        residuals = us_fit - fit_model(wavelength_fit, b_mie)
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((us_fit - np.mean(us_fit))**2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
                        fit_results.append({"sample": sample_name, "a0": a0, "b_mie": b_mie, "r2": r2})
                        fit_curves[sample_name] = us_smoothed
                        self.log_message(f"Fit {sample_name}: a0 = {a0:.4f}, b_mie = {b_mie:.4f}, R^2 = {r2:.4f}")
                    except Exception as e:
                        self.log_message(f"Fitting failed for {sample_name}: {str(e)}", "ERROR")
                    # Update progress bar for each sample
                    if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                        progress = (i + 1) / num_samples if num_samples else 1.0
                        self.gui_queue.put(('set_value', 'progress_bar', (1/3) + progress * (1/3)))
                        self.gui_queue.put(('set_value', 'progress_text', f'Step 2/3: Power law fitting ({i+1}/{num_samples})'))
                summary_csv = os.path.join(self.output_dir, "powerlaw_summary.csv")
                smoothed_csv = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
                if fit_results:
                    pd.DataFrame(fit_results).to_csv(summary_csv, index=False)
                fit_curves.to_csv(smoothed_csv, index=False)
                self.fit_results = fit_results
                self.log_message(f"Power law fitting complete. Fitted {len(fit_results)} samples.")
                if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                    self.gui_queue.put(('set_value', 'progress_bar', 2/3))
                    self.gui_queue.put(('set_value', 'status_label', 'Ready for final analysis'))
                    self.gui_queue.put(('set_value', 'progress_text', 'Step 2/3: Power law fitting complete'))
            except Exception as e:
                self.log_message(f"Error in power law fitting: {str(e)}", "ERROR")
            finally:
                self.processing = False
        Thread(target=process, daemon=True).start()

    def run_final_iad(self, sender=None, data=None):
        if self.processing:
            return
        self.processing = True
        def process():
            try:
                self.log_message("Initializing Step 3: Final IAD analysis...")
                if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                    self.gui_queue.put(('set_value', 'progress_bar', 2/3))
                    self.gui_queue.put(('set_value', 'status_label', 'Running final analysis...'))
                    self.gui_queue.put(('set_value', 'progress_text', 'Step 3/3: Final analysis in progress'))
                # mode and model_type should be passed in or set as attributes
                mode = getattr(self, 'analysis_mode', 'Power Law')
                model_type = getattr(self, 'model_type', 'R')
                if mode == "Power Law":
                    self.run_powerlaw_mode(model_type)
                else:
                    self.run_fixed_mode()
                self.log_message("Final IAD analysis complete.")
                if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                    self.gui_queue.put(('set_value', 'progress_bar', 1.0))
                    self.gui_queue.put(('set_value', 'status_label', 'Analysis complete'))
                    self.gui_queue.put(('set_value', 'progress_text', 'Step 3/3: Final analysis complete'))
            except Exception as e:
                self.log_message(f"Error in final IAD analysis: {str(e)}", "ERROR")
            finally:
                self.processing = False
        Thread(target=process, daemon=True).start()

    def run_powerlaw_mode(self, model_type):
        summary_path = os.path.join(self.output_dir, "powerlaw_summary.csv")
        if not os.path.exists(summary_path):
            self.log_message("Power law summary not found. Run fitting first.", "ERROR")
            return
        summary_df = pd.read_csv(summary_path)
        lambda0 = self.reference_wavelength
        g = self.g_value
        if abs(1.0 - g) < 1e-9:
            self.log_message("Anisotropy (g) cannot be 1, this would cause a division by zero.", "ERROR")
            return
        rxt_files = [f for f in os.listdir(self.input_dir) if f.endswith(".rxt")]
        num_files = len(rxt_files)
        for i, fname in enumerate(rxt_files):
            sample_name = os.path.splitext(fname)[0]
            input_path = os.path.join(self.input_dir, fname)
            expected_output_txt = input_path.replace(".rxt", ".txt")
            row = summary_df[summary_df["sample"] == sample_name]
            if row.empty:
                self.log_message(f"No fit data found for {sample_name}, skipping.", "WARNING")
                continue
            a0_reduced = row["a0"].item()
            b_mie = row["b_mie"].item()
            a0_scattering = a0_reduced / (1 - g)
            dual_beam_flag = '-X' if self.use_dual_beam else ''
            cmd = [self.iad_exe]
            if dual_beam_flag:
                cmd.append(dual_beam_flag)
            cmd.extend(["-F", f"{model_type} {lambda0} {a0_scattering:.6f} {b_mie:.6f}", input_path])
            self.log_message(f"Processing final analysis for: {sample_name}")
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(expected_output_txt):
                dest_path = os.path.join(self.output_dir, os.path.basename(expected_output_txt))
                shutil.copy(expected_output_txt, dest_path)
                os.remove(expected_output_txt)
            else:
                self.log_message(f"No output generated for: {sample_name}", "WARNING")
            # Update progress bar for each file
            if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                progress = (i + 1) / num_files if num_files else 1.0
                self.gui_queue.put(('set_value', 'progress_bar', (2/3) + progress * (1/3)))
                self.gui_queue.put(('set_value', 'progress_text', f'Step 3/3: Final analysis ({i+1}/{num_files})'))
        self.export_final_absorption_scattering_csv()

    def run_fixed_mode(self):
        scatter_path = os.path.join(self.output_dir, "scattering_values_multi.csv")
        if not os.path.exists(scatter_path):
            self.log_message("Scattering data not found. Run initial analysis first.", "ERROR")
            return
        scatter_df = pd.read_csv(scatter_path)
        lambda_values = scatter_df["lambda"].values
        temp_rxt = os.path.join(self.run_dir, "temp_single.rxt")
        combined_output = os.path.join(self.output_dir, "combined_output.txt")
        if os.path.exists(combined_output):
            os.remove(combined_output)
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
                        current_task += 1
                        # Update progress bar for each task
                        if hasattr(self, 'gui_queue') and self.gui_queue is not None:
                            progress = current_task / total_tasks if total_tasks else 1.0
                            self.gui_queue.put(('set_value', 'progress_bar', (2/3) + progress * (1/3)))
                            self.gui_queue.put(('set_value', 'progress_text', f'Step 3/3: Final analysis ({current_task}/{total_tasks})'))
                except Exception as e:
                    self.log_message(f"Could not process {sample_name}: {str(e)}", "ERROR")
                    continue
        self.export_final_absorption_scattering_csv()

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

    def export_final_absorption_scattering_csv(self):
        """Create a CSV with Sample, Wavelength, Absorption (from final step), and ReducedScattering (from powerlaw_smoothed.csv)."""
        output_csv = os.path.join(self.output_dir, "final_absorption_scattering.csv")
        smoothed_csv = os.path.join(self.output_dir, "powerlaw_smoothed.csv")
        smoothed_df = pd.read_csv(smoothed_csv)
        # Remove any columns with 'final_' prefix for matching
        smoothed_df = smoothed_df.rename(columns={col: col.replace('final_', '') for col in smoothed_df.columns})
        # Build a dict: (sample, wavelength) -> reduced scattering
        scattering_dict = {}
        for _, row in smoothed_df.iterrows():
            wavelength = float(row["lambda"])
            for sample in smoothed_df.columns:
                if sample == "lambda":
                    continue
                value = row[sample]
                if isinstance(value, pd.Series):
                    value = float(value.values[0])
                elif isinstance(value, np.ndarray):
                    value = float(value[0])
                scattering_dict[(sample, wavelength)] = float(value)
        # Prepare output rows
        output_rows = []
        # Check if combined_output.txt exists (fixed mode)
        combined_path = os.path.join(self.output_dir, "combined_output.txt")
        if os.path.exists(combined_path):
            # Fixed mode: parse combined_output.txt
            with open(combined_path) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            sample = parts[0]
                            wavelength = float(parts[1])
                            absorption = float(parts[6])
                            key = (sample, wavelength)
                            reduced_scattering = scattering_dict.get(key, None)
                            output_rows.append({
                                "Sample": sample,
                                "Wavelength": wavelength,
                                "Absorption": absorption,
                                "ReducedScattering": reduced_scattering
                            })
        else:
            # Power law mode: parse each .txt file in output_dir
            for fname in os.listdir(self.output_dir):
                if fname.endswith(".txt") and not fname.startswith("combined"):
                    sample = fname.replace(".txt", "")
                    with open(os.path.join(self.output_dir, fname)) as f:
                        for line in f:
                            if line.strip() and not line.startswith("#"):
                                parts = line.strip().split()
                                if len(parts) >= 7:
                                    wavelength = float(parts[0])
                                    absorption = float(parts[6])
                                    key = (sample, wavelength)
                                    reduced_scattering = scattering_dict.get(key, None)
                                    output_rows.append({
                                        "Sample": sample,
                                        "Wavelength": wavelength,
                                        "Absorption": absorption,
                                        "ReducedScattering": reduced_scattering
                                    })
        # Write to CSV
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Sample", "Wavelength", "Absorption", "ReducedScattering"])
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)
        self.log_message(f"Exported final absorption and scattering CSV: {os.path.basename(output_csv)}") 