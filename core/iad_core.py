from pathlib import Path
from typing import List, Dict, Any
import subprocess
import shutil
import pandas as pd
import numpy as np
import tempfile
from scipy.optimize import curve_fit
from core.iad_model import IADModel

def check_iad_executable(model: IADModel) -> bool:
    """Check if iad.exe exists."""
    return model.paths.iad_exe.exists()

def count_input_files(model: IADModel) -> int:
    """Count .rxt files in input directory."""
    if not model.paths.input_dir.exists():
        return 0
    return len(list(model.paths.input_dir.glob('*.rxt')))

def run_initial_iad(model: IADModel) -> List[Path]:
    """Run initial IAD analysis for all .rxt files. Returns list of output files."""
    dual_beam_flag = '-X' if model.params.use_dual_beam else ''
    g_flag = f'-g {model.params.g_value}'
    processed_files = []
    for rxt_file in model.paths.input_dir.glob('*.rxt'):
        output_txt = rxt_file.with_suffix('.txt')
        cmd = [str(model.paths.iad_exe)]
        if dual_beam_flag:
            cmd.append(dual_beam_flag)
        cmd.extend([g_flag, str(rxt_file)])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            print(f"IAD stderr for {rxt_file.name}: {result.stderr}")
        if output_txt.exists():
            dest = model.paths.output_dir / output_txt.name
            shutil.copy(output_txt, dest)
            output_txt.unlink()
            processed_files.append(dest)
    return processed_files

def extract_scattering_data(model: IADModel) -> Path:
    """Extract mu_s' data from IAD outputs and save as CSV. Returns path to CSV."""
    from collections import defaultdict
    wavelength_to_us = defaultdict(dict)
    for txt_file in model.paths.output_dir.glob('*.txt'):
        if txt_file.name.startswith('combined'):
            continue
        parsed = extract_us_from_output(txt_file)
        sample_name = txt_file.stem
        for wl, us in parsed:
            wavelength_to_us[wl][sample_name] = us
    df = pd.DataFrame.from_dict(wavelength_to_us, orient='index').sort_index()
    df.index.name = 'lambda'
    out_csv = model.paths.output_dir / 'scattering_values_multi.csv'
    df.to_csv(out_csv)
    return out_csv

def extract_us_from_output(file_path: Path) -> List[tuple]:
    """Parse mu_s' values from IAD output file."""
    data = []
    in_data_section = False
    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not in_data_section:
                if 'mu_s' in stripped and 'wave' in stripped:
                    in_data_section = True
                    continue
            elif stripped and not stripped.startswith('#'):
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        wavelength = float(parts[0])
                        mu_s_prime = float(parts[6])
                        data.append((wavelength, mu_s_prime))
                    except ValueError:
                        continue
    return data

def fit_powerlaw(model: IADModel) -> Path:
    """Fit power law to scattering data and save summary CSV. Returns path to summary CSV."""
    csv_path = model.paths.output_dir / 'scattering_values_multi.csv'
    df = pd.read_csv(csv_path)
    wavelengths = df['lambda'].values.astype(float)
    samples = df.drop(columns=['lambda'])
    lambda0 = model.params.reference_wavelength
    fit_min = model.params.fit_min
    fit_max = model.params.fit_max
    fit_mask = (wavelengths >= fit_min) & (wavelengths <= fit_max)
    wavelength_fit = wavelengths[fit_mask]
    fit_results = []
    fit_curves = pd.DataFrame({'lambda': wavelengths})
    def mie_powerlaw(lambda_, b_mie, a0, lambda0):
        return a0 * (lambda_ / lambda0) ** (-b_mie)
    for sample_name in samples.columns:
        us = samples[sample_name].values.astype(float)
        us_fit = us[fit_mask]
        try:
            idx = np.where(wavelengths == lambda0)[0][0]
            a0 = us[idx]
        except IndexError:
            continue
        def fit_model(lambda_, b):
            return mie_powerlaw(lambda_, b, a0, lambda0)
        try:
            popt, _ = curve_fit(fit_model, wavelength_fit, us_fit, p0=[1.0])
            b_mie = abs(popt[0])
            us_smoothed = fit_model(wavelengths, b_mie)
            residuals = us_fit - fit_model(wavelength_fit, b_mie)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((us_fit - np.mean(us_fit))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
            fit_results.append({'sample': sample_name, 'a0': a0, 'b_mie': b_mie, 'r2': r2})
            fit_curves[sample_name] = us_smoothed
        except Exception:
            continue
    summary_csv = model.paths.output_dir / 'powerlaw_summary.csv'
    smoothed_csv = model.paths.output_dir / 'powerlaw_smoothed.csv'
    pd.DataFrame(fit_results).to_csv(summary_csv, index=False)
    fit_curves.to_csv(smoothed_csv, index=False)
    model.results.fit_results = fit_results
    return summary_csv 