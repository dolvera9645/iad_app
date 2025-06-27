from pathlib import Path

class Paths:
    def __init__(self, run_dir):
        internal_dir = Path(run_dir) / "_internal"
        self.iad_exe = internal_dir / "iad.exe"
        self.input_dir = internal_dir / "iad_inputs"
        self.output_dir = internal_dir / "iad_outputs"

class Params:
    def __init__(self):
        self.use_dual_beam = False
        self.g_value = 0.8
        self.reference_wavelength = 600
        self.fit_min = 600
        self.fit_max = 750

class Results:
    def __init__(self):
        self.fit_results = []

class IADModel:
    def __init__(self, run_dir="."):
        self.paths = Paths(run_dir)
        self.params = Params()
        self.results = Results()
        # Initialize model attributes here
        pass 