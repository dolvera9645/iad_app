# iad_app

# IAD Batch Automation Tool

**IAD Batch Automation Tool** is a portable Windows application that automates batch processing of `.rxt` input files using the Inverse Adding-Doubling (IAD) method. This tool simplifies running `iad.exe` across multiple datasets, provides a progress bar and splash screen, and organizes outputs into a clean directory.

## ✨ Features

- 🖥️ Standalone `.exe` — no Python required
- 🧠 Automatically processes all `.rxt` files using `iad.exe -X`
- 📂 Inputs/outputs handled automatically
- 🎛️ Visual splash screen and progress bar
- 🧹 Cleanup and file organization included
- 🧑‍💻 Easily extensible and customizable

## 📁 Folder Structure
IADBatchApp/
```
IADBatchApp/ 
├── iad_automation.exe # Main executable
├── iad.exe # IAD processor (from Scott Prahl's repo)
├── splash.png # Splash image on launch
├── icon.ico # Custom app icon (used for .exe)
├── iad_inputs/ # Place .rxt input files here
└── iad_outputs/ # Processed output files (.txt) are saved here ``` </pre>


## 🚀 Getting Started

1. Place your `.rxt` files into the `iad_inputs` folder.
2. Run `iad_automation.exe` by double-clicking it.
3. Wait for processing to finish (watch the progress bar).
4. Check the `iad_outputs` folder for your results.

## 🛠 Building From Source

If you'd like to modify or rebuild the tool from source:

### Prerequisites
- Python 3.11+
- `pyinstaller`, `tqdm`, `pillow`

### Installation
```bash
pip install pyinstaller tqdm pillow




