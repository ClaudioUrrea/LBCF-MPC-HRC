# Installation Guide

This guide provides detailed instructions for installing and setting up the LBCF-MPC-HRC framework.

## Table of Contents

- [System Requirements](#system-requirements)
- [CoppeliaSim Installation](#coppeliasim-installation)
- [Python Environment Setup](#python-environment-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

- **CPU**: Intel Core i7 or equivalent (quad-core minimum).
- **RAM**: 8GB minimum, 16GB recommended.
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training).
- **Storage**: 10GB free disk space.

### Operating System

- Ubuntu 20.04 LTS or later (recommended).
- Windows 10/11 (supported).
- macOS 11+ (experimental support).

### Software

- Python 3.8 or higher.
- CoppeliaSim EDU 4.6.0 or later.
- CUDA 11.0+ (optional, for GPU acceleration).
- Git (for cloning repository).

---

## CoppeliaSim Installation

### Linux (Ubuntu)

```bash
# Download CoppeliaSim EDU 4.6.0
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04.tar.xz

# Extract
tar -xf CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04.tar.xz

# Move to /opt
sudo mv CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04 /opt/coppeliasim

# Add to PATH
echo 'export PATH=$PATH:/opt/coppeliasim' >> ~/.bashrc
source ~/.bashrc

# Test installation
coppeliaSim.sh
```

### Windows

1. Download CoppeliaSim EDU from: https://www.coppeliarobotics.com/downloads
2. Run the installer `CoppeliaSim_Edu_V4_6_0_Setup.exe`.
3. Follow installation wizard.
4. Add CoppeliaSim to system PATH:
   - Right-click "This PC" → Properties → Advanced system settings.
   - Environment Variables → Path → Edit → New.
   - Add: `C:\Program Files\CoppeliaSim`

### macOS

```bash
# Download .dmg from CoppeliaSim website
# Open CoppeliaSim_Edu_V4_6_0_macOS.dmg
# Drag CoppeliaSim to Applications

# Test installation
/Applications/CoppeliaSim/coppeliaSim.app/Contents/MacOS/coppeliaSim
```

---

## Python Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/ClaudioUrrea/LBCF-MPC-HRC.git
cd LBCF-MPC-HRC
```

### 2. Create Virtual Environment

#### Using venv (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

#### Using conda (alternative)

```bash
conda create -n lbcf python=3.8
conda activate lbcf
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install PyRep for CoppeliaSim integration
pip install pyrep
```

### 4. Install CoppeliaSim Python API

#### Linux/macOS

```bash
# Add CoppeliaSim to Python path
export COPPELIASIM_ROOT=/opt/coppeliasim
export PYTHONPATH=$PYTHONPATH:$COPPELIASIM_ROOT/programming/remoteApiBindings/python/python

# Add to ~/.bashrc for persistence
echo 'export COPPELIASIM_ROOT=/opt/coppeliasim' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:$COPPELIASIM_ROOT/programming/remoteApiBindings/python/python' >> ~/.bashrc
```

#### Windows

```cmd
set COPPELIASIM_ROOT=C:\Program Files\CoppeliaSim
set PYTHONPATH=%PYTHONPATH%;%COPPELIASIM_ROOT%\programming\remoteApiBindings\python\python
```

### 5. Install PyTorch with CUDA (Optional)

For GPU acceleration:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Verification

### Test Python Environment

```bash
python -c "import torch; import numpy; import osqp; print('All core packages installed successfully!')"
```

### Test CoppeliaSim Integration

```bash
# Start CoppeliaSim in headless mode
coppeliaSim.sh -h

# In another terminal, run test script
python tests/test_coppeliasim.py
```

Expected output:
```
Testing CoppeliaSim connection...
✓ CoppeliaSim connected successfully
✓ Scene loaded: hrc_workspace.ttt
✓ Simulation started
✓ Robot controlled successfully
Test completed!
```

### Run Quick Demo

```bash
# Generate sample data
python scripts/generate_sample_data.py --samples 1000

# Train small CBF (fast test)
python scripts/train_cbf.py --samples 1000 --epochs 10 --output models/test/

# Run MPC test
python scripts/test_mpc.py --model models/test/cbf_model_final.pth
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sim'"

**Solution**:
Ensure CoppeliaSim Python API is in PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/opt/coppeliasim/programming/remoteApiBindings/python/python
```

### Issue: "CUDA out of memory"

**Solution**:
Reduce batch size in training:
```bash
python scripts/train_cbf.py --batch-size 128  # Instead of 256
```

### Issue: CoppeliaSim won't start in headless mode

**Solution**:
Install required X11 libraries (Linux):
```bash
sudo apt-get install xvfb
xvfb-run coppeliaSim.sh -h
```

### Issue: "OSError: libcoppeliaSim.so: cannot open shared object file"

**Solution**:
Add CoppeliaSim library path:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/coppeliasim
```

### Issue: PyRep installation fails

**Solution**:
Install build dependencies:
```bash
# Ubuntu
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Then retry
pip install pyrep
```

---

## Next Steps

After successful installation:

1. Read [USAGE.md](USAGE.md) for basic usage examples.
2. Explore [notebooks/](../notebooks/) for interactive tutorials.
3. Run full experiments: `python scripts/run_experiments.py`
4. Check [API.md](API.md) for detailed API reference.

---

## Support

For installation issues:

- **GitHub Issues**: https://github.com/ClaudioUrrea/LBCF-MPC-HRC/issues
- **Email**: claudio.urrea@usach.cl
- **CoppeliaSim Forum**: https://forum.coppeliarobotics.com/

---

**Last Updated**: December 26, 2025
