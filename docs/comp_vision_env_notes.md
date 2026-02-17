# `comp_vision` Environment Notes

This project is configured to run inside your Conda env `comp_vision`.

## Activation Repair (PowerShell)

If you see errors pointing to `C:\ProgramData\anaconda3\...` while you use
`miniconda3`, reset the shell hook in the current terminal:

```powershell
Remove-Module Conda -ErrorAction SilentlyContinue
$env:CONDA_EXE = "C:\Users\umair\miniconda3\Scripts\conda.exe"
(& $env:CONDA_EXE "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate comp_vision
python -c "import sys, numpy; print(sys.executable); print(numpy.__version__)"
```

If activation remains unreliable, bypass Conda activation and run directly with
the environment python executable:

```powershell
& "C:\Users\umair\miniconda3\envs\comp_vision\python.exe" scripts/train.py --help
```

If `train.py` import fails, run:

```powershell
conda activate comp_vision
python scripts/check_env.py --save-json runs/env_report.json
```

## Known Failure Pattern

A common failure is:
- NumPy 2.x installed
- TensorFlow/TensorBoard packages built against NumPy 1.x
- Stable-Baselines3 import chain triggers TensorBoard/TensorFlow import errors

Symptoms include:
- `_ARRAY_API not found`
- `numpy.core._multiarray_umath failed to import`
- TensorBoard/TensorFlow compatibility tracebacks

## Recommended Fix Path

Use a clean RL-compatible dependency set in `comp_vision`:

1. Pin `numpy<2`.
2. If TensorFlow is not needed for this RL project, remove it from `comp_vision`.
3. Reinstall SB3 stack against the pinned NumPy.

Example command sequence (run manually when ready):

```powershell
conda activate comp_vision
pip install --upgrade "numpy<2"
pip uninstall -y tensorflow tensorflow-intel
pip install --upgrade --force-reinstall stable-baselines3 torch tensorboard matplotlib gymnasium
```

Then verify:

```powershell
python scripts/train.py --help
```

## Current Project Status

- Wrapper scripts now auto-relaunch with `python -s` to ignore user-site packages
  (prevents `AppData\\Roaming\\Python` shadowing Conda packages).
- `scripts/compose_env.py` is operational in `comp_vision`.
- `configs/env_hw_merged.yaml` is generated and valid.
- `scripts/train.py --help` and `scripts/eval.py --help` run successfully in `comp_vision`.
