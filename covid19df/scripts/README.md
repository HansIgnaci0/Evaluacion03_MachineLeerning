Run the ad-hoc unsupervised analysis script

Usage (PowerShell):

1. Activate the project's virtual environment (adjust path if needed):

```powershell
& 'C:\Users\hansi\Desktop\kedro\covid19DF_Ev02\venv_kedro\Scripts\Activate.ps1'
```

2. Run the script (choose the command that matches your current working directory):

- From the project root `C:\Users\hansi\Desktop\kedro\covid19DF_Ev02`:

```powershell
python covid19df\scripts\run_unsupervised_ad_hoc.py
```

- Or, if your current directory is `C:\Users\hansi\Desktop\kedro\covid19DF_Ev02\covid19df`:

```powershell
python scripts\run_unsupervised_ad_hoc.py
```

Note: the script is located at `covid19df/scripts/run_unsupervised_ad_hoc.py` (not under `src`).

Exporting the report notebook to HTML
-----------------------------------
After running the ad-hoc script, you can generate a presentable HTML report from the notebook we added at `notebooks/unsupervised_report.ipynb`:

From the project root:
```powershell
python -m nbconvert --to html notebooks\unsupervised_report.ipynb --output unsupervised_report.html
```

This produces `unsupervised_report.html` in your current directory; attach that file for delivery.

What it does:
- Detects the project root and adds `src` to `sys.path`.
- Loads the cleaned CSV from `data/03_intermediate`.
- Runs preprocessing, elbow, KMeans and PCA and saves outputs to `data/05_train`.

If you see import errors, ensure the venv in step 1 is the same Python kernel/environment you use for JupyterLab. To make the venv available inside Jupyter, install `ipykernel` in the venv and run:

```powershell
pip install ipykernel
python -m ipykernel install --user --name covid19df-venv --display-name "covid19df (venv_kedro)"
```

Then select that kernel in JupyterLab and re-run the notebook.
