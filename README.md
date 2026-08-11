# Learning Glacier Runoff from Streamflow: A Counterfactual Evaluation of Deep Hydrological Models

Official code repository and computational pipeline for:
> **Learning Glacier Runoff from Streamflow: A Counterfactual Evaluation of Deep Hydrological Models**  
> Tyler Wilson and Valentina Radić 
> *Department of Earth, Ocean and Atmospheric Sciences, University of British Columbia*  
> (Submitted 2026)  
> **Correspondence:** Tyler Wilson (twilson@eoas.ubc.ca)

## Citation

**APA:**
> Wilson, T. (2026). LSTM Glacier Counterfactuals Code (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21879838

**BibTeX**
```bibtex
@misc{wilson_lstm_counterfactual_2026_zenodo,
  author = {Wilson, Tyler},
  title = {LSTM Glacier Counterfactuals Code},
  month = aug,
  year = 2026,
  copyright = {Creative Commons Attribution 4.0 International},
  howpublished = {Zenodo},
  doi = {https://doi.org/10.5281/zenodo.21879838}
  }
```

---

## Overview

This repository implements a regional deep-learning framework to investigate whether glacier contributions to streamflow can be inferred directly from hydrological observations without explicit glacier physics or mass-balance training data.

Using a regional Long Short-Term Memory (LSTM) network trained across 269 catchments in southwestern Canada (1980–2022), we perform a **counterfactual experiment**:
1. **Factual Simulation ($Q_f$):** Streamflow predicted using observed meteorological forcing and actual catchment attributes (including fractional glacier cover $g$).
2. **Counterfactual Simulation ($Q_{cf}$):** Streamflow predicted after setting fractional glacier cover to zero ($g = 0$) while holding all meteorological forcings and other topographic attributes identical.
3. **Inferred Glacier Runoff ($Q_g$):** $Q_g = Q_f - Q_{cf}$, representing the glacier-driven streamflow component learned by the model.

The code in this repository can reproduce all figures and findings in the study. All data used is publicly available. Setup instructions are given below.

---

## Local Setup & Preprocessing
Before training on the cluster, it is recommended that the data be downloaded and preprocessed locally. To avoid conflicts with other projects, you should do this within an isolated Python virtual environment.

1. **Create and Activate a Virtual Environment:**
   Ensure you have Python 3.10+ installed. Open your terminal in the project root and run:

   *For Windows (Command Prompt / PowerShell):*
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
   *(Note: If PowerShell returns an "UnauthorizedAccess" or "running scripts is disabled" error, run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as an administrator or for your current user, then try activating again).*

   *For macOS/Linux*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Dependencies:**
   With the environment active (you should see `(.venv)` in your terminal prompt), install the training requirements and the local package:

   ```bash
   pip install -r preprocessing_requirements.txt
   pip install -e .
   ```

3. **Run Preprocessing:**
   Launch Jupyter Notebook from your active environment:
   ```bash
   jupyter notebook
   ```
   Open and run `notebooks/01_data_preprocessing.ipynb` in full.
   * Note: This notebook downloads ERA5 reanalysis data which can take a significant amount of time depending on the Copernicus server queues.
   * Outcome: This generates the lightweight CSVs in `data/processed/` required for the training pipeline.

---

**Files of particular note:**
* `data/processed/combined_streamflow.csv`: this is the ground-truth streamflow data in units of $m^3/s$. Note that there will be some gaps in this data.
* `data/processed/glacier_volume_change_x.csv`: this is the monthly changes in mass balance from the mass balance model aggregated for each station. Units are in millions of cubic meters of water (MCM). The three files are different outputs of the mass balance model. `x=1` is the best performing model. Using all three models allows for an estimate of model confidence.
* `data/processed/static_attributes.csv`: this is the values of static variables for each station. Area is in units of $\mathrm{km}^2$, elevation is in $\mathrm{m}$, and slope is unitless.
* `data/processed/climate/`: this folder contains CSV files of the dynamic variables, structured in the same way as `combined_streamflow.csv`. Temperature has units of degrees celcius while precipitation variables are in units of millimeters averaged over the basin.
* `data/output` contains daily predictions for each model, structured in the same manner as `combined_streamflow` except using units of millimeters over the basin area.
* `src/config.py` is used to consistantly reference common files and directories. Include this in your import statment when developing code or performing analysis. `from src.config import ___`.

---

## Repository Structure

```text
lstm_glacier_counterfactuals/
├── data/
│   ├── output/                  # model predictions of streamflow
│   ├── processed/               # data processed for use in model training and analysis
│   │   ├── climate/                    # daily climate data for each basin
│   │   │   ├── daily_precipitation.csv # daily precipitation by basin
│   │   │   ├── daily_rainfall.csv      # daily rainfall by basin
│   │   │   ├── daily_snowfall.csv      # daily snowfall by basin
│   │   │   ├── daily_temp_max.csv      # daily max temperature by basin
│   │   │   └── daily_temp_min.csv      # daily min temperature by basin
│   │   ├── combined_streamflow.csv     # daily observed mean streamflow per basin
│   │   ├── glacier_volume_change_1.csv # monthly mass balance by basin for fnn file
│   │   ├── glacier_volume_change_2.csv # monthly mass balance by basin for fnn_cluster file
│   │   ├── glacier_volume_change_3.csv # monthly mass balance by basin for lstm file
│   │   └── static_attributes.csv       # static attributes for each basin
│   └── raw/                     # raw data downloads (unmodified)
│       ├── dem_data/            # downloaded digital elevation maps
│       ├── drainage_areas/      # drainage area polygons
│       ├── era5/                # temperature and precipitation data downloaded from ERA5
│       │   ├── precipitation/   # raw hourly precipitation data
│       │   └── temperature/     # raw hourly temperature data
│       ├── mass_balance/        # Modeled monthly mass balance data
│       │   ├── ts_monthly_const_area_fnn.csv
│       │   ├── ts_monthly_const_area_fnn_cluster.csv
│       │   └── ts_monthly_const_area_lstm.csv
│       ├── RGI-western-canada/  # RGI glacier data and polygons
│       ├── spatial_bounds.csv   # latitude and longitude bounds used for downloads
│       └── station_metadata.csv # raw HYDAT station metadata
├── hpc/                         # HPC (UBC Sockeye) submission & environment scripts
│   ├── job.sh                   # Main SLURM execution script
│   ├── setup_env.sh             # Conda environment setup for PyTorch/CUDA
│   └── submit.sh                # Job submission wrapper
├── models/                      # Saved models
├── local_postprocessing/        # Local setup for figures notebook
│   ├── clean_local.sh           # Cleanup script for local cached evaluation runs
│   └── setup_local.sh           # Setup script for local analysis environment
├── notebooks/                   # Jupyter notebooks for data processing & paper figures
│   ├── 01_data_preprocessing.ipynb # Gauge filtering, ERA5-Land extraction, & attribute building
│   └── 02_paper_figures.ipynb      # Main manuscript figure generation
├── src/                         # Core Python package
│   ├── __init__.py
│   ├── climate.py               # ERA5-Land climate processing routines
│   ├── config.py                # Central experiment configuration & default paths
│   ├── data_ingestion.py        # HYDAT hydrometric data readers
│   ├── data_utils.py            # Normalization, sequence building, & masking utilities
│   ├── dataset.py               # PyTorch Dataset classes for sequence windows
│   ├── inference.py             # Factual and Counterfactual (g=0) prediction engine
│   ├── models.py                # Regional LSTM architecture definitions
│   ├── processing.py            # Data pipeline orchestration
│   ├── spatial_utils.py         # Geospatial intersections & polygon operations
│   └── training.py              # Loss functions (Masked NSE*) and training loop
├── .gitignore
├── bundle_project.py            # Utility to bundle project dependencies for HPC transfer
├── postprocessing_requirements.txt # Dependencies for local analysis & figure generation
├── requirements.txt             # Primary PyTorch training dependencies for HPC
├── README.md
├── run_training.py              # Main CLI entry point for training and evaluation
├── secrets.env                  # Email and allocation code for HPC setup
└── setup.py                     # Local package installation setup
```

---

## Model & Experiment Configurations
The `run_training.py` pipeline supports multiple architectures, loss functions, and input feature sets to facilitate broad experiment testing. The specific setup used in the paper was the combination of `lstm`, NSE*, and `topographic` input set.

1. **Model Architectures** (`<model_type>`):
* `lstm`: A standard Long Short-Term Memory network where static catchment attributes are concatenated with dynamic meteorological inputs at each timestep.
* `ealstm` An Entity-Aware LSTM where static catchment attributes are explicitly separated from dynamic inputs to modulate the input gate.

2. **Loss Functions**:
* NSE* (`BasinAveragedNSELoss`): A basin-averaged Nash-Sutcliffe Efficiency loss function. This normalizes the loss across catchments to prevent high-variance basins from dominating the optimization. (Set as default in `run_training.py`)
* MSE (`MaskedMSELoss`): A standard Mean Square Error loss function. (To use this, modify the `criterion` variable inside `run_training.py`)

3. **Input Feature Configurations** (`<exp_name>`)
* `baseline`:
  * Dynamic: `temp_max`, `temp_min`, `precip`
  * Static: `glacier_pct`
* `area`:
  * Dynamic: `temp_max`, `temp_min`, `precip`
  * Static: `basin_area_km2`, `glacier_pct`
* `topographic`:
  * Dynamic: `temp_max`, `temp_min`, `precip`
  * Static: `basin_area_km2`, `mean_elev`, `elev_range`, `mean_slope`, `glacier_pct`
* `phase-split`:
  * Dynamic: `temp_max`, `temp_min`, `rain`, `snow`
  * Static: `basin_area_km2`, `mean_elev`, `elev_range`, `mean_slope`, `glacier_pct`

---

## High Performance Compute Setup (UBC ARC Sockeye)
To train the EA-LSTM model, this project utilized UBC ARC Sockeye. The following steps can be followed to set up this project on Sockeye:

1. **Create a Secrets File**
   Create a file named `secrets.env` in the project root containing your email and Sockeye allocation code.
   ```text
   # secrets.env
   EMAIL="<your email>"
   ACCOUNT="<alloc-code>-gpu"
   ```
   Note: The -gpu suffix is required for running in a GPU environment.
2. **Bundle the Project**
   Run the bundle script, `python bundle_project.py`, in your terminal from the project root. This creates a zip file excluding the raw data.
3. **Upload to Sockeye**
   Run `scp project_upload.zip <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/` replacing `<cwl>` with your UBC CWL and `<alloc-code>` with your Sockeye allocation code.
   Note that to connect to Sockeye you must be connected to a UBC secure network or connect to [UBC myVPN](https://it.ubc.ca/services/email-voice-internet/myvpn/setup-documents)
4. **Connect and Extract**
   SSH into Sockeye and unzip the project.
   ```bash
   ssh <cwl>@sockeye.arc.ubc.ca
   cd /scratch/<alloc-code>
   unzip project_upload.zip -d lstm_glacier_counterfactuals
   cd lstm_glacier_counterfactuals
   ```
5. **Setup Environment (One-time)**
   This script loads the required Python modules, creates a virtual environment, and installs dependencies.
   ```bash
   chmod +x hpc/setup_env.sh
   ./hpc/setup_env.sh
   ```
   *Troubleshooting*: If you recieve an error, try running this first and then trying again:
   ```bash
   sed -i 's/\r$//' setup_env.sh
   sed -i 's/\r$//' submit.sh
   sed -i 's/\r$//' job.sh
   ```
6. **Submit the Job**
   The submit script automatically handles directory setup, secrets injection, and SLURM submission. You must provide the experiment name and model type as command-line arguments. 
   ```bash
   chmod +x hpc/submit.sh
   ./hpc/submit.sh <experiment_name> <model_type>
   ```
   (Example: `./hpc/submit.sh topographic lstm`) 

### Monitoring and Results
* **Check Status:** Run `squeue -u <cwl>` to see your job in the queue.
* **View Logs:** Once running, track progress live: `tail -f logs/train_*.out`
* **Retrieve Results:**
   Once the job status has changed to `COMPLETED`, you can download the trained model and the factual/counterfactual predictions to your local machine. Saved files can be found at the following locations, with `member_id` corresponding to each of the 10 ensemble runs (0-9):
   * Trained models: `models/<exp_name>_<model_type>_member_<member_id>.pth` (*Example:* `topographic_lstm_member_0.pth`)
   * Factual predictions: `data/output/<exp_name>_<model_type>_preds_member_<member_id>.csv` (*Example:* `topographic_lstm_preds_member_0.csv`)
   * Counterfactual predictions: `data/output/<exp_name>_<model_type>_preds_noglacier_member_<member_id>.csv` (*Example:* `topographic_lstm_preds_noglacier_member_0.csv`)
   The following code will download the necessary files:
   ```bash
   # download the factual and counterfactual predictions
   scp -r <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/lstm_glacier_counterfactuals/data/output/* ./data/output/
   ```
   ```bash
   # download saved model
   scp -r <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/lstm_glacier_counterfactuals/models/* ./models/
   ```

---

## Postprocessing & Figure Generation
After retrieving the factual and counterfactual predictions from the HPC cluster, use the local postprocessing environment to generate the manuscript figures.
1. **Set up the local environment:** Execute the local setup script and install the dependencies required for analysis and plotting.
```bash
bash local_postprocessing/setup_local.sh
pip install -r postprocessing_requirements.txt
```
2. **Run the figure notebook:**
Execute the code blocks in `notebooks/02_paper_figures.ipynb`. This notebook processes the downloaded prediction arrays and reproduces all figures and performance summary tables from the manuscript.

---

## Funding & Acknowledgements
* **Funding** This research was supported by a Canada Graduate Scholarship – Master's (CGS M) from the Natural Sciences and Engineering Research Council of Canada (NSERC) awarded to Tyler Wilson.
* **Compute** Computational resources and systems administration were provided by Advanced Research Computing at the University of British Columbia (UBC ARC Sockeye).