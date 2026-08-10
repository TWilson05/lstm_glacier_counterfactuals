# Learning Glacier Runoff from Streamflow: A Counterfactual Evaluation of Deep Hydrological Models

Official code repository and computational pipeline for:
> **Learning Glacier Runoff from Streamflow: A Counterfactual Evaluation of Deep Hydrological Models**  
> Tyler Wilson and Valentina Radić 
> *Department of Earth, Ocean and Atmospheric Sciences, University of British Columbia*  
> (Submitted 2026)  
> **Correspondence:** Tyler Wilson (twilson@eoas.ubc.ca)

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
Before training on the cluster, it is recommended for data to be downloaded and preprocessed locally.

1. **Install Dependencies:**
   Ensure you have Python 3.10+ installed, then run:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
2. **Run Preprocessing:**
   Run `notebooks/01_data_preprocessing.ipynb` in full.
   * Note: This notebook downloads ERA5 reanalysis data which can take a significant amount of time depending on the server queues.
   * Outcome: This generates the lightweight CSVs in `data/processed/` required for training.

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
│   ├── output/
│   │   ├── area/
│   │   ├── baseline/
│   │   ├── phase_split/
│   │   └── topographic/
│   ├── processed/
│   │   ├── climate/
│   │   │   ├── daily_fraction_below_zero.csv
│   │   │   ├── daily_precipitation.csv
│   │   │   ├── daily_rainfall.csv
│   │   │   ├── daily_snowfall.csv
│   │   │   ├── daily_temp_max.csv
│   │   │   └── daily_temp_min.csv
│   │   ├── combined_streamflow.csv
│   │   ├── glacier_volume_change_1.csv
│   │   ├── glacier_volume_change_2.csv
│   │   ├── glacier_volume_change_3.csv
│   │   └── static_attributes.csv
│   └── raw/
│       ├── dem_data/
│       ├── drainage_areas/
│       ├── era5/
│       │   ├── precipitation/
│       │   └── temperature/
│       ├── mass_balance/
│       │   ├── ts_monthly_const_area_fnn.csv
│       │   ├── ts_monthly_const_area_fnn_cluster.csv
│       │   └── ts_monthly_const_area_lstm.csv
│       ├── RGI-western-canada/
│       ├── spatial_bounds.csv
│       └── station_metadata.csv
├── hpc/                         # HPC (UBC Sockeye) submission & environment scripts
│   ├── job.sh                   # Main SLURM execution script
│   ├── setup_env.sh             # Conda environment setup for PyTorch/CUDA
│   └── submit.sh                # Job submission wrapper
├── models/
├── local_postprocessing/        # Local evaluation & SOM clustering utilities
│   ├── clean_local.sh           # Cleanup script for local cached evaluation runs
│   └── setup_local.sh           # Setup script for local analysis environment
├── notebooks/                   # Jupyter notebooks for data processing & paper figures
│   ├── 01_data_preprocessing.ipynb # Gauge filtering, ERA5-Land extraction, & attribute building
│   └── 02_paper_figures.ipynb      # Main manuscript figure generation (Figs 1-7)
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
├── secrets.env
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

3. **Input Feature Configurations** (`<experiment_name>`)
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
   ```
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
   ```
   ssh <cwl>@sockeye.arc.ubc.ca
   cd /scratch/<alloc-code>
   unzip project_upload.zip -d lstm_glacier_counterfactuals
   cd lstm_glacier_counterfactuals
   ```
5. **Setup Environment (One-time)**
   This script loads the required Python modules, creates a virtual environment, and installs dependencies.
   ```
   chmod +x hpc/setup_env.sh
   ./hpc/setup_env.sh
   ```
   *Troubleshooting*: If you recieve an error, try running this first and then trying again:
   ```
   sed -i 's/\r$//' setup_env.sh
   sed -i 's/\r$//' submit.sh
   sed -i 's/\r$//' job.sh
   ```
6. **Submit the Job**
   The submit script automatically handles directory setup, secrets injection, and SLURM submission. You must provide the experiment name and model type as command-line arguments. 
   ```
   chmod +x hpc/submit.sh
   ./hpc/submit.sh
   ```

### Monitoring and Results
* **Check Status:** Run `squeue -u <cwl>` to see your job in the queue.
* **View Logs:** Once running, track progress live: `tail -f logs/train_*.out`
* **Retrieve Results:**
   Once the job status has changed to `COMPLETED`, you can download the trained model and the factual/counterfactual predictions to your local machine.
   The following code will download the necessary files:
   ```bash
   # download the factual and counterfactual predictions
   scp -r <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/lstm_glacier_counterfactuals/data/output/test_set_predictions.csv ./data/output/
   ```
   ```bash
   # download saved model
   scp -r <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/lstm_glacier_counterfactuals/models/ ./models/
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