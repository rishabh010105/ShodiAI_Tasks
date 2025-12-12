
Policy Optimization for Financial Decision-Making (LendingClub)
Short instructions to reproduce EDA, supervised model, and offline RL results.

Prerequisites

Python 3.9–3.11 recommended

~10–20 GB free disk (dataset and memmaps)

Git, (optional) conda

1. Clone repo & place data
git clone <your-repo-url>
cd lendingclub-policy-optimization
mkdir -p data models
# Download the dataset from Kaggle:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Place accepted_2007_to_2018Q4.csv.gz into ./data/

2. Create environment & install packages
Option A — conda (recommended)
conda create -n lendingclub-env python=3.10 -y
conda activate lendingclub-env
pip install -r requirements.txt
# If you need CPU-only PyTorch:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

Option B — venv / pip
python -m venv lendingclub-env
# Windows: .\lendingclub-env\Scripts\activate
# macOS/Linux: source lendingclub-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


requirements.txt (minimum):

numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
torch
d3rlpy
jupyterlab


Note: The notebooks include compatibility code for different d3rlpy versions. If you get API errors for CQL, run the diagnostic cell in Task 3 notebook.

3. Run the notebooks (order matters)

Open JupyterLab:

jupyter lab


Then run these notebooks top-to-bottom (execute all cells in each):

notebooks/01_task1_EDA_preprocessing.ipynb

Cleans data, does EDA, saves data/df_encoded.joblib and data/X_preprocessed.joblib, data/y_preprocessed.joblib.

notebooks/02_task2_supervised_training.ipynb

Trains MLP, saves models/best_mlp.pth, models/scaler.joblib, and task2 metrics.

notebooks/03_task3_offline_rl_cql.ipynb

Builds MDPDataset, trains offline RL (CQL).

NOTE: Two scaling options are provided:

Option A (quick) — sample-based scaler (low RAM)

Option B (robust) — partial_fit + memmap (no big-memory spike)

After training it saves RL policy and models/policy_values_summary.joblib.

If you hit memory errors, use Option A or lower batch sizes.

notebooks/04_task4_analysis_and_report.ipynb

Loads artifacts and produces Task 4 analysis, figures, models/task4_analysis.md, and disagreement CSVs.

4. Quick commands (non-interactive)

If you prefer to run notebooks non-interactively (optional), use papermill:

pip install papermill
papermill notebooks/01_task1_EDA_preprocessing.ipynb outputs/01_out.ipynb
papermill notebooks/02_task2_supervised_training.ipynb outputs/02_out.ipynb
papermill notebooks/03_task3_offline_rl_cql.ipynb outputs/03_out.ipynb


5. Outputs & where to find results

Models: models/best_mlp.pth, models/cql_policy/

Scaler: models/scaler.joblib

Saved arrays / memmaps (if used): data/X_*_scaled.dat

Analysis & figures: models/task4_analysis.md, models/figures/, models/rl_vs_supervised_comparison.joblib, models/rl_supervised_disagreements_sample.csv

Numeric summary: models/task4_summary.joblib, models/policy_values_summary.joblib

6. Repro tips & troubleshooting

If you restart kernel, re-run earlier notebook cells (or reload .joblib artifacts from data/ and models/).

If d3rlpy constructor/fit errors occur, run the diagnostic cell included in Task 3 to adapt to your installed version.

If scaling raises MemoryError, switch to Option A (sample-based) or use Option B with smaller batch sizes.

Use RND = 42 seed in notebooks for reproducible splits.
