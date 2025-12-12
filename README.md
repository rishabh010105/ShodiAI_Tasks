# Policy Optimization for Financial Decision-Making (LendingClub)

Short instructions to reproduce EDA, supervised model, and offline RL results.

---

## 🔧 Prerequisites

- Python **3.9–3.11** recommended  
- **10–20 GB** free disk (dataset + memmaps)  
- Git  
- (Optional) Conda  

---

## 1. Clone Repo & Place Data

```bash
git clone <your-repo-url>
cd lendingclub-policy-optimization

mkdir -p data models
Download dataset from Kaggle:
https://www.kaggle.com/datasets/wordsforthewise/lending-club
```
Place the file here:

bash
Copy code
data/accepted_2007_to_2018Q4.csv.gz
2. Create Environment & Install Packages
Option A — Conda (recommended)
bash
Copy code
conda create -n lendingclub-env python=3.10 -y
conda activate lendingclub-env
pip install -r requirements.txt
CPU-only PyTorch (optional):

bash
Copy code
pip install torch --index-url https://download.pytorch.org/whl/cpu
Option B — venv / pip
bash
Copy code
python -m venv lendingclub-env
# Windows
.\lendingclub-env\Scripts\activate
# macOS/Linux
source lendingclub-env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
Minimum requirements.txt
nginx
Copy code
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
torch
d3rlpy
jupyterlab
Note: The notebooks include compatibility code for multiple d3rlpy versions.
If you get CQL API errors, run the diagnostic cell in the Task 3 notebook.

3. Run the Notebooks (order matters)
Start JupyterLab:

bash
Copy code
jupyter lab
Then run each notebook top-to-bottom:

✔ Task 1
notebooks/01_task1_EDA_preprocessing.ipynb
Cleans data and saves:

data/df_encoded.joblib

data/X_preprocessed.joblib

data/y_preprocessed.joblib

✔ Task 2
notebooks/02_task2_supervised_training.ipynb
Trains MLP and saves:

models/best_mlp.pth

models/scaler.joblib

task2 metrics

✔ Task 3
notebooks/03_task3_offline_rl_cql.ipynb
Builds MDPDataset and trains offline RL (CQL).

Two scaling modes:

Option A (quick) — sample-based scaler (low RAM)

Option B (robust) — partial_fit + memmap (no memory spike)

Saves:

RL policy inside models/cql_policy/

models/policy_values_summary.joblib

If memory errors occur → use Option A or reduce batch size.

✔ Task 4
notebooks/04_task4_analysis_and_report.ipynb
Loads all artifacts and produces:

models/task4_analysis.md

models/figures/

disagreement CSVs

4. Quick Commands (Non-Interactive Execution)
Install papermill:

bash
Copy code
pip install papermill
Run all notebooks automatically:

bash
Copy code
papermill notebooks/01_task1_EDA_preprocessing.ipynb outputs/01_out.ipynb
papermill notebooks/02_task2_supervised_training.ipynb outputs/02_out.ipynb
papermill notebooks/03_task3_offline_rl_cql.ipynb outputs/03_out.ipynb
5. Outputs & Results Location
Models
models/best_mlp.pth

models/cql_policy/

models/scaler.joblib

Saved Arrays / Memmaps
data/X_*_scaled.dat

Analysis & Figures
models/task4_analysis.md

models/figures/

models/rl_vs_supervised_comparison.joblib

models/rl_supervised_disagreements_sample.csv

Numeric Summaries
models/task4_summary.joblib

models/policy_values_summary.joblib

6. Reproducibility Tips & Troubleshooting
If kernel restarts → rerun earlier notebook cells or reload .joblib files.

If d3rlpy constructor/fit errors occur → run the diagnostic cell in Task 3.

If scaling triggers MemoryError →

use sample-based scaling (Option A)

or reduce batch sizes in Option B

Always use RND = 42 for reproducible splits.

