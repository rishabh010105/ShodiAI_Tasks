```markdown
# 📘 Policy Optimization for Financial Decision-Making (LendingClub)

This repository contains an end-to-end pipeline for:  
✔ Exploratory Data Analysis (EDA)  
✔ A supervised Deep Learning model for loan default prediction  
✔ An Offline Reinforcement Learning (CQL) policy for loan approval decisions  
✔ A full analytical comparison and recommendations report (Task 4)

The goal is to **maximize expected financial return** using the LendingClub accepted loan dataset.

---

## 🚀 1. Project Structure
```

lendingclub-policy-optimization/
├─ data/                         # dataset + preprocessed files
├─ models/                       # trained models + scalers + RL policy
├─ notebooks/                    # Jupyter notebooks for Tasks 1–4
├─ requirements.txt
└─ README.md

```

---

## 📥 2. Dataset Download

Download the LendingClub dataset from Kaggle:

🔗 **https://www.kaggle.com/datasets/wordsforthewise/lending-club**

Place the file inside `data/`:

```

accepted_2007_to_2018Q4.csv.gz

````

(This file is NOT included in the repo due to size.)

---

## 🛠 3. Environment Setup

### ✔ Option A — Conda (Recommended)

```bash
conda create -n lendingclub-env python=3.10 -y
conda activate lendingclub-env
pip install -r requirements.txt
````

### ✔ Option B — venv / pip

```bash
python -m venv lendingclub-env
# Windows
.\lendingclub-env\Scripts\activate
# macOS/Linux
source lendingclub-env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### requirements.txt (included)

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
torch
d3rlpy
jupyterlab
```

> The notebooks include compatibility code to handle different `d3rlpy` versions.

---

## ▶️ 4. Running the Project (Order Matters)

Start JupyterLab:

```bash
jupyter lab
```

Then run notebooks in this exact order:

---

### 📌 4.1 Task 1 — EDA & Preprocessing

Notebook: `notebooks/01_task1_EDA_preprocessing.ipynb`

This notebook:

* Loads raw CSV
* Performs EDA
* Cleans data
* Encodes categorical variables
* Saves:

Outputs (saved to `data/`):

* `df_encoded.joblib`
* `X_preprocessed.joblib`
* `y_preprocessed.joblib`

---

### 📌 4.2 Task 2 — Supervised Deep Learning Model

Notebook: `notebooks/02_task2_supervised_training.ipynb`

This notebook:

* Trains MLP classifier
* Computes AUC & F1
* Saves trained models

Outputs (saved to `models/`):

* `best_mlp.pth`
* `final_mlp.pth`
* `scaler.joblib`

---

### 📌 4.3 Task 3 — Offline RL (CQL Policy Learning)

Notebook: `notebooks/task3_offline_rl_&_TASK4.ipynb`

This notebook:

* Builds RL dataset
* Computes loan rewards
* Trains Conservative Q-Learning (CQL)
* Includes fallback compatibility code for older `d3rlpy` versions

Outputs (saved to `models/`):

* `cql_policy/`
* `policy_values_summary.joblib`

---

### 📌 4.4 Task 4 — Analysis & Final Report

Notebook: `notebooks/task3_offline_rl_&_TASK4.ipynb`

This notebook:

* Evaluates supervised vs RL policies
* Computes disagreement cases
* Produces final analysis & summary report

Outputs (saved to `models/`):

* `task4_analysis.md`
* Task 4 figures + CSVs
* Summary joblib files

---

## 🧪 5. Expected Results (Reference)

| Model                              | Value    |
| ---------------------------------- | -------- |
| **Supervised Model AUC**           | ~0.717   |
| **Supervised F1 (best threshold)** | ~0.434   |
| **Supervised Policy Value**        | −1395.72 |
| **RL Policy Value (CQL)**          | −1604.37 |
| **Approve-All Baseline**           | −1651.94 |
| **Deny-All Baseline**              | 0        |

---

## 🆘 6. Troubleshooting

* **Cannot find CSV** → Ensure `accepted_2007_to_2018Q4.csv.gz` is in `data/`.
* **d3rlpy errors** → Run compatibility cell in Task 3.
* **MemoryError** → Use sample-based scaler or reduce batch size.
* **Missing test_preds** → Re-run Task 2 evaluation cells.


