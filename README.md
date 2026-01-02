# Project structure

rcfst_keml_project/
│
├─ data/
│   └─ rcfst_database.csv           # your 447-specimen database
│
├─ src/
│   ├─ config.py
│   ├─ data_utils.py
│   ├─ features.py
│   ├─ keml.py
│   ├─ models.py
│   └─ evaluation.py
│
├─ artifacts/
│   ├─ models/
│   └─ optuna/
│
├─ outputs/
│   ├─ figures/
│   ├─ metrics/
│   └─ predictions/
│
└─ notebooks/
    └─ keml_rcfst_framework.ipynb   # main Jupyter notebook (code below as #%% cells)
