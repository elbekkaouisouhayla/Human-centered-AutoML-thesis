# Trade-offs Between Automation and User Control in AutoML Systems for Expert and Non-Expert Users

**Author:** Souhayla El Bekkaoui  
**Degree:** Master of Specialization in Big Data  
**Institution:** Université Libre de Bruxelles (ULB)

---

## Overview

This repository contains the source code, experiments, and supporting materials for my Master's thesis entitled:

> **Trade-offs Between Automation and User Control in AutoML Systems for Expert and Non-Expert Users**

The objective of this research is to compare modern Automated Machine Learning (AutoML) frameworks from both a **technical** and **human-centered** perspective.

Rather than focusing solely on predictive performance, this study investigates the trade-offs between:

- Automation
- User control
- Usability
- Transparency
- Explainability
- Computational efficiency

The experiments are designed to evaluate how different AutoML frameworks support users with different levels of machine learning expertise.

---

## AutoML Frameworks

The following AutoML frameworks are evaluated:

- **PyCaret**
- **TPOT**
- **H2O AutoML**

---

## Datasets

The experiments are conducted using three publicly available classification datasets:

- **Breast Cancer Wisconsin Diagnostic Dataset**
  - Binary classification
  - Numerical features

- **Wine Recognition Dataset**
  - Multiclass classification
  - Numerical features

- **Titanic Dataset**
  - Binary classification
  - Mixed numerical and categorical features with missing values

---

## Experimental Design

Each framework is evaluated under identical experimental conditions.

### Configuration

- 3 datasets
- 3 AutoML frameworks
- 3 random seeds (42, 123, 2026)
- Stratified 80/20 train-test split
- 5-fold cross-validation
- 10-minute runtime limit per AutoML run
- CPU execution only (no GPU)

This results in:

**3 × 3 × 3 = 27 main AutoML experiments**

---

## Evaluation Criteria

### Quantitative Evaluation

- Macro F1-score
- Accuracy
- Macro Precision
- Macro Recall
- ROC-AUC
- Runtime
- Selected model
- Number of evaluated models/pipelines (when available)

### Qualitative Evaluation

The frameworks are also compared according to:

- Degree of automation
- Usability
- Transparency & explainability
- User control & configurability

---

## Repository Structure

```text
├── Datasets/
│
├── Notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_data_preparation.ipynb
│   ├── 02_preprocessing_function.ipynb
│   ├── 03_pycaret_experiments.ipynb
│   ├── 04_tpot_experiments.ipynb
│   ├── 05_h2o_automl_experiments.ipynb
│   └── 06_results_analysis.ipynb
│
├── Results/
│   ├── master_results.xlsx
│   ├── figures/
│   └── tables/
│
├── Screenshots/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Development Environment

- Python 3.11
- Visual Studio Code
- Jupyter Notebook
- scikit-learn
- PyCaret
- TPOT
- H2O AutoML
- SHAP
- Pandas
- NumPy
- Matplotlib

---

## Current Status

-  Repository created
-  Project structure completed
-  Development environment configured
-  Data preprocessing implemented
-  AutoML experiments in progress
-  Comparative analysis in progress

---

## Thesis Information

**Author:** Souhayla El Bekkaoui

**Program:** Master of Specialization in Big Data

**Institution:** Université Libre de Bruxelles (ULB)

**Academic Year:** 2025–2026
