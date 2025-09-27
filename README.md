# Early Prediction of Cerebral Vasospasm After Aneurysmal Subarachnoid Hemorrhage

This repository contains the dataset and source code accompanying the manuscript:

**Early Prediction of Cerebral Vasospasm After Aneurysmal SAH Using a Machine Learning Model and Interactive Web Application**

## Overview

Cerebral vasospasm is one of the most severe complications following aneurysmal subarachnoid hemorrhage (aSAH). Early prediction is critical for timely interventions and improved outcomes.
In this work, we developed and validated several machine learning models to predict vasospasm using routinely collected clinical and radiological variables.

Our best-performing model was **logistic regression**, which provided the most favorable trade-off between accuracy, sensitivity, specificity, and interpretability.

An interactive **web application** was implemented with Gradio and deployed on HuggingFace. The deployed model was trained on the **entire dataset** for maximum predictive accuracy in real-time use.

Both the dataset and source code are openly available to support transparency and reproducibility.

## Dataset

The published dataset is provided in CSV format:

[`data/dataset_vasospasm.csv`](data/dataset_vasospasm.csv)

### Features

| Feature              | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `id`                 | Patient identifier (ascending integers                  )                   |
| `age_diagnose`       | Age at diagnosis (in years)                                                 |
| `sex`                | Patient sex (`male` / `female`)                                             |
| `hunt_hess`          | Hunt and Hess grade at admission (clinical severity scale)                  |
| `fisher`             | Fisher scale (radiological blood load grading)                              |
| `cns_infection`      | Presence of central nervous system infection (`0` = no, `1` = yes)          |
| `EVD`                | External ventricular drainage placement (`0` = no, `1` = yes)               |
| `aneurysm`           | Presence of aneurysm (`0` = no, `1` = yes)                                  |
| `aneurysm_location`  | Anatomical location of the aneurysm (e.g., ACom, ACA, MCA)                  |
| `aneurysm_diameter`  | Maximum aneurysm diameter (mm)                                              |
| `aneurysm_height`    | Aneurysm dome height (mm)                                                   |
| `aneurysm_treatment` | Treatment modality (e.g., `coil`, `clip`)                                   |
| `vasospasm`          | Binary target variable: occurrence of vasospasm (`0` = no, `1` = yes)       |
| `vasospasm_severity` | Severity of vasospasm (`moderate`, `severe`, or empty if none) – *not used* |

## Local Modelling (Python)

1. **Clone this repository**

   ```bash
   git clone https://github.com/RISCSoftware/vasospasm-prediction.git
   cd vasospasm-prediction
   ```

2. **Set up the environment**
   You can use either **Poetry** (recommended) or **pip**.

   * **Poetry (reproducible)**
     First install [Poetry](https://python-poetry.org/docs/#installation) if it’s not already available. Then run:

     ```bash
     poetry install
     poetry env activate
     ```

     This creates and activates a virtual environment with exact dependency versions from the [`poetry.lock`](poetry.lock) file.

   * **Pip (lightweight)**
     First create and activate a virtual environment, e.g.
     ```
     python -m venv .venv && source .venv/bin/activate
     ```
     Then run:

     ```bash
     pip install -r requirements.txt
     ```

     This installs dependencies directly; quicker but versions may vary slightly.

3. **Run analyses**
   The main entry point for training and evaluation is:
   [`vasospasm/main.py`](vasospasm/main.py)

   This script trains the classifiers using **nested cross-validation** and saves results to [`output/vasospasm/current/`](output/vasospasm/current/):

   * [`report.csv`](output/vasospasm/current/report.csv): Performance metrics for predictive models for vasospasm.
   * [`feature_importance.json`](output/vasospasm/current/feature_importance.json): Feature importance, coefficients, and odds ratios for logistic regression.

## R Environment & Statistical Tables

Generate the baseline characteristics table (with statistical tests) from the paper using **R + renv** and **Quarto**.

**Prerequisites**

* R (≥ 4.2 recommended)
* [Quarto CLI](https://quarto.org/docs/get-started/)
* From the repo root, the dataset is at [`data/dataset_vasospasm.csv`](data/dataset_vasospasm.csv)

**1) Create/restore the R environment**
Run the setup script once from the repository root:

```bash
Rscript renv-setup.R
```

This installs/activates **renv**, restores packages from [`renv.lock`](renv.lock) if present, or installs a minimal set and snapshots to create [`renv.lock`](renv.lock).

**2) Render the statistics table**
Render the Quarto document that computes summary statistics and p-values:

```bash
quarto render vasospasm/statistics_table.qmd
```

To restrict the analysis to aneurysmal SAH only, set `only_with_aneurysm <- TRUE` near the top of [`vasospasm/statistics_table.qmd`](vasospasm/statistics_table.qmd) before rendering.

**Outputs**

* An HTML render of the document (default Quarto output).
* A Word table is written by the script to:

  * [`output/table_1_all.docx`](output/table_1_all.docx) (default, all patients)
  * or [`output/table_1.docx`](output/table_1.docx) if the document is configured to only include aneurysmal cases.

## Web Application

For interactive use, access the deployed HuggingFace space:
🔗 [Web App on HuggingFace Spaces](https://huggingface.co/spaces/risc42/vasospasm-prediction)

You can also run the Gradio web app locally by running [vasospasm/web/gradio_app.py](vasospasm/web/gradio_app.py).

## License

- The source code of this project is licensed under the [MIT License](LICENSE).
- The dataset provided in [data/dataset_vasospasm.csv](data/dataset_vasospasm.csv) is licensed under the Creative Commons Attribution 4.0 International License. See the [data/LICENSE-DATA.txt](data/LICENSE-DATA.txt) file for more details.

## Citation

If you use this dataset or code, please cite the following paper:

> Gollwitzer M, Mazanec V, Steindl M, Atli B, Stroh N, Hauser A, Sardi G, Rossmann T, Aspalter S, Rauch P, Horner E, Sonnberger M, Gruber A, Gmeiner M. *Early Prediction of Cerebral Vasospasm After Aneurysmal SAH Using a Machine Learning Model and Interactive Web Application*. Brain Sci. 2025.
