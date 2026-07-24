# Quantathon — Water Potability Classification

This project addresses **Challenge 2 of Quantathon CR 2026**: predicting whether a water sample is potable from physicochemical measurements. It supports the United Nations' **Sustainable Development Goal 6 (Clean Water and Sanitation)**.

The challenge compares a classical support vector machine (SVM) with a quantum support vector machine (QSVM) based on quantum kernels. The repository currently contains the dataset, the official challenge brief, and an initial classical SVM baseline.

## Dataset

The Water Potability dataset contains **3,276 samples**, nine input features, and a binary target:

| Column | Description |
| --- | --- |
| `ph` | Acidity or alkalinity of the water |
| `Hardness` | Mineral hardness |
| `Solids` | Total dissolved solids |
| `Chloramines` | Chloramine concentration |
| `Sulfate` | Sulfate concentration |
| `Conductivity` | Electrical conductivity |
| `Organic_carbon` | Organic carbon concentration |
| `Trihalomethanes` | Trihalomethane concentration |
| `Turbidity` | Water clarity |
| `Potability` | Target: `0` = non-potable, `1` = potable |

The raw data includes missing values in `ph`, `Sulfate`, and `Trihalomethanes`. The supplied baseline notebook removes incomplete rows; the challenge brief recommends median imputation by class for the final comparison.

## Classical Machine Learning Pipeline

The implemented pipeline follows the Challenge methodology:

- Exploratory Data Analysis (EDA)
- Stratified 80/20 train-test split
- Median imputation by class for missing values
- Feature standardization (StandardScaler)
- Class balancing with SMOTE
- Data leakage validation
- SVM with RBF kernel
- Hyperparameter tuning using **GridSearchCV (5-fold cross-validation)**

## Project structure

```text
Quantathon/
│
├── data/
│   ├── water_potability.csv
│   ├── kernel/
│   │   ├── n_24_dim_2_z_feature_map.csv
│   │   └── ...
│   ├── kernel_h2/
│   ├── test/
│   └── test_h2/
│
├── docs/
│   ├── challenge-2-water-potability.pdf
│   └── challenge-2-water-potability.docx
│
├── img/
│   ├── kernel heatmaps
│   ├── circuit figures
│   ├── evaluation plots
│   └── experimental results
│
├── notebooks/
│   ├── SVM_RBF.ipynb
│   ├── qsvm.ipynb
│   ├── qsvm nexus.ipynb
│   ├── analisis_experimental_qsvm.ipynb
│   └── heatmaps_kernel_vs_h2.ipynb
│
├── output/
│   └── pdf/
│       └── informe_tecnico_quantathon.pdf
│
├── reports/
│   ├── generate_figures.py
│   ├── informe_tecnico_quantathon.tex
│   └── references.bib
│
├── README.md
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Quantum-Odyssey/Quantathon.git
cd Quantathon
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with `.venv\Scripts\activate`.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the notebooks

```bash
# Classical baseline
jupyter notebook notebooks/SVM_RBF.ipynb

# Quantum experiments
jupyter notebook notebooks/analisis_experimental_qsvm.ipynb
```

Run the cells from top to bottom. The notebook expects the dataset at `data/water_potability.csv`.

## Challenge roadmap

- Improve preprocessing with median imputation, stratification, and class balancing.
- Tune the classical RBF SVM using five-fold cross-validation over the required `C` and `gamma` grid.
- Report accuracy, precision, recall, F1 score, balanced accuracy, and a confusion matrix.
- Select a balanced subset of 16–64 training samples for the quantum experiment.
- Implement and compare quantum feature maps and precomputed quantum kernels.
- Compare the SVM and QSVM on the same held-out test set.
- Analyze kernel alignment, intra/inter-class similarity, eigenvalue spectrum, circuit depth, noise sensitivity, and computational cost.
- Document limitations honestly; demonstrating quantum advantage is not required.

## Quantum stack

The Quantum Stack utilized was **qiskit**, as proviciency with its capabilities was adquired during the bootcamp.

## Source

Dataset: [Water Quality — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## License

No project license has been selected yet. Review the dataset's usage terms before redistribution or publication.
