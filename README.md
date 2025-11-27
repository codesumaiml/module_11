# What Drives the Price of a Car?

This repository contains a Jupyter notebook (`prompt_II.ipynb`) that explores a used-vehicle dataset (approx. 426K rows) to identify the factors that influence resale price and to build simple regression models to predict price.

**Purpose**
- Provide a clear, reproducible analysis that helps used-car dealerships understand what drives resale value.
- Produce a predictive baseline (linear, Lasso, Ridge) and surface the top features that affect price.

**Files of interest**
- `prompt_II.ipynb` — analysis notebook (data loading, exploration, cleaning, modeling, evaluation, visualizations).
- `data/vehicles.csv` — raw dataset used by the notebook.
- `images/` — images referenced in the notebook (figures/illustrations).

Dependencies
Run the following to create a Python environment and install minimal dependencies used by the notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Running the notebook
1. Activate the environment (see commands above).
2. Start Jupyter in the repository root and open the notebook:

```bash
jupyter notebook prompt_II.ipynb
```

Notebook overview (high-level)
- Data loading and initial inspection: reads `data/vehicles.csv`, prints shape, describes columns, checks missing values and duplicates.
- Data understanding: visualizes distributions (price, year), categorical counts, and correlations.
- Data preparation: sampling for speed, dropping/transforming fields (e.g., `id`, `VIN`, region), converting ordinal fields (condition), extracting numeric cylinder counts, imputing missing values, and one-hot encoding categorical variables.
- Modeling: scales numeric features, splits train/test, fits Linear, Lasso, and Ridge regressions, computes MSE/RMSE, and uses `GridSearchCV` to tune Ridge `alpha`.
- Feature importance: prints & plots top features from the Ridge model coefficients.

Key findings (from the notebook)
- Ridge Regression was reported as the best-performing model (lower RMSE/MSE than Linear and Lasso on the sampled/processed data).
- Features strongly influencing price included `year`, `odometer` (mileage), `manufacturer`, `model`, `condition`, and `title_status`.

Notes & caveats
- The notebook samples the dataset (20k rows) for modeling; results depend on sampling and preprocessing choices.
- Many VIN duplicates and missing values exist — these should be handled carefully in a production workflow.
- Some columns (e.g., `region`) were dropped because of high cardinality; depending on business scope, region-level pricing adjustments may be valuable.

Suggested next steps
- Improve data cleaning: remove or consolidate duplicate listings, verify zero/invalid prices, and more thoughtfully impute missing values.
- Feature engineering: add age (current year - year), interactions (manufacturer × model), demand proxies, and location features if available.
- Try stronger models: tree-based models (RandomForest, XGBoost, LightGBM) or ensemble stacking to improve predictive performance.
- Evaluate with more robust CV and metrics: use cross-validated scores, stratify sampling by manufacturer/model where appropriate, and report error percentiles.


