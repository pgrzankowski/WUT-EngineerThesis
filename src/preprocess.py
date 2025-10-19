import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from typing import Tuple


CFG = {
    "csv_path": "../data/raw/sbdb_query_results_final.csv",
    "target": "diameter",
    "random_state": 42,
    "test_size": 0.15,
    "val_size": 0.15,
    "use_iterative_imputer": True,
}

na_vals = ["", "NA", "N/A", "na", "null", "None", "-", "?"]

print("Loading dataset from csv")
df = pd.read_csv(CFG["csv_path"], na_values=na_vals, low_memory=False)

print("Preprocessing...")
df = df.drop_duplicates()
df = df.dropna(subset=[CFG["target"]])
df = df[["diameter", "H", "a", "moid", "data_arc", "n_obs_used", "rms"]]

train_df, test_df = train_test_split(
    df, test_size=CFG["test_size"], random_state=CFG["random_state"]
)

y_train = train_df[CFG["target"]]
X_train = train_df.drop(columns=[CFG["target"]])
y_test = test_df[CFG["target"]]
X_test = test_df.drop(columns=[CFG["target"]])

# Map binary Y/N only if such columns exist; don't touch purely numeric columns
for bin_col in ["neo", "pha"]:
    if bin_col in X_train.columns:
        X_train[bin_col] = X_train[bin_col].map({"Y": 1, "N": 0})
    if bin_col in X_test.columns:
        X_test[bin_col] = X_test[bin_col].map({"Y": 1, "N": 0})
# Determine column types explicitly to make ColumnTransformer robust
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_steps = []
if CFG["use_iterative_imputer"]:
    numeric_steps.append(("impute", IterativeImputer(random_state=CFG["random_state"])))
else:
    numeric_steps.append(("impute", SimpleImputer(strategy="median")))

numeric_steps += [
    ("power", PowerTransformer(method="yeo-johnson", standardize=True))
]

num_pipe = Pipeline(numeric_steps)

# TODO: Remove this pipe since categorical features
#       don't have missing values
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)),
])

# Build the ColumnTransformer only with non-empty selectors to avoid 0-feature branches
transformers = []
if len(numeric_cols) > 0:
    transformers.append(("num", num_pipe, numeric_cols))
if len(categorical_cols) > 0:
    transformers.append(("cat", cat_pipe, categorical_cols))

pre = ColumnTransformer(transformers, remainder="drop")

models = {
    "KNN": KNeighborsRegressor(),
    "RandomForest": RandomForestRegressor()
}

cv = KFold(n_splits=5, shuffle=True, random_state=CFG["random_state"])
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",    
}

print("Cross validation")
results = {}
for name, est in models.items():
    pipe = Pipeline([("pre", pre), ("model", est)])
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False)
    results[name] = {k: np.mean(v) for k, v in cv_res.items() if k.startswith("test_")}

print("Results:")
for name, scores in results.items():
    print(name, ':')
    for score_name, score in scores.items():
        print(f'\t{score_name}: {score:.2f}')