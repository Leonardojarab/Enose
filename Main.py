# -*- coding: utf-8 -*-
"""
Main script for COPD/Smokers/Control classification using
Electronic Nose signals, PCA and XGBoost.
Author: Leonardo Jara
Created on Sat Aug 16 08:55:42 2025
"""

#Import custom functions from src/
from src.format import f_format
from src.signal import f_signalcheck, f_signal
from src.pca import f_pca, pca_elbow, plot_pca_scatter, f_pcagraf
from src.split import fsplit
from src.xgboost_func import xgb_grid_mc
from src.metrics_func import metrics_multiclass

#Standard libraries
import numpy as np, pandas as pd
from sklearn.preprocessing import  LabelEncoder

# 1) Load datasets
control_path = "data/CONTROL.csv"
copd_path = "data/COPD.csv"
smokers_path = "data/SMOKERS.csv"
general_path = "data/General_data_from_the_dataset.csv"

control = pd.read_csv(control_path, sep=",")
copd = pd.read_csv(copd_path, sep=",")
smokers = pd.read_csv(smokers_path, sep=",")
general = pd.read_csv(general_path, sep=",", skiprows=1)


# 2) Data formatting & correlation check
df = f_format(control, copd, smokers, general)

# Check correlation between repeated sensor signals
f_signalcheck(control, copd, smokers, df)

# Visualize signal of one patient (example: D01)
f_signal(df, "D01")




# 3) PCA analysis
y = df["target"]
X = df.loc[:, "t0":].astype(np.float32)

# Find optimal number of components for 95% variance
k = pca_elbow(X, th=0.95, estandarizar=True)

# Apply PCA
pca_model, pca_scores = f_pca(X, n=k, estandarizar=True)

# Scatter plot of PCA (COMP1 vs COMP2)
plot_pca_scatter(pca_scores, y, comp1=0, comp2=1)


# 4) XGBoost multiclass classification
le = LabelEncoder()
y_enc = le.fit_transform(y)  # [0..K-1]
classlabel = dict(enumerate(le.classes_))
n_classes = len(le.classes_)
print("Classes:", le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = fsplit(pca_scores, y_enc, cut_size=0.30, seed=34)

# Train XGBoost with hyperparameter grid search
y_pred, bestxgb, y_proba = xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=n_classes,
                                       kn=3)

# Evaluate metrics
metrics_multiclass(y_test, y_pred, y_proba, n_classes=n_classes, target_names=le.classes_)


# 5) Extended experiment: Merge metadata (sex, age) + PCA features
merge =pd.concat([df[["sex", "age"]], pca_scores], axis=1)

# Encode categorical variable 'sex'
lemerge = LabelEncoder()
merge["sex"] = lemerge.fit_transform(merge["sex"])

# Train-test split with extended dataset
X_train, X_test, y_train, y_test = fsplit(merge, y_enc, cut_size=0.30, seed=34)

# Train XGBoost again
y_pred, bestxgb, y_proba = xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=n_classes,
                                       kn=3)

# Evaluate metrics
metrics_multiclass(y_test, y_pred, y_proba, n_classes=n_classes, target_names=le.classes_)



