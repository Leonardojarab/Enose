# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 08:55:42 2025

@author: leoja
"""

from src.format import f_format
from src.signal import f_signalcheck, f_signal
from src.pca import f_pca, pca_elbow, plot_pca_scatter, f_pcagraf
from src.split import fsplit
from src.xgboost_func import xgb_grid_mc
from src.metrics_func import metrics_multiclass

import os, re, numpy as np, pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize


control_path = "data/CONTROL.csv"
control = pd.read_csv(control_path, sep=",")
copd_path = "data/COPD.csv"
copd = pd.read_csv(copd_path, sep=",")
smokers_path = "data/SMOKERS.csv"
smokers = pd.read_csv(smokers_path, sep=",")
general_path = "data/General_data_from_the_dataset.csv"
general = pd.read_csv(general_path, sep=",", skiprows=1)








df = f_format(control, copd, smokers, general)
f_signalcheck(control, copd, smokers, df)




f_signal(df, "D01")





y = df["target"]
X = df.loc[:, "t0":].astype(np.float32)

k = pca_elbow(X, th=0.95, estandarizar=True)





pca_model, pca_scores = f_pca(X, n=k, estandarizar=True)





plot_pca_scatter(pca_scores, y, comp1=0, comp2=1)





comp1 = 0
comp2 = 1
estilo = "RdYlBu"
f_pcagraf(y, pca_scores, comp1, comp2, estilo)

# -------------------------------------------------
# 5) XGBoost multiclase (COPD vs SMOKERS vs CONTROL)
# -------------------------------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)  # [0..K-1]
classlabel = dict(enumerate(le.classes_))
n_classes = len(le.classes_)
print("Clases presentes:", le.classes_)





X_train, X_test, y_train, y_test = fsplit(pca_scores, y_enc, cut_size=0.30, seed=34)

y_pred, bestxgb, y_proba = xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=n_classes,
                                       kn=3)
metrics_multiclass(y_test, y_pred, y_proba, n_classes=n_classes, target_names=le.classes_)


# SEGUIR  la concatenacion entre los datos y los pca
merge =pd.concat([df[["sex", "age"]], pca_scores], axis=1)

lemerge = LabelEncoder()
sex_enc = lemerge.fit_transform(merge["sex"])
merge["sex"]=sex_enc

X_train, X_test, y_train, y_test = fsplit(merge, y_enc, cut_size=0.30, seed=34)

y_pred, bestxgb, y_proba = xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=n_classes,
                                       kn=3)
metrics_multiclass(y_test, y_pred, y_proba, n_classes=n_classes, target_names=le.classes_)



