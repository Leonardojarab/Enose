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
from src.utils import get_git_commit, get_dataset_version_hash, log_dataframe_as_artifact, log_dict_as_artifact, log_current_time

#Standard libraries
import numpy as np, pandas as pd
from sklearn.preprocessing import  LabelEncoder
import mlflow



if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Enose2")
    import mlflow.sklearn

    mlflow.sklearn.autolog(
        log_models=False,  # evitas que guarde *todos* los modelos intermedios
        log_post_training_metrics=True,  # te registra automáticamente métricas post-fit
        max_tuning_runs=100  # límite de runs hijos del GridSearchCV
    )
    with mlflow.start_run(run_name="Enose_XGB_GridSearch"):
        mlflow.set_tags({
            "project": "enose",
            "author": "leo.jara",
            "git_commit": get_git_commit(),
        })

        log_current_time()

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

        #dataset version
        dataset_hash = get_dataset_version_hash(df, extra_info="control-copd-smokers-general")
        mlflow.set_tag("dataset_version", dataset_hash)
        # Log artifact
        log_dataframe_as_artifact(df.head(20), "sample_data.csv")
        log_dict_as_artifact({"note": "first experiment with MLflow"})

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


        # =========================
        # Experiment 1: PCA only
        # =========================
        with mlflow.start_run(nested=True, run_name="PCA_features"):
            # Tags/params específicos del experimento
            mlflow.set_tags({
                "feature_set": "pca_only",
                "labels": ",".join(map(str, le.classes_)),
            })
            mlflow.log_param("pca_n_components", int(k))
            mlflow.log_param("split_seed", 34)
            mlflow.set_tag("cv_scoring", "recall_macro")
            mlflow.set_tag("cv_folds", "3")

            # Train-test split
            X_train, X_test, y_train, y_test = fsplit(pca_scores, y_enc, cut_size=0.30, seed=34)

            # Train XGBoost with hyperparameter grid search
            y_pred, bestxgb, y_proba = xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=n_classes,
                                                   kn=3,class_names=list(le.classes_))

            # Evaluate metrics
            metrics_multiclass(y_test, y_pred, y_proba, n_classes=n_classes, target_names=le.classes_)

        # # ==========================================
        # # Experiment 2: PCA + metadata (sex, age)
        # # ==========================================
        # merge =pd.concat([df[["sex", "age"]], pca_scores], axis=1)
        #
        # # Encode categorical variable 'sex'
        # lemerge = LabelEncoder()
        # merge["sex"] = lemerge.fit_transform(merge["sex"])
        #
        # with mlflow.start_run(nested=True, run_name="PCA_plus_metadata"):
        #     mlflow.set_tags({
        #         "feature_set": "pca_plus_sex_age",
        #         "labels": ",".join(map(str, le.classes_)),
        #     })
        #     mlflow.log_param("pca_n_components", int(k))
        #     mlflow.log_param("split_seed", 34)
        #     mlflow.set_tag("cv_scoring", "recall_macro")
        #     mlflow.set_tag("cv_folds", "3")
        #
        #     # Train-test split with extended dataset
        #     X_train, X_test, y_train, y_test = fsplit(merge, y_enc, cut_size=0.30, seed=34)
        #
        #     # Train XGBoost again
        #     y_pred, bestxgb, y_proba = xgb_grid_mc(X_train, X_test, y_train, y_test, score='recall_macro', n_classes=n_classes,
        #                                            kn=3
        #                                            ,class_names=list(le.classes_))
        #
        #     # Evaluate metrics
        #     metrics_multiclass(y_test, y_pred, y_proba, n_classes=n_classes, target_names=le.classes_)



