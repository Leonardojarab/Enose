# src/utils.py
import hashlib
import os
import subprocess
import datetime
import pandas as pd
import mlflow


def get_git_commit() -> str:
    """
    Return the current Git commit hash.
    If not in a Git repository, return 'no_git'.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except Exception:
        return "no_git"


def get_dataset_version_hash(df: pd.DataFrame, extra_info: str = "") -> str:
    """
    Generate a short hash representing the dataset version.
    Uses shape + column names + optional extra info (e.g., preprocessing steps).
    """
    raw = str(df.shape) + str(sorted(df.columns.tolist())) + extra_info
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def log_dataframe_as_artifact(df: pd.DataFrame, filename: str = "data.csv"):
    """
    Save a DataFrame as CSV and log it to MLflow as an artifact.
    The file is removed locally after logging.
    """
    df.to_csv(filename, index=False)
    mlflow.log_artifact(filename)
    #os.remove(filename)


def log_dict_as_artifact(d: dict, filename: str = "info.txt"):
    """
    Save a dictionary as a text file and log it to MLflow as an artifact.
    The file is removed locally after logging.
    """
    with open(filename, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")
    mlflow.log_artifact(filename)
    os.remove(filename)

def log_current_time():
    """
    Registra la hora actual como tag en MLflow.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mlflow.set_tag("run_timestamp", now)
    return now
