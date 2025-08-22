"""
Signal utilities for correlation analysis and visualization.

This module provides:
- f_signalcheck: checks correlation between duplicate sensor signals
- f_signal: plots the time-series signal of a specific patient
"""


import numpy as np
import matplotlib.pyplot as plt


def f_signalcheck(control, copd, smokers, df):
    """
        Check correlation between duplicate sensor measurements for each patient.

        Each patient has two measurements per sensor (e.g., S1_patient01, S1_patient02).
        This function compares them and computes the percentage of pairs that are not
        highly correlated (threshold < 0.85).

        Parameters
        ----------
        control : pd.DataFrame
            DataFrame containing control group sensor data.
        copd : pd.DataFrame
            DataFrame containing COPD patient sensor data.
        smokers : pd.DataFrame
            DataFrame containing smoker patient sensor data.
        df : pd.DataFrame
            DataFrame containing patient metadata (index = patient codes).

        Returns
        -------
        None
            Prints the percentage of non-correlated signals.
        """
    count = 0
    pairs = 0
    for patient in df.index:
        for sen in range(1, 9, 1):

            if patient[0] == "D":
                corr = np.corrcoef(copd[f"S{sen}_{patient}01"], copd[f"S{sen}_{patient}02"])[0, 1]
            if patient[0] == "S":
                corr = np.corrcoef(smokers[f"S{sen}_{patient}01"], smokers[f"S{sen}_{patient}02"])[0, 1]
            if patient[0] == "C":
                corr = np.corrcoef(control[f"S{sen}_{patient}01"], control[f"S{sen}_{patient}02"])[0, 1]
            if corr < 0.85:
                count = count + 1

            pairs = pairs + 1
    percentage = count / pairs * 100
    return print(f"the percentage of signal not corralate are {percentage:.1f}%")

def f_signal(df, patient):
    """
        Plot the time-series signal of a given patient.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing patient signals with columns t0, t1, ..., tn.
        patient : str
            Patient code (index in df).

        Returns
        -------
        None
            Displays the signal plot.
        """

    signal = df.loc[patient, "t0":].to_numpy(dtype=np.float32)
    plt.figure(figsize=(12, 5))
    plt.plot(signal)
    plt.title(f"Patient {patient}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Signal amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
