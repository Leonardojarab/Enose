"""
Data formatting utilities.

This module prepares the main DataFrame by:
- Extracting patient metadata (code, sex, age).
- Concatenating sensor signals across all sensors and sessions.
- Creating a final dataset with metadata + signal time series.
"""
import pandas as pd


def f_format(control, copd, smokers, general):
    """
        Build a formatted DataFrame with patient metadata and concatenated signals.

        Each patient has 8 sensors measured in two sessions.
        This function concatenates those signals into a single row per patient,
        while keeping metadata such as sex, age, and group (COPD, Smoker, Control).

        Parameters
        ----------
        control : pd.DataFrame
            Raw signal data for control patients.
        copd : pd.DataFrame
            Raw signal data for COPD patients.
        smokers : pd.DataFrame
            Raw signal data for smoker patients.
        general : pd.DataFrame
            General metadata table (patient codes, gender, age, etc.).

        Returns
        -------
        df : pd.DataFrame
            Final DataFrame with:
            - Index: patient_code
            - Columns: sex, age, target (group label), and signals (t0, t1, ..., tn).
        """

    # Initialize DataFrame with metadata fields
    df = pd.DataFrame(columns=["patiente_code", "sex", "age"])

    # Collect patient codes and metadata from general dataset
    df["patiente_code"] = pd.concat(
        [general["COPD_patient_code"], general["SMOKERS_patient_code"], general["CONTROL_patient_code"]],
        ignore_index=True).dropna()
    df["sex"] = pd.concat([general["Gender"], general["Gender.1"], general["Gender.2"]], ignore_index=True).dropna()
    df["age"] = pd.concat([general["Age"], general["Age.1"], general["Age.2"]], ignore_index=True).dropna()

    # Use patient_code as DataFrame index
    df = df.set_index("patiente_code")

    signals_df = []  # list of rows (each one is the full signal per patient)

    # Loop through each patient
    for patient in df.index:
        if patient[0] == "D":  # COPD patient
            s1_cols = [c for c in copd.columns if f"_{patient}0" in c]
            signal = pd.concat([copd[c] for c in s1_cols], ignore_index=True).reset_index(drop=True)
            df.at[patient, "target"] = patient[0]

        elif patient[0] == "S": # Smoker patient
            s1_cols = [c for c in smokers.columns if f"_{patient}0" in c]
            signal = pd.concat([smokers[c] for c in s1_cols], ignore_index=True).reset_index(drop=True)
            df.at[patient, "target"] = patient[0]

        if patient[0] == "C":  # Control patient
            s1_cols = [c for c in control.columns if f"_{patient}0" in c]
            signal = pd.concat([control[c] for c in s1_cols], ignore_index=True).reset_index(drop=True)
            df.at[patient, "target"] = patient[0]

        # Convert the concatenated signal into a one-row DataFrame
        signal_row = pd.DataFrame([signal.values], index=[patient])
        signals_df.append(signal_row)

    # Concatenate all patients' signals
    signals_df = pd.concat(signals_df)

    # Rename signal columns as t0, t1, t2, ...
    signals_df.columns = [f"t{i}" for i in range(signals_df.shape[1])]

    # Merge metadata with signal block
    df = pd.concat([df, signals_df], axis=1)

    return df