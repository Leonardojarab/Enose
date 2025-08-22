
import pandas as pd


def f_format(control, copd, smokers, general):
    df = pd.DataFrame(columns=["patiente_code", "sex", "age"])

    df["patiente_code"] = pd.concat(
        [general["COPD_patient_code"], general["SMOKERS_patient_code"], general["CONTROL_patient_code"]],
        ignore_index=True).dropna()
    df["sex"] = pd.concat([general["Gender"], general["Gender.1"], general["Gender.2"]], ignore_index=True).dropna()
    df["age"] = pd.concat([general["Age"], general["Age.1"], general["Age.2"]], ignore_index=True).dropna()

    df = df.set_index("patiente_code")
    # concatenar los 8 sensores por pacientes de la primera señal
    signals_df = []
    for patient in df.index:
        if patient[0] == "D":
            # selecciono todas las columnas que quier concatenar
            s1_cols = [c for c in copd.columns if f"_{patient}0" in c]
            # las agrego al dataset como listas usan el comando at, con iloc da error
            signal = pd.concat([copd[c] for c in s1_cols], ignore_index=True).reset_index(drop=True)
            # df.at[patient,"signal"] = pd.concat([copd[c] for c in s1_cols], ignore_index=True).tolist()
            df.at[patient, "target"] = patient[0]
        if patient[0] == "S":
            s1_cols = [c for c in smokers.columns if f"_{patient}0" in c]
            signal = pd.concat([smokers[c] for c in s1_cols], ignore_index=True).reset_index(drop=True)
            # df.at[patient,"signal"] = pd.concat([smokers[c] for c in s1_cols], ignore_index=True).tolist()
            df.at[patient, "target"] = patient[0]
        if patient[0] == "C":
            s1_cols = [c for c in control.columns if f"_{patient}0" in c]
            signal = pd.concat([control[c] for c in s1_cols], ignore_index=True).reset_index(drop=True)
            # df.at[patient,"signal"] = pd.concat([control[c] for c in s1_cols], ignore_index=True).tolist()
            df.at[patient, "target"] = patient[0]

        # convierto la señal en un DataFrame de 1 fila
        signal_row = pd.DataFrame([signal.values], index=[patient])
        signals_df.append(signal_row)
    # I concatenate the entire block of signals
    signals_df = pd.concat(signals_df)

    # renombro las columnas de la señal como t0, t1, t2, ...
    signals_df.columns = [f"t{i}" for i in range(signals_df.shape[1])]

    # uno metadata con señales
    df = pd.concat([df, signals_df], axis=1)

    return df