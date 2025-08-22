import numpy as np
import matplotlib.pyplot as plt


def f_signalcheck(control, copd, smokers, df):
    # ∟ correlacion alta entre la señal señal repetida, no hace falta agregar la 2da señal repetida por sensor
    # de 272 señales solo 62 no estan correlacionadas
    contador = 0
    combinaciones = 0
    for patient in df.index:
        for sen in range(1, 9, 1):

            if patient[0] == "D":
                corr = np.corrcoef(copd[f"S{sen}_{patient}01"], copd[f"S{sen}_{patient}02"])[0, 1]
            if patient[0] == "S":
                corr = np.corrcoef(smokers[f"S{sen}_{patient}01"], smokers[f"S{sen}_{patient}02"])[0, 1]
            if patient[0] == "C":
                corr = np.corrcoef(control[f"S{sen}_{patient}01"], control[f"S{sen}_{patient}02"])[0, 1]
            if corr < 0.85:
                contador = contador + 1

            combinaciones = combinaciones + 1
    percentage = contador / combinaciones * 100
    return print(f"the percentage of signal not corralate are {percentage:.1f}%")

def f_signal(df, patient):
    signal = df.loc[patient, "t0":].to_numpy(dtype=np.float32)
    plt.figure(figsize=(12, 5))
    plt.plot(signal)
    plt.title(f"Patient {patient}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
