import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import mplcursors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def pca_elbow(df: pd.DataFrame, th: float = 0.95, estandarizar: bool = True) -> int:
    print("Cálculo codo PCA…", end='\r')
    X_in = df.copy()
    if estandarizar:
        X_in = pd.DataFrame(StandardScaler().fit_transform(X_in), index=df.index, columns=df.columns)
    pca = PCA().fit(X_in)
    var_cum = pca.explained_variance_ratio_.cumsum()
    k = int(np.argmax(var_cum >= th) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(var_cum) + 1), var_cum, marker='o', linestyle='--')
    plt.title('Codo PCA')
    plt.xlabel('Componentes')
    plt.ylabel('Varianza acumulada')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Cálculo codo PCA: OK → {k} componentes para ≥{int(th * 100)}% varianza")
    return k

def f_pca(df: pd.DataFrame, n: int, estandarizar: bool = True):
    X_in = df.copy()
    if estandarizar:
        X_in = pd.DataFrame(StandardScaler().fit_transform(X_in), index=df.index, columns=df.columns)
    pca = PCA(n)
    pdata = pca.fit_transform(X_in)
    pca_df = pd.DataFrame(pdata, index=df.index, columns=[f"COMP_{i + 1}" for i in range(n)])
    return pca, pca_df

def plot_pca_scatter(pca_df: pd.DataFrame, y: pd.Series, comp1: int = 0, comp2: int = 1):
    """comp1/comp2 son índices base 0."""
    plot_df = pca_df.copy()
    plot_df["Clase"] = y.loc[plot_df.index]
    plot_df["patient_code"] = plot_df.index

    # Mapeo dinámico según clases presentes
    classes_present = sorted(plot_df["Clase"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(classes_present)}
    id2label = {i: lab for lab, i in label2id.items()}
    plot_df["Clase_id"] = plot_df["Clase"].map(label2id)

    xcol = f"COMP_{comp1 + 1}"
    ycol = f"COMP_{comp2 + 1}"

    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colors.ListedColormap(
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'][:len(classes_present)])
    sc = plt.scatter(plot_df[xcol], plot_df[ycol], c=plot_df["Clase_id"], cmap=cmap, edgecolor='k', s=50, alpha=0.95)
    cbar = plt.colorbar(sc)
    ticks = list(range(len(classes_present)))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([id2label[t] for t in ticks])
    cbar.set_label('Clase')

    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.axvline(0, color='grey', lw=1, linestyle='--')
    plt.title('PCA por clase')
    plt.xlabel(xcol);
    plt.ylabel(ycol)

    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def _(sel):
        i = sel.index
        pid = plot_df["patient_code"].iloc[i]
        xv = plot_df[xcol].iloc[i]
        yv = plot_df[ycol].iloc[i]
        cl = plot_df["Clase"].iloc[i]
        sel.annotation.set_text(f"patient_code: {pid}\nClase: {cl}\n{xcol}: {xv:.2f}\n{ycol}: {yv:.2f}")

    plt.tight_layout()
    plt.show()


def f_pcagraf(df_out, pcas, comp1, comp2, estilo):
    print("Grafico PCA castor:", end='\r')
    pca = pcas.copy()
    pca['Clase'] = df_out.values
    # pca['participant_id']= df_out.index.values

    pca_df = pca.sort_values(by='Clase')

    # Mapear clases a IDs numéricos
    classdic = {'C': 0, 'D': 1, 'S': 2}
    pca_df['Clase'] = pca_df['Clase'].map(classdic)

    plt.figure(figsize=(10, 6))
    unique_classes = pca_df['Clase'].unique()
    # Definir paleta de colores (3 clases máximo)
    colors = ['black', 'red', 'green']
    cmap = matplotlib.colors.ListedColormap(colors[:len(unique_classes)])

    scatter = plt.scatter(
        pca_df[f'COMP_{comp1 + 1}'],
        pca_df[f'COMP_{comp2 + 1}'],
        c=pca_df['Clase'], cmap=cmap, alpha=1, edgecolor='k', s=50
    )

    # Barra de color
    cbar = plt.colorbar(scatter)
    cbar.set_label('Clase')
    cbar.set_ticks(list(classdic.values()))  # ticks numéricos
    cbar.set_ticklabels(list(classdic.keys()))  # etiquetas como C, D, S

    plt.xlabel(f'COMP_{comp1 + 1}')
    plt.ylabel(f'COMP_{comp2 + 1}')

    cursor = mplcursors.cursor(hover=True)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(
            # f"participant_id: {pca_df['participant_id'][sel.target.index]}"))
            # f"ID: {pca_df.iloc[sel.index]['participant_id']} \n"
            f"participant_id: {pca_df['participant_id'].iloc[sel.index]}\n"
            f"COMP_{comp1 + 1}:{sel.target[0]:.2f}\n"
            f"COMP_{comp2 + 1}:{sel.target[1]:.2f}"
        )
    )
    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.axvline(0, color='grey', lw=1, linestyle='--')
    plt.show
    print("Grafico PCA castor: OK")