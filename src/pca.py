"""
PCA utilities for dimensionality reduction, visualization, and exploratory analysis.
Includes functions to compute the optimal number of components,
apply PCA transformation, and generate scatter plots with class labels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import mplcursors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def pca_elbow(df: pd.DataFrame, th: float = 0.95, estandarizar: bool = True) -> int:
    """
        Determine the optimal number of PCA components using the elbow (cumulative variance) method.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset (features only, numeric).
        th : float, default=0.95
            Threshold of cumulative explained variance ratio (e.g. 0.95 = 95%).
        standardize : bool, default=True
            Whether to standardize features before PCA.

        Returns
        -------
        k : int
            Minimum number of components needed to explain at least `th` variance.
        """

    print("Computing PCA elbow…", end='\r')
    X_in = df.copy()
    if estandarizar:
        X_in = pd.DataFrame(StandardScaler().fit_transform(X_in), index=df.index, columns=df.columns)
    pca = PCA().fit(X_in)
    var_cum = pca.explained_variance_ratio_.cumsum()
    k = int(np.argmax(var_cum >= th) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(var_cum) + 1), var_cum, marker='o', linestyle='--')
    plt.title('PCA Elbow (Cumulative Variance)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"PCA Elbow: {k} componentes para ≥{int(th * 100)}% variance")
    return k

def f_pca(df: pd.DataFrame, n: int, estandarizar: bool = True):
    """
        Apply PCA transformation and return the transformed dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset (features only).
        n : int
            Number of PCA components to compute.
        standardize : bool, default=True
            Whether to standardize features before PCA.

        Returns
        -------
        pca : PCA
            Fitted PCA model.
        pca_df : pd.DataFrame
            Transformed dataset with `COMP_1 ... COMP_n` as columns.
        """

    X_in = df.copy()
    if estandarizar:
        X_in = pd.DataFrame(StandardScaler().fit_transform(X_in), index=df.index, columns=df.columns)
    pca = PCA(n)
    pdata = pca.fit_transform(X_in)
    pca_df = pd.DataFrame(pdata, index=df.index, columns=[f"COMP_{i + 1}" for i in range(n)])
    return pca, pca_df

def plot_pca_scatter(pca_df: pd.DataFrame, y: pd.Series, comp1: int = 0, comp2: int = 1):
    """
        Plot PCA scatter plot for two selected components, colored by class.

        Parameters
        ----------
        pca_df : pd.DataFrame
            DataFrame with PCA-transformed features (COMP_1 ... COMP_n).
        y : pd.Series
            Class labels aligned with `pca_df` index.
        comp1 : int, default=0
            First component index (zero-based).
        comp2 : int, default=1
            Second component index (zero-based).
        """

    plot_df = pca_df.copy()
    plot_df["Class"] = y.loc[plot_df.index]
    plot_df["patient_code"] = plot_df.index

    # Map class labels to numeric IDs dynamically
    classes_present = sorted(plot_df["Class"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(classes_present)}
    id2label = {i: lab for lab, i in label2id.items()}
    plot_df["Class_id"] = plot_df["Class"].map(label2id)

    xcol = f"COMP_{comp1 + 1}"
    ycol = f"COMP_{comp2 + 1}"

    # Scatter plot
    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colors.ListedColormap(
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'][:len(classes_present)])
    sc = plt.scatter(plot_df[xcol], plot_df[ycol], c=plot_df["Class_id"], cmap=cmap, edgecolor='k', s=50, alpha=0.95)

    # Add colorbar with class labels
    cbar = plt.colorbar(sc)
    ticks = list(range(len(classes_present)))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([id2label[t] for t in ticks])
    cbar.set_label('Class')

    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.axvline(0, color='grey', lw=1, linestyle='--')
    plt.title('PCA Scatter by Class')
    plt.xlabel(xcol)
    plt.ylabel(ycol)

    # Interactive hover labels
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def _(sel):
        i = sel.index
        pid = plot_df["patient_code"].iloc[i]
        xv = plot_df[xcol].iloc[i]
        yv = plot_df[ycol].iloc[i]
        cl = plot_df["Class"].iloc[i]
        sel.annotation.set_text(f"patient_code: {pid}\nClass: {cl}\n{xcol}: {xv:.2f}\n{ycol}: {yv:.2f}")

    plt.tight_layout()
    plt.show()


def f_pcagraf(df_out, pcas, comp1, comp2, style):
    """
        Plot PCA scatter for fixed class dictionary (C, D, S).

        Parameters
        ----------
        y : pd.Series
            True class labels.
        pca_scores : pd.DataFrame
            DataFrame of PCA scores (COMP_1 ...).
        comp1 : int
            First component index (zero-based).
        comp2 : int
            Second component index (zero-based).
        style : str
            Colormap style (currently unused, kept for extension).
        """

    print("Generating PCA scatter plot…", end='\r')
    pca = pcas.copy()
    pca['Class'] = df_out.values
    pca_df = pca.sort_values(by='Class')

    # Fixed mapping for 3 known classes
    classdic = {'C': 0, 'D': 1, 'S': 2}
    pca_df['Class'] = pca_df['Class'].map(classdic)

    plt.figure(figsize=(10, 6))
    unique_classes = pca_df['Clase'].unique()
    # Definir paleta de colores (3 clases máximo)
    colors = ['black', 'red', 'green']
    cmap = matplotlib.colors.ListedColormap(colors[:len(unique_classes)])

    scatter = plt.scatter(
        pca_df[f'COMP_{comp1 + 1}'],
        pca_df[f'COMP_{comp2 + 1}'],
        c=pca_df['Class'], cmap=cmap, alpha=1, edgecolor='k', s=50
    )

    # Colorbar with fixed labels
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class')
    cbar.set_ticks(list(classdic.values()))
    cbar.set_ticklabels(list(classdic.keys()))

    plt.xlabel(f'COMP_{comp1 + 1}')
    plt.ylabel(f'COMP_{comp2 + 1}')

    # Interactive hover info
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
    print("PCA scatter plot: OK")