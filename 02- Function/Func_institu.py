import os
# ---------------------------------------------------------
# 1. Fonction : lecture + restructuration
# ---------------------------------------------------------
def reshape_government_data(file_path, sheet_name=0):
    """
    Transforme les données WGI format large en format tidy.
    Attend colonnes : Country | Country Code | Series | 2010...2023
    Retourne colonnes : Country | Year | gov_eff | con_corr | ...
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Passer en format long
    df_long = df.melt(id_vars=["Country", "Country Code", "Series"],
                      var_name="Year", value_name="Value")
    df_long["Year"] = df_long["Year"].astype(int)

    # Pivot pour avoir une colonne par indicateur
    df_pivot = df_long.pivot_table(index=["Country", "Country Code", "Year"],
                                   columns="Series", values="Value").reset_index()
    
    # Harmonisation noms si nécessaire
    rename_map = {
        "reg_quali": "reg_qual",
        "con_corr": "con_corr",
        "gov_eff": "gov_eff",
        "pol_sta": "pol_sta",
        "rul_law": "rul_law",
        "voi_acc": "voi_acc"
    }
    df_pivot = df_pivot.rename(columns=rename_map)

    return df_pivot


# ---------------------------------------------------------
# 1. Fonction : Save Dataframe & Figure
# ---------------------------------------------------------



def save_to_excel(df, path):
    """
    Save a DataFrame to an Excel file, creating folders if necessary.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save.
    path : str
        File path (including filename) where the Excel file will be saved.
        Example: "05-Result/join data/df_p_enrichi.xlsx"
    """
    # ensure folders exist
    folder = os.path.dirname(path)
    if folder:  # avoid errors if path is only a filename
        os.makedirs(folder, exist_ok=True)

    # save file
    df.to_excel(path, index=False)
    print(f"✅ DataFrame saved successfully to: {path}")



def save_figure(fig, path, dpi=300):
    """
    Save a matplotlib figure to a file, creating folders if necessary.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    path : str
        File path (including filename) where the figure will be saved.
        Example: "05-Result/plots/my_plot.png"
    dpi : int, optional
        Resolution in dots per inch (default = 300).
    """
    folder = os.path.dirname(path)
    if folder:  # avoid issues if only filename is passed
        os.makedirs(folder, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Figure saved successfully to: {path}")



# ---------------------------------------------------------
# 2. Fonction : imputation + scaling
# ---------------------------------------------------------

from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

def preprocess_data(df, var_cols, scaler_type='standard'):
    """
    Impute missing values and scale selected variables.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input dataframe.
    var_cols : list
        List of variable (column) names to impute and scale.
    scaler_type : str, optional
        Type of scaler to use: 'standard' or 'robust'. Default is 'standard'.

    Returns:
    -------
    df_imputed : pandas.DataFrame
        DataFrame with imputed values.
    X_scaled : numpy.ndarray
        Scaled version of the selected columns.
    """

    # Step 1: Imputation
    imputer = IterativeImputer(random_state=42, max_iter=10)
    df_imputed = df.copy()
    df_imputed[var_cols] = imputer.fit_transform(df[var_cols])

    # Step 2: Scaling
    if scaler_type == 'robust':
        scaler = RobustScaler(quantile_range=(5, 95))
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'robust'")

    X_scaled = scaler.fit_transform(df_imputed[var_cols])

    return df_imputed, X_scaled

# ---------------------------------------------------------
# 3. Fonction : calcul poids PCA
# ---------------------------------------------------------

from sklearn.decomposition import PCA 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_pca_weights(X_scaled, var_cols, threshold=0.7):
    """
    Calcule les poids des variables selon PCA et retourne :
    - weights : importance normalisée des variables
    - explained_var : variance expliquée par chaque axe
    - cum_var : variance cumulée
    - pca : objet PCA entraîné
    """
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    n_pc = np.argmax(cum_var >= threshold) + 1

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(var_cols))],
        index=var_cols
    )
    weights = loadings.iloc[:, :n_pc].abs().sum(axis=1)
    weights /= weights.sum()

    return weights, explained_var, cum_var, pca


def pca_plot(X_scaled, var_cols, pca, ax1=1, ax2=2, plot_type="variables",
             save_path=None, show=True):
    """
    Affiche une carte PCA (variables ou individus).

    Paramètres :
    ------------
    X_scaled : ndarray
        Données standardisées.
    var_cols : list
        Noms des variables.
    pca : objet PCA
        Résultat du fit PCA.
    ax1, ax2 : int
        Numéros des axes à représenter (1-based).
    plot_type : str
        'variables' ou 'individus'.
    save_path : str
        Chemin pour sauvegarder l'image (PNG, PDF, etc.).
    show : bool
        Afficher le graphique.
    """
    pc_labels = [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_)+1)]
    X_pca = pca.transform(X_scaled)

    plt.figure(figsize=(10, 8))

    if plot_type == "variables":
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=pc_labels,
            index=var_cols
        )
        for i in range(len(loadings)):
            plt.arrow(0, 0,
                      loadings.iloc[i, ax1-1],
                      loadings.iloc[i, ax2-1],
                      color="red", alpha=0.7,
                      head_width=0.03, head_length=0.03)
            plt.text(loadings.iloc[i, ax1-1]*1.1,
                     loadings.iloc[i, ax2-1]*1.1,
                     loadings.index[i],
                     fontsize=12, color="red")

    elif plot_type == "individus":
        plt.scatter(X_pca[:, ax1-1], X_pca[:, ax2-1],
                    alpha=0.6, c="blue", edgecolors="k")
    
    plt.xlabel(f"PC{ax1} ({pca.explained_variance_ratio_[ax1-1]*100:.1f}%)")
    plt.ylabel(f"PC{ax2} ({pca.explained_variance_ratio_[ax2-1]*100:.1f}%)")
    plt.title(f"Projection PCA - {plot_type.capitalize()}")
    plt.axhline(0, color="grey", lw=1)
    plt.axvline(0, color="grey", lw=1)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()



from sklearn.cluster import KMeans


def pca_individus_clusters(X_scaled, df, pca, selected_countries, n_clusters=4, save_path=None, show=True):
    """
    Affiche la projection PCA des individus sélectionnés,
    regroupés par clusters.
    """
    X_pca = pca.transform(X_scaled)
    df_proj = df.copy()
    df_proj["PC1"] = X_pca[:,0]
    df_proj["PC2"] = X_pca[:,1]

    # Moyenne par pays
    df_indiv = df_proj.groupby("Country")[["PC1","PC2"]].mean().reset_index()
    df_indiv = df_indiv[df_indiv["Country"].isin(selected_countries)]

    # Clustering (KMeans par défaut)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_indiv["Cluster"] = kmeans.fit_predict(df_indiv[["PC1","PC2"]])

    # Figure
    plt.figure(figsize=(12,8))
    colors = plt.cm.get_cmap("tab10", n_clusters)

    for cluster in range(n_clusters):
        cluster_data = df_indiv[df_indiv["Cluster"] == cluster]
        plt.scatter(cluster_data["PC1"], cluster_data["PC2"],
                    color=colors(cluster), alpha=0.7, label=f"Cluster {cluster+1}")
        for i, txt in enumerate(cluster_data["Country"]):
            plt.text(cluster_data["PC1"].iloc[i]+0.02,
                     cluster_data["PC2"].iloc[i]+0.02,
                     txt, fontsize=9)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Projection PCA des pays sélectionnés avec clustering")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()





# ---------------------------------------------------------
# 4. Fonction finale : score + risque
# ---------------------------------------------------------



def compute_government_stability(df, X_scaled, weights,
                                 csv_path=None, img_path=None):
    """
    Calcule l'indice de Stabilité Gouvernementale et l'expose en CSV et figure.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame avec colonnes ['Country', 'Year', ...] correspondant aux X_scaled.
    X_scaled : np.ndarray ou pd.DataFrame
        Matrice des variables normalisées.
    weights : pd.Series ou np.ndarray
        Poids pour la combinaison linéaire.
    csv_path : str, optionnel
        Chemin complet pour sauvegarder le CSV. Si None, le CSV n'est pas exporté.
    img_path : str, optionnel
        Chemin complet pour sauvegarder la figure. Si None, la figure n'est pas sauvegardée.

    Retour
    ------
    df_scores : pd.DataFrame
        DataFrame contenant les colonnes ['SG', 'SG_norm', 'SG_vol', 'P', 'I', 'R_cat'].
    """
    df_scores = df.copy()
    df_scores["SG"] = np.dot(X_scaled, weights.values)

    # Normalisation 0–100
    df_scores["SG_norm"] = (df_scores["SG"] - df_scores["SG"].min()) / \
                           (df_scores["SG"].max() - df_scores["SG"].min()) * 100

    # Probabilité (volatilité mobile)
    df_scores["SG_vol"] = df_scores.groupby("Country")["SG_norm"].rolling(
        3, min_periods=2).std().reset_index(level=0, drop=True)
    df_scores["P"] = (df_scores["SG_vol"] - df_scores["SG_vol"].min()) / \
                     (df_scores["SG_vol"].max() - df_scores["SG_vol"].min())

    # Impact = inverse stabilité
    df_scores["I"] = 1 - (df_scores["SG_norm"] / 100)

    # Risque
    df_scores["R_cat"] = df_scores["P"] * df_scores["I"]

    # --- Export CSV ---
    if csv_path:
        folder_csv = os.path.dirname(csv_path)
        if folder_csv:
            os.makedirs(folder_csv, exist_ok=True)
        df_scores.to_csv(csv_path, index=False)
        print(f"✅ Résultats exportés : {csv_path}")

    # --- Tracé exemple ---
    if img_path:
        folder_img = os.path.dirname(img_path)
        if folder_img:
            os.makedirs(folder_img, exist_ok=True)

        plt.figure(figsize=(10,6))
        for country in df_scores["Country"].unique()[:5]:
            subset = df_scores[df_scores["Country"] == country]
            plt.plot(subset["Year"], subset["SG_norm"], label=country)
        plt.title("Indice de Stabilité Gouvernementale (SG_norm)")
        plt.xlabel("Année")
        plt.ylabel("SG (0–100)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure sauvegardée : {img_path}")
        plt.show()

    return df_scores




    
def construire_indice_synthetique(X_scaled, df, pca, method="pc1"):
    """
    Construit un indice institutionnel synthétique à partir de l'ACP.
    
    method : 
        - "pc1" : on prend seulement le premier axe principal.
        - "weighted" : combinaison pondérée des axes par variance expliquée.
    """
    X_pca = pca.transform(X_scaled)
    
    if method == "pc1":
        indice = X_pca[:,0]   # Premier axe
    elif method == "weighted":
        weights = pca.explained_variance_ratio_
        indice = (X_pca * weights).sum(axis=1)
    else:
        raise ValueError("Méthode inconnue. Choisir 'pc1' ou 'weighted'.")

    # Associer à chaque pays et année
    df_indice = df.copy()
    df_indice["Indice_Instit"] = indice
    return df_indice






def compute_doing_business(df, X_scaled, weights,
                                 csv_path=None, img_path=None):
    """
    Calcule l'indice de Stabilité Gouvernementale et l'expose en CSV et figure.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame avec colonnes ['Country', 'Year', ...] correspondant aux X_scaled.
    X_scaled : np.ndarray ou pd.DataFrame
        Matrice des variables normalisées.
    weights : pd.Series ou np.ndarray
        Poids pour la combinaison linéaire.
    csv_path : str, optionnel
        Chemin complet pour sauvegarder le CSV. Si None, le CSV n'est pas exporté.
    img_path : str, optionnel
        Chemin complet pour sauvegarder la figure. Si None, la figure n'est pas sauvegardée.

    Retour
    ------
    df_scores : pd.DataFrame
        DataFrame contenant les colonnes ['SG', 'SG_norm', 'SG_vol', 'P', 'I', 'R_cat'].
    """
    df_scores = df.copy()
    df_scores["SG"] = np.dot(X_scaled, weights.values)

    # Normalisation 0–100
    df_scores["SG_norm"] = (df_scores["SG"] - df_scores["SG"].min()) / \
                           (df_scores["SG"].max() - df_scores["SG"].min()) * 100

    # Probabilité (volatilité mobile)
    df_scores["SG_vol"] = df_scores.groupby("Country")["SG_norm"].rolling(
        3, min_periods=2).std().reset_index(level=0, drop=True)
    df_scores["P"] = (df_scores["SG_vol"] - df_scores["SG_vol"].min()) / \
                     (df_scores["SG_vol"].max() - df_scores["SG_vol"].min())

    # Impact = inverse stabilité
    df_scores["I"] = 1 - (df_scores["SG_norm"] / 100)

    # Risque
    df_scores["R_cat"] = df_scores["P"] * df_scores["I"]

    # --- Export CSV ---
    if csv_path:
        folder_csv = os.path.dirname(csv_path)
        if folder_csv:
            os.makedirs(folder_csv, exist_ok=True)
        df_scores.to_csv(csv_path, index=False)
        print(f"✅ Résultats exportés : {csv_path}")

    # --- Tracé exemple ---
    if img_path:
        folder_img = os.path.dirname(img_path)
        if folder_img:
            os.makedirs(folder_img, exist_ok=True)

        plt.figure(figsize=(10,6))
        for country in df_scores["Country"].unique()[:5]:
            subset = df_scores[df_scores["Country"] == country]
            plt.plot(subset["Year"], subset["SG_norm"], label=country)
        plt.title("Indice de Stabilité Gouvernementale (SG_norm)")
        plt.xlabel("Année")
        plt.ylabel("SG (0–100)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure sauvegardée : {img_path}")
        plt.show()

    return df_scores




# ---------------------------------------------------------
# Score analysis
# ---------------------------------------------------------


def handle_missing_values(df, method="interpolate"):
    """
    Gère les valeurs manquantes dans le DataFrame panel (Country, Year, scores...).
    method : "interpolate", "mean", "median"
    """
    df_clean = df.copy()
    score_cols = ["score_wb", "score_cpi", "score_fh", "score_ief", "score_wgi"]
    
    if method == "interpolate":
        # interpolation par pays sur les années
        df_clean = df_clean.sort_values(["Country", "Year"])
        for col in score_cols:
            df_clean[col] = df_clean.groupby("Country")[col].transform(
                lambda x: x.interpolate(limit_direction="both")
            )
    
    elif method == "mean":
        df_clean[score_cols] = df_clean.groupby("Country")[score_cols].transform(
            lambda group: group.fillna(group.mean())
        )
    
    elif method == "median":
        df_clean[score_cols] = df_clean[score_cols].fillna(df_clean[score_cols].median())
    
    else:
        raise ValueError("Méthode inconnue: choisir 'interpolate', 'mean' ou 'median'.")
    
    return df_clean



from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def compute_weights_stat(df, method="pca"):
    """
    Calcule les poids statistiques (ACP ou Entropie).
    Gère automatiquement les NaN résiduels.
    """
    X = df.drop(columns=["Country", "Year"]).values
    
    # 🔹 Étape de sécurité : imputation finale
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    if method == "pca":
        pca = PCA(n_components=1)
        pca.fit(X)
        loadings = np.abs(pca.components_[0])
        weights = loadings / loadings.sum()
        return dict(zip(df.drop(columns=["Country", "Year"]).columns, weights))
    
    elif method == "entropy":
        X = X / X.sum(axis=0)  # probabilité relative
        k = 1 / np.log(X.shape[0])
        entropy = -k * (X * np.log(X + 1e-12)).sum(axis=0)
        d = 1 - entropy
        weights = d / d.sum()
        return dict(zip(df.drop(columns=["Country", "Year"]).columns, weights))
    
    else:
        raise ValueError("Méthode inconnue: choisir 'pca' ou 'entropy'.")
    

def compute_global_score_hybrid(df_clean, method_stat="pca", weights_expert=None, alpha=0.5):
    """
    Calcule le score global institutionnel hybride.
    """
    # 🔹 1. Nettoyage des NaN
    #df_clean = handle_missing_values(df, method=na_method)

    # 🔹 2. Normaliser entre 0 et 1
    scaler = MinMaxScaler()
    df_scaled = df_clean.copy()
    score_cols = ["score_wb", "score_cpi", "score_fh", "score_ief", "score_wgi"]
    df_scaled[score_cols] = scaler.fit_transform(df_clean[score_cols])

    # 🔹 3. Poids statistiques
    weights_stat = compute_weights_stat(df_scaled, method=method_stat)

    # 🔹 4. Poids experts (par défaut égalitaires si non fournis)
    if weights_expert is None:
        weights_expert = {var: 1/5 for var in ["score_wb", "score_cpi", "score_fh", "score_ief", "score_wgi"]}

    # 🔹 5. Poids hybrides
    weights_hybrid = {
        var: alpha * weights_stat[var] + (1 - alpha) * weights_expert[var]
        for var in ["score_wb", "score_cpi", "score_fh", "score_ief", "score_wgi"]
    }

    # 🔹 6. Score global (somme pondérée)
    df_clean["score_global"] = sum(df_scaled[var] * w for var, w in weights_hybrid.items())

    return df_clean, weights_stat, weights_expert, weights_hybrid




# ---------------------------------------------------------
# Expo & Vuln
# ---------------------------------------------------------

def ensure_scores_panel(df, score_cols=None, country_col="Country", year_col="Year",
                        impute_method="interpolate"):
    """
    - s'assure que les colonnes score_cols existent ;
    - impute NaN par country (interpolate / mean / median) ;
    - met les scores en échelle 0-1 (détecte si valeurs >1 => divise par 100).
    Retourne une copie.
    """
    df = df.copy()
    if score_cols is None:
        # par défaut, toutes les colonnes qui commencent par 'score_' sauf 'score_global' optionnel
        score_cols = [c for c in df.columns if c.startswith("score_")]

    # interpolation/imputation par pays
    score_cols_present = [c for c in score_cols if c in df.columns]
    df = df.sort_values([country_col, year_col])

    if impute_method == "interpolate":
        for col in score_cols_present:
            df[col] = df.groupby(country_col)[col].transform(lambda s: s.interpolate(limit_direction="both"))
            # si toujours NaN => remplir par la moyenne pays
            df[col] = df.groupby(country_col)[col].transform(lambda s: s.fillna(s.mean()))
    elif impute_method == "mean":
        for col in score_cols_present:
            df[col] = df.groupby(country_col)[col].transform(lambda s: s.fillna(s.mean()))
    elif impute_method == "median":
        for col in score_cols_present:
            df[col] = df.groupby(country_col)[col].transform(lambda s: s.fillna(s.median()))
    else:
        raise ValueError("impute_method doit être 'interpolate','mean' ou 'median'")

    # Si colonnes encore NaN (ex: pays totalement manquant), imputer par global mean
    imputer = SimpleImputer(strategy="mean")
    df[score_cols_present] = imputer.fit_transform(df[score_cols_present])

    # Normalisation de l'échelle : si la médiane ou max > 1 => on suppose 0-100 et on divise par 100
    for col in score_cols_present:
        if df[col].max() > 1.0:
            df[col] = df[col] / 100.0

    # S'assurer bornes [0,1]
    df[score_cols_present] = df[score_cols_present].clip(0.0, 1.0)

    return df, score_cols_present

 
# EXPO : méthodes 

def compute_expo(df, expo_column=None, expo_map=None, fallback="score_global"):
    """
    - expo_column : nom de la colonne (ex: 'GDP' ou 'InsurancePremiums')
    - expo_map : dict country -> expo value (utiliser si fourni)
    - fallback : 'score_global' or 'constant' : fallback si la colonne n'existe pas
    Retourne df avec colonne EXPO (0-1)
    """
    df = df.copy()
    scaler = MinMaxScaler()

    if expo_map is not None:
        # map values (user-provided)
        df["EXPO_raw"] = df["Country"].map(expo_map)
        # si NaN -> fallback
        if expo_column is None or expo_column not in df.columns:
            df["EXPO_raw"] = df["EXPO_raw"].fillna(np.nan)
    else:
        if expo_column is not None and expo_column in df.columns:
            df["EXPO_raw"] = df[expo_column]
        else:
            df["EXPO_raw"] = np.nan

    # fallback: if EXPO_raw all NaN, use fallback metric
    if df["EXPO_raw"].isna().all():
        if fallback == "score_global" and "score_global" in df.columns:
            df["EXPO_raw"] = df["score_global"]
        elif fallback == "constant":
            df["EXPO_raw"] = 1.0
        else:
            # as last resort use 1
            df["EXPO_raw"] = 1.0

    # Imputer remaining NaN by country mean
    df["EXPO_raw"] = df.groupby("Country")["EXPO_raw"].transform(lambda s: s.fillna(s.mean()))
    df["EXPO_raw"] = df["EXPO_raw"].fillna(df["EXPO_raw"].mean())

    # normalize 0-1
    df["EXPO"] = scaler.fit_transform(df[["EXPO_raw"]]).ravel()
    df["EXPO"] = df["EXPO"].clip(0,1)
    return df


# VULN : méthodes 

def compute_vuln_methods(df, score_cols, method="composite", indicator_weights=None):
    """
    Retourne df avec colonnes VULN_inverse, VULN_weighted, VULN_pca et VULN_final
    - method: 'inverse'|'weighted'|'pca'|'composite'
    - indicator_weights: dict mapping score_col -> weight (used for 'weighted' or composite)
    Convention : les score_cols sont normalisés 0-1 (1 = pire).
    """
    df = df.copy()
    scaler = MinMaxScaler()

    # VULN_inverse : simple reprise du score_global si existant, sinon moyenne des indicateurs
    if "score_global" in df.columns:
        # assume score_global is 0-1 (if >1, user should pre-scale)
        v_inv = df["score_global"].copy()
        # ensure 0-1
        if v_inv.max() > 1.0: v_inv = v_inv / 100.0
        df["VULN_inverse"] = v_inv.clip(0,1)
    else:
        df["VULN_inverse"] = df[score_cols].mean(axis=1)

    # VULN_weighted : moyenne pondérée sur indicateurs (weights must sum to 1)
    if indicator_weights is None:
        # default sensible weights: give more weight to WGI-like indicators if present
        default = {}
        n = len(score_cols)
        for c in score_cols:
            default[c] = 1.0 / n
        indicator_weights = default
    # normalize weights
    s = sum(indicator_weights.get(c, 0.0) for c in score_cols)
    if s == 0:
        # fall back
        indicator_weights = {c: 1.0/len(score_cols) for c in score_cols}
    else:
        indicator_weights = {c: indicator_weights.get(c, 0.0)/s for c in score_cols}

    df["VULN_weighted"] = 0.0
    for c in score_cols:
        df["VULN_weighted"] += df[c] * indicator_weights.get(c, 0.0)
    df["VULN_weighted"] = df["VULN_weighted"].clip(0,1)

    # VULN_pca : première composante (après scaling) -> normaliser en 0-1
    X = df[score_cols].fillna(df[score_cols].mean())
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    f = pca.fit_transform(Xs).ravel()
    # normaliser 0-1
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-12)
    df["VULN_pca"] = f_norm.clip(0,1)

    # VULN_final selon méthode choisie
    if method == "inverse":
        df["VULN_final"] = df["VULN_inverse"]
    elif method == "weighted":
        df["VULN_final"] = df["VULN_weighted"]
    elif method == "pca":
        df["VULN_final"] = df["VULN_pca"]
    elif method == "composite":
        # moyenne des 3 méthodes (ou pondérée)
        df["VULN_final"] = (df["VULN_inverse"] + df["VULN_weighted"] + df["VULN_pca"]) / 3.0
    else:
        raise ValueError("method VULN inconnu")

    df["VULN_final"] = df["VULN_final"].clip(0,1)
    return df


# PROBABILITÉ / IMPACT / RISQUE 

def compute_prob_impact_risk(df, score_cols, vuln_col="VULN_final", expo_col="EXPO",
                             gamma=(0.5, 0.3, 0.2), kappa=(1.0, 0.5),
                             volatility_window=3, threshold_S=0.7):
    """
    Calcule P_{i,c}, I_{i,c}, R_{i,c} pour chaque score_col (catégorie).
    Hypothèse : score_cols sont en 0-1, vuln_col et expo_col existent.
    - gamma: tuple (γ1,γ2,γ3)
    - kappa: tuple (κ1,κ2)
    - volatility_window: fenêtre rolling (années) pour la volatilité
    - threshold_S: seuil binaire (S > threshold) pour composante γ3
    Renvoie df enrichi + dictionnaire des noms de colonnes créées.
    """
    df = df.copy()
    gamma1, gamma2, gamma3 = gamma
    k1, k2 = kappa  # k1 = κ0, k2 = κ1 (notation interne)

    created_cols = {"P": [], "I": [], "R": [], "sigma": []}

    for col in score_cols:
        S_col = col  # score normalisé 0-1

        # volatilité temporelle rolling par pays
        sigma_name = f"sigma_{col}"
        df[sigma_name] = df.groupby("Country")[S_col].transform(
            lambda s: s.rolling(window=volatility_window, min_periods=1).std().fillna(0)
        )
        sigma_t = df[sigma_name]
        sigma_max = sigma_t.max() if sigma_t.max() > 0 else 1.0

        P_col = f"P_{col}"
        I_col = f"I_{col}"
        R_col = f"R_{col}"

        # Probabilité : gamma1*S + gamma2*(sigma/sigma_max) + gamma3*indicator(S>threshold)
        # S_col est 0-1 => on l'utilise directement
        df[P_col] = gamma1 * df[S_col] + gamma2 * (df[sigma_name] / sigma_max) + gamma3 * (df[S_col] > threshold_S).astype(float)
        df[P_col] = df[P_col].clip(0, 1).fillna(0)

        # Impact : EXPO * VULN * (k1 + k2 * S)
        # IMPORTANT: k1 et k2 sont scalaires ; df[S_col] est un vecteur -> opération élément-wise valide
        df[I_col] = df[expo_col] * df[vuln_col] * (k1 + k2 * df[S_col])

        # Risque
        df[R_col] = df[P_col] * df[I_col]

        created_cols["P"].append(P_col)
        created_cols["I"].append(I_col)
        created_cols["R"].append(R_col)
        created_cols["sigma"].append(sigma_name)

    return df, created_cols


# AGRÉGATION SRI + CLASSIFICATION

def aggregate_and_classify(df, risk_cols, cat_weights=None,
                           sri_scale=100, classify_method="quantile",
                           n_classes=3):
    """
    Agrège les risques par catégories et calcule le SRI global + classes.
    """

    df = df.copy()

    # 1) Agrégation multi-catégorie
    if cat_weights is None:
        cat_weights = {col: 1/len(risk_cols) for col in risk_cols}

    df["SRI_raw"] = 0.0
    for col, w in cat_weights.items():
        if col in df.columns:
            df["SRI_raw"] += w * df[col]
    df["SRI_raw"] = 1- df["SRI_raw"]
    # 2) Mise à l'échelle SRI
    if isinstance(sri_scale, (int, float)):
        # simple multiplication
        df["SRI"] = df["SRI_raw"] * float(sri_scale)
    elif isinstance(sri_scale, (list, tuple)) and len(sri_scale) == 2:
        mn, mx = float(sri_scale[0]), float(sri_scale[1])
        raw = df["SRI_raw"].values
        if raw.max() - raw.min() < 1e-12:
            # valeurs constantes -> assigner milieu de l'intervalle
            df["SRI"] = (mn + mx) / 2.0
        else:
            df["SRI"] = mn + (raw - raw.min()) * (mx - mn) / (raw.max() - raw.min())
    else:
        raise ValueError("sri_scale doit être un scalaire ou un tuple/list (min,max)")

    # 3) Inversion du SRI (risque inverse)
    # df["SRI"] = 1 - df["SRI"]

    # 4) Gestion du cas NaN
    if df["SRI"].isna().all():
        print("⚠️ Toutes les valeurs de SRI sont NaN → impossible de classifier")
        df["Risk_Class"] = np.nan
        return df

    # 3) Labels textuels
    labels_map = ["Faible", "Moyen", "Elevé"]  # toujours défini
    n_classes = min(n_classes, len(labels_map))  # s'assurer que <=3

    # 4) Classification
    if classify_method == "quantile":
        df["Risk_Class"] = pd.qcut(df["SRI"], q=n_classes, labels=labels_map[:n_classes], duplicates="drop")

    elif classify_method == "kmeans":
        from sklearn.cluster import KMeans
        valid = df["SRI"].dropna().values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(valid)
        mapping = {i: labels_map[i] for i in range(len(np.unique(cluster_labels)))}
        df.loc[df["SRI"].notna(), "Risk_Class"] = [mapping[i] for i in cluster_labels]

    elif classify_method == "jenks":
        try:
            import mapclassify
            classifier = mapclassify.NaturalBreaks(df["SRI"].dropna(), k=n_classes)
            numeric_class = classifier.yb
            df.loc[df["SRI"].notna(), "Risk_Class"] = [labels_map[i] for i in numeric_class]
        except ImportError:
            print("⚠️ mapclassify non installé → fallback sur quantile")
            df["Risk_Class"] = pd.qcut(df["SRI"], q=n_classes, labels=labels_map[:n_classes], duplicates="drop")
    else:
        raise ValueError("Méthode de classification non reconnue.")


    return df



# PIPELINE COMPLET (exécutable) 

def compute_panel_institutional_risk(
        df,
        score_cols=None,
        country_col="Country",
        year_col="Year",
        impute_method="interpolate",
        expo_column="GDP",
        expo_map=None,
        expo_fallback="score_global",
        vuln_method="composite",
        indicator_weights=None,
        gamma=(0.5,0.3,0.2),
        kappa=(1.0,0.5),
        volatility_window=3,
        threshold_S=0.7,
        cat_weights=None,
        sri_scale=(0,100),
        classify_method="quantile",
        n_classes=5):
    """
    Orchestrateur complet pour panel (tous pays x années).
    """
    # 1) Pre-traitement scores & mise à l'échelle
    df_pp, score_cols_present = ensure_scores_panel(df, score_cols=score_cols,
                                                    country_col=country_col, year_col=year_col,
                                                    impute_method=impute_method)

    # 2) EXPO
    df_pp = compute_expo(df_pp, expo_column=expo_column, expo_map=expo_map, fallback=expo_fallback)

    # 3) VULN
    df_pp = compute_vuln_methods(df_pp, score_cols_present, method=vuln_method,
                                 indicator_weights=indicator_weights)

    # 4) P,I,R par catégorie
    df_pi, created = compute_prob_impact_risk(df_pp, score_cols_present,
                                              vuln_col="VULN_final", expo_col="EXPO",
                                              gamma=gamma, kappa=kappa,
                                              volatility_window=volatility_window,
                                              threshold_S=threshold_S)

    # 5) Agrégation & classification
    risk_cols = created["R"]
    df_final = aggregate_and_classify(df_pi, risk_cols, cat_weights=cat_weights,
                                      sri_scale=sri_scale, classify_method=classify_method,
                                      n_classes=n_classes)

    return df_final












# ---------------------------------------------------------
# Plot risk matrices
# ---------------------------------------------------------





from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

def animate_institutional_risk_matrices(df, group_dict, save_folder="animations",
                                        years_to_plot=None, interval=1000, max_show=3,
                                        prob_col="SRI"):
    """
    Crée une animation de l'évolution des matrices Impact × Probabilité par groupe.

    Paramètres
    ----------
    df : pd.DataFrame
        Contient ['Country','Risk_Class', prob_col]. Optionnel : 'Year'.
        'Risk_Class' doit être labellisé "Faible", "Moyen", "Élevé".
    group_dict : dict
        Dictionnaire {Groupe: [liste de pays]}.
    save_folder : str
        Dossier où sauvegarder les GIFs.
    years_to_plot : list[int], optionnel
        Liste des années à inclure. Si None, toutes les années présentes sont utilisées.
    interval : int
        Intervalle entre les frames en ms.
    max_show : int
        Nombre maximum d'animations affichées à l’écran.
    prob_col : str
        Colonne représentant la probabilité.
    """
    os.makedirs(save_folder, exist_ok=True)

    df_plot = df.copy()

    if 'Year' in df_plot.columns:
        if years_to_plot is None:
            years = sorted(df_plot['Year'].unique())
        else:
            years = [y for y in years_to_plot if y in df_plot['Year'].unique()]
    else:
        years = [None]  # pas de colonne année

    impact_map = {"Faible": 1, "Moyen": 2, "Élevé": 3, "Elevé": 3}
    risk_palette = {"Faible": "green", "Moyen": "orange", "Élevé": "red", "Elevé": "red"}

    df_plot["Impact"] = df_plot["Risk_Class"].map(impact_map)
    df_plot["Probabilité"] = df_plot[prob_col] / 100.0  # normalisation

    for idx_g, (group, pays_list) in enumerate(group_dict.items()):
        df_group = df_plot[df_plot["Country"].isin(pays_list)].copy()
        if df_group.empty:
            print(f"⚠️ Aucun résultat pour le groupe '{group}'")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        def update(year):
            ax.clear()
            if year is not None:
                df_year = df_group[df_group["Year"] == year]
            else:
                df_year = df_group.copy()

            for _, row in df_year.iterrows():
                ax.scatter(row["Impact"], row["Probabilité"],
                           color=risk_palette[row["Risk_Class"]],
                           edgecolor="black", s=150, alpha=0.8)
                ax.text(row["Impact"] + 0.05, row["Probabilité"] + 0.01,
                        row["Country"], fontsize=8)

            titre = f"Matrice Impact–Probabilité – {group}"
            if year is not None:
                titre += f" ({year})"
            ax.set_title(titre, fontsize=14, weight="bold")
            ax.set_xlabel("Impact")
            ax.set_ylabel("Probabilité")
            ax.set_xticks([1,2,3])
            ax.set_xticklabels(["Faible","Moyen","Élevé"])
            ax.set_ylim(0,1.05)
            ax.grid(True, linestyle="--", alpha=0.5)

            legend_elements = [Patch(facecolor=color, edgecolor='black', label=risk)
                               for risk, color in risk_palette.items()]
            ax.legend(handles=legend_elements, title="Classe de Risque")

        anim = FuncAnimation(fig, update, frames=years, interval=interval, repeat=True)

        save_path = os.path.join(save_folder, f"{group.replace(' ', '_')}_institutional_risk.gif")
        anim.save(save_path, writer="pillow", dpi=150)
        print(f"✅ Animation sauvegardée pour le groupe '{group}': {save_path}")

        if idx_g < max_show:
            plt.show()
        else:
            plt.close(fig)



def plot_institutional_risk_matrices(df, group_dict, save_path, year=None,
                                     prob_col="SRI", max_show=3):
    """
    Trace et sauvegarde les matrices Impact × Probabilité pour chaque groupe.
    Permet de sélectionner une année spécifique si le DataFrame contient une colonne 'Year'.

    Paramètres
    ----------
    df : pd.DataFrame
        Contient au moins ['Pays', prob_col, 'Risk_Class']. Optionnel : 'Year'.
        'Risk_Class' doit être labellisé "Faible", "Moyen", "Élevé".
    group_dict : dict
        Dictionnaire {Groupe: [liste de pays]}.
    save_path : str
        Dossier où sauvegarder les figures.
    year : int ou str, optionnel
        Année à sélectionner. Si None, toutes les années sont utilisées.
    prob_col : str
        Colonne représentant la probabilité.
    max_show : int
        Nombre maximum de figures affichées à l’écran.
    """
    os.makedirs(save_path, exist_ok=True)

    df_plot = df.copy()

    # Filtrer par année si spécifié
    if year is not None:
        if 'Year' not in df_plot.columns:
            raise ValueError("La colonne 'Year' n'existe pas dans le DataFrame.")
        df_plot = df_plot[df_plot["Year"] == year]
        if df_plot.empty:
            print(f"⚠️ Aucun résultat pour l'année {year}.")
            return

    # Mapping Impact texte → valeur numérique
    impact_map = {"Faible": 1, "Moyen": 2, "Elevé": 3}
    risk_palette = {"Faible": "green", "Moyen": "orange", "Elevé": "red"}

    df_plot["Impact"] = df_plot["Risk_Class"].map(impact_map)
    df_plot["Probabilité"] = df_plot[prob_col] / 100.0  # Normalisation

    for idx, (group, pays_list) in enumerate(group_dict.items()):
        df_group = df_plot[df_plot["Country"].isin(pays_list)].copy()
        if df_group.empty:
            print(f"⚠️ Aucun résultat pour le groupe '{group}' (année {year})")
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot
        for _, row in df_group.iterrows():
            ax.scatter(row["Impact"], row["Probabilité"], 
                       color=risk_palette[row["Risk_Class"]],
                       edgecolor="black", s=150, alpha=0.8)
            # Annoter le pays
            ax.text(row["Impact"] + 0.05, row["Probabilité"] + 0.01, row["Country"], fontsize=8)

        # Axes et titres
        titre = f"Matrice Impact–Probabilité – {group}"
        if year is not None:
            titre += f" ({year})"
        ax.set_title(titre, fontsize=14, weight="bold")
        ax.set_xlabel("Impact")
        ax.set_ylabel("Probabilité")
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(["Faible", "Moyen", "Elevé"])
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Légende
        legend_elements = [Patch(facecolor=color, edgecolor='black', label=risk)
                           for risk, color in risk_palette.items()]
        ax.legend(handles=legend_elements, title="Classe de Risque")

        # Sauvegarde
        filename = os.path.join(save_path, f"risk_matrix_{group.replace(' ', '_')}_{year if year else 'all'}.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✅ Figure sauvegardée : {filename}")

        # Affichage limité
        if idx < max_show:
            plt.show()
        else:
            plt.close(fig)



def plot_institutional_risk_matrices_group(df, group_dict, save_path, year=None,
                           prob_col="SRI", 
                           agg_func='mean', max_show=3):
    """
    Trace et sauvegarde une matrice Impact × Probabilité par groupe.
    Les valeurs Probabilité et Impact sont agrégées sur les pays du groupe.

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir ['Pays', prob_col, 'Risk_Class']. Optionnel : 'Year'.
    group_dict : dict
        Dictionnaire {Groupe: [liste de pays]}.
    save_path : str
        Dossier pour sauvegarder la figure.
    year : int ou None
        Année à filtrer. Si None, toutes les années sont utilisées.
    prob_col : str
        Colonne représentant la probabilité.
    agg_func : str
        Fonction d’agrégation pour Probabilité et Impact ('mean' ou 'median').
    max_show : int
        Nombre maximum de figures affichées à l’écran.
    """
    os.makedirs(save_path, exist_ok=True)

    df_plot = df.copy()

    # Filtrer par année si spécifié
    if year is not None:
        if 'Year' not in df_plot.columns:
            raise ValueError("La colonne 'Year' n'existe pas dans le DataFrame.")
        df_plot = df_plot[df_plot["Year"] == year]
        if df_plot.empty:
            print(f"⚠️ Aucun résultat pour l'année {year}.")
            return

    # Mapping Impact texte → valeur numérique
    impact_map = {"Faible": 1, "Moyen": 2, "Elevé": 3}
    risk_palette = {"Faible": "green", "Moyen": "orange", "Elevé": "red"}
    df_plot["Impact_num"] = df_plot["Risk_Class"].map(impact_map)
    df_plot["Probabilité"] = df_plot[prob_col] / 100.0  # Normalisation

    # Agréger par groupe
    group_data = []
    for group, pays_list in group_dict.items():
        df_group = df_plot[df_plot["Country"].isin(pays_list)]
        if df_group.empty:
            continue

        if agg_func == 'mean':
            prob_agg = df_group["Probabilité"].mean()
            impact_agg = df_group["Impact_num"].mean()
        elif agg_func == 'median':
            prob_agg = df_group["Probabilité"].median()
            impact_agg = df_group["Impact_num"].median()
        else:
            raise ValueError("agg_func doit être 'mean' ou 'median'.")

        # Déterminer classe de risque majoritaire
        impact_class = df_group["Risk_Class"].mode()[0]

        group_data.append({
            "Groupe": group,
            "Probabilité": prob_agg,
            "Impact": impact_agg,
            "Classe": impact_class
        })

    if not group_data:
        print("⚠️ Aucun groupe disponible pour le graphique.")
        return

    df_group_plot = pd.DataFrame(group_data)

    # --- Tracé ---
    fig, ax = plt.subplots(figsize=(10,6))
    for _, row in df_group_plot.iterrows():
        ax.scatter(row["Impact"], row["Probabilité"], 
                   color=risk_palette[row["Classe"]],
                   edgecolor="black", s=300, alpha=0.8)
        ax.text(row["Impact"] + 0.05, row["Probabilité"] + 0.01, row["Groupe"], fontsize=10)

    # Axes et titres
    titre = "Matrice Impact–Probabilité par Groupe"
    if year is not None:
        titre += f" ({year})"
    ax.set_title(titre, fontsize=14, weight="bold")
    ax.set_xlabel("Impact")
    ax.set_ylabel("Probabilité")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Faible", "Moyen", "Elevé"])
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Légende
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=risk)
                       for risk, color in risk_palette.items()]
    ax.legend(handles=legend_elements, title="Classe de Risque")

    # Sauvegarde
    filename = os.path.join(save_path, f"group_risk_matrix_{year if year else 'all'}.png")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Figure sauvegardée : {filename}")

    plt.show()




def animate_group_risk_matrix_institu(df, group_dict, save_folder="animations_group",
                              years_to_plot=None, agg_func='mean', interval=1000,
                              max_show=3, prob_col="SRI"):
    """
    Crée une animation montrant l'évolution des matrices Impact × Probabilité par groupe (valeurs agrégées).

    Paramètres
    ----------
    df : pd.DataFrame
        Contient ['Country','Risk_Class', prob_col]. Optionnel : 'Year'.
    group_dict : dict
        Dictionnaire {Groupe: [liste de pays]}.
    save_folder : str
        Dossier pour sauvegarder les GIFs.
    years_to_plot : list[int], optionnel
        Liste des années à inclure. Si None, toutes les années présentes sont utilisées.
    agg_func : str
        Fonction d’agrégation pour Probabilité et Impact ('mean' ou 'median').
    interval : int
        Intervalle entre les frames en ms.
    max_show : int
        Nombre maximum d'animations affichées à l’écran.
    prob_col : str
        Colonne représentant la probabilité.
    """
    os.makedirs(save_folder, exist_ok=True)

    df_plot = df.copy()

    # Déterminer années
    if 'Year' in df_plot.columns:
        if years_to_plot is None:
            years = sorted(df_plot['Year'].unique())
        else:
            years = [y for y in years_to_plot if y in df_plot['Year'].unique()]
    else:
        years = [None]

    impact_map = {"Faible": 1, "Moyen": 2, "Elevé": 3}
    risk_palette = {"Faible": "green", "Moyen": "orange", "Elevé": "red"}

    df_plot["Impact_num"] = df_plot["Risk_Class"].map(impact_map)
    df_plot["Probabilité"] = df_plot[prob_col] / 100.0

    for idx_g, (group, pays_list) in enumerate(group_dict.items()):
        df_group = df_plot[df_plot["Country"].isin(pays_list)].copy()
        if df_group.empty:
            print(f"⚠️ Aucun résultat pour le groupe '{group}'")
            continue

        fig, ax = plt.subplots(figsize=(10,6))

        def update(year):
            ax.clear()
            if year is not None:
                df_year = df_group[df_group["Year"] == year]
            else:
                df_year = df_group.copy()

            group_data = []
            for g, pays_list_inner in group_dict.items():
                df_g = df_year[df_year["Country"].isin(pays_list_inner)]
                if df_g.empty:
                    continue
                if agg_func=='mean':
                    prob_agg = df_g["Probabilité"].mean()
                    impact_agg = df_g["Impact_num"].mean()
                elif agg_func=='median':
                    prob_agg = df_g["Probabilité"].median()
                    impact_agg = df_g["Impact_num"].median()
                else:
                    raise ValueError("agg_func doit être 'mean' ou 'median'")
                impact_class = df_g["Risk_Class"].mode()[0]
                group_data.append({"Groupe": g, "Probabilité": prob_agg,
                                   "Impact": impact_agg, "Classe": impact_class})

            df_group_plot = pd.DataFrame(group_data)
            for _, row in df_group_plot.iterrows():
                ax.scatter(row["Impact"], row["Probabilité"],
                           color=risk_palette[row["Classe"]],
                           edgecolor="black", s=300, alpha=0.8)
                ax.text(row["Impact"] + 0.05, row["Probabilité"] + 0.01,
                        row["Groupe"], fontsize=10)

            titre = "Matrice Impact–Probabilité par Groupe"
            if year is not None:
                titre += f" ({year})"
            ax.set_title(titre, fontsize=14, weight="bold")
            ax.set_xlabel("Impact")
            ax.set_ylabel("Probabilité")
            ax.set_xticks([1,2,3])
            ax.set_xticklabels(["Faible","Moyen","Elevé"])
            ax.set_ylim(0,1.05)
            ax.grid(True, linestyle="--", alpha=0.5)

            legend_elements = [Patch(facecolor=color, edgecolor='black', label=risk)
                               for risk, color in risk_palette.items()]
            ax.legend(handles=legend_elements, title="Classe de Risque")

        anim = FuncAnimation(fig, update, frames=years, interval=interval, repeat=True)

        save_path = os.path.join(save_folder, f"{group.replace(' ', '_')}_group_risk.gif")
        anim.save(save_path, writer="pillow", dpi=150)
        print(f"✅ Animation sauvegardée pour le groupe '{group}': {save_path}")

        if idx_g < max_show:
            plt.show()
        else:
            plt.close(fig)





