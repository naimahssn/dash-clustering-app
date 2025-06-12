import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import base64
import logging
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
sbert_model = SentenceTransformer('distiluse-base-multilingual-cased')

def handle_file_upload(contenu: str, nom_fichier: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Gère l'upload de fichiers (CSV/Excel avec gestion multi-feuilles)
    Args:
        contenu: Contenu encodé en base64
        nom_fichier: Nom du fichier uploadé
    Returns:
        Tuple: (Dictionnaire des feuilles/DataFrames, message d'erreur)
    """
    try:
        type_contenu, chaine_contenu = contenu.split(',')
        decode = base64.b64decode(chaine_contenu)

        if 'csv' in nom_fichier:
            df = pd.read_csv(StringIO(decode.decode('utf-8')))
            df = nettoyer_donnees_brutes(df)
            return {'feuilles': ['Feuille1'], 'dataframes': {'Feuille1': df}}, None

        elif 'xls' in nom_fichier:
            xls = pd.ExcelFile(BytesIO(decode))
            return {
                feuille: nettoyer_donnees_brutes(xls.parse(feuille)).to_dict()
                for feuille in xls.sheet_names
            }, None

        else:
            return None, "Format de fichier non supporté (CSV/Excel uniquement)"

    except Exception as e:
        logger.error(f"Erreur d'upload : {str(e)}")
        return None, f"Erreur de traitement : {str(e)}"

def nettoyer_donnees_brutes(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données brutes avant traitement"""
    # Suppression des colonnes entièrement vides
    df = df.dropna(axis=1, how='all')

    # Conversion et nettoyage des types
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='ignore').fillna(df[col])
            df[col] = df[col].str.strip()

    return df

def process_financial_data(df: pd.DataFrame, col_initial: str, col_engage: str) -> pd.DataFrame:
    """
    Calcule les indicateurs financiers principaux
    Args:
        df: DataFrame source
        col_initial: Colonne du montant initial
        col_engage: Colonne du montant engagé
    Returns:
        DataFrame enrichi avec les indicateurs
    """
    df_propre = df.copy()

    montant_initial = pd.to_numeric(df_propre[col_initial], errors='coerce').fillna(0).round(2)
    montant_engage = pd.to_numeric(df_propre[col_engage], errors='coerce').fillna(0).round(2)

    # Calcul des indicateurs
    df_propre['ECART D\'ENGAGEMENT'] = montant_engage- montant_initial
    df_propre['TAUX_ENGAGEMENT'] = np.where(
        montant_initial > 0,
        (montant_engage / montant_initial) * 100,
        0
    )
    df_propre['ENGAGEMENT_STATUS'] = df_propre['TAUX_ENGAGEMENT'].apply(
        lambda x: 'Sous-engagé' if x < 100 else ('Normal' if x == 100 else 'Sur-engagé')
    )

    return df_propre

def normaliser_donnees(df: pd.DataFrame, cols_numeriques: list, cols_categorielles: list) -> Tuple[pd.DataFrame, dict]:
    """
    Normalise les données pour le machine learning
    Args:
        df: DataFrame à normaliser
        cols_numeriques: Colonnes numériques àscaler
        cols_categorielles: Colonnes catégorielles à encoder
    Returns:
        Tuple: (DataFrame normalisé, encodeurs)
    """
    normaliseurs = {}
    encodeurs = {}
    df_normalise = df.copy()

    if cols_numeriques:
        scaler = MinMaxScaler()
        df_normalise[cols_numeriques] = scaler.fit_transform(df_normalise[cols_numeriques])
        normaliseurs['numerique'] = scaler

    for col in cols_categorielles:
        if col in df_normalise.columns:
            le = LabelEncoder()
            df_normalise[col] = le.fit_transform(df_normalise[col].astype(str))
            encodeurs[col] = le

    return df_normalise, {'normaliseurs': normaliseurs, 'encodeurs': encodeurs}

def detect_column_types(df: pd.DataFrame) -> Dict[str, list]:
    """
    Détecte automatiquement les types de colonnes
    Returns:
        Dict: {
            'numerique': [...],
            'categoriel': [...],
            'texte': [...],
            'date': [...]
        }
    """
    types = {
        'numerique': [],
        'categoriel': [],
        'texte': [],
        'date': []
    }

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types['date'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            types['numerique'].append(col)
        else:
            ratio_unique = df[col].nunique() / len(df[col])
            types['categoriel' if ratio_unique < 0.1 else 'texte'].append(col)

    return types

def calculer_kpis(df: pd.DataFrame, col_initial: str, col_engage: str) -> Dict[str, float]:
    """Calcule les indicateurs clés de performance"""
    montant_initial = pd.to_numeric(df[col_initial], errors='coerce').fillna(0)
    montant_engage = pd.to_numeric(df[col_engage], errors='coerce').fillna(0)

    taux_engagement = np.where(
        montant_initial > 0,
        (montant_engage / montant_initial) * 100,
        0
    )

    return {
        'total_projets': len(df),
        'total_initial': montant_initial.sum(),
        'total_engage': montant_engage.sum(),
        'engagement_moyen': taux_engagement[montant_initial > 0].mean(),
        'variance_totale': (montant_engage- montant_initial).sum(),
        'sur_engagement': (taux_engagement > 100).sum()
    }

def preparer_deep_clustering(df):
    cols_numeriques = ['Montant Initial', 'Montant engagé', 'ECART D\'ENGAGEMENT', 'TAUX_ENGAGEMENT']
    cols_categorielles = ['Classification 1','Classification 3 ','Classification 2','Statut AE', 'ENGAGEMENT_STATUS']
    texte = ['Description Projet','Designation AE']

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[cols_numeriques])

    encodeur = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = encodeur.fit_transform(df[cols_categorielles])

    X_texte = sbert_model.encode(df[texte].fillna('').astype(str).tolist())

    return np.concatenate([X_num, X_cat, X_texte], axis=1)

def preparer_export(df: pd.DataFrame, format: str = 'excel') -> BytesIO:
    """
    Prépare l'export des données
    Args:
        format: 'excel' ou 'csv'
    Returns:
        Flux mémoire des données exportées
    """
    sortie = BytesIO()

    if format == 'excel':
        with pd.ExcelWriter(sortie, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Données')
    else:
        sortie.write(df.to_csv(index=False).encode('utf-8'))

    sortie.seek(0)
    return sortie