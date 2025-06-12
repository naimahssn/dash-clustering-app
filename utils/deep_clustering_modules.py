import pandas as pd
import numpy as np
import plotly.express as px
import logging

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# Keras/TensorFlow imports
from keras.layers import Input, Dense
from keras.models import Model

# UMAP import
import umap.umap_ as umap

# NLTK for stopwords (ensure you have downloaded 'stopwords' corpus: nltk.download('stopwords'))
import nltk
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK stopwords if not already downloaded
try:
    french_stopwords = stopwords.words('french')
except LookupError:
    nltk.download('stopwords')
    french_stopwords = stopwords.words('french')


# =================================================================================
# HELPER FUNCTIONS (Ces fonctions peuvent être utilisées indépendamment ou par la classe)
# =================================================================================

def create_empty_figure(title="Aucune donnée à afficher"):
    """
    Crée une figure Plotly vide avec un message spécifié.
    Utilisé pour afficher des graphiques d'erreur ou d'absence de données.
    """
    fig = px.scatter(title=title)
    fig.update_layout(
        xaxis={'visible': False, 'showgrid': False},
        yaxis={'visible': False, 'showgrid': False},
        annotations=[
            dict(
                text="Pas de données ou erreur lors du traitement.",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
        ]
    )
    return fig

def convert_dates(date_series):
    """
    Convertit une série de dates en caractéristiques numériques (année, mois, jour, jour de la semaine).
    Gère les valeurs manquantes en les remplaçant par -1.
    """
    try:
        # Squeeze pour s'assurer que c'est une série 1D
        dates = pd.to_datetime(date_series.squeeze(), errors='coerce')
        return np.c_[
            dates.dt.year.fillna(-1).astype(int),
            dates.dt.month.fillna(-1).astype(int),
            dates.dt.day.fillna(-1).astype(int),
            dates.dt.dayofweek.fillna(-1).astype(int)
        ]
    except Exception as e:
        logger.error(f"Échec conversion dates: {e}", exc_info=True)
        raise ValueError("Format de date invalide pour la conversion.") from e

def process_mixed_features(df, feature_types):
    """
    Traite un DataFrame en appliquant des préprocesseurs spécifiques
    en fonction des types de caractéristiques définis.
    Gère les colonnes numériques, catégorielles, texte et date.
    Convertit les matrices creuses en matrices denses si nécessaire.
    """
    processors = {
        'numerique': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]),
        'categoriel': Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='manquant')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]),
        'texte': Pipeline([
            ('cleaner', FunctionTransformer(
                lambda x: x.fillna('').astype(str).values.ravel(), validate=False
            )),
            ('vectorizer', TfidfVectorizer(
                max_features=100,
                token_pattern=r'(?u)\b\w+\b',
                min_df=1,
                stop_words=french_stopwords
            )),
            ('densifier', FunctionTransformer(lambda x: x.toarray(), validate=False))
        ]),
        'date': Pipeline([
            ('converter', FunctionTransformer(convert_dates, validate=False))
        ])
    }

    processed_features = []

    for col in df.columns:
        try:
            col_type = feature_types.get(col, 'texte')

            if col_type == 'texte':
                text_data = df[col].fillna('')
                if text_data.str.strip().eq('').all():
                    raise ValueError(f"La colonne texte '{col}' est vide ou ne contient que des espaces.")

            processor = processors.get(col_type)
            if processor is None:
                raise ValueError(f"Type de caractéristique non supporté: {col_type} pour la colonne {col}")

            transformed = processor.fit_transform(df[[col]])

            if sparse.issparse(transformed):
                transformed = transformed.toarray()

            if transformed.shape[0] != len(df):
                raise ValueError(f"Dimensions incohérentes après traitement pour la colonne {col}. "
                                 f"Attendu {len(df)}, obtenu {transformed.shape[0]}.")

            processed_features.append(transformed)

        except Exception as e:
            logger.error(f"Échec du traitement de la colonne '{col}' de type '{col_type}': {e}", exc_info=True)
            raise ValueError(f"Erreur lors du traitement de la colonne '{col}': {str(e)}") from e

    try:
        return np.hstack(processed_features)
    except Exception as e:
        logger.error(f"Échec de l'assemblage final des caractéristiques traitées: {e}", exc_info=True)
        raise ValueError(f"Erreur lors de la combinaison des caractéristiques: {str(e)}") from e

def create_tsne_plot(projection, clusters, hover_data):
    """
    Crée un graphique de dispersion Plotly utilisant les projections (t-SNE ou UMAP)
    et les clusters pour la coloration. Permet d'afficher des données au survol.
    """
    df_plot = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'cluster': clusters
    })
    for col in hover_data.columns:
        df_plot[col] = hover_data[col]

    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='cluster',
        title='Projection UMAP des Clusters Profonds',
        hover_data=hover_data.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# =================================================================================
# DEEP CLUSTERING MODEL CLASS
# =================================================================================

class DeepClusteringModel:
    """
    Une classe pour effectuer le clustering profond en utilisant un autoencodeur
    pour la réduction de dimensionnalité et KMeans pour le clustering.
    Inclut la sélection automatique du nombre optimal de clusters (k)
    et la projection UMAP pour la visualisation.
    """
    def __init__(self, encoding_dim=32, autoencoder_epochs=50, autoencoder_batch_size=256,
                 kmeans_n_init=10, umap_n_components=2, umap_n_neighbors=15, umap_min_dist=0.1,
                 random_state=42):
        """
        Initialise le modèle de clustering profond.

        Args:
            encoding_dim (int): Dimension de la couche d'encodage de l'autoencodeur.
            autoencoder_epochs (int): Nombre d'epochs pour l'entraînement de l'autoencodeur.
            autoencoder_batch_size (int): Taille du batch pour l'entraînement de l'autoencodeur.
            kmeans_n_init (int): Nombre de fois que l'algorithme k-means sera exécuté avec différentes
                                 graines centroïdes.
            umap_n_components (int): Nombre de dimensions pour la projection UMAP.
            umap_n_neighbors (int): Paramètre n_neighbors pour UMAP.
            umap_min_dist (float): Paramètre min_dist pour UMAP.
            random_state (int): Graine pour la reproductibilité des résultats.
        """
        self.encoding_dim = encoding_dim
        self.autoencoder_epochs = autoencoder_epochs
        self.autoencoder_batch_size = autoencoder_batch_size
        self.kmeans_n_init = kmeans_n_init
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state

        self.autoencoder = None
        self.encoder_model = None
        self.kmeans_model = None
        self.umap_reducer = None
        self.optimal_k = None

    def _build_autoencoder(self, input_dim):
        """
        Construit et compile le modèle d'autoencodeur et le modèle d'encodeur.

        Args:
            input_dim (int): Dimension des données d'entrée.

        Returns:
            tuple: (autoencodeur, encodeur_model)
        """
        # Assurer que encoding_dim est au moins 2 et pas plus grand que input_dim
        effective_encoding_dim = max(2, min(self.encoding_dim, input_dim // 2))

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(effective_encoding_dim * 2, activation='relu')(input_layer)
        encoder = Dense(effective_encoding_dim, activation='relu')(encoder) # Couche d'encodage

        decoder = Dense(effective_encoding_dim * 2, activation='relu')(encoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder) # Utiliser sigmoid pour des données normalisées [0,1]

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        encoder_model = Model(inputs=input_layer, outputs=encoder) # Modèle pour obtenir les embeddings

        autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder = autoencoder
        self.encoder_model = encoder_model
        logger.info(f"Autoencodeur construit avec input_dim={input_dim}, encoding_dim={effective_encoding_dim}")

    def _select_optimal_k(self, embeddings):
        """
        Sélectionne le nombre optimal de clusters (k) en utilisant une combinaison
        des scores Silhouette, Calinski-Harabasz et Davies-Bouldin.

        Args:
            embeddings (np.array): Les caractéristiques encodées par l'autoencodeur.

        Returns:
            int: Le nombre optimal de clusters.
        """
        max_k = min(100, len(embeddings) - 1)
        if max_k < 2:
            logger.warning("Pas assez de données pour effectuer un clustering (moins de 2 échantillons). Retourne k=1.")
            return 1 # Ou lever une erreur si le clustering n'est pas possible

        scores = {'silhouette': {}, 'calinski': {}, 'davies': {}}

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.kmeans_n_init)
            labels = kmeans.fit_predict(embeddings)
            if len(np.unique(labels)) < 2:
                # Si KMeans ne parvient pas à créer au moins 2 clusters distincts, ignorer ce k
                continue

            scores['silhouette'][k] = silhouette_score(embeddings, labels)
            scores['calinski'][k] = calinski_harabasz_score(embeddings, labels)
            scores['davies'][k] = -davies_bouldin_score(embeddings, labels) # Inversé pour que "plus grand = mieux"

        votes = {}
        for metric in scores:
            if scores[metric]:
                best_k_for_metric = max(scores[metric], key=scores[metric].get)
                votes[best_k_for_metric] = votes.get(best_k_for_metric, 0) + 1

        optimal_k = max(votes, key=votes.get) if votes else 2
        logger.info(f"Nombre optimal de clusters (k) sélectionné: {optimal_k}")
        return optimal_k

    def fit_predict(self, df_original, selected_features, feature_types):
        """
        Exécute le processus complet de clustering profond.

        Args:
            df_original (pd.DataFrame): Le DataFrame original contenant toutes les données.
            selected_features (list): Liste des noms de colonnes à utiliser pour le clustering.
            feature_types (dict): Dictionnaire mappant les noms de caractéristiques à leurs types
                                  ('numerique', 'categoriel', 'texte', 'date').

        Returns:
            tuple: (result_df, umap_fig, optimal_k)
                result_df (pd.DataFrame): Le DataFrame original avec une nouvelle colonne 'DEEP_CLUSTER'.
                umap_fig (plotly.graph_objects.Figure): Le graphique de projection UMAP.
                optimal_k (int): Le nombre optimal de clusters trouvé.
        """
        if not selected_features:
            raise ValueError("Aucune caractéristique sélectionnée pour le clustering.")
        if not feature_types:
            raise ValueError("Les types de caractéristiques ne sont pas définis.")

        # 1. Préparation des données
        df_to_process = df_original[selected_features].copy()
        processed_data = process_mixed_features(df_to_process, feature_types)
        logger.info(f"Données traitées. Forme: {processed_data.shape}")

        # 2. Construction et entraînement de l'autoencodeur
        self._build_autoencoder(input_dim=processed_data.shape[1])
        logger.info("Début de l'entraînement de l'autoencodeur...")
        self.autoencoder.fit(processed_data, processed_data,
                              epochs=self.autoencoder_epochs,
                              batch_size=self.autoencoder_batch_size,
                              shuffle=True, verbose=0)
        logger.info("Autoencodeur entraîné.")

        # 3. Encodage des données
        encoded_features = self.encoder_model.predict(processed_data)
        logger.info(f"Caractéristiques encodées. Forme: {encoded_features.shape}")

        # 4. Sélection automatique de k
        self.optimal_k = self._select_optimal_k(encoded_features)
        if self.optimal_k < 2:
            # Gérer le cas où le clustering n'est pas significatif
            logger.warning("Clustering non significatif, moins de 2 clusters formés. Retourne des résultats vides.")
            result_df = df_original.copy()
            result_df['DEEP_CLUSTER'] = "Non classifié"
            umap_fig = create_empty_figure("Clustering non significatif")
            return result_df, umap_fig, self.optimal_k

        # 5. Clustering final
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=self.random_state, n_init=self.kmeans_n_init)
        clusters = self.kmeans_model.fit_predict(encoded_features)
        logger.info(f"Clustering KMeans effectué avec {self.optimal_k} clusters.")

        # 6. Projection UMAP
        self.umap_reducer = umap.UMAP(n_components=self.umap_n_components,
                                       n_neighbors=self.umap_n_neighbors,
                                       min_dist=self.umap_min_dist,
                                       random_state=self.random_state)
        umap_projection = self.umap_reducer.fit_transform(encoded_features)
        logger.info(f"Projection UMAP effectuée. Forme: {umap_projection.shape}")

        # 7. Préparation des résultats
        result_df = df_original.copy()
        result_df['DEEP_CLUSTER'] = clusters.astype(str) # Convertir en chaîne pour Plotly
        logger.info("Colonne 'DEEP_CLUSTER' ajoutée au DataFrame des résultats.")

        umap_fig = create_tsne_plot(
            projection=umap_projection,
            clusters=result_df['DEEP_CLUSTER'],
            hover_data=result_df
        )
        umap_fig.update_traces(
            marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        logger.info("Figure UMAP créée.")

        return result_df, umap_fig, self.optimal_k