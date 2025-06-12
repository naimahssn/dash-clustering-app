import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
import hdbscan
from hdbscan import validity
import plotly.express as px
import plotly.graph_objects as go # Import for interactive table
from typing import List, Optional, Dict, Union
from sklearn.metrics import silhouette_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumericalClusterAnalyzer:
    """
    Une classe pour effectuer le clustering numérique de données à l'aide de UMAP et HDBSCAN,
    avec des fonctionnalités d'analyse et de visualisation.
    La méthode de normalisation est fixée à StandardScaler (moyenne=0, écart-type=1).
    L'embedding UMAP 2D et 3D sont calculés lors de l'entraînement.
    """
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        random_state: int = 42,
        umap_params: Optional[dict] = None,
        hdbscan_params: Optional[dict] = None
    ):
        """
        Initialise l'analyseur de clusters.

        Args:
            min_cluster_size (int): La taille minimale d'un cluster pour HDBSCAN.
            min_samples (Optional[int]): Le nombre de points dans le voisinage d'un point
                                         pour qu'il soit considéré comme un point central.
                                         Par défaut, min_cluster_size.
            random_state (int): Graine aléatoire pour la reproductibilité.
            umap_params (Optional[dict]): Paramètres supplémentaires pour UMAP.
            hdbscan_params (Optional[dict]): Paramètres supplémentaires pour HDBSCAN.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        # La méthode de normalisation est fixée à 'standard' (StandardScaler)
        self.scaling_method = 'standard'
        self.random_state = random_state
        self.umap_params = umap_params or {}
        self.hdbscan_params = hdbscan_params or {}

        # Initialisation des composants et des résultats
        self.scaler: Optional[StandardScaler] = None
        self.reducer: Optional[umap.UMAP] = None # UMAP reducer for 2D embedding
        self.reducer_3d: Optional[umap.UMAP] = None # UMAP reducer for 3D embedding
        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.labels_: Optional[np.ndarray] = None
        self.embedding_2d_: Optional[np.ndarray] = None # Always store 2D embedding
        self.embedding_3d_: Optional[np.ndarray] = None # Always store 3D embedding
        self.df_: Optional[pd.DataFrame] = None # DataFrame after processing (with UMAP and CLUSTER)
        self.original_df_: Optional[pd.DataFrame] = None # Original DataFrame for categorical analysis
        self.numeric_cols_: Optional[List[str]] = None
        self._processed_data_: Optional[np.ndarray] = None # Store processed numeric data (X)
        self.cluster_stats_: Optional[pd.DataFrame] = None
        self.cluster_quality_: Optional[Dict[str, Union[float, int]]] = None
        self.color_discrete_map_: Optional[Dict[str, str]] = None # Stockage des couleurs pour les clusters

        self._validate_parameters()

    def _validate_parameters(self):
        """Valide les paramètres d'entrée de la classe."""
        if self.min_cluster_size <= 0:
            raise ValueError("`min_cluster_size` doit être un entier positif.")
        if self.min_samples is not None and self.min_samples <= 0:
            raise ValueError("`min_samples` doit être un entier positif.")

    def _initialize_scaler(self) -> StandardScaler:
        """Initialise le scaler, qui est toujours StandardScaler."""
        return StandardScaler()

    def _preprocess_data(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Prétraite les données: sélectionne les colonnes numériques, gère les NaN,
        et applique la normalisation.
        """
        self.original_df_ = df.copy()

        self.numeric_cols_ = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
        if not self.numeric_cols_:
            raise ValueError("Aucune colonne numérique valide n'a été fournie pour le clustering parmi les features.")

        missing_features = [f for f in self.numeric_cols_ if f not in self.original_df_.columns]
        if missing_features:
            raise ValueError(f"Les colonnes numériques suivantes sont manquantes dans le DataFrame fourni: {missing_features}")

        for col in self.numeric_cols_:
            if pd.api.types.is_object_dtype(self.original_df_[col]):
                try:
                    self.original_df_[col] = pd.to_numeric(self.original_df_[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"La colonne '{col}' est de type objet et ne peut pas être convertie en numérique. Elle sera ignorée pour le clustering. Erreur: {e}")
                    self.numeric_cols_.remove(col)

        if not self.numeric_cols_:
            raise ValueError("Après vérification, aucune colonne numérique valide pour le clustering.")

        X_numeric = self.original_df_[self.numeric_cols_].copy()

        if X_numeric.isna().any().any():
            logger.warning("Remplissage des NaN par la médiane dans les colonnes numériques pour le clustering.")
            median_values = X_numeric.median()
            X_numeric = X_numeric.fillna(median_values)
            self.original_df_[self.numeric_cols_] = X_numeric # Update original_df_ with imputed values

        X = X_numeric.values

        # Le scaler est toujours StandardScaler maintenant
        self.scaler = self._initialize_scaler()
        X = self.scaler.fit_transform(X)
        return X

    def fit(self, df: pd.DataFrame, features: List[str]) -> 'NumericalClusterAnalyzer':
        """
        Entraîne le modèle de clustering sur les données fournies.
        Calcule les projections UMAP 2D et 3D.

        Args:
            df (pd.DataFrame): Le DataFrame d'entrée contenant les données.
            features (List[str]): Une liste des noms de colonnes numériques à utiliser pour le clustering.

        Returns:
            NumericalClusterAnalyzer: L'instance de la classe entraînée.
        """
        logger.info("Début du prétraitement des données...")
        self._processed_data_ = self._preprocess_data(df, features) # Store processed data
        logger.info("Prétraitement terminé.")

        n_samples = len(self._processed_data_)
        if n_samples <= 1:
            raise ValueError("Le DataFrame contient trop peu d'échantillons pour effectuer le clustering.")
        n_neighbors_umap = min(15, n_samples - 1) if n_samples > 1 else 1

        # Calcul de l'embedding UMAP 2D
        default_umap_2d = {
            'n_components': 2,
            'random_state': self.random_state,
            'n_neighbors': n_neighbors_umap,
            'min_dist': 0.1,
            'metric': 'euclidean'
        }
        self.reducer = umap.UMAP(**{**default_umap_2d, **self.umap_params})
        logger.info(f"Démarrage de la réduction de dimension avec UMAP (n_components=2) pour le clustering...")
        self.embedding_2d_ = self.reducer.fit_transform(self._processed_data_)
        logger.info("Réduction de dimension UMAP 2D terminée.")

        # Calcul UMAP 3D
        logger.info("Calcul de l'embedding UMAP 3D...")
        umap_3d_params = {
            'n_components': 3,
            'random_state': self.random_state,
            'n_neighbors': n_neighbors_umap,
            'min_dist': 0.1,
            'metric': 'euclidean'
        }
        self.reducer_3d = umap.UMAP(**{**umap_3d_params, **self.umap_params})
        self.embedding_3d_ = self.reducer_3d.fit_transform(self._processed_data_)
        logger.info("Réduction de dimension UMAP 3D terminée.")

        default_hdbscan = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'cluster_selection_method': 'eom',
            'prediction_data': True,
            'gen_min_span_tree': True
        }
        self.clusterer = hdbscan.HDBSCAN(**{**default_hdbscan, **self.hdbscan_params})
        logger.info("Démarrage du clustering avec HDBSCAN...")
        self.labels_ = self.clusterer.fit_predict(self._processed_data_) # Use processed data for clustering
        logger.info("Clustering HDBSCAN terminé.")

        self.df_ = self.original_df_.copy()
        self.df_['CLUSTER'] = self.labels_.astype(str) # Convert to string here once

        self.df_['UMAP-1'] = self.embedding_2d_[:, 0]
        self.df_['UMAP-2'] = self.embedding_2d_[:, 1]
        self.df_['UMAP-3'] = self.embedding_3d_[:, 2]

        logger.info("Labels de clusters et embeddings UMAP 2D/3D ajoutés au DataFrame.")

        self._compute_cluster_stats()
        self._compute_cluster_quality(self._processed_data_) # Use processed data for quality metrics
        self._generate_cluster_colors()
        logger.info("Analyse de clustering numérique terminée avec succès.")
        return self

    def _compute_cluster_stats(self):
        """Calcule les statistiques (moyenne, médiane, écart-type, count) pour chaque cluster."""
        if self.df_ is None or 'CLUSTER' not in self.df_:
            logger.warning("Impossible de calculer les statistiques: DataFrame ou colonne CLUSTER manquante.")
            self.cluster_stats_ = None
            return

        # self.df_['CLUSTER'] is already str from fit() now
        stats = self.df_.groupby("CLUSTER")[self.numeric_cols_].agg(['mean', 'median', 'std', 'count'])
        stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
        self.cluster_stats_ = stats.reset_index()

        cluster_counts = self.df_['CLUSTER'].value_counts(normalize=True).mul(100).round(1)
        self.cluster_stats_['percentage'] = self.cluster_stats_['CLUSTER'].map(cluster_counts)
        logger.info("Statistiques par cluster calculées.")

    def _compute_cluster_quality(self, X: np.ndarray):
        """Calcule les métriques de qualité des clusters (Silhouette, DBCV)."""
        unique_labels = np.unique(self.labels_)
        labels_for_metrics = self.labels_[self.labels_ != -1]
        X_for_metrics = X[self.labels_ != -1]

        self.cluster_quality_ = {
            'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
            'noise_points': np.sum(self.labels_ == -1),
            'noise_percentage': np.mean(self.labels_ == -1) * 100
        }

        if len(np.unique(labels_for_metrics)) < 2:
            logger.warning("Moins de 2 clusters valides (hors bruit) pour calculer le score de Silhouette. DBCV non applicable non plus.")
            self.cluster_quality_['silhouette'] = np.nan
            self.cluster_quality_['dbcv'] = np.nan
        else:
            try:
                self.cluster_quality_['silhouette'] = silhouette_score(X_for_metrics, labels_for_metrics)
            except Exception as e:
                logger.warning(f"Erreur calcul du score de Silhouette: {str(e)}. Définition à NaN.")
                self.cluster_quality_['silhouette'] = np.nan
            try:
                self.cluster_quality_['dbcv'] = validity.validity_index(X, self.labels_)
            except Exception as e:
                logger.warning(f"Erreur calcul de l'indice DBCV: {str(e)}. Définition à NaN.")
                self.cluster_quality_['dbcv'] = np.nan
        logger.info("Métriques de qualité des clusters calculées.")

    def _generate_cluster_colors(self):
        """Génère une palette de couleurs pour les clusters, y compris le bruit."""
        if self.df_ is None or 'CLUSTER' not in self.df_.columns:
            logger.warning("Impossible de générer la palette de couleurs: DataFrame ou colonne CLUSTER manquante.")
            return

        clusters = self.df_['CLUSTER'].astype(str).unique()
        sorted_clusters = sorted([c for c in clusters if c != '-1'], key=int)
        if '-1' in clusters:
            sorted_clusters.append('-1')

        self.color_discrete_map_ = {}
        for i, cluster_id in enumerate([c for c in sorted_clusters if c != '-1']):
            self.color_discrete_map_[cluster_id] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        if '-1' in sorted_clusters:
            self.color_discrete_map_["-1"] = "#D3D3D3" # Light grey for noise
        logger.info("Palette de couleurs des clusters générée.")

    def plot_clusters(self, projection_dim: int = 2, hover_data: Optional[List[str]] = None, **kwargs) -> Union[px.scatter, px.scatter_3d]:
        """
        Visualisation 2D ou 3D des clusters en utilisant UMAP.

        Args:
            projection_dim (int): La dimension pour la visualisation (2 ou 3). Par défaut à 2.
            hover_data (Optional[List[str]]): Colonnes supplémentaires à afficher au survol.
            **kwargs: Arguments supplémentaires pour plotly express (ex: title, size, opacity).
        Returns:
            Union[px.scatter, px.scatter_3d]: Figure Plotly des clusters.
        """
        if self.df_ is None or 'CLUSTER' not in self.df_.columns:
            raise ValueError("Aucune donnée de clustering disponible. Exécutez 'fit' d'abord.")
        if self.color_discrete_map_ is None:
            self._generate_cluster_colors()
        if projection_dim not in [2, 3]:
            raise ValueError("`projection_dim` pour le plot doit être 2 ou 3.")

        umap_x_col, umap_y_col, umap_z_col = 'UMAP-1', 'UMAP-2', 'UMAP-3' # Column names for plot

        if projection_dim == 2:
            if self.embedding_2d_ is None:
                raise ValueError("L'embedding UMAP 2D n'a pas été calculé. Exécutez `fit`.")
            # Ensure UMAP-1 and UMAP-2 are correctly set from 2D embedding
            self.df_['UMAP-1'] = self.embedding_2d_[:, 0]
            self.df_['UMAP-2'] = self.embedding_2d_[:, 1]
            logger.info("Préparation du plot UMAP 2D.")
        else: # projection_dim == 3
            if self.embedding_3d_ is None:
                # This case should ideally not be hit if fit always calculates 3D.
                # If it is hit, it means fit failed or wasn't called.
                raise ValueError("L'embedding UMAP 3D n'a pas été calculé. Exécutez `fit`.")
            # Ensure UMAP-1, UMAP-2, UMAP-3 are correctly set from 3D embedding
            self.df_['UMAP-1'] = self.embedding_3d_[:, 0]
            self.df_['UMAP-2'] = self.embedding_3d_[:, 1]
            self.df_['UMAP-3'] = self.embedding_3d_[:, 2]
            logger.info("Préparation du plot UMAP 3D.")

        hover_data = hover_data or []
        # Add UMAP coordinates to hover data if not already present
        if umap_x_col not in hover_data: hover_data.append(umap_x_col)
        if umap_y_col not in hover_data: hover_data.append(umap_y_col)
        if projection_dim == 3 and umap_z_col not in hover_data: hover_data.append(umap_z_col)

        fig_args = {
            'data_frame': self.df_,
            'color': 'CLUSTER',
            'color_discrete_map': self.color_discrete_map_,
            'hover_data': hover_data,
            'title': f'Visualisation des Clusters (UMAP - {projection_dim}D)',
            **kwargs
        }

        if projection_dim == 3:
            fig = px.scatter_3d(**fig_args, x=umap_x_col, y=umap_y_col, z=umap_z_col)
        else:
            fig = px.scatter(**fig_args, x=umap_x_col, y=umap_y_col)

        fig.update_traces(
            marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        fig.update_layout(legend_title_text='Cluster')
        logger.info(f"Figure des clusters UMAP {projection_dim}D générée.")
        return fig

    def plot_feature_distributions(self, features: Optional[List[str]] = None, **kwargs) -> px.box:
        """
        Boîtes à moustaches des features numériques par cluster.
        """
        if self.df_ is None or 'CLUSTER' not in self.df_.columns:
            raise ValueError("Données de clustering non disponibles. Exécutez 'fit' d'abord.")
        if self.color_discrete_map_ is None:
            self._generate_cluster_colors()

        features_to_plot = features or self.numeric_cols_
        if not features_to_plot:
            raise ValueError("Aucune caractéristique numérique à afficher.")

        # Filter for features that are actually in the DataFrame and are numeric
        features_to_plot = [f for f in features_to_plot if f in self.df_.columns and pd.api.types.is_numeric_dtype(self.df_[f])]
        if not features_to_plot:
            raise ValueError("Aucune caractéristique numérique valide à afficher parmi celles fournies.")

        melted_df = self.df_.melt(id_vars="CLUSTER", value_vars=features_to_plot, var_name="Caractéristique", value_name="Valeur")

        # Sort clusters for consistent plotting order (noise last)
        melted_df['CLUSTER_sort'] = melted_df['CLUSTER'].apply(lambda x: float(x) if x != '-1' else np.inf)
        melted_df = melted_df.sort_values(by='CLUSTER_sort')

        fig = px.box(
            melted_df,
            x="Caractéristique", y="Valeur", color="CLUSTER",
            title="Distribution des caractéristiques numériques par cluster",
            labels={"Caractéristique": "Caractéristique Numérique", "Valeur": "Valeur de la Caractéristique"},
            points="outliers", # Show individual outlier points
            color_discrete_map=self.color_discrete_map_,
            **kwargs
        )
        fig.update_layout(
            xaxis_title="Caractéristique",
            yaxis_title="Valeur",
            boxmode="group", # Group boxes by feature
            legend_title_text='Cluster'
        )
        logger.info("Figure des distributions de caractéristiques générée.")
        return fig

    def plot_feature_importance(self, top_n: int = 10) -> px.bar:
        """
        Visualisation de l'importance des features pour le clustering, basée sur la variance inter-cluster.
        Cette métrique est un ratio de la variance des moyennes de cluster à la moyenne des écarts-types intra-cluster.
        """
        if self.cluster_stats_ is None:
            raise ValueError("Les statistiques des clusters ne sont pas disponibles. Exécutez 'fit' d'abord.")

        importance = {}
        # Exclude noise cluster (-1) for importance calculation
        cluster_stats_non_noise = self.cluster_stats_[self.cluster_stats_['CLUSTER'] != '-1']

        if cluster_stats_non_noise.empty:
            logger.warning("Aucun cluster valide (non-bruit) pour calculer l'importance des caractéristiques.")
            return px.bar(title="Aucune caractéristique discriminante trouvée (pas de clusters valides).")

        for col in self.numeric_cols_:
            cluster_means = cluster_stats_non_noise[f'{col}_mean']
            cluster_stds = cluster_stats_non_noise[f'{col}_std']

            # Avoid division by zero if all stds are zero (feature is constant within clusters)
            mean_std = cluster_stds.mean()
            if not cluster_means.empty and mean_std > 1e-9: # Add a small epsilon to avoid near-zero division
                importance[col] = cluster_means.std() / mean_std
            else:
                importance[col] = 0 # Feature is not discriminant or constant

        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False).head(top_n)

        if importance_df.empty:
            return px.bar(title="Aucune caractéristique discriminante trouvée.")

        fig = px.bar(
            importance_df,
            x='importance', y='feature',
            orientation='h',
            title=f"Top {top_n} des caractéristiques discriminantes entre clusters",
            labels={'importance': 'Importance (Écart relatif)', 'feature': 'Caractéristique'},
            color_discrete_sequence=px.colors.qualitative.Plotly # Use Plotly's default qualitative colors
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort features by importance on y-axis
        logger.info("Figure des caractéristiques importantes générée.")
        return fig

    def plot_categorical_distribution_for_selection(
        self,
        df_clustered: pd.DataFrame,
        categorical_column: str,
        selected_cluster: Union[str, int],
        color_discrete_map: Optional[Dict[str, str]] = None
    ) -> px.bar:
        """
        Visualise la distribution d'une feature catégorielle spécifique au sein d'un cluster donné.

        Args:
            df_clustered (pd.DataFrame): Le DataFrame contenant les données clusterisées (avec la colonne 'CLUSTER').
            categorical_column (str): Le nom de la colonne catégorielle à visualiser.
            selected_cluster (Union[str, int]): L'identifiant du cluster à analyser.
            color_discrete_map (Optional[Dict[str, str]]): Une carte des couleurs pour les clusters.

        Returns:
            px.bar: Une figure Plotly Express représentant la distribution.
        """
        if 'CLUSTER' not in df_clustered.columns:
            raise ValueError("Le DataFrame fourni ne contient pas la colonne 'CLUSTER'.")
        if categorical_column not in df_clustered.columns:
            raise ValueError(f"La colonne '{categorical_column}' n'est pas trouvée dans le DataFrame.")
        if pd.api.types.is_numeric_dtype(df_clustered[categorical_column]):
            raise ValueError(f"La colonne '{categorical_column}' est numérique. Ce plot est pour les données catégorielles.")

        # S'assurer que la colonne CLUSTER est de type string pour la comparaison
        df_clustered['CLUSTER'] = df_clustered['CLUSTER'].astype(str)
        selected_cluster_str = str(selected_cluster)

        # Filtrer le DataFrame pour le cluster sélectionné
        df_filtered = df_clustered[df_clustered['CLUSTER'] == selected_cluster_str].copy()

        if df_filtered.empty:
            logger.warning(f"Aucune donnée pour le cluster '{selected_cluster_str}'. Retourne une figure vide.")
            return px.bar(title=f"Aucune donnée pour le cluster '{selected_cluster_str}'")

        # Gérer les NaN dans la colonne catégorielle
        if df_filtered[categorical_column].isnull().any():
            logger.info(f"Remplissage des NaN dans la colonne catégorielle '{categorical_column}' par 'NaN_Category'.")
            df_filtered[categorical_column] = df_filtered[categorical_column].fillna('NaN_Category')

        # Calculer la distribution des catégories dans le cluster sélectionné
        category_counts = df_filtered[categorical_column].value_counts(normalize=True).mul(100).reset_index()
        category_counts.columns = [categorical_column, 'percentage']

        fig = px.bar(
            category_counts,
            x=categorical_column,
            y='percentage',
            title=f"Distribution de '{categorical_column}' dans le Cluster {selected_cluster_str}",
            labels={categorical_column: 'Catégorie', 'percentage': 'Pourcentage (%)'},
            color=categorical_column, # Couleur par catégorie
            color_discrete_map=color_discrete_map # Utilisez la carte des couleurs si fournie, sinon Plotly par défaut
        )
        fig.update_layout(yaxis_title="Pourcentage dans le Cluster")
        logger.info(f"Figure de distribution catégorielle pour '{categorical_column}' dans le cluster '{selected_cluster_str}' générée.")
        return fig

    def plot_cluster_sizes(self) -> px.bar:
        """
        Visualise la taille (nombre de points) de chaque cluster, y compris le bruit.
        """
        if self.df_ is None or 'CLUSTER' not in self.df_.columns:
            raise ValueError("Données de clustering non disponibles. Exécutez 'fit' d'abord.")
        if self.color_discrete_map_ is None:
            self._generate_cluster_colors()

        # Calculate value counts for the 'CLUSTER' column
        cluster_counts = self.df_['CLUSTER'].value_counts().reset_index()
        cluster_counts.columns = ['CLUSTER', 'count']
        cluster_counts['percentage'] = (cluster_counts['count'] / len(self.df_)) * 100

        # Sort clusters for consistent plotting order (noise last)
        cluster_counts['sort_key'] = cluster_counts['CLUSTER'].apply(
            lambda x: float(x) if x != '-1' else float('inf')
        )
        cluster_counts = cluster_counts.sort_values(by='sort_key').drop('sort_key', axis=1)

        fig = px.bar(
            cluster_counts,
            x='CLUSTER',
            y='count',
            color='CLUSTER',
            color_discrete_map=self.color_discrete_map_,
            text=cluster_counts['percentage'].apply(lambda x: f'{x:.1f}%'), # Display percentage on bars
            title="Taille des Clusters (y compris les points de bruit)",
            labels={'CLUSTER': 'Numéro de Cluster', 'count': 'Nombre de points'},
            height=500
        )
        fig.update_traces(textposition='outside') # Position text labels outside the bars
        fig.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Nombre de points",
            uniformtext_minsize=8, # Minimum font size for text labels
            uniformtext_mode='hide', # Hide text if it doesn't fit
            hovermode="x unified", # Show hover info for all bars at a given x-coordinate
            legend_title_text='Cluster'
        )
        logger.info("Figure des tailles de clusters générée.")
        return fig

    def display_cluster_stats_table(self) -> go.Figure:
        """
        Affiche un tableau interactif des statistiques des clusters (moyenne, médiane, std, count, pourcentage).
        """
        if self.cluster_stats_ is None:
            raise ValueError("Les statistiques des clusters ne sont pas disponibles. Exécutez 'fit' d'abord.")

        # Create a copy to avoid modifying the original stats DataFrame
        display_df = self.cluster_stats_.copy()

        # Sort clusters for consistent display, with noise cluster (-1) at the end
        display_df['sort_key'] = display_df['CLUSTER'].apply(
            lambda x: float(x) if x != '-1' else float('inf')
        )
        display_df = display_df.sort_values(by='sort_key').drop('sort_key', axis=1)

        # Prepare headers and cells for the Plotly table
        header_values = ['Cluster', 'Pourcentage (%)']
        cell_values = [display_df['CLUSTER'], display_df['percentage'].apply(lambda x: f'{x:.1f}')]

        # Add statistics for each numeric column
        for col in self.numeric_cols_:
            header_values.extend([f'{col} (Moyenne)', f'{col} (Médiane)', f'{col} (Écart-type)', f'{col} (Count)'])
            cell_values.extend([
                display_df[f'{col}_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A'),
                display_df[f'{col}_median'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A'),
                display_df[f'{col}_std'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A'),
                display_df[f'{col}_count'].astype(int).astype(str) # Ensure count is integer and then string
            ])

        # Create the Plotly table
        table = go.Table(
            header=dict(
                values=header_values,
                fill_color='paleturquoise',
                align='left',
                font=dict(color='black', size=12)
            ),
            cells=dict(
                values=cell_values,
                fill_color='lavender',
                align='left',
                font=dict(color='black', size=11)
            )
        )

        fig = go.Figure(data=[table])
        fig.update_layout(title_text="Statistiques des Clusters", height=600)
        logger.info("Tableau des statistiques des clusters généré.")
        return fig

    def get_clustered_data(self) -> Optional[pd.DataFrame]:
        """Retourne le DataFrame enrichi avec les clusters et projections UMAP."""
        return self.df_.copy() if self.df_ is not None else None

    def get_cluster_quality(self) -> Optional[dict]:
        """Retourne les métriques de qualité des clusters (Silhouette, DBCV, etc.)."""
        return self.cluster_quality_.copy() if self.cluster_quality_ is not None else None

    def get_stats_table(self) -> Optional[pd.DataFrame]:
        """Retourne un DataFrame avec les statistiques (moyenne, std, min, max) par cluster."""
        return self.cluster_stats_.copy() if self.cluster_stats_ is not None else None


    def get_summary(self) -> List[str]:
        """
        Retourne un résumé textuel des résultats du clustering.
        """
        summary_lines = []
        if self.cluster_quality_ is not None:
            summary_lines.append(
                f"Nombre de clusters trouvés (hors bruit): {self.cluster_quality_.get('n_clusters', 'N/A')}")
            summary_lines.append(
                f"Points de bruit: {self.cluster_quality_.get('noise_points', 'N/A')} ({self.cluster_quality_.get('noise_percentage', 0):.1f}%)")

            silhouette = self.cluster_quality_.get('silhouette', np.nan)
            if not np.isnan(silhouette):
                summary_lines.append(f"Score de Silhouette (hors bruit): {silhouette:.3f}")
            else:
                summary_lines.append("Score de Silhouette: N/A (moins de 2 clusters valides)")

            dbcv = self.cluster_quality_.get('dbcv', np.nan)
            if not np.isnan(dbcv):
                summary_lines.append(f"Indice DBCV: {dbcv:.3f}")
            else:
                summary_lines.append("Indice DBCV: N/A (moins de 2 clusters valides ou erreur)")

        return summary_lines

    def get_color_map(self) -> Optional[Dict[str, str]]:
        """Retourne la palette de couleurs des clusters."""
        if self.color_discrete_map_ is None:
            self._generate_cluster_colors()  # Ensure colors are generated
        return self.color_discrete_map_
