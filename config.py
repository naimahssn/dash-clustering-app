# config.py
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()  # Charge les variables d'environnement


class Config:
    CACHE_CONFIG = {
        'CACHE_TYPE': 'FileSystemCache',
        'CACHE_DIR': os.getenv('CACHE_DIR', 'cache'),
        'CACHE_THRESHOLD': int(os.getenv('CACHE_THRESHOLD', 500))
    }

    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'fr')
    TRANSLATIONS = {
        'fr': {
            'app_title': "Outil d'Analyse de Projets",
            'data_import': "Import & Traitement des Données",
            'advanced_analysis': "Analyse Avancée",
            'user_guide': "Guide Utilisateur",
            'step1': "Étape 1 : Import des Données",
            'import_button': "Importer CSV/Excel",
            'drag_drop_text': "Glisser-déposer un fichier ou cliquer pour sélectionner",
            'select_column': "Sélectionner une colonne",
            'step2': "Étape 2 : Calcul des Indicateurs",
            'initial_amount': "Montant Initial :",
            'committed_amount': "Montant Engagé :",
            'calculate_button': "Calculer les Indicateurs",
            'results': "Résultats avec Indicateurs",
            'export_results': "Exporter les Résultats",
            'save_analysis': "Sauvegarder l'Analyse",
            'project_classification': "Classification des Projets",
            'classification_features': "Variables de Classification :",
            'target_feature': "Variable Cible (optionnelle) :",
            'hdbscan_params': "Paramètres HDBSCAN :",
            'min_cluster_size': "min_cluster_size",
            'min_samples': "min_samples",
            'run_classification': "Exécuter Classification",
            'cluster_visualization': "Visualisation des Clusters",
            'tsne_projection': "Projection T-SNE",
            'perplexity': "Perplexité :",
            'feature_analysis': "Analyse des Variables",
            'feature_to_analyze': "Variable à analyser :",
            'cluster_stats': "Statistiques des Clusters",
            'cluster_distribution': "Distribution des Clusters",
            'detailed_results': "Résultats Détailés par Cluster",
            'global_view': "Vue Globale",
            'cluster_analysis': "Analyse par Cluster",
            'select_cluster': "Sélectionnez un Cluster :",
            'export_clusters': "Exporter les Clusters",
            'save_clusters': "Sauvegarder les Clusters",
            'success': "Succès",
            'error': "Erreur",
            'file_loaded': "Fichier {} chargé avec succès !",
            'only_csv_excel': "Seuls les fichiers CSV ou Excel sont supportés",
            'indicators_calculated': "Indicateurs calculés avec succès !",
            'classification_completed': "Classification effectuée avec succès !",
            'analysis_saved': "Analyse sauvegardée avec succès !",
            'clusters_saved': "Clusters sauvegardés avec succès !",
            'select_at_least_2': "Sélectionnez au moins 2 caractéristiques",
            'clustering_error': "Erreur lors de la classification : {}",
            'calculation_error': "Erreur lors des calculs : {}",
            'file_error': "Erreur lors du chargement : {}",
            'save_error': "Erreur lors de la sauvegarde : {}",
            'text_clustering': "Regroupement de Texte",
            'text_features': "Variables Textuelles :",
            'num_clusters': "Nombre de Clusters :",
            'run_text_clustering': "Exécuter Regroupement",
            'export_format': "Format d'Export :",
            'excel': "Excel",
            'pdf': "PDF",
            'text_clustering_results': "Résultats du Regroupement Textuel",
            'cluster': "Cluster",
            'text': "Texte",
            'text_clustering_completed': "Regroupement textuel effectué avec succès !",
            'total_projects': "Total Projets",
            'total_initial': "Montant Initial Total",
            'total_committed': "Montant Engagé Total",
            'avg_engagement': "Taux Engagement Moyen",
            'amount_comparison': "Montants Initial vs Engagé",
            'engagement_dist': "Distribution Taux d'Engagement",
            'variance_analysis': "Analyse des Écarts Budgétaires",
            'risk_distribution': "Distribution des Catégories de Risque",
            'correlation_matrix': "Matrice de Corrélation",
            'cluster_breakdown': "Répartition des Clusters",
            'trend_analysis': "Analyse des Tendances",
            'nature_breakdown': "Répartition par Nature Comptable",
            'outlier_detection': "Détection d'Anomalies",
            'relationships': "Relations entre Éléments",
            'graph_type': "Type de Graphique",
            'bar_chart': "Histogramme",
            'pie_chart': "Camembert",
            'sunburst': "Sunburst",
            'donut': "Anneau",
            'line_chart': "Courbe",
            'scatter_plot': "Nuage de Points",
            'network_graph': "Graphe de Relations",
            'dashboard_title': "Tableau de Bord Interactif",
            'time_column': "Colonne Temporelle :",
            'amount_column': "Colonne Montant :",
            'text_column': "Colonne Texte :",
            'similarity_threshold': "Seuil de Similarité :",
            'mixed_clustering_settings': "Paramètres de Regroupement Mixte",
            'optimal_k': "Clusters optimaux : {}",
            'no_clusters_found': "Aucun cluster détecté",
            'error_stats': "Erreur lors de la génération des statistiques",
            'min_clusters_error': "Au moins 2 clusters sont requis",
            'cluster_legend': "Clusters"

        }
    }

# Instance unique
config = Config()