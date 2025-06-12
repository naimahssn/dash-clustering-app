from dash import no_update, html, dash_table, dcc, ALL, callback_context
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import base64
import logging
import os
import base64
import io
from io import StringIO, BytesIO
from datetime import datetime

# Import your custom utility modules and classes
from utils.data_processing import detect_column_types
from utils.visualizations import  create_empty_figure
from utils.numerical_clustering import NumericalClusterAnalyzer
from utils.Postanalysis import (
    EnhancedParetoAnalysis,
    create_enhanced_pareto_results_layout,generate_comprehensive_report
)

# Make sure DeepClusteringModel, create_empty_figure, process_mixed_features, create_tsne_plot, convert_dates
from utils.deep_clustering_modules import DeepClusteringModel

# Scikit-learn and Keras/TensorFlow imports (ensure these are available in your environment)
from nltk.stem.snowball import FrenchStemmer
from dash.exceptions import PreventUpdate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Téléchargement des stopwords français si non déjà présents
try:
    stopwords.words('french')
except LookupError:
    nltk.download('stopwords')
french_stopwords = stopwords.words('french')

# Initial configuration
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
nltk.download('stopwords', quiet=True)  # Added quiet=True to avoid verbose output
french_stopwords = stopwords.words('french') + ['', 'nan', 'na', 'manquant']
stemmer = FrenchStemmer()

# Global instance of TopicClustering processor (initialized once)


def register_clustering_callbacks(app):
    """Enregistre tous les callbacks liés au clustering"""

    # Callback commun pour les options de caractéristiques
    @app.callback(
        [Output('numerical-features', 'options'),
         Output('text-features', 'options'),
         Output('deep-cluster-features', 'options')],
        [Input('processed-data', 'data')]
    )
    def update_feature_options(data):
        """
        Met à jour les options des dropdowns de sélection de caractéristiques
        en fonction des colonnes disponibles dans les données traitées.
        """
        if not data:
            raise PreventUpdate

        df = pd.DataFrame(data)
        col_types = detect_column_types(df)

        return (
            [{'label': col, 'value': col} for col in df.columns],  # For general selection
            [{'label': col, 'value': col} for col in df.columns],  # For text analysis
            [{'label': col, 'value': col} for col in df.columns]  # For deep clustering
        )

    # =================================================================================
    # CALLBACK PRINCIPAL POUR LE CLUSTERING NUMERIQUE
    # =================================================================================
    @app.callback(
        Output("numerical-cluster-graph", "figure"),
        Output("numerical-cluster-distribution", "figure"),
        Output("numerical-cluster-importance", "figure"),
        Output("cluster-sizes-graph", "figure"),
        Output("cluster-summary-output", "children"),  # This output is for your summary text
        Output("cluster-data-table", "data"),
        Output("cluster-data-table", "columns"),
        Output("interactive-cluster-stats-table", "figure"),  # NEW: Output for the interactive stats table
        Output('stored-clustering-data', 'data'),
        Output('stored-color-map', 'data'),
        Output('cluster-table-filter', 'options'),
        Output('stored-numeric-cols', 'data'),
        Input("numerical-cluster-btn", "n_clicks"),
        Input('cluster-table-filter', 'value'),
        Input('projection-dim', 'value'),
        State("processed-data", "data"),
        State("numerical-features", "value"),
        State("min-cluster-size", "value"),
        State('stored-clustering-data', 'data'),
        State('stored-color-map', 'data'),
        prevent_initial_call=True
    )
    def update_numerical_cluster(n_clicks, selected_clusters_for_table_filter, current_projection_dim, processed_data,
                                 selected_features,
                                 min_cluster_size,
                                 stored_clustering_data, stored_color_map):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Handle dropdown filter update
        if trigger_id == 'cluster-table-filter':
            if not stored_clustering_data:
                raise no_update

            df_original_stored = pd.DataFrame(stored_clustering_data)

            # Ensure numeric types for UMAP columns
            for col in ['UMAP-1', 'UMAP-2', 'UMAP-3']:
                if col in df_original_stored.columns:
                    df_original_stored[col] = pd.to_numeric(df_original_stored[col], errors='coerce')

            if not selected_clusters_for_table_filter:
                df_filtered = df_original_stored
            else:
                # Convert selected_clusters_for_table_filter to string to match 'CLUSTER' column type
                df_filtered = df_original_stored[
                    df_original_stored['CLUSTER'].isin([str(c) for c in selected_clusters_for_table_filter])]
                if df_filtered.empty:
                    logger.warning("Aucune donnée pour les clusters sélectionnés après filtrage.")
                    # Adjust return values for removed outputs
                    return (
                        create_empty_figure("Aucune donnée pour les clusters sélectionnés."),
                        no_update, no_update, no_update, no_update,
                        [], [],  # cluster-data-table data and columns
                        create_empty_figure("Aucune statistique à afficher."),  # interactive-cluster-stats-table
                        no_update, no_update, no_update, no_update
                    )

            table_data = df_filtered.to_dict("records")
            table_columns = [{"name": col, "id": col} for col in df_filtered.columns]

            umap_x_col, umap_y_col, umap_z_col = 'UMAP-1', 'UMAP-2', 'UMAP-3'

            fig_clusters_args = {
                'data_frame': df_filtered,
                'color': 'CLUSTER',
                'color_discrete_map': stored_color_map if stored_color_map else px.colors.qualitative.Plotly,
                'hover_data': df_filtered.columns,
                'title': f'Visualisation des Clusters (UMAP - {current_projection_dim}D) - Filtré',
            }

            if current_projection_dim == 3:
                if umap_z_col not in df_filtered.columns:
                    logger.warning("UMAP-3 column not found for 3D filtered plot, defaulting to 2D.")
                    fig_clusters = px.scatter(**fig_clusters_args, x=umap_x_col, y=umap_y_col)
                else:
                    fig_clusters = px.scatter_3d(**fig_clusters_args, x=umap_x_col, y=umap_y_col, z=umap_z_col)
            else:
                fig_clusters = px.scatter(**fig_clusters_args, x=umap_x_col, y=umap_y_col)

            fig_clusters.update_traces(
                marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
                selector=dict(mode='markers')
            )
            fig_clusters.update_layout(legend_title_text='Cluster')

            # For filter updates, the stats table and other plots don't need to be re-generated
            return (
                fig_clusters,
                no_update, no_update, no_update, no_update,
                table_data, table_columns,
                no_update,  # interactive-cluster-stats-table
                no_update, no_update, no_update, no_update
            )
        # Handle initial clustering button click
        elif trigger_id == 'numerical-cluster-btn':
            if processed_data is None or selected_features is None or len(selected_features) < 2:
                logger.warning("Données ou caractéristiques numériques insuffisantes pour le clustering.")
                raise no_update

            try:
                df = pd.DataFrame(processed_data)
                missing_cols = [col for col in selected_features if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Colonnes sélectionnées manquantes dans les données: {', '.join(missing_cols)}")

                analyzer = NumericalClusterAnalyzer(
                    min_cluster_size=min_cluster_size,
                    random_state=42
                )

                analyzer.fit(df, selected_features)

                # cluster_quality = analyzer.get_cluster_quality() # No longer directly outputting silhouette/dbcv/n_clusters
                fig_clusters = analyzer.plot_clusters(projection_dim=current_projection_dim,
                                                      hover_data=selected_features)
                fig_distribution = analyzer.plot_feature_distributions(features=selected_features)
                fig_importance = analyzer.plot_feature_importance()
                fig_cluster_sizes = analyzer.plot_cluster_sizes()
                interactive_stats_table_figure = analyzer.display_cluster_stats_table()  # NEW: Get the interactive table figure

                summary_lines = analyzer.get_summary()
                summary_text = [html.P(line) for line in summary_lines]

                table_df = analyzer.get_clustered_data()
                table_data = table_df.to_dict("records")
                table_columns = [{"name": col, "id": col} for col in table_df.columns]

                cluster_options = [{'label': f'CLUSTER {c}', 'value': c} for c in sorted(table_df['CLUSTER'].unique(),
                                                                                         key=lambda x: (
                                                                                             int(x) if x != '-1' else float(
                                                                                                 'inf')))]

                color_map_to_store = analyzer.get_color_map()
                numeric_cols_to_store = analyzer.numeric_cols_

                # Adjust return values to match new outputs
                return (
                    fig_clusters,
                    fig_distribution,
                    fig_importance,
                    fig_cluster_sizes,
                    summary_text,
                    table_data,
                    table_columns,
                    interactive_stats_table_figure,  # NEW: Return the interactive stats table figure
                    table_df.to_dict("records"),
                    color_map_to_store,
                    cluster_options,
                    numeric_cols_to_store,
                )

            except Exception as e:
                logger.error(f"Erreur lors du clustering numérique: {str(e)}", exc_info=True)
                # Adjust return values for error case
                return (
                    create_empty_figure("Erreur de visualisation"),
                    create_empty_figure("Erreur de visualisation"),
                    create_empty_figure("Erreur de visualisation"),
                    create_empty_figure("Erreur de visualisation"),
                    [html.P(f"Erreur : {str(e)}", className="text-danger")],
                    [], [],  # cluster-data-table data and columns
                    create_empty_figure("Erreur de chargement des statistiques."),  # interactive-cluster-stats-table
                    no_update, no_update, no_update, no_update
                )
        else:  # If projection-dim changes without re-clustering
            if not stored_clustering_data:
                raise no_update

            df_original_stored = pd.DataFrame(stored_clustering_data)

            # Ensure numeric types for UMAP columns
            for col in ['UMAP-1', 'UMAP-2', 'UMAP-3']:
                if col in df_original_stored.columns:
                    df_original_stored[col] = pd.to_numeric(df_original_stored[col], errors='coerce')

            umap_x_col, umap_y_col, umap_z_col = 'UMAP-1', 'UMAP-2', 'UMAP-3'

            fig_clusters_args = {
                'data_frame': df_original_stored,
                'color': 'CLUSTER',
                'color_discrete_map': stored_color_map if stored_color_map else px.colors.qualitative.Plotly,
                'hover_data': df_original_stored.columns,  # Use all columns for hover
                'title': f'Visualisation des Clusters (UMAP - {current_projection_dim}D)',
            }

            if current_projection_dim == 3:
                if umap_z_col not in df_original_stored.columns:
                    logger.warning("UMAP-3 column not found for 3D filtered plot, defaulting to 2D.")
                    fig_clusters = px.scatter(**fig_clusters_args, x=umap_x_col, y=umap_y_col)
                else:
                    fig_clusters = px.scatter_3d(**fig_clusters_args, x=umap_x_col, y=umap_y_col, z=umap_z_col)
            else:
                fig_clusters = px.scatter(**fig_clusters_args, x=umap_x_col, y=umap_y_col)

            fig_clusters.update_traces(
                marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
                selector=dict(mode='markers')
            )
            fig_clusters.update_layout(legend_title_text='Cluster')

            # For projection-dim changes, only the cluster graph updates
            return (
                fig_clusters,
                no_update, no_update, no_update, no_update,
                no_update, no_update,
                no_update,  # interactive-cluster-stats-table
                no_update, no_update, no_update, no_update
            )

    # --- Callback pour contrôler la visibilité des onglets de visualisation ---
    @app.callback(
        Output("numerical-cluster-graph", "style"),
        Output("numerical-cluster-distribution", "style"),
        Output("numerical-cluster-importance", "style"),
        Output("cluster-sizes-graph", "style"),  # Added for cluster sizes graph
        Output("detailed-analysis-content", "style"),
        # NEW: Add output for the interactive stats table if it's in a tab
        Output("interactive-cluster-stats-table", "style"),
        Input("cluster-visualization-tabs", "active_tab"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def render_cluster_visualization_tabs(active_tab):
        hidden = {'height': '600px', 'display': 'none'}
        visible = {'height': '600px', 'display': 'block'}  # Ensure display: 'block' for visibility
        div_hidden = {'display': 'none'}
        div_visible = {'display': 'block', 'height': 'auto'}

        return (
            visible if active_tab == "umap" else hidden,
            visible if active_tab == "distribution" else hidden,
            visible if active_tab == "importance" else hidden,
            visible if active_tab == "cluster-sizes" else hidden,  # Control visibility of cluster sizes graph
            div_visible if active_tab == "detailed_analysis" else div_hidden,
            # NEW: Control visibility for interactive stats table
            visible if active_tab == "cluster-stats" else hidden  # Assuming a tab with id "cluster-stats"
        )

    # --- Callback pour la coloration conditionnelle du tableau numérique ---
    @app.callback(
        Output('cluster-data-table', 'style_data_conditional'),
        Input('stored-clustering-data', 'data'),
        Input('stored-color-map', 'data'),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def apply_numerical_cluster_colors(data, color_map):
        if not data or not color_map:
            return []

        df = pd.DataFrame(data)
        if 'CLUSTER' not in df.columns:
            return []

        styles = []
        for cluster_id, color in color_map.items():
            styles.append({
                'if': {'filter_query': f'{{CLUSTER}} = "{cluster_id}"'},
                'backgroundColor': color,
                'color': 'white'
            })
        return styles

    # REMOVED: The old update_cluster_stats_table callback is removed as it's replaced by interactive-cluster-stats-table

    # Callback pour les distributions de caractéristiques catégorielles
    @app.callback(
        Output("categorical-cluster-selector", "options"),
        Output("categorical-features-dropdown", "options"),
        Input("stored-clustering-data", "data"),
        prevent_initial_call=True
    )
    def populate_categorical_dropdowns(clustered_data):
        if not clustered_data:
            raise PreventUpdate

        df = pd.DataFrame(clustered_data)

        cluster_options = [{'label': str(c), 'value': c} for c in sorted(df['CLUSTER'].unique())]

        # Détection des colonnes catégorielles (type object, category ou peu de modalités)
        categorical_cols = [
            col for col in df.columns
            if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 20
            if col != 'CLUSTER'  # exclure CLUSTER
        ]
        feature_options = [{'label': col, 'value': col} for col in categorical_cols]

        return cluster_options, feature_options

    @app.callback(
        Output('categorical-distribution-graph', 'figure'),
        Input('categorical-cluster-selector', 'value'),
        Input('categorical-features-dropdown', 'value'),
        State('stored-clustering-data', 'data'),
        prevent_initial_call=True
    )
    def update_categorical_distribution_plot(cluster_id, feature, clustered_data):
        if not clustered_data or cluster_id is None or feature is None:
            raise PreventUpdate

        df = pd.DataFrame(clustered_data)
        df_cluster = df[df['CLUSTER'] == cluster_id]

        if df_cluster.empty:
            return create_empty_figure(f"Aucune donnée trouvée pour le cluster {cluster_id}.")

        # Calcul des pourcentages
        value_counts = df_cluster[feature].value_counts(normalize=True).sort_index() * 100
        df_percent = value_counts.reset_index()
        df_percent.columns = [feature, 'Pourcentage']

        # Création du graphique
        fig = px.bar(
            df_percent,
            x=feature,
            y='Pourcentage',
            text='Pourcentage',
            title=f"Distribution (%) de '{feature}' dans le cluster {cluster_id}"
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis_title='Pourcentage', xaxis_title=feature, height=500)

        return fig


    # Votre code existant pour les stopwords français
    french_stopwords = [
        'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
        'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
        'par', 'grand', 'mais', 'si', 'ou', 'leur', 'bien', 'encore', 'pouvoir', 'aussi',
        'sans', 'autre', 'après', 'très', 'donc', 'aller', 'savoir', 'faire', 'voir',
        'donner', 'prendre', 'venir', 'falloir', 'vouloir', 'dire', 'quelque', 'où',
        'lui', 'nous', 'comme', 'alors', 'même', 'cette', 'votre', 'notre', 'trop',
        'peu', 'avant', 'deux', 'trois', 'toujours', 'jamais', 'souvent', 'peut',
        'peuvent', 'cette', 'ces', 'cela', 'celui', 'celle', 'ceux', 'celles', 'aussi',
        'ainsi', 'alors', 'après', 'avant', 'avec', 'beaucoup', 'bien', 'car', 'chez',
        'comme', 'comment', 'dans', 'depuis', 'dont', 'encore', 'entre', 'jusqu',
        'lors', 'pendant', 'plutôt', 'pour', 'près', 'puis', 'quand', 'que', 'quel',
        'quelle', 'quels', 'quelles', 'qui', 'sans', 'selon', 'sous', 'sur', 'tous',
        'toute', 'toutes', 'tout', 'très', 'vers', 'voici', 'voilà', 'là', 'où'
    ]

    # Fonction pour créer le rapport Excel
    def create_excel_report(df, stats_data, keywords_data, text_features):
        """
        Crée un rapport Excel complet avec tous les résultats du clustering
        """
        try:
            # Créer un buffer en mémoire
            output = io.BytesIO()

            # Créer un writer Excel avec xlsxwriter
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book

                # Définir les formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })

                cluster_format = workbook.add_format({
                    'bold': True,
                    'fg_color': '#F2F2F2',
                    'border': 1,
                    'align': 'center'
                })

                # 1. Feuille de résumé exécutif
                summary_data = {
                    'Métrique': [
                        'Date du rapport',
                        'Nombre total de documents',
                        'Nombre de clusters',
                        'Colonnes analysées',
                        'Méthode de clustering',
                        'Algorithme de réduction dimensionnelle'
                    ],
                    'Valeur': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(df),
                        len(df['TEXT_CLUSTER'].unique()),
                        ', '.join(text_features),
                        'K-Means avec TF-IDF',
                        't-SNE'
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Résumé', index=False)

                worksheet = writer.sheets['Résumé']
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 50)

                # 2. Feuille des documents avec clusters
                documents_df = df[text_features + ['TEXT_CLUSTER', 'TEXT_COMBINED']].copy()
                documents_df = documents_df.rename(columns={'TEXT_CLUSTER': 'Cluster_Assigné'})
                documents_df.to_excel(writer, sheet_name='Documents_Clusters', index=False)

                worksheet = writer.sheets['Documents_Clusters']
                worksheet.set_row(0, None, header_format)

                # Ajuster la largeur des colonnes
                for i, col in enumerate(documents_df.columns):
                    if col == 'TEXT_COMBINED':
                        worksheet.set_column(i, i, 50)
                    elif col == 'Cluster_Assigné':
                        worksheet.set_column(i, i, 15)
                    else:
                        worksheet.set_column(i, i, 25)

                # 3. Feuille des statistiques par cluster
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistiques_Clusters', index=False)

                worksheet = writer.sheets['Statistiques_Clusters']
                worksheet.set_row(0, None, header_format)

                # 4. Feuille des mots-clés par cluster
                keywords_df = pd.DataFrame(keywords_data)
                keywords_df.to_excel(writer, sheet_name='Mots_Clés_Clusters', index=False)

                worksheet = writer.sheets['Mots_Clés_Clusters']
                worksheet.set_row(0, None, header_format)
                worksheet.set_column('A:A', 15)
                worksheet.set_column('B:B', 80)

                # 5. Feuille détaillée par cluster
                for cluster_id in sorted(df['TEXT_CLUSTER'].unique()):
                    cluster_data = df[df['TEXT_CLUSTER'] == cluster_id][text_features + ['TEXT_COMBINED']].copy()
                    sheet_name = f'Cluster_{cluster_id}'
                    cluster_data.to_excel(writer, sheet_name=sheet_name, index=False)

                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_row(0, None, header_format)

                    # Ajuster les colonnes
                    for i, col in enumerate(cluster_data.columns):
                        if col == 'TEXT_COMBINED':
                            worksheet.set_column(i, i, 60)
                        else:
                            worksheet.set_column(i, i, 25)

                # 6. Feuille de métadonnées techniques
                metadata = {
                    'Paramètre': [
                        'Vectorisation',
                        'Stop words',
                        'Max features TF-IDF',
                        'Algorithme clustering',
                        'Random state',
                        'Perplexité t-SNE',
                        'Itérations t-SNE'
                    ],
                    'Valeur': [
                        'TF-IDF',
                        'Français personnalisé',
                        '10000',
                        'K-Means',
                        '42',
                        'Auto-ajustée',
                        '1000'
                    ]
                }
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_excel(writer, sheet_name='Métadonnées', index=False)

                worksheet = writer.sheets['Métadonnées']
                worksheet.set_row(0, None, header_format)
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:B', 30)

            # Récupérer les données du buffer
            output.seek(0)
            return output.getvalue()

        except Exception as e:
            print(f"Erreur lors de la création du rapport Excel : {str(e)}")
            return None

    # CALLBACK PRINCIPAL POUR LE CLUSTERING TEXTUELLE

    def _generate_elbow_method_plot(k_range, sse, optimal_n_clusters):
        fig = px.line(
            x=list(k_range),
            y=sse,
            markers=True,
            labels={'x': 'Nombre de Clusters (k)', 'y': 'Inertie (SSE)'},
            title='Méthode du Coude pour Déterminer K'
        )
        fig.add_vline(x=optimal_n_clusters, line_dash="dot", line_color="red",
                      annotation_text=f"K optimal: {optimal_n_clusters}",
                      annotation_position="top right")
        fig.update_layout(xaxis_title="Nombre de Clusters (k)", yaxis_title="Inertie (Somme des Carrés des Erreurs)")
        return fig

    def _generate_silhouette_score_plot(k_range, silhouette_scores, optimal_n_clusters):
        valid_silhouette_k_range = [k for k, score in zip(k_range, silhouette_scores) if score != -1]
        valid_silhouette_values = [score for score in silhouette_scores if score != -1]

        fig = px.line(
            x=valid_silhouette_k_range,
            y=valid_silhouette_values,
            markers=True,
            labels={'x': 'Nombre de Clusters (k)', 'y': 'Score de Silhouette'},
            title='Score de Silhouette pour Déterminer K'
        )
        fig.add_vline(x=optimal_n_clusters, line_dash="dot", line_color="red",
                      annotation_text=f"K optimal: {optimal_n_clusters}",
                      annotation_position="top right")
        fig.update_layout(xaxis_title="Nombre de Clusters (k)", yaxis_title="Score de Silhouette")
        return fig

    def _generate_cluster_distribution_chart(df, colors):
        cluster_counts = df['TEXT_CLUSTER'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Nombre de documents'},
            title='Distribution des Documents par Cluster',
            color=cluster_counts.index.astype(str),
            color_discrete_sequence=colors
        )
        fig.update_layout(xaxis_title="Numéro du Cluster", yaxis_title="Nombre de Documents")
        return fig

    def _generate_cluster_scatter_plot(df, text_features, colors):
        fig = px.scatter(
            df,
            x='TSNE_X',
            y='TSNE_Y',
            color=df['TEXT_CLUSTER'].astype(str),
            labels={'TSNE_X': 't-SNE Composante 1', 'TSNE_Y': 't-SNE Composante 2', 'color': 'Cluster'},
            title='Visualisation des Clusters Textuels (t-SNE)',
            hover_data=text_features + ['TEXT_CLUSTER'],
            color_discrete_sequence=colors
        )
        fig.update_layout(xaxis_title="t-SNE X", yaxis_title="t-SNE Y")
        return fig

    def _prepare_text_results_table_data(df, valid_text_features, colors):
        cols_for_table = valid_text_features + ['TEXT_CLUSTER']
        text_results_df = df[cols_for_table].copy()
        text_results_df['TEXT_CLUSTER'] = text_results_df['TEXT_CLUSTER'].astype(str)

        table_data = text_results_df.to_dict('records')
        table_columns = [{'name': col, 'id': col} for col in text_results_df.columns]

        tooltip_data = [
            {
                'TEXT_CLUSTER': {
                    'value': 'Filtrage exact : tapez le numéro exact du cluster (ex: 0, 1, 2...)',
                    'type': 'markdown'
                }
            } for _ in range(len(text_results_df))
        ]

        return table_data, table_columns, tooltip_data

    def _prepare_cluster_statistics_data(df, cluster_counts, num_documents_clustered, french_stopwords, colors):
        stats_data = []
        keywords_data = []

        for cluster_id in sorted(cluster_counts.index):
            count = cluster_counts.loc[cluster_id]
            percentage = (count / num_documents_clustered * 100) if num_documents_clustered > 0 else 0

            cluster_texts = df[df['TEXT_CLUSTER'] == cluster_id]['TEXT_COMBINED']
            top_keywords_str = "N/A"

            if not cluster_texts.empty:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    cluster_vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=500)
                    cluster_X = cluster_vectorizer.fit_transform(cluster_texts)

                    sums_tfidf = cluster_X.sum(axis=0)
                    feature_names = cluster_vectorizer.get_feature_names_out()
                    tfidf_scores = [(word, sums_tfidf[0, idx]) for word, idx in
                                    zip(feature_names, range(len(feature_names)))]
                    tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

                    most_common_words = [word for word, score in tfidf_scores[:10]]
                    top_keywords_str = ", ".join(most_common_words)
                except Exception as e:
                    print(f"Erreur lors de l'extraction des mots-clés pour le cluster {cluster_id}: {e}")
                    top_keywords_str = "Erreur lors de l'extraction"

            stats_data.append({
                "Cluster": str(cluster_id),
                "Nombre de documents": count,
                "Pourcentage": f"{percentage:.2f}%",
                "Densité relative": round(percentage / 100, 3)
            })

            keywords_data.append({
                "Cluster": str(cluster_id),
                "Mots-clés représentatifs": top_keywords_str
            })

        return stats_data, keywords_data

    extended_colors = [
        "#FF8A65", "#81C784", "#64B5F6", "#BA68C8", "#FF77A9", "#4DB6AC", "#FFB74D", "#7986CB",
    ]

    # NOUVEAU CALLBACK POUR LE TÉLÉCHARGEMENT EXCEL
    @app.callback(
        Output("download-excel-report", "data"),
        Input("download-excel-btn", "n_clicks"),
        State('text-clustering-result', 'data'),
        State('stats-table', 'data'),
        State('keywords-table', 'data'),
        State('text-features', 'value'),
        prevent_initial_call=True
    )
    def download_excel_report(n_clicks, clustering_data, stats_data, keywords_data, text_features):
        if n_clicks is None or not clustering_data:
            raise PreventUpdate

        try:
            df = pd.DataFrame.from_dict(clustering_data)

            # Créer le rapport Excel
            excel_data = create_excel_report(df, stats_data, keywords_data, text_features)

            if excel_data is None:
                raise Exception("Erreur lors de la génération du rapport")

            # Nom du fichier avec timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"rapport_clustering_textuel_{timestamp}.xlsx"

            return dcc.send_bytes(excel_data, filename)

        except Exception as e:
            print(f"Erreur lors du téléchargement du rapport Excel : {str(e)}")
            # Vous pourriez aussi ajouter une alerte utilisateur ici
            raise PreventUpdate

    @app.callback(
        Output('text-clustering-result', 'data', allow_duplicate=True),
        Output('text-results-table', 'data', allow_duplicate=True),
        Output('text-results-table', 'columns', allow_duplicate=True),
        Output('text-results-table', 'tooltip_data', allow_duplicate=True),
        Output('cluster-distribution-chart', 'figure', allow_duplicate=True),
        Output('cluster-scatter-plot', 'figure', allow_duplicate=True),
        Output('total-documents-clustered', 'children', allow_duplicate=True),
        Output('num-clusters-found', 'children', allow_duplicate=True),
        Output('stats-table', 'data', allow_duplicate=True),
        Output('stats-table', 'style_data_conditional', allow_duplicate=True),
        Output('keywords-table', 'data', allow_duplicate=True),
        Output('keywords-table', 'style_data_conditional', allow_duplicate=True),
        Output('status-alerts-container', 'is_open', allow_duplicate=True),
        Output('status-alerts-container', 'children', allow_duplicate=True),
        Output('status-alerts-container', 'color', allow_duplicate=True),
        Output('elbow-method-chart', 'figure', allow_duplicate=True),
        Output('silhouette-score-chart', 'figure', allow_duplicate=True),
        Output('global-cluster-filter', 'options', allow_duplicate=True),
        Output('global-cluster-filter', 'value', allow_duplicate=True),
        # NOUVEAU : Activer le bouton de téléchargement
        Output('download-excel-btn', 'disabled', allow_duplicate=True),
        Input('text-cluster-btn', 'n_clicks'),
        State('processed-data', 'data'),
        State('text-features', 'value'),
        State('min-clusters-auto', 'value'),
        State('max-clusters-auto', 'value'),
        prevent_initial_call=True
    )
    def run_text_clustering(n_clicks, data, text_features, min_clusters_auto, max_clusters_auto):
        if n_clicks is None or not text_features or data is None:
            raise PreventUpdate

        if not isinstance(text_features, list) or not text_features:
            return (
                dash.no_update, [], [], [], go.Figure(), go.Figure(),
                dash.no_update, dash.no_update, [], [], [], [],
                True, "Veuillez sélectionner au moins une colonne de texte pour le clustering.", "warning",
                go.Figure(), go.Figure(), [], [], True  # Bouton désactivé
            )

        if min_clusters_auto is None or max_clusters_auto is None or min_clusters_auto >= max_clusters_auto or min_clusters_auto < 2:
            return (
                dash.no_update, [], [], [], go.Figure(), go.Figure(),
                dash.no_update, dash.no_update, [], [], [], [],
                True, "Veuillez définir une plage valide pour le nombre de clusters (minimum 2, min < max).", "warning",
                go.Figure(), go.Figure(), [], [], True  # Bouton désactivé
            )

        try:
            df = pd.DataFrame.from_dict(data)

            valid_text_features = [col for col in text_features if col in df.columns]
            if not valid_text_features:
                return (
                    dash.no_update, [], [], [], go.Figure(), go.Figure(),
                    dash.no_update, dash.no_update, [], [], [], [],
                    True, "Aucune des colonnes sélectionnées n'existe dans les données fournies.", "danger",
                    go.Figure(), go.Figure(), [], [], True  # Bouton désactivé
                )

            df['TEXT_COMBINED'] = df[valid_text_features].apply(
                lambda x: ' '.join(x.dropna().astype(str)), axis=1
            )
            df = df[df['TEXT_COMBINED'].str.strip() != ''].reset_index(drop=True)
            if df.empty:
                return (
                    dash.no_update, [], [], [], go.Figure(), go.Figure(),
                    dash.no_update, dash.no_update, [], [], [], [],
                    True, "Aucune donnée textuelle valide trouvée pour le clustering après nettoyage.", "warning",
                    go.Figure(), go.Figure(), [], [], True  # Bouton désactivé
                )

            vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=10000)
            X = vectorizer.fit_transform(df['TEXT_COMBINED'])

            sse = []
            silhouette_scores = []
            k_range = range(min_clusters_auto, max_clusters_auto + 1)

            if X.shape[0] < 2:
                return (
                    dash.no_update, [], [], [], go.Figure(), go.Figure(),
                    dash.no_update, dash.no_update, [], [], [], [],
                    True, "Pas assez de documents pour calculer les métriques de clustering (moins de 2 documents).",
                    "danger",
                    go.Figure(), go.Figure(), [], [], True  # Bouton désactivé
                )

            best_silhouette_score = -1
            optimal_n_clusters = 0

            for k in k_range:
                if k == 1:
                    sse.append(KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X).inertia_)
                    silhouette_scores.append(-1)
                    continue

                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(X)
                sse.append(kmeans.inertia_)

                if X.shape[0] >= k:
                    score = silhouette_score(X, kmeans.labels_)
                    silhouette_scores.append(score)
                    if score > best_silhouette_score:
                        best_silhouette_score = score
                        optimal_n_clusters = k
                else:
                    silhouette_scores.append(-1)

            if optimal_n_clusters == 0:
                optimal_n_clusters = min_clusters_auto if min_clusters_auto >= 2 else 2

            print(f"Nombre optimal de clusters détecté via Silhouette Score: {optimal_n_clusters}")

            fig_elbow = _generate_elbow_method_plot(k_range, sse, optimal_n_clusters)
            fig_silhouette = _generate_silhouette_score_plot(k_range, silhouette_scores, optimal_n_clusters)

            kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init='auto')
            df['TEXT_CLUSTER'] = kmeans.fit_predict(X)

            n_components_tsne = 2
            if X.shape[0] < n_components_tsne + 1:
                return (
                    dash.no_update, [], [], [], go.Figure(), go.Figure(),
                    dash.no_update, dash.no_update, [], [], [], [],
                    True,
                    f"Pas assez de documents pour la réduction de dimensionnalité t-SNE (besoin d'au moins {n_components_tsne + 1} documents).",
                    "danger",
                    go.Figure(), go.Figure(), [], [], True  # Bouton désactivé
                )

            tsne_perplexity = min(30, X.shape[0] - 1)
            if tsne_perplexity < 1:
                tsne_perplexity = 1

            tsne = TSNE(n_components=n_components_tsne, random_state=42,
                        perplexity=tsne_perplexity, n_iter=1000, learning_rate='auto',
                        init='random')
            X_tsne = tsne.fit_transform(X.toarray())
            df['TSNE_X'] = X_tsne[:, 0]
            df['TSNE_Y'] = X_tsne[:, 1]

            cluster_counts = df['TEXT_CLUSTER'].value_counts().sort_index()
            num_documents_clustered = len(df)
            cluster_options = [{'label': 'Tous les clusters', 'value': 'all'}]
            cluster_options.extend([
                {'label': f'Cluster {cluster_id} ({count} docs)', 'value': str(cluster_id)}
                for cluster_id, count in cluster_counts.items()
            ])

            fig_cluster_distribution = _generate_cluster_distribution_chart(df, extended_colors)
            fig_cluster_scatter = _generate_cluster_scatter_plot(df, valid_text_features, extended_colors)

            text_results_data, text_results_columns, text_results_tooltip_data = _prepare_text_results_table_data(df,
                                                                                                                  valid_text_features,
                                                                                                                  extended_colors)

            stats_data_dict, keywords_data_dict = \
                _prepare_cluster_statistics_data(df, cluster_counts, num_documents_clustered, french_stopwords,
                                                 extended_colors)

            stats_table_style_data_conditional = [
                {
                    'if': {'column_id': 'Cluster'},
                    'fontWeight': 'bold'
                },
                *[
                    {
                        'if': {
                            'filter_query': '{Cluster} = "' + str(cluster) + '"',
                            'column_id': 'Cluster'
                        },
                        'backgroundColor': extended_colors[i % len(extended_colors)],
                        'color': 'white'
                    }
                    for i, cluster in enumerate(sorted(cluster_counts.index))
                ]
            ]
            keywords_table_style_data_conditional = [
                {
                    'if': {'column_id': 'Cluster'},
                    'fontWeight': 'bold'
                },
                *[
                    {
                        'if': {
                            'filter_query': '{Cluster} = "' + str(cluster) + '"',
                            'column_id': 'Cluster'
                        },
                        'backgroundColor': extended_colors[i % len(extended_colors)],
                        'color': 'white'
                    }
                    for i, cluster in enumerate(sorted(cluster_counts.index))
                ]
            ]

            print("Clustering textuel effectué avec succès et visualisations générées.")

            return (
                df.to_dict('records'),  # text-clustering-result
                text_results_data,  # text-results-table data
                text_results_columns,  # text-results-table columns
                text_results_tooltip_data,  # text-results-table tooltip_data
                fig_cluster_distribution,
                fig_cluster_scatter,
                f"Nombre total de documents analysés : {num_documents_clustered}",  # total-documents-clustered
                f"Nombre de clusters trouvés : {len(cluster_counts)}",  # num-clusters-found
                stats_data_dict,  # stats-table data
                stats_table_style_data_conditional,  # stats-table styling
                keywords_data_dict,  # keywords-table data
                keywords_table_style_data_conditional,  # keywords-table styling
                True, f"Clustering textuel terminé avec succès ! Nombre de clusters détecté : {optimal_n_clusters}",
                "success",
                fig_elbow, fig_silhouette,
                cluster_options, ['all'],
                False  # NOUVEAU : Activer le bouton de téléchargement
            )
        except Exception as e:
            print(f"Erreur lors de l'exécution du clustering textuel : {str(e)}")
            return (
                dash.no_update, [], [], [], go.Figure(), go.Figure(),
                dash.no_update, dash.no_update, [], [], [], [],
                True, f"Erreur lors du clustering textuel : {str(e)}", "danger",
                go.Figure(), go.Figure(), [], [], True  # Bouton désactivé
            )

    @app.callback(
        Output('cluster-distribution-chart', 'figure', allow_duplicate=True),
        Output('cluster-scatter-plot', 'figure', allow_duplicate=True),
        Output('text-results-table', 'data', allow_duplicate=True),
        Output('stats-table', 'data', allow_duplicate=True),
        Output('keywords-table', 'data', allow_duplicate=True),
        Output('text-results-table', 'style_data_conditional', allow_duplicate=True),
        Output('stats-table', 'style_data_conditional', allow_duplicate=True),
        Output('keywords-table', 'style_data_conditional', allow_duplicate=True),
        Output('total-documents-clustered', 'children', allow_duplicate=True),
        Output('num-clusters-found', 'children', allow_duplicate=True),
        Input('global-cluster-filter', 'value'),
        State('text-clustering-result', 'data'),
        State('text-features', 'value'),
        prevent_initial_call=True
    )
    def update_visualizations_with_global_filter(selected_clusters, clustering_data, text_features):
        if not clustering_data or not selected_clusters:
            raise PreventUpdate

        try:
            df = pd.DataFrame.from_dict(clustering_data)

            if 'all' not in selected_clusters:
                df['TEXT_CLUSTER'] = df['TEXT_CLUSTER'].astype(int)
                selected_cluster_ids = [int(cluster) for cluster in selected_clusters if cluster != 'all']
                df_filtered = df[df['TEXT_CLUSTER'].isin(selected_cluster_ids)].copy()
            else:
                df_filtered = df.copy()

            if df_filtered.empty:
                empty_fig = go.Figure()
                return empty_fig, empty_fig, [], [], [], [], [], [], \
                    f"Nombre total de documents analysés : 0", \
                    f"Nombre de clusters trouvés : 0"

            fig_cluster_distribution = _generate_cluster_distribution_chart(df_filtered, extended_colors)
            fig_cluster_scatter = _generate_cluster_scatter_plot(df_filtered, text_features, extended_colors)

            text_results_data, _, text_results_tooltip_data = _prepare_text_results_table_data(df_filtered,
                                                                                               text_features,
                                                                                               extended_colors)
            cluster_counts_filtered = df_filtered['TEXT_CLUSTER'].value_counts().sort_index()
            num_documents_filtered = len(df_filtered)

            stats_data_dict, keywords_data_dict = \
                _prepare_cluster_statistics_data(df_filtered, cluster_counts_filtered,
                                                 num_documents_filtered, french_stopwords, extended_colors)

            # Recalculate style_data_conditional based on filtered data
            stats_table_style_data_conditional = [
                {
                    'if': {'column_id': 'Cluster'},
                    'fontWeight': 'bold'
                },
                *[
                    {
                        'if': {
                            'filter_query': '{Cluster} = "' + str(cluster) + '"',
                            'column_id': 'Cluster'
                        },
                        'backgroundColor': extended_colors[i % len(extended_colors)],
                        'color': 'white'
                    }
                    for i, cluster in enumerate(sorted(cluster_counts_filtered.index))
                ]
            ]
            keywords_table_style_data_conditional = [
                {
                    'if': {'column_id': 'Cluster'},
                    'fontWeight': 'bold'
                },
                *[
                    {
                        'if': {
                            'filter_query': '{Cluster} = "' + str(cluster) + '"',
                            'column_id': 'Cluster'
                        },
                        'backgroundColor': extended_colors[i % len(extended_colors)],
                        'color': 'white'
                    }
                    for i, cluster in enumerate(sorted(cluster_counts_filtered.index))
                ]
            ]
            text_results_table_style_data_conditional = [
                {
                    'if': {'column_id': 'TEXT_CLUSTER'},
                    'fontWeight': 'bold'
                },
                *[
                    {
                        'if': {
                            'filter_query': '{TEXT_CLUSTER} = "' + str(cluster) + '"',
                            'column_id': 'TEXT_CLUSTER'
                        },
                        'backgroundColor': extended_colors[i % len(extended_colors)],
                        'color': 'white'
                    }
                    for i, cluster in enumerate(sorted(df_filtered['TEXT_CLUSTER'].unique()))
                ]
            ]
            return (
                fig_cluster_distribution,
                fig_cluster_scatter,
                text_results_data,
                stats_data_dict,
                keywords_data_dict,
                text_results_table_style_data_conditional,
                stats_table_style_data_conditional,
                keywords_table_style_data_conditional,
                f"Nombre total de documents analysés : {num_documents_filtered}",
                f"Nombre de clusters trouvés : {len(cluster_counts_filtered)}",
            )

        except Exception as e:
            print(f"Erreur lors du filtrage global : {str(e)}")
            empty_fig = go.Figure()
            return empty_fig, empty_fig, [], [], [], [], [], [], \
                f"Erreur de mise à jour: {str(e)}", \
                f"Erreur de mise à jour: {str(e)}"

    def update_filter_style(selected_values):
        if selected_values and 'all' not in selected_values:
            return {'border': '2px solid var(--bs-primary)', 'borderRadius': '5px'}
        return {}

    # COMPOSANTS UI POUR LE TÉLÉCHARGEMENT EXCEL
    def create_download_section():
        """
        Crée la section de téléchargement pour le rapport Excel
        """
        return html.Div([
            html.Hr(),
            html.H4("📊 Téléchargement du Rapport", className="mb-3"),
            html.P(
                "Téléchargez un rapport Excel complet contenant tous les résultats de votre analyse de clustering textuel.",
                className="text-muted mb-3"),

            html.Div([
                html.Button(
                    [
                        html.I(className="fas fa-download me-2"),
                        "Télécharger le Rapport Excel"
                    ],
                    id="download-excel-btn",
                    className="btn btn-success btn-lg",
                    disabled=True,  # Désactivé initialement
                    style={"width": "100%"}
                ),
                dcc.Download(id="download-excel-report")
            ], className="d-grid gap-2"),

            html.Div([
                html.Small([
                    html.Strong("Le rapport inclut :"),
                    html.Ul([
                        html.Li("Résumé exécutif avec métriques clés"),
                        html.Li("Documents originaux avec clusters assignés"),
                        html.Li("Statistiques détaillées par cluster"),
                        html.Li("Mots-clés représentatifs de chaque cluster"),
                        html.Li("Feuilles séparées pour chaque cluster"),
                        html.Li("Métadonnées techniques du processus")
                    ])
                ], className="text-muted")
            ], className="mt-3")
        ], className="card-body")

    # EXEMPLE D'INTÉGRATION DANS VOTRE LAYOUT
    def create_clustering_results_layout():
        """
        Layout pour les résultats de clustering avec section de téléchargement
        """
        return html.Div([
            # Vos composants existants...
            html.Div(id="clustering-results-container", children=[
                # Graphiques et tableaux existants
                dcc.Graph(id="cluster-distribution-chart"),
                dcc.Graph(id="cluster-scatter-plot"),
                dcc.Graph(id="elbow-method-chart"),
                dcc.Graph(id="silhouette-score-chart"),

                # Tableaux
                dash_table.DataTable(
                    id="text-results-table",
                    columns=[],
                    data=[],
                    filter_action="native",
                    sort_action="native",
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),

                dash_table.DataTable(
                    id="stats-table",
                    columns=[
                        {"name": "Cluster", "id": "Cluster"},
                        {"name": "Nombre de documents", "id": "Nombre de documents"},
                        {"name": "Pourcentage", "id": "Pourcentage"},
                        {"name": "Densité relative", "id": "Densité relative"}
                    ],
                    data=[],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),

                dash_table.DataTable(
                    id="keywords-table",
                    columns=[
                        {"name": "Cluster", "id": "Cluster"},
                        {"name": "Mots-clés représentatifs", "id": "Mots-clés représentatifs"}
                    ],
                    data=[],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),

                # NOUVELLE SECTION DE TÉLÉCHARGEMENT
                html.Div([
                    html.Div([
                        create_download_section()
                    ], className="card mt-4")
                ])
            ])
        ])



    @app.callback(
        Output('pareto-cluster-selector', 'options'),
        Output('pareto-variable-selector', 'options'),
        Output('pareto-display-selector', 'options'),
        Input('text-clustering-result', 'data'),
        prevent_initial_call=True
    )
    def update_pareto_options(clustering_data):
        """Met à jour les options de sélection pour l'analyse de Pareto"""
        if not clustering_data:
            return [], [], []

        df = pd.DataFrame.from_dict(clustering_data)
        if clustering_data:
            df = pd.DataFrame(clustering_data)
            # Drop TSNE_X, TSNE_Y, and TEXT_COMBINED columns if they exist
            columns_to_drop = []
            if 'TSNE_X' in df.columns:
                columns_to_drop.append('TSNE_X')
            if 'TSNE_Y' in df.columns:
                columns_to_drop.append('TSNE_Y')

            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)

        # Créer une instance temporaire pour obtenir les colonnes
        analyzer = EnhancedParetoAnalysis(df)

        # Options des clusters
        cluster_counts = df['TEXT_CLUSTER'].value_counts().sort_index()
        cluster_options = [{'label': 'Tous les clusters', 'value': 'all'}]
        cluster_options.extend([
            {'label': f'Cluster {cluster_id} ({count} docs)', 'value': str(cluster_id)}
            for cluster_id, count in cluster_counts.items()
        ])

        # Options des variables numériques (utilise les méthodes de la classe)
        numeric_options = [{'label': col, 'value': col} for col in analyzer.numeric_columns]

        # Options des colonnes d'affichage (utilise les méthodes de la classe)
        display_options = [{'label': col, 'value': col} for col in analyzer.text_columns]
        display_options.extend([
            {'label': col, 'value': col}
            for col in analyzer.numeric_columns
            if col not in analyzer.text_columns
        ])

        return cluster_options, numeric_options, display_options

    @app.callback(
        Output('pareto-analysis-data', 'data'),
        Output('pareto-results-container', 'children'),
        Output('pareto-insights-data', 'data'),
        Input('pareto-analyze-enhanced-btn', 'n_clicks'),
        State('text-clustering-result', 'data'),
        State('pareto-cluster-selector', 'value'),
        State('pareto-variable-selector', 'value'),
        State('pareto-threshold-slider', 'value'),
        State('pareto-display-selector', 'value'),
        prevent_initial_call=True
    )
    def perform_pareto_analysis_callback(enhanced_clicks, clustering_data,
                                         selected_clusters, variable_column, threshold_percent,
                                         display_columns):
        """Effectue l'analyse de Pareto détaillée"""
        if not enhanced_clicks or not clustering_data or not selected_clusters or not variable_column:
            return no_update, no_update, no_update

        try:
            df = pd.DataFrame.from_dict(clustering_data)
            analyzer = EnhancedParetoAnalysis(df, 'TEXT_CLUSTER')

            # Analyse détaillée avec insights
            results = analyzer.perform_enhanced_pareto_analysis(
                df, selected_clusters, variable_column,
                threshold_percent, display_columns
            )

            if not results:
                return no_update, dbc.Alert(
                    "Aucune donnée valide pour l'analyse. Vérifiez que la variable contient des valeurs positives.",
                    color="warning"
                ), no_update

            # Créer le layout
            results_layout = create_enhanced_pareto_results_layout(results)
            # Stocke les insights séparément
            insights = results.get('business_insights', {})
            return results, results_layout, insights

        except Exception as e:
            return no_update, dbc.Alert(f"Erreur lors de l'analyse: {str(e)}", color="danger"), no_update

    @app.callback(
        Output("pareto-download-csv", "data"),
        Input("pareto-export-csv-btn", "n_clicks"),
        State('pareto-analysis-data', 'data'),
        prevent_initial_call=True
    )
    def export_pareto_csv(n_clicks, analysis_data):
        """Exporte les résultats en CSV"""
        if not n_clicks or not analysis_data:
            return no_update

        try:
            full_data = pd.DataFrame(analysis_data['full_data'])
            key_elements = pd.DataFrame(analysis_data['key_elements'])

            output = StringIO()

            # Écrire un résumé enrichi
            output.write(f"ANALYSE DE PARETO AVANCÉE - {analysis_data['variable_column']}\n")
            output.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output.write(f"Seuil: {analysis_data['threshold_percent']}%\n")
            output.write(f"Total éléments: {len(full_data)}\n")
            output.write(f"Éléments clés: {analysis_data['num_key_elements']}\n")
            output.write(
                f"Contribution clés: {analysis_data['key_elements_contribution'] / analysis_data['total_sum'] * 100:.2f}%\n\n")

            # Éléments clés
            output.write("ÉLÉMENTS CLÉS\n")
            key_elements.to_csv(output, index=False)
            output.write("\n\nANALYSE COMPLÈTE\n")
            full_data.to_csv(output, index=False)

            return {
                "content": output.getvalue(),
                "filename": f"pareto_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }

        except Exception as e:
            return no_update

    @app.callback(
        Output("pareto-download-report", "data"),
        Input("pareto-export-report-btn", "n_clicks"),
        State('pareto-insights-data', 'data'),
        State('pareto-analysis-data', 'data'),
        prevent_initial_call=True
    )
    def export_comprehensive_report(n_clicks, insights_data, analysis_data):
        """Exporte le rapport complet avec insights"""
        if not n_clicks or not insights_data or not analysis_data:
            return no_update

        try:
            # Générer le rapport structuré
            report = generate_comprehensive_report({
                **analysis_data,
                'business_insights': insights_data
            })

            # Créer un buffer pour le rapport
            output = BytesIO()

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Feuille de résumé exécutif
                summary_df = pd.DataFrame({
                    'Métrique': [
                        'Variable analysée', 'Seuil Pareto', 'Total éléments',
                        'Éléments clés', 'Contribution clés', 'Index Gini',
                        'Score stabilité'
                    ],
                    'Valeur': [
                        analysis_data['variable_column'],
                        f"{analysis_data['threshold_percent']}%",
                        len(pd.DataFrame(analysis_data['full_data'])),
                        analysis_data['num_key_elements'],
                        f"{analysis_data['key_elements_contribution'] / analysis_data['total_sum'] * 100:.1f}%",
                        f"{insights_data['concentration']['gini_coefficient']:.3f}",
                        f"{insights_data['cluster_stability']['stability_score']:.2f}"
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Résumé Exécutif', index=False)

                # Feuille des recommandations
                rec_df = pd.DataFrame(insights_data['recommendations'])
                rec_df.to_excel(writer, sheet_name='Recommandations', index=False)

                # Feuille des éléments clés
                key_elements = pd.DataFrame(analysis_data['key_elements'])
                key_elements.to_excel(writer, sheet_name='Éléments Clés', index=False)

                # Feuille des KPIs
                kpis_df = pd.DataFrame(report['kpis_to_monitor'])
                kpis_df.to_excel(writer, sheet_name='KPIs', index=False)

            output.seek(0)
            return {
                "content": base64.b64encode(output.read()).decode(),
                "filename": f"rapport_pareto_complet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "base64": True
            }

        except Exception as e:
            return no_update

    # Callback pour la génération dynamique de l'UI de sélection de type de caractéristique
    @app.callback(
        Output('feature-type-selection-ui', 'children'),
        Input('deep-cluster-features', 'value'),
        State('processed-data', 'data')
    )
    def generate_type_selection_ui(features, data):
        """
        Génère dynamiquement l'interface utilisateur pour la sélection du type de chaque caractéristique
        choisie pour le clustering profond.
        """
        if not features or not data:
            return []

        df = pd.DataFrame(data)
        options = [
            {'label': 'Numérique', 'value': 'numerique'},
            {'label': 'Catégoriel', 'value': 'categoriel'},
            {'label': 'Texte', 'value': 'texte'},
            {'label': 'Date', 'value': 'date'}
        ]

        rows = []
        for feature in features:
            sample_value = df[feature].dropna().iloc[0] if not df.empty and not df[feature].dropna().empty else ''
            rows.append(
                dbc.Row([
                    dbc.Col(html.Div([
                        html.B(feature),
                        html.Small(f" (ex: {str(sample_value)[:30]})", className="text-muted")
                    ]), width=6),
                    dbc.Col(
                        dcc.Dropdown(
                            id={'type': 'feature-type-selector', 'feature': feature},
                            options=options,
                            placeholder="Sélectionner le type...",
                            clearable=False
                        ),
                        width=6
                    )
                ], className="mb-2")
            )
        return rows

    # Callback pour stocker les types de caractéristiques sélectionnés pour le clustering profond
    @app.callback(
        Output('deep-feature-types-store', 'data', allow_duplicate=True),
        Input({'type': 'feature-type-selector', 'feature': ALL}, 'value'),
        State('deep-cluster-features', 'value'),
        prevent_initial_call=True
    )
    def update_feature_types_store(selected_types, features):
        """
        Met à jour un dcc.Store avec la cartographie des caractéristiques vers leurs types sélectionnés
        pour le clustering profond.
        """
        if not features or None in selected_types:
            return {'feature_types': {}}

        feature_type_map = {
            feature: type_
            for feature, type_ in zip(features, selected_types)
        }
        return {'feature_types': feature_type_map}
    # =================================================================================
    # CALLBACK PRINCIPAL POUR LE CLUSTERING PROFOND
    # =================================================================================

    @app.callback(
        [Output('deep-clustering-result', 'data', allow_duplicate=True),
         Output('deep-clustering-graph', 'figure', allow_duplicate=True),
         Output('cluster-results-container', 'children', allow_duplicate=True),
         Output('status-alerts-container', 'children', allow_duplicate=True)],
        [Input('deep-clustering-button', 'n_clicks')],
        [State('processed-data', 'data'),
         State('deep-cluster-features', 'value'),
         State('deep-feature-types-store', 'data')],
        prevent_initial_call=True
    )
    def run_deep_clustering(n_clicks, data, features, feature_types_store):
        """
        Exécute le processus de clustering profond en utilisant la classe DeepClusteringModel.
        Affiche les résultats sous forme de tableau et de graphique UMAP.
        """
        if n_clicks is None or not features:
            raise PreventUpdate

        # Default outputs in case of error or no update
        default_error_outputs = (
            no_update,
            create_empty_figure('Erreur de visualisation'),
            html.Div(),
            dbc.Alert(f'Une erreur inattendue est survenue.', color="danger")
        )

        try:
            # 1. Retrieve and validate feature types
            feature_types = feature_types_store.get('feature_types', {})
            if not isinstance(feature_types, dict):
                raise ValueError("Structure de types de caractéristiques invalide. Attendu un dictionnaire.")
            missing_types = [col for col in features if col not in feature_types or feature_types[col] is None]
            if missing_types:
                raise ValueError(
                    f"Types de caractéristiques manquants ou non sélectionnés pour: {', '.join(missing_types)}. "
                    f"Veuillez sélectionner un type pour chaque caractéristique.")

            df_original = pd.DataFrame(data)

            # 2. Instantiate and run the DeepClusteringModel
            # Parameters like encoding_dim, epochs, batch_size are set during DeepClusteringModel initialization.
            # You can make these configurable via Dash UI States if needed.
            deep_model = DeepClusteringModel(
                encoding_dim=32,
                autoencoder_epochs=50,
                autoencoder_batch_size=256,
                random_state=42
            )
            result_df, umap_fig, optimal_k = deep_model.fit_predict(df_original, features, feature_types)

            # 3. Create Dash DataTable for deep clustering results
            cluster_table = dash_table.DataTable(
                data=result_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in result_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto', 'minWidth': '100%'},
                style_cell={'textAlign': 'left', 'padding': '8px'},
                filter_action="native",
                sort_action="native",
                export_format="csv"
            )

            # 4. Return results
            return (
                result_df.to_dict('records'),
                umap_fig,
                html.Div([
                    html.H5(f'Résultats du clustering profond (k = {optimal_k})', className='mt-3'),
                    cluster_table
                ]),
                dbc.Alert(f'Clustering profond réussi ! {optimal_k} clusters identifiés.', color="success")
            )

        except PreventUpdate:
            raise  # Re-raise PreventUpdate to stop the callback

        except Exception as e:
            logger.error(f"Erreur de deep clustering: {str(e)}", exc_info=True)
            return default_error_outputs  # Return predefined error outputs

    # =================================================================================
    # CALLBACK POUR LE TABLEAU DES RÉSULTATS GLOBAUX (CONSOLIDÉ)
    # =================================================================================

    @app.callback(
        [Output('global-results-table', 'data'),
         Output('global-results-table', 'columns'),
         Output('global-results-table', 'style_data_conditional')],
        [Input('cluster-data-table', 'data'),
         Input('text-clustering-result', 'data'),  # Use the dcc.Store for full text data
         Input('deep-clustering-result', 'data')]
    )
    def update_global_table(num_data, text_clustering_data, deep_data):
        """
        Combines clustering results (numerical, text, deep) into a single global table.
        Adds a 'Type' column to identify the clustering source.
        Applies conditional styling to cluster columns.
        Drops TSNE_X and TSNE_Y columns from text clustering data for global table clarity.
        """
        dfs = []

        if num_data:
            num_df = pd.DataFrame(num_data)
            num_df['Type'] = 'Numérique'
            dfs.append(num_df)

        if text_clustering_data:
            text_df = pd.DataFrame(text_clustering_data)
            # --- MODIFICATION START ---
            columns_to_drop = []
            if 'TSNE_X' in text_df.columns:
                columns_to_drop.append('TSNE_X')
            if 'TSNE_Y' in text_df.columns:
                columns_to_drop.append('TSNE_Y')

            if columns_to_drop:
                text_df = text_df.drop(columns=columns_to_drop)
            # --- MODIFICATION END ---

            text_df['Type'] = 'Textuel'
            dfs.append(text_df)

        if deep_data:
            deep_df = pd.DataFrame(deep_data)
            deep_df['Type'] = 'Mixte'
            dfs.append(deep_df)

        merged_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        columns = []
        if not merged_df.empty:
            # Add 'Type' column first
            columns.append({'name': 'Type', 'id': 'Type', 'deletable': False, 'selectable': False, 'hideable': False})

            # Add other columns, excluding 'Type'
            for col in merged_df.columns:
                if col != 'Type':
                    columns.append({'name': col, 'id': col, 'deletable': True})

        styles = []
        for col in merged_df.columns:
            if 'CLUSTER' in col.upper():
                styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': 'rgba(220, 220, 220, 0.3)',
                    'fontWeight': 'bold',
                    'color': 'black'
                })
            elif col == 'Type':
                styles.append({
                    'if': {'column_id': 'Type'},
                    'backgroundColor': 'rgba(173, 216, 230, 0.3)',
                    'fontWeight': 'bold',
                    'color': 'black'
                })

        return (
            merged_df.to_dict('records'),
            columns,
            styles
        )

    # NOUVEAU CALLBACK pour le téléchargement Excel
    @app.callback(
        Output("download-excel", "data"),
        Input("download-excel-btnn", "n_clicks"),
        [Input('global-results-table', 'data')],
        prevent_initial_call=True
    )
    def download_excel(n_clicks, table_data):
        """
        Génère et télécharge un fichier Excel contenant les données du tableau global.
        """
        if n_clicks is None or not table_data:
            return None

        # Convertir les données en DataFrame
        df = pd.DataFrame(table_data)

        # Créer un buffer en mémoire pour le fichier Excel
        output = io.BytesIO()

        # Utiliser ExcelWriter pour un meilleur contrôle du formatage
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Écrire les données dans une feuille
            df.to_excel(writer, sheet_name='Résultats_Clustering', index=False)

            # Optionnel : formatage du fichier Excel
            workbook = writer.book
            worksheet = writer.sheets['Résultats_Clustering']

            # Auto-ajuster la largeur des colonnes
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Limite la largeur maximale
                worksheet.column_dimensions[column_letter].width = adjusted_width

        # Préparer le téléchargement
        output.seek(0)

        return dcc.send_bytes(
            output.getvalue(),
            filename="resultats_clustering.xlsx",
            type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
