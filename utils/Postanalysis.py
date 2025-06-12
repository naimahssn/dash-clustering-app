import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, no_update
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import base64
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class EnhancedParetoAnalysis:
    """Classe enrichie pour gérer l'analyse de Pareto avec insights avancés"""

    def __init__(self, data, cluster_column='TEXT_CLUSTER'):
        self.data = data.copy()
        self.cluster_column = cluster_column
        self.numeric_columns = self._get_numeric_columns()
        self.text_columns = self._get_text_columns()
        self.analysis_history = []  # Historique des analyses

    def _get_numeric_columns(self):
        """Retourne la liste des colonnes numériques disponibles"""
        numeric_cols = []
        for col in self.data.columns:
            if col != self.cluster_column and self.data[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        return numeric_cols

    def _get_text_columns(self):
        """Retourne la liste des colonnes textuelles disponibles pour l'indexing"""
        text_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object' and col != self.cluster_column:
                text_cols.append(col)
        return text_cols

    def calculate_business_insights(self, filtered_data, variable_column, threshold_percent):
        """Calcule des insights business avancés"""
        insights = {}

        # 1. Analyse de concentration (Coefficient de Gini)
        values = filtered_data[variable_column].values
        gini_coeff = self._calculate_gini_coefficient(values)
        insights['concentration'] = {
            'gini_coefficient': gini_coeff,
            'interpretation': self._interpret_gini(gini_coeff)
        }

        # 2. Analyse des outliers
        outliers_info = self._analyze_outliers(filtered_data, variable_column)
        insights['outliers'] = outliers_info

        # 3. Analyse de stabilité des clusters
        cluster_stability = self._analyze_cluster_stability(filtered_data, variable_column)
        insights['cluster_stability'] = cluster_stability

        # 4. Potentiel d'optimisation
        optimization_potential = self._calculate_optimization_potential(filtered_data, variable_column,
                                                                        threshold_percent)
        insights['Écart de performance'] = optimization_potential

        # 5. Analyse comparative par cluster
        cluster_comparison = self._compare_clusters(filtered_data, variable_column)
        insights['cluster_comparison'] = cluster_comparison

        # 6. Recommandations stratégiques
        strategic_recommendations = self._generate_strategic_recommendations(insights, threshold_percent)
        insights['recommendations'] = strategic_recommendations

        return insights

    def _calculate_gini_coefficient(self, values):
        """Calcule le coefficient de Gini pour mesurer l'inégalité"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _interpret_gini(self, gini):
        """Interprète le coefficient de Gini"""
        if gini < 0.3:
            return "Faible concentration - Distribution relativement équitable"
        elif gini < 0.5:
            return "Concentration modérée - Quelques éléments dominent"
        elif gini < 0.7:
            return "Forte concentration - Principe de Pareto visible"
        else:
            return "Très forte concentration - Domination extrême de quelques éléments"

    def _analyze_outliers(self, data, variable_column):
        """Analyse les valeurs aberrantes et retourne le DataFrame des outliers."""
        Q1 = data[variable_column].quantile(0.25)
        Q3 = data[variable_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[variable_column] < lower_bound) | (data[variable_column] > upper_bound)].copy()

        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'upper_outliers': len(data[data[variable_column] > upper_bound]),
            'lower_outliers': len(data[data[variable_column] < lower_bound]),
            'impact': outliers[variable_column].sum() / data[variable_column].sum() * 100,
            'outlier_elements': outliers.to_dict('records') # Store outliers as list of dicts
        }

    def _analyze_cluster_stability(self, data, variable_column):
        """Analyse la stabilité des clusters"""
        cluster_stats = data.groupby(self.cluster_column)[variable_column].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)

        # Coefficient de variation par cluster
        cluster_stats['cv'] = (cluster_stats['std'] / cluster_stats['mean'] * 100).round(2)

        # Évaluation de la cohérence
        stability_score = 1 - (cluster_stats['cv'].std() / 100)  # Score de stabilité

        return {
            'cluster_stats': cluster_stats.to_dict('index'),
            'stability_score': max(0, stability_score),
            'most_stable_cluster': cluster_stats['cv'].idxmin(),
            'least_stable_cluster': cluster_stats['cv'].idxmax()
        }

    def _calculate_optimization_potential(self, data, variable_column, threshold_percent):
        """Calcule le potentiel d'optimisation"""
        total_sum = data[variable_column].sum()

        # Analyse des éléments sous-performants
        data_sorted = data.sort_values(variable_column, ascending=False)
        cumulative_pct = (data_sorted[variable_column].cumsum() / total_sum * 100)

        # Éléments dans les 20% inférieurs qui contribuent moins
        bottom_20_pct_threshold = len(data) * 0.8
        bottom_elements = data_sorted.iloc[int(bottom_20_pct_threshold):]

        potential_improvement = {
            'low_performers_count': len(bottom_elements),
            'low_performers_contribution': bottom_elements[variable_column].sum() / total_sum * 100,
            'average_low_performer_value': bottom_elements[variable_column].mean(),
            'improvement_target': data_sorted[variable_column].quantile(0.5),  # Médiane comme cible
        }

        # Potentiel d'amélioration si les éléments faibles atteignent la médiane
        current_bottom_sum = bottom_elements[variable_column].sum()
        target_bottom_sum = len(bottom_elements) * potential_improvement['improvement_target']
        potential_gain = target_bottom_sum - current_bottom_sum

        potential_improvement['potential_gain'] = potential_gain
        potential_improvement['potential_gain_percentage'] = potential_gain / total_sum * 100

        return potential_improvement

    def _compare_clusters(self, data, variable_column):
        """Compare les performances des clusters"""
        cluster_metrics = data.groupby(self.cluster_column).agg({
            variable_column: ['count', 'sum', 'mean', 'median', 'std']
        }).round(2)

        cluster_metrics.columns = ['Count', 'Total', 'Mean', 'Median', 'Std']
        cluster_metrics['Contribution_%'] = (cluster_metrics['Total'] / cluster_metrics['Total'].sum() * 100).round(2)
        cluster_metrics['Efficiency'] = (cluster_metrics['Total'] / cluster_metrics['Count']).round(2)

        # Classement des clusters
        cluster_metrics['Rank_by_Total'] = cluster_metrics['Total'].rank(ascending=False)
        cluster_metrics['Rank_by_Efficiency'] = cluster_metrics['Efficiency'].rank(ascending=False)

        return cluster_metrics.to_dict('index')

    def _generate_strategic_recommendations(self, insights, threshold_percent):
        """Génère des recommandations stratégiques basées sur l'analyse"""
        recommendations = []

        # Recommandations basées sur la concentration
        gini = insights['concentration']['gini_coefficient']
        if gini > 0.6:
            recommendations.append({
                'type': 'Concentration',
                'priority': 'Haute',
                'title': 'Forte concentration détectée',
                'description': f'Un indice de concentration relativement élevé est observé (Gini: {gini:.2f}). '
                           f'Les éléments représentant le top {threshold_percent}% génèrent une part importante de la valeur.',
                'action': 'Envisager une analyse approfondie de ces éléments pour évaluer leur rôle stratégique'
        })

        # Recommandations basées sur les outliers
        outliers = insights['outliers']
        if outliers['upper_outliers'] > 0:
            recommendations.append({
                'type': 'Outliers',
                'priority': 'Moyenne',
                'title': 'Éléments exceptionnels identifiés',
                'description': f'{outliers["upper_outliers"]} éléments exceptionnels représentent '
                               f'{outliers["impact"]:.1f}% de la valeur totale.',
                'action': 'Explorer ces éléments pour mieux comprendre les facteurs associés à leur performance'
            })

        # Recommandations basées sur le potentiel d'optimisation
        optimization = insights['Écart de performance']
        if optimization['potential_gain_percentage'] > 10:
            recommendations.append({
                'type': 'Écart de performance',
                'priority': 'Haute',
                'title': 'Fort potentiel d\'amélioration',
                 'description': f'{optimization["low_performers_count"]} éléments présentent des performances '
                       f'inférieures aux autres, avec un différentiel global estimé à {optimization["potential_gain_percentage"]:.1f}%.',
        'action': 'Il peut être pertinent d’examiner ces éléments afin d’en comprendre les spécificités'})

        # Recommandations basées sur la stabilité des clusters
        stability = insights['cluster_stability']
        if stability['stability_score'] < 0.7:
            recommendations.append({
                'type': 'Clusters',
                'priority': 'Moyenne',
                'title': 'Instabilité des clusters détectée',
                'description': f'Score de stabilité: {stability["stability_score"]:.2f}. '
                               f'Le cluster {stability["least_stable_cluster"]} est particulièrement instable.',
                'action': 'Revoir la segmentation ou analyser les causes de variabilité'
            })

        return recommendations

    def perform_enhanced_pareto_analysis(self, data_to_analyze, selected_clusters, variable_column,
                                         threshold_percent, display_columns=None, include_insights=True):
        """Effectue l'analyse de Pareto enrichie avec insights business"""
        # Analyse de base
        basic_results = self.perform_pareto_analysis(data_to_analyze, selected_clusters,
                                                     variable_column, threshold_percent, display_columns)

        if not basic_results or not include_insights:
            return basic_results

        # Données filtrées pour l'analyse
        if 'all' in selected_clusters:
            filtered_data = data_to_analyze.copy()
        else:
            cluster_ids = [int(c) for c in selected_clusters if c != 'all']
            filtered_data = data_to_analyze[data_to_analyze[self.cluster_column].isin(cluster_ids)].copy()

        filtered_data = filtered_data.dropna(subset=[variable_column])
        filtered_data = filtered_data[filtered_data[variable_column] > 0]

        # Calcul des insights avancés
        business_insights = self.calculate_business_insights(filtered_data, variable_column, threshold_percent)

        # Ajout des insights aux résultats
        basic_results['business_insights'] = business_insights
        basic_results['analysis_timestamp'] = datetime.now().isoformat()

        return basic_results

    def perform_pareto_analysis(self, data_to_analyze, selected_clusters, variable_column, threshold_percent,
                                display_columns=None):
        """Effectue l'analyse de Pareto de base (méthode originale)"""
        # [Code original de perform_pareto_analysis reste identique]
        if 'all' in selected_clusters:
            filtered_data = data_to_analyze.copy()
        else:
            cluster_ids = [int(c) for c in selected_clusters if c != 'all']
            filtered_data = data_to_analyze[data_to_analyze[self.cluster_column].isin(cluster_ids)].copy()

        if filtered_data.empty or variable_column not in filtered_data.columns:
            return None

        filtered_data = filtered_data.dropna(subset=[variable_column])
        filtered_data = filtered_data[filtered_data[variable_column] > 0]

        if filtered_data.empty:
            return None

        working_data = filtered_data.copy()
        working_data = working_data.sort_values(variable_column, ascending=False).reset_index(drop=True)

        total_sum = working_data[variable_column].sum()
        working_data['Valeur_Cumulative'] = working_data[variable_column].cumsum()
        working_data['Pourcentage_Cumulatif'] = (working_data['Valeur_Cumulative'] / total_sum * 100).round(2)
        working_data['Pourcentage_Individuel'] = (working_data[variable_column] / total_sum * 100).round(2)

        threshold_mask = working_data['Pourcentage_Cumulatif'] <= threshold_percent
        key_elements = working_data[threshold_mask].copy()

        if key_elements.empty and not working_data.empty:
            key_elements = working_data.iloc[:1].copy()

        return {
            'full_data': working_data.to_dict('records'),
            'key_elements': key_elements.to_dict('records'),
            'total_sum': float(total_sum),
            'threshold_percent': threshold_percent,
            'variable_column': variable_column,
            'display_columns': display_columns if display_columns else [],
            'num_key_elements': len(key_elements),
            'key_elements_contribution': float(key_elements[variable_column].sum() if not key_elements.empty else 0),
            'cluster_column': self.cluster_column
        }


def create_enhanced_pareto_analysis_layout():
    """Crée le layout enrichi pour l'analyse de Pareto"""
    return html.Div([
        # Contrôles de sélection et d'action
        dbc.Card([
            dbc.CardHeader(
                html.H4("Paramètres et Actions", className="text-primary mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Clusters à analyser:", className="form-label"),
                        html.Div(
                            dcc.Dropdown(
                                id='pareto-cluster-selector',
                                multi=True,
                                placeholder="Sélectionner les clusters...",
                                value=['all']
                            ),
                            style={"zIndex": 1050, "position": "relative"}
                        )
                    ], md=3),

                    dbc.Col([
                        html.Label("Variable quantitative:", className="form-label"),
                        html.Div(
                            dcc.Dropdown(
                                id='pareto-variable-selector',
                                placeholder="Choisir une variable..."
                            ),
                            style={"zIndex": 1040, "position": "relative"}
                        )
                    ], md=3),

                    dbc.Col([
                        html.Label("Colonnes à afficher:", className="form-label"),
                        html.Div(
                            dcc.Dropdown(
                                id='pareto-display-selector',
                                multi=True,
                                placeholder="Choisir des colonnes supplémentaires...",
                                clearable=True
                            ),
                            style={"zIndex": 1030, "position": "relative"}
                        )
                    ], md=3),

                    dbc.Col([
                        html.Label("Seuil cumulatif (%):", className="form-label"),
                        dcc.Slider(
                            id='pareto-threshold-slider',
                            min=50,
                            max=95,
                            step=5,
                            value=80,
                            marks={i: f"{i}%" for i in range(50, 100, 10)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=3)
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button(
                                "Analyser",
                                id='pareto-analyze-enhanced-btn',
                                color="primary",
                                size="sm"
                            ),
                        ], className="me-2"),

                        dbc.ButtonGroup([
                            dbc.Button(
                                "Exporter Rapport Complet",
                                id='pareto-export-report-btn',
                                color="success",
                                size="sm"
                            ),
                            dbc.Button(
                                "Exporter CSV",
                                id='pareto-export-csv-btn',
                                color="info",
                                size="sm",
                                outline=True
                            )
                        ])
                    ])
                ], className="mb-3"),

                # Section avancée (cachée par défaut)
                html.Div([
                    html.Hr(),
                    html.H5("Actions et Simulations Avancées", className="card-title mt-4"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Simulation d'amélioration:", className="form-label"),
                            dcc.Slider(
                                id='improvement-simulation-slider',
                                min=0,
                                max=50,
                                step=5,
                                value=10,
                                marks={i: f"+{i}%" for i in range(0, 51, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Small("Simuler l'amélioration des éléments sous-performants",
                                       className="form-text text-muted")
                        ], md=4),

                        dbc.Col([
                            html.Label("Seuil de détection d'outliers:", className="form-label"),
                            dcc.Slider(
                                id='outlier-threshold-slider',
                                min=1.0,
                                max=3.0,
                                step=0.5,
                                value=1.5,
                                marks={i: f"{i}" for i in [1.0, 1.5, 2.0, 2.5, 3.0]},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Small("Multiplicateur IQR pour détecter les outliers",
                                       className="form-text text-muted")
                        ], md=4),

                        dbc.Col([
                            dbc.Button(
                                "Analyser Impact",
                                id='impact-analysis-btn',
                                color="warning",
                                size="sm",
                                className="mt-4"
                            ),
                            dbc.Button(
                                "Simuler Scénarios",
                                id='scenario-simulation-btn',
                                color="info",
                                size="sm",
                                className="mt-4 ms-2"
                            )
                        ], md=4)
                    ])
                ], id="advanced-features-container", style={"display": "none"})
            ])
        ], className="mb-4", style={"minHeight": "300px", "overflow": "visible"}),

        # Zone de résultats enrichis
        html.Div(id='pareto-results-container'),

        # Stores pour les données et l'état
        dcc.Store(id='pareto-analysis-data'),
        dcc.Store(id='current-data-store'),
        dcc.Store(id='pareto-insights-data'),
        dcc.Store(id='simulation-results'),
        dcc.Download(id="pareto-download-report"),
        dcc.Download(id="pareto-download-csv")
    ])



def create_enhanced_pareto_results_layout(results):
    """Crée le layout enrichi des résultats avec insights business"""
    if not results:
        return dbc.Alert("Aucune donnée disponible pour l'analyse.", color="warning")

    # Layout de base
    basic_layout = create_pareto_results_layout(results)

    # Ajouter les insights si disponibles
    if 'business_insights' in results:
        insights_layout = create_business_insights_layout(results['business_insights'])

        # Combiner les layouts
        enhanced_layout = html.Div([
            # Dashboard des insights en premier
            insights_layout,
            html.Hr(className="my-4"),
            # Suivi par l'analyse détaillée
            basic_layout
        ])

        return enhanced_layout

    return basic_layout


def create_business_insights_layout(insights):
    """Crée le layout pour les insights business"""
    # Récupération des outliers
    outliers_data = insights['outliers']['outlier_elements']

    # Création du DataFrame (même vide)
    df_outliers = pd.DataFrame(outliers_data)

    # Génération des colonnes avec typage dynamique
    columns = [
        {
            "name": col,
            "id": col,
            "type": "numeric" if pd.api.types.is_numeric_dtype(df_outliers[col]) else "text"
        }
        for col in df_outliers.columns
    ] if not df_outliers.empty else []
    return html.Div([
        # Dashboard des métriques clés
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(" Tableau de Bord d’Analyse", className="card-title text-primary")
                    ])
                ])
            ], md=12)
        ], className="mb-4"),

        # Métriques de concentration
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🎯 Concentration", className="card-title"),
                        html.H4(f"{insights['concentration']['gini_coefficient']:.3f}",
                                className="text-primary mb-2"),
                        html.P("Index de Gini", className="text-muted small mb-1"),
                        html.P(insights['concentration']['interpretation'],
                               className="small text-success")
                    ])
                ])
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("⚡ Outliers", className="card-title"),
                        html.H4(f"{insights['outliers']['count']}",
                                className="text-warning mb-2"),
                        html.P("Éléments exceptionnels", className="text-muted small mb-1"),
                        html.P(f"Impact: {insights['outliers']['impact']:.1f}%",
                               className="small text-info")
                    ])
                ])
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🚀 Écart de performance", className="card-title"),
                        html.H4(f"+{insights['Écart de performance']['potential_gain_percentage']:.1f}%",
                                className="text-success mb-2"),
                        html.P("Écart de performance", className="text-muted small mb-1"),
                        html.P(f"{insights['Écart de performance']['low_performers_count']} éléments",
                               className="small text-secondary")
                    ])
                ])
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("⚖️ Stabilité", className="card-title"),
                        html.H4(f"{insights['cluster_stability']['stability_score']:.2f}",
                                className="text-info mb-2"),
                        html.P("Score de stabilité", className="text-muted small mb-1"),
                        html.P("Cohérence clusters", className="small text-secondary")
                    ])
                ])
            ], md=3)
        ], className="mb-4"),

        # Composant principal
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("🔍 Détail des Outliers", className="card-title"),
                        html.P(
                            f"Les éléments suivants ont été identifiés comme des outliers "
                            f"({insights['outliers']['count']} au total, représentant {insights['outliers']['impact']:.1f}% de la valeur).",
                            className="card-text mb-3"
                        ),
                        # Si des données existent, afficher le tableau
                        dash_table.DataTable(
                            id='outliers-table',
                            columns=columns,
                            data=df_outliers.to_dict('records'),
                            page_size=10,
                            filter_action="native",
                            sort_action="native",
                            export_format="xlsx",
                            export_headers="display",
                            style_table={
                                'overflowX': 'auto',
                                'border': '1px solid #dee2e6',
                                'minWidth': '100%',
                            },
                            style_cell={
                                'textAlign': 'left',
                                'padding': '8px',
                                'fontSize': '12px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'maxWidth': '200px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                            },
                            style_header={
                                'backgroundColor': '#f8f9fa',
                                'fontWeight': 'bold',
                                'borderBottom': '2px solid #dee2e6',
                                'textAlign': 'center',
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f9f9f9',
                                },
                                {
                                    'if': {'state': 'active'},
                                    'backgroundColor': '#e6f7ff',
                                    'border': '1px solid #91d5ff',
                                },
                            ],
                            style_as_list_view=True,
                        ) if not df_outliers.empty else html.Div(
                            "Aucun outlier détecté dans les données.",
                            className="text-muted fst-italic"
                        )
                    ])
                ])
            ], md=12)
        ], className="mb-4"),

        # Recommandations stratégiques
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(" Recommandations Stratégiques", className="card-title"),
                        html.Div([
                            create_recommendation_card(rec) for rec in insights['recommendations']
                        ])
                    ])
                ])
            ], md=8),
            # Analyse comparative des clusters
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Performance Clusters", className="card-title"),
                        create_cluster_comparison_chart(insights['cluster_comparison'])
                    ])
                ])
            ], md=4)
        ], className="mb-4")
    ])


def create_recommendation_card(recommendation):
    """Crée une carte de recommandation"""
    priority_colors = {
        'Haute': 'danger',
        'Moyenne': 'warning',
        'Faible': 'info'
    }

    return dbc.Alert([
        html.H6([
            dbc.Badge(recommendation['priority'],
                      color=priority_colors.get(recommendation['priority'], 'secondary'),
                      className="me-2"),
            recommendation['title']
        ], className="alert-heading"),
        html.P(recommendation['description'], className="mb-2"),
        html.Hr(),
        html.P([
            html.Strong("Action recommandée: "),
            recommendation['action']
        ], className="mb-0 small")
    ], color="light", className="mb-3")


def create_cluster_comparison_chart(cluster_comparison):
    """Crée un graphique de comparaison des clusters"""
    clusters = list(cluster_comparison.keys())
    contributions = [cluster_comparison[c]['Contribution_%'] for c in clusters]
    efficiencies = [cluster_comparison[c]['Efficiency'] for c in clusters]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Contribution %',
        x=clusters,
        y=contributions,
        yaxis='y',
        offsetgroup=1,
        marker_color='lightblue'
    ))

    fig.add_trace(go.Scatter(
        name='Efficacité',
        x=clusters,
        y=efficiencies,
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Performance par Cluster',
        xaxis_title='Clusters',
        yaxis=dict(title='Contribution (%)', side='left'),
        yaxis2=dict(title='Efficacité', side='right', overlaying='y'),
        height=300,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig





def create_optimization_chart(optimization_data):
    """Crée un graphique du potentiel d'optimisation"""
    categories = ['Performance Actuelle', 'Performance Cible', 'Gain Potentiel']
    current_value = 100 - optimization_data['low_performers_contribution']
    potential_value = current_value + optimization_data['potential_gain_percentage']
    gain_value = optimization_data['potential_gain_percentage']

    values = [current_value, potential_value, gain_value]
    colors = ['lightcoral', 'lightgreen', 'gold']

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Écart de performance',
        xaxis_title='Scénarios',
        yaxis_title='Performance (%)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def create_pareto_results_layout(results):
    """Crée le layout des résultats d'analyse de Pareto (version originale améliorée)"""
    if not results:
        return dbc.Alert("Aucune donnée disponible pour l'analyse.", color="warning")

    full_data = pd.DataFrame(results['full_data'])
    full_data['_original_index'] = full_data.index.values

    # Définir les colonnes du tableau avec amélioration de la présentation
    table_columns = []
    table_columns.append({'name': 'Rang', 'id': 'index'})

    display_columns = results.get('display_columns', [])
    if display_columns:
        for col in display_columns:
            if col in full_data.columns:
                table_columns.append({
                    'name': col,
                    'id': col,
                    'type': 'text' if full_data[col].dtype == 'object' else 'numeric'
                })

    table_columns.append({'name': 'Cluster', 'id': results['cluster_column']})
    table_columns.extend([
        {
            'name': f'{results["variable_column"]}',
            'id': results['variable_column'],
            'type': 'numeric',
            'format': Format(precision=2, scheme='f')
        },
        {
            'name': 'Cumul',
            'id': 'Valeur_Cumulative',
            'type': 'numeric',
            'format': Format(precision=2, scheme='f')
        },
        {
            'name': '% Individuel',
            'id': 'Pourcentage_Individuel',
            'type': 'numeric',
            'format': Format(precision=2, scheme='f')
        },
        {
            'name': '% Cumulé',
            'id': 'Pourcentage_Cumulatif',
            'type': 'numeric',
            'format': Format(precision=2, scheme='f')
        }
    ])

    table_data = full_data.reset_index()
    table_data['index'] = range(1, len(table_data) + 1)

    displayed_columns_text = ", ".join(display_columns) if display_columns else "Aucune"

    return html.Div([
        # Résumé enrichi des résultats
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📈 Résumé Exécutif", className="card-title text-primary"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5(f"{len(full_data)}", className="text-dark mb-1"),
                                    html.P("Éléments analysés", className="text-muted small mb-0")
                                ])
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.H5(f"{results['num_key_elements']}", className="text-success mb-1"),
                                    html.P(f"Éléments clés ({results['threshold_percent']}%)",
                                           className="text-muted small mb-0")
                                ])
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.H5(f"{results['key_elements_contribution']:,.0f}", className="text-info mb-1"),
                                    html.P("Valeur générée", className="text-muted small mb-0")
                                ])
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.H5(f"{results['key_elements_contribution'] / results['total_sum'] * 100:.1f}%",
                                            className="text-warning mb-1"),
                                    html.P("Contribution clés", className="text-muted small mb-0")
                                ])
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.H5(f"{results['total_sum']:,.0f}", className="text-secondary mb-1"),
                                    html.P("Total général", className="text-muted small mb-0")
                                ])
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.H5(f"{results['total_sum'] / len(full_data):,.0f}",
                                            className="text-dark mb-1"),
                                    html.P("Moyenne", className="text-muted small mb-0")
                                ])
                            ], md=2)
                        ])
                    ])
                ])
            ], md=12)
        ], className="mb-4"),

        # Graphique de Pareto amélioré
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(" Diagramme de Pareto Interactif", className="card-title"),
                        dcc.Graph(
                            id='pareto-chart',
                            figure=create_enhanced_pareto_chart(results)
                        )
                    ])
                ])
            ], md=12)
        ], className="mb-4"),

        # Tableau interactif amélioré
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("🔍 Analyse Détaillée", className="card-title"),
                        html.P([
                            "Colonnes supplémentaires: ",
                            dbc.Badge(displayed_columns_text, color="info", className="ms-1")
                        ], className="text-muted small mb-3"),
                        dash_table.DataTable(
                            id='pareto-full-table',
                            data=table_data.to_dict('records'),
                            columns=table_columns,
                            row_selectable='multi',
                            selected_rows=[],
                            style_cell={
                                'textAlign': 'left',
                                'fontSize': '12px',
                                'padding': '8px',
                                'fontFamily': 'Arial'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                },
                                {
                                    'if': {
                                        'filter_query': f'{{Pourcentage_Cumulatif}} <= {results["threshold_percent"]}',
                                    },
                                    'backgroundColor': '#d4edda',
                                    'color': 'black',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {'column_id': 'index'},
                                    'backgroundColor': '#f8f9fa',
                                    'fontWeight': 'bold'
                                }
                            ],
                            style_header={
                                'backgroundColor': '#343a40',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'fontSize': '12px',
                                'textAlign': 'center'
                            },
                            page_size=20,
                            sort_action="native",
                            filter_action="native",
                            style_table={'overflowX': 'auto'},
                            export_format='xlsx',
                            export_headers='display'
                        )
                    ])
                ])
            ], md=12)
        ], className="mb-4"),


    ])


def create_enhanced_pareto_chart(results):
    """Crée un graphique de Pareto amélioré avec plus d'interactivité"""
    full_data = pd.DataFrame(results['full_data'])
    variable_column = results['variable_column']
    threshold_percent = results['threshold_percent']
    display_columns = results.get('display_columns', [])

    # Limiter à 100 premiers éléments pour performance
    display_data = full_data.head(100).copy()

    # Créer les labels pour l'axe X
    if display_columns and len(display_columns) > 0:
        first_display_col = display_columns[0]
        if first_display_col in display_data.columns:
            x_labels = [str(val)[:20] + '...' if len(str(val)) > 20 else str(val)
                        for val in display_data[first_display_col]]
            x_title = f'{first_display_col} (top 100)'
        else:
            x_labels = [f"Élément {i + 1}" for i in range(len(display_data))]
            x_title = 'Éléments (classés par ordre décroissant)'
    else:
        x_labels = [f"Élément {i + 1}" for i in range(len(display_data))]
        x_title = 'Éléments (classés par ordre décroissant)'

    fig = go.Figure()

    # Barres avec gradient de couleur
    colors = ['#1f77b4' if pct <= threshold_percent else '#ff7f0e'
              for pct in display_data['Pourcentage_Cumulatif']]

    fig.add_trace(go.Bar(
        name=f'{variable_column}',
        x=list(range(len(display_data))),
        y=display_data[variable_column],
        yaxis='y',
        marker_color=colors,
        text=[f"{val:,.0f}" for val in display_data[variable_column]],
        textposition='outside',
        customdata=list(zip(x_labels, display_data['Pourcentage_Cumulatif'])),
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      f'{variable_column}: %{{y:,.2f}}<br>' +
                      '% Cumulé: %{customdata[1]:.2f}%<br>' +
                      '<extra></extra>'
    ))

    # Ligne de pourcentage cumulé avec zone colorée
    fig.add_trace(go.Scatter(
        name='% Cumulé',
        x=list(range(len(display_data))),
        y=display_data['Pourcentage_Cumulatif'],
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=6, color='red'),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        customdata=x_labels,
        hovertemplate='<b>%{customdata}</b><br>' +
                      '% Cumulé: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))

    # Ligne de seuil avec annotation améliorée
    fig.add_hline(
        y=threshold_percent,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text=f"Seuil {threshold_percent}% - Éléments Critiques",
        annotation_position="top right",
        yref="y2"
    )

    # Zone de focus (éléments clés)
    key_elements_count = len([x for x in display_data['Pourcentage_Cumulatif'] if x <= threshold_percent])
    if key_elements_count > 0:
        fig.add_vrect(
            x0=-0.5, x1=key_elements_count - 0.5,
            fillcolor="green", opacity=0.1,
            layer="below",
            annotation_text="Zone Critique",
            annotation_position="top left"
        )

    # Mise en forme avancée
    fig.update_layout(
        title={
            'text': f'Analyse de Pareto - {variable_column}<br><sub>Focus sur les éléments à fort impact</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title=x_title,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, len(display_data), max(1, len(display_data) // 20))),
            ticktext=[x_labels[i][:10] + '...' if len(x_labels[i]) > 10 else x_labels[i]
                      for i in range(0, len(display_data), max(1, len(display_data) // 20))],
            tickangle=45,
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=f'{variable_column}',
            side='left',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis2=dict(
            title='Pourcentage Cumulé (%)',
            side='right',
            overlaying='y',
            range=[0, 105],
            showgrid=False
        ),
        legend=dict(
            x=0.7, y=0.95,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=600,
        margin=dict(b=150, t=100),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    return fig


def calculate_pareto_rule(results):
    """Calcule si la règle 80/20 est respectée"""
    full_data = pd.DataFrame(results['full_data'])

    # Vérifier le pourcentage d'éléments qui génèrent 80% de la valeur
    threshold_80 = full_data['Pourcentage_Cumulatif'] <= 80
    elements_80 = len(full_data[threshold_80])
    percentage_elements = elements_80 / len(full_data) * 100

    if percentage_elements <= 25:
        return f"✅ Règle 80/20 respectée: {elements_80} éléments ({percentage_elements:.1f}%) génèrent 80% de la valeur"
    else:
        return f"⚠️ Distribution moins concentrée: {elements_80} éléments ({percentage_elements:.1f}%) pour 80% de la valeur"


def analyze_concentration(results):
    """Analyse le niveau de concentration"""
    num_elements = len(pd.DataFrame(results['full_data']))
    key_elements = results['num_key_elements']
    contribution = results['key_elements_contribution'] / results['total_sum'] * 100

    if contribution >= 80 and key_elements / num_elements <= 0.2:
        return f"🔴 Très forte concentration: {key_elements} éléments ({key_elements / num_elements * 100:.1f}%) = {contribution:.1f}%"
    elif contribution >= 70:
        return f" Forte concentration: Focus sur {key_elements} éléments prioritaires"
    else:
        return f" Distribution équilibrée: Attention répartie sur {key_elements} éléments"


def generate_quick_recommendation(results):
    """Génère une recommandation rapide"""
    contribution_pct = results['key_elements_contribution'] / results['total_sum'] * 100

    if contribution_pct >= 80:
        return f"🎯 Concentrez 80% de vos efforts sur les {results['num_key_elements']} premiers éléments"
    elif contribution_pct >= 60:
        return f"⚡ Optimisez d'abord les {results['num_key_elements']} éléments clés, puis élargissez"
    else:
        return f"📈 Approche équilibrée: Travaillez sur plusieurs fronts simultanément"


def get_enhanced_pareto_analysis_tab():
    """Retourne le layout de l'onglet d'analyse de Pareto enrichi"""
    return dbc.Container([
        create_enhanced_pareto_analysis_layout()
    ], fluid=True)


# Fonctions utilitaires pour les callbacks (à implémenter selon votre framework)

def generate_comprehensive_report(results):
    """Génère un rapport complet avec tous les insights"""
    if not results or 'business_insights' not in results:
        return None

    insights = results['business_insights']

    report = {
        'executive_summary': {
            'total_elements': len(pd.DataFrame(results['full_data'])),
            'key_elements': results['num_key_elements'],
            'concentration_level': insights['concentration']['gini_coefficient'],
            'optimization_potential': insights['Écart de performance']['potential_gain_percentage']
        },
        'detailed_analysis': {
            'pareto_compliance': calculate_pareto_rule(results),
            'concentration_analysis': analyze_concentration(results),
            'cluster_performance': insights['cluster_comparison'],
            'outliers_impact': insights['outliers']
        },
        'strategic_recommendations': insights['recommendations'],
        'action_plan': generate_action_plan(insights),
        'kpis_to_monitor': generate_kpis(results, insights)
    }

    return report


def generate_action_plan(insights):
    """Génère un plan d'action basé sur les insights"""
    actions = []

    # Actions basées sur les recommandations
    for rec in insights['recommendations']:
        if rec['priority'] == 'Haute':
            actions.append({
                'priority': 1,
                'action': rec['action'],
                'timeline': 'Court terme (1-3 mois)',
                'resources': 'Équipe dédiée',
                'kpi': 'Amélioration de la métrique principale'
            })
        elif rec['priority'] == 'Moyenne':
            actions.append({
                'priority': 2,
                'action': rec['action'],
                'timeline': 'Moyen terme (3-6 mois)',
                'resources': 'Ressources partagées',
                'kpi': 'Réduction de la variabilité'
            })

    return actions


def generate_kpis(results, insights):
    """Génère une liste de KPIs à surveiller"""
    kpis = [
        {
            'name': 'Indice de Concentration',
            'current_value': insights['concentration']['gini_coefficient'],
            'target': 'Maintenir > 0.6 pour forte concentration',
            'frequency': 'Mensuel'
        },
        {
            'name': 'Contribution Top Elements',
            'current_value': f"{results['key_elements_contribution'] / results['total_sum'] * 100:.1f}%",
            'target': f"Maintenir > {results['threshold_percent']}%",
            'frequency': 'Hebdomadaire'
        },
        {
            'name': 'Stabilité Clusters',
            'current_value': insights['cluster_stability']['stability_score'],
            'target': 'Maintenir > 0.7',
            'frequency': 'Mensuel'
        },
        {
            'name': 'Éléments Exceptionnels',
            'current_value': insights['outliers']['count'],
            'target': 'Analyser impact mensuel',
            'frequency': 'Mensuel'
        }
    ]

    return kpis