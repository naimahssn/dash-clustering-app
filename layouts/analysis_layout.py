from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from utils.Postanalysis import get_enhanced_pareto_analysis_tab


def create_analysis_layout():
    return html.Div([
        # Stockage des données
        dcc.Store(id="processed-data"),
        dcc.Store(id='deep-clustering-result'),
        dcc.Store(id='text-clustering-results'),
        dcc.Store(id='numerical-clustering-results'),
        dcc.Store(id='optimal-clusters-store'),
        dcc.Store(id='deep-feature-types-store'),
        dcc.Store(id='stored-clustering-data'),
        dcc.Store(id='stored-categorical-features'),
        dcc.Store(id='stored-color-map'),
        dcc.Store(id='stored-cluster-stats'),
        dcc.Store(id='stored-numeric-cols'),
        dcc.Store(id='text-clustering-result', data={}),
        html.Div(id='status-alerts-container', children=[],
                 style={'position': 'fixed', 'top': '10px', 'right': '10px', 'zIndex': '1000'}),

        # Conteneur d'alertes global
        dbc.Alert(
            id='status-alerts-container',
            children=[],
            is_open=False,
            duration=4000,
            dismissable=True,
            color="info",
            className="mt-3"
        ),

        # Barres de progression
        dbc.Progress(id='clustering-progress', animated=True, striped=True, style={'height': '3px'}),
        dbc.Progress(id='text-clustering-progress', animated=True, striped=True, style={'height': '3px'}),
        dbc.Progress(id='deep-clustering-progress', animated=True, striped=True, style={'height': '3px'}),

        # Principaux onglets d'analyse
        dbc.Tabs([
            # -----------------------------------------------
            # ONGLET CLUSTERING NUMÉRIQUE
            # -----------------------------------------------
            dbc.Tab(
                label="Clustering Numérique",
                tab_id="numerical-clustering-tab",
                children=[
                    dbc.Container(fluid=True, className="py-4", children=[
                        # Section Paramètres
                        dbc.Card(className="mb-4 shadow-sm", children=[
                            dbc.CardHeader(
                                html.H4("Paramètres de Clustering Numérique", className="text-primary mb-0")
                            ),
                            dbc.CardBody([
                                dbc.Row(className="g-3", children=[
                                    dbc.Col(md=4, children=[
                                        dbc.Label("Caractéristiques pour le Clustering",
                                                  className="fw-bold text-primary mb-2"),
                                        dcc.Dropdown(
                                            id='numerical-features',
                                            options=[],
                                            multi=True,
                                            placeholder="Sélectionner les variables à clusteriser",
                                            maxHeight=200,
                                            className="mb-2"
                                        ),
                                        dbc.FormText("Variables numériques à utiliser pour le clustering")
                                    ]),
                                    dbc.Col(md=4, children=[
                                        dbc.Label("Dimensions de Projection UMAP",
                                                  className="fw-bold text-primary mb-2"),
                                        dcc.Dropdown(
                                            id='projection-dim',
                                            options=[
                                                {'label': '2D', 'value': 2},
                                                {'label': '3D', 'value': 3}
                                            ],
                                            value=2,
                                            clearable=False,
                                            className="mb-2"
                                        ),
                                        dbc.FormText("Type de projection pour la visualisation")
                                    ]),
                                    dbc.Col(md=4, children=[
                                        dbc.Label("Taille Minimale de Cluster (HDBSCAN)",
                                                  className="fw-bold text-primary mb-2"),
                                        dcc.Input(
                                            id='min-cluster-size',
                                            type='number',
                                            min=2,
                                            value=5,
                                            step=1,
                                            className="form-control mb-2"
                                        ),
                                        dbc.FormText("Nombre minimum d'échantillons dans un cluster.")
                                    ]),
                                ]),
                                dbc.Row(className="g-3 mt-3", children=[
                                    dbc.Col(md=12, className="d-flex align-items-end", children=[
                                        dbc.Button(
                                            "Lancer le Clustering Numérique",
                                            id='numerical-cluster-btn',
                                            color="primary",
                                            className="w-100",
                                            size="lg"
                                        )
                                    ])
                                ])
                            ])
                        ]),

                        # Section Visualisations et Métriques
                        dbc.Row(className="g-4", children=[
                            dbc.Col(md=8, children=[
                                dbc.Card(className="shadow-sm", children=[
                                    dbc.CardHeader(
                                        html.H4("Visualisations des Clusters", className="text-primary mb-0")
                                    ),
                                    dbc.CardBody([
                                        dbc.Tabs(
                                            id="cluster-visualization-tabs",
                                            active_tab="umap",
                                            className="mb-3",
                                            children=[
                                                dbc.Tab(label="Projection UMAP", tab_id="umap"),
                                                dbc.Tab(label="Distribution des Caractéristiques",
                                                        tab_id="distribution"),
                                                dbc.Tab(label="Importance des Caractéristiques", tab_id="importance"),
                                                dbc.Tab(label="Tailles des Clusters", tab_id="cluster-sizes"),
                                                dbc.Tab(label="Statistiques des Clusters", tab_id="cluster-stats"),
                                                dbc.Tab(label="Analyse Approfondie", tab_id="detailed_analysis")
                                            ]
                                        ),
                                        html.Div(id="cluster-visualization-content", children=[
                                            dcc.Graph(id='numerical-cluster-graph', style={'height': '600px'}),
                                            dcc.Graph(id='numerical-cluster-distribution',
                                                      style={'height': '600px', 'display': 'none'}),
                                            dcc.Graph(id='numerical-cluster-importance',
                                                      style={'height': '600px', 'display': 'none'}),
                                            dcc.Graph(id='cluster-sizes-graph',
                                                      style={'height': '600px', 'display': 'none'}),
                                            dcc.Graph(id='interactive-cluster-stats-table',
                                                      style={'height': '600px', 'display': 'none'}),
                                            html.Div(id='detailed-analysis-content',
                                                     style={'display': 'none'}, children=[
                                                    dbc.Row(className="g-3 mb-4", children=[
                                                        dbc.Card([
                                                            dbc.CardHeader("Distributions Catégorielles par Cluster"),
                                                            dbc.CardBody([
                                                                html.Div([
                                                                    dbc.Label("Sélectionnez un Cluster:",
                                                                              className="fw-bold mb-2"),
                                                                    dcc.Dropdown(
                                                                        id="categorical-cluster-selector",
                                                                        options=[],
                                                                        placeholder="Sélectionnez un cluster...",
                                                                        clearable=False,
                                                                        className="mb-3"
                                                                    )
                                                                ]),
                                                                html.Div([
                                                                    dbc.Label(
                                                                        "Sélectionnez une Caractéristique Catégorielle:",
                                                                        className="fw-bold mb-2"),
                                                                    dcc.Dropdown(
                                                                        id="categorical-features-dropdown",
                                                                        options=[],
                                                                        placeholder="Sélectionnez une caractéristique...",
                                                                        clearable=False,
                                                                        className="mb-3"
                                                                    )
                                                                ]),
                                                                dcc.Graph(id='categorical-distribution-graph')
                                                            ])
                                                        ])
                                                    ])
                                                ])
                                        ])
                                    ])
                                ])
                            ]),

                            dbc.Col(md=4, children=[
                                dbc.Card(className="shadow-sm h-100", children=[
                                    dbc.CardHeader(
                                        html.H4("Résumé et Qualité", className="text-primary mb-0")
                                    ),
                                    dbc.CardBody([
                                        html.Div(id="cluster-summary-output"),
                                        html.Hr(),
                                    ])
                                ]),
                            ])
                        ]),

                        # Section Données Clusterisées
                        dbc.Card(className="mt-4 shadow-sm", children=[
                            dbc.CardHeader(
                                html.H4("Données Clusterisées", className="text-primary mb-0")
                            ),
                            dbc.CardBody([
                                dcc.Dropdown(
                                    id='cluster-table-filter',
                                    options=[],
                                    multi=True,
                                    placeholder="Filtrer les données...",
                                    className="mb-3"
                                ),
                                dash_table.DataTable(
                                    id='cluster-data-table',
                                    page_size=10,
                                    style_table={'maxHeight': '600px', 'overflowX': 'auto',
                                                 'border': '1px solid #eee', 'borderRadius': '5px'},
                                    style_cell={'textAlign': 'left', 'padding': '12px'},
                                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
                                )
                            ])
                        ])
                    ])
                ]
            ),

            # -----------------------------------------------
            # ONGLET CLUSTERING TEXTUEL
            # -----------------------------------------------
            dbc.Tab(label="Clustering Textuel", children=[
                dbc.Container(fluid=True, className="py-4", children=[
                    dbc.Card(className="mb-4 shadow-sm", children=[
                        dbc.CardHeader(
                            html.H4("Paramètres Sémantiques", className="text-primary mb-0")
                        ),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(md=6, children=[
                                    dbc.Label("Colonnes Textuelles", className="fw-bold text-primary mb-2"),
                                    dcc.Dropdown(
                                        id='text-features',
                                        options=[
                                            {'label': 'Description du Produit', 'value': 'description_produit'},
                                            {'label': 'Commentaires Clients', 'value': 'commentaires_clients'},
                                            {'label': 'Résumé Article', 'value': 'resume_article'}
                                        ],
                                        multi=True,
                                        placeholder="Sélectionnez une ou plusieurs colonnes...",
                                        className='mb-3',
                                        persistence=True,
                                        persistence_type='session'
                                    ),
                                    dbc.FormText("Combinaison automatique des colonnes sélectionnées",
                                                 color="secondary")
                                ]),
                                dbc.Col(md=6, children=[
                                    dbc.Label("Plage de Thématiques (K) pour la détection automatique :",
                                              className="fw-bold text-primary mb-2"),
                                    dbc.FormText("Le nombre optimal sera détecté entre ces valeurs.",
                                                 className="text-muted d-block mb-2"),
                                    dbc.Row([
                                        dbc.Col(md=6, children=[
                                            dbc.Input(
                                                id='min-clusters-auto',
                                                type='number',
                                                value=2,
                                                min=2,
                                                placeholder="Min K",
                                                className='form-control mb-3'
                                            )
                                        ]),
                                        dbc.Col(md=6, children=[
                                            dbc.Input(
                                                id='max-clusters-auto',
                                                type='number',
                                                value=10,
                                                min=2,
                                                placeholder="Max K",
                                                className='form-control mb-3'
                                            )
                                        ])
                                    ]),
                                    dbc.FormText(
                                        "Minimum 2 clusters. Il est recommandé de ne pas choisir une plage trop large pour des raisons de performance.",
                                        color="secondary")
                                ])
                            ]),

                            dbc.Button(
                                "Analyser les Thématiques",
                                id='text-cluster-btn',
                                color="primary",
                                className="w-100 mt-3",
                                size="lg"
                            ),

                            # Spinner wraps the entire statistics section
                            dbc.Spinner(
                                html.Div(id='cluster-statistics-content-wrapper', className="mt-4", children=[
                                    html.Div(className="bg-light p-3 rounded mb-3", children=[
                                        html.H5("Récapitulatif du Clustering", className="text-primary mb-3"),
                                        html.P(id='total-documents-clustered',
                                               children="Nombre total de documents analysés : N/A"),
                                        html.P(id='num-clusters-found',
                                               children="Nombre de clusters trouvés : N/A"),
                                    ]),
                                    html.Hr(),
                                    html.H6("Tableau des Statistiques par Cluster", className="text-primary mb-3"),
                                    dash_table.DataTable(
                                        id='stats-table',
                                        data=[],
                                        columns=[
                                            {'name': 'Cluster', 'id': 'Cluster'},
                                            {'name': 'Nombre de documents', 'id': 'Nombre de documents',
                                             'type': 'numeric'},
                                            {'name': 'Pourcentage', 'id': 'Pourcentage'},
                                            {'name': 'Densité relative', 'id': 'Densité relative', 'type': 'numeric'}
                                        ],
                                        sort_action="native",
                                        style_table={'overflowX': 'auto', 'border': '1px solid #eee',
                                                     'borderRadius': '5px'},
                                        style_cell={
                                            'textAlign': 'left',
                                            'minWidth': '120px', 'maxWidth': '200px', 'whiteSpace': 'normal',
                                            'padding': '12px'
                                        },
                                        style_header={
                                            'backgroundColor': '#f8f9fa',
                                            'fontWeight': 'bold'
                                        },
                                        style_data_conditional=[
                                            {
                                                'if': {'column_id': 'Cluster'},
                                                'fontWeight': 'bold'
                                            },
                                        ]
                                    ),
                                    html.Hr(),
                                ]),
                                color="primary",
                                size="lg"
                            )
                        ])
                    ]),

                    dbc.Card(className="mb-4 shadow-sm", children=[
                        dbc.CardHeader(
                            html.H4("Diagnostic de la Détection Automatique de K", className="text-primary mb-0")
                        ),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(md=6, children=[
                                    html.H5("Méthode du Coude (Inertie)", className="text-secondary mb-3"),
                                    dcc.Graph(id='elbow-method-chart', figure=go.Figure(), style={'height': '400px'})
                                ]),
                                dbc.Col(md=6, children=[
                                    html.H5("Score de Silhouette", className="text-secondary mb-3"),
                                    dcc.Graph(id='silhouette-score-chart', figure=go.Figure(),
                                              style={'height': '400px'})
                                ])
                            ])
                        ])
                    ]),

                    dbc.Card(className="mb-4 shadow-sm", children=[
                        dbc.CardHeader(
                            html.H4("Filtrage Global des Clusters", className="text-primary mb-0")
                        ),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(md=8, children=[
                                    dbc.Label("Sélectionner les Clusters à Afficher",
                                              className="fw-bold text-primary mb-2"),
                                    dcc.Dropdown(
                                        id='global-cluster-filter',
                                        options=[{'label': 'Tous les clusters', 'value': 'all'}],
                                        value=['all'],
                                        multi=True,
                                        placeholder="Filtrer par cluster...",
                                        className='mb-2',
                                        persistence=True,
                                        persistence_type='session'
                                    ),
                                ]),
                            ]),
                            html.Div(id='filter-status-indicator', className="mt-2"),
                        ])
                    ]),

                    dbc.Row(className="g-4 mt-3", children=[
                        dbc.Col(md=4, children=[
                            dbc.Card(className="shadow-sm h-100", children=[
                                dbc.CardHeader([
                                    html.Span([
                                        "Mots-clés par Thématique ",
                                        dbc.Badge("Nouveau", color="success", className="ms-2")
                                    ]),
                                ]),
                                dbc.CardBody([
                                    html.Div(
                                        id='keywords-table-container',
                                        children=[
                                            dash_table.DataTable(
                                                id='keywords-table',
                                                data=[],
                                                columns=[
                                                    {'name': 'Cluster', 'id': 'Cluster'},
                                                    {'name': 'Mots-clés représentatifs',
                                                     'id': 'Mots-clés représentatifs'}
                                                ],
                                                filter_action="native",
                                                sort_action="native",
                                                style_table={'overflowX': 'auto', 'border': '1px solid #eee',
                                                             'borderRadius': '5px'},
                                                style_cell={
                                                    'textAlign': 'left',
                                                    'minWidth': '150px', 'maxWidth': '500px', 'whiteSpace': 'normal',
                                                    'padding': '12px'
                                                },
                                                style_header={
                                                    'backgroundColor': '#f8f9fa',
                                                    'fontWeight': 'bold'
                                                },
                                            )
                                        ]
                                    )
                                ]),
                                dbc.CardFooter([
                                    dbc.Button(
                                        "Exporter les Mots-clés",
                                        id="download-keywords-btn",
                                        color="success",
                                        className="w-100",
                                        n_clicks=0
                                    ),
                                    dcc.Download(id="download-keywords")
                                ])
                            ])
                        ]),

                        dbc.Col(md=8, children=[
                            dbc.Card(className="shadow-sm h-100", children=[
                                dbc.CardHeader([
                                    html.Span([
                                        "Analyse Multidimensionnelle et Corpus ",
                                        dbc.Badge("Viz", color="info", className="ms-2")
                                    ]),
                                ]),
                                dbc.CardBody([
                                    dbc.Tabs([
                                        dbc.Tab(label="Visualisation t-SNE", children=[
                                            html.Div(
                                                dcc.Graph(id='cluster-scatter-plot', figure=go.Figure(),
                                                          style={'height': '600px'}),
                                                className="mt-3"
                                            )]),
                                        dbc.Tab(label="Distribution des Clusters", children=[
                                            html.Div(
                                                dcc.Graph(id='cluster-distribution-chart', figure=go.Figure(),
                                                          style={'height': '600px'}),
                                                className="mt-3"
                                            )
                                        ]),
                                        dbc.Tab(label="Corpus Analysé", children=[
                                            html.Div(
                                                id='text-cluster-results-container',
                                                children=[
                                                    dash_table.DataTable(
                                                        id='text-results-table',
                                                        data=[],
                                                        columns=[],
                                                        filter_action="native",
                                                        sort_action="native",
                                                        page_size=10,
                                                        style_table={'overflowX': 'auto', 'maxHeight': '500px',
                                                                     'border': '1px solid #eee', 'borderRadius': '5px'},
                                                        style_header={
                                                            'backgroundColor': '#f8f9fa',
                                                            'fontWeight': 'bold'
                                                        },
                                                        style_cell={
                                                            'minWidth': '100px', 'maxWidth': '300px',
                                                            'whiteSpace': 'normal',
                                                            'textAlign': 'left',
                                                            'padding': '12px'
                                                        },
                                                        filter_options={"case": "insensitive"},
                                                    )
                                                ]
                                            ),
                                            html.Div([
                                                html.Button(
                                                    [html.I(className="fas fa-download me-2"),
                                                     "Télécharger le Rapport Excel"],
                                                    id="download-excel-btn",
                                                    className="btn btn-success btn-lg",
                                                    disabled=True
                                                ),
                                                dcc.Download(id="download-excel-report")
                                            ])
                                        ])

                                    ])
                                ])
                            ])
                        ])
                    ]),

                ])
            ]),
            dbc.Tab(label="Post Analysis", tab_id="pareto-analysis", children=[
                get_enhanced_pareto_analysis_tab()
            ]),
            dbc.Tab(label="Clustering Profond", children=[
                dbc.Container(fluid=True, className="py-4", children=[
                    dbc.Card(className="mb-4 shadow-sm", children=[
                        dbc.CardHeader(
                            html.H4("Paramètres Avancés", className="text-primary mb-0")
                        ),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(md=6, children=[
                                    dbc.Label("Caractéristiques", className="fw-bold text-primary mb-2"),
                                    dcc.Dropdown(
                                        id='deep-cluster-features',
                                        options=[],
                                        multi=True,
                                        placeholder="Sélectionner colonnes",
                                        maxHeight=150,
                                        className="mb-3"
                                    ),
                                    html.Div(id='feature-type-selection-ui', className="mt-3")
                                ]),
                            ]),
                            dbc.Button(
                                "Démarrer l'Apprentissage Profond",
                                id='deep-clustering-button',
                                color="primary",
                                className="w-100 mt-4",
                                size="lg"
                            )
                        ])
                    ]),
                    dbc.Card(className="shadow-sm", children=[
                        dbc.CardHeader(
                            html.H4("Visualisation des Résultats", className="text-primary mb-0")
                        ),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-deep",
                                type="circle",
                                children=dcc.Graph(
                                    id='deep-clustering-graph',
                                    config={'displayModeBar': True},
                                    style={'height': '600px'}
                                )
                            )
                        ])
                    ]),
                    dbc.Card(className="shadow-sm mt-4", children=[
                        dbc.CardHeader(
                            html.H4("Résultats Détaillés", className="text-primary mb-0")
                        ),
                        dbc.CardBody([
                            html.Div(
                                id='cluster-results-container',
                                style={'maxHeight': '500px', 'overflowY': 'auto'}
                            )
                        ])
                    ])
                ])
            ]),
        ]),

        # Section des Résultats Globaux
        dbc.Card(className="shadow-sm mt-4", children=[
            dbc.CardHeader(
                html.H4("Synthèse des Analyses", className="text-primary mb-0")
            ),
            dbc.CardBody([
                html.Div([
                    html.Button(
                        "Télécharger en Excel",
                        id="download-excel-btnn",
                        style={
                            'marginBottom': '15px',
                            'padding': '10px 20px',
                            'backgroundColor': '#28a745',  # vert Bootstrap
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }
                    ),
                    dcc.Download(id="download-excel")
                ]),

                dash_table.DataTable(
                    id='global-results-table',
                    columns=[
                        {"name": "ID", "id": "ID", "deletable": False, "selectable": False},
                        {"name": "Type", "id": "Type", "deletable": False, "selectable": False},
                    ],
                    data=[],
                    page_size=10,
                    filter_action="native",
                    sort_action="native",
                    style_table={'overflowX': 'auto', 'maxHeight': '600px',
                                 'border': '1px solid #eee', 'borderRadius': '5px'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '12px',
                        'minWidth': '80px', 'width': '120px', 'maxWidth': '200px',
                        'whiteSpace': 'normal'
                    },
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold'
                    },
                    tooltip_delay=0,
                    tooltip_duration=None,
                )
            ]),
        ])
    ])