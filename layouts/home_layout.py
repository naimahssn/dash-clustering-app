from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable

def create_home_layout():
    return html.Div([
        # Section d'upload de données
        dbc.Card([
            dbc.CardHeader(html.H4('Étape 1 : Importation des données', className="mb-0")),
            dbc.CardBody([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt me-2"),
                        'Glissez-déposez un fichier ou cliquez ici'
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                dcc.Store(id='stored-sheets'),
                dcc.Dropdown(id='sheet-selection-dropdown', placeholder="Sélectionnez une feuille",
                             style={'marginTop': '10px'}),
                html.Div(id='upload-status-message', className="mt-2"),
                html.Div([
                    dbc.Label("Choix de la feuille"),
                    dcc.Dropdown(
                        id='sheet-selector',
                        options=[],
                        placeholder="Sélectionner une feuille",
                        className="mb-3"
                    )
                ], id='sheet-selector-container', style={"display": "none"}),
                dbc.Spinner(html.Div(id='data-preview-container'), color="primary")
            ])
        ], className="mb-4 shadow-sm"),

        # Section de calcul
        dbc.Card([
            dbc.CardHeader(html.H4('Étape 2 : Configuration', className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Colonne Montant Initial', className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='montant-initial-col',
                            options=[],
                            placeholder='Sélectionnez une colonne',
                            clearable=False,
                            className="mb-3"
                        )
                    ], md=6),
                    dbc.Col([
                        dbc.Label('Colonne Montant Engagé', className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='montant-engage-col',
                            options=[],
                            placeholder='Sélectionnez une colonne',
                            clearable=False
                        )
                    ], md=6)
                ], className="mb-4"),

                dbc.Button(
                    [html.I(className="fas fa-calculator me-2"), 'Lancer le calcul'],
                    id='calculate-btn',
                    color="primary",
                    className="w-100 mb-3",
                    disabled=False
                ),
                html.Div(id='status-alerts-container', className='mt-3'),
                dbc.Spinner(html.Div(id='calculation-spinner'), color="primary")
            ])
        ], className="mb-4 shadow-sm"),

        # Section KPI
        dbc.Card([
            dbc.CardHeader(html.H4("Indicateurs Clés", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(create_summary_card("Total Montant Initial", "total-initial", "primary"), md=3),
                    dbc.Col(create_summary_card("Total Montant Engagé", "total-committed", "success"), md=3),
                    dbc.Col(create_summary_card("Nombre de projets", "total-projects", "warning"), md=3),
                    dbc.Col(create_summary_card("% Taux d'engagement moyen", "avg-engagement", "info"), md=3),
                ], className="mb-4"),
            ])
        ], className="mb-4 shadow-sm"),

        # Section Résultats
        dbc.Card([
            dbc.CardHeader(html.H4('Résultats de l\'analyse', className="mb-0")),
            dbc.CardBody([
                html.Div(
                    id='results-table-container',
                    style={
                        'maxHeight': '600px',
                        'overflowY': 'auto',
                        'border': '1px solid #eee',
                        'borderRadius': '5px'
                    }
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), 'Exporter les résultats'],
                            id='export-results-btn',
                            color="success",
                            className="w-100 mt-3"
                        ),
                        md=4
                    )
                ], className="mb-3"),
                dcc.Download(id="download-dataframe-xlsx"),

            ])
        ], className="mb-4 shadow-sm"),
        dcc.Store(id='processed-data')
    ])

def create_summary_card(title: str, element_id: str, color: str) -> dbc.Card:
    """Crée une carte de résumé KPI"""
    return dbc.Card([
        dbc.CardHeader(title, className="small"),
        dbc.CardBody([
            html.Div(id=element_id, className="h4 mb-0")
        ])
    ], color=color, inverse=True, className="text-center shadow-sm")