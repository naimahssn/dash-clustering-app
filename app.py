from dash import Dash, html, dcc, callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import os
from flask_caching import Cache
import logging

# Import des layouts
from layouts.home_layout import create_home_layout
from layouts.analysis_layout import create_analysis_layout

# Import des callbacks
from callbacks.data_callbacks import register_data_callbacks
from callbacks.clustering_callbacks import register_clustering_callbacks
from callbacks.ui_callbacks import register_ui_callbacks

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
            "/assets/custom.css"
        ],
        suppress_callback_exceptions=True,
        meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
        assets_folder="assets",
        background_callback_manager=True
    )

    server = app.server
    app.title = "Outil d'Analyse de Projet"

    # Configuration du cache
    cache = Cache()
    cache_config = {
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': os.path.join(os.path.dirname(__file__), 'cache'),
        'CACHE_DEFAULT_TIMEOUT': 3600,
        'CACHE_THRESHOLD': 100
    }
    cache.init_app(server, config=cache_config)

    # Layout principal
    app.layout = dbc.Container([
        # Stores
        dcc.Store(id='stored-data', storage_type='memory'),
        dcc.Store(id='processed-data', storage_type='memory'),
        dcc.Store(id='numerical-clustering-results', storage_type='memory'),
        dcc.Store(id='text-clustering-results', storage_type='memory'),
        dcc.Store(id='deep-clustering-result', storage_type='memory'),
        dcc.Store(id='upload-status-store', storage_type='session'),
        dcc.Store(id='optimal-clusters-store', storage_type='memory'),
        dcc.Store(id='user-settings', storage_type='local'),
        dcc.Store(id='saved-clusters', storage_type='local'),
        dcc.Store(id='feature-types-store', storage_type='memory'),

        # Navbar
        dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.Img(src="/assets/logo.png", height="40px")),
                        dbc.Col(dbc.NavbarBrand("Outil d'Analyse de Projet", className="ms-2 fw-bold")),
                    ], align="center", className="g-0"),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink(
                            [html.I(className="fas fa-upload me-2"), "Importation Données"],
                            href="/", id="nav-link-1", className="nav-link-hover"
                        )),
                        dbc.NavItem(dbc.NavLink(
                            [html.I(className="fas fa-chart-line me-2"), "Analyse Avancée"],
                            href="/analyse", id="nav-link-2", className="nav-link-hover"
                        )),
                    ], className="ms-auto", navbar=True),
                    id="navbar-collapse",
                    navbar=True,
                    is_open=False,
                ),
            ], fluid=True),
            color="secondary",
            dark=False,
            sticky="top",
            className="shadow-sm navbar-custom"  # Classe personnalisée
        ),

        # Floating alerts
        html.Div(
            id='status-alerts-container',
            className="mt-3",
            style={
                'position': 'fixed',
                'top': '80px',
                'right': '20px',
                'zIndex': 1050,
                'maxWidth': '400px'
            }
        ),

        # Progress bars
        html.Div([
            dbc.Progress(id='global-progress', color="info", style={'height': '3px'}, className="mb-0"),
            dbc.Progress(id='clustering-progress', color="primary", animated=True, striped=True, style={'height': '3px'}, className="mb-0"),
            dbc.Progress(id='text-clustering-progress', color="success", animated=True, striped=True, style={'height': '3px'}, className="mb-0"),
            dbc.Progress(id='deep-clustering-progress', color="warning", animated=True, striped=True, style={'height': '3px'}, className="mb-0")
        ], style={
            'position': 'fixed',
            'top': '60px',
            'width': '100%',
            'zIndex': 1030
        }),

        # Routing + content
        dcc.Location(id='url', refresh=False),

        html.Div(
            id='page-content',
            className="pt-5",
            style={
                'backgroundColor': '#fefefe',
                'minHeight': 'calc(100vh - 120px)',
                'padding': '20px',
                'paddingTop': '100px',
                'color': '#212529'
            }
        )
    ], fluid=True, style={
        'padding': '0',
        'paddingTop': '80px',
        'backgroundColor': '#f6f9fc',
        'color': '#212529'
    })

    register_all_callbacks(app)

    server.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    server.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

    return app


def register_all_callbacks(app):
    try:
        register_data_callbacks(app)
        register_clustering_callbacks(app)
        register_ui_callbacks(app)

        @app.callback(
            Output('page-content', 'children'),
            [Input('url', 'pathname')],
            [State('user-settings', 'data')],
            prevent_initial_call=True
        )
        def display_page(pathname, settings):
            if pathname == '/analyse':
                return create_analysis_layout()
            return create_home_layout()

        @app.callback(
            [Output('nav-link-1', 'active'),
             Output('nav-link-2', 'active'),
             Output('navbar-collapse', 'is_open')],
            [Input('url', 'pathname'),
             Input('navbar-toggler', 'n_clicks')],
            [State('navbar-collapse', 'is_open')]
        )
        def update_nav_links(pathname, n_clicks, is_open):
            ctx = callback_context
            if not ctx.triggered:
                return True, False, False
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'navbar-toggler':
                return pathname != '/analyse', pathname == '/analyse', not is_open
            return pathname != '/analyse', pathname == '/analyse', False

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement des callbacks: {str(e)}", exc_info=True)


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=8051)
