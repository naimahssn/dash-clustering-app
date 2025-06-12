from dash import Input, Output, State, callback, clientside_callback, callback_context
import dash_bootstrap_components as dbc
from config import Config
import logging

logger = logging.getLogger(__name__)


def register_ui_callbacks(app):
    """Enregistre les callbacks liés à l'interface utilisateur"""

    @app.callback(
        [Output('user-guide-modal', 'is_open'),
         Output('language-store', 'data')],
        [Input('user-guide-link', 'n_clicks'),
         Input('close-guide', 'n_clicks')],
        [State('user-guide-modal', 'is_open'),
         State('language-store', 'data')]
    )
    def toggle_modal_and_language(guide_open, guide_close, is_open, lang_data):
        ctx = callback_context
        lang = lang_data.get('language', Config.DEFAULT_LANGUAGE)

        if not ctx.triggered:
            return False, lang_data

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Gestion du modal
        if trigger_id in ['user-guide-link', 'close-guide']:
            return not is_open, lang_data

    @app.callback(
        [Output('nav', 'children'),
         Output('url', 'pathname')],
        [Input('language-store', 'modified_timestamp')],
        [State('language-store', 'data'),
         State('url', 'pathname')]
    )
    def update_language(ts, lang_data, current_path):
        """Met à jour la langue de l'interface"""
        lang = "fr"

        nav_links = [
            dbc.NavItem(dbc.NavLink('data_import', href="/page-1", id="nav-link-1")),
            dbc.NavItem(dbc.NavLink('advanced_analysis', href="/page-2", id="nav-link-2")),
            dbc.NavItem(dbc.NavLink('user_guide', href="#", id="user-guide-link")),
        ]

        return nav_links, current_path

    # Callback client-side pour le thème sombre/clair
    clientside_callback(
        """
        function(checked) {
            return checked ? 'dark' : 'light';
        }
        """,
        Output('theme-switch', 'className'),
        Input('theme-switch-checkbox', 'checked')
    )

    @app.callback(
        Output('page-content', 'className'),
        [Input('theme-switch-checkbox', 'checked')]
    )
    def update_theme(checked):
        """Met à jour les classes CSS pour le thème"""
        return 'dark-mode' if checked else 'light-mode'

    @app.callback(
        [Output('nav-link-1', 'active',allow_duplicate=True),
         Output('nav-link-2', 'active',allow_duplicate=True)],
        [Input('url', 'pathname')],
        prevent_initial_call=True
    )
    def update_nav_links(pathname):
        """Met à jour l'état actif des liens de navigation"""
        if pathname == '/page-2':
            return False, True
        return True, False