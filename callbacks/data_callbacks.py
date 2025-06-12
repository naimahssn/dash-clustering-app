from dash import Input, Output, State, callback, no_update, dash_table
import dash_bootstrap_components as dbc
import dash.dcc as dcc
import pandas as pd
import numpy as np
from utils.data_processing import handle_file_upload, process_financial_data, calculer_kpis
from dash.exceptions import PreventUpdate
import logging
from typing import Tuple, Dict, Optional
logger = logging.getLogger(__name__)
from utils.data_processing  import preparer_export

def register_data_callbacks(app):
    """Enregistre tous les callbacks liés aux données"""

    @app.callback(
        [Output('stored-sheets', 'data'),
         Output('sheet-selection-dropdown', 'options'),
         Output('upload-status-message', 'children')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename'),
         State('upload-data', 'last_modified')]
    )
    def upload_data_callback(contents: str, filename: str, last_modified: str):
        if not contents:
            raise PreventUpdate

        try:
            sheets_dict, error = handle_file_upload(contents, filename)
            if error:
                return no_update, [], dbc.Alert(error, color="danger")

            options = [{'label': name, 'value': name} for name in sheets_dict.keys()]
            return sheets_dict, options, dbc.Alert(f'Fichier {filename} chargé avec succès', color="success")

        except Exception as e:
            logger.error(f"Erreur d'upload: {str(e)}")
            return no_update, [], dbc.Alert(f'Erreur avec le fichier : {str(e)}', color="danger")

    @app.callback(
        [Output('stored-data', 'data'),
         Output('montant-initial-col', 'options'),
         Output('montant-engage-col', 'options'),
         Output('data-preview-container', 'children')],
        [Input('sheet-selection-dropdown', 'value')],
        [State('stored-sheets', 'data')]
    )
    def select_sheet_callback(sheet_name: str, sheets_data: Dict) -> Tuple:
        if not sheet_name or not sheets_data:
            raise PreventUpdate

        try:
            df = pd.DataFrame(sheets_data[sheet_name])

            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            options = [{'label': col, 'value': col} for col in numeric_cols]

            preview_table = dash_table.DataTable(
                data=df.head(5).to_dict('records'),
                columns=[{'name': col, 'id': col} for col in df.columns],
                style_table={
                    'height': '300px',
                    'overflowY': 'scroll',
                    'overflowX': 'auto',
                    'maxWidth': '95%',
                    'margin': 'auto',
                    'border': '1px solid #ccc',
                    'borderRadius': '5px'
                },
                style_cell={
                    'textAlign': 'center',
                    'minWidth': '120px',
                    'maxWidth': '200px',
                    'whiteSpace': 'normal',
                    'fontSize': '12px'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold'
                }
            )

            return df.to_dict('records'), options, options, preview_table

        except Exception as e:
            logger.error(f"Erreur de sélection de feuille : {str(e)}")
            return no_update, [], [], no_update

    @app.callback(
        [Output('processed-data', 'data'),
         Output('results-table-container', 'children'),
         Output('calculation-spinner', 'children'),
         Output('status-alerts-container', 'children'),
         Output('export-results-btn', 'disabled'),
         Output('sheet-selector', 'options'),
         Output('sheet-selector', 'value')],
        [Input('calculate-btn', 'n_clicks')],
        [State('stored-data', 'data'),
         State('montant-initial-col', 'value'),
         State('montant-engage-col', 'value')],
        prevent_initial_call=True
    )
    def calculate_indicators_callback(n_clicks: int, data: Dict, initial_col: str, engage_col: str) -> Tuple:
        """Callback pour calculer les indicateurs financiers"""
        if n_clicks is None or not initial_col or not engage_col:
            raise PreventUpdate

        try:
            df = pd.DataFrame.from_dict(data)
            processed_df = process_financial_data(df, initial_col, engage_col)

            table = create_interactive_table(processed_df)

            return (
                processed_df.to_dict('records'),
                table,
                None,
                dbc.Alert('Calcul des indicateurs réussi !', color="success", duration=4000),
                False,
                no_update,
                no_update
            )
        except Exception as e:
            logger.error(f"Erreur de calcul: {str(e)}")
            return (
                no_update,
                no_update,
                None,
                dbc.Alert(f'Erreur de calcul : {str(e)}', color="danger"),
                True,
                no_update,
                no_update
            )

    @app.callback(
        [Output('total-projects', 'children'),
         Output('total-initial', 'children'),
         Output('total-committed', 'children'),
         Output('avg-engagement', 'children')],
        [Input('processed-data', 'modified_timestamp')],
        [State('processed-data', 'data'),
         State('montant-initial-col', 'value'),
         State('montant-engage-col', 'value')
         ]
    )
    def update_kpis_callback(ts, data, initial_col, engage_col):
        """Callback pour mettre à jour les KPI"""
        if not data or not initial_col or not engage_col:
            raise PreventUpdate

        try:
            df = pd.DataFrame.from_dict(data)
            kpis = calculer_kpis(df, initial_col, engage_col)

            return (
                f" {kpis['total_projets']:,}",
                f" {kpis['total_initial']:,.2f} ",
                f" {kpis['total_engage']:,.2f} ",
                f"  {kpis['engagement_moyen']:.1f}%"
            )
        except Exception as e:
            logger.error(f"Erreur KPI: {str(e)}")
            return no_update, no_update, no_update, no_update

    @callback(
        Output("download-dataframe-xlsx", "data"),
        Input("export-results-btn", "n_clicks"),
        State("processed-data", "data"),
        prevent_initial_call=True
    )
    def export_to_excel(n_clicks, processed_data):
        if processed_data is None:
            return None

        df = pd.DataFrame(processed_data)
        output = preparer_export(df, format_type="excel")

        return dcc.send_bytes(output.read(), "resultats_analyse.xlsx")

def create_interactive_table(df: pd.DataFrame) -> dbc.Table:
    """Crée une table interactive avec styles conditionnels"""
    styled_df = df.copy()
    for col in ['ECART D\'ENGAGEMENT']:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

    if 'TAUX_ENGAGEMENT' in styled_df.columns:
        styled_df['TAUX_ENGAGEMENT'] = styled_df['TAUX_ENGAGEMENT'].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) else ""
        )

    table = dash_table.DataTable(
        data=styled_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in styled_df.columns],
        filter_action="native",
        sort_action="native",
        page_size=70,
        fixed_rows={'headers': True},
        style_table={
            'maxWidth': '95%',
            'maxHeight': '1000px',
            'overflowY': 'scroll',
            'overflowX': 'auto',
            'margin': 'auto',
            'border': '1px solid #ccc',
            'borderRadius': '5px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'minWidth': '100px',
            'maxWidth': '200px',
            'whiteSpace': 'normal',
            'fontSize': '12px'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{ECART D\'ENGAGEMENT} < 0', 'column_id': 'ECART D\'ENGAGEMENT'},
                'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                'color': 'darkred',
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': '{TAUX_ENGAGEMENT} > 100', 'column_id': 'TAUX_ENGAGEMENT'},
                'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                'color': 'darkred',
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': '{ENGAGEMENT_STATUS} = "Sous-engagé"', 'column_id': 'ENGAGEMENT_STATUS'},
                'backgroundColor': 'rgba(0, 0, 255, 0.1)',
                'color': 'blue',
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': '{ENGAGEMENT_STATUS} = "Normal"', 'column_id': 'ENGAGEMENT_STATUS'},
                'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                'color': 'green',
                'fontWeight': 'bold'
            },
            {
                'if': {'filter_query': '{ENGAGEMENT_STATUS} = "Sur-engagé"', 'column_id': 'ENGAGEMENT_STATUS'},
                'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                'color': 'darkred',
                'fontWeight': 'bold'
            }
        ]
    )

    return table