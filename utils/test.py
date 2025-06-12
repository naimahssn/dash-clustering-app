import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

# Exemple de données
data = [
    {"id": 1, "texte": "Analyse du budget 2024", "cluster_auto": "Finance"},
    {"id": 2, "texte": "Gestion des ressources humaines", "cluster_auto": "RH"}
]

clusters_detectes = ["Finance", "RH", "IT", "Autres"]

app.layout = html.Div([
    dash_table.DataTable(
        id='table-classification',
        columns=[
            {"name": "ID", "id": "id"},
            {"name": "Texte", "id": "texte"},
            {"name": "Cluster Auto", "id": "cluster_auto"},
            {"name": "Cluster Manuel", "id": "cluster_manuel", "presentation": "dropdown"}
        ],
        data=[{**row, "cluster_manuel": row["cluster_auto"]} for row in data],
        editable=True,
        dropdown={
            "cluster_manuel": {
                "options": [{"label": c, "value": c} for c in clusters_detectes]
            }
        }
    ),
    html.Button("Valider les changements", id="valider-btn", n_clicks=0),
    html.Div(id="output-valide")
])

@app.callback(
    Output("output-valide", "children"),
    Input("valider-btn", "n_clicks"),
    State("table-classification", "data")
)
def valider_clusters(n, rows):
    if n > 0:
        resultats = [f"Texte: {r['texte']} → Nouveau cluster: {r['cluster_manuel']}" for r in rows]
        return html.Ul([html.Li(res) for res in resultats])

if __name__ == '__main__':
    app.run(debug=True)
