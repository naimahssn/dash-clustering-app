import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, html
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
from typing import Optional, Dict, List, Tuple
import numpy as np
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import base64
from io import BytesIO
from PIL import Image
import logging

# Configuration des palettes de couleurs
CLUSTER_COLORS = px.colors.qualitative.Plotly
MAX_CLUSTERS = 20

logger = logging.getLogger(__name__)


def apply_standard_styling(fig: go.Figure) -> go.Figure:
    """Applique le style visuel standard à toutes les visualisations"""
    fig.update_layout(
        plot_bgcolor='rgba(245,245,245,0.95)',
        paper_bgcolor='white',
        font={'family': "Segoe UI, Arial, sans-serif", 'size': 12},
        margin={'t': 50, 'b': 30, 'l': 50, 'r': 30},
        hovermode='closest',
        legend=dict(
            title_text="Légende des Clusters",
            title_font_size=14,
            font_size=12,
            bgcolor='rgba(255,255,255,0.9)',
            itemsizing='constant'
        ),
        title_x=0.05,
        title_y=0.95,
        title_font_size=18
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(200,200,200,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(200,200,200,0.3)')
    return fig


def generate_color_mapping(clusters: pd.Series) -> Dict:
    """Génère un mapping de couleurs cohérent pour les clusters"""
    unique_clusters = sorted(clusters.unique())
    return {cluster: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cluster in enumerate(unique_clusters)}


def create_tsne_plot(projection, clusters, hover_data, dimensions=2):
    """
    Crée une visualisation t-SNE interactive 2D/3D avec Plotly Express
    """
    # Validation des dimensions
    if dimensions not in [2, 3]:
        raise ValueError("Les dimensions doivent être 2 ou 3")

    if len(projection.shape) != 2 or projection.shape[1] not in [2, 3]:
        raise ValueError("Format de projection invalide")

    if len(projection) != len(clusters) or len(projection) != len(hover_data):
        raise ValueError("Dimensions incohérentes entre les données")

    # Création du DataFrame
    tsne_data = pd.DataFrame({
        'Dimension 1': projection[:, 0],
        'Dimension 2': projection[:, 1],
        'Cluster': clusters.astype(str)
    })

    if dimensions == 3 and projection.shape[1] == 3:
        tsne_data['Dimension 3'] = projection[:, 2]

    tsne_data = pd.concat([tsne_data, hover_data.reset_index(drop=True)], axis=1)

    # Configuration des couleurs
    unique_clusters = pd.Series(tsne_data['Cluster']).astype(str).unique()
    color_mapping = {
        cluster: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, cluster in enumerate(sorted(unique_clusters, key=lambda x: (x.isdigit(), x)))
    }

    # Création de la figure
    if dimensions == 3 and 'Dimension 3' in tsne_data:
        fig = px.scatter_3d(
            data_frame=tsne_data,
            x='Dimension 1',
            y='Dimension 2',
            z='Dimension 3',
            color='Cluster',
            color_discrete_map=color_mapping,
            hover_data=hover_data.columns.tolist(),
            title="Visualisation 3D des clusters",
            labels={'Cluster': 'Cluster'}
        )
    else:
        fig = px.scatter(
            data_frame=tsne_data,
            x='Dimension 1',
            y='Dimension 2',
            color='Cluster',
            color_discrete_map=color_mapping,
            hover_data=hover_data.columns.tolist(),
            title="Projection t-SNE des clusters",
            labels={'Cluster': 'Cluster'}
        )

    # Configuration des marqueurs
    fig.update_traces(
        marker=dict(
            size=6 if dimensions == 3 else 8,
            opacity=0.8,
            line=dict(width=0.5, color='DarkSlateGrey'),
            symbol='diamond' if dimensions == 3 else 'circle-open'
        )
    )

    # Ajout des annotations (uniquement en 2D)
    if dimensions == 2:
        annotations = []
        for cluster in unique_clusters:
            cluster_points = tsne_data[tsne_data['Cluster'] == cluster]
            if len(cluster_points) > 0:
                annotations.append(dict(
                    x=cluster_points['Dimension 1'].median(),
                    y=cluster_points['Dimension 2'].median(),
                    text=f"Cluster {cluster}",
                    showarrow=False,
                    font=dict(size=12, color=color_mapping[cluster])
                ))
        fig.update_layout(annotations=annotations)

    # Configuration globale du layout
    fig.update_layout(
        clickmode='event+select',
        dragmode='pan' if dimensions == 3 else 'select',
        plot_bgcolor='rgba(240,240,240,0.9)',
        paper_bgcolor='white',
        font={'family': "Arial, sans-serif"},
        margin={'t': 40, 'b': 20, 'l': 20, 'r': 20},
        hoverlabel=dict(bgcolor="white", font_size=12),
        scene=dict(  # Spécifique 3D
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ) if dimensions == 3 else {}
    )

    return fig


def create_cluster_distribution(clusters: pd.Series) -> go.Figure:
    """Visualisation de la distribution des clusters"""
    try:
        counts = (
            clusters.astype(str)
            .value_counts(normalize=True)
            .reset_index()
            .rename(columns={'proportion': 'POURCENTAGE', 'index': 'CLUSTER'})
        )

        fig = px.bar(
            counts,
            x='CLUSTER',
            y='POURCENTAGE',
            color='CLUSTER',
            color_discrete_map=generate_color_mapping(counts['CLUSTER']),
        title = "Distribution des Clusters",
        labels = {'POURCENTAGE': 'Pourcentage'},
        text_auto = '.1%'
        )

        fig.update_layout(
            xaxis={'type': 'category', 'title': 'Cluster'},
            yaxis={'tickformat': ',.0%'},
            hoverlabel={'namelength': -1},
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )

        return apply_standard_styling(fig)

    except Exception as e:
        logger.error(f"Erreur distribution clusters: {str(e)}")
        return create_empty_figure("Erreur de visualisation")


def create_interactive_table(df: pd.DataFrame, cluster_col: str = 'CLUSTER') -> dash_table.DataTable:
    """Crée un tableau interactif avec fonctionnalités avancées"""
    color_mapping = generate_color_mapping(df[cluster_col])

    style_conditions = []
    for cluster, color in color_mapping.items():
        style_conditions.append({
            'if': {
                'filter_query': f'{{{cluster_col}}} = "{cluster}"',
                'column_id': cluster_col
            },
            'backgroundColor': color,
            'color': 'white' if px.colors.sequential.aggregate_luminance([color])[0] < 128 else 'black'
        })

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        filter_action="native",
        sort_action="native",
        page_size=15,
        style_table={'overflowX': 'auto', 'maxHeight': '70vh'},
        style_header={
            'backgroundColor': 'rgb(240,240,240)',
            'fontWeight': 'bold',
            'position': 'sticky',
            'top': 0
        },
        style_cell={
            'minWidth': '120px',
            'maxWidth': '250px',
            'whiteSpace': 'normal',
            'textAlign': 'left',
            'padding': '10px'
        },
        style_data_conditional=style_conditions + [
            {
                'if': {'state': 'selected'},
                'backgroundColor': 'rgba(200,200,200,0.3)',
                'border': '2px solid rgb(0,116,217)'
            }
        ],
        tooltip_data=[{
            col: {'value': str(value), 'type': 'markdown'}
            for col, value in row.items()
        } for row in df.to_dict('records')],
        tooltip_duration=None
    )


def create_word_cloud(keywords: Dict[int, List[str]]) -> html.Img:
    """Génère un nuage de mots à partir des mots-clés"""
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='tab20'
        ).generate_from_frequencies(
            {word: weight for cluster_words in keywords.values() for word, weight in cluster_words}
        )

        img = Image.fromarray(wordcloud.to_array())
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return html.Img(src=f"data:image/png;base64,{img_str}",
                        style={'width': '100%', 'padding': '20px'})

    except Exception as e:
        logger.error(f"Erreur génération wordcloud: {str(e)}")
        return html.Div("Erreur d'affichage")


def create_parallel_coordinates(df: pd.DataFrame, cluster_col: str = 'CLUSTER') -> go.Figure:
    """Crée une visualisation en coordonnées parallèles"""
    dimensions = [
        dict(label=col, values=df[col])
        for col in df.columns if col != cluster_col and df[col].nunique() > 1
    ]

    color_mapping = generate_color_mapping(df[cluster_col])

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=[color_mapping[str(c)] for c in df[cluster_col]],
            colorscale=CLUSTER_COLORS,
            showscale=True
        ),
        dimensions=dimensions
    ))

    fig.update_layout(
        title="Coordonnées Parallèles par Cluster",
        margin={'t': 50, 'b': 30}
    )

    return apply_standard_styling(fig)


def create_empty_figure(message="Aucune donnée à afficher"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20)
    )
    return fig


def create_cluster_summary(df: pd.DataFrame, cluster_col: str = 'CLUSTER') -> List[dbc.Card]:
    """Génère des cartes récapitulatives pour chaque cluster"""
    stats = df.groupby(cluster_col).agg({
        col: ['mean', 'count'] if np.issubdtype(df[col].dtype, np.number) else 'count'
        for col in df.columns if col != cluster_col
    }).reset_index()

    cards = []
    for cluster in stats[cluster_col].unique():
        cluster_data = stats[stats[cluster_col] == cluster].iloc[0]
        color = CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)]

        content = []
        for col in df.columns:
            if col != cluster_col:
                content.append(html.P(
                    f"{col}: {cluster_data[col]['mean'] if isinstance(cluster_data[col], dict) else cluster_data[col]}",
                    className="mb-1"
                ))

        cards.append(dbc.Card(
            [
                dbc.CardHeader(f"Cluster {cluster}", style={'backgroundColor': color}),
                dbc.CardBody(content)
            ],
            className="m-2 shadow-sm",
            style={'minWidth': '300px'}
        ))

    return cards