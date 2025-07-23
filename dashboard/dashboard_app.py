import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Chargement des données
df = pd.read_csv('data/cleaned/train_FD001_cleaned.csv')

# Création du dashboard
app = dash.Dash(__name__)

# Graphique simple : Histogramme du nombre de cycles par moteur
fig = px.histogram(df, x='cycle', nbins=50, title='Distribution des cycles')

app.layout = html.Div([
    html.H1("Dashboard Maintenance Prédictive"),

    dcc.Graph(figure=fig),

    html.Br(),

    html.Button("Télécharger CSV", id="download-btn"),
    dcc.Download(id="download-dataframe-csv")
])

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n_clicks):
    return dcc.send_data_frame(df.to_csv, "donnees_maintenance.csv")

if __name__ == '__main__':
    app.run(debug=True)
