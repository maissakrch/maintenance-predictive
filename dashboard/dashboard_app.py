import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
import dash_auth


# Charger les prédictions
df = pd.read_csv("data/cleaned/train_FD001_cleaned.csv")

# Initialiser l'app Dash
app = dash.Dash(__name__)

VALID_USERNAME_PASSWORD_PAIRS = {
    'admin': 'password123'
}

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


# Exemple de figure : distribution du cycle de vie
fig = px.histogram(df, x='RUL', nbins=50, title='Distribution de la Remaining Useful Life (RUL)')

# Layout de l'application
app.layout = html.Div(children=[
    html.H1(children='Dashboard Maintenance Prédictive', style={'textAlign': 'center'}),
    html.Div(children='Analyse de la vie restante des moteurs à partir du dataset FD001.'),
    dcc.Graph(
        id='rul-distribution',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run(debug=True)
