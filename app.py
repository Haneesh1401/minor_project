# =========================
# IMPORTS
# =========================
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import xarray as xr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# LOAD DATA (CRITICAL)
# =========================
# This replaces notebook memory
dust_ds = xr.load_dataset("outputs/dust_visualization_dataset.nc")

# Simulated prediction layer (DL-ready placeholder)
predicted_ds = dust_ds.shift(time=-1)

# =========================
# ANALYTICAL FUNCTIONS
# =========================
def classify_risk(data):
    mean_val = np.nanmean(data)
    if mean_val < 0.3:
        return "LOW", "#00FFC8", "🟢"
    elif mean_val < 0.6:
        return "MODERATE", "#FFB300", "🟡"
    else:
        return "HIGH", "#FF4B4B", "🔴"

def compute_uncertainty(series, window=5):
    return np.array([
        np.std(series[max(0, i-window):i+1])
        for i in range(len(series))
    ])

def detect_hotspots(data, threshold=0.7):
    return (data > threshold).sum() / data.size * 100

def health_impact(risk):
    if risk == "LOW":
        return "Minimal respiratory impact expected."
    elif risk == "MODERATE":
        return "Sensitive groups may experience discomfort."
    else:
        return "High risk for respiratory issues. Precaution advised."

def compute_metrics(obs, pred):
    mae = mean_absolute_error(obs.flatten(), pred.flatten())
    rmse = np.sqrt(mean_squared_error(obs.flatten(), pred.flatten()))
    return mae, rmse

# =========================
# CREATE DASH APP
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # REQUIRED FOR DEPLOYMENT

# =========================
# DASHBOARD LAYOUT
# =========================
app.layout = dbc.Container(fluid=True, style={"padding": "30px"}, children=[

    dbc.Row(dbc.Col(
        html.Div([
            html.H1(
                "GEO-DUST ANALYTICS",
                style={
                    "color": "white",
                    "fontWeight": "900",
                    "letterSpacing": "4px",
                    "textAlign": "center"
                }
            ),
            html.P(
                "AI-powered spatio-temporal dust forecasting & risk intelligence",
                style={
                    "color": "#58a6ff",
                    "textAlign": "center",
                    "fontSize": "14px",
                    "letterSpacing": "1px"
                }
            )
        ])
    )),

    dbc.Row([
        # LEFT PANEL
        dbc.Col(width=3, children=[
            html.Div(style={
                "background": "#11141d",
                "padding": "20px",
                "borderRadius": "12px",
                "border": "1px solid #2d3039"
            }, children=[

                html.Label("Variable"),
                dcc.Dropdown(
                    id="variable-dropdown",
                    options=[{"label": v, "value": v} for v in dust_ds.data_vars],
                    value=list(dust_ds.data_vars)[0],
                    clearable=False
                ),

                html.Br(),
                html.Label("Mode"),
                dbc.RadioItems(
                    id="data-mode",
                    options=[
                        {"label": "Observed", "value": "obs"},
                        {"label": "Predicted", "value": "pred"},
                        {"label": "Difference", "value": "diff"},
                    ],
                    value="obs"
                ),

                html.Br(),
                html.Label("Forecast Horizon"),
                dcc.Dropdown(
                    id="forecast-horizon",
                    options=[{"label": f"+{h}", "value": h} for h in [1, 3, 6]],
                    value=1,
                    clearable=False
                ),

                html.Br(),
                html.Label("Time Index"),
                dcc.Slider(
                    id="time-slider",
                    min=0,
                    max=len(dust_ds.time) - 1,
                    step=1,
                    value=0,
                    tooltip={"always_visible": True}
                ),

                html.Hr(),
                html.Div(id="risk-indicator"),
                html.Div(id="health-indicator", style={"fontSize": "13px", "color": "#CCCCCC"})
            ])
        ]),

        # RIGHT PANEL
        dbc.Col(width=9, children=[
            dcc.Graph(id="spatial-map", style={"height": "55vh"}),
            dcc.Graph(id="time-series", style={"height": "25vh"})
        ])
    ])
])

# =========================
# CALLBACKS
# =========================
@app.callback(
    Output("spatial-map", "figure"),
    Output("risk-indicator", "children"),
    Output("health-indicator", "children"),
    Input("variable-dropdown", "value"),
    Input("time-slider", "value"),
    Input("data-mode", "value"),
    Input("forecast-horizon", "value"),
)
def update_spatial(variable, t, mode, horizon):

    future_t = min(t + horizon, len(dust_ds.time) - 1)
    obs = dust_ds[variable].isel(time=t).values
    pred = predicted_ds[variable].isel(time=future_t).values

    if mode == "obs":
        data = obs
    elif mode == "pred":
        data = pred
    else:
        data = pred - obs

    risk, color, icon = classify_risk(data)

    fig = px.imshow(
        data,
        origin="upper",
        color_continuous_scale="Inferno"
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig, f"{icon} Risk Level: {risk}", health_impact(risk)

@app.callback(
    Output("time-series", "figure"),
    Input("spatial-map", "clickData"),
    Input("variable-dropdown", "value")
)
def update_timeseries(clickData, variable):

    if clickData:
        x = int(clickData["points"][0]["x"])
        y = int(clickData["points"][0]["y"])
    else:
        y = dust_ds.dims["y"] // 2
        x = dust_ds.dims["x"] // 2

    hist = dust_ds[variable][:, y, x].values
    pred = predicted_ds[variable][:, y, x].values
    unc = compute_uncertainty(pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist, name="Observed"))
    fig.add_trace(go.Scatter(y=pred, name="Forecast", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(
        y=pred + unc,
        showlegend=False,
        line=dict(width=0)
    ))
    fig.add_trace(go.Scatter(
        y=pred - unc,
        fill="tonexty",
        fillcolor="rgba(255,75,75,0.25)",
        line=dict(width=0),
        name="Uncertainty"
    ))

    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=20, t=30, b=30))
    return fig

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)
