import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finance_gs import *
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

# Set the colors for the app
colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'accent': '#0074D9'
}

# Create the app layout with a dark theme
app = dash.Dash(__name__, 
                external_stylesheets=[
                    {'href': 'https://fonts.googleapis.com/css?family=Lato', 'rel': 'stylesheet'},
                    {'href': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/7ecfa3c1/dash-analytics-report.css', 'rel': 'stylesheet'}
                ],
                suppress_callback_exceptions=True,
               )

app.layout = html.Div([
    
    html.Div([
        
        html.H1('Stock Ticker Dashboard',
                style={
                    'textAlign': 'center',
                    'color': colors['accent'],
                    'padding': '20px',
                    'fontFamily': 'Lato'
                }
               ),
        
        html.Div([
            dcc.Input(id='input', type='text', placeholder='Enter a value', style={'width': '80%', 'marginRight': '10px'}),
            html.Button('Add Stock Ticker', id='add-button', n_clicks=0, style={'width': '20%'}),
        ], style={'display': 'flex'}),
        
        dcc.Dropdown(
            id='dropdown',
            options=[],
            value=[],
            multi=True,
            style={
                'width': '60%',
                'marginTop': '20px',
                'marginBottom': '20px',
                'marginRight': '10px'
            }
        ),
        
        dcc.DatePickerRange(
            id='date-range',
            max_date_allowed=datetime.now().date(),
            start_date=None,
            end_date=None,
            style={
                'width': '40%',
                'marginTop': '20px',
                'marginBottom': '20px',
            }
        ),
        
        html.Button('Print Selected', id='print-button', n_clicks=0, style={'width': '20%'}),
        
    ], style={'background': colors['background'], 'padding': '30px'}),
    
    html.Div(id='output', style={'padding': '20px'}),
    
    html.Div(id='charts', style={'padding': '20px'}),
    
], style={'background': colors['background'], 'fontFamily': 'Lato'})



@app.callback(
    dash.dependencies.Output('dropdown', 'options'),
    dash.dependencies.Output('dropdown', 'value'),
    dash.dependencies.Input('add-button', 'n_clicks'),
    dash.dependencies.State('input', 'value'),
    dash.dependencies.State('dropdown', 'options'),
    dash.dependencies.State('dropdown', 'value')
)
def update_dropdown_options(n_clicks, value, options, selected):
    if n_clicks > 0 and value:
        options.append({'label': value, 'value': value})
        selected.append(value)
    return options, selected

@app.callback(
    dash.dependencies.Output('output', 'children'),
    dash.dependencies.Output('charts', 'children'),
    dash.dependencies.Input('print-button', 'n_clicks'),
    dash.dependencies.State('dropdown', 'value'),
    dash.dependencies.Input('date-range', 'start_date'),
    dash.dependencies.Input('date-range', 'end_date'),
)
def print_selected_values(n_clicks, value, start, end):
    charts = []
    
    if n_clicks > 0 and len(value) == 0:
        
        return html.Div(), charts
    
    elif len(value) > 0 and start is not None and end is not None and n_clicks > 0:
        
        for each in value:
            df = getTickerData(each, start, end)
            fig = px.histogram(df, x = df['Daily Return'])
            charts.append(dcc.Graph(figure=fig))
            
        return html.Div([
            html.H3('Selected values:'),
            html.Ul([html.Li(val) for val in value])
        ]), charts 
    
    else:
        return html.Div(), charts

if __name__ == '__main__':
    app.run_server(debug=True)
