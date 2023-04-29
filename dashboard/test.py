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
    'background': '#ffffff',
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
        
        html.H1('Stock Dashboard',
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
        
        
        stock_data = yf.download(value, start=start, end=end, group_by='ticker')

        # Calculate daily returns
        daily_returns = stock_data.xs('Adj Close', level = 1, axis = 1).pct_change()
        
        # Calculate annual returns and covariances
        annual_returns = daily_returns.mean() * 252
        cov_matrix = daily_returns.cov() * 252
        
        # Define objective function
        def objective(weights, returns, cov_matrix):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std_dev = np.sqrt(portfolio_variance)
            return portfolio_return, portfolio_std_dev
        
        # Define function to simulate random portfolios
        def simulate_random_portfolios(returns, cov_matrix, num_portfolios):
            num_assets = len(returns)
            all_weights = np.zeros((num_portfolios, num_assets))
            rets = np.zeros(num_portfolios)
            stds = np.zeros(num_portfolios)
            for i in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                all_weights[i,:] = weights
                rets[i], stds[i] = objective(weights, returns, cov_matrix)
            return all_weights, rets, stds
        
        # Perform Monte Carlo simulation
        num_portfolios = 10000
        all_weights, rets, stds = simulate_random_portfolios(annual_returns, cov_matrix, num_portfolios)
        
        # Find portfolio with highest Sharpe ratio
        sharpe_ratios = rets / stds
        max_sharpe_ratio_index = np.argmax(sharpe_ratios)
        optimal_weights = all_weights[max_sharpe_ratio_index,:]

        
        
        for each in value:
            df = getTickerData(each, start, end)
            fig = px.histogram(df, x = df['Daily Return'])
            charts.append(dcc.Graph(figure=fig))
            
        return html.Div([
            html.H3('Selected values:'),
            html.Ul([html.Li(val) for val in optimal_weights])
        ]), charts 
    
    else:
        return html.Div(), charts

if __name__ == '__main__':
    app.run_server(debug=True)
