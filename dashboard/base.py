import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as plt
from finance_gs import *
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go


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
    
    dcc.Graph(id='scatter-plot'),
    
    html.Div(id='charts', style={'padding': '20px'}),

    html.Div(id='corr-chart', style={'padding': '20px'})
    
], style={'background': colors['background'], 'fontFamily': 'Lato'})



@app.callback(
    dash.dependencies.Output('dropdown', 'options'),
    dash.dependencies.Output('dropdown', 'value'),
     dash.dependencies.Output('input', 'value'),
    dash.dependencies.Input('add-button', 'n_clicks'),
    dash.dependencies.State('input', 'value'),
    dash.dependencies.State('dropdown', 'options'),
    dash.dependencies.State('dropdown', 'value')
)
def update_dropdown_options(n_clicks, value, options, selected):
    if n_clicks > 0 and value:
        options.append({'label': value, 'value': value})
        selected.append(value)
    return options, selected, ''

@app.callback(
    dash.dependencies.Output('output', 'children'),
    dash.dependencies.Output('charts', 'children'),
    dash.dependencies.Output('scatter-plot', 'figure'),
    dash.dependencies.Output('corr-chart', 'children'),
    dash.dependencies.Input('print-button', 'n_clicks'),
    dash.dependencies.State('dropdown', 'value'),
    dash.dependencies.Input('date-range', 'start_date'),
    dash.dependencies.Input('date-range', 'end_date'),
)
def print_selected_values(n_clicks, value, start, end):
    charts = []
    chart_corr = []    

    tickers = value

    if n_clicks > 0 and len(value) == 0:
        
        return html.Div(), charts, go.Figure(), html.Div()
    
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
        
        fig_ptf = go.Figure(
                        go.Scatter(
                            x=stds,
                            y=rets,
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=rets/stds,
                                
                                symbol='circle'
                                        ),
                            hovertemplate='Standard Deviation: %{x}<br>Return: %{y}<br>Sharpe Ratio: %{marker.color:.2f}<extra></extra>'
                                  )
                            )
        fig_ptf.update_layout(
                        xaxis_title='Standard Deviation',
                        yaxis_title='Return',
                        coloraxis=dict(
                            colorbar=dict(
                                title='Sharpe Ratio'
                            )
                        ),
                        height=500,
                        margin=dict(l=50, r=50, b=50, t=50),
                    )
        
        #include the correlation matrix of the PTF
        #histogram of the returns for each pair

        corr_matrix = daily_returns.corr()

        fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values,x=corr_matrix.columns,y=corr_matrix.columns,colorscale='RdBu',zmin=-1,zmax=1,hoverongaps=False))
        fig_corr.update_layout(title='Correlation Matrix of Stock Returns')
        
        chart_corr.append(dcc.Graph(figure=fig_corr))

        for each in value:
            df = getTickerData(each, start, end)
            fig = px.histogram(df, x = df['Daily Return'],  opacity=0.8, color_discrete_sequence=['#636EFA'])
            fig.update_layout(title='Distribution of '+ each +' Stock Returns', xaxis_title='Return', yaxis_title='Count')
            charts.append(dcc.Graph(figure=fig))
            
        return html.Div([
            html.H3('Optimal Portfolio Weights:'),
            html.Ul(
                    [html.Li(val_e + ': ' + str(val)) for val,val_e in zip(list(optimal_weights), list(tickers))]
                    )
        ]), charts , fig_ptf, chart_corr
    
    else:
        return html.Div(), charts, go.Figure(), html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)
