import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finance_gs import *

from dash import Dash, Input, Output, callback, dash_table, html, dcc, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

df = pd.read_csv('https://git.io/Juf1t')

app = Dash( )

app.layout = dbc.Container([
    dbc.Label('Insert a Stock Ticker:'),
    html.Br(),
    
    dbc.Input(id='ticker', debounce = True, value = 'AAPL'),
    html.Br(),
    
    dcc.Dropdown( id = 'stock-dropdown', 
                 options = [ {'label': 'GS', 'value': 'GS'},],
                 multi=True
    ),
    
    
    html.H2(  id='tbl'),
    
    html.Button('Submit', id='submit-val', n_clicks=0),
    
    html.H2(  id='ptf'),
    
    
    dcc.Graph(id='graph-content')
])


@app.callback(
    Output("tbl", "children"),
    Input("ticker", "value"),
)
def update_output(ticker):
    return getTickerDataNoDate(ticker).Open[-1]
 

@app.callback(
    Output("stock-dropdown", "options"),
    Input("ticker", "value"),
    State('stock-dropdown', 'options')
)
def add_stock_to_list(ticker, pre_list):
    if not ticker:
        return pre_list
    pre_list.append({'label': ticker, 'value': ticker})
    return pre_list
    

@app.callback(
    Output("ptf", "children"),
    Output('graph-content', 'figure'),
    Input("submit-val", "n_clicks"),
    State('stock-dropdown', 'value')
)
def update_ptf(n_clicks, ptf):
    
    
    return ptf, px.line(getTickerDataNoDate(ptf), x = 'Date' , y = 'Open')
 
    
 
    
 
    

if __name__ == "__main__":
    app.run_server(debug=True)