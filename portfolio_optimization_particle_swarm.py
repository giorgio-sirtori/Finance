import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyswarms as ps

# Define your stock tickers and the date range for historical data
tickers = ['KO', 'CAT', 'MSFT', 'GS', 'COST']
start_date = '2015-01-01'
end_date = '2023-04-28'

# Download historical data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change()

# Define the objective function (maximize Sharpe Ratio)
def objective_function(weights):
    weights = np.array(weights).reshape(-1, len(tickers))
    portfolio_return = np.sum(returns.mean().values * weights, axis=1) * 252
    portfolio_volatility = np.sqrt(np.diag(np.dot(weights, np.dot(returns.cov() * 252, weights.T))))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return -sharpe_ratio


# Constraints for PSO
constraints = (np.zeros(len(tickers)), np.ones(len(tickers)))

# Set up the PSO
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=len(tickers), options=options, bounds=constraints)
cost, optimal_weights = optimizer.optimize(objective_function, iters=100)

print("Optimal weights:", optimal_weights)

# Normalize the weights so they sum to 1
optimal_weights = optimal_weights / np.sum(optimal_weights)

# Calculate the portfolio performance
portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights)))
sharpe_ratio = portfolio_return / portfolio_volatility

print("Portfolio Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
print("Sharpe Ratio:", sharpe_ratio)

