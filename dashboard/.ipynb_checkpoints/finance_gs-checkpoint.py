# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:12:00 2023

@author: Giorgio
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import datetime as datetime
#import time
#import os
#import seaborn as sns
#import plotly.express as px
#import plotly.graph_objects as go

def adjust(date, close, adj_close, in_col, rounding=4):
    '''
    If using forex or Crypto - Change the rounding accordingly!
    '''
    try:
        factor = adj_close / close
        return round(in_col * factor, rounding)
    except ZeroDivisionError:
        print('WARNING: DIRTY DATA >> {} Close: {} | Adj Close {} | in_col: {}'.format(date, close, adj_close, in_col))
        return 0

def getTickerData(ticker, start,end):
    '''
    funciton to fetch data of a stock (ticker) and compute the daily returns, log returns, cumulative returns usind split adjusted data
    
    Parameters:
    -----------  
     ticker: symbol of the stock to fetch data (US)
      start: start date in ISO format yyyy-mm-dd
        end: end date in ISO format yyyy-mm-dd
            
    Returns:
    --------
        df: with the following columns
            'adj open','adj close','adj high', 'adj low', 'daily_return', 'cumluative_return' , 'log_return'
    '''
    df = yf.download(ticker,start,end)
    df['Adj Open'] = np.vectorize(adjust)(df.index.date, df['Close'], df['Adj Close'], df['Open'])
    df['Adj High'] = np.vectorize(adjust)(df.index.date, df['Close'], df['Adj Close'], df['High'])
    df['Adj Low'] = np.vectorize(adjust)(df.index.date, df['Close'], df['Adj Close'], df['Low'])    
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Log Return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1)) #np.log(1+df['adj close'].pct_change())
    df['Cumulative Return'] = np.exp(np.log1p(df['Daily Return']).cumsum())
    ma_day = [10,20,50]
    for ma in ma_day:
        column_name = "MA %s Days" %(str(ma))
        df[column_name] = df['Adj Close'].rolling(window=ma,center=False).mean()
    
    
    return df


def getTickerDataNoDate(ticker):
    '''
    funciton to fetch data of a stock (ticker) and compute the daily returns, log returns, cumulative returns usind split adjusted data
    
    Parameters:
    -----------  
     ticker: symbol of the stock to fetch data (US)
      start: start date in ISO format yyyy-mm-dd
        end: end date in ISO format yyyy-mm-dd
            
    Returns:
    --------
        df: with the following columns
            'adj open','adj close','adj high', 'adj low', 'daily_return', 'cumluative_return' , 'log_return'
    '''
    df = yf.download(ticker)
    df['Adj Open'] = np.vectorize(adjust)(df.index.date, df['Close'], df['Adj Close'], df['Open'])
    df['Adj High'] = np.vectorize(adjust)(df.index.date, df['Close'], df['Adj Close'], df['High'])
    df['Adj Low'] = np.vectorize(adjust)(df.index.date, df['Close'], df['Adj Close'], df['Low'])    
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Log Return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1)) #np.log(1+df['adj close'].pct_change())
    df['Cumulative Return'] = np.exp(np.log1p(df['Daily Return']).cumsum())
    ma_day = [10,20,50]
    for ma in ma_day:
        column_name = "MA %s Days" %(str(ma))
        df[column_name] = df['Adj Close'].rolling(window=ma,center=False).mean()
    
    
    return df


def getPtf (tickers, start, end):
    '''
    funciton to fetch data of a series of tickers and build a portfolio
    
    Parameters:
    -----------  
     tickers: list of tickers for which to fetch data
            
    Returns:
    --------
     ptf: multi index data frame with stock data divided on the columns
       
    '''
    ptf = pd.DataFrame()
    df = pd.DataFrame()
    for stock in tickers:
        df = getTickerData(stock, start, end)
        df.sort_index(ascending=True, inplace = True)
        df.columns = pd.MultiIndex.from_product([[stock], df.columns])
        ptf =  pd.concat([df, ptf], axis=1)
    return ptf

def randomWeights(n):
    '''
    funciton to retrun a np array of n random numbers, wich all add up to 1
    
    Parameters:
    -----------  
     n: number of assets
            
    Returns:
    --------
     list: np array 
       
    '''
    k = np.random.rand(n)
    return k/sum(k)


def getSpecificColumns(ptf, column, stocks):
    '''
    funciton to get the specific columns for each stock contained in the ptf
    
    Parameters:
    -----------  
     ptf: ptf obtained with getPtfColumns 
     column: name of the column
     stocks: list of the tickers    
            
    Returns:
    --------
     new_ptf: dataframe of the specific columns for each stock
       
    '''
    df = pd.DataFrame()
    new_ptf = pd.DataFrame()
    for each in stocks:
        df = ptf.loc[:, each][[column]]
        df.columns = [each]
        new_ptf = pd.concat([df, new_ptf], axis=1)
    return new_ptf

def ptfPerformance(weights, ptf):
    '''
    funciton to return a tuple of ptf return and std dev annualised, function accepts df from getPtfColumns, and isolates the 
    returns and computes std dev
    
    Parameters:
    -----------  
     weight: weight of the assets
     ptf: dataframe of stock ptf  
            
    Returns:
    --------
     ptf_return: return of the portfolio
     ptf_std: std dev of the portfolio
       
    '''    
    returns = getSpecificColumns(ptf, 'Daily Return', ptf.columns.levels[0])
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    weights_a = np.array(weights)
    ptf_return = np.sum(mean_returns*weights_a)*252
    ptf_std = np.sqrt(np.dot(weights_a.T, np.dot(cov_matrix, weights_a ))) * np.sqrt(252)
    #be careful the order is ptf.columns.levels[0] reversed
    return ptf_return, ptf_std

def retrurnsAndStd(ptf, stocks, iterations):
    '''
           
    '''    
    df = pd.DataFrame(columns = ['Return', 'Std', 'Weights'])
    for i in range(iterations):
        weights = randomWeights(len(stocks))
        returns, std = ptfPerformance(weights, ptf)
        df.loc[i] = returns, std, weights
    return df

def stockPerformanceAnnualised(stock_daily_return):
    '''
    function to annualise the returns of a stock
    
    Parameter:
    -----------
        stock_daily_return:
    
    Returns:
    -----------
        
    ''' 
    return np.mean(stock_daily_return)*252, np.std(stock_daily_return)*np.sqrt(252)


def portfolio_returns_simulation(stocks,ptf,iterations):
    returns = pd.DataFrame()
    for each in stocks:
        column = each+' Daily Log Return'
        returns[column] = ptf.loc[:, each]['Log Return']
    
    port_returns = []
    port_vols = []
    
    for i in range(iterations):
        weights = np.random.dirichlet(np.ones(len(stocks)), size=1)
        weights = weights[0]
        port_returns.append(np.sum(returns.mean() * weights) * 252)
        port_vols.append(np.sqrt(
                            np.dot(
                                weights.T, np.dot(
                                                returns.cov() * 252, weights)
                                  )
                            )
        )
    
    # Convert lists to arrays
    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)
    
    
    plt.figure(figsize = (18,10))
    plt.scatter(port_vols,port_returns,c = (port_returns / port_vols), marker='o')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label = 'Sharpe ratio (not adjusted for short rate)')
    
    
    return port_returns, port_vols

def portfolio_return(stocks,ptf):
    returns = pd.DataFrame()
    for each in stocks:
        column = each+' Daily Log Return'
        returns[column] = ptf.loc[:, each]['Log Return']
    return returns



def portfolio_stats(weights, returns):
    
    '''
    We can gather the portfolio performance metrics for a specific set of weights.
    This function will be important because we'll want to pass it to an optmization
    function to get the portfolio with the best desired characteristics.
    
    Note: Sharpe ratio here uses a risk-free short rate of 0.
    
    Paramaters: 
    -----------
        weights: array, 
            asset weights in the portfolio.
        returns: dataframe
            a dataframe of returns for each asset in the trial portfolio    
    
    Returns: 
    --------
        dict of portfolio statistics - mean return, volatility, sharp ratio.
    '''

    # Convert to array in case list was passed instead.
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = port_return/port_vol

    return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}

def minimize_sharpe(weights, returns):  
    return -portfolio_stats(weights,returns)['sharpe'] 

def minimize_volatility(weights, returns):  
    # Note that we don't return the negative of volatility here because we 
    # want the absolute value of volatility to shrink, unlike sharpe.
    return portfolio_stats(weights,returns)['volatility'] 

def minimize_return(weights, returns): 
    return -portfolio_stats(weights,returns)['return']


#Function takes in stock price, number of days to run, mean and standard deviation values
def stock_monte_carlo(start_price,days,mu,sigma):
    '''
    function to simulate, using monte carlo, the price of an asset for the next X number of days
    
    Paramaters: 
    -----------
        start_price: starting price of the asset
        days: days to simulate
        mu: median return of an asset
        sigma: std dev of the returns of an asset
    
    Returns: 
    --------
        dict of portfolio statistics - mean return, volatility, sharp ratio.
    '''
    #delta t
    dt = 1/days
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        
        #Shock and drift formulas taken from the Monte Carlo formula
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        
        drift[x] = mu * dt
        
        #New price = Old price + Old price*(shock+drift)
        price[x] = price[x-1] + (price[x-1] * (drift[x]+shock[x]))
    return price