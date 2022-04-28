from django.shortcuts import render
import plotly
import plotly.express as px
from rest_framework.exceptions import *
from rest_framework.views import APIView
from rest_framework.response import Response

import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
from past.builtins import xrange

from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf
import datetime as dt
#import numpy_financial as npf
from scipy.stats import norm
import seaborn as sns
import plotly.graph_objs as go

import mpld3

# Create your views here.
def home(request):
    return render(request, "myapp/portfolio_c.html")

def equity(request):
    return render(request, "myapp/eq.html")

def bond(request):
    return render(request, "myapp/checkbox.html")

def risk(request):
    return render(request, "myapp/risk.html")

# def ticker_data(tickers, start, end):
#     df = yf.download(tickers=tickers, start=start, end=end)
#     for i in tickers:
#         print('df')
#     fig = px.line(df['Adj Close'])
#
#     fig.update_layout(
#         margin=dict(l=0, r=0, b=50, t=10),
#         height=260, width=400,
#         paper_bgcolor='rgba(0, 0, 0, 0)',
#         plot_bgcolor='rgba(0, 0, 0, 0)',
#         yaxis=dict(color="#ffffff"),
#         xaxis=dict(color="#ffffff"),
#         font_color="white",
#     )
#
#
#     plotly.offline.plot(fig, filename="myapp/static/files/mom.html", config={'displayModeBar': False}, auto_open=False)
#
#     fig.update_layout(
#         margin=dict(l=0, r=0, b=50, t=10),
#         height=350, width=600,
#         paper_bgcolor='rgba(0, 0, 0, 0)',
#         plot_bgcolor='rgba(0, 0, 0, 0)',
#         yaxis=dict(color="#ffffff"),
#         xaxis=dict(color="#ffffff"),
#         font_color="white",
#     )
#
#     plotly.offline.plot(fig, filename="myapp/static/files/portfolio.html", config={'displayModeBar': False}, auto_open=False)
#
#
# ticker_data(['AAPL', 'CVM', 'TSLA', 'F'], dt.datetime(2010, 1, 1), dt.datetime(2017, 1, 1))

# VAR
def get_var(data, weights, confidence_level):
    returns = data.pct_change()
    returns.dropna(axis=0, inplace=True)
    cov_matrix = returns.cov()
    avg_returns = returns.mean()
    count = returns.count()[0]
    port_mean = avg_returns @ weights
    port_variance = weights.T @ cov_matrix @ weights
    port_stdev = np.sqrt(port_variance)
    x = np.arange(-0.05, 0.05, 0.001)
    norm_dist = norm.pdf(x, port_mean, port_stdev)
    var = norm.ppf(confidence_level, port_mean, port_stdev)
    # change background style from these two lines
    df = pd.DataFrame(norm_dist, x, columns=['Frequency'])
    df = df.reset_index()
    df1 = df[df['index'] <= var]
    df2 = df[df['index'] > var]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df1['index'],
                         y=df1.Frequency,
                         marker_color='red'
                         ))
    fig.add_trace(go.Bar(x=df2['index'],
                         y=df2.Frequency,
                         marker_color='blue'
                         ))

    fig.update_xaxes(title='', visible=True, showticklabels=True)
    fig.add_vline(x=var, line_width=3, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=var,
        y=15,
        xref="x",
        yref="y",
        text="Risk",
        showarrow=True,
        font=dict(
            family="Times new",
            size=16,
            color="black"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        height=200, width=275,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )

    fig.update_layout(template='plotly_dark', showlegend=False)
    # fig.show()
    plotly.offline.plot(fig, filename="myapp/static/files/var.html", config={'displayModeBar': False}, auto_open=False)

    # print('Value at risk : ', round(var, 4))
    lower = port_mean - 2 * port_stdev / np.sqrt(count)
    higher = port_mean + 2 * port_stdev / np.sqrt(count)
    # print(f"Daily portfolio return will be between : {round(lower, 4) * 100}% and {round(higher, 4) * 100}%")
    # return var

weights = np.array([0.15,0.15,0.18,0.22])
tickers=['AAPL','CVM','TSLA','FB']
data = pdr.get_data_yahoo(tickers, start="2018-01-01", end=dt.date.today())['Adj Close']
# get_var(data,weights,0.05)

# EFFICIENT FRONTIER

def get_efp():

    # list of stocks in portfolio
    stocks = ['AAPL', 'CVM', 'TSLA', 'FB']

    # download daily price data for each of the stocks in the portfolio
    data = web.DataReader(stocks, data_source='yahoo', start="2020-01-01", end=dt.date.today())['Adj Close']
    data.sort_index(inplace=True)

    # convert daily stock prices into daily returns
    returns = data.pct_change()

    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    # set number of runs of random portfolio weights
    num_portfolios = 25000

    # set up array to hold results
    # We have increased the size of the array to hold the weight values for each stock
    results = np.zeros((4 + len(stocks) - 1, num_portfolios))
    for i in xrange(num_portfolios):
        # select random weights for portfolio holdings
        weights = np.array(np.random.random(4))
        # rebalance weights to sum to 1
        weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        # store results in results array
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev

        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2, i] = results[0, i] / results[1, i]

        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j + 3, i] = weights[j]

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe', stocks[0], stocks[1], stocks[2], stocks[3]])

    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]

    # locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

    # create scatter plot coloured by Sharpe Ratio
    # plt.figure(facecolor='#191c24')
    # ax = plt.axes()
    # ax.set_facecolor('#191c24')
    plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
    plt.xlabel('Volatility', color='white')
    plt.ylabel('Returns', color='white')
    plt.tick_params(axis='y', colors='white')
    plt.tick_params(axis='x', colors='white')

    cb = plt.colorbar()

    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')

    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    # plot red star to highlight position of portfolio with highest Sharpe Ratio
    plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=1000, label='Max Sharpe Ratio')
    # plot green star to highlight position of minimum variance portfolio
    plt.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=1000, label='Minimum Volatility')
    plt.legend(labelspacing=1.0)

    plt.savefig("myapp/static/files/efp.png", bbox_inches='tight', transparent=True)

get_efp()

