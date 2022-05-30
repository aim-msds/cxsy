import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pymannkendall as mk

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip
from patsy import dmatrices

from scipy.stats import skew, kurtosis
from itertools import product



from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt


# used to detrend
def difference(dataset, interval = 1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


def roll_window(dataset, win_size, func):
    return dataset.rolling(int(win_size)).apply(func)

def do_ews_std(dataset, win_size):
    return dataset.rolling(int(win_size)).std()

def do_ews_skew(dataset, win_size):
    return dataset.rolling(int(win_size)).apply(skew)

def do_ews_kurt(dataset, time, win_size):
    return dataset.rolling(int(win_size)).apply(kurtosis)

def get_auto(dataset, lag=1):
    return sm.tsa.acf(dataset)[lag]

def do_ews_auto(dataset, win_size, lag=1):
    return dataset.rolling(int(win_size)).apply(get_auto)

def get_ch(dataset, time):
    df = pd.DataFrame(dataset)
    df['Time'] = time.loc[dataset.index]
    df[f'LOG_'] = np.log(dataset)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    if df.empty:
        return np.nan
    expr = f'LOG_ ~ Time'
    y, X = dmatrices(expr, df, return_type='dataframe')
    olsr_results = smf.ols(expr, df).fit()
    keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
    results = het_white(olsr_results.resid, X)
    return results[1]

def do_ews_ch(dataset, time, win_size):
    return dataset.rolling(int(win_size)).apply(get_ch, args =(time,) )

def do_ar(dataset, lag=1):
    try:
        model = AutoReg(dataset, lags=lag)
        model_fit = model.fit()
        return model_fit.params[1]
    except Exception as e: 
        print(f"Warning: {e}")
        return np.nan

def do_ews_ar(dataset, win_size, lag=1):
    return dataset.rolling(int(win_size)).apply(do_ar)

def plot(data, dic):
    method_name = dic['method_name']
    orig = dic['data']
    diff = pd.DataFrame(difference(dic['data']))
    
    if method_name == 'Std':
        orig_process = do_ews_std(orig, win_size=int(len(orig) / 2))
        resid = do_ews_std(diff, win_size=int(len(diff) / 2))
        
    elif method_name == 'Autocorrelation':
        orig_process = do_ews_auto(orig, win_size=int(len(orig) / 2))
        resid = do_ews_auto(diff, win_size=int(len(diff) / 2))
    
    elif method_name == 'Skewness':
        orig_process= do_ews_skew(orig, win_size=int(len(orig)/2))
        resid= do_ews_skew(diff, win_size=int(len(diff)/2))
        
    elif method_name == 'CH':
        time = dic['time']
        orig_process = do_ews_ch(orig, time, win_size=int(len(orig) / 2))
        resid = do_ews_ch(diff, time, win_size=int(len(diff) / 2))
        
    elif method_name == 'Autoregression':
        orig_process = do_ews_ar(orig, win_size=int(len(orig) / 2), lag=1)
        resid = do_ews_ar(diff, win_size=int(len(diff) / 2), lag=1)
        
    orig_process.index = orig_process.index + 1
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 5)) 

    axes[0, 0].plot(orig)
    axes[0, 0].set_title(dic['title'])
    axes[0, 0].set_xlabel("Timestep")

    axes[0, 1].plot(diff)
    axes[0, 1].set_title("Residuals")
    axes[0, 1].set_xlabel("Timestep")

    axes[1, 0].plot(orig_process, color="red")
    axes[1, 0].set_title(method_name + " on Data")
    axes[1, 0].set_xlabel("Timestep")

    axes[1, 1].plot(resid, color="red")
    axes[1, 1].set_title(method_name + " on Residuals")
    axes[1, 1].set_xlabel("Timestep")
    
    fig.tight_layout()
    
    ml_result = mk.original_test(orig_process)
    print ('Mann Kendall Test Results')
    print ('-------------------------')
    print(f'Trend : {ml_result.trend}')
    print(f'h : {ml_result.h}')
    print(f'p : {ml_result.p}')
    print(f'Tau : {ml_result.Tau}')
    print(f'Slope : {ml_result.slope}')