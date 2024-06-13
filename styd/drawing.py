
import lifetimes
import numpy as np


import matplotlib
from matplotlib import pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')

import scipy.stats as ss
from scipy import optimize
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import pandas as pd
from scipy.special import beta as beta_fun
from scipy.integrate import quad
import statsmodels.api as sm

from itertools import product
import datetime as dt



def draw_betas(alpha, beta, gamma, delta, figure=None):
    
    if figure:
        fig, axs = figure
    else:
        fig, axs = plt.subplots(1, 1)
    
    I = np.linspace(0.01, 0.99, 100)
    axs.plot(I, ss.beta(alpha, beta).pdf(I))
    axs.plot(I, ss.beta(gamma, delta).pdf(I))
    axs.axvline(alpha / (alpha + beta), ls='-.', color='blue', label="transaction rate = {:.3f}".format(alpha / (alpha + beta)))
    axs.axvline(gamma / (gamma + delta), ls='-.', color='red', label="dropout rate = {:.3f}".format(gamma / (gamma + delta)))
    fig.legend()
    return fig, axs


def draw_hists(df, bins=30,figure=None):
    
    if figure:
        fig, axs = figure
    else:
        fig, axs = plt.subplots(1, 1)
        
    axs.hist(df['p'], bins=bins, alpha=0.25, color='blue', density=True, label='p')
    axs.hist(df['frequency'] / df['recency'], bins=bins, alpha=0.25, color='purple', density=True, label='F/R')
    axs.hist(df['theta'], bins=bins, alpha=0.25, color='orange', density=True, label='theta')
    fig.legend()
    
    return fig, axs


def draw_daily_retention(transaction_data_, activation_date=None):
    fig, axs = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    #period = np.arange(T)
    #R_win = lambda t: R_w(t, ALPHA, BETA, GAMMA, DELTA, c=C, theta=THETA, p=P)
    #axs.plot(period, R_win(period), color='black', ls='--')
    if not activation_date:
        activation_date = min(transaction_data_['date'])

    cap = len(transaction_data_[transaction_data_['date'] == pd.to_datetime(activation_date)]['id'].unique())
    R_dayly = transaction_stream(transaction_data_, 'id', 'date', freq='D')
    axs.plot(np.arange(len(R_dayly)), R_dayly['id'] / cap,color='#75bbfd', label="R_sample")

    axs.legend(fontsize=15)
    axs.set_title("E(Retention)")
    axs.set(xlabel='Period', ylabel='Retention')

    return fig, axs

def draw_daily_wretention(T, alpha, beta, gamma, delta, c, theta, p, figure=None, clr='black'):
    
    if figure:
        fig, axs = figure
    else:
        raise BaseException('Give me FACT!')
        
    period = np.arange(T)
    R_win = lambda t: R_w(t, alpha, beta, gamma, delta, c=c, theta=theta, p=p)
    axs.plot(period, R_win(period), color=clr, ls='--')
    
    axs.legend(fontsize=15)
    axs.set_title("E(Retention)")
    axs.set(xlabel='Period', ylabel='Retention')
    
    return fig, axs

# draw_betas(ALPHA, BETA, GAMMA, DELTA)[0].show()



def transform_transactions(transaction, customer_id_col, datetime_col, monetary_value_col=None, activation_date=dt.datetime(1970, 1, 1)):
    transaction[datetime_col] = pd.to_datetime(transaction[datetime_col], yearfirst=True)
    activations = pd.DataFrame(data={customer_id_col:transaction[customer_id_col].unique()})
    activations[datetime_col] = activation_date
    if monetary_value_col:
        activations[monetary_value_col] = 0
    
    transaction = pd.concat([transaction, activations])
    return transaction.groupby(by=[datetime_col, customer_id_col], as_index=False).sum([monetary_value_col])
    

def transaction_stream(transaction, customer_id_col, datetime_col, freq='D'):
    ret_table = transaction.copy()
    ret_table[datetime_col] = ret_table[datetime_col].dt.to_period(freq).dt.to_timestamp()
    ret_table = ret_table.groupby(by=[datetime_col], as_index=False, sort=True)[[customer_id_col]].nunique()
    # apply(lambda customer_ids: len(set(customer_ids)))
    return ret_table

def value_stream(transaction, monetary_col, datetime_col, freq='D'):
    value_table = transaction.copy()
    value_table[datetime_col] = value_table[datetime_col].dt.to_period(freq).dt.to_timestamp()
    value_table = value_table.groupby(by=[datetime_col], as_index=False, sort=True)[[monetary_col]].sum()
    # apply(lambda customer_ids: len(set(customer_ids)))
    return value_table


def S_w(t, gamma, delta, c=1, theta=None):#survival function
    if theta:
        return (1 - theta) ** (t ** c)
    else:
        return beta_fun(gamma, delta + t**c) / beta_fun(gamma, delta)

    
def r_w(t, gamma, delta, c=1, theta=None):#retention rate
    '''
    t must by greater zero
    '''
    if theta:
        return (1 - theta) ** (t**c - (t - 1)**c)
    else:
        return S_w(t, gamma, delta, c) / S_agg(t - 1, gamma, delta, c)

    
def R_w(t, alpha, beta, gamma, delta, c=1, theta=None, p=None): #retention function
    S_win = lambda t: S_w(t, gamma, delta, c, theta)
    
    if p:
        return p * (t > 0) * S_win(t) + (t == 0)
    else:
        return (alpha / (alpha + beta)) * (t > 0) * S_win(t) + (t == 0)
