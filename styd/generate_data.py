from numpy import random
#import requirements
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

def dWeib(theta,c=1,size=1):
    u = random.uniform(0, 1, size=size)
    dw = (np.log(1 - u) / np.log(1 - theta)) ** (1 / c)
    return np.ceil(dw)



def beta_dWeib_transaction_model(N, 
                                 alpha, beta,
                                 gamma, delta,
                                 p=None,
                                 theta=None, c=1,
                                 size=1):
    T = int(N)
    if type(N) in [float, int, np.int64]:
        N = N * np.ones(size)
    else:
        N = np.asarray(N)

    if p:
        probability_of_post_purchase_death = p * np.ones(size)
    else:
        probability_of_post_purchase_death = random.beta(a=alpha, b=beta, size=size)
        
    if theta:
        thetas = theta * np.ones(size)
    else:
        thetas = random.beta(a=gamma, b=delta, size=size)
        
        
    death_time = dWeib(thetas, c=c, size=size)
    alive = N <= death_time
    transactions = random.binomial(1, probability_of_post_purchase_death , size=(T, size)).T
    E = np.ones_like(transactions)
    mask = (E * (np.arange(T) + 1) < (E.T * death_time).T)
    transactions = transactions * mask
     
    frequency = transactions.sum(axis=-1)
    trace = transactions * (np.arange(T) + 1)
    recency = trace.max(axis=-1)
    
    opportunities = death_time - 1
    opportunities[T < opportunities] = T
    
    df = pd.DataFrame({
        'customer_id' : np.arange(len(N)),
        'frequency' : frequency,
        'recency' : recency,
        'n_periods' : N,
        'p' : probability_of_post_purchase_death,
        'theta' : thetas,
        'alive' : alive.astype(np.float64),
        'death_time' : death_time,
        'opportunities': opportunities
    })
    return df, trace
