from __future__ import division
from __future__ import print_function
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from autograd.numpy import log, exp, logaddexp
from pandas import DataFrame
from autograd.scipy.special import gammaln, betaln, beta as betaf
from scipy.special import binom

from lifetimes.utils import _check_inputs
from lifetimes import BaseFitter

#import requirements
import lifetimes
import numpy as np



class BlankLLH():
    @staticmethod
    def _loglikelihood(params, x, tx, T, c):

        """Log likelihood for optimizer."""
        alpha, beta, gamma, delta = params

        betaln_ab = betaln(alpha, beta)
        betaln_gd = betaln(gamma, delta)
        A1 = betaln(alpha + x, beta + T - x)
        A2 = betaln(gamma, delta + T**c)
        log_A = A1 - betaln_ab + A2 - betaln_gd

        B = np.zeros_like(T)
        recency_T = T - tx - 1

        for j in np.arange(recency_T.max() + 1):
            ix = recency_T >= j
            B1 = betaln(alpha + x, beta + tx - x + j)
            B2 = betaln(gamma, delta + (tx + j)**c)
            B3 = betaln(gamma, delta + (tx + j + 1)**c)
            B = B + ix * (exp(B1 - betaln_gd + B2 - betaln_ab) - exp(B1 - betaln_gd + B3 - betaln_ab))
        
        answer = log(exp(log_A) + B)
        return answer


class BlankFitter(BaseFitter):
    "Also known as the as Beta-dWeibull / Beta-Binomial model."
    
    def __init__(self, penalizer_coef=0.0, c=1):
        """Initialization, set penalizer_coef, set c hyper param."""
        self.penalizer_coef = penalizer_coef
        self.c = c
    
    @staticmethod
    def _loglikelihood(params, x, tx, T, c):

        """Log likelihood for optimizer."""
        alpha, beta, gamma, delta = params

        betaln_ab = betaln(alpha, beta)
        betaln_gd = betaln(gamma, delta)
        A1 = betaln(alpha + x, beta + T - x)
        A2 = betaln(gamma, delta + T**c)
        log_A = A1 - betaln_ab + A2 - betaln_gd

        B = np.zeros_like(T)
        recency_T = T - tx - 1

        for j in np.arange(recency_T.max() + 1):
            ix = recency_T >= j
            B1 = betaln(alpha + x, beta + tx - x + j)
            B2 = betaln(gamma, delta + (tx + j)**c)
            B3 = betaln(gamma, delta + (tx + j + 1)**c)
            B = B + ix * (exp(B1 - betaln_gd + B2 - betaln_ab) - exp(B1 - betaln_gd + B3 - betaln_ab))
        
        answer = log(exp(log_A) + B)
        return answer
    
    
    @staticmethod
    def _negative_log_likelihood(log_params, frequency, recency, n_periods, weights, penalizer_coef=0, c=1):
        params = exp(log_params)
        penalizer = lambda x: sum(x**2)
        llh = BlankLLH._loglikelihood
        return (
            -(llh(params, frequency, recency, n_periods, c) * weights).sum()
            / weights.sum()
            + penalizer_coef * penalizer(log_params)
        )
    
    def fit(
        self,
        frequency,
        recency,
        n_periods,
        weights=None,
        initial_params=None,
        verbose=False,
        tol=1e-7,
        index=None,
        **kwargs
    ):

        frequency = np.asarray(frequency).astype(int)
        recency = np.asarray(recency).astype(int)
        n_periods = np.asarray(n_periods).astype(int)

        if weights is None:
            weights = np.ones_like(recency)
        else:
            weights = np.asarray(weights)

        _check_inputs(frequency, recency, n_periods)

        log_params_, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (frequency, recency, n_periods, weights, self.penalizer_coef, self.c), initial_params, 4, verbose, tol, **kwargs
        )
        self.params_ = pd.Series(np.exp(log_params_), index=["alpha", "beta", "gamma", "delta"])

        self.data = DataFrame(
            {"frequency": frequency, "recency": recency, "n_periods": n_periods, "weights": weights}, index=index
        )
        
        self.generate_new_data = None
        '''
        self.generate_new_data = lambda size=1: beta_geometric_beta_binom_model(
            # Making a large array replicating n by n_custs having n.
            np.array(sum([n_] * n_cust for (n_, n_cust) in zip(n_periods, weights))),
            *self._unload_params("alpha", "beta", "gamma", "delta"),
            size=size
        )
        '''

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()
        return self
    
    
    def conditional_expected_number_of_purchases_up_to_time(self, m_periods_in_future, frequency, recency, n_periods, c=1):
        r"""
        Conditional expected purchases in future time period.

        The  expected  number  of  future  transactions across the next m_periods_in_future
        transaction opportunities by a customer with purchase history
        (x, tx, n).

        .. math:: E(X(n_{periods}, n_{periods}+m_{periods_in_future})| \alpha, \beta, \gamma, \delta, frequency, recency, n_{periods})

        See (13) in Fader & Hardie 2010.

        Parameters
        ----------
        t: array_like
            time n_periods (n+t)

        Returns
        -------
        array_like
            predicted transactions

        """
        x = frequency
        tx = recency
        n = n_periods

        params = self._unload_params("alpha", "beta", "gamma", "delta")
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(self._loglikelihood(params, x, tx, n, c))
        p2 = exp(betaln(alpha + x + 1, beta + n - x) - betaln(alpha, beta))
        p3 = delta / (gamma - 1) * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        p4 = exp(gammaln(1 + delta + n) - gammaln(gamma + delta + n))
        p5 = exp(gammaln(1 + delta + n + m_periods_in_future) - gammaln(gamma + delta + n + m_periods_in_future))

        return p1 * p2 * p3 * (p4 - p5)
    
    
    
    
class BetaGeoBetaBinomEditFitter(BaseFitter):
    def __init__(self, penalizer_coef=0.0):
        """Initialization, set penalizer_coef."""
        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _loglikelihood(params, x, tx, T):
        warnings.simplefilter(action="ignore", category=FutureWarning)

        """Log likelihood for optimizer."""
        alpha, beta, gamma, delta = params

        betaln_ab = betaln(alpha, beta)
        betaln_gd = betaln(gamma, delta)

        log_A = betaln(alpha + x, beta + T - x) - betaln_ab + betaln(gamma, delta + T) - betaln_gd
        B = exp(log_A)
        recency_T = T - tx - 1

        for j in np.arange(recency_T.max() + 1):
            ix = recency_T >= j
            B1 = betaln(alpha + x, beta + tx - x + j)
            B2 = betaln(gamma + 1, delta + tx + j)
            B = B + ix * exp(B1 + B2 - betaln_gd - betaln_ab)
            
        return log(B)

    @staticmethod
    def _negative_log_likelihood(log_params, frequency, recency, n_periods, weights, penalizer_coef=0):
        params = exp(log_params)
        penalizer_term = penalizer_coef * sum(params ** 2)
        return (
            -(BetaGeoBetaBinomEditFitter._loglikelihood(params, frequency, recency, n_periods) * weights).sum()
            / weights.sum()
            + penalizer_term
        )

    def fit(
        self,
        frequency,
        recency,
        n_periods,
        weights=None,
        initial_params=None,
        verbose=False,
        tol=1e-7,
        index=None,
        **kwargs
    ):
        frequency = np.asarray(frequency).astype(int)
        recency = np.asarray(recency).astype(int)
        n_periods = np.asarray(n_periods).astype(int)

        if weights is None:
            weights = np.ones_like(recency)
        else:
            weights = np.asarray(weights)

        _check_inputs(frequency, recency, n_periods)

        log_params_, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (frequency, recency, n_periods, weights, self.penalizer_coef), initial_params, 4, verbose, tol, **kwargs
        )
        self.params_ = pd.Series(np.exp(log_params_), index=["alpha", "beta", "gamma", "delta"])

        self.data = DataFrame(
            {"frequency": frequency, "recency": recency, "n_periods": n_periods, "weights": weights}, index=index
        )

        self.generate_new_data = lambda size=1: beta_geometric_beta_binom_model(
            # Making a large array replicating n by n_custs having n.
            np.array(sum([n_] * n_cust for (n_, n_cust) in zip(n_periods, weights))),
            *self._unload_params("alpha", "beta", "gamma", "delta"),
            size=size
        )

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()
        return self
    
    
    
    

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
