import aesara.tensor as at
import numpy as np

def betaln(x, y):
  return at.gammaln(x) + at.gammaln(y) - at.gammaln(x + y)

def betaf(x, y):
  return at.gamma(x) * at.gamma(y) / at.gamma(x + y)

def _loglikelihood(x, tx, T, alpha, beta, gamma, delta):
    """Log likelihood for optimizer."""
    x = at.as_tensor_variable(x)
    tx = at.as_tensor_variable(tx)
    T = at.as_tensor_variable(T)

    betaln_ab = betaln(alpha, beta)
    betaln_gd = betaln(gamma, delta)

    A = betaln(alpha + x, beta + T - x) - betaln_ab + betaln(gamma, delta + T) - betaln_gd

    recency_T = T - tx - 1
    j = at.arange(recency_T.max() + 1)[:, None]
    ix = at.where(recency_T[None, :] >= j, 1, 0)

    # Вычисляем логарифм бета-функции
    log_beta_values = (betaln(alpha + x, beta + tx - x + j) + betaln(gamma + 1, delta + tx + j) - betaln_ab - betaln_gd)

    # Используем logsumexp для суммирования в логарифмической шкале
    B = at.sum(ix * at.exp(log_beta_values), axis=0)

    # Возвращаем результат в линейной шкале
    loglike = at.log(B + at.exp(A))
    return loglike

def _loglikelihood_dw(x, tx, T, alpha, beta, gamma, delta, c):
    """Log likelihood for optimizer."""
    x = at.as_tensor_variable(x)
    tx = at.as_tensor_variable(tx)
    T = at.as_tensor_variable(T)

    betaln_ab = betaln(alpha, beta)
    betaln_gd = betaln(gamma, delta)

    A1 = betaln(alpha + x, beta + T - x)
    A2 = betaln(gamma, delta + T**c)
    A = A1 - betaln_ab + A2 - betaln_gd

    recency_T = T - tx - 1
    j = at.arange(recency_T.max() + 1)[:, None]
    ix = at.where(recency_T[None, :] >= j, 1, 0)

    # Вычисляем логарифм бета-функции
    b0 = betaln(alpha + x, beta + tx - x + j)
    b1 = betaln(gamma, delta + (tx + j)**c) #betaln(gamma + 1, delta + tx + j)
    b2 = betaln(gamma, delta + (tx + j + 1)**c)

    log_beta_values_1 = (b0 + b1 - betaln_ab - betaln_gd)
    log_beta_values_2 = (b0 + b2 - betaln_ab - betaln_gd)

    # Используем logsumexp для суммирования в логарифмической шкале
    #B = at.sum(ix * (at.exp(log_beta_values_1) - at.exp(log_beta_values_2)), axis=0)
    B1 = at.sum(ix * at.exp(log_beta_values_1), axis=0)
    B2 = at.sum(ix * at.exp(log_beta_values_2), axis=0)
    B = B1 - B2

    # Возвращаем результат в линейной шкале
    loglike = at.log(B + at.exp(A))
    return loglike

def _loglikelihood_sbb(x, tx, T, p, theta, c):

    x = at.as_tensor_variable(x)
    tx = at.as_tensor_variable(tx)
    T = at.as_tensor_variable(T)

    
    A = x * at.log(p) + (T - x) * at.log(1 - p) + (T**c) * at.log(1 - theta)

    recency_T = T - tx - 1
    j = at.arange(recency_T.max() + 1)[:, None]
    ix = at.where(recency_T[None, :] >= j, 1, 0)

    log_beta_values_1 = x * at.log(p) + (tx - x + j) * at.log(1 - p) + ((tx + j)**c) * at.log(1 - theta)
    log_beta_values_2 = x * at.log(p) + (tx - x + j) * at.log(1 - p) + ((tx + j + 1)**c) * at.log(1 - theta)

    B1 = at.sum(ix * at.exp(log_beta_values_1), axis=0)
    B2 = at.sum(ix * at.exp(log_beta_values_2), axis=0)
    B = B1 - B2

    loglike = at.log(B + at.exp(A))
    return loglike
