
import pandas as pd
import datetime as dt
import numpy as np


def transform_transactions(transaction, customer_id_col, datetime_col, monetary_value_col=None, activation_date=dt.datetime(1970, 1, 1)):

    # Convert the transaction date column to datetime format, assuming year is first
    transaction[datetime_col] = pd.to_datetime(transaction[datetime_col], yearfirst=True)

    # Initialize activations DataFrame with unique customer IDs and set activation date
    activations = pd.DataFrame(data={customer_id_col: transaction[customer_id_col].unique()})
    activations[datetime_col] = activation_date

    # If monetary value column is specified, initialize it with 0 in activations DataFrame
    if monetary_value_col:
        activations[monetary_value_col] = 0
    
    # Concatenate original transaction data with the activations data
    transaction = pd.concat([transaction, activations])

    # Group by date and customer ID, and sum the monetary values if column specified
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


# trunc
def trunc(transaction_data, freq='M', monetary_value_col='monetary'):
    summary_monthly = summary_data_from_transaction_data(transaction_data, 'id', 'date', monetary_value_col=monetary_value_col,freq=freq, freq_multiplier=1)
    summary_monthly['frequency'] = summary_monthly['frequency'].apply(int)
    summary_monthly['recency'] = summary_monthly['recency'].apply(int)
    summary_monthly['T'] = summary_monthly['T'].apply(int)
    trunc = summary_monthly[summary_monthly['frequency'] == summary_monthly['recency']]
    return transaction_data.set_index('id').loc[trunc.index].reset_index().copy()


# trunc
def get_trunc(transaction_data, freq='M', monetary_value_col='monetary'):
    summary_monthly = summary_data_from_transaction_data(transaction_data, 'id', 'date', monetary_value_col=monetary_value_col,freq=freq, freq_multiplier=1)
    summary_monthly['frequency'] = summary_monthly['frequency'].apply(int)
    summary_monthly['recency'] = summary_monthly['recency'].apply(int)
    summary_monthly['T'] = summary_monthly['T'].apply(int)
    trunc = summary_monthly[summary_monthly['frequency'] == summary_monthly['recency']]
    return trunc.index


def transform_trace(trace, customer_id_col='customer_id', datetime_col='date', activation_date='2022-01-01'):
    customers = trace.shape[0]
    customers_df = pd.DataFrame(data={customer_id_col: np.arange(customers)})
    customers_df[datetime_col] = pd.to_datetime(activation_date)
    customers_df['period'] = 0
    
    customers_ids = (np.ones_like(trace).T * np.arange(customers)).T
    df = pd.DataFrame(data={customer_id_col : customers_ids.ravel(), "period": trace.ravel()})
    df = df[df['period'] != 0]
    df[datetime_col] = pd.to_datetime(activation_date) + pd.to_timedelta(df['period'], unit="day")
    return pd.concat([customers_df, df])
