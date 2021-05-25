# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:06:28 2021

@author: aschauer
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def get_gap_HS_data(modeling_data, gap_period, gap_station):
    """
    Get original HS data in the gap period. Only winter values (Nov-Apr) will 
    be returned.
    """
    y_true = (modeling_data
              .loc[modeling_data['stn']==gap_station, 'HS']
              .loc[gap_period]
              )
    y_true = y_true.loc[y_true.index.month.isin([11,12,1,2,3,4])]
    y_true.name = f'Measured HS Data'
    return y_true


def get_station_altitude(modeling_data, gap_period, gap_station):
    altitude = (modeling_data
                .loc[modeling_data['stn']==gap_station, 'Z']
                .loc[gap_period]
                .iloc[0])
    return altitude

def HSavg(series):
    return series.mean()

def dHS1(series):
    if series.isna().all():
        result = np.nan
    else:
        result = np.count_nonzero(series.values >= 1)
    return result

def HSmax(series):
    return series.max()

def _HSavg_diff(y_true, y_hat):
    return np.mean(y_hat) - np.mean(y_true)

def _HSavg_abs_diff(y_true, y_hat):
    return np.abs(np.mean(y_hat) -np.mean(y_true))
    
def _HSavg_relative_diff(y_true, y_hat):
    return (np.mean(y_hat) - np.mean(y_true))/np.mean(y_true)

def _HSavg_relative_abs_diff(y_true, y_hat):
    return np.abs(np.mean(y_hat) - np.mean(y_true))/np.mean(y_true)

def _dHS1_diff(y_true, y_hat):
    return dHS1(y_hat)-dHS1(y_true)

def _dHS1_abs_diff(y_true, y_hat):
    return np.abs(_dHS1_diff(y_true, y_hat))

def _dHS1_relative_diff(y_true, y_hat):
    return (_dHS1_diff(y_true, y_hat)) / dHS1(y_true)

def _dHS1_relative_abs_diff(y_true, y_hat):
    return (_dHS1_abs_diff(y_true, y_hat)) / dHS1(y_true)

def _HSmax_diff(y_true, y_hat):
    return np.max(y_hat) - np.max(y_true)

def _HSmax_abs_diff(y_true, y_hat):
    return np.abs(np.max(y_hat) -np.max(y_true))
    
def _HSmax_relative_diff(y_true, y_hat):
    return (np.max(y_hat) - np.max(y_true))/np.max(y_true)

def _HSmax_relative_abs_diff(y_true, y_hat):
    return np.abs(np.max(y_hat) - np.max(y_true))/np.max(y_true)

def _maape_score(y_true, y_hat):
    """
    mean arctangent absolute percentage error (MAAPE) calculated by::
        mean(artan(abs(error/y_true))))
        
    Reference: 
    Kim, S., & Kim, H. (2016). A new metric of absolute percentage error for 
    intermittent demand forecasts. International Journal of Forecasting, 32(3), 
    669-679.
    """
    assert(len(y_true) == len(y_hat))
    error = y_true-y_hat
    # only divide if error is not zero (leave it as zero, avoid 0/0), and dont 
    # divide if y is zero (avoide division by zero):
    percentage_error = np.divide(error, y_true, out=np.zeros_like(y_true), 
                                 where=(error!=0) & (y_true!=0))
    # if error is not zero and y is zero set percentage error to infinity
    percentage_error[(error!=0) & (y_true==0)] = np.inf
    return np.mean(np.arctan(np.abs(percentage_error)))


def _bias_score(y_true, y_hat):
    
    assert(len(y_true) == len(y_hat))
    error = y_hat-y_true
    return np.average(error)


def get_climate_score_value(y_true, y_hat, metric):
    
    func = {'HSavg_diff': _HSavg_diff,
            'HSavg_abs_diff': _HSavg_abs_diff,
            'HSavg_relative_diff': _HSavg_relative_diff,
            'HSavg_relative_abs_diff': _HSavg_relative_abs_diff,
            'dHS1_diff': _dHS1_diff,
            'dHS1_abs_diff': _dHS1_abs_diff,
            'dHS1_relative_diff': _dHS1_relative_diff,
            'dHS1_relative_abs_diff': _dHS1_relative_abs_diff,
            'HSmax_diff': _HSmax_diff,
            'HSmax_abs_diff': _HSmax_abs_diff,
            'HSmax_relative_diff': _HSmax_relative_diff,
            'HSmax_relative_abs_diff': _HSmax_relative_abs_diff}
    assert metric in func.keys()
    try:
        result = func[metric](y_true, y_hat)
    except ZeroDivisionError: # exception for the relative errors
        result = np.nan
    return result


def get_daily_score_value(y_true, y_pred, metric):
    
    assert metric in ['RMSE',
                      'RMSE_nonzero',
                      'RMSE_nonzero_true',
                      'RMSE_nonzero_pred',
                      'MAAPE',
                      'MAAPE_nonzero',
                      'MAAPE_nonzero_true',
                      'MAAPE_nonzero_pred',
                      'bias',
                      'r2_score',
                      'r2_score_nonzero',
                      'r2_score_nonzero_true',
                      'r2_score_nonzero_pred']
    
    if 'nonzero' in metric:
        data = pd.DataFrame({'y_true': y_true,
                             'y_pred': y_pred},
                            index = y_true.index)
        if 'true' in metric:
            data = data.loc[data['y_true']!=0, :]
        elif 'pred' in metric:
            data = data.loc[data['y_pred']!=0, :]
        else:  # no zero in true and pred
            data = data.loc[data.all(axis=1)]
        
        y_true = data['y_true']
        y_pred = data['y_pred']

    if y_true.size == 0 or y_pred.isna().all():
        score_value = np.nan

    elif 'RMSE' in metric:
        score_value = np.sqrt(mean_squared_error(y_true.values,
                                                 y_pred.values))
    elif 'MAAPE' in metric:
        score_value = _maape_score(y_true.values,
                                   y_pred.values)
    elif 'bias' in metric:
        score_value = _bias_score(y_true.values,
                                  y_pred.values)
    elif 'r2_score' in metric:
        score_value = r2_score(y_true.values, y_pred.values)
    
    return score_value