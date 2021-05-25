# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:52:14 2021

@author: aschauer
"""

import pandas as pd
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

import gap_modeling as gm
import scoring_utils as scu

ABO_INPUT_DATA = pd.read_csv(
    '../input_data/data_for_ABO_1984.csv',
    parse_dates = ['time'],
    index_col='time')

plot_output = '../results/abo_reconstruction/'

if __name__ == '__main__':
    
    # remove 1AD from ABO input data:
    ABO_INPUT_DATA = (ABO_INPUT_DATA
                      .loc[(ABO_INPUT_DATA['stn']!='1AD'), :])
   
    
    models = OrderedDict([
        ('elastic_net_regression', gm.ElasticNetRegressionFilling()),
        ('GIDS', gm.GidsFilling()),
        ('inverse_distance', gm.InverseDistanceSquaredFilling()),
        # ('SWE2HS_SLFTI', gm.SWE2HSSLFTI()),
        ('SWE2HS_snow17', gm.SWE2HSSnow17(shifted_dates=False)),
        # ('random_forest_V3', gm.RandomForestFilling3(grid_search=False)),
        ('random_forest_V3_5', gm.RandomForestFilling3_5(grid_search=False)),
        ('matiu', gm.MatiuFilling(weighting='vertical'))
        ])
    
    #full year missing for temp-index/SWE2HS:
    gap_period = pd.date_range(start='1983-09-01', end=f'1984-08-31')
    train_periods = {}
    train_periods['10_before_10_after'] = pd.date_range(start='1973-09-01', end='1983-08-31')
    train_periods['10_before_10_after'].append(pd.date_range(start='1984-09-01', end='1994-06-01'))
    
    train_periods['10_before_10_after_plus_NovDez1983'] = pd.date_range(start='1973-09-01', end='1983-12-31')
    train_periods['10_before_10_after_plus_NovDez1983'].append(pd.date_range(start='1984-09-01', end='1994-06-01'))
    
    train_periods['10_before'] = pd.date_range(start='1973-09-01', end='1983-08-31')
    
    train_periods['10_before_plus_NovDez1983'] = pd.date_range(start='1973-09-01', end='1983-12-31')
    
    train_periods['10_after'] = pd.date_range(start='1984-09-01', end='1994-06-01')
    
    
    
    train_periods['1982_and_fall_1983'] = pd.date_range(start='1982-11-01', end='1983-12-31')

    # other methods: missing from 01-01-1984
    
    y_preds = OrderedDict()
    for modelname, model in models.items():
        if modelname == 'SWE2HS_snow17':
            train_period = train_periods['10_before_10_after']
        else:
            train_period = train_periods['10_before_10_after_plus_NovDez1983']

            
        y_preds[modelname] = model.fit_predict(
            data=ABO_INPUT_DATA,
            train_period=train_period,
            gap_period=gap_period,
            gap_station='ABO')

    #%%
    predictions = pd.concat(list(y_preds.values()), axis=1)
    y_true = (scu.get_gap_HS_data(ABO_INPUT_DATA,
                                 gap_period,
                                 'ABO')
              .loc['1983']
              .rename('ABO measured'))
    
    chris = (scu.get_gap_HS_data(ABO_INPUT_DATA,
                                 gap_period,
                                 'ABO')
              .loc['1984']
              .rename('interpolation Christoph'))
    
    y_1ad = (scu.get_gap_HS_data(ABO_INPUT_DATA,
                               gap_period,
                               '1AD')
           .rename("1AD measured"))
    
    result = pd.concat([y_true, y_1ad, chris, predictions], axis=1)
    
    fig, ax = plt.subplots(1,1,figsize=[17,13])
    result.loc[:,['ABO measured','1AD measured']].plot(ax=ax, lw=4)

    result.drop(['ABO measured','1AD measured'], axis=1).plot(ax=ax)
    
    ax.legend(loc='upper left')
    fig.savefig('abo_reconstruction.png', bbox_inches='tight', dpi=300)
    
    
    
    