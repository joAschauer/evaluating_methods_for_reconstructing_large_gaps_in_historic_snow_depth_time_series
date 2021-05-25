# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:27:51 2021

@author: aschauer
"""
import os
import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from cv_results_database import get_cv_results_as_df
import plotting_utils as pu
import scoring_utils as scu
from sklearn.metrics import r2_score, mean_squared_error


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sns.set_color_codes(palette='deep')

sc = get_cv_results_as_df()

sc = sc.loc[sc['gap_type']=='LOWO']
sc = sc.rename(columns={'bias': 'BIAS'})

methods_used = [
    'SingleStation_best_correlated_mean_ratio',
    'Inverse distance squared',
    'matiu vertical weighted_min_corr_-1.0',
    'Elastic Net Regression',
    'RandomForest_V3.5',
    'SWE2HS_Snow17_shifted_dates']


plot_data = sc.loc[sc['fill_method'].isin(methods_used)].copy()
plot_data.replace(to_replace={'fill_method':pu.METHOD_NAMES}, inplace=True)

def calculate_scores(df):
    scores = {}
    for climate_metric in ['HSavg','HSmax','dHS1']:
        data = (df.loc[:,[f'{climate_metric}_true',f'{climate_metric}_pred']]
                .copy()
                .dropna())
        scores[f'r2_{climate_metric}'] = r2_score(data[f'{climate_metric}_true'], data[f'{climate_metric}_pred'])
        scores[f'rmse_{climate_metric}'] = np.sqrt(mean_squared_error(data[f'{climate_metric}_true'], data[f'{climate_metric}_pred']))
        scores[f'MAAPE_{climate_metric}'] = scu._maape_score(data[f'{climate_metric}_true'], data[f'{climate_metric}_pred'])
        scores[f'bias_{climate_metric}'] = scu._bias_score(data[f'{climate_metric}_true'], data[f'{climate_metric}_pred'])
    return pd.Series(scores)

grouped_scores = (plot_data.groupby(['fill_method','station_grid'])
                  .apply(calculate_scores)
                  .reset_index()
                  )

for metric in ['rmse','r2','MAAPE','bias']:
    for clim_indi in ['HSavg', 'HSmax', 'dHS1']:
        sns.barplot(data=grouped_scores,
                    x='fill_method',
                    y=f'{metric}_{clim_indi}',
                    order=[pu.METHOD_NAMES[m] for m in methods_used],
                    hue = 'station_grid',
                    hue_order = ['full', 'only_target_stations'],
                    palette=['C1', 'C0']
                    )
        plt.show()