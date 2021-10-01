# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:51:45 2021

@author: aschauer
"""
import pandas as pd
import numpy as np
import re

from sklearn.metrics import r2_score, mean_squared_error
import scoring_utils as scu
import plotting_utils as pu
from cv_results_database import get_cv_results_as_df

def score_table_true_pred(score_data,
                          methods_used,
                          climate_metrics,
                          filename=None,
                          **to_latex_kwargs):
    """
    

    Parameters
    ----------
    score_data : TYPE
        DESCRIPTION.
    methods_used : TYPE
        DESCRIPTION.
    climate_metrics : TYPE
        DESCRIPTION.
    filename : TYPE, optional
        DESCRIPTION. The default is None.
    **to_latex_kwargs : kwargs
        e.g. caption or label

    Returns
    -------
    out : TYPE
        DESCRIPTION.
    tablestr : TYPE
        DESCRIPTION.

    """
    score_annotations = {}
    
    
    grid_idx = []
    metric_idx = []
    method_idx = []
    r2 = []
    rmse = []
    bias = []
    

    for station_grid in ['full','only_target_stations']:
        for row, metric in enumerate(climate_metrics):
            for column, method in enumerate(methods_used):
                subset = score_data.loc[(score_data['fill_method']==method) & (score_data['station_grid']==station_grid)].dropna()
                true = subset[f'{metric}_true']
                pred = subset[f'{metric}_pred']
                
                grid_idx.append(station_grid)
                metric_idx.append(metric)
                method_idx.append(method)
                try:
                    r2.append(r2_score(true, pred))
                    rmse.append(np.sqrt(mean_squared_error(true, pred)))
                    bias.append(scu._bias_score(true, pred))
                except TypeError:
                    # only nans are in y_pred (for some stations/years for IDS)
                    r2.append(np.nan)
                    rmse.append(np.nan)
                    bias.append(np.nan)
    
    scores = pd.DataFrame({'station network':grid_idx,
                           'method':method_idx,
                           'climate metric':metric_idx,
                           '$r^{2}$':r2,
                           'RMSE':rmse,
                           'BIAS':bias})
    
    scores['method'] = scores['method'].map(pu.METHOD_NAMES)
    # using pd.Categorical for keeping right order while unstacking:
    scores['climate metric'] = pd.Categorical(scores['climate metric'],
                                              categories=climate_metrics,
                                              ordered=True)
    scores['station network'] = scores['station network'].map(pu.GRIDLABELS)
    
    scores = scores.set_index(['method','climate metric','station network'])
    scores.columns.name = 'score metric'
    
    scores = scores.stack()
    
    out = scores.unstack(level=['station network', 'method'])
        
    tablestr = out.to_latex(
        # "climate_metrics_scores_table.tex",
        float_format="%.2f",
        sparsify=True,
        multirow=True,
        position='t',
        escape=False,
        bold_rows=True,
        multicolumn_format='l',
        **to_latex_kwargs
        )
    
    tablestr = re.sub('toprule', 'tophline', tablestr)
    tablestr = re.sub('midrule', 'middlehline', tablestr)
    tablestr = re.sub('bottomrule', 'bottomhline', tablestr)
    
    if filename is not None:
        with open(filename, "w") as f:
            f.write(tablestr)
        
    return out, tablestr

if __name__ == '__main__':

    df = get_cv_results_as_df()
    df = df.loc[df['gap_type']=='LOWO']
    df = df.rename(columns={'bias': 'BIAS'})
    
    methods = [
    'SingleStation_best_correlated_mean_ratio',
    'Inverse distance squared',
    'matiu vertical weighted_min_corr_-1.0',
    'Elastic Net Regression',
    'RandomForest_V3.5',
    'SWE2HS_Snow17_shifted_dates',
    ]
    
    metrics = ['HSavg','HSmax','dHS1']
    
    scores_table, latex_string = score_table_true_pred(
        df,
        methods,
        metrics,
        filename="../results/revision/climate_metrics_scores_table.tex",
        label="tab:climate_metrics_scores_table",
        caption=("BIAS, RMSE and coefficient of determination ($r^2$) for"
                 " the three climate metrics HSavg, HSmax and dHS1"
                 " reconstructed with the different methods in the dense and"
                 " sparse station networks as seen in "
                 "Figure~\ref{fig:scatter_true_vs_pred_climate_metrics}."),
        )
    
    
    scores_table, latex_string = score_table_true_pred(
        df,
        ['ERA5-land_no_scaling',
         'ERA5-land_mean_ratio',
         'ERA5-land_RF_surrounding_gridcells_max_depth_70_n_estimators_200'
         ],
        metrics,
        filename="../results/revision/era5_climate_metrics_scores_table.tex",
        
        )