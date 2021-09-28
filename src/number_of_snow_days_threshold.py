# -*- coding: utf-8 -*-
"""
Evaluate the influence of different HS thresholds for number of snow days dHSn.

Created on Mon Sep 20 16:44:23 2021

@author: aschauer
"""
import os
import pandas as pd

from cv_results_database import get_cv_results_as_df
import scoring_utils as scu

import cv_evaluate_scores

METHODS = [
    'SingleStation_best_correlated_mean_ratio',
    'Inverse distance squared',
    'matiu vertical weighted_min_corr_-1.0',
    'Elastic Net Regression',
    'RandomForest_V3.5',
    'SWE2HS_Snow17_shifted_dates',
    'ERA5-land_mean_ratio',
    'ERA5-land_no_scaling'
    ]

def calculate_dHSn(HS_file, n=1):
    hs_series = pd.read_pickle(HS_file)
    return scu.dHSn(hs_series, n)

if __name__ == '__main__':
    
    df = get_cv_results_as_df()
    df = df.loc[df['gap_type']=='LOWO']
    df = df.rename(columns={'bias': 'BIAS'})
    df = df.loc[df.fill_method.isin(METHODS), :]

    for n in {2,3,4,5,10,20,30,40}:
        df[f'dHS{n}_true'] = df.apply(lambda x: calculate_dHSn(x.HS_true_file, n=n), axis=1)
        df[f'dHS{n}_pred'] = df.apply(lambda x: calculate_dHSn(x.HS_pred_file, n=n), axis=1)
    
    plot_output = '../results/revision/'
    if not os.path.isdir(plot_output):
        os.makedirs(plot_output)
    
    cv_evaluate_scores.scatterplot_true_vs_pred(
        METHODS,
        ['dHS1','dHS2','dHS3','dHS4','dHS5','dHS10','dHS20','dHS30','dHS40'],
        filename=f"{plot_output}number_of_snowdays_different_HS_thresholds.png",
        dpi=150,
        legend_kw={
            'bbox_to_anchor': (0.02, 1.01),
            'loc': 'upper left',
            'ncol': 2},
        equal_xy_axes=True,
        fitlines=True,
        score_df=df)