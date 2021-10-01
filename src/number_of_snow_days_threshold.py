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
import cv_score_tables

METHODS = [
    'SingleStation_best_correlated_mean_ratio',
    'Inverse distance squared',
    'matiu vertical weighted_min_corr_-1.0',
    'Elastic Net Regression',
    'RandomForest_V3.5',
    'SWE2HS_Snow17_shifted_dates',
    # 'ERA5-land_mean_ratio',
    # 'ERA5-land_no_scaling',
    # 'ERA5-land_RF_surrounding_gridcells_max_depth_70_n_estimators_200'
    ]

def calculate_dHSn(HS_file, n=1):
    hs_series = pd.read_pickle(HS_file)
    return scu.dHSn(hs_series, n)

if __name__ == '__main__':
    
    df = get_cv_results_as_df()
    df = df.loc[df['gap_type']=='LOWO']
    df = df.rename(columns={'bias': 'BIAS'})
    df = df.loc[df.fill_method.isin(METHODS), :]

    for n in {2,5,10,30}:
        df[f'dHS{n}_true'] = df.apply(lambda x: calculate_dHSn(x.HS_true_file, n=n), axis=1)
        df[f'dHS{n}_pred'] = df.apply(lambda x: calculate_dHSn(x.HS_pred_file, n=n), axis=1)
    
    plot_output = '../results/revision/'
    if not os.path.isdir(plot_output):
        os.makedirs(plot_output)
    
    cv_evaluate_scores.scatterplot_true_vs_pred(
        METHODS,
        ['dHS1','dHS2','dHS5','dHS10','dHS30'],
        filename=f"{plot_output}number_of_snowdays_different_HS_thresholds.png",
        dpi=150,
        legend_kw={
            'bbox_to_anchor': 'below_titles',
            },
        equal_xy_axes=True,
        fitlines=True,
        print_score_values=False,
        score_df=df,
        panel_height=2.0,
        panel_width=2.0,
        sharex=True,
        sharey=True,
        individual_panel_labels=False,
        global_x_label='measured [days]',
        global_y_label='modeled [days]'
        )
    
    
        
    score_df, latexstring = cv_score_tables.score_table_true_pred(
        score_data=df,
        methods_used=METHODS,
        climate_metrics=['dHS1','dHS2','dHS5','dHS10','dHS30'],
        filename=f"{plot_output}score_table_number_of_snowdays_thresholds.tex"
        )
    
    
    for n in {2,5,10,30}:
        df[f'dHS{n}_abs_diff'] = df[f'dHS{n}_true'].sub(df[f'dHS{n}_pred']).abs()
    cv_evaluate_scores.scatterboxbins(
        METHODS,
        metrics_used=['dHS1_abs_diff',
                     'dHS2_abs_diff',
                     'dHS5_abs_diff',
                     'dHS10_abs_diff',
                     'dHS30_abs_diff'],
        xaxs_value='HSavg_true',
        filename=f'{plot_output}scatterbox_dHSn_abs_diff_vs_HSavg.png',
        dpi=600,
        legend_kw={'bbox_to_anchor':'below_titles'},
        showfliers=True,
        score_df=df
        )
    
    
    for n in {2,5,10,30}:
        df[f'dHS{n}_diff'] = df[f'dHS{n}_true'].sub(df[f'dHS{n}_pred'])
    cv_evaluate_scores.scatterboxbins(
        METHODS,
        metrics_used=['dHS1_diff',
                     'dHS2_diff',
                     'dHS5_diff',
                     'dHS10_diff',
                     'dHS30_diff'],
        xaxs_value='HSavg_true',
        filename=f'{plot_output}scatterbox_dHSn_diff_vs_HSavg.png',
        dpi=600,
        legend_kw={'bbox_to_anchor':'below_titles'},
        showfliers=True,
        score_df=df
        )