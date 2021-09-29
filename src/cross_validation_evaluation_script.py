# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:14:33 2021

@author: aschauer
"""
import os
import logging

import cv_evaluate_scores
import cv_evaluate_good_and_bad_examples
import cv_evaluate_true_vs_pred_daily
import cv_evaluate_climate_indices_timeseries
import plot_influence_of_number_of_stations
import station_elevation_distribution

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PAPER_FIGURE_DIR = '../results/paper_figures/'
if not os.path.isdir(PAPER_FIGURE_DIR):
    os.makedirs(PAPER_FIGURE_DIR)


METHODS = [
    'SingleStation_best_correlated_mean_ratio',
    'Inverse distance squared',
    'matiu vertical weighted_min_corr_-1.0',
    'Elastic Net Regression',
    'RandomForest_V3.5',
    'SWE2HS_Snow17_shifted_dates',
    # 'ERA5-land_mean_ratio',
    # 'ERA5-land_no_scaling',
    'ERA5-land_RF_surrounding_gridcells_max_depth_70_n_estimators_200'
    ]

def main():
    
    # # run the different evaluation scripts (generates a lot of figures
    # # for exploration, figure storage locations are logged)
    
    # cv_evaluate_scores.main(METHODS)
    # cv_evaluate_good_and_bad_examples.main(methods_used=METHODS)
    # cv_evaluate_climate_indices_timeseries.main(methods_used=METHODS)
    # cv_evaluate_true_vs_pred_daily.main(methods_used=METHODS)
    # station_elevation_distribution.main()
    
    
    # Make Figures for the paper and save to PAPER_FIGURE_DIR
    logger.info(("create figures for paper and save to\n"
                  f"{os.path.abspath(PAPER_FIGURE_DIR)}"))
    
    # NUMBER OF STATIONS:
    # -------------------

    # FIGURE 2: influence of number of stations:
    plot_influence_of_number_of_stations.make_paper_figure_boxplots(
        filename=f'{PAPER_FIGURE_DIR}fig02_boxplots_number_of_stations.png',
        dpi=500,
        station_grid='full',
        methods_used=[
            'Inverse distance squared',
            'matiu vertical weighted_min_corr_-1.0',
            'Elastic Net Regression',
            'RandomForest_V3.5']
        )
    
    
    # DAILY VALUES / SCORES:
    # ----------------------
    
    # FIGURE 3: Scatterplots of reconstructed vs true daily values
    cv_evaluate_true_vs_pred_daily.scatterplot_true_pred_daily(
        METHODS,
        f'{PAPER_FIGURE_DIR}fig03_scatter_true_pred_daily.png',
        dpi=700,
        equal_xy_axes=True)
    
    # Scatterplots rmse/MAAPE vs HSavg along with boxplots:
    cv_evaluate_scores.scatterboxbins(
        METHODS,
        ['RMSE', 'BIAS'],
        'HSavg_true',
        filename=f'{PAPER_FIGURE_DIR}figA01_scatterbox_RMSE_BIAS_vs_HSavg.png',
        dpi=1000,
        legend_kw={
            'bbox_to_anchor':'below_titles',
            'frameon': True},
        showfliers=False)
    
    # Boxplots of RMSE and MAAPE (include maybe)
    cv_evaluate_scores.evaluation_boxplot(
        METHODS,
        ['RMSE', 'BIAS'],
        f'{PAPER_FIGURE_DIR}figA02_boxplot_RMSE_BIAS.png',
        dpi=700,
        showfliers=False)
    
    
    # CLIMATE METRICS / SCORES:
    # -------------------------
    
    # FIGURE 5: Scatterplots of predicted vs true: HSavg, HSmax and dHS1
    cv_evaluate_scores.scatterplot_true_vs_pred(
        METHODS,
        ['HSavg','HSmax','dHS1'],
        f'{PAPER_FIGURE_DIR}fig04_scatter_true_pred_climate_metrics.png',
        dpi=500,
        legend_kw={
            'bbox_to_anchor':'below_titles'},
        equal_xy_axes=True,
        fitlines=True)

    # FIGURE 6: scatterplot climate metrics abs. error vs. HSavg:
    cv_evaluate_scores.scatterboxbins(
        METHODS,
        ['HSavg_abs_diff', 'HSmax_abs_diff', 'dHS1_abs_diff'],
        'HSavg_true',
        filename=f'{PAPER_FIGURE_DIR}fig05_scatterbox_climate_metrics_abs_diff_vs_HSavg.png',
        dpi=1000,
        legend_kw={'bbox_to_anchor':'below_titles'},
        showfliers=True
        )
    
    # not used figure:
    cv_evaluate_scores.scatterboxbins(
        METHODS,
        ['HSavg_diff', 'HSmax_diff', 'dHS1_diff'],
        'HSavg_true',
        filename=f'{PAPER_FIGURE_DIR}fig06_1_scatterbox_climate_metrics_diff_vs_HSavg.png',
        dpi=1000,
        legend_kw={'bbox_to_anchor':'below_titles'},
        showfliers=False
        )


if __name__ =='__main__':
    main()
