# -*- coding: utf-8 -*-
"""
plotting_utils

Created on Mon Mar 29 15:31:09 2021

@author: aschauer
"""

METHOD_NAMES = {
    'SingleStation_best_correlated': 'BCS',
    'SingleStation_best_correlated_mean_ratio': 'BCS',
    'Elastic Net Regression': 'ENET',
    'GIDS': 'GIDS',
    'Inverse distance squared': 'IDW',
    'Lasso Regression': 'LASSO',
    'Multilinear Regression': 'mulitlinregr',
    'PCA Regression': 'PCR',
    'SNOWGRID_CL': 'snowgrid',
    'SWE2HS_HS2SWE': 'HS2SWE_SWE2HS',
    'SWE2HS_SLFTI': 'SM',
    'Snow17': 'Snow17',
    'Snow17_shifted_dates': 'Snow17_shifted_dates',
    'SWE2HS_Snow17': 'SM:S17',
    'SWE2HS_Snow17_shifted_dates': 'SM',
    'matiu correlation weighted': 'matiu corr',
    'matiu horizontal weighted': 'matiu x_dist',
    'matiu vertical weighted': 'Matiu',
    'matiu vertical weighted_min_corr_-1.0': 'WNR',
    'RandomForest': 'RandomForest',
    'RandomForest_V2': 'RandomForest_V2',
    'RandomForest_V3': 'RandomForest_V3',
    'RandomForest_V3.1': 'RandomForest_V3.1',
    'RandomForest_V3.2': 'RandomForest_V3.2',
    'RandomForest_V3.3': 'RandomForest_V3.3',
    'RandomForest_V3.5': 'RF',
    'RandomForest_V4': 'RandomForest_V4',
    'RandomForest_V4.1': 'RandomForest_V4.1',
    'ERA5-land_mean_ratio':'ERA5mrbc',
    'ERA5-land_no_scaling':'ERA5nobc',
    'ERA5-land_RF_surrounding_gridcells_max_depth_70_n_estimators_200':'ERA5rf'}

GRIDLABELS = {
    'full':'dense station network',
    'only_target_stations':'evaluation stations only'}

