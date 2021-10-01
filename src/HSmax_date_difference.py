# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:44:55 2021

@author: aschauer
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cv_results_database import get_cv_results_as_df
import plotting_utils as pu

sns.set_color_codes(palette='deep')


METHODS = [
    'SingleStation_best_correlated_mean_ratio',
    'Inverse distance squared',
    'matiu vertical weighted_min_corr_-1.0',
    'Elastic Net Regression',
    'RandomForest_V3.5',
    'SWE2HS_Snow17_shifted_dates',
    # 'ERA5-land_RF_surrounding_gridcells_max_depth_70_n_estimators_200'
    ]

def calculate_HSmax_date_difference(HS_true_file, HS_pred_file):
    hs_true = pd.read_pickle(HS_true_file)
    hs_pred = pd.read_pickle(HS_pred_file)
    
    true_max_date = hs_true.idxmax()
    pred_max_date = hs_pred.idxmax()
    try:
        datediff = (true_max_date-pred_max_date).days
    except TypeError: 
        # only nans in predicted series will cause idxmax to be 
        # nan and result in TypeError when nan is subtracted from timestamp
        datediff = np.nan
    return datediff

if __name__ == '__main__':
    
    df = get_cv_results_as_df()
    df = df.loc[df['gap_type']=='LOWO']
    df = df.rename(columns={'bias': 'BIAS'})
    df = df.loc[df.fill_method.isin(METHODS), :]
    df['fill_method'] = df['fill_method'].map(pu.METHOD_NAMES)

    df['HSmax_datediff'] = df.apply(lambda x: calculate_HSmax_date_difference(x.HS_true_file, x.HS_pred_file), axis=1)
    
    
    # ax = df['HSmax_datediff'].plot.hist(bins=71)
    # ax.set_xlabel("HSmax date measured - HSmax date predicted")
    # plt.gcf().set_dpi(250)
    # plt.show()
    
    # g = sns.FacetGrid(df, col='fill_method')
    # g.map(plt.hist, 'HSmax_datediff', bins=41)
    # plt.gcf().set_dpi(250)
    # plt.show()
    
    fig, axs = plt.subplots(3,2, figsize=(5, 6), sharex=True, sharey=True)
    
    
    for (method, data) in df.groupby('fill_method', sort=False):
        if method=='BCS':
            ax = axs[0,0]
        if method=='IDW':
            ax = axs[0,1]
        if method=='WNR':
            ax = axs[1,0]
        if method=='ENET':
            ax = axs[1,1]
        if method=='RF':
            ax = axs[2,0]
        if method=='SM':
            ax = axs[2,1]
        
        
        # sns.histplot(data, x='HSmax_datediff', hue='station_grid', ax=ax, bins=21, stat='count', multiple="stack")
        colors={'full': 'tab:orange',
                'only_target_stations': "b"}
        for grid, griddata in data.groupby('station_grid'):
            ax.hist(griddata['HSmax_datediff'], bins=31, color=colors[grid], alpha=0.7, label=pu.GRIDLABELS[grid], edgecolor='grey')
            
        ax.text(0.05, 0.9, method, ha='left', va='top', transform=ax.transAxes)
    plt.tight_layout()
    legend_kw = {}
    top = axs[0,-1].get_position().ymax
    right = axs[0,-1].get_position().xmax
    legend_kw['bbox_to_anchor'] = [right, top+0.01]
    legend_kw['borderaxespad'] = 0
    legend_kw['edgecolor'] = 'black'
    legend_kw['fancybox'] = False
    legend_kw['framealpha'] = 1
    # legend_kw['bbox_transform'] = fig.transFigure
    legend_kw['loc'] = 4
    legend_kw['ncol'] = 2
        # 'fontsize': 11,
        # 'frameon': False
    
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, **legend_kw)
    [ax.set_ylabel('# gaps') for ax in axs[:,0]]
    fig.text(0.55, 0., 'HSmax date measured - HSmax date predicted [days]', ha='center', va='center')
    plt.tight_layout()
    fig.savefig('../results/revision/HSmax_date_differences.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    description = df.groupby(['fill_method','station_grid'])['HSmax_datediff'].describe()
    
    # sample = df.sort_values('HSmax_datediff', ascending=False).head(100)
    # for x in sample.itertuples():
    #     hs_true = pd.read_pickle(x.HS_true_file)
    #     hs_pred = pd.read_pickle(x.HS_pred_file)
        
    #     pd.DataFrame({'true':hs_true, 'pred':hs_pred}, index=hs_true.index).plot()
    #     plt.show()