# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:44:24 2021

@author: aschauer
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

import scoring_utils as scu
import cv_results_database as db
import plotting_utils as pu

sns.set_color_codes(palette='deep')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


sc = db.get_cv_results_as_df()

def scatterplot_true_pred_daily(
        methods_used,
        filename=None,
        dpi=300,
        stn=None,
        equal_xy_axes=False):
    
    fig, axes = plt.subplots(2, len(methods_used),
                              figsize=[len(methods_used)*2.3,2*2.55],
                              sharex=False, sharey=False)
    
    #different markers and colors for different station grids:
    markers={'full': "s",
             'only_target_stations': "^"}
    colors={'full': 'tab:orange',
            'only_target_stations': "b"}
    grids = ['full', 'only_target_stations']
    gridlabel = {'full':'dense station network',
                 'only_target_stations':'evaluation stations only'}
    
    for row, station_grid in enumerate(grids):
        color = colors[station_grid]
        marker = markers[station_grid]
        for column, method in enumerate(methods_used):
            if stn is None:
                view = sc.loc[(sc['fill_method']==method) & (sc['station_grid']==station_grid)]
            else:
                view = sc.loc[(sc['fill_method']==method) & (sc['station_grid']==station_grid) & (sc['gap_stn']==stn)]
            
            hs_true_files = view['HS_true_file'].tolist()
            hs_pred_files = view['HS_pred_file'].tolist()
            hs_true = pd.concat([pd.read_pickle(file) for file in hs_true_files],
                                axis=0)
            hs_pred = pd.concat([pd.read_pickle(file) for file in hs_pred_files],
                                axis=0)
                
            # remove nans
            concated_series = pd.concat([hs_true, hs_pred], axis=1).dropna()
            hs_true = concated_series.iloc[:,0]
            hs_pred = concated_series.iloc[:,1]

            # markersize = 0.1 if stn is None else 2
            if stn is None:
                markersize = 0.1
            else:
                if hs_true.max() <= 300:
                    markersize = -0.0133*hs_true.max()+4.1
                else:
                    markersize = 0.1
            
            axes[row, column].scatter(
                hs_true,
                hs_pred,
                s=markersize,
                marker='o',
                facecolor=color,
                lw=0,
                alpha=0.9,
                label=station_grid)
            
            try:
                # linear fit to the scatterplot:
                #obtain m (slope) and b(intercept) of linear regression line
                m, b = np.polyfit(hs_true, hs_pred, 1)
                # new x-vector
                x_fitline = np.linspace(hs_true.min(), hs_true.max())
                #add linear regression line to scatterplot 
                axes[row,column].plot(
                    x_fitline,
                    m*x_fitline+b,
                    linestyle='--',
                    color='k',
                    lw=0.8)
                
                # coefficient of determination
                r2 = r2_score(hs_true, hs_pred)
                rmse = np.sqrt(mean_squared_error(hs_true, hs_pred))
                maape = scu._maape_score(hs_true, hs_pred)
                bias = scu._bias_score(hs_true, hs_pred)
                plt.rcParams.update({
                    "text.usetex": True})
                axes[row,column].text(
                    0.95,
                    0.05,
                    f"$r^2$ = {r2:.2f}\nRMSE = {rmse:.1f}\nBIAS = {bias:.2f}", 
                    ha='right',
                    va='bottom',
                    transform=axes[row,column].transAxes,
                    fontsize=11
                    )
                plt.rcParams.update({
                    "text.usetex": False})
            except TypeError:
                # only nans are in y_pred (for some stations/years for IDS)
                pass

            # y-labels
            if column == 0:
                axes[row, column].set_ylabel(f'modeled [cm] with\n{gridlabel[station_grid]}', fontsize=11)
            else:
                axes[row, column].set_ylabel(None)
                axes[row, column].tick_params(labelleft=False)
            
            # x-labels
            if row == 1:
                axes[row, column].set_xlabel(f'measured [cm]',
                                            fontsize=11)
            else:
                axes[row, column].set_xlabel(None)
                axes[row, column].tick_params(labelbottom=False)

            # titles
            if row == 0:
                axes[row, column].set_title(pu.METHOD_NAMES[method], fontsize=13)
    
    ygmin = 0.; ygmax = 0.
    xgmin = 0.; xgmax = 0.
    for ax in axes.flatten():
        #Get global minimum and maximum y values accross all axis
        ymin, ymax = ax.get_ylim()
        ygmin = min(ygmin,ymin)
        ygmax = max(ygmax,ymax)
        xmin, xmax = ax.get_xlim()
        xgmin = min(xgmin,xmin)
        xgmax = max(xgmax,xmax)
    [ax.set_ylim((ygmin,ygmax)) for ax in axes.flatten()]
    [ax.set_xlim((xgmin,xgmax)) for ax in axes.flatten()]
    
    if equal_xy_axes:
        gmin = min(xgmin,ygmin)
        gmax = max(xgmax,ygmax)
        for ax in axes.flatten():
            ax.set_ylim((gmin,gmax))
            ax.set_xlim((gmin,gmax))
            ax.set_aspect(1, adjustable='box')
    
    # draw x=y line:
    for ax in axes.flatten():
        ax.axline([0, 0], [1, 1], linestyle='-', color='k', lw=0.8)

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return None

def main(methods_used):
    
    plot_output = '../results/cross_validation/scatter_true_vs_predicted_daily_values/'
    if not os.path.isdir(plot_output):
        os.makedirs(plot_output)
    logger.info(("generate true vs predicted scatterplots of daily values"
                 f"and save to\n{os.path.abspath(plot_output)}"))
    
    scatterplot_true_pred_daily(
        methods_used=methods_used,
        filename=f'{plot_output}all_stations_scatterplots_true_vs_pred_daily_values.png',
        equal_xy_axes=True
        )
    
    for stn in sc['gap_stn'].unique():
        scatterplot_true_pred_daily(
            methods_used=methods_used,
            filename=f'{plot_output}{stn}_scatterplots_true_vs_pred_daily_values.png',
            equal_xy_axes=True,
            stn=stn)
