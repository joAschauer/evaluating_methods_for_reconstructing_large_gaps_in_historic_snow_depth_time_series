# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:09:19 2021

@author: aschauer
"""
import os
import shutil
import logging

import matplotlib.pyplot as plt
import cv_results_database as db

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# %% Bad predictions

def n_worst_RMSE(scores, n=100, methods_used=None):
    # 100 worst RMSE values
    view = scores.sort_values('RMSE', ascending=False).head(n)
    outdir = '../results/bad_predictions/large_RMSE'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)
    logger.info(f"save plots to {os.path.abspath(outdir)}")
    for n, l in enumerate(view.itertuples()):
        ax = db.get_predictions_from_one_gap_as_df(
            l.gap_stn,
            l.gap_winter,
            fill_methods=methods_used,
            station_grids=[l.station_grid]).plot(figsize=[20,15])
        ax.set_title(f"bad RMSE: {l.fill_method}")
        ax.figure.savefig(f'{outdir}/{str(n).zfill(3)}_{l.gap_stn}_{l.gap_winter}_{l.station_grid}.png')
        plt.close(ax.figure)

# %%
def n_worst_dHS1(scores, n=100, methods_used=None):
    # dHS1 predictions with dHS1=0 (occors for IDS and GIDS, sparse network):
    view = scores.sort_values('dHS1_pred', ascending=True).head(n)
    outdir = '../results/bad_predictions/too_small_dHS1'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)
    logger.info(f"save plots to {os.path.abspath(outdir)}")
    for n, l in enumerate(view.itertuples()):
        ax = db.get_predictions_from_one_gap_as_df(
            l.gap_stn,
            l.gap_winter,
            fill_methods=methods_used,
            station_grids=[l.station_grid]).plot(figsize=[20,15])
        ax.set_title(f"small dHS1: {l.fill_method}")
        ax.figure.savefig(f'{outdir}/{str(n).zfill(3)}_{l.gap_stn}_{l.gap_winter}_{l.station_grid}.png')
        plt.close(ax.figure)

# %%
def n_best_MAAPE(scores, n=100, HSavg_gt=20, methods_used=None):
    """
    very good predictions:
        - HSavg measured > 20
        - ascending MAAPE
    """
    view = (scores
            .loc[scores['HSavg_true']>HSavg_gt, :]
            .sort_values('MAAPE', ascending=True).head(n)
            )
    outdir = '../results/good_predictions/good_MAAPE'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)
    logger.info(f"save plots to {os.path.abspath(outdir)}")
    for n, l in enumerate(view.itertuples()):
        ax = db.get_predictions_from_one_gap_as_df(
            l.gap_stn,
            l.gap_winter,
            fill_methods=methods_used,
            station_grids=[l.station_grid]).plot(figsize=[20,15])
        ax.set_title(f"good MAAPE: {l.fill_method}")
        ax.figure.savefig(f'{outdir}/{str(n).zfill(3)}_{l.gap_stn}_{l.gap_winter}_{l.station_grid}.png')
        plt.close(ax.figure)

# %%
def n_best_RMSE(scores, n=100, HSavg_gt=20, methods_used=None):
    """
    very good predictions:
        - HSavg measured > 20
        - ascending RMSE
    """
    view = (scores
            .loc[scores['HSavg_true']>HSavg_gt, :]
            .sort_values('RMSE', ascending=True).head(n)
            )
    outdir = '../results/good_predictions/good_RMSE'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)
    logger.info(f"save plots to {os.path.abspath(outdir)}")
    for n, l in enumerate(view.itertuples()):
        ax = db.get_predictions_from_one_gap_as_df(
            gap_stn=l.gap_stn,
            gap_winter=l.gap_winter,
            fill_methods=methods_used,
            station_grids=[l.station_grid]).plot(figsize=[20,15])
        
        ax.set_title(f"good RMSE: {l.fill_method}")
        ax.figure.savefig(f'{outdir}/{str(n).zfill(3)}_{l.gap_stn}_{l.gap_winter}_{l.station_grid}.png')
        plt.close(ax.figure)

def main(methods_used=None):
    logger.info("generate plots for good and bad examples...")
    scores = db.get_cv_results_as_df()
    
    if methods_used is not None:
        scores = scores.loc[scores['fill_method'].isin(methods_used)]
    
    n_worst_RMSE(scores,
                 n=100,
                 methods_used=methods_used)
    
    n_worst_dHS1(scores,
                 n=100,
                 methods_used=methods_used)
    
    n_best_MAAPE(scores,
                 n=100,
                 HSavg_gt=20,
                 methods_used=methods_used)
    
    n_best_RMSE(scores,
                n=100,
                HSavg_gt=20,
                methods_used=methods_used)

if __name__ == '__main__':
    standard_methods = ['Inverse distance squared',
                        'GIDS',
                        'matiu vertical weighted',
                        'Elastic Net Regression',
                        'RandomForest_V3.5',
                        'SWE2HS_SLFTI']
    
    main(methods_used=standard_methods)