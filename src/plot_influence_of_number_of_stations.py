# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:48:34 2021

@author: aschauer
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import influence_of_number_of_stations
import plotting_utils as pu


SELECTED_METHODS = [
    'IDS',
    'GIDS',
    'Matiu',
    'ENET',
    'RF']

SELECTED_METHODS = [
    'Inverse distance squared',
    'matiu vertical weighted_min_corr_-1.0',
    'Elastic Net Regression',
    'RandomForest_V3.5']

def scatter_nstationsused_yval(
        yval,
        station_grid='only_target_stations',
        only_HSavg_smaller=500,
        only_HSavg_larger=0
        ):
    
    
    df = influence_of_number_of_stations.get_results()
    df['fill_method'] = df['fill_method'].map(pu.METHOD_NAMES)
    df['dHS1_abs_diff'] = df['dHS1_diff'].abs()
    df['HSmax_abs_diff'] = df['HSmax_diff'].abs()
    
    plot_data = df.loc[((df['station_grid']==station_grid) &
                        (df['HSavg_true']<only_HSavg_smaller) &
                        (df['HSavg_true']>only_HSavg_larger)), :]
    
    
    sns.stripplot(data=plot_data,
                  x='number_stations_used',
                  y=yval,
                  hue='fill_method',
                  dodge=True)
    plt.show()
    return None

# EXPLORATION
def exploration_plots(station_grid='full',
                      only_HSavg_smaller=500,
                      only_HSavg_larger=0
                      ):
    """
    xval: 'max_station_number' or 'number_stations_used'
    """
    
    method_order = ['IDS',
                    'GIDS',
                    'Matiu 2021',
                    'Elastic Net',
                    'Random Forest']
    
    markers = ['o','D','s','v','^']
    linestyles = '--'
    
    df = influence_of_number_of_stations.get_results()
    # plot_data = sc.loc[sc['fill_method'].isin(methods_used)].copy()
    # plot_data.replace(to_replace={'fill_method':pu.METHOD_NAMES}, inplace=True)
    df['fill_method'] = df['fill_method'].map(pu.METHOD_NAMES)
    
    
    df['dHS1_abs_diff'] = df['dHS1_diff'].abs()
    df['HSmax_abs_diff'] = df['HSmax_diff'].abs()
    
    xval = 'max_station_number'
    # xval = 'number_stations_used'
    
    # yval = 'RMSE'
    # yval = 'MAAPE'
    # yval = 'HSavg_diff'
    # yval = 'dHS1_abs_diff'
    # yval = 'HSmax_abs_diff'
    # yvals = ['RMSE','MAAPE','HSavg_diff','dHS1_abs_diff','HSmax_abs_diff']
    yvals = ['RMSE','MAAPE']

    plot_data = df.loc[((df['station_grid']==station_grid) &
                        (df['HSavg_true']<only_HSavg_smaller) &
                        (df['HSavg_true']>only_HSavg_larger)), :]
    
    fig, axs = plt.subplots(len(yvals),2, figsize=[15,len(yvals)*5],
                            sharex=True)
    for yval, ax in zip(yvals, axs[:,0]):

        sns.pointplot(
            x=xval,
            y=yval,
            hue='fill_method',
            data=plot_data,
            hue_order=method_order,
            markers=markers,
            linestyles=linestyles,
            estimator=np.median,
            errorbar=(),
            capsize=.05,
            ax=ax
            )
        ax.grid()
    
        
    # axs[0].set_title(f'{station_grid}, {only_HSavg_larger} < HSavg < {only_HSavg_smaller}')
    # plt.show()
    
    # fig, axs = plt.subplots(len(yvals),1, figsize=[10,len(yvals)*5],
    #                         sharex=True)
    for yval, ax in zip(yvals, axs[:,1]):
        sns.boxplot(
            x=xval,
            y=yval,
            hue='fill_method',
            hue_order=method_order,
            data=plot_data,
            ax=ax,
            showfliers=False
            )
        
        ax.grid()
        
    for row, yval in enumerate(yvals):
        ygmin = 0.; ygmax = 0.
        for ax in axs[row,:]:
            #Get global minimum and maximum y values accross all axis
            ymin, ymax = ax.get_ylim()
            ygmin = min(ygmin,ymin)
            ygmax = max(ygmax,ymax)
        [ax.set_ylim((ygmin,ygmax)) for ax in axs[row,:]]
        
    # axs[0].set_title(f'{station_grid}, {only_HSavg_larger} < HSavg < {only_HSavg_smaller}')
    fig.suptitle(f'{station_grid}, {only_HSavg_larger} < HSavg < {only_HSavg_smaller}')
    plt.show()

# %%

def make_paper_figure(filename=None):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=[6,7.5],
                                   sharex=True)
    
    df = influence_of_number_of_stations.get_results()
    df['fill_method'] = df['fill_method'].map(pu.METHOD_NAMES)
    plot_data = df.loc[df['station_grid']=='full', :]
    
    method_order = [
        # 'IDS',
        # 'GIDS',
        'Matiu',
        'MatiuNC',
        # 'ENET',
        # 'RF'
        ]
    
    markers = ['o','D','s','v','^']
    linestyles = '--'
    fontsize = 13
    
    # RMSE
    p1 = sns.pointplot(
        x='max_station_number',
        y='RMSE',
        hue='fill_method',
        data=plot_data,
        hue_order=method_order,
        markers=markers,
        linestyles=linestyles,
        estimator=np.median,
        errorbar=("pi",95),
        capsize=.05,
        ax=ax1,
        legend=None
        )
    
    ax1.set_xlabel('')
    ax1.set_ylabel('RMSE [cm]')
    ax1.get_legend().remove()
    ax1.yaxis.label.set_size(fontsize)
    ax1.tick_params(labelsize=fontsize)
    
    # MAAPE
    p2 = sns.pointplot(
        x='max_station_number',
        y='MAAPE',
        hue='fill_method',
        data=plot_data,
        hue_order=method_order,
        markers=markers,
        linestyles=linestyles,
        estimator=np.median,
        errorbar=("pi",95),
        capsize=.05,
        ax=ax2
        )
    
    p2.legend_.set_title(None)
    ax2.set_xlabel('Maximum number of predictor stations',
                   fontsize=fontsize)
    ax2.yaxis.label.set_size(fontsize)
    ax2.tick_params(labelsize=fontsize)
    fig.tight_layout()
    fig.align_ylabels()
    
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

def describe_data(station_grid='full'):
    df = influence_of_number_of_stations.get_results()
    df['fill_method'] = df['fill_method'].map(pu.METHOD_NAMES)
    df = df.rename(columns={'bias': 'BIAS'})
    data = df.loc[df['station_grid']==station_grid, :]
    data = data.loc[data['fill_method'].isin(['IDS','Matiu','MatiuNC','ENET','RF'])]
    
    description =  data.groupby(['fill_method','max_station_number']).describe()
    rmse = description['RMSE']
    maape = description['MAAPE']
    return rmse, maape
    
def make_paper_figure_boxplots(filename=None,
                               dpi=300,
                               station_grid='full',
                                   methods_used=[
                                    'Inverse distance squared',
                                    'matiu vertical weighted',
                                    'matiu vertical weighted_min_corr_-1.0',
                                    'Elastic Net Regression',
                                    'RandomForest_V3.5']):
    
    assert station_grid in {'full','only_target_stations'}
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=[6.5,9.5],
                                   sharex=True)
    
    df = influence_of_number_of_stations.get_results()
    df['fill_method'] = df['fill_method'].map(pu.METHOD_NAMES)
    df = df.rename(columns={'bias': 'BIAS'})
    plot_data = df.loc[df['station_grid']==station_grid, :]

    fontsize = 13
    
    #RMSE
    p1 = sns.boxplot(
        x='fill_method',
        y='RMSE',
        hue='max_station_number',
        data=plot_data,
        order=[pu.METHOD_NAMES[m] for m in methods_used],
        ax=ax1,
        showfliers=False,
        medianprops={'color':'yellow'}
        )
    
    p1.legend_.set_title('max. number of\npredictor stations')
    ax1.grid(axis='y',zorder=-1)
    ax1.set_xlabel('')
    ax1.set_ylabel('RMSE [cm]')
    
    ax1.yaxis.label.set_size(fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.set_axisbelow(True)
    
    # MAAPE
    p2 = sns.boxplot(
        x='fill_method',
        y='MAAPE',
        hue='max_station_number',
        data=plot_data,
        order=[pu.METHOD_NAMES[m] for m in methods_used],
        ax=ax2,
        showfliers=False,
        medianprops={'color':'yellow'}
        )
    
    
    ax2.grid(axis='y',zorder=-1)
    ax2.set_xlabel(None)
    ax2.get_legend().remove()
    ax2.yaxis.label.set_size(fontsize)
    # ax2.set_xticklabels(ax2.get_xticklabels(), rotation=55, ha='right')
    ax2.tick_params(labelsize=fontsize)
    # ax2.xaxis.tick_params
    ax2.set_axisbelow(True)
    fig.tight_layout()
    fig.align_ylabels()
    
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

if __name__ == '__main__':
    rmse, maape = describe_data()

    make_paper_figure()
    plt.show()
    make_paper_figure_boxplots(station_grid='only_target_stations')
    plt.show()

    # scatter_nstationsused_yval('RMSE')
    # scatter_nstationsused_yval('MAAPE')
    # scatter_nstationsused_yval('HSavg_diff')
    
    # exploration_plots(
    #     station_grid='only_target_stations',
    #     only_HSavg_smaller=500,
    #     only_HSavg_larger=0)
    
    # exploration_plots(
    #     station_grid='only_target_stations',
    #     only_HSavg_smaller=10,
    #     only_HSavg_larger=0)
    
    # exploration_plots(
    #     station_grid='only_target_stations',
    #     only_HSavg_smaller=500,
    #     only_HSavg_larger=30)
    
    # exploration_plots(
    #     station_grid='only_target_stations',
    #     only_HSavg_smaller=500,
    #     only_HSavg_larger=50)
    
    # exploration_plots(
    #     station_grid='full',
    #     only_HSavg_smaller=500,
    #     only_HSavg_larger=0)
    
    # exploration_plots(
    #     station_grid='full',
    #     only_HSavg_smaller=10,
    #     only_HSavg_larger=0)
    
    # exploration_plots(
    #     station_grid='full',
    #     only_HSavg_smaller=500,
    #     only_HSavg_larger=30)
    
    # exploration_plots(
    #     station_grid='full',
    #     only_HSavg_smaller=500,
    #     only_HSavg_larger=50)
