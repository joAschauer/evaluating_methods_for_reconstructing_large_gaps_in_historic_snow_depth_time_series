# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:12:22 2021

@author: aschauer
"""
import os
import logging
from collections import defaultdict
from matplotlib.transforms import Affine2D

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import sqlalchemy as sa

from cv_results_database import get_cv_results_as_df
import plotting_utils as pu
import scoring_utils as scu


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sns.set_color_codes(palette='deep')

sc = get_cv_results_as_df()

sc = sc.loc[sc['gap_type']=='LOWO']
sc = sc.rename(columns={'bias': 'BIAS'})

class HandlerTupleHorizontal(HandlerTuple):
    """
    https://stackoverflow.com/a/59068881
    """
    def __init__(self, **kwargs):
        HandlerTuple.__init__(self, **kwargs)
    
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # How many lines are there.
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()
        
        # divide the horizontal space where the lines will go
        # into equal parts based on the number of lines
        # width_x = (width / numlines)
        width_x = width
    
        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)
         
            legline = handler.create_artists(legend, handle,
                                             xdescent,
                                             height,
                                             (2 * i + 1) * width_x,
                                             2 * height,
                                             fontsize, trans)
            leglines.extend(legline)
    
        return leglines


def scatterplot(methods_used,
                metrics_used,
                xaxs_value,
                filename,
                dpi=300,
                no_legend=False,
                legend_kw=None,
                score_df=sc):
    
    fig, axes = plt.subplots(len(metrics_used),len(methods_used),
                             figsize=[len(methods_used)*2,len(metrics_used)*2.1], 
                             sharex=True, sharey=False)
    
    x_labels = {'gap_stn_altitude': 'Altitude [m]',
                'HSavg_true': 'HSavg [cm]',
                'gap_winter': 'gap winter'}
    
    ylabels = {'dHS1_abs_diff': 'dHS1\nabs. error [days]',
               'HSmax_abs_diff': 'HSmax\nabs. error [cm]',
               'HSavg_abs_diff': 'HSavg\nabs. error [cm]',
               'RMSE': 'RMSE',
               'MAAPE': 'MAAPE'}
    
    #different markers and colors for different station grids:
    markers={'full': "s",
             'only_target_stations': "^"}
    colors={'full': 'tab:orange',
            'only_target_stations': "b"}
    
    if legend_kw is None:
            legend_kw = {}
        
    default_legend_kwargs={
        'bbox_to_anchor':[0.99, 0],
        'loc': 1,
        'ncol': 1,
        'bbox_transform':fig.transFigure,
        'fontsize': 11,
        'frameon': False}
        
    for key, value in default_legend_kwargs.items():
        legend_kw.setdefault(key, value)

    for station_grid in ['full','only_target_stations']:
        color = colors[station_grid]
        marker = markers[station_grid]
        for row, metric in enumerate(metrics_used):
            for column, method in enumerate(methods_used):
                score_df.loc[(score_df['fill_method']==method) & (score_df['station_grid']==station_grid)].plot(xaxs_value,
                                                                                              metric, 
                                                                                              kind='scatter', 
                                                                                              ax=axes[row,column],
                                                                                              color=color,
                                                                                              marker=marker,
                                                                                              alpha=0.4,
                                                                                              label=station_grid)

                axes[row,column].get_legend().remove()

                # y_labels
                if column == 0:
                    try:
                        axes[row,column].set_ylabel(ylabels[metric], fontsize=13)
                    except KeyError:
                        axes[row,column].set_ylabel(metric, fontsize=13)
                else:
                    axes[row,column].set_ylabel(None)
                    axes[row,column].tick_params(labelleft=False)

                # x_labels
                if row == len(metrics_used)-1:
                    axes[row,column].set_xlabel(x_labels[xaxs_value],
                                                fontsize=13)

                # titles
                if row == 0:
                    if legend_kw['bbox_to_anchor']=='below_titles':
                        axes[row,column].set_title(f'{pu.METHOD_NAMES[method]}\n', fontsize=13)
                    else:
                        axes[row,column].set_title(pu.METHOD_NAMES[method], fontsize=13)

    # adapt y-lim for both station grids in every row
    for row, metric in enumerate(metrics_used):
        ygmin = 0.; ygmax = 0.
        for ax in axes[row,:]:
            #Get global minimum and maximum y values accross all axis
            ymin, ymax = ax.get_ylim()
            ygmin = min(ygmin,ymin)
            ygmax = max(ygmax,ymax)
        [ax.set_ylim((ygmin,ygmax)) for ax in axes[row,:]]

    plt.tight_layout()
    if no_legend:
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
    else:

        handles, labels = axes[-1,-1].get_legend_handles_labels()

        
        if legend_kw['bbox_to_anchor']=='top_right_axis':
            top = axes.flatten()[0].get_position().ymax
            right = axes.flatten()[-1].get_position().xmax
            legend_kw['bbox_to_anchor'] = [right, top]
            legend_kw['borderaxespad'] = 0
            legend_kw['edgecolor'] = 'black'
            legend_kw['fancybox'] = False
            legend_kw['framealpha'] = 1
        
        if legend_kw['bbox_to_anchor']=='below_titles':
            legend_kw['loc'] = 'upper center'
            legend_kw['bbox_to_anchor'] = (0.515, 0.95)
            legend_kw['borderaxespad'] = 0.
            legend_kw['ncol'] = 2
            legend_kw['fancybox'] = False
            legend_kw['framealpha'] = 0
            legend_kw['columnspacing'] = 2
            legend_kw['handletextpad'] = 0.2

        leg = fig.legend(handles, ['dense station network', 'only evaluation stations'],
                         **legend_kw)

        for l in leg.legendHandles:
            l.set_alpha(1)

        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
    return None

def scatterplot_true_vs_pred(
        methods_used,
        climate_metrics,
        filename=None,
        dpi=300,
        no_legend=False,
        legend_kw=None,
        equal_xy_axes=False,
        fitlines=False,
        score_df=sc
        ):

    fig, axes = plt.subplots(len(climate_metrics),len(methods_used),
                             figsize=[len(methods_used)*2.1,len(climate_metrics)*2.5], 
                             sharex=False, sharey=False)
    
    #different markers and colors for different station grids:
    markers={'full': "s",
             'only_target_stations': "^"}
    colors={'full': 'tab:orange',
            'only_target_stations': "b"}
    units = defaultdict(lambda: '',
                        {'HSavg': ' [cm]',
                         'HSmax': ' [cm]',
                         'dHS1': ' [days]'})
    
    if legend_kw is None:
            legend_kw = {}
    default_legend_kwargs={
        'bbox_to_anchor':[0.99, 0],
        'loc': 1,
        'ncol': 1,
        'bbox_transform':fig.transFigure,
        'fontsize': 11,
        'frameon': False}
    for key, value in default_legend_kwargs.items():
        legend_kw.setdefault(key, value)
    
    score_annotations = {}

    for station_grid in ['full','only_target_stations']:
        color = colors[station_grid]
        marker = markers[station_grid]
        for row, metric in enumerate(climate_metrics):
            for column, method in enumerate(methods_used):
                score_df.loc[(score_df['fill_method']==method) & (score_df['station_grid']==station_grid)].plot(
                    f'{metric}_true',
                    f'{metric}_pred', 
                    kind='scatter', 
                    ax=axes[row,column],
                    color=color,
                    marker=marker,
                    alpha=0.4,
                    label=station_grid)

                axes[row,column].get_legend().remove()
                
                if fitlines:
                    try:
                        linestyles = {'full':'--', 'only_target_stations':':'}
                        plot_data = score_df.loc[(score_df['fill_method']==method) & (score_df['station_grid']==station_grid)].dropna()
                        true = plot_data[f'{metric}_true']
                        pred = plot_data[f'{metric}_pred']
                        # linear fit to the scatterplot:
                        #obtain m (slope) and b(intercept) of linear regression line
                        m, b = np.polyfit(true, pred, 1)
                        # new x-vector
                        x_fitline = np.linspace(true.min(), true.max())
                        #add linear regression line to scatterplot 
                        axes[row,column].plot(
                            x_fitline,
                            m*x_fitline+b,
                            linestyle=linestyles[station_grid],
                            color='k',
                            lw=1)

                        score_annotations[f"{method}{metric}{station_grid}r2"] = r2_score(true, pred)
                        score_annotations[f"{method}{metric}{station_grid}rmse"] = np.sqrt(mean_squared_error(true, pred))
                        score_annotations[f"{method}{metric}{station_grid}bias"] = scu._bias_score(true, pred)
                    except TypeError:
                        # only nans are in y_pred (for some stations/years for IDS)
                        pass

                # y-labels
                if column == 0:
                    axes[row,column].set_ylabel(f'{metric} modeled{units[metric]}', fontsize=11)
                else:
                    axes[row,column].set_ylabel(None)
                    axes[row,column].tick_params(labelleft=False)
                
                axes[row,column].set_xlabel(f'{metric} measured{units[metric]}',
                                            fontsize=11)

                # titles
                if row == 0:
                    axes[row,column].set_title(pu.METHOD_NAMES[method], fontsize=13)
                    if legend_kw['bbox_to_anchor']=='below_titles':
                        axes[row,column].set_title(f'{pu.METHOD_NAMES[method]}\n', fontsize=13)
    
    if fitlines:
        # Annotations with mixed colors: extremely hacky...
        for score_metric in ['rmse','r2','bias']:
            for row, metric in enumerate(climate_metrics):
                    for column, method in enumerate(methods_used):
                        score_printed={
                            'rmse':'RMSE:',
                            'r2':'$r^2$:',
                            'bias':'BIAS:'}
                        float_format = {
                            'rmse':'1f',
                            'r2':'2f',
                            'bias':'2f'}
                        fontheight=0.095
                        y_pos = {'rmse':0.01+fontheight,
                                 'r2':0.01+2*fontheight,
                                 'bias':0.01}
                        score_dense = score_annotations[f'{method}{metric}full{score_metric}']
                        score_sparse = score_annotations[f"{method}{metric}only_target_stations{score_metric}"]
                        plt.rcParams.update({
                                "text.usetex": True})
                        fs = 12.5
    
                        
                        x_pos = {'full':0.80, 'only_target_stations':0.99}
                        widths = {
                            '2f':{'negative':0.20,
                                  'below_ten':0.17,
                                  'above_ten':0.23},
                            '1f':{'negative':0.14,
                                  'below_ten':0.13,
                                  'above_ten':0.17}}
                        above_ten_add = 0.23
                        below_ten_add = 0.17
                        negative_add = 0.20
                        
                        offset=0
                        axes[row,column].text(
                            0.99,
                            y_pos[score_metric],
                            f"{score_sparse:.{float_format[score_metric]}}",
                            ha='right',
                            va='bottom',
                            color=colors['only_target_stations'],
                            fontsize=fs,
                            transform=axes[row,column].transAxes
                            )
                        if score_sparse < 0:
                            offset += widths[float_format[score_metric]]['negative']
                        elif score_sparse < 10:
                            offset += widths[float_format[score_metric]]['below_ten']
                        else:
                            offset += widths[float_format[score_metric]]['above_ten']
                        
                        axes[row,column].text(
                            0.99-offset,
                            y_pos[score_metric],
                            r"$\mid$",
                            ha='right',
                            va='bottom',
                            fontsize=fs,
                            transform=axes[row,column].transAxes
                            )
                        offset += 0.045
                        axes[row,column].text(
                            0.99-offset,
                            y_pos[score_metric],
                            f"{score_dense:.{float_format[score_metric]}}",
                            ha='right',
                            va='bottom',
                            color=colors['full'],
                            fontsize=fs,
                            transform=axes[row,column].transAxes
                            )
                        if score_dense < 0:
                            offset += widths[float_format[score_metric]]['negative']
                        elif score_dense < 10:
                            offset += widths[float_format[score_metric]]['below_ten']
                        else:
                            offset += widths[float_format[score_metric]]['above_ten']
                        offset += 0.005
                        axes[row,column].text(
                            0.99-offset,
                            y_pos[score_metric],
                            score_printed[score_metric],
                            ha='right',
                            va='bottom',
                            fontsize=fs,
                            transform=axes[row,column].transAxes
                            )
                        plt.rcParams.update({
                            "text.usetex": False})
    
    # adapt y-lim for both station grids in every row
    for row, metric in enumerate(climate_metrics):
        ygmin = 0.; ygmax = 0.
        xgmin = 0.; xgmax = 0.
        for ax in axes[row,:]:
            #Get global minimum and maximum y values accross all axis
            ymin, ymax = ax.get_ylim()
            ygmin = min(ygmin,ymin)
            ygmax = max(ygmax,ymax)
            xmin, xmax = ax.get_xlim()
            xgmin = min(xgmin,xmin)
            xgmax = max(xgmax,xmax)
        [ax.set_ylim((ygmin,ygmax)) for ax in axes[row,:]]
    
        if equal_xy_axes:
            gmin = min(xgmin,ygmin)
            gmax = max(xgmax,ygmax)
            for ax in axes[row,:]:
                ax.set_ylim((gmin,gmax))
                ax.set_xlim((gmin,gmax))
                ax.set_aspect(1, adjustable='box')
    
    # draw x=y line:
    for ax in axes.flatten():
        ax.axline([0, 0], [1, 1], color='k',lw=0.9)
            
    plt.tight_layout()
    if no_legend:
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
        else:
            plt.show()
    else:
        handles, labels = axes[-1,-1].get_legend_handles_labels()
        
        if fitlines:
            custom_handles = []
            for station_grid in ['full','only_target_stations']:
                custom_handles.append(mlines.Line2D([], [], ls=linestyles[station_grid], color='k'))
            handles = [(handles[0],custom_handles[0]),(handles[1],custom_handles[1])]
        
        if legend_kw['bbox_to_anchor']=='top_right_axis':
            top = axes.flatten()[0].get_position().ymax
            right = axes.flatten()[-1].get_position().xmax
            legend_kw['bbox_to_anchor'] = [right, top]
            legend_kw['borderaxespad'] = 0
            legend_kw['edgecolor'] = 'black'
            legend_kw['fancybox'] = False
            legend_kw['framealpha'] = 1
        
        if legend_kw['bbox_to_anchor']=='below_titles':
            legend_kw['loc'] = 'upper center'
            legend_kw['bbox_to_anchor'] = (0.515, 0.96)
            legend_kw['borderaxespad'] = 0.
            legend_kw['ncol'] = 2
            legend_kw['fancybox'] = False
            legend_kw['framealpha'] = 0
            legend_kw['columnspacing'] = 2
            legend_kw['handletextpad'] = 0.2
        
        if fitlines:
            legend_kw['handler_map'] = {tuple: HandlerTuple()}
        
        leg = fig.legend(handles, ['dense station network', 'only evaluation stations'],
                         **legend_kw)

        for l in leg.legendHandles:
            l.set_alpha(1)
        
        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
    return None

# Boxplots
def evaluation_boxplot(methods_used,
                       metrics_used,
                       filename,
                       dpi=300,
                       legend_axis=-1,
                       boxstyle='whisker-box',
                       showfliers=False,
                       score_df=sc):
    
    plot_func = {'whisker-box': sns.boxplot,
                 'letter-value': sns.boxenplot,
                 'violin': sns.violinplot}
    
    assert boxstyle in plot_func.keys()
    
    ylabels = {'dHS1_abs_diff': 'dHS1 abs. error [days]',
               'HSmax_abs_diff': 'HSmax abs. error [cm]',
               'HSavg_abs_diff': 'HSavg abs. error [cm]',
               'RMSE': 'RMSE [cm]',
               'MAAPE': 'MAAPE'}
    
    plot_data = score_df.loc[score_df['fill_method'].isin(methods_used)].copy()
    plot_data.replace(to_replace={'fill_method':pu.METHOD_NAMES}, inplace=True)
    fig, axes = plt.subplots(1,len(metrics_used),
                             figsize=[(len(methods_used)*len(metrics_used))*0.7,10*0.65],
                             sharey=False)
    
    if len(metrics_used) == 1:
        axes = np.array([axes])

    for ax, metric in zip(axes.flat, metrics_used):
        plot_func[boxstyle](
            data = plot_data,
            x = 'fill_method', 
            y = metric,
            order=[pu.METHOD_NAMES[m] for m in methods_used],
            hue = 'station_grid',
            hue_order = ['full', 'only_target_stations'],
            palette=['C1', 'C0'],
            # sym='',
            showfliers=showfliers,
            flierprops={'marker':'x'},
            ax=ax)

        try:
            ax.set_ylabel(ylabels[metric])
        except KeyError:
            ax.set_ylabel(metric)

        ax.set_xlabel(None)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation = 55, ha='right')
        ax.grid(axis='y',zorder=-1)
        ax.set_axisbelow(True)
        ax.get_legend().remove()

    handles, labels = axes[legend_axis].get_legend_handles_labels()
    
    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    l = axes[legend_axis].legend(handles, ['dense station network', 'only evaluation stations'],
                                 bbox_to_anchor=(0.99, 0.99),
                                 loc=1,
                                 borderaxespad=0.,
                                 frameon=False)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    
    fig.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return None




def scatter_and_boxplot_subgrid(
        methods_used,
        metrics_used,
        xaxs_value,
        filename,
        dpi=300,
        no_legend=False,
        legend_kw=None,
        score_df=sc):

# methods_used= ['Inverse distance squared',
#         'GIDS',
#         'matiu vertical weighted',
#         'Elastic Net Regression',
#         'RandomForest_V3.5',
#         'SWE2HS_SLFTI']
# metrics_used = ['HSavg_abs_diff', 'HSmax_abs_diff', 'dHS1_abs_diff']
# xaxs_value = 'HSavg_true'
# filename=None
# legend_kw={
#     'bbox_to_anchor':'below_titles',
#     'frameon': True}
# no_legend=False


    # Figure setup
    fig = plt.figure(figsize=[len(methods_used)*2.2,len(metrics_used)*2.4])
    
    outer_gs = gridspec.GridSpec(len(metrics_used), len(methods_used), figure=fig)
    axs = []
    inner_gs = []
    for g in outer_gs:
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=g, width_ratios=[3,1],wspace=0.)
        inner_gs.append(inner)
        inner_axs = [fig.add_subplot(ax) for ax in inner]
        axs.append(inner_axs)
    
    x_labels = {'gap_stn_altitude': 'Altitude [m]',
                'HSavg_true': 'HSavg [cm]',
                'gap_winter': 'gap winter'}
    
    ylabels = {'dHS1_abs_diff': 'dHS1\nabs. error [days]',
                'HSmax_abs_diff': 'HSmax\nabs. error [cm]',
                'HSavg_abs_diff': 'HSavg\nabs. error [cm]',
                'RMSE': 'RMSE',
                'MAAPE': 'MAAPE'}
    
    #different markers and colors for different station grids:
    markers={'full': "s",
              'only_target_stations': "^"}
    colors={'full': 'tab:orange',
            'only_target_stations': "b"}
    linestyles={'full': '-',
               'only_target_stations': "--"}
    
    if legend_kw is None:
            legend_kw = {}
        
    default_legend_kwargs={
        'bbox_to_anchor':[0.99, 0],
        'loc': 1,
        'ncol': 1,
        'bbox_transform':fig.transFigure,
        'fontsize': 11,
        'frameon': False}
        
    for key, value in default_legend_kwargs.items():
        legend_kw.setdefault(key, value)
    
    for station_grid in ['full','only_target_stations']:
        color = colors[station_grid]
        marker = markers[station_grid]
        for row, metric in enumerate(metrics_used):
            for column, method in enumerate(methods_used):
                outer_ax = axs[(row*len(methods_used))+column]
                plot_data = score_df.loc[(score_df['fill_method']==method) & (score_df['station_grid']==station_grid)].copy()
                plot_data.plot(
                    xaxs_value,
                    metric, 
                    kind='scatter', 
                    ax=outer_ax[0],
                    color=color,
                    marker=marker,
                    alpha=0.4,
                    label=station_grid)
                
                # binning based on xaxs_val:
                bins = np.arange(0,140,20)
                labels = np.arange(10,130,20)
                # bins = np.arange(0,130,10)
                # labels = np.arange(5,125,10)
                plot_data['binned_xval'] = pd.cut(plot_data[xaxs_value], bins,labels=labels)
                median_bins = plot_data.groupby('binned_xval').median()
                
                outer_ax[0].plot(
                    median_bins.index,
                    median_bins[metric],
                    color='k',
                    marker=marker,
                    ls=linestyles[station_grid],
                    label=station_grid)
                
                if station_grid == 'only_target_stations':
                    sns.boxplot(
                        data = score_df.loc[(score_df['fill_method']==method)],
                        x = 'fill_method', 
                        y = metric,
                        hue = 'station_grid',
                        hue_order = ['full', 'only_target_stations'],
                        palette=['C1', 'C0'],
                        # sym='',
                        # showfliers=False,
                        flierprops={'marker':'x'},
                        ax=outer_ax[1])
                    for ax in outer_ax:
                        ax.get_legend().remove()
    
                # y_labels
                if column == 0:
                    try:
                        outer_ax[0].set_ylabel(ylabels[metric], fontsize=13)
                    except KeyError:
                        outer_ax[0].set_ylabel(metric, fontsize=13)
                else:
                    outer_ax[0].set_ylabel(None)
                    outer_ax[0].tick_params(labelleft=False)
    
                # x_labels
                if row == len(metrics_used)-1:
                    outer_ax[0].set_xlabel(x_labels[xaxs_value],
                                                fontsize=13)
                else:
                    outer_ax[0].set_xlabel(None)
                    outer_ax[0].tick_params(labelbottom=False)
    
                # titles
                if row == 0:
                    if legend_kw['bbox_to_anchor']=='below_titles':
                        outer_ax[0].set_title(f'{pu.METHOD_NAMES[method]}\n', fontsize=13)
                    else:
                        outer_ax[0].set_title(pu.METHOD_NAMES[method], fontsize=13)
                        
                outer_ax[1].set(xticks=[], yticks=[])
                outer_ax[1].set_ylabel(None)
                outer_ax[1].set_xlabel(None)
                    
    
    
    # adapt y-lim for both station grids in every row
    for row, metric in enumerate(metrics_used):
        ygmin = 0.; ygmax = 0.
        row_axs = []
        for outer_ax in axs[row*len(methods_used):row*len(methods_used)+len(methods_used)]:
            for ax in outer_ax:
                #Get global minimum and maximum y values accross all axis
                ymin, ymax = ax.get_ylim()
                ygmin = min(ygmin,ymin)
                ygmax = max(ygmax,ymax)
                row_axs.append(ax)
        [ax.set_ylim((ygmin,ygmax)) for ax in row_axs]
        # if metric =='r2_score':
        #     [ax.set_ylim((-1.5,1.1)) for ax in row_axs]
    
    plt.tight_layout()
    if no_legend:
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
    else:
    
        handles, labels = axs[0][0].get_legend_handles_labels()
        custom_handles = []
        for station_grid in ['full','only_target_stations']:
            custom_handles.append(mlines.Line2D([], [], 
                                          color=colors[station_grid], 
                                          marker=markers[station_grid],
                                          mfc=colors[station_grid],
                                          mec=colors[station_grid],
                                          ls='')
                                          )
    
        
        if legend_kw['bbox_to_anchor']=='below_titles':
            legend_kw['loc'] = 'upper center'
            if len(metrics_used)==3:
                legend_kw['bbox_to_anchor'] = (0.515, 0.955)
            elif len(metrics_used)==2:
                legend_kw['bbox_to_anchor'] = (0.515, 0.935)
            else:
                legend_kw['bbox_to_anchor'] = (0.515, 0.975)
            legend_kw['borderaxespad'] = 0.
            legend_kw['ncol'] = 2
            legend_kw['fancybox'] = False
            legend_kw['framealpha'] = 0
            legend_kw['columnspacing'] = 4
            legend_kw['handletextpad'] = 1.8
    
        leg = fig.legend([(handles[0],custom_handles[0]),(handles[1],custom_handles[1])],
                         ['dense station network', 'only evaluation stations'],
                         handler_map={tuple: HandlerTupleHorizontal()},
                         **legend_kw, )
    
        for l in leg.legendHandles:
            l.set_alpha(1)
    
        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
    return None

def scatterboxbins(
        methods_used,
        metrics_used,
        xaxs_value,
        filename,
        dpi=300,
        no_legend=False,
        legend_kw=None,
        showfliers=False,
        score_df=sc):
    
    fig, axs = plt.subplots(len(metrics_used),len(methods_used),
                             figsize=[len(methods_used)*2,len(metrics_used)*2.1], 
                             sharex=True, sharey=False)
    
    x_labels = {'gap_stn_altitude': 'Altitude [m]',
                'HSavg_true': 'HSavg [cm]',
                'gap_winter': 'gap winter'}
    
    ylabels = {'dHS1_abs_diff': 'dHS1\nabs. error [days]',
               'HSmax_abs_diff': 'HSmax\nabs. error [cm]',
               'HSavg_abs_diff': 'HSavg\nabs. error [cm]',
               'RMSE': 'RMSE',
               'MAAPE': 'MAAPE'}
    
    #different markers and colors for different station grids:
    markers={'full': "s",
             'only_target_stations': "^"}
    colors={'full': 'tab:orange',
            'only_target_stations': "b"}
    
    if legend_kw is None:
            legend_kw = {}
        
    default_legend_kwargs={
        'bbox_to_anchor':[0.99, 0],
        'loc': 1,
        'ncol': 1,
        'bbox_transform':fig.transFigure,
        'fontsize': 11,
        'frameon': False}
        
    for key, value in default_legend_kwargs.items():
        legend_kw.setdefault(key, value)

    
    for row, metric in enumerate(metrics_used):
        for column, method in enumerate(methods_used):
            plt_data = score_df.loc[(score_df['fill_method']==method)].copy()
            # binning based on xaxs_val:
            if xaxs_value == 'HSavg_true':
                bins = np.arange(0,140,20)
                labels = np.arange(10,130,20)
            if xaxs_value == 'gap_stn_altitude':
                bins = np.linspace(200,2000,5)
                labels = None

            plt_data['binned_xval'] = pd.cut(plt_data[xaxs_value], bins,labels=labels)
            sns.boxplot(
                data = plt_data,
                x = 'binned_xval', 
                y = metric,
                hue = 'station_grid',
                hue_order = ['full', 'only_target_stations'],
                palette=['C1', 'C0'],
                # sym='',
                showfliers=showfliers,
                flierprops={'marker':'d',
                            'markersize':2},
                ax=axs[row,column])
            
            axs[row,column].get_legend().remove()
            axs[row,column].yaxis.grid(True)

            # y_labels
            if column == 0:
                try:
                    axs[row,column].set_ylabel(ylabels[metric], fontsize=13)
                except KeyError:
                    axs[row,column].set_ylabel(metric, fontsize=13)
            else:
                axs[row,column].set_ylabel(None)
                axs[row,column].tick_params(labelleft=False)

            # x_labels
            if row == len(metrics_used)-1:
                axs[row,column].set_xlabel(x_labels[xaxs_value],
                                            fontsize=13)
            else:
                axs[row,column].set_xlabel(None)
                
            # titles
            if row == 0:
                if legend_kw['bbox_to_anchor']=='below_titles':
                    axs[row,column].set_title(f'{pu.METHOD_NAMES[method]}\n', fontsize=13)
                else:
                    axs[row,column].set_title(pu.METHOD_NAMES[method], fontsize=13)

    # adapt y-lim for both station grids in every row
    for row, metric in enumerate(metrics_used):
        ygmin = 0.; ygmax = 0.
        for ax in axs[row,:]:
            #Get global minimum and maximum y values accross all axis
            ymin, ymax = ax.get_ylim()
            ygmin = min(ygmin,ymin)
            ygmax = max(ygmax,ymax)
        [ax.set_ylim((ygmin,ygmax)) for ax in axs[row,:]]
    
    # somehow changing ylim semms to move the grid to the front...
    [ax.set_axisbelow(True) for ax in axs.flatten()]

    plt.tight_layout()
    if no_legend:
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
    else:

        handles, labels = axs[-1,-1].get_legend_handles_labels()

        
        if legend_kw['bbox_to_anchor']=='below_titles':
            legend_kw['loc'] = 'upper center'
            if len(metrics_used)==3:
                legend_kw['bbox_to_anchor'] = (0.515, 0.96)
            elif len(metrics_used)==2:
                legend_kw['bbox_to_anchor'] = (0.515, 0.94)
            else:
                legend_kw['bbox_to_anchor'] = (0.515, 0.975)
            legend_kw['ncol'] = 2
            legend_kw['fancybox'] = False
            legend_kw['framealpha'] = 0
            legend_kw['columnspacing'] = 2

        leg = fig.legend(handles, ['dense station network', 'only evaluation stations'],
                         **legend_kw)

        for l in leg.legendHandles:
            l.set_alpha(1)

        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
    return None

# %% ########### Standard Methods
def main(used_methods):
    
    plot_output = '../results/cross_validation/score_box_and_scatterplots/'
    if not os.path.isdir(plot_output):
        os.makedirs(plot_output)
    logger.info(("generate boxplots and scatterplots for scores and save to\n"
                 f"{os.path.abspath(plot_output)}"))
    # list of methods
    all_methods = sc['fill_method'].unique()
    
    climate_metrics = ['HSavg_diff', 'HSmax_diff', 'dHS1_diff']
    
    climate_metrics_abs_diff = ['HSavg_abs_diff', 'HSmax_abs_diff', 'dHS1_abs_diff']
    
    metrics_used = ['RMSE', 'MAAPE', 'BIAS']
            
    scatterplot(used_methods,
                ['RMSE', 'MAAPE'],
                'HSavg_true',
                f'{plot_output}fig03_scatterplots_HSavg_vs_RMSE_MAAPE_used_methods.png',
                legend_kw={
                    'bbox_to_anchor':'top_right_axis',
                    'frameon': True})
    
    scatterplot(used_methods,
                climate_metrics,
                'HSavg_true',
                f'{plot_output}scatterplots_HSavg_vs_climate_metrics_used_methods.png',
                legend_kw={
                    'bbox_to_anchor':'top_right_axis',
                    'frameon': True})
    
    scatterplot(used_methods,
                ['RMSE', 'MAAPE'],
                'gap_stn_altitude',
                f'{plot_output}scatterplots_stationaltitude_vs_RMSE_MAAPE_used_methods.png')
    
    scatterplot(used_methods,
                metrics_used=climate_metrics,
                xaxs_value='HSavg_true',
                filename=f'{plot_output}scatterplots_HSavg_vs_climate_metrics_used_methods.png')
    
    scatterplot(used_methods,
                metrics_used=climate_metrics,
                xaxs_value='HSavg_true',
                filename=f'{plot_output}scatterplots_stationaltitude_vs_climate_metrics_used_methods.png')
    
    evaluation_boxplot(used_methods,
                        climate_metrics,
                        f'{plot_output}boxplots_climate_metrics_standar_methods.png')
    
    evaluation_boxplot(used_methods,
                        climate_metrics_abs_diff,
                        f'{plot_output}fig06_boxplots_climate_metrics_abs_diff_used_methods.png',
                        legend_axis=-1)
    
    evaluation_boxplot(['Elastic Net Regression',
                        'RandomForest_V3.5',
                        'SWE2HS_SLFTI'],
                        ['HSavg_diff'],
                        f'{plot_output}boxplots_HSavg_diff_BIAS_ela_rand_swe2hs_for_bias_estimation.png')
    
    
    
    evaluation_boxplot(used_methods,
                        ['RMSE', 'MAAPE'],
                        f'{plot_output}fig02_boxplots_RMSE_MAAPE_used_methods.png'
                        )
    
    
    scatterplot(all_methods,
                ['RMSE', 'MAAPE'],
                'HSavg_true',
                f'{plot_output}scatterplots_HSavg_vs_RMSE_MAAPE_all_methods.png')
    
    scatterplot(all_methods,
                ['HSavg_pred','HSavg_true'],
                'HSavg_true',
                f'{plot_output}scatterplots_HSavg_true_vs_HSavg_pred_all_methods.png')
    
    scatterplot_true_vs_pred(
        used_methods,
        ['HSavg','HSmax','dHS1'],
        f'{plot_output}fig05_scatterplots_true_vs_pred_climate_metrics_used_methods.png',
        legend_kw={'bbox_to_anchor':'top_right_axis',
                   'frameon':True})
    
    scatterplot_true_vs_pred(
        all_methods,
        ['HSavg','HSmax','dHS1'],
        f'{plot_output}scatterplots_true_vs_pred_climate_metrics_all_methods.png')
    
    
    
    scatterplot(all_methods,
                metrics_used=climate_metrics,
                xaxs_value='HSavg_true',
                filename=f'{plot_output}scatterplots_HSavg_vs_climate_metrics_all_methods.png')
    
    scatterplot(all_methods,
                metrics_used=['RMSE','MAAPE','BIAS'],
                xaxs_value='gap_winter',
                filename=f'{plot_output}scatterplots_gap_winter_vs_RMSE_MAAPE_BIAS_all_methods.png')
    
    evaluation_boxplot(all_methods,
                        ['RMSE', 'MAAPE', 'BIAS'],
                        filename=f'{plot_output}boxplots_all_methods.png')
    
    evaluation_boxplot(all_methods,
                        climate_metrics,
                        f'{plot_output}boxplots_climate_metrics_all_methods.png')
    
    scatterplot(used_methods,
                ['HSavg_abs_diff', 'HSmax_abs_diff', 'dHS1_abs_diff'],
                'HSavg_true',
                filename=f'{plot_output}scatter_climate_metrics_abs_diff_vs_HSavg.png',
                legend_kw={
                    'bbox_to_anchor':'top_right_axis',
                    'frameon': True})
