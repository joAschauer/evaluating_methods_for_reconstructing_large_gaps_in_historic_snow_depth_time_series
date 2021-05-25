# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 08:32:38 2020

@author: aschauer
"""
import os
import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import cv_results_database as db
import plotting_utils as pu

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def fill_plot(plot_data, 
              ax, 
              stn, 
              grid, 
              metric,
              methods_used='all'):
    
    if methods_used != 'all':
        if 'Measured' not in methods_used:
            methods_used = ['Measured', *methods_used] # insert at first position
        plot_data = plot_data.loc[:, methods_used]

    plot_data = plot_data.rename(columns=pu.METHOD_NAMES,
                                 errors='ignore')
    # Plotting.
    plot_data.iloc[:,0].plot(ax=ax, 
                             lw=3, 
                             legend=False, 
                             zorder=10,
                             color='black',
                             linestyle='-')
    
    plot_data.iloc[:,1:].plot(ax=ax,
                              legend=False
                              )
    
    ax.text(0.02, 1.05, f"{metric}: {stn}, {grid.replace('_',' ')} grid",
            fontsize =15,
            ha='left', va='center', 
            transform=ax.transAxes)
    
    handles, labels = ax.get_legend_handles_labels()
    l = ax.legend(handles, labels,
                  bbox_to_anchor=(1.01, 1), loc='upper left',
                  borderaxespad=0.)
    
        
    ax.set_ylabel(metric)
    ax.set_xlabel('Hydrological Winter')
    
    loc = ticker.MultipleLocator(base=1.0)
    ax.xaxis.set_major_locator(loc)
    plt.tight_layout()
    return None


def main(methods_used='all'):
    
    figdir = '../results/cross_validation/climate_indices_timeseries_plots'
    
    logger.info(("generate climate indices timeseries from cross validation"
                 f"and save to\n{os.path.abspath(figdir)}"))
    df = db.get_cv_results_as_df()

    fig, ax = plt.subplots(1,1,figsize=[12,6])
    
    for stn, stn_df in df.groupby('gap_stn',
                                  as_index=False,
                                  sort=False):
        for grid, grid_df in stn_df.groupby('station_grid',
                                            as_index=False,
                                            sort=False):
            
            for metric in ['HSavg', 'dHS1', 'HSmax']:
                true = grid_df.groupby('gap_winter').first()[f'{metric}_true']
                true.name = 'Measured'
                pred = grid_df.pivot(index='gap_winter',
                                     columns='fill_method',
                                     values=f'{metric}_pred')
                
                plot_data = pd.concat([true, pred], axis=1)
                
                fill_plot(plot_data, ax, stn, grid, metric, methods_used=methods_used)
                
                    
                fig_filename = f'{figdir}/{stn}_{metric}_timeseries_{grid}_grid.png'
                fig.savefig(fig_filename,
                            bbox_inches="tight")
                plt.close()
                ax.clear()
    return None

if __name__ =='__main__':
    
    main()








