# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:52:12 2020

@author: aschauer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import json

def main(adjust_text=False):
    CV_INPUT_DATA = pd.read_csv(
        '../input_data/data_for_cross_validation_non_homogenized.csv',
        parse_dates = ['time'],
        index_col='time')
    
    # Stations where the gaps will be simulated.
    with open('../input_data/cv_evaluation_stations_non_homogenized.json', mode='r') as f:
            EVAL_STATIONS = [stn.replace('*','') for stn in json.load(f)]
    
    predictor_stations = [x for x in CV_INPUT_DATA['stn'].unique().tolist() if x not in EVAL_STATIONS]
    
    df = CV_INPUT_DATA.groupby('stn').mean().loc[:,'Z'].to_frame()
    
    df['eval_stns'] = df.index.isin(EVAL_STATIONS)
    df['stn_type'] = 'reference stations'
    df.loc[df.index.isin(EVAL_STATIONS), 'stn_type'] = 'evaluation stations'
    
    df = df.sort_values('Z').reset_index()
    df['x1'] = df.index
    
    fig, ax = plt.subplots(figsize=[13,9])
    
    # we need all points in onelist for adjust_text function.
    points = [plt.scatter(l.x1, l.Z, s=5) for l in df.itertuples()]
    
    for name, group in df.groupby('stn_type'):
        
        ax.plot(group.x1, group.Z, marker='o', linestyle='', markersize=5, label=name)
        if name == 'evaluation stations':
            texts = [plt.text(l.x1, l.Z, l.stn) for l in group.itertuples()]
    
    ax.set_ylim([200, 2600])
    ax.set_ylabel('elevation [m a.s.l.]')
    ax.xaxis.set_visible(False)
    plt.legend()
    
    if adjust_text:
        adjust_text(texts,
                    arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
                    add_objects=points,
                    expand_objects=(1.2,1.2),
                    expand_text=(1.1,1.1),
                    expand_points=(1.1,1.1))

    fig.savefig('../results/stations_elevation_distribution.png')

if __name__ == '__main__':
    main()