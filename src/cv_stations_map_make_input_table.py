# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:28:11 2021

@author: aschauer

Map of cv stations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

CV_INPUT_DATA = pd.read_csv(
    '../input_data/data_for_cross_validation_non_homogenized.csv',
    parse_dates = ['time'],
    index_col='time')

# Stations where the gaps will be simulated.
with open('../input_data/cv_evaluation_stations_non_homogenized.json', mode='r') as f:
        EVAL_STATIONS = [stn.replace('*','') for stn in json.load(f)]

PREDICTOR_STATIONS = [x for x in CV_INPUT_DATA['stn'].unique().tolist() if x not in EVAL_STATIONS]

df = CV_INPUT_DATA.groupby('stn').mean().loc[:,['X','Y','Z']]
df['type'] = 'reference station'
df.loc[df.index.isin(EVAL_STATIONS), 'type'] = 'evaluation station'
df.to_csv('H:/gap_filling/paper_gap_filling/map_of_cv_stations/input_coordinates.csv')
