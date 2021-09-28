# -*- coding: utf-8 -*-
"""
Extract closest ERA5land gridpoint for each evaluation station.

@author: Johannes Aschauer
"""

import json
import xarray as xr
import pandas as pd

from gap_modeling import GapModel, _select_winter_months


# Functions for conversion of LV03 to WGS84:
# snippet from: https://github.com/ValentinMinder/Swisstopo-WGS84-LV03
#  The MIT License (MIT)
#  
#  Copyright (c) 2014 Federal Office of Topography swisstopo, Wabern, CH and Aaron Schmocker 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
def _CHtoWGSlat(x, y):
    # Ayiliarx values (% Bern)
    x_auy = (x - 600000) / 1000000
    y_auy = (y - 200000) / 1000000
    lat = (16.9023892 + (3.238272 * y_auy)) + \
            - (0.270978 * pow(x_auy, 2)) + \
            - (0.002528 * pow(y_auy, 2)) + \
            - (0.0447 * pow(x_auy, 2) * y_auy) + \
            - (0.0140 * pow(y_auy, 3))
    # Unit 10000" to 1" and convert seconds to degrees (dec)
    lat = (lat * 100) / 36
    return lat


def _CHtoWGSlon(x, y):
    # Ayiliarx values (% Bern)
    x_auy = (x - 600000) / 1000000
    y_auy = (y - 200000) / 1000000
    lng = (2.6779094 + (4.728982 * x_auy) + \
            + (0.791484 * x_auy * y_auy) + \
            + (0.1306 * x_auy * pow(y_auy, 2))) + \
            - (0.0436 * pow(x_auy, 3))
    # Unit 10000" to 1" and convert seconds to degrees (dec)
    lng = (lng * 100) / 36
    return lng


class ERA5landSingleGridcellFilling(GapModel):
    """
    Use snow dpth of the closest gridcell from ERA5-land to fill missing data.

    Parameters
    ----------
    era5_data : xr.Dataset
        xarray dataset containing ERA5-land data.
    scaling : str, optional
        bias correction applied to the reference ERA5 gridcell:
            - no_scaling (default): directly use data from reference gridcell.
            - mean_ratio: scale by the mean ration of the refernece gridcell 
              to the target station in the train period.

    """
    
    def __init__(self,
                 era5_data,
                 scaling='no_scaling'):

        assert scaling in {'no_scaling', 'mean_ratio'}
        
        self.method = f'ERA5-land_{scaling}'
        self.scaling = scaling
        self.era5_data = era5_data


    def train_gap_split(self,
                        cv_input_data,
                        train_period,
                        gap_period,
                        gap_station):
        
        # get coordinates from station (in LV03)
        X = cv_input_data.loc[cv_input_data['stn']==gap_station,'X'].mean()
        Y = cv_input_data.loc[cv_input_data['stn']==gap_station,'Y'].mean()
        
        # exctract data from reference ERA5 gridcell
        station_gridcell = self.era5_data.sel(
            latitude=_CHtoWGSlat(X, Y),
            longitude=_CHtoWGSlon(X, Y),
            method='nearest')['sde'].to_dataframe()
        
        # take mean from 7 and 8 am from ERA5
        station_gridcell = (station_gridcell
                        .between_time('5:00', '6:00', include_start=True, include_end=True) #measurements between 7 and 8 am local swiss time
                        .resample("D")
                        .mean()
                        )
        station_gridcell = station_gridcell['sde'] /0.01 #conversion to cm
        
        station_data = cv_input_data.loc[cv_input_data['stn']==gap_station, 'HS']
        
        #merge era5 and measured station data to one df
        data = pd.concat([station_data, station_gridcell], axis=1)
        
        # split in gap and train period:
        train_data = (data
            .loc[train_period,:]
            .pipe(_select_winter_months)
            )
        gap_data = (data
            .loc[gap_period,:]
            .pipe(_select_winter_months)
            )
        
        y_train = train_data.loc[:, 'HS']
        X_train = train_data.loc[:, 'sde']
        y_gap = gap_data.loc[:, 'HS']
        X_gap = gap_data.loc[:, 'sde']
        
        return y_train, X_train, y_gap, X_gap
    
    
    def fit_predict(self,
                    data,
                    train_period,
                    gap_period,
                    gap_station):

        y_train, X_train, y_gap, X_gap = self.train_gap_split(cv_input_data=data,
                                                              train_period=train_period,
                                                              gap_period=gap_period,
                                                              gap_station=gap_station)

        if self.scaling == 'no_scaling':
            y_pred = pd.Series(X_gap.values,
                               index=X_gap.index,
                               name=self.method)

        if self.scaling == 'mean_ratio':
            ratio = y_train.mean() / X_train.mean()
            y_pred = pd.Series(X_gap.mul(ratio).values,
                               index=X_gap.index,
                               name=self.method)
        
        return self._postprocess_predictions(y_pred)