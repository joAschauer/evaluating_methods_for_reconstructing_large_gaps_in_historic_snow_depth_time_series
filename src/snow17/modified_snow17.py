# -*- coding: utf-8 -*-
"""
Snow 17 with time reset at 1st September.

Created on Wed Apr 22 15:18:52 2020

@author: aschauer
"""
import pandas as pd
from .snow_17 import snow17

def _assign_hydr_year_column(data):
    """
    Assign a new column containing the hydrological year to the data. Here we 
    start the hydrological year with the begining of September

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe where we want to assign the hydrological year to. Has to
        have a pd.DatetimeIndex as
        index

    Returns
    -------
    data : pd.DataFrame
        The modified dataframe with the new column 'hydr_year' containing the 
        hydrological year.
    """

    fall_mask = pd.Series(data.index.month.isin([9,10,11,12]),
                          index=data.index)    
    data['hydr_year'] = data.index.year
    data.loc[fall_mask,'hydr_year'] += 1
    
    return data

    
def hy_batch_snow17(time, precip, tair, shift_dates=False, **kwargs):
    """
    Execute snow17 in hydrological year (September-August) batches.
    
    
    Parameters
    ----------
    time : 1d numpy.ndarray or scalar
        Array of datetime objects.
    prec : 1d numpy.ndarray or scalar
        Array of precipitation forcings, size of `time`.
    tair : 1d numpy.ndarray or scalar
        Array of air temperature forcings, size of `time`.
    shift_dates : bool (default=False)
        Whether to shift dates to comply with measurement scheme of 
        precipitation from the MeteoSwiss.
        
        MeteoSwiss assigns the precipitation from 6am-6am to the date when the 
        measurements start. Temperatures are means from 0am-0am. The snow
        measurements take place at ~7-8am.
        
        When True, the temperature is a weighted mean from the
        date before and the date of HS measurement and the precipitation is 
        shifted by one day.
    **kwargs : keyword arguments
        Keyword arguments passed to the ´snow17´ call

    Returns
    -------
    result : pd.DataFrame
        DateFrame with pd.DatetimeIndex and the columns:
            -'model_swe'
            -'outflow'
            -'model_HN'
            -'model_HS'

    """
    # make a dataframe for yearly grouping:
    meteo_data = pd.DataFrame({'precip':precip,
                               'tair': tair},
                              index=time)

    if shift_dates:
        meteo_data['tair'] = meteo_data['tair'].shift(1, fill_value=0)
        meteo_data['precip'] = meteo_data['precip'].shift(1, fill_value=0)

    years_modeled = []
    # divide in hydro year slices.
    for year, data in (meteo_data.pipe(_assign_hydr_year_column)
                                 .groupby('hydr_year')):
        
        # run snow17 for every hydro year independently.
        model_swe, outflow, model_HN, model_HS = snow17(list(data.index),
                                                        data['precip'],
                                                        data['tair'],
                                                        **kwargs)

        years_modeled.append(pd.DataFrame({'model_swe': model_swe,
                                           'outflow': outflow,
                                           'model_HN': model_HN,
                                           'model_HS': model_HS},
                                          index=data.index))

    # concatenate results of single hydro years.
    result = pd.concat(years_modeled, axis=0)
    return result



