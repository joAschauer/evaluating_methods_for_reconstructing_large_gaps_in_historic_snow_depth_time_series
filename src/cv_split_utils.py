# -*- coding: utf-8 -*-
"""
Daterange period generators for synthetic gaps.

Created on Mon May 25 15:32:24 2020

@author: Johannes Aschauer
"""
import pandas as pd
import numpy as np


def _no_common_member(a, b):
    """
    Validates that there is no common member in the two lists a and b. Returns
    False if there is a common member.

    """
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return False 
    else: 
        return True


def leave_one_winter_out_splitter(first_year=2000,
                                  last_year=2020):

    years = np.arange(first_year,last_year+1)
    dateranges = {year: pd.date_range(start=f'{year-1}-09-01',
                                      end=f'{year}-08-31')
                  for year in years}
    if 2020 in years:
        dateranges[2020] = pd.date_range(start='2019-09-01', end='2020-06-01')

    for gap_year in years:
        gap_dates = dateranges[gap_year]
        train_years = years[np.logical_not(years==gap_year)]
        for i, train_year in enumerate(train_years):
            if i == 0:
                train_dates = dateranges[train_year]
            else:
                train_dates = train_dates.append(dateranges[train_year])
        
        assert _no_common_member(gap_dates, train_dates)
        yield train_dates, gap_dates

