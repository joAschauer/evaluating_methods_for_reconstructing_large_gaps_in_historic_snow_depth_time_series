# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:05:40 2020

@author: aschauer
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from collections import OrderedDict
import json

import gap_modeling as gm
import era5land_gap_modeling as era5gm
from cv_split_utils import leave_one_winter_out_splitter
import scoring_utils as scu
from cv_results_database import ModeledGap, make_session

CV_INPUT_DATA = pd.read_csv(
    '../input_data/data_for_cross_validation_non_homogenized.csv',
    parse_dates = ['time'],
    index_col='time')

ERA5_DATA = xr.open_dataset("../input_data/era5land/era5-land_snowdepth_switzerland_05_06_07_UTC.nc")

with open('../input_data/cv_evaluation_stations_non_homogenized.json', mode='r') as f:
        EVAL_STATIONS = [stn.replace('*','') for stn in json.load(f)]

if __name__ == '__main__':
    
    session = make_session()

    MODELING = True
    SCORING = True

    # Station grids with different station densities as dict of different
    # DataFrames.
    stations_grids = OrderedDict([
        ('full', CV_INPUT_DATA),
        ('only_target_stations', (CV_INPUT_DATA
                                  .loc[CV_INPUT_DATA['stn'].isin(EVAL_STATIONS),:]))
        ])
    
    gap_generators = OrderedDict([('LOWO', leave_one_winter_out_splitter)])

    models = OrderedDict([
        # ('single_station_best_correlated',
        #   gm.SingleStationFilling(
        #       distance_metric='best_correlated',
        #       scaling='no_scaling')
        #    ),
        # ('single_station_best_correlated_mean_scaled',
        #   gm.SingleStationFilling(
        #       distance_metric='best_correlated',
        #       scaling='mean_ratio')
        #    ),
        # ('elastic_net_regression',
        #   gm.ElasticNetRegressionFilling(
        #       n_predictor_stations=15)
        #   ),
        # ('GIDS',
        #   gm.GidsFilling(
        #       n_predictor_stations=10)
        #     ),
        # ('inverse_disance',
        #   gm.InverseDistanceSquaredFilling(
        #       n_predictor_stations=3)
        #   ),
        # ('SWE2HS_SLFTI',
        #   gm.SWE2HSSLFTI()
        #   ),
        # ('Snow17_SWE2HS',
        #   gm.SWE2HSSnow17(shifted_dates=False,
        #                   n_jobs=-1),
        #   ),
        # ('Snow17_SWE2HS_shifted_dates',
        #   gm.SWE2HSSnow17(shifted_dates=True,
        #                   n_jobs=-1),
        #   ),
        ('ERA5land_mean_ratio_scaled',
          era5gm.ERA5landSingleGridcellFilling(
              era5_data=ERA5_DATA,
              scaling='mean_ratio'),
          ),
        ('ERA5land_no_scaling',
          era5gm.ERA5landSingleGridcellFilling(
              era5_data=ERA5_DATA,
              scaling='no_scaling'),
         ),
        # # ('random_forest_V3',
        # #  gm.RandomForestFilling3(
        # #      grid_search=False)
        # #  ),
        # ('random_forest_V3_5',
        #   gm.RandomForestFilling3_5(
        #       grid_search=False,
        #       n_predictor_stations=10)
        #   ),
        # ('matiu',
        #   gm.MatiuFilling(
        #       weighting='vertical',
        #       n_predictor_stations=5)
        #   ),
        # ('matiu_no_correlation_constraint',
        #   gm.MatiuFilling(
        #       weighting='vertical',
        #       n_predictor_stations=3,
        #       minimum_correlation=-1
        #       )
        #   )
        ])

    for station in EVAL_STATIONS:
        print(f'Do stuff for station {station}...')
        for station_grid, input_data_grid in stations_grids.items():
            print(f'   ...{station_grid} station grid')

            for gap_type in gap_generators.keys():
                print(f'      ...{gap_type} gap type')

                for train_period, gap_period in gap_generators[gap_type](first_year=2000,
                                                                         last_year=2020):

                    for model in models.values():
                        # SNOWGRID run only until 2019-12-31:
                        if model.method == 'SNOWGRID_CL' and gap_period[-1].year==2020:
                            continue
                        # BUF missing meteo in 10/2010 and 11/2010:
                        elif station=='BUF' and gap_period[-1].year==2011 and model.method in gm.models_that_use_meteo:
                            continue
                        # BUF two weeks missing precip in 09/2017 (only affects temp index model from Tobias):
                        elif station=='BUF' and gap_period[-1].year==2018 and model.method=='SWE2HS_SLFTI':
                            continue
                        # CHD missing meteo in 11/2016:
                        elif station=='CHD' and gap_period[-1].year==2017 and model.method in gm.models_that_use_meteo:
                            continue
                        # GRA missing meteo in Winter 2005/2006:
                        elif station=='GRA' and gap_period[-1].year==2006 and model.method in gm.models_that_use_meteo:
                            continue
                        else:
                            # try to query an entry from the database
                            gap_result = (session.query(ModeledGap)
                                          .filter_by(gap_stn=station,
                                                     fill_method=model.method,
                                                     station_grid=station_grid,
                                                     gap_type=gap_type,
                                                     gap_winter=gap_period[-1].year,
                                                     gap_start=gap_period[0].strftime('%d-%m-%Y'),
                                                     gap_end=gap_period[-1].strftime('%d-%m-%Y'),
                                                     train_start=train_period[0].strftime('%d-%m-%Y'),
                                                     train_end=train_period[-1].strftime('%d-%m-%Y'))
                                          .first())
                            
                            if gap_result is None: # no entry in database
                                # Create new entry in database
                                new_entry = True
                                gap_result = ModeledGap(gap_stn=station,
                                                        fill_method=model.method,
                                                        station_grid=station_grid,
                                                        gap_type=gap_type,
                                                        gap_winter=gap_period[-1].year,
                                                        gap_start=gap_period[0].strftime('%d-%m-%Y'),
                                                        gap_end=gap_period[-1].strftime('%d-%m-%Y'),
                                                        train_start=train_period[0].strftime('%d-%m-%Y'),
                                                        train_end=train_period[-1].strftime('%d-%m-%Y'))
                            
                                gap_result.create_file_references()
                                
                                setattr(gap_result,
                                        'gap_stn_altitude',
                                        scu.get_station_altitude(
                                            CV_INPUT_DATA,
                                            gap_period,
                                            station)
                                        )
                            else:
                                new_entry = False
                                
                            
    
                            if MODELING:
                                # full station grid always has to be modeled first!
                                if 'SWE2HS' in model.method and station_grid!='full':
                                    # get already modeled y_pred from full grid.
                                    # query file reference:
                                    gap_full_grid = (session.query(ModeledGap)
                                                     .filter_by(gap_stn=station,
                                                                fill_method=model.method,
                                                                station_grid='full',
                                                                gap_type=gap_type,
                                                                gap_winter=gap_period[-1].year,
                                                                gap_start=gap_period[0].strftime('%d-%m-%Y'),
                                                                gap_end=gap_period[-1].strftime('%d-%m-%Y'),
                                                                train_start=train_period[0].strftime('%d-%m-%Y'),
                                                                train_end=train_period[-1].strftime('%d-%m-%Y'))
                                                     .first())
                                    y_pred = pd.read_pickle(gap_full_grid.HS_pred_file)

                                else:
                                    if model.method in gm.models_that_use_meteo:
                                        if station=='BUF': # 10,11/2010 missing
                                            train_period = train_period.drop(pd.date_range('2010-09-01','2011-08-31'))
                                        if station=='CHD': # 11/2016 missing
                                            train_period = train_period.drop(pd.date_range('2016-09-01','2017-08-31'))
                                        if station=='GRA': #winter 2006 missing
                                            train_period = train_period.drop(pd.date_range('2005-09-01','2006-08-31'))

                                    y_pred = model.fit_predict(
                                        data=input_data_grid,
                                        train_period=train_period,
                                        gap_period=gap_period,
                                        gap_station=station)

                            else:
                                y_pred = pd.read_pickle(gap_result.HS_pred_file)
                            
                            y_true = scu.get_gap_HS_data(CV_INPUT_DATA,
                                                         gap_period,
                                                         station)
                            
                            if SCORING:
                                # climate indices:
                                setattr(gap_result, 'HSavg_true', scu.HSavg(y_true))
                                setattr(gap_result, 'HSavg_pred', scu.HSavg(y_pred))
                                setattr(gap_result, 'dHS1_true', scu.dHS1(y_true))
                                setattr(gap_result, 'dHS1_pred', scu.dHS1(y_pred))
                                setattr(gap_result, 'HSmax_true', scu.HSmax(y_true))
                                setattr(gap_result, 'HSmax_pred', scu.HSmax(y_pred))

                                # daily metrics:
                                for metric in ['RMSE',
                                               'RMSE_nonzero',
                                               'RMSE_nonzero_true',
                                               'RMSE_nonzero_pred',
                                               'MAAPE',
                                               'MAAPE_nonzero',
                                               'MAAPE_nonzero_true',
                                               'MAAPE_nonzero_pred',
                                               'bias',
                                               'r2_score',
                                               'r2_score_nonzero',
                                               'r2_score_nonzero_true',
                                               'r2_score_nonzero_pred'
                                               ]:

                                    setattr(gap_result,
                                            metric,
                                            scu.get_daily_score_value(
                                                y_true,
                                                y_pred, 
                                                metric)
                                            )

                                # climate metrics:
                                for metric in ['HSavg_diff',
                                               'HSavg_abs_diff',
                                               'HSavg_relative_diff',
                                               'HSavg_relative_abs_diff',
                                               'dHS1_diff',
                                               'dHS1_abs_diff',
                                               'dHS1_relative_diff',
                                               'dHS1_relative_abs_diff',
                                               'HSmax_diff',
                                               'HSmax_abs_diff',
                                               'HSmax_relative_diff',
                                               'HSmax_relative_abs_diff']:
                                    setattr(gap_result,
                                            metric,
                                            scu.get_climate_score_value(
                                                y_true,
                                                y_pred,
                                                metric))
                                # print('end score calculations')
                                y_true.to_pickle(gap_result.HS_true_file)
                                y_pred.to_pickle(gap_result.HS_pred_file)

                            if new_entry:
                                # add the new entry to the database:
                                session.add(gap_result)

                            # in all cases commit changes to the database:
                            session.commit()

