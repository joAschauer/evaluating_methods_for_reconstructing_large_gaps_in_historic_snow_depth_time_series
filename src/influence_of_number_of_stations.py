# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:19:19 2021

@author: aschauer
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict
import json
import socket

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float

import gap_modeling as gm
from cv_split_utils import leave_one_winter_out_splitter
import scoring_utils as scu

CV_INPUT_DATA = pd.read_csv(
    '../input_data/data_for_cross_validation_non_homogenized.csv',
    parse_dates = ['time'],
    index_col='time')

with open('../input_data/cv_evaluation_stations_non_homogenized.json', mode='r') as f:
        EVAL_STATIONS = [stn.replace('*','') for stn in json.load(f)]

# Database setup:

# store locally if on my machine (database loses connection to H: drive)
if socket.gethostname() == 'SLFPC954':
    CV_RESULTS_DIR = Path(r'D:\\HS_gap_filling_paper\results\influence_of_station_number')
    CV_MODELED_SERIES_DIR = CV_RESULTS_DIR / 'modeled_series'
else:
    PROJECT_DIR = Path(__file__).parent.parent
    CV_RESULTS_DIR = PROJECT_DIR / 'results' / 'influence_of_station_number'
    CV_MODELED_SERIES_DIR = CV_RESULTS_DIR / 'modeled_series'

for d in [CV_RESULTS_DIR, CV_MODELED_SERIES_DIR]:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)

DBFILE = CV_RESULTS_DIR / 'database.db'

Base = declarative_base()
    
class ModeledGap(Base):
    __tablename__ = "modeled_gaps_from_different_station_numbers"
    gap_stn = Column(String, primary_key=True)
    fill_method = Column(String, primary_key=True)
    station_grid = Column(String, primary_key=True)
    gap_type = Column(String, primary_key=True)
    gap_winter = Column(Integer, primary_key=True)
    gap_start = Column(String, primary_key=True)
    gap_end = Column(String, primary_key=True)
    train_start = Column(String, primary_key=True)
    train_end = Column(String, primary_key=True)
    max_station_number = Column(Integer, primary_key=True)
    gap_stn_altitude = Column(Integer)
    stations_used = Column(String)
    number_stations_used = Column(Integer)
    HS_true_file = Column(String) # file reference to pickled Series
    HS_pred_file = Column(String) # file reference to pickled Series
    HSavg_true = Column(Float)
    HSavg_pred = Column(Float)
    dHS1_true = Column(Float)
    dHS1_pred = Column(Float)
    HSmax_true = Column(Float)
    HSmax_pred = Column(Float)
    RMSE = Column(Float)
    RMSE_nonzero = Column(Float)
    RMSE_nonzero_true = Column(Float)
    RMSE_nonzero_pred = Column(Float)
    MAAPE = Column(Float)
    MAAPE_nonzero = Column(Float)
    MAAPE_nonzero_true = Column(Float)
    MAAPE_nonzero_pred = Column(Float)
    bias = Column(Float)
    HSavg_diff = Column(Float)
    HSavg_abs_diff = Column(Float)
    HSavg_relative_diff = Column(Float)
    HSavg_relative_abs_diff = Column(Float)
    dHS1_diff = Column(Float)
    dHS1_abs_diff = Column(Float)
    dHS1_relative_diff = Column(Float)
    dHS1_relative_abs_diff = Column(Float)
    HSmax_diff = Column(Float)
    HSmax_abs_diff = Column(Float)
    HSmax_relative_diff = Column(Float)
    HSmax_relative_abs_diff = Column(Float)
    
    def create_file_references(self):
        gap_base = f"{self.gap_stn}_{self.gap_type}_{self.gap_start}-{self.gap_end}"
        model_base = f"{self.fill_method}_{self.station_grid}_{self.max_station_number}_{self.train_start}-{self.train_end}"
        setattr(self, 'HS_true_file',
                str(CV_MODELED_SERIES_DIR / f"_y_true_{gap_base}.pkl"))
        setattr(self, 'HS_pred_file',
                str(CV_MODELED_SERIES_DIR / f"_y_pred_{gap_base}_{model_base}.pkl"))

engine = sa.create_engine(f'sqlite:///{DBFILE}', echo=False)
Base.metadata.create_all(engine)

def make_session():
    session_factory = sa.orm.sessionmaker()
    session_factory.configure(bind=engine)
    session = session_factory()
    return session

def get_results():
    return pd.read_sql('modeled_gaps_from_different_station_numbers', engine)

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
    
    for max_station_number in [1, 3, 5, 10, 15, 20, 25]:

        models = OrderedDict([
            ('elastic_net_regression', 
              gm.ElasticNetRegressionFilling(
                  n_predictor_stations=max_station_number)
              ),
            ('inverse_disance',
              gm.InverseDistanceSquaredFilling(
                  n_predictor_stations=max_station_number)
              ),
            ('random_forest_V3_5',
              gm.RandomForestFilling3_5(
                  n_predictor_stations=max_station_number,
                  grid_search=False)
              ),
            ('matiu_no_correlation_constraint',
              gm.MatiuFilling(
                  n_predictor_stations=max_station_number,
                  weighting='vertical',
                  minimum_correlation=-1))
            ])
    
        for station in EVAL_STATIONS:
            
            for station_grid, input_data_grid in stations_grids.items():
                
    
                for gap_type in gap_generators.keys():
                    print(f'Do stuff for station {station}...')
                    print(f'max station number = {max_station_number}')
                    print(f'   ...{station_grid} station grid')
                    print(f'      ...{gap_type} gap type')
    
                    for train_period, gap_period in gap_generators[gap_type](first_year=2000,
                                                                             last_year=2020):
    
                        for model in models.values():
                            # SNOWGRID run only until 2019-12-31:
                            if model.method == 'SNOWGRID_CL' and gap_period[-1].year==2020:
                                continue
                            # BUF missing meteo in 10/2020 and 11/2010:
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
                                                         train_end=train_period[-1].strftime('%d-%m-%Y'),
                                                         max_station_number=max_station_number)
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
                                                            train_end=train_period[-1].strftime('%d-%m-%Y'),
                                                            max_station_number=max_station_number)
                                
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
                                    
                                    y_pred = model.fit_predict(
                                        data=input_data_grid,
                                        train_period=train_period,
                                        gap_period=gap_period,
                                        gap_station=station)
                                    
                                    stations_used = model.get_used_predictor_stations(
                                        data=input_data_grid,
                                        train_period=train_period,
                                        gap_period=gap_period,
                                        gap_station=station)
                                    setattr(gap_result,
                                            'stations_used',
                                            json.dumps(stations_used))
                                    setattr(gap_result,
                                            'number_stations_used',
                                            len(stations_used))
                                    
    
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
                                                   # 'RMSE_nonzero',
                                                   # 'RMSE_nonzero_true',
                                                   # 'RMSE_nonzero_pred',
                                                   'MAAPE',
                                                   # 'MAAPE_nonzero',
                                                   # 'MAAPE_nonzero_true',
                                                   # 'MAAPE_nonzero_pred',
                                                   'bias']:
    
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
                                                    'HSmax_relative_abs_diff'
                                                    ]:
                                        setattr(gap_result,
                                                metric,
                                                scu.get_climate_score_value(
                                                    y_true,
                                                    y_pred,
                                                    metric))
                                        
                                    y_true.to_pickle(gap_result.HS_true_file)
                                    y_pred.to_pickle(gap_result.HS_pred_file)
    
                                if new_entry:
                                    # add the new entry to the database:
                                    session.add(gap_result)
    
                                # in all cases commit changes to the database:
                                session.commit()

