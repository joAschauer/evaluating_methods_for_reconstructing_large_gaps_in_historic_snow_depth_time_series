# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:54:41 2020

@author: aschauer
"""
import socket
import pandas as pd
from pathlib import Path

import sqlalchemy as sa

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float

# store locally if on my machine (database loses connection to H: drive)
if socket.gethostname() == 'SLFPC954':
    CV_RESULTS_DIR = Path(r'D:\\HS_gap_filling_paper\results\cross_validation')
    CV_MODELED_SERIES_DIR = CV_RESULTS_DIR / 'modeled_series'
else:
    PROJECT_DIR = Path(__file__).parent.parent
    CV_RESULTS_DIR = PROJECT_DIR / 'results' / 'cross_validation'
    CV_MODELED_SERIES_DIR = CV_RESULTS_DIR / 'modeled_series'

for d in [CV_RESULTS_DIR, CV_MODELED_SERIES_DIR]:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)

DBFILE = CV_RESULTS_DIR / 'cv_scores_database.db'

Base = declarative_base()
    
class ModeledGap(Base):
    __tablename__ = "modeled_gaps"
    gap_stn = Column(String, primary_key=True)
    fill_method = Column(String, primary_key=True)
    station_grid = Column(String, primary_key=True)
    gap_type = Column(String, primary_key=True)
    gap_winter = Column(Integer, primary_key=True)
    gap_start = Column(String, primary_key=True)
    gap_end = Column(String, primary_key=True)
    train_start = Column(String, primary_key=True)
    train_end = Column(String, primary_key=True)
    gap_stn_altitude = Column(Integer)
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
    r2_score = Column(Float)
    r2_score_nonzero = Column(Float)
    r2_score_nonzero_true = Column(Float)
    r2_score_nonzero_pred = Column(Float)
    
    def create_file_references(self):
        gap_base = f"{self.gap_stn}_{self.gap_type}_{self.gap_start}-{self.gap_end}"
        model_base = f"{self.fill_method}_{self.station_grid}_{self.train_start}-{self.train_end}"
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

def get_cv_results_as_df():
    return pd.read_sql('modeled_gaps', engine)

def get_predictions_from_one_gap_as_df(gap_stn,
                                       gap_winter,
                                       fill_methods=None,
                                       station_grids=None):
    """
    Query predictions and true data from one gap and gap station and concatenate
    result into a single dataframe.

    Parameters
    ----------
    gap_stn : str
    gap_winter : int
    fill_methods : list or tuple, optional
        The default is None and queries all methods.
    station_grids : list or tuple, optional
        The default is None and queris all station grids.

    Returns
    -------
    out_df : TYPE
        DESCRIPTION.

    """
    
    query = f"""select *
            from modeled_gaps
            where gap_winter=?
            and gap_stn=?"""

    res = pd.read_sql(query, engine, params=[gap_winter, gap_stn])
    
    if fill_methods is not None:
        res = res.loc[res.fill_method.isin(fill_methods)]
    if station_grids is not None:
        res = res.loc[res.station_grid.isin(station_grids)]
    
    # station:
    predictions = []
    hs_true_file = res['HS_true_file'].unique().tolist()
    predictions.append(pd.read_pickle(hs_true_file[0]))
    for station_grid, data in res.groupby('station_grid'):
        
        hs_pred_files = data['HS_pred_file'].tolist()
        df = pd.concat([pd.read_pickle(file) for file in hs_pred_files],
                       axis=1)
        df = df.add_suffix(f'_{station_grid}_station_grid')
        predictions.append(df)
    
    out_df = pd.concat(predictions, axis=1)
    out_df = out_df.add_prefix(f'{gap_stn}_')

    return out_df
