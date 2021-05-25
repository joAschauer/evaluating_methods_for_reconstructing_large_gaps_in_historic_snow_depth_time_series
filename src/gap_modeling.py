# -*- coding: utf-8 -*-
"""
Methods to perform gap filling in snow depth (HS) time series.





@author: aschauer
"""
import pandas as pd
import numpy as np
import math

import logging

# scikit-learn modules: 
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, KBinsDiscretizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# with sklearn version '0.24.1' HalvingGridSearch is experimental:
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

#yellowbrick (qq-plots)
#from yellowbrick.regressor import ResidualsPlot

# Snow17 and SWE2HS
from snow17.modified_snow17 import hy_batch_snow17
from SWE2HS.density_model_numba import SnowDepthParamEstimator, SnowDepthEstimator

logger = logging.getLogger(__name__)


def _select_winter_months(data):
    """
    Select only data in winer months Nov-Apr. For .pipe() calls.
    """
    assert isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex))
    if isinstance(data, pd.Series):
        winter_data = data.loc[data.index.month.isin([11,12,1,2,3,4])]
    else:
        winter_data = data.loc[data.index.month.isin([11,12,1,2,3,4]),:]
    return winter_data


def _haversine(lat_point1, lon_point1, lat_point2, lon_point2):
    """ 
    Distance on sphere between two lat lon points.
    
    Adapted from python haversine package.
    https://github.com/mapado/haversine/blob/master/haversine/haversine.py
    """
    # mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
    _AVG_EARTH_RADIUS_M = 6371008.8

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = math.radians(lat_point1)
    lon1 = math.radians(lon_point1)
    lat2 = math.radians(lat_point2)
    lon2 = math.radians(lon_point2)

    # calculate haversine
    lat = lat2 - lat1
    lon = lon2 - lon1
    d = math.sin(lat * 0.5) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(lon * 0.5) ** 2

    return 2 * _AVG_EARTH_RADIUS_M * math.asin(math.sqrt(d))


def _make_dist_columns(data, gap_station):
    """
    Calculate distance between taget station and surrounding stations can be 
    called on input data with `pd.pipe`.

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    gap_station : str
        DESCRIPTION.

    Raises
    ------
    ValueError
        Error if `data` does not contain coordinate columns.

    Returns
    -------
    data : pd.DataFrame
        modified input dataframe with newly appended column `x_dist`.

    """
    # vertical distances:
    Z = data.loc[data['stn']==gap_station,'Z'][0]
    data['z_dist'] = data['Z']-Z
    
    if 'X' in data.columns and 'Y' in data.columns:
        # Get location and height of the gap station.
        X = data.loc[data['stn']==gap_station,'X'][0]
        Y = data.loc[data['stn']==gap_station,'Y'][0]
        # pythagoras for swiss projection
        data['x_dist'] = np.sqrt((data['X']-X)**2 + (data['Y']-Y)**2)
        
    elif 'lon' in data.columns and 'lat' in data.columns:
        #haversine formula for wgs84 projection
        lon = data.loc[data['stn']==gap_station,'lon'][0]
        lat = data.loc[data['stn']==gap_station,'lat'][0]
        data['x_dist'] = data.apply(lambda df: _haversine(lat, lon, df['lat'], df['lon']),
                                    axis=1)
    
    else:
        raise ValueError(("coordinates in the input data have to be denoted "
                          "with the column names\n 'X', 'Y' for LV03 projection "
                          "or 'lon', 'lat' for wgs84 projection"))
    return data


def _filter_stations(data,
                     data_period,
                     gap_station,
                     drop_gap_station,
                     n_predictors,
                     sort_mode,
                     minimum_correlation,
                     max_horizontal_distance,
                     max_vertical_distance):
    """
    Filter stations based on input data and target station.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with columns:
            - stn: str with stationnames
            - HS: measured HS data
            - X: x-coordinate in LV03
            - Y: y-coordinate in LV03
            - Z: elevation
    data_period : pd.DatetimeIndex
        Period based on which the stations are filtered (i.e. correlations are
        calculated in that period).
    gap_station : str
        Name of the target station.
    drop_gap_station : bool
        whether to include the gap station or not.
    n_predictors : int
        Number of used neighboring stations.
    sort_mode : str {'best_correlated', 'nearest'}
        wheter the best `n_predictors` correlated or nearest station are 
        selected..
    minimum_correlation : float, optional
        Minimum allowed correlation between neighboring and target station.
        The default is -1.0.
    max_horizontal_distance : int, optional
        Maximum allowed horizontal distance in [m]. The default is 10000000.
    max_vertical_distance : int, optional
        Maximum allowed absolute vertical distance in [m].
        The default is 10000.



    Returns
    -------
    filtered_stns : list of str
        DESCRIPTION.
    correlations : pd.Series
        index is name of respective neighboring station.
    distances : pd.Series
        index is name of respective neighboring station.

    """
    assert sort_mode in {'best_correlated', 'nearest'}
    
    corrs = (data
             .loc[data_period,:]
             .pipe(_select_winter_months)
             .pivot(columns='stn', values='HS')
             .corr()
             .loc[gap_station]
             .rename('corr'))
    
    dists = (data
             .loc[data_period, ['stn','X','Y','Z']]
             .pipe(_select_winter_months)
             .pipe(_make_dist_columns, gap_station=gap_station)
             .groupby('stn').first()
             .loc[:,['x_dist','z_dist']])
    
    data = pd.concat([corrs, dists], axis=1)
    
    # appy constraints:
    filtered_data = data.loc[((data['corr']>minimum_correlation) &
                              (data['x_dist']<max_horizontal_distance) &
                              (data['z_dist'].abs()<max_vertical_distance)), :]
    
    if sort_mode == 'best_correlated':
        filtered_data = filtered_data.sort_values(by='corr', ascending=False)
    elif sort_mode == 'nearest':
        filtered_data = filtered_data.sort_values(by='x_dist')
    
    filtered_data = filtered_data.iloc[:1+n_predictors]
    
    if drop_gap_station:
        filtered_data = filtered_data.drop(gap_station)
    
    filtered_stns = filtered_data.index.tolist()
    correlations = filtered_data.loc[:, 'corr']
    distances = filtered_data.loc[:, ['x_dist', 'z_dist']]
    
    return filtered_stns, correlations, distances


def _select_data_for_SWE2HS_model(data,
                                  train_period,
                                  gap_period,
                                  gap_station,
                                  swe_source):

    assert swe_source in {'swe_SLFTI', 'swe_HS2SWE'}

    train_data = (data
                  .loc[data['stn']==gap_station, ['HS', swe_source]]
                  .loc[train_period]
                  .dropna())

    gap_data = (data
                .loc[data['stn']==gap_station, ['HS', swe_source]]
                .loc[gap_period]
                .dropna())

    y_train = train_data.loc[:, 'HS']
    X_train = train_data.loc[:, swe_source]
    y_gap = gap_data.loc[:,'HS']
    X_gap = gap_data.loc[:, swe_source]

    return y_train, X_train, y_gap, X_gap


def _select_data_for_regr(data,
                          train_period,
                          gap_period,
                          gap_station,
                          n_predictors,
                          sort_mode,
                          minimum_correlation=-1.0,
                          max_horizontal_distance=10000000,
                          max_vertical_distance=10000):
    """
    Selecting data for regression `train_gap_split` operations.
    X will be pivoted DataFrame with station HS values as columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with columns:
            - stn: str with stationnames
            - HS: measured HS data
            - X: x-coordinate in LV03
            - Y: y-coordinate in LV03
            - Z: elevation
    train_period : pd.DatetimeIndex
        Period from which train data is selected.
    gap_period : pd.DatetimeIndex
        Period in which the gap is simulated.
    gap_station : str
        Station at which th gap is simulated.
    n_predictors : int
        Number of used neighboring stations.
    sort_mode : str {'best_correlated', 'nearest'}
        wheter the best `n_predictors` correlated or nearest station are 
        selected..
    minimum_correlation : float, optional
        Minimum allowed correlation between neighboring and target station.
        The default is -1.0.
    max_horizontal_distance : int, optional
        Maximum allowed horizontal distance in [m]. The default is 10000000.
    max_vertical_distance : int, optional
        Maximum allowed absolute vertical distance in [m].
        The default is 10000.

    Returns
    -------
    y_train : pd.Series
        DESCRIPTION.
    X_train : pd.DataFrame
        DESCRIPTION.
    y_gap : pd.Series
        DESCRIPTION.
    X_gap : pd.DataFrame
        DESCRIPTION.

    """

    
    assert isinstance(train_period, pd.DatetimeIndex)
    assert isinstance(gap_period, pd.DatetimeIndex)
    assert isinstance(gap_station, str)
    
    # take winter HS data in train and test period
    train_data = (data
                  .loc[train_period,:]
                  .pipe(_select_winter_months)
                  .pivot(columns='stn', values='HS')
                  )

    gap_data = (data
                .loc[gap_period,:]
                .pipe(_select_winter_months)
                .pivot(columns='stn', values='HS')
                )
    
    
    filtered_stns, corrs, dists = _filter_stations(
            data,
            train_period,
            gap_station,
            drop_gap_station=True,
            n_predictors=n_predictors,
            sort_mode=sort_mode,
            minimum_correlation=minimum_correlation,
            max_horizontal_distance=max_horizontal_distance,
            max_vertical_distance=max_vertical_distance)

    
    y_train = train_data.loc[:, gap_station]
    X_train = train_data.loc[:, filtered_stns]
    y_gap = gap_data.loc[:, gap_station]
    X_gap = gap_data.loc[:, filtered_stns]
    
    return y_train, X_train, y_gap, X_gap


def _select_data_for_dist(data,
                          train_period,
                          gap_period,
                          gap_station,
                          n_predictors,
                          sort_mode,
                          minimum_correlation=-1,
                          max_horizontal_distance=10000000,
                          max_vertical_distance=10000):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    train_period : TYPE
        DESCRIPTION.
    gap_period : TYPE
        DESCRIPTION.
    gap_station : TYPE
        DESCRIPTION.
    n_predictors : TYPE
        DESCRIPTION.
    sort_mode : TYPE
        DESCRIPTION.
    minimum_correlation : TYPE, optional
        DESCRIPTION. The default is -1.
    max_horizontal_distance : TYPE, optional
        DESCRIPTION. The default is 10000000.
    max_vertical_distance : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    y_train : TYPE
        DESCRIPTION.
    X_train : TYPE
        DESCRIPTION.
    y_gap : TYPE
        DESCRIPTION.
    X_gap : TYPE
        DESCRIPTION.

    """
    """
    y_gap and y_train are Dataframes and not Series because we need location
    info for GIDS.
    """
    filtered_stns, corrs, dists = _filter_stations(
            data,
            train_period,
            gap_station,
            drop_gap_station=False,
            n_predictors=n_predictors,
            sort_mode=sort_mode,
            minimum_correlation=minimum_correlation,
            max_horizontal_distance=max_horizontal_distance,
            max_vertical_distance=max_vertical_distance)
    
    train_data = (data
                  .loc[data['stn'].isin(filtered_stns)]
                  .loc[train_period, ['stn','HS','X','Y','Z']]
                  .pipe(_select_winter_months)
                  )
    
    gap_data = (data
                .loc[data['stn'].isin(filtered_stns)]
                .loc[gap_period, ['stn','HS','X','Y','Z']]
                .pipe(_select_winter_months)
                )
    
    # add vertical distance columns.    
    train_data['z_dist'] = train_data['stn'].map(dists['z_dist'])
    gap_data['z_dist'] = gap_data['stn'].map(dists['z_dist'])
    # add horizontal distance columns:
    train_data['x_dist'] = train_data['stn'].map(dists['x_dist'])
    gap_data['x_dist'] = gap_data['stn'].map(dists['x_dist'])
    # add corelations columns.
    train_data['corr'] = train_data['stn'].map(corrs)
    gap_data['corr'] = gap_data['stn'].map(corrs)

    X_train = train_data.loc[(train_data['stn'] != gap_station)]
    y_train = train_data.loc[(train_data['stn'] == gap_station)]
    X_gap = gap_data.loc[(gap_data['stn'] != gap_station)]
    y_gap = gap_data.loc[(gap_data['stn'] == gap_station)]

    return y_train, X_train, y_gap, X_gap


class GapModel():

    
    def fit_predict(self):
        raise NotImplementedError


    def train_gap_split(self):
        raise NotImplementedError


    def _postprocess_predictions(self, y_pred):
        """
        Round to nearest integer and set negative predictions to zero.
        """
        return y_pred.clip(lower=0).round()
    
    def get_used_predictor_stations(self,
                                    data,
                                    train_period,
                                    gap_period,
                                    gap_station):
        
        y_train, X_train, y_gap, X_gap = self.train_gap_split(data,
                                                              train_period,
                                                              gap_period,
                                                              gap_station)
        
        if 'stn' in X_train.columns:
            predictor_stations = X_train['stn'].unique().tolist()
        else:
            predictor_stations = X_train.columns.tolist()
            predictor_stations = [x 
                                  for x in predictor_stations 
                                  if x not in ['month', 'season']]
        
        return predictor_stations

class RegressionModel(GapModel):
    """
    Base class for GapFilling classes that use a sklearn estimator for gap 
    filling. Implements fit_predict method.
    """
    def fit_predict(self,
                    data,
                    train_period,
                    gap_period,
                    gap_station):

        y_train, X_train, y_gap, X_gap = self.train_gap_split(data,
                                                              train_period,
                                                              gap_period,
                                                              gap_station)

        self.model.fit(X_train, y_train)

        y_pred = pd.Series(self.model.predict(X_gap),
                           index=X_gap.index,
                           name=self.method)

        return self._postprocess_predictions(y_pred)


class RandomForestFilling3_5(RegressionModel):

    def __init__(self,
                 n_predictor_stations=10,
                 grid_search=False,
                 param_grid='default'):
        
        
        self.method = 'RandomForest_V3.5'
        self.n_predictor_stations = n_predictor_stations
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), ['season']),
                ('droper', 'drop', ['month'])],
            remainder=StandardScaler())
        pipe = Pipeline(
            [('preprocess', preprocessor),
             ('forest', RandomForestRegressor(max_depth=70,
                                              n_estimators=200,
                                              n_jobs=-1))])
        if grid_search:
            if param_grid=='default':
                param_grid = {
                    'forest__max_depth': [20,100],
                    'forest__n_estimators':[200,1000]
                    }
            self.model = HalvingGridSearchCV(pipe,
                                      param_grid,
                                      n_jobs=-1)
        else:
            self.model = pipe
        
    
    def train_gap_split(self,
                        data,
                        train_period,
                        gap_period,
                        gap_station):

        y_train, X_train, y_gap, X_gap = _select_data_for_regr(data,
                                                               train_period,
                                                               gap_period,
                                                               gap_station,
                                                               n_predictors=self.n_predictor_stations,
                                                               sort_mode='best_correlated',
                                                               max_horizontal_distance=200000,
                                                               max_vertical_distance=500
                                                               )
        # mapping for months in seasons:
        seasons = {11: 'early', 12: 'early',
                   1: 'mid', 2: 'mid',
                   3: 'late', 4: 'late'}
        
        X_train['month'] = X_train.index.month
        X_gap['month'] = X_gap.index.month
        X_train['season'] = X_train.index.month.map(seasons)
        X_gap['season'] = X_gap.index.month.map(seasons)
        return y_train, X_train, y_gap, X_gap


class ElasticNetRegressionFilling(RegressionModel):
    """
    Gap filling with ElasticNetCV.
    """
    def __init__(self,
                 n_predictor_stations=15):
        
        self.method = 'Elastic Net Regression'
        self.n_predictor_stations = n_predictor_stations
        self.model = Pipeline([
            ('scaling', StandardScaler()),
            ('regr', ElasticNetCV(
                n_jobs=-1,
                selection='cyclic'))
            ])

    def train_gap_split(self,
                        data,
                         train_period,
                         gap_period,
                         gap_station):

        y_train, X_train, y_gap, X_gap = _select_data_for_regr(data,
                                                               train_period,
                                                               gap_period,
                                                               gap_station,
                                                               n_predictors=self.n_predictor_stations,
                                                               sort_mode='best_correlated',
                                                               max_horizontal_distance=200000,
                                                               max_vertical_distance=500
                                                               )
        return y_train, X_train, y_gap, X_gap


class MatiuFilling(GapModel):
    """
    Implementation of the imputation method described in Matiu et al 2021.
    
    Refererenced in the Paper as weighted normal ratio (WNR) method.
    
    Additional to vertical distance weighting, correlation weighting and 
    horizontal distance weighting can be selected.
    
    Default conditions for neighboring station selection as described in 
    Matiu (2021)::
        - horizontal distance < 200km
        - vertical distance < 500m
        - correlation of neighboring station to target station > 0.7
        - maximum 5 neighboring stations (if more, best 5 correlated)
    """

    def __init__(self,
                 n_predictor_stations=5,
                 weighting='vertical',
                 minimum_correlation=0.7):
        
        
        assert weighting in {'vertical', 'horizontal', 'correlation'}
        if minimum_correlation == 0.7:
            self.method = f'matiu {weighting} weighted'
        else:
            self.method = f'matiu {weighting} weighted_min_corr_{minimum_correlation:.1f}'
        self.weighting = weighting
        self.minimum_correlation = minimum_correlation
        self.n_predictor_stations = n_predictor_stations
        

    def train_gap_split(self,
                        data,
                         train_period,
                         gap_period,
                         gap_station):

        y_train, X_train, y_gap, X_gap = _select_data_for_dist(
            data,
            train_period,
            gap_period,
            gap_station,
            n_predictors=self.n_predictor_stations,
            sort_mode='best_correlated',
            minimum_correlation=self.minimum_correlation,
            max_horizontal_distance=200000,
            max_vertical_distance=500)

        return y_train, X_train, y_gap, X_gap

    def fit_predict(self,
                    data,
                    train_period,
                    gap_period,
                    gap_station):
        
        y_train, X_train, y_gap, X_gap = self.train_gap_split(data,
                                                              train_period,
                                                              gap_period,
                                                              gap_station)
        
        # calculate ratios of mean HS between reference and canditate series
        # in train period.
        ratios = y_train['HS'].mean() / X_train.groupby('stn')['HS'].mean()
        
        # map the ratios on X_gap for simple usage:
        X_gap['ratios'] = X_gap['stn'].map(ratios)

        # calculate the weights based on specified measure.
        if self.weighting == 'vertical':
            # weighting based on gaussian fuction with full width at half
            # maximum of 500:
            X_gap['weights'] = np.exp(-np.log(2)*(X_gap['z_dist']**2)/(250**2))
        elif self.weighting == 'horizontal':
            X_gap['weights'] = 1 / X_gap['x_dist']
        elif self.weighting == 'correlation':
            X_gap['weights'] = X_gap['corr'].copy()
        else:
            raise ValueError
        # remove infinity weights for stations with equal height
        X_gap['weights'] = X_gap['weights'].replace(np.inf, 1000)
        
        dates = []
        HS_modeled = []
        
        for target_day in list(y_gap.index):
            try:
                # get data for the target day and do IDW:
                day_data = X_gap.loc[target_day]
                if isinstance(day_data, pd.Series): # happens if only one neighboring station
                    day_data = day_data.to_frame().T
                # check if there is a data point at the exact same location (distance==0):
                zero_distance_HS = day_data.loc[day_data['x_dist'] == 0].HS
                if zero_distance_HS.size > 0 and not np.isnan(zero_distance_HS.iloc[0]):
                    HS_at_target_day = zero_distance_HS.iloc[0]
                else:
                    HS_at_target_day = np.average(day_data['HS']*day_data['ratios'],
                                                  weights=day_data['weights'])
                    
                
                dates.append(target_day)
                HS_modeled.append(HS_at_target_day)
            except KeyError:
                dates.append(target_day)
                HS_modeled.append(np.nan)
                continue
        
        y_pred = pd.Series(HS_modeled, index=dates, name=self.method)
        
        
        return self._postprocess_predictions(y_pred)


class SingleStationFilling(GapModel):
    """
    Use data from a single neighboring station is used to fill a gap.
    
    Parameters
    ----------
    distance_metric : {'best_correlated' or 'nearest'}, default 'best_correlated'
        whether the best correlated station or the station that has lowest
        horizontal distance to the target station is used.
    scaling : str, optional
        bias correction applied to the enighboring station:
            - no_scaling (default): directly use neihboring station data
            - mean_ratio: scale by the mean ration ofthe two station in the
              train period.
            - quantile_ratios: scale by mean ratios of the quantiles defined 
              wit `quantiles` argument.
    quantiles : int or list-like of float
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles. Passed
        to `pandas.qcut`. The default is [0., 0.75, 1.]
    """
    
    def __init__(self,
                 distance_metric='best_correlated',
                 scaling='no_scaling',
                 quantiles=[0.,0.75,1.]
                 ):

        assert distance_metric in {'best_correlated', 'nearest'}
        assert scaling in {'no_scaling', 'mean_ratio','quantile_ratios'}
        assert isinstance(quantiles, (int, list))

        if scaling == 'no_scaling':
            self.method = f'SingleStation_{distance_metric}'
        elif scaling == 'quantile_ratios':
            self.method = f'SingleStation_{distance_metric}_{quantiles}_{scaling}'
        else:
            self.method = f'SingleStation_{distance_metric}_{scaling}'

        self.distance_metric = distance_metric
        self.scaling = scaling
        self.quantiles = quantiles

    def train_gap_split(self,
                        data,
                        train_period,
                        gap_period,
                        gap_station):

        y_train, X_train, y_gap, X_gap = _select_data_for_regr(
                    data,
                    train_period=train_period,
                    gap_period=gap_period,
                    gap_station=gap_station,
                    n_predictors=1,
                    sort_mode=self.distance_metric,
                    minimum_correlation=-1,
                    max_horizontal_distance=200000,
                    max_vertical_distance=500)
        return y_train, X_train, y_gap, X_gap

    def fit_predict(self,
                    data,
                    train_period,
                    gap_period,
                    gap_station):

        y_train, X_train, y_gap, X_gap = self.train_gap_split(data,
                                                              train_period,
                                                              gap_period,
                                                              gap_station)

        if self.scaling == 'no_scaling':
            y_pred = pd.Series(X_gap.iloc[:,0].values,
                               index=X_gap.index,
                               name=self.method)

        if self.scaling == 'mean_ratio':
            ratio = y_train.mean() / X_train.mean()
            y_pred = pd.Series(X_gap.iloc[:,0]*ratio.values,
                               index=X_gap.index,
                               name=self.method)

        if self.scaling == 'quantile_ratios':
            # get ratios in the different quantiles:
            ratios = (y_train.groupby(pd.qcut(y_train, self.quantiles, labels=False)).mean() /
                      X_train.iloc[:,0].groupby(pd.qcut(X_train.iloc[:,0], self.quantiles, labels=False)).mean())

            # get quantile edges in y_train (we want to use the same quantile ranges
            # in X_gap):
            _, qranges_y_train = pd.qcut(y_train, self.quantiles, retbins=True)
            # expand qranges upper and lower limit:
            qranges_y_train[0] = 0
            qranges_y_train[-1]= 10000

            # get series with quantile labels of X_gap:
            qlabels_X_gap = pd.cut(X_gap.iloc[:,0], qranges_y_train, labels=False)
            # map ratios on labels:
            mapped_ratios_X_gap = qlabels_X_gap.map(ratios)

            y_pred = pd.Series(X_gap.iloc[:,0]*mapped_ratios_X_gap,
                               index=X_gap.index,
                               name=self.method)
            raise NotImplementedError("quantile ratios scaling is experimental and buggy")

        return self._postprocess_predictions(y_pred)


class GidsFilling(GapModel):
    """
    Implementation of gradient-plus-inverse-distance-squared method (GIDS) as
    described in Nalder and Wein (1998).
    """
    def __init__(self,
                 n_predictor_stations=10):
        
        self.method = 'GIDS'
        self.n_predictor_stations = n_predictor_stations

    def train_gap_split(self,
                        data,
                        train_period,
                        gap_period,
                        gap_station):
        
        y_train, X_train, y_gap, X_gap = _select_data_for_dist(
                    data,
                    train_period=train_period,
                    gap_period=gap_period,
                    gap_station=gap_station,
                    n_predictors=self.n_predictor_stations,
                    sort_mode='best_correlated',
                    minimum_correlation=-1,
                    max_horizontal_distance=200000,
                    max_vertical_distance=500)
        return y_train, X_train, y_gap, X_gap

    def fit_predict(self,
                    data,
                    train_period,
                    gap_period,
                    gap_station):

        y_train, X_train, y_gap, X_gap = self.train_gap_split(data,
                                                              train_period,
                                                              gap_period,
                                                              gap_station)

        gap_period_days = list(y_gap.index)

        dates = []
        HS_modeled = []

        for target_day in gap_period_days:
            try:
                y = X_gap.loc[target_day,'HS']
                X = X_gap.loc[target_day,['X','Y','Z']]
                model = LinearRegression()

                model.fit(X, y)
                coefs = model.coef_
            except (ValueError, AttributeError):
                # # we get Errors when only one neighboring station is selected
                # # set coefs to zero and do normal idw
                # coefs = np.array([0.,0.,0.])
                dates.append(target_day)
                HS_modeled.append(np.nan)
                continue

            try:
                # Step 3: get data for the target day and do IDW:
                day_data = X_gap.loc[target_day]
                if isinstance(day_data, pd.Series): # happens if only one neighboring station
                    day_data = day_data.to_frame().T
                # check if there is a data point at the exact same location (distance==0):
                zero_distance_HS = day_data.loc[day_data['x_dist'] == 0].HS
                if zero_distance_HS.size > 0 and not np.isnan(zero_distance_HS.iloc[0]):
                    HS_at_target_day = zero_distance_HS.iloc[0]
                else:
                    numerator_sum = np.sum((day_data['HS'] + 
                            coefs[0]*(y_gap.loc[y_gap.index.isin([target_day]),'X']-day_data['X']) + 
                            coefs[1]*(y_gap.loc[y_gap.index.isin([target_day]),'Y']-day_data['Y']) + 
                            coefs[2]*(y_gap.loc[y_gap.index.isin([target_day]),'Z']-day_data['Z'])) *
                            (1/day_data['x_dist'])**2)

                    denominator_sum = np.sum((1/day_data['x_dist'])**2)
                    
                    HS_at_target_day = numerator_sum / denominator_sum
                
                dates.append(target_day)
                HS_modeled.append(HS_at_target_day)
            except KeyError:
                dates.append(target_day)
                HS_modeled.append(np.nan)
                continue
        
        y_pred = pd.Series(HS_modeled, index=dates, name=self.method)

        return self._postprocess_predictions(y_pred)


class InverseDistanceSquaredFilling(GapModel):
    """
    IDW implementation.
    """
    def __init__(self,
                 n_predictor_stations=5):
        
        self.method = 'Inverse distance squared'
        self.n_predictor_stations = n_predictor_stations

    def train_gap_split(self,
                        data,
                        train_period,
                        gap_period,
                        gap_station):
        
        y_train, X_train, y_gap, X_gap = _select_data_for_dist(
                    data,
                    train_period=train_period,
                    gap_period=gap_period,
                    gap_station=gap_station,
                    n_predictors=self.n_predictor_stations,
                    sort_mode='best_correlated',
                    minimum_correlation=-1,
                    max_horizontal_distance=200000,
                    max_vertical_distance=500)
        return y_train, X_train, y_gap, X_gap

    def fit_predict(self,
                    data,
                    train_period,
                    gap_period,
                    gap_station):
        
        y_train, X_train, y_gap, X_gap = self.train_gap_split(data,
                                                              train_period,
                                                              gap_period,
                                                              gap_station)

        gap_period_days = list(y_gap.index)
        
        dates = []
        HS_modeled = []

        for target_day in gap_period_days:
            try:
                # get data for the target day and do IDW:
                day_data = X_gap.loc[target_day,:]
                if isinstance(day_data, pd.Series): # happens if only one neighboring station
                    day_data = day_data.to_frame().T
                # check if there is a data point at the exact same location (distance==0):
                zero_distance_HS = day_data.loc[day_data['x_dist'] == 0].HS
                if zero_distance_HS.size > 0 and not np.isnan(zero_distance_HS.iloc[0]):
                    HS_at_target_day = zero_distance_HS.iloc[0]
                else:
                    numerator_sum = np.sum((day_data['HS'] / (day_data['x_dist'])**2))
                    
                    denominator_sum = np.sum((1/day_data['x_dist'])**2)
                    
                    HS_at_target_day = numerator_sum / denominator_sum
                
                dates.append(target_day)
                HS_modeled.append(HS_at_target_day)
            except KeyError:
                dates.append(target_day)
                HS_modeled.append(np.nan)
                continue
        
        y_pred = pd.Series(HS_modeled, index=dates, name=self.method)
        
        return self._postprocess_predictions(y_pred)


class SWE2HSModel(GapModel):

    param_grid = [{'rho_new': list(range(60,161,10)),
                   'rho_max': list(range(400,751,50)),
                   'tau': list(range(10,71,10))}]

    def fit_predict(self,
                    data,
                    train_period,
                    gap_period,
                    gap_station):
        
        y_train, X_train, y_gap, X_gap = self.train_gap_split(data,
                                                              train_period,
                                                              gap_period,
                                                              gap_station)
        
        if 'snow17' in self.method.lower():
            # altitude of gap station:
            z_gap = (data
                .loc[data['stn']==gap_station, 'Z']
                .loc[gap_period]
                .iloc[0])
            # transform X_train and Y_train to swe series:
            X_train = (hy_batch_snow17(X_train.index,
                                       X_train['rre150d0'],
                                       X_train['tre200d0'],
                                       shift_dates=self.shifted_dates,
                                       elevation=z_gap,
                                       scf=1,
                                       lat=46,
                                       mbase=1)
                      .loc[:, 'model_swe']
                      )
            X_gap = (hy_batch_snow17(X_gap.index,
                                     X_gap['rre150d0'],
                                     X_gap['tre200d0'],
                                     shift_dates=self.shifted_dates)
                     .loc[:, 'model_swe']
                     )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_gap)
        y_pred.name = self.method
        # only return predictons in the winter months for being consistent.
        y_pred = _select_winter_months(y_pred)
        return self._postprocess_predictions(y_pred)
    
    

class SWE2HSSnow17(SWE2HSModel):

    def __init__(self,
                 shifted_dates=False,
                 n_jobs=-1):
        if not shifted_dates:
            self.method = 'SWE2HS_Snow17'
        else:
            self.method = 'SWE2HS_Snow17_shifted_dates'
        
        self.shifted_dates = shifted_dates
        
        self.model = SnowDepthParamEstimator(
            param_grid=[{
                'rho_new': list(range(70,131,5)),
                'rho_max': list(range(250,751,25)),
                'tau': list(range(10,81,5))}],
            n_jobs=n_jobs)


    def train_gap_split(self,
                        data,
                        train_period,
                        gap_period,
                        gap_station):
        
        train_data = (data
                 .loc[data['stn']==gap_station, ['HS','tre200d0','rre150d0']]
                 .loc[train_period]
                 .dropna()
                 )
    
        gap_data = (data
                    .loc[data['stn']==gap_station, ['HS','tre200d0','rre150d0']]
                    .loc[gap_period]
                    .dropna()
                    )
        
        y_train = train_data.loc[:, 'HS']
        X_train = train_data.loc[:, ['tre200d0','rre150d0']]
        y_gap = gap_data.loc[:,'HS']
        X_gap = gap_data.loc[:, ['tre200d0','rre150d0']]
        return y_train, X_train, y_gap, X_gap


models_that_use_meteo = {'SWE2HS_Snow17',
                         'SWE2HS_Snow17_shifted_dates'}

