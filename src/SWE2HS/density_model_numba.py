# -*- coding: utf-8 -*-
"""
This module holds the a Model for transferring modeled swe data to HS.

The model can be used to transfer modeled snow water equivalent (swe) time 
series to snow depth series. It can be used as if it was a scikit-learn 
estimator in order to tune the parameters in the density growth function, 
the melting scheme and the density growth function itself.

The top level API is provided by the SnowDepthParamEstimator class. 

You can also use the SnowDepthEstimator class if you do not want to tune the 
parameters or already know what good parameter settings are. Both classes have 
the same interface and can be interchanged within a script.

@author: aschauer
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import logging

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from numba import jit
 
class SnowDepthParamEstimator(BaseEstimator, RegressorMixin):
    """
    Parameter tuning for the SnowDepthEstimator.
    
    This is a density model that tunes the hyperparameters in the density 
    function and does an exhaustive non cross-validated grid search over the 
    parameter space when being fitted.
    
    Parameters
    ----------
    param_grid : list of dicts, dict, 'minimal' (default) or 'extensive'
        The parameter spache that is evaluated during the fit. Is passed
        to the GridSearchCV and has to comply to the format defined in 
        GridSearchCV.
        
        Either a dictionary with parameters names (string) as keys and 
        lists of parameter settings to try as values, or a list of such 
        dictionaries, in which case the grids spanned by each dictionary in the
        list are explored. This enables searching over any sequence of 
        parameter settings.
        
        Two predefined parameter spaces exist which can be used by setting the
        strings 'minimal' (default) or 'extensive'. The extensive mode will 
        take long runtime for fitting the model.
        
    scoring : str or callable (default='neg_root_mean_squared_error')
        The scoring strategy for evaluating the model skill during the
        parameter optimization. For datails on scoring metrics check
        https://scikit-learn.org/stable/modules/model_evaluation.html where 
        you can find a list of predefined valid scoring strategies or read how
        make your own scorers with sklearn.metrics.make_scorer()

    n_jobs : int (default=1)
        number of CPU cores for parallel grid searching. If set to -1 all
        available cores are used. WARNING: parallel jobs can cause trouble in 
        ipython IDEs such as Spyder. If so, try to run outside ipython/Spyder.
        
    
    Attributes
    ----------
    best_params_ : dict
        parameter settings of the best estimator 
    
    best_estimator_ : SnowDepthEstimator
        The best SnowDepthEstimator which is used for prediction of new unseen
        swe data
    """
    def __init__(self, 
                 param_grid='minimal', 
                 scoring='neg_root_mean_squared_error',
                 n_jobs=1):
        """
        Initialize the default parameters
        """
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
    

        
    def fit(self, swe_train, HS_train):
        """
        Optimize the parameters of the SnowDepthEstimator.
        
        The parameters are searched with GridSearchCV but no cross-validation
        takes place. This is because the SnowDepthEstimator itself fits no
        parameters and therefore we do not need to hide data from it.
        
        Parameters
        ----------
        swe_train : pd.Series
            training data with swe from any model, needs to have a 
            pd.DateTimeIndex
            
        HS_train : pd.Series
            training data with measured HS series that are used to optimize 
            the coefficients in the density increase function. Preferably
            same Index as in swe_train.
            
        Returns
        -------
        self
        """
        if (isinstance(self.param_grid, list) or 
                isinstance(self.param_grid, dict)):
            param_grid = self.param_grid
            
        elif self.param_grid == 'extensive':
            param_grid = [
                {'rho_new': list(range(60,121,1)),
                 'rho_max': list(range(400,551,10)),
                 'tau': list(range(20,71,1))}]
            
        elif self.param_grid == 'minimal':
            param_grid = [
                {'rho_new': [60,70,80,90,100,110,120],
                 'rho_max': [450,500],
                 'tau': [20,30,40,50,60]}]
        
        gs = GridSearchCV(
                estimator=SnowDepthEstimator(), 
                param_grid=param_grid,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                cv=DisabledCV())
        
        gs.fit(swe_train, HS_train)
        self.best_params_ = gs.best_params_
        self.best_estimator_ = gs.best_estimator_ 
        return self
    
    def predict(self, swe_test):
        check_is_fitted(self)
        return self.best_estimator_.predict(swe_test)
    

class SnowDepthEstimator(BaseEstimator, RegressorMixin):
    """
    Implementation of a conceptual snow density model.
    
    The model is used for transforming snow water equivalent (swe) to snow 
    depth (HS). With the default parameters, it is assumed, that swe is given 
    in [mm] and HS will be predicted in [cm].
    
    The model searches for swe increases and treats each increase as a new snow
    layer. The density of each layer increases with time. The density increase
    is realized by the following exponential decay function:

            rho(t) = rho_max + (rho_new-rho_max) * e^(-t/tau)
    
    If the snow water eqivalent decreases, melt is realized starting from 
    the top layers.
    
    Parameters
    ----------
    rho_new : float (default=100.0)
        The assumed density of new snow
    rho_max : float (default=450.0)
        The assumed maximum density of snow
    tau : float (default=50.0)
        The decay constant used for the exponential decay function.
        
    Attributes
    ----------
    coef_ : dict
        the coefficients of the fitted model. Fit refers to the translation of
        strings to callables only. No parameter optimization is realized in 
        this class and the default values are taken only. If you want to do 
        parameter optimization, use the SnowDepthParamEstimator class.
    """
    def __init__(self,
                 rho_new=100.0,
                 rho_max=450.0,
                 tau=50.0):

        self.rho_new = rho_new
        self.rho_max = rho_max
        self.tau = tau


    def fit(self, swe_train, HS_train):
        """
        This fit function does not perform any optimization or learning. 
        
        It only transforms some parameter definition to callables and sets the 
        ``coef_`` attribute. This is necessary in order to be in compliance
        with the scikit-learn API guidelines for an estimator. 
        
        We need to comply to this structure in order to be able to use this 
        class within sklearn.model_selection.GridSearchCV for parameter 
        optimization.
        """
        coefs = self.get_params()
        
        self.coef_ = coefs
        return self
    
    def predict(self, swe_series):
        """
        Predict the HS series based on the set parameter values in self.coef_
        """
        check_is_fitted(self)
        HS = self._calculate_HS(swe_series, **self.coef_)
        return HS
    
    def _calculate_HS(self, swe_series, **coefs):
        """
        This method does the actual snow depth modeling.
        
        First, an instance of the SnowPack class is created. In a for loop, the 
        snowpack is dynamically updated for each day and the snow depth (HS) of
        that day is calculated.
        
        Parameters
        ----------
        swe_series : pd.Series
            Snow water equivalent model output from any snow model. Has to be
            a pd.Series with a DateTimeIndex
        
        **coefs : kwargs
            Usually a dict with keyword arguments passed to 
            Snowpack.get_total_depth()
        
        Returns
        -------
        HS : pd.Series
            modeled HS with same index as in the input swe_series
            
        """
        assert isinstance(swe_series, pd.Series), "input has to be pd.Series"
        assert isinstance(swe_series.index, (pd.DatetimeIndex, 
                                             pd.PeriodIndex)), \
               "index of the input has to be datetime."
        
        HS = call_ts_swe2hs_1D_numba(swe_series, **coefs)
        
        return HS

def dt2cal_numpy(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.
    
    Credit to RBF06: https://stackoverflow.com/a/56260054

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 2)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output 
    out = np.empty(dt.shape, dtype="u4")
    # decompose calendar floors
    Y, M = [dt.astype(f"M8[{x}]") for x in "YM"]
    out = (M - Y) + 1 # month
    return out.astype(int)

def call_ts_swe2hs_1D_numba(swe_series, rho_new, rho_max, tau):
    """
    

    Parameters
    ----------
    sample_data : pd.Series
        Series of SWE with DatetimeIndex.

    Returns
    -------
    out : pd.Series
        DESCRIPTION.

    """
    # transfer to numpy
    swe_input = swe_series.to_numpy()
    months = dt2cal_numpy(swe_series.index.to_numpy())
    # call numba function:
    hs = ts_swe2hs_1D_numba(months, swe_input, rho_new, rho_max, tau)
    # combine to pd.Series
    out = pd.Series(hs, index=swe_series.index)
    return out
    
@jit(nopython=True)
def ts_swe2hs_1D_numba(months, swe_input, rho_new, rho_max, tau):
    hs = np.zeros(len(swe_input))
    month_before = 8
    sept = 9
    for l, (month, swe) in enumerate(zip(months, swe_input)):
        
        if l == 0:  # set up layers at beginning
            layer_ages = np.zeros(366)
            layer_swes = np.zeros(366)
            i = 0
        # reset layers at beginning of September.
        elif month_before < sept and month >= sept:
            layer_ages = np.zeros(366)
            layer_swes = np.zeros(366)
            i = 0
        
        month_before = month
        if i==0:
            delta_swe = swe
        else:
            delta_swe = swe - swe_input[l-1]

        if delta_swe>0:
            layer_swes[i] = delta_swe
            
        if delta_swe<0:
            # melt_from_top: go backwards in layers (j decreasing)
            swe_removed = 0
            j = i-1
            while swe_removed > delta_swe: # both are (will be) negative
                swe_removed = swe_removed-layer_swes[j]
                layer_swes[j] = 0
                j = j-1
                # minimal floating point errors can cause the while loop to run away
                if j == -1: 
                    break

            # fill up last removed layer with excess swe:
            layer_swes[j+1] = delta_swe - swe_removed
    
        layer_ages[:i] = layer_ages[:i]+1
        i = i+1
        # density calculation over layers:
        hs[l] = np.sum(np.divide(layer_swes*100, (rho_max + (rho_new-rho_max)*np.exp(-layer_ages/tau))))
    return hs

class DisabledCV:
    """
    Disabled cross-validation for sklearn.
    
    This helper class is used to disable cross-validation in the parameter 
    estimation of GridSearchCV. 
    
    Taken from: https://stackoverflow.com/a/55326439
    """
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits    
