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
    
    n_jobs : int
        number of cores to be used in parallel. If n_jobs=-1 uses all available 
        cores.
    
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
    
    def get_cumulative_layer_depths(self, swe_test):
        
        check_is_fitted(self)
        return self.best_estimator_.get_cumulative_layer_depths(swe_test)
    
    def get_layer_depths(self, swe_test):
        
        check_is_fitted(self)
        return self.best_estimator_.get_layer_depths(swe_test)
        
    def fit(self, swe_train, HS_train):
        """
        Optimize the parameters of the SnowDepthEstimator.
        
        The parameters are searched with GridSearchCV but no cross-validation
        takes place. This is because the SnowDepthEstimator itself fits no
        parameters and therefore we do not need to hide data from it.
        
        Parameters
        ----------
        swe_train : pd.Series
            training data with swe [mm] from any model, needs to have a 
            pd.DateTimeIndex
            
        HS_train : pd.Series
            training data with measured HS series [cm] that are used to optimize 
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
                {'melt_method': ['from_top','from_bottom','proportional'],
                 'rho_increase_func': ['exponential_decay'],
                 'rho_new': list(range(60,121,1)),
                 'rho_max': list(range(400,551,10)),
                 'tau': list(range(20,71,1))},
                {'melt_method': ['from_top','from_bottom','proportional'],
                 'rho_increase_func': ['martinec_rho_function'],
                 'rho_new': list(range(60,121,1)),
                 'k': [x / 100.0 for x in range(15,46,1)]}]
            
        elif self.param_grid == 'minimal':
            param_grid = [
                {'melt_method': ['from_top','from_bottom','proportional'],
                 'rho_increase_func': ['exponential_decay'],
                 'rho_new': [60,70,80,90,100,110,120],
                 'rho_max': [450,500],
                 'tau': [20,30,40,50,60]},
                {'melt_method': ['from_top','from_bottom','proportional'],
                 'rho_increase_func': ['martinec_rho_function'],
                 'rho_new': [60,70,80,90,100,110,120],
                 'k': [0.2,0.25,0.3,0.35,0.4]}]
        
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
    can be realized by two functions:
        - 'exponential_decay'
            the density function is of the form
            rho(t) = rho_max + (rho_new-rho_max) * e^(-t/tau)
        - 'martinec_rho_function'
            the density function is of the form
            rho(t) = rho_new * (t+1)^k
    
    If the snow water eqivalent decreses, melt is either realized starting from 
    the top layers, from the bottom layers or proportional over all layers.
    
    Parameters
    ----------
    melt_method : str (default='from_top')
        The melting realization. Can be one of the following possibilities:
          - 'from_top': melting started from at the top of snow pack layers
          - 'from_bottom': melting strating from bottom
          - 'proportional': each layer is melted proportionally
    rho_increase_func : str (default='exponential_decay')
        The density increase function for the descriptipn of snow 
        snow settlement. Can be one of the following:
            - 'exponential_decay':
                the density function is of the form
                rho(t) = rho_max + (rho_new-rho_max) * e^(-t/tau)
            - 'martinec_rho_function':
                the density function is of the form
                rho(t) = rho_new * (t+1)^k
    rho_new : float (default=100.0)
        The assumed density of new snow
    rho_max : float (default=450.0)
        The assumed maximum density of snow
    tau : float (default=50.0)
        The decay constant used for the exponential decay function.
    k : float (default=0.3)
        The exponent in the martinec rho function
        
    Attributes
    ----------
    coef_ : dict
        the coefficients of the fitted model. Fit refers to the translation of
        strings to callables only. No parameter optimization is realized in 
        this class and the default values are taken only. If you want to do 
        parameter optimization, use the SnowDepthParamEstimator class.
    """
    def __init__(self, 
                 melt_method='from_top', 
                 rho_increase_func='exponential_decay',
                 rho_new=100.0,
                 rho_max=450.0,
                 tau=50.0,
                 k=0.3):
       
        self.melt_method = melt_method
        self.rho_increase_func = rho_increase_func
        self.rho_new = rho_new
        self.rho_max = rho_max
        self.tau = tau
        self.k = k
    
    def get_cumulative_layer_depths(self, swe_series):
        """
        Calculate the cumulative layer depths.
        
        The cumulative layer depths can be used to plot the evolution of the 
        snowpack over time.
        
        This function can only be called when the estimator is already fitted.
        
        Parameters
        ----------
        swe_test : pd.Series
            Snow water equivalent model output from any snow model. Has to be
            a pd.Series with a DateTimeIndex
        
        Returns
        -------
        layer_depths : pd.Dataframe
            Dataframe with the cumulative layer depths. Each column of the 
            dataframe refers to a single layer and each row refers to a 
            timestep.
        """
        check_is_fitted(self)
        layer_depths = self._get_layer_depths(swe_series)
        cumulation = None
        for column in layer_depths.columns:
            if cumulation is None:
                cumulation = layer_depths.iloc[:,0]
            else:
                cumulation += layer_depths[column]
                layer_depths[column] = cumulation
        return layer_depths
    
    def _get_layer_depths(self, swe_test):
        """
        Calculate the individual layer depths.
        
        Can only be called when the estimator is already fitted.
        
        Parameters
        ----------
        swe_test : pd.Series
            Snow water equivalent model output from any snow model. Has to be
            a pd.Series with a DateTimeIndex
        
        Returns
        -------
        layer_depths : pd.Dataframe
            Dataframe with the individual layer depths. Each column of the 
            dataframe refers to a single layer and each row refers to a 
            timestep.
        """
        check_is_fitted(self)
        snowpack = SnowPack()

        layer_depths = []
        
        # Initialize date of previous entry.
        date_before = swe_test.index[0]
        
        for date, swe in swe_test.items():
            delta_swe = swe - snowpack.get_total_swe()
                
            if delta_swe > 0:
                snowpack.add_layer(delta_swe)
            if delta_swe < 0:
                snowpack.melt(delta_swe, self.coef_['melt_method'])
            
            layer_depths.append(snowpack.get_layer_depths(**self.coef_))
            
            snowpack.increase_age()
            
            snowpack.remove_old_layers()
            # Reset the snowpack if we pass by the begining of September.
            if  date_before.dayofyear <= 244 <= date.dayofyear:
                snowpack.reset_snowpack()
            
            # Update the date of the previous entry.
            date_before = date

        layer_depths = pd.concat(layer_depths, axis=1)
        layer_depths = layer_depths.transpose()
        layer_depths.index = swe_test.index
        return layer_depths
    
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
        if coefs['rho_increase_func'] == 'exponential_decay':    
            coefs['rho_increase_func'] = self.__exponential_decay
        elif coefs['rho_increase_func'] == 'martinec_rho_function':
            coefs['rho_increase_func'] = self.__martinec_rho_function
        else: 
            raise ValueError("rho_increase_func must be one of the following:"
                             "'exponential_decay' or 'martinec_rho_function'")
        self.coef_ = coefs
        return self
    
    def predict(self, swe_test):
        """
        Predict the HS series based on the set parameter values in self.coef_
        """
        check_is_fitted(self)
        HS = self._calculate_HS(swe_test, **self.coef_)
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
        # if swe_series.iloc[0] > 0:
        #     logging.warning('You do not start modeling with an empty Snowpack')
        
        
        snowpack = SnowPack()
        
        # Initialize a list for intermediate HS storage.
        HS = []
        
        # Initialize date of previous entry.
        date_before = swe_series.index[0]
        
        for date, swe in swe_series.items():
            # Reset the snowpack if we pass by the begining of September.
            if  date_before.month < 9 <= date.month:
                snowpack.reset_snowpack()
                # if swe > 0:
                #     logging.warning("A new winter starts without empty snowpack")
            
            delta_swe = swe - snowpack.get_total_swe()
                
            if delta_swe > 0:
                snowpack.add_layer(delta_swe)
            if delta_swe < 0:
                snowpack.melt(delta_swe, coefs['melt_method'])
            
            HS.append(snowpack.get_total_depth(**coefs))
            
            snowpack.increase_age()
            snowpack.remove_old_layers()

            # Update the date of the previous entry.
            date_before = date

        HS = pd.Series(HS, index=swe_series.index)
        return HS
    
    @staticmethod
    def __exponential_decay(age, rho_new=80, rho_max=500, tau=50, **kwargs):
        """
        Desnity growth function of snow of the form::
            
            rho(t) = rho_max + (rho_new-rho_max) * e^(-t/tau)
            
        Where rho(t) is the density at a certain time, rho_new is the density of 
        new snow, rho_max is the maximum density of snow, tau is the decay 
        constant for the exponential function
        
        Parameters
        ----------
        age: int
            Age of the snow layer
        rho_new : int or float (default=80)
            density of the new snow in [kg/m^3]
        rho_max : int or float (default=500)
            maximum density of settled snow in [kg/m^3]
        tau : int or float (default=50)
            decay constant for the density increase
        """
        return rho_max + (rho_new-rho_max)*np.exp(-age/tau)
    
    @staticmethod
    def __martinec_rho_function(age, rho_new=100, k=0.3, **kwargs):
        """
        Density growth function of snow after Martinec (1977)::
            
            rho(t) = rho_new * (t+1)^k
        
        Where rho_new is the density of new snow and k is an empirical exponent.
        """
        return rho_new * ((age+1)**k)


class SnowPack():
    """
    The SnowPack class bundels SnowLayer objects, can add and melt snow 
    layers, calculate swe in the snowpack, depths of the layers and the total 
    depth of the snowpack. 
    """
    def __init__(self):
        self.layers = OrderedDict()
        self.n_layers = 0
        
    def add_layer(self, delta_swe):
        assert delta_swe > 0, 'delta_swe has to be positive for a new layer'
        self.layers[self.n_layers] = SnowLayer(delta_swe)
        self.n_layers += 1
        
    def get_layer_depths(self, rho_increase_func, **kwargs):
        """
        Returns a pd.Series with the layer depths, the index of the layer depth
        refers to the layer number
        
        Parameters
        ----------
        rho_increase_func : function
            the function which defines the density increase of the snow layer
            time
        **kwargs : keyword arguments
            keyword arguments passed to the density increase function
        
        Returns
        -------
            pd.Series with the layer depths and layer numbers as index
        """
        layer_depths = []
        layer_numbers = []
        for n, layer in self.layers.items():
            layer_depths.append(layer.get_layer_depth(rho_increase_func,
                                                      **kwargs))
            layer_numbers.append(n)
        return pd.Series(layer_depths, index=layer_numbers, dtype='float64')
        
    def get_total_depth(self, rho_increase_func, **kwargs):
        """
        Returns the total thickness of the snowpack.
        
        Parameters
        ----------
        rho_increase_func : function
            the function which defines the density increase of the snow layer
            time
        **kwargs : keyword arguments
            keyword arguments passed to the density increase function
        """
        return self.get_layer_depths(rho_increase_func, **kwargs).sum()
    
    def get_total_swe(self):
        """
        Returns the total snow water equivalent of the snowpack.
        """
        total_swe = .0
        for layer in self.layers.values():
            total_swe += layer.get_swe()
        return total_swe
    
    def increase_age(self):
        """
        Increase the age of each layer by 1.
        
        Returns
        -------
        None
        """
        for layer in self.layers.values():
            layer.increase_age()
    
    def melt(self, delta_swe, method='from_top'):
        """
        Melt the snowpack by a given amount of snow water equivalent.
        
        Parameters
        ----------
        delta_swe : float
            The amount of water that has to be melted from the snowpack
        method : {'from_top', from_bottom', 'proportional'}
            The melt method that is applied. Either layers are melted from top 
            of the snowpack, from bottom or proportionally over all layers of 
            the snowpack.
        
        Returns
        -------
        None
        """
        if method == 'from_top':
            self.melt_layers_from_top(delta_swe)
        elif method == 'from_bottom':
            self.melt_layers_from_bottom(delta_swe)
        elif method == 'proportional':
            self.melt_layers_proportional(delta_swe)
        else: 
            raise ValueError('No valid melt method given')
            
    def melt_layers_from_bottom(self, delta_swe):
        """
        Melts layers starting from the bottom of the snowpack.
        
        Parameters
        ----------
        delta_swe : float
            The amount of water that has to be melted from the snowpack
            
        Returns
        -------
        None
        """
        # Iterate from bottom to top in layers dictionary.
        for n, layer in self.layers.items():
            if delta_swe < 0:
                # The layer is melted and delta_swe is updated accordingly.
                delta_swe = layer.melt_absolute(delta_swe)
                # We do not delete the empty layers here, we leave them at the 
                # bottom of the snowpack. This is necessary for the cumulative
                # layer depth calculation (we do not want to get nans at the 
                # bottom)
                
    def melt_layers_from_top(self, delta_swe):
        """
        Melts layers starting at the top of the snowpack.
        
        Parameters
        ----------
        delta_swe : float
            The amount of water that has to be melted from the snowpack
            
        Returns
        -------
        None
        """
        empty_layers = []
        # Iterate from top to bottom in layers dictionary (reversed order)
        for n, layer in reversed(self.layers.items()):
            if delta_swe < 0:
                delta_swe = layer.melt_absolute(delta_swe)
                if layer.is_empty():
                    # Store the empty layers in a list for subsequent deletion 
                    # of the empty layers. The layers cannot be deleted whithin 
                    # this loop because an OrderedDict (self.layers) cannot be 
                    # modified when being iterated over. 
                    empty_layers.append(n)
        for n in empty_layers:
            del self.layers[n]
            self.n_layers -= 1
            
    def melt_layers_proportional(self, delta_swe):
        """
        Melts proportionally over all layers of the snowpack.
        
        Parameters
        ----------
        delta_swe : float
            The amount of water that has to be melted from the snowpack
            
        Returns
        -------
        None
        """
        assert delta_swe < 0, 'delta_swe has to be negative for melt'
        old_swe = self.get_total_swe()    
        percentage = (old_swe+delta_swe)/old_swe
        for layer in self.layers.values():
            layer.melt_proportional(percentage)
    
    def remove_old_layers(self):
        """
        Remove all layers if there is no snow present anymore.
        """
        if self.get_total_swe() == 0:
            self.layers.clear()
            self.n_layers = 0
    
    def reset_snowpack(self):
        """
        Remove all layers in the snowpack and set the number of layers to zero.
        """
        self.layers.clear()
        self.n_layers = 0
        return None


class SnowLayer():
    """
    A single snow layer. 
    
    Keeps track of the age and the swe which allows to calculate the thickness 
    via a density function.
    """
    def __init__(self, initial_swe):
        self.swe = initial_swe
        self.age = 0
        
    def get_layer_depth(self, rho_increase_func, **kwargs):
        """
        Calculate the snow depth of the layer based on swe and density.
        
        Parameters
        ----------
        rho_increase_func : function
            the function which defines the density increase of the snow layer
            time
        **kwargs : keyword arguments
            keyword arguments passed to the density increase function
        
        Returns
        -------
        depth : float
            the depth of the SnowLayer
        
        """
        # The swe has to be multiplied by 100 in order to get HS in [cm] if we 
        # assume that swe is given in [mm] and rho is given in [kg/m^2].
        return np.divide(self.swe*100, rho_increase_func(self.age, **kwargs))
        
    def get_swe(self):
        """
        Return the swe of the layer
        """
        return self.swe
    
    def increase_age(self):
        """
        Increase the age of the layer by one
        """
        self.age += 1
        return None
    
    def is_empty(self):
        """
        Check if there is any water in the layer.
        
        Returns
        -------
        bool
        """
        return self.swe == 0
    
    def melt_absolute(self, delta_swe):
        """
        Reduces the swe of the layer according to delta_swe. 
        
        If there needs to be more melt than can be realized in this layer, swe 
        will be set to zero and delta_swe will be propagated. This function is
        only able to be called when delta_swe is negative.
        """
        assert delta_swe < 0, 'delta_swe has to be negative for melt'
        # The swe is decreased by delta_swe. 
        self.swe = np.add(self.swe, delta_swe)
        delta_swe = 0
        # If there is more melt required than swe in the layer, swe is set to 
        # zero and delta_swe is retured for melting in the next layer.
        if self.swe < 0:
            delta_swe = self.swe
            self.swe = 0
        return delta_swe
        
    def melt_proportional(self, percentage):
        """
        Reduces swe in the layer based on a percentage value.
        
        The percentage of remaining swe has to be between 0 and 1.
        """
        assert 0 <=  percentage <= 1, 'melt percentage must be between 0 and 1' 
        self.swe = np.multiply(percentage, self.swe)
        return None


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