import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numba as nb
from numba import jit



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


def swe2hs_1D_numpy(swe_input, rho_new, rho_max, tau):
    # initialize when passing 1.September:
    layer_ages = np.zeros(366, dtype=int)
    layer_swes = np.zeros(366, dtype=float)
    hs = np.zeros(len(swe_input))
    
    delta_swes = np.diff(swe_input, prepend=[0])
    for i, swe in enumerate(swe_input):
        if delta_swes[i]>0:
            layer_swes[i] = delta_swes[i]
            
        if delta_swes[i]<0:
            # melt_from_top(layer_swes, delta_swes[i])
            swe_removed = 0
            j = i-1
            while swe_removed > delta_swes[i]:
                swe_removed -= layer_swes[j]
                layer_swes[j] = 0
                j -= 1
            layer_swes[j+1] = delta_swes[i] - swe_removed
    
        layer_ages[:i] +=1
        # dichteberechnung
        hs[i] = np.sum(np.divide(layer_swes*100, (rho_max + (rho_new-rho_max)*np.exp(-layer_ages/tau))))
    return hs

def swe2hs_2D_numpy(swe_input, rho_new, rho_max, tau):
    hs = np.zeros(swe_input.shape)
    for r, c in np.ndindex(swe_input.shape[:2]):
        hs[r,c] = swe2hs_1D_numpy(swe_input[r,c,:], rho_new, rho_max, tau)
    return hs

@jit(nopython=True)
def swe2hs_1D_numba(swe_input, rho_new, rho_max, tau):
    # initialize when passing 1.September:
    layer_ages = np.zeros(366)
    layer_swes = np.zeros(366)
    hs = np.zeros(swe_input.shape)

    for i, swe in enumerate(swe_input):
        if i==0:
            delta_swe = swe
        else:
            delta_swe = swe - swe_input[i-1]

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
            # fill up last removed layer with excess swe:
            layer_swes[j+1] = delta_swe - swe_removed
    
        layer_ages[:i] = layer_ages[:i]+1
        # density calculation over layers:
        hs[i] = np.sum(np.divide(layer_swes*100, (rho_max + (rho_new-rho_max)*np.exp(-layer_ages/tau))))
    return hs

@jit(nopython=True)
def swe2hs_2D_numba_V1(swe_input, rho_new, rho_max, tau):
    hs = np.zeros(swe_input.shape)
    for r, c in np.ndindex(swe_input.shape[:2]):
        hs[r,c] = swe2hs_1D_numba(swe_input[r,c,:], rho_new, rho_max, tau)
    return hs

@jit(nopython=True)
def swe2hs_2D_numba_V2(swe_input, rho_new, rho_max, tau):
    # initialize when passing 1.September:
    layer_ages = np.zeros(366)
    layer_swes = np.zeros(swe_input.shape[:2]+(366,))
    hs = np.zeros(swe_input.shape)
    
    for i in range(swe_input.shape[2]):
        for r, c in np.ndindex(swe_input.shape[:2]):
            swe = swe_input[:, :, i]
            if i == 0:
                delta_swe = swe
            else:
                delta_swe = swe - swe_input[:,:,i-1]

            if delta_swe[r,c] > 0:
                layer_swes[r,c,i] = delta_swe[r,c]
                
            if delta_swe[r,c] < 0:
                # melt_from_top: go backwards in layers (j decreasing)
                swe_removed = 0
                j = i-1
                while swe_removed > delta_swe[r,c]: # both are (will be) negative
                    swe_removed = swe_removed-layer_swes[r,c,j]
                    layer_swes[r,c,j] = 0
                    j = j-1
                # fill up last removed layer with excess swe:
                layer_swes[r,c,j+1] = delta_swe[r,c] - swe_removed
    
        layer_ages[:i] = layer_ages[:i]+1
        # density calculation over layers:
        hs[:,:,i] = np.sum(np.divide(layer_swes*100, (rho_max + (rho_new-rho_max)*np.exp(-layer_ages/tau))), axis=2)
    return hs

