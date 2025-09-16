from functools import partial
import numpy as np
from pystorm import mnt
from MHT.utils import *

def feature_for_node(gamma,node,feature_fn,**kwargs):
    feature_value = feature_fn(gamma,**kwargs)
    return feature_value[...,node]

def freq_weightedAverage_along_axis(x,band_range=[2,49],freq_res=0.5,freq_offset=0.0,axis=-1):
    freqs = np.arange(band_range[0],band_range[1], freq_res)
    freqs_format = [1 for _ in range(len(x.shape))]
    freqs_format[axis] = len(freqs)
    formatted_freqs = freqs.reshape(freqs_format)
    indices = (freqs/freq_res).astype(int)
    power_values =  indices_along_axis(x,axis=axis,indices=indices)
    power_along = power_values.sum(axis,keepdims=True)
    #power_along[power_along==0] = 1
    power_values = zpd(power_values,power_along)
    return (power_values * formatted_freqs).sum(axis) + freq_offset

def freq_sumnormpower_along_axis(x,band_range = [0,47],full_band_range=[0,47,[-1,-2]],freq_res=0.5,axis=-1):
    freqs = np.arange(full_band_range[0],full_band_range[1], freq_res)
    freqs_band = np.arange(band_range[0],band_range[1], freq_res)
    indices = (freqs/freq_res).astype(int)
    full_power = indices_along_axis(x,axis=axis,indices=indices)
    full_power_sum = full_power
    for ax in full_band_range[2]:
        full_power_sum = np.sum(full_power_sum,axis=ax,keepdims=True)
    full_power_sum_norm = zpd(full_power,full_power_sum)
    band_indices = (freqs_band/freq_res).astype(int)
    band_power = indices_along_axis(full_power_sum_norm,axis=axis,indices=band_indices).sum(-1)
    return band_power

def within_band_power_coupl(gamma,band_freq_range,freq_res,node_sel=0,full_band_range=[0,40,[-1,-2]],**kwargs):
    return feature_for_node(gamma,node_sel,freq_sumnormpower_along_axis,band_range=band_freq_range,full_band_range=full_band_range,freq_res=freq_res)

def sum_normpower_for_freqs_along_axis(x,band_range = [0,47],full_band_range=[0,47,[-1,-2]],freq_res=0.5,axis=-1):
    x_ = mnt.ensure_numpy(x)
    freqs = np.arange(full_band_range[0],full_band_range[1]+freq_res, freq_res)
    if freqs.shape[-1] > x.shape[-1]:
        freqs = freqs[:x.shape[-1]]
    freqs_band = np.arange(band_range[0],band_range[1]+freq_res, freq_res)
    indices = (freqs/freq_res).astype(int)
    full_power = indices_along_axis(x_,axis=axis,indices=indices)
    full_power_sum = full_power
    for ax in full_band_range[2]:
        full_power_sum = np.sum(full_power_sum,axis=ax,keepdims=True)
    full_power_sum_norm = zpd(full_power,full_power_sum)
    band_indices = (freqs_band/freq_res).astype(int)
    power = indices_along_axis(full_power_sum_norm,axis=axis,indices=band_indices)
    return mnt.ensure_numpy(power)
