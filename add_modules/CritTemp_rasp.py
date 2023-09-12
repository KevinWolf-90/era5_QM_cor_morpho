#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:10:21 2021
following
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD012443

@author: kwolf
"""

import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

import os.path
import sys
from rh_ice_to_rh_liquid import rh_ice_to_rh_liquid
from rh_liquid_to_rh_ice import rh_liquid_to_rh_ice
from saturation import eSatLiquid, eSatIce


def heatcapacity(T):
    cp = 0.12125 * T + 977.25  # self-made linear relation of the hrat capacity
    return cp


def TLM_rap(Gin,Tamb):
    TLM = 226.69 + 9.43*np.log(Gin - 0.053) + 0.72*np.log(Gin - 0.053)**2    
    rh_TLM = (Gin*(Tamb - TLM) + eSatLiquid(TLM)) / (eSatLiquid(Tamb))
    return np.asarray([TLM, rh_TLM]) 

def TIM_rap(Gin,Tamb):
    TIM = 229.79 + 9.08*np.log(Gin - 0.02) + 0.49*np.log(Gin - 0.02)**2.
    rh_TIM = (Gin*(Tamb - TIM) + eSatIce(TIM)) / (eSatIce(Tamb))
    return np.asarray([TIM,rh_TIM])

def CritTemp_rasp(Tin,pin,U,eta,Q=43E6,Ein=1.25): #Tin in K and pin in Pa  U between 0 and 1

    Tin =np.asarray([Tin])    
    pin =np.asarray([pin])    
    U =np.asarray([U])

    E_h2o = Ein # kg kg-1
    c_p = heatcapacity(Tin) #--calculate temperature tempendent heat capacity # J kg-1
    epsilon = 0.622
    G = (E_h2o * c_p * pin) / (epsilon * Q * (1-eta)) # contrail factor; slope of the tangent
    #print('G: ',G)
    #print('          ')
    TLM_out,TLM_rh = TLM_rap(G,Tin)
    TIM_out,TIM_rh = TIM_rap(G,Tin)
    
    #--calculate supersaturation with respect to ice
    rh_amb_ice = rh_liquid_to_rh_ice(U,Tin)

    
    out_arr = np.asarray([TLM_out,TLM_rh,rh_amb_ice]) # transpose to have the variables in columns
    return out_arr.T

