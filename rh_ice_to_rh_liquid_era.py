#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:14:45 2021

@author: kwolf
"""

import numpy as np
import os.path
import sys
from ecmwf_saturation import eSatLiquid_ecmwf, eSatIce_ecmwf

#--this function converts rh from era5 to a consistent rh liquid
#--to be comparable with the observations 
    
def rh_ice_to_rh_liquid_era(rhin,Tin):   # input in rel hum [0-1], T in K
    #--output is rel. hum in %
    #--following: https://digital.library.unt.edu/ark:/67531/metadc693874/m1/6/
    if (isinstance(Tin, float)) or (isinstance(Tin, int)):
        T_a = np.float(Tin)
    else:
        T_a = np.asarray(Tin,dtype=np.float)
    rh_i = rhin 
      
    Ew = eSatLiquid_ecmwf(T_a)
    
    if (isinstance(T_a, float)) or (isinstance(T_a, int)):
        if T_a < 0:
            T_a = np.nan
    else:
       nan_where = np.where(T_a < 0)
       T_a[nan_where] = np.nan
    
    Ei = eSatIce_ecmwf(T_a)
    
    #following th equations out of
    #https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf#subsection.7.4.2
    #pages 114 to 115
    
    Tice = 250.16
    T0 = 273.16
    scale_alpha = ((Tin - Tice) / (T0 - Tice))**2
    
    if isinstance(scale_alpha, float):
        if Tin < Tice:
            scale_alpha = 0
        elif Tin > T0:
            scale_alpha = 1
        else:
            scale_alpha = ((Tin - Tice) / (T0 - Tice))**2
    else:    
        locator = (Tin < Tice)
        scale_alpha[locator] = 0
        locator = (Tin > T0)
        scale_alpha[locator] = 1 
    

    rh_w = np.asarray((rh_i * (Ei * (1-scale_alpha) + Ew * (scale_alpha))) / Ew)
    rh_w[rh_w < 0] = 0    
    
    return(rh_w) 

