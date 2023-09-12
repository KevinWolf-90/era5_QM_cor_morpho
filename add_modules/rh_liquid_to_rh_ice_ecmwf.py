#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:14:45 2021

@author: kwolf
"""

#--for conversion of rh liquid to rh ice
#--not to be used for unifying era5 data, no temp. depended selection of rh reference
#--uses rh definitions / equation for esat

import numpy as np
import os.path
import sys
from ecmwf_saturation import eSatLiquid_ecmwf, eSatIce_ecmwf
 

def rh_liquid_to_rh_ice_ecmwf(rhin,Tin):   # input in rel hum [0-1], T in K
    #--output is rel. hum in 0-1
    #--following: https://digital.library.unt.edu/ark:/67531/metadc693874/m1/6/
    if (isinstance(Tin, float)) or (isinstance(Tin, int)):
        T_a = np.float(Tin)
    else:
        T_a = np.asarray(Tin,dtype=np.float)
    rh_w = rhin 
    Ew = eSatLiquid_ecmwf(T_a)
    Ei = eSatIce_ecmwf(T_a)
    rh_i = np.asarray((rh_w * Ew) / Ei)
    rh_i[rh_i < 0] = np.nan    
    return(rh_i) 

