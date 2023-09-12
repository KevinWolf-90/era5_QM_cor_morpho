#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 08:15:57 2022

@author: kwolf
"""

#Example and test code to download ERA5 data from MARS


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
import copy
import sys
import os
from pathlib import Path
import pandas as pd
from itertools import chain
import xarray as xr

#statistical imports
from scipy import stats

import os.path
from matplotlib.colors import LogNorm

import cdsapi

#--get number of days in month
def day_in_month(yyyy,mm):
    if mm in [1,3,5,7,8,10,12]:
        days = 31
    if mm in [4,6,9,11]:
        days = 30
    if ((mm == 2) & (yyyy%4 != 0)):
        days = 28
    if ((mm == 2) & (yyyy%4 == 0)):
        days = 29
    return days


##select time stop to be extracte##
timestep = 1 # to load the era5 reanalysis data on 1h steps
#timestep = 3 # to load the era5 3hourly ensemble spread; no reanalysis data
#timestep = 6

if timestep == 1:
    hours = list(np.arange(0,24,1))
    foo  = [str(f'{x:02.0f}')+':00' for x in hours]
    hours = foo

if timestep == 3:
    hours = list(np.arange(0,24,3))
    foo  = [str(f'{x:02.0f}')+':00' for x in hours]
    hours = foo
    
if timestep == 6:
    hours = list(np.arange(0,24,6))
    foo  = [str(f'{x:02.0f}')+':00' for x in hours]
    hours = foo



prod_type = 'reanalysis' #--download the main run; the re-analysis


#--which years to extract
years = ['2015','2016','2017','2018','2019','2020','2021']

#--which months to extract
months = [str(x) for x in range(1,13)]

#--which pressure levels to extract
plevels = ['125', '150', '175', '200', '225', '250', '300', '350'] #-- at which levels to extract

#--which domain to extract
areafoo = [70, -140, 30, 40]   #max lat, min lon, min lat, max lon

#--which variables to extract
variables = ['relative_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'fraction_of_cloud_cover']

#--where to store                                                                                                                                                                       
outpath = '/scratchx/kwolf/ERA5/' 

print('****************************')
print('Downloading ERA5: '+prod_type)
print('For years: '+str(years))
print('For months: '+str(months))
print('For hours: '+str(hours))
print('***************************')

for year in years: # loop over years
    #--check if directory for year exists, if not, create
    if  not os.path.exists(outpath+'/'+year):
        os.makedirs(outpath+'/'+year)
    for month in months: # loop over moths
            

        ndays = day_in_month(int(year),int(month))
        days = np.arange(1,ndays+1)
        foo = [str(x) for x in days] # make a list of strings
        days = foo

        print('Days: ',str(days))

        print('')
        print('Starting request for: '+str(year)+' and month: '+str(month))
        print('**********************************************************')
        
    
        c = cdsapi.Client()
        
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
               'product_type': prod_type,
                'variable': variables,
                'pressure_level': plevels,
                'year': year,
                'month': month,
                'day': days,
                'time': hours,
                'area': areafoo,  #--max lat, min lon, max lat, max lon
                'format': 'grib',
                #'format': 'netcdf',
            },
            outpath+str(f'{int(year):04.0f}')+'/'+str(f'{int(year):04.0f}')+'_'+str(f'{int(month):02.0f}')+'_era5_'+str(timestep)+'hour_'+str(f'{int(areafoo[1]):04.0f}')+'_'+str(f'{int(areafoo[3]):04.0f}')+'_'+str(f'{int(areafoo[0]):03.0f}')+'_'+str(f'{int(areafoo[2]):03.0f}')+'_'+str(prod_type)+'.grib')
    
    print('Saved to: ' + outpath+str(f'{int(year):04.0f}')+'/'+str(f'{int(year):04.0f}')+'_'+str(f'{int(month):02.0f}')+'_era5_'+str(timestep)+'hour_'+str(f'{int(areafoo[1]):04.0f}')+'_'+str(f'{int(areafoo[3]):04.0f}')+'_'+str(f'{int(areafoo[0]):03.0f}')+'_'+str(f'{int(areafoo[2]):03.0f}')+'_'+str(prod_type)+'.grib') 
        
print('DONE.')
