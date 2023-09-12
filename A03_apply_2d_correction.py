#!/usr/bin/env python3/
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:39:57 2022

@author: kwolf
"""

#Programto read IAGOS data

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
from datetime import timedelta
import netCDF4 as nc
#from datetime import datetime
#from time import mktime
import copy
import sys
import os
from pathlib import Path
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import great_circle_calculator.great_circle_calculator as gcc

#--get string of alphabet
import string
alpha_string = list(string.ascii_lowercase)


from PIL import Image


import matplotlib.colors as colors

import xarray as xr

#statistical imports
from scipy import stats


import os.path
from matplotlib.colors import LogNorm

#--create normalized histogram
def my_histogram(value,xmin,xmax,step):
        #used to calculate my own histograms
        dummy = copy.deepcopy(value)
        dummy = dummy.reshape(dummy[:].size)
        yyy, xxx = np.histogram(dummy[:], bins=np.linspace(xmin, xmax, int((xmax-xmin)/step)))
        y_total = np.nansum(yyy)
        yyy = np.divide(yyy,y_total)

        return(xxx,yyy)

#--calculate cumulative distribution function
def cum_sum(values_in):
    #--incoming data sorted
    data_cum_sorted = np.sort(values_in)
    # calculate the proportional values of samples
    p = np.linspace(0, 1, len(data_cum_sorted), endpoint=False)
    return p,data_cum_sorted

############
#-my metrics
#--mae
def mae(predictions, y_true):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.nanmean(np.abs(y_true - predictions))


#--rsquare
def rSquare(estimations, measureds):
    """ Compute the coefficient of determination of random data. 
    This metric gives the level of confidence about the model used to model data"""
    SEE =  np.nansum((np.array(measureds) - np.array(estimations))**2)
    mMean = np.nansum(np.array(measureds)) / float(len(measureds))
    dErr = np.nansum((mMean - measureds)**2)
    return 1 - (SEE / dErr)

#--mse mean squared error
def mse(estimations, measured):
    return np.nansum((estimations - measured)**2.) / len(estimations)
    
def rmse(estimations, measured):
    return np.sqrt(np.nansum((estimations - measured)**2.) / len(estimations))

#--mean difference
def md(estiamtions, measured):
    return np.nanmean(estiamtions - measured)
#####################


#disable warning. keep output clean
#import warnings
#warnings.filterwarnings("ignore")

print('#################################')
print('Filter warnings are switched off!')
print('#################################')
time.sleep(1)


#server = 0 # 0 if local, 1 if on server
server = 1


if server == 1:  # to not use Xwindow
    if any('SPYDER' in name for name in os.environ):
        print('Activated plotting on screen')
    else:
        print('Deactivated plotting on screen for terminal and batch')
        matplotlib.use('Agg')



if server == 0:
    sys.path.insert(1, '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines')
if server == 1:
    sys.path.insert(1, '/homedata/kwolf/40_era_iagos/00_code')


from rh_ice_to_rh_liquid import rh_ice_to_rh_liquid  #assumes that there is pure ice, which is true for the extracted data!
#--import module to calculate the Schmidt-Appleman-criterion
from CritTemp_rasp import CritTemp_rasp

#%%
#### write diagnose ouput file
filename = 'diagnose.txt'

if server == 0:
    filename_diag = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename
if server == 1:
    filename_diag = '/homedata/kwolf/40_era_iagos/'+filename



outfile2 = open(filename_diag ,'w')
outfile2.write('Diagnose \n')
outfile2.write('========\n')

########################
#year_to_read = '*2018'
year_to_read = '*'
########################

#%%
#--1 hour time step data
filedummz = year_to_read+'_era_1h.npz'
if server == 0:
    path = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    files = Path(path).glob(filedummz)
if server == 1:    
    path='/homedata/kwolf/40_era_iagos/'
    files = Path(path).glob(filedummz)

#--sort all the files
files = sorted(files)

out_arr   = np.zeros((25,0))
pres_out  = np.zeros((0))
t_out     = np.zeros((0))
r_out     = np.zeros((0))
r_out_ice = np.zeros((0))
wspd_out  = np.zeros((0))
wndr_out  = np.zeros((0))

outfile2.write('===============================\n')
outfile2.write('1h Timestepfiles \n')

for f in files:
    print('###############################')
    print('Read the extracted data from:!')
    print(f)
    print('###############################')
    time.sleep(1)
    dummy     = np.load(f,allow_pickle=True)
    out_arr   = np.append(out_arr, np.asarray(dummy['arr_0']),axis=1)#iagos out arr
    pres      = np.asarray(dummy['arr_1'])  #era tempf
    pres_out  = np.append(pres_out, np.asarray(dummy['arr_2'])) #era tempf
    t_out     = np.append(t_out, np.asarray(dummy['arr_3'])) #era tempf
    r_out     = np.append(r_out, np.asarray(dummy['arr_4'])) #era tempf
    r_out_ice = np.append(r_out_ice, np.asarray(dummy['arr_5'])) #era tempf
    wspd_out  = np.append(wspd_out, np.asarray(dummy['arr_6'])) #era tempf
    wndr_out  = np.append(wndr_out, np.asarray(dummy['arr_7'])) #era tempf

    #write diagnose to file
    
    outfile2.write('Reading file: '+str(f)+'\n')

#--get only every 4th step
out_arr   = out_arr[:,:]
pres      = pres
pres_out  = pres_out[:]
t_out     = t_out[:]
r_out     = r_out[:]
r_out_ice = r_out_ice[:]
wspd_out  = wspd_out[:]
wndr_out  = wndr_out[:]



#%%
#--1 hour time step data for cloud fraction
filedummz = year_to_read+'_era_1h_cc.npz'
if server == 0:
    path = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    files = Path(path).glob(filedummz)
if server == 1:    
    path='/homedata/kwolf/40_era_iagos/'
    files = Path(path).glob(filedummz)

#--sort all the files
files = sorted(files)

out_arr_cc   = np.zeros((25,0))
pres_out_cc  = np.zeros((0))
t_out_cc     = np.zeros((0))
r_out_cc     = np.zeros((0))
r_out_ice_cc = np.zeros((0))
cc_out  = np.zeros((0))

outfile2.write('===============================\n')
outfile2.write('1h Timestepfiles \n')

for f in files:
    print('###############################')
    print('Read the extracted data from:!')
    print(f)
    print('###############################')
    time.sleep(1)
    dummy     = np.load(f,allow_pickle=True)
    out_arr_cc   = np.append(out_arr_cc, np.asarray(dummy['arr_0']),axis=1)#iagos out arr
    pres_cc      = np.asarray(dummy['arr_1'])  #era tempf
    pres_out_cc  = np.append(pres_out_cc, np.asarray(dummy['arr_2'])) #era tempf
    t_out_cc     = np.append(t_out_cc, np.asarray(dummy['arr_3'])) #era tempf
    r_out_cc     = np.append(r_out_cc, np.asarray(dummy['arr_4'])) #era tempf
    r_out_ice_cc = np.append(r_out_ice_cc, np.asarray(dummy['arr_5'])) #era tempf
    cc_out  = np.append(cc_out, np.asarray(dummy['arr_6'])) #era tempf

    #write diagnose to file    
    outfile2.write('Reading file: '+str(f)+'\n')


#%%
#--1 hour time step data smoothed
filedummz = year_to_read+'_era_1h_smooth.npz'
if server == 0:
    path = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    files = Path(path).glob(filedummz)
if server == 1:    
    path='/homedata/kwolf/40_era_iagos/'
    files = Path(path).glob(filedummz)

#--sort all the files
files = sorted(files)

out_arr_smooth   = np.zeros((25,0))
pres_out_smooth  = np.zeros((0))
t_out_smooth     = np.zeros((0))
r_out_smooth     = np.zeros((0))
r_out_ice_smooth = np.zeros((0))
wspd_out_smooth  = np.zeros((0))
wndr_out_smooth  = np.zeros((0))

outfile2.write('===============================\n')
outfile2.write('1h Timestepfiles smoothed\n')

for f in files:
    print('###############################')
    print('Read the extracted data from:!')
    print(f)
    print('###############################')
    time.sleep(1)
    dummy     = np.load(f,allow_pickle=True)
    out_arr_smooth   = np.append(out_arr_smooth, np.asarray(dummy['arr_0']),axis=1)#iagos out arr
    pres_smooth      = np.asarray(dummy['arr_1'])  #era tempf
    pres_out_smooth  = np.append(pres_out_smooth, np.asarray(dummy['arr_2'])) #era tempf
    t_out_smooth     = np.append(t_out_smooth, np.asarray(dummy['arr_3'])) #era tempf
    r_out_smooth     = np.append(r_out_smooth, np.asarray(dummy['arr_4'])) #era tempf
    r_out_ice_smooth = np.append(r_out_ice_smooth, np.asarray(dummy['arr_5'])) #era tempf
    wspd_out_smooth  = np.append(wspd_out_smooth, np.asarray(dummy['arr_6'])) #era tempf
    wndr_out_smooth  = np.append(wndr_out_smooth, np.asarray(dummy['arr_7'])) #era tempf

    outfile2.write('Reading file: '+str(f)+'\n')

print(out_arr.shape)
print(out_arr_smooth.shape)



#%%
#--read the 3 hour ensemble spread
filedummz = year_to_read+'_era_3h_spread.npz'
if server == 0:
    path = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    files = Path(path).glob(filedummz)
if server == 1:    
    path='/homedata/kwolf/40_era_iagos/'
    files = Path(path).glob(filedummz)

#--sort all the files
files = sorted(files)

out_arr_3h   = np.zeros((25,0))  #--keep only year month day as I do have the rest already
pres_out_3h  = np.zeros((0))
t_out_3h     = np.zeros((0))
r_out_3h     = np.zeros((0))
wspd_out_3h  = np.zeros((0))
wndr_out_3h  = np.zeros((0))

outfile2.write('===============================\n')
outfile2.write('3h Spread Timestepfiles \n')

for f in files:
    print('###############################')
    print('Read the extracted data from:!')
    print(f)
    print('###############################')
    time.sleep(1)
    dummy        = np.load(f,allow_pickle=True)
    out_arr_3h   = np.append(out_arr_3h, np.asarray(dummy['arr_0']),axis=1)#iagos out arr
    pres_3h      = np.asarray(dummy['arr_1'])  #era tempf
    pres_out_3h  = np.append(pres_out_3h, np.asarray(dummy['arr_2'])) #era tempf
    t_out_3h     = np.append(t_out_3h, np.asarray(dummy['arr_3'])) #era tempf
    r_out_3h     = np.append(r_out_3h, np.asarray(dummy['arr_4'])) #era tempf
    wspd_out_3h  = np.append(wspd_out_3h, np.asarray(dummy['arr_5'])) #era tempf
    wndr_out_3h  = np.append(wndr_out_3h, np.asarray(dummy['arr_6'])) #era tempf
    
    outfile2.write('Reading file: '+str(f)+'\n')

#--get only every 4th step
out_arr_3h_spread   = out_arr_3h[:,:]
pres_3h_spread      = pres_3h
pres_out_3h_spread  = pres_out_3h[:]
t_out_3h_spread     = t_out_3h[:]
r_out_3h_spread     = r_out_3h[:]
wspd_out_3h_spread  = wspd_out_3h[:]
wndr_out_3h_spread  = wndr_out_3h[:]


#%%
#--3 hour time step data
filedummz = year_to_read+'_era_3h.npz'
if server == 0:
    path = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    files = Path(path).glob(filedummz)
if server == 1:    
    path='/homedata/kwolf/40_era_iagos/'
    files = Path(path).glob(filedummz)

#--sort all the files
files = sorted(files)

out_arr_3h   = np.zeros((25,0))
pres_out_3h  = np.zeros((0))
t_out_3h     = np.zeros((0))
r_out_3h     = np.zeros((0))
r_out_ice_3h = np.zeros((0))
wspd_out_3h  = np.zeros((0))
wndr_out_3h  = np.zeros((0))

outfile2.write('===============================\n')
outfile2.write('3h Timestepfiles \n')

for f in files:
    print('###############################')
    print('Read the extracted data from:!')
    print(f)
    print('###############################')
    time.sleep(1)
    dummy        = np.load(f,allow_pickle=True)
    out_arr_3h   = np.append(out_arr_3h, np.asarray(dummy['arr_0']),axis=1)#iagos out arr
    pres_3h      = np.asarray(dummy['arr_1'])  #era tempf
    pres_out_3h  = np.append(pres_out_3h, np.asarray(dummy['arr_2'])) #era tempf
    t_out_3h     = np.append(t_out_3h, np.asarray(dummy['arr_3'])) #era tempf
    r_out_3h     = np.append(r_out_3h, np.asarray(dummy['arr_4'])) #era tempf
    r_out_ice_3h = np.append(r_out_ice_3h, np.asarray(dummy['arr_5'])) #era tempf
    wspd_out_3h  = np.append(wspd_out_3h, np.asarray(dummy['arr_6'])) #era tempf
    wndr_out_3h  = np.append(wndr_out_3h, np.asarray(dummy['arr_7'])) #era tempf
    
    outfile2.write('Reading file: '+str(f)+'\n')

#--get only every 4th step
out_arr_3h   = out_arr_3h[:,:]
pres_3h      = pres_3h
pres_out_3h  = pres_out_3h[:]
t_out_3h     = t_out_3h[:]
r_out_3h     = r_out_3h[:]
r_out_ice_3h = r_out_ice_3h[:]
wspd_out_3h  = wspd_out_3h[:]
wndr_out_3h  = wndr_out_3h[:]


#%%
#--6 hour time step data
filedummz = year_to_read+'_era_6h.npz'
if server == 0:
    path = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    files = Path(path).glob(filedummz)
if server == 1:    
    path='/homedata/kwolf/40_era_iagos/'
    files = Path(path).glob(filedummz)

#--sort all the files
files = sorted(files)

out_arr_6h = np.zeros((25,0))
pres_out_6h = np.zeros((0))
t_out_6h = np.zeros((0))
r_out_6h = np.zeros((0))
r_out_ice_6h = np.zeros((0))
wspd_out_6h = np.zeros((0))
wndr_out_6h = np.zeros((0))

outfile2.write('===============================\n')
outfile2.write('6h Timestepfiles \n')

for f in files:
    print('###############################')
    print('Read the extracted data from:!')
    print(f)
    print('###############################')
    time.sleep(1)

    dummy        = np.load(f,allow_pickle=True)
    out_arr_6h   = np.append(out_arr_6h, np.asarray(dummy['arr_0']),axis=1)#iagos out arr
    pres_6h      = np.asarray(dummy['arr_1'])  #era tempf
    pres_out_6h  = np.append(pres_out_6h, np.asarray(dummy['arr_2'])) #era tempf
    t_out_6h     = np.append(t_out_6h, np.asarray(dummy['arr_3'])) #era tempf
    r_out_6h     = np.append(r_out_6h, np.asarray(dummy['arr_4'])) #era tempf
    r_out_ice_6h = np.append(r_out_ice_6h, np.asarray(dummy['arr_5'])) #era tempf
    wspd_out_6h  = np.append(wspd_out_6h, np.asarray(dummy['arr_6'])) #era tempf
    wndr_out_6h  = np.append(wndr_out_6h, np.asarray(dummy['arr_7'])) #era tempf
    
    outfile2.write('Reading file: '+str(f)+'\n')

#--get only every 4th step
out_arr_6h   = out_arr_6h[:,:]
pres_6h      = pres_6h
pres_out_6h  = pres_out_6h[:]
t_out_6h     = t_out_6h[:]
r_out_6h     = r_out_6h[:]
r_out_ice_6h = r_out_ice_6h[:]
wspd_out_6h  = wspd_out_6h[:]
wndr_out_6h  = wndr_out_6h[:]

#%%
#--convert list do np float array
pres = np.asarray(pres,dtype=np.float64)
print('Pressure levels in the file: ',pres)
outfile2.write('Pressure levels in the files: '+str(pres)+'\n')

#--calculate pres brakes
#for individual alt levels
pres_diff = np.diff(pres)
pres_brak = np.append(pres[1:] - np.diff(pres[0:])/2,pres[-1])
print('Applied pressure brakets: ',pres_brak)
outfile2.write('Applied pressure brakets: '+str(pres_brak)+'\n')


#%%

print('Shapes of the input data before filtering for -178 178 32 68')
outfile2.write('Shapes of the input data before filtering for -178 178 32 68 \n')
print('=============================')
outfile2.write('=============================')
print(out_arr.shape)
print(out_arr_smooth.shape)
print(out_arr_3h_spread.shape)
print(out_arr_3h.shape)
print(out_arr_6h.shape)
print(out_arr_cc.shape)

outfile2.write(str(out_arr.shape)+'\n')
outfile2.write(str(out_arr_smooth.shape)+'\n')
outfile2.write(str(out_arr_3h_spread.shape)+'\n')
outfile2.write(str(out_arr_3h.shape)+'\n')
outfile2.write(str(out_arr_6h.shape)+'\n')
outfile2.write(str(out_arr_cc.shape)+'\n')




#%%
#--calculate the number of measurements points per level and per region

cords = [-105, 30, 32, 68]
foo =  (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 350) & (out_arr[4,:]/100 > 162.5)
F,x=plt.subplots(1,5,figsize=(25,6),squeeze=False)
#--for 175 hpa
cords = [-105, 30, 32,68]
foo_all = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 187.5) & (out_arr[4,:]/100 > 162.5)

print(np.nansum(foo_all))
cords = [-105, -65, 32,68]
foo_us = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 187.5) & (out_arr[4,:]/100 > 162.5)
print(np.nansum(foo_us) / np.nansum(foo_all))
us_ratio = np.nansum(foo_us) / np.nansum(foo_all)
cords = [-65, -5, 32,68]
foo_na = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 187.5) & (out_arr[4,:]/100 > 162.5)
print(np.nansum(foo_na) / np.nansum(foo_all))
na_ratio = np.nansum(foo_na) / np.nansum(foo_all)
cords = [-5, 30, 32,68]
foo_eu = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 187.5) & (out_arr[4,:]/100 > 162.5)
print(np.nansum(foo_eu) / np.nansum(foo_all))
eu_ratio = np.nansum(foo_eu) / np.nansum(foo_all)
x1=x[0,4].scatter(-3,0,color='red')
x[0,4].bar(1,us_ratio,0.5,0,label='US')
x[0,4].bar(3,na_ratio,0.5,0,label='NA')
x[0,4].bar(5,eu_ratio,0.5,0,label='EU')
x[0,4].set_xticklabels([])

outfile2.write('================================== \n')
outfile2.write('Total Number of smaples at 175 hPa:  '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:5.1f}')+' %\n')
outfile2.write('Fraction US: '+str(us_ratio)+' \n')
outfile2.write('Fraction EU: '+str(eu_ratio)+' \n')
outfile2.write('Fraction NA: '+str(na_ratio)+' \n')

x[0,4].set_xlim(0,6)
x[0,4].set_ylim(0,1)
x[0,4].tick_params(labelsize=20)
x[0,4].xaxis.set_tick_params(width=2,length=5)
x[0,4].yaxis.set_tick_params(width=2,length=5)
x[0,4].spines['top'].set_linewidth(1.5)
x[0,4].spines['left'].set_linewidth(1.5)
x[0,4].spines['right'].set_linewidth(1.5)
x[0,4].spines['bottom'].set_linewidth(1.5)
x[0,4].text(0.4,0.9,'N: '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:5.1f}')+ ' %',fontsize=18)
x[0,4].set_title('(e) 175 hPa',fontsize=20)


#--for 200 hpa
cords = [-105, 30, 32,68]
foo_all = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 212.5) & (out_arr[4,:]/100 > 187.5)
print(np.nansum(foo_all))
cords = [-105, -65, 32,68]
foo_us = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 212.5) & (out_arr[4,:]/100 > 187.5)
print(np.nansum(foo_us) / np.nansum(foo_all))
us_ratio = np.nansum(foo_us) / np.nansum(foo_all)
cords = [-65, -5, 32,68]
foo_na = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 212.5) & (out_arr[4,:]/100 > 187.5)
print(np.nansum(foo_na) / np.nansum(foo_all))
na_ratio = np.nansum(foo_na) / np.nansum(foo_all)
cords = [-5, 30, 32,68]
foo_eu = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 212.5) & (out_arr[4,:]/100 > 187.5)
print(np.nansum(foo_eu) / np.nansum(foo_all))
eu_ratio = np.nansum(foo_eu) / np.nansum(foo_all)
x1=x[0,3].scatter(-3,0,color='red')
x[0,3].bar(1,us_ratio,0.5,0,label='US')
x[0,3].bar(3,na_ratio,0.5,0,label='NA')
x[0,3].bar(5,eu_ratio,0.5,0,label='EU')
x[0,3].set_xticklabels([])

outfile2.write('================================== \n')
outfile2.write('Total Number of smaples at 200 hPa:  '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:05.1f}')+' %\n')
outfile2.write('Fraction US: '+str(us_ratio)+' \n')
outfile2.write('Fraction EU: '+str(eu_ratio)+' \n')
outfile2.write('Fraction NA: '+str(na_ratio)+' \n')


x[0,3].set_xlim(0,6)
x[0,3].set_ylim(0,1)
x[0,3].tick_params(labelsize=20)
x[0,3].xaxis.set_tick_params(width=2,length=5)
x[0,3].yaxis.set_tick_params(width=2,length=5)
x[0,3].spines['top'].set_linewidth(1.5)
x[0,3].spines['left'].set_linewidth(1.5)
x[0,3].spines['right'].set_linewidth(1.5)
x[0,3].spines['bottom'].set_linewidth(1.5)
x[0,3].set_yticklabels([])

x[0,3].text(0.3,0.9,'N: '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:5.1f}')+ ' %',fontsize=18)
x[0,3].set_title('(d) 200 hPa',fontsize=20)

#--for 225 hpa
cords = [-105, 30, 32,68]
foo_all = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 237.5) & (out_arr[4,:]/100 > 212.5)
print(np.nansum(foo_all))
cords = [-105, -65, 32,68]
foo_us = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 237.5) & (out_arr[4,:]/100 > 212.5)
print(np.nansum(foo_us) / np.nansum(foo_all))
us_ratio = np.nansum(foo_us) / np.nansum(foo_all)
cords = [-65, -5, 32,68]
foo_na = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 237.5) & (out_arr[4,:]/100 > 212.5)
print(np.nansum(foo_na) / np.nansum(foo_all))
na_ratio = np.nansum(foo_na) / np.nansum(foo_all)
cords = [-5, 30, 32,68]
foo_eu = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 237.5) & (out_arr[4,:]/100 > 212.5)
print(np.nansum(foo_eu) / np.nansum(foo_all))
eu_ratio = np.nansum(foo_eu) / np.nansum(foo_all)
x1=x[0,2].scatter(-3,0,color='red')
x[0,2].bar(1,us_ratio,0.5,0,label='US')
x[0,2].bar(3,na_ratio,0.5,0,label='NA')
x[0,2].bar(5,eu_ratio,0.5,0,label='EU')
x[0,2].set_xticklabels([])


outfile2.write('================================== \n')
outfile2.write('Total Number of smaples at 225 hPa:  '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:05.1f}')+' %\n')
outfile2.write('Fraction US: '+str(us_ratio)+' \n')
outfile2.write('Fraction EU: '+str(eu_ratio)+' \n')
outfile2.write('Fraction NA: '+str(na_ratio)+' \n')



x[0,2].set_xlim(0,6)
x[0,2].set_ylim(0,1)
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].text(0.3,0.9,'N: '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:5.1f}')+ ' %',fontsize=18)
x[0,2].set_title('(c) 225 hPa',fontsize=20)
x[0,2].set_yticklabels([])

#--for 250 hpa
cords = [-105, 30, 32,68]
foo_all = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 275) & (out_arr[4,:]/100 > 237.5)
print(np.nansum(foo_all))
cords = [-105, -65, 32,68]
foo_us = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 275) & (out_arr[4,:]/100 > 237.5)
print(np.nansum(foo_us) / np.nansum(foo_all))
us_ratio = np.nansum(foo_us) / np.nansum(foo_all)
cords = [-65, -5, 32,68]
foo_na = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 275) & (out_arr[4,:]/100 > 237.5)
print(np.nansum(foo_na) / np.nansum(foo_all))
na_ratio = np.nansum(foo_na) / np.nansum(foo_all)
cords = [-5, 30, 32,68]
foo_eu = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 275) & (out_arr[4,:]/100 > 237.5)
print(np.nansum(foo_eu) / np.nansum(foo_all))
eu_ratio = np.nansum(foo_eu) / np.nansum(foo_all)
x1=x[0,1].scatter(-3,0,color='red')
x[0,1].bar(1,us_ratio,0.5,0,label='US')
x[0,1].bar(3,na_ratio,0.5,0,label='NA')
x[0,1].bar(5,eu_ratio,0.5,0,label='EU')
x[0,1].set_xticklabels([])

outfile2.write('================================== \n')
outfile2.write('Total Number of smaples at 250 hPa:  '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:05.1f}')+' %\n')
outfile2.write('Fraction US: '+str(us_ratio)+' \n')
outfile2.write('Fraction EU: '+str(eu_ratio)+' \n')
outfile2.write('Fraction NA: '+str(na_ratio)+' \n')


x[0,1].set_xlim(0,6)
x[0,1].set_ylim(0,1)
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].text(0.3,0.9,'N: '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:5.1f}')+ ' %',fontsize=18)
x[0,1].set_title('(b) 250 hPa',fontsize=20)
x[0,1].set_yticklabels([])

#--for 300 hpa
cords = [-105, 30, 32,68]
foo_all = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 325) & (out_arr[4,:]/100 > 275)
print(np.nansum(foo_all))
cords = [-105, -65, 32,68]
foo_us = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 325) & (out_arr[4,:]/100 > 275)
print(np.nansum(foo_us) / np.nansum(foo_all))
us_ratio = np.nansum(foo_us) / np.nansum(foo_all)
cords = [-65, -5, 32,68]
foo_na = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 325) & (out_arr[4,:]/100 > 275)
print(np.nansum(foo_na) / np.nansum(foo_all))
na_ratio = np.nansum(foo_na) / np.nansum(foo_all)
cords = [-5, 30, 32,68]
foo_eu = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 < 325) & (out_arr[4,:]/100 > 275)
print(np.nansum(foo_eu) / np.nansum(foo_all))
eu_ratio = np.nansum(foo_eu) / np.nansum(foo_all)
x1=x[0,0].scatter(-3,0,color='red')
x[0,0].bar(1,us_ratio,0.5,0,label='US')
x[0,0].bar(3,na_ratio,0.5,0,label='NA')
x[0,0].bar(5,eu_ratio,0.5,0,label='EU')
x[0,0].set_xticklabels([])

outfile2.write('================================== \n')
outfile2.write('Total Number of smaples at 300 hPa:  '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:05.1f}')+' %\n')
outfile2.write('Fraction US: '+str(us_ratio)+' \n')
outfile2.write('Fraction EU: '+str(eu_ratio)+' \n')
outfile2.write('Fraction NA: '+str(na_ratio)+' \n')

x[0,0].set_xlim(0,7)
x[0,0].set_ylim(0,1)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('(a) 300 hPa',fontsize=20)
x[0,0].set_ylabel('Fraction of Samples',fontsize = 20)
x[0,0].text(0.3,0.9,'N: '+str(np.nansum(foo_all))+' '+str(f'{np.nansum(foo_all) / np.nansum(foo)*100:5.1f}')+ ' %',fontsize=18)
x[0,0].legend(shadow=True,fontsize=20,loc='center left')
filename = 'iagos_measurements_per_region_and_altitude.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()


#%%

flight_dist = 1

if flight_dist == 1:
    #--get the flight altitude distribution from iagos
    #-- for the entire region, for us, the atlantic, and the eu
    #--for pressure bins of 25 hpa
    
    p_range = np.flip(np.arange(100,400,20))
    
    fad_array = np.zeros((4,len(p_range))) # 1+3 regions; all, us, na, eu
    fad_quantiles = np.zeros((4,5)) # 1+3 regions, for 5 quantiles of 0.1,0.25,0.5,0.75,0.9
    
    print('#all')
    cords = [-105, 30, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) &\
                (p_range[p]+10 > out_arr[4]/100.) & (out_arr[4]/100. >= p_range[p]-10) )
        
        fad_array[0,p] = np.nansum(foo) / np.nansum(foo_all)
        fad_quantiles[0,:] = np.nanquantile(out_arr[4,foo_all]/100,(0.1,0.25,0.5,0.75,0.9))
    print('#us')
    cords = [-105, -60, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) &\
                (p_range[p]+10 > out_arr[4]/100.) & (out_arr[4]/100. >= p_range[p]-10) )
        
        fad_array[1,p] = np.nansum(foo) / np.nansum(foo_all)
        fad_quantiles[1,:] = np.nanquantile(out_arr[4,foo_all]/100,(0.1,0.25,0.5,0.75,0.9))
    print('#na')
    cords = [-60, -5, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) &\
                (p_range[p]+10 > out_arr[4]/100.) & (out_arr[4]/100. >= p_range[p]-10) )
        
        fad_array[2,p] = np.nansum(foo) / np.nansum(foo_all)
        fad_quantiles[2,:] = np.nanquantile(out_arr[4,foo_all]/100,(0.1,0.25,0.5,0.75,0.9))
    print('#eu')
    cords = [-5, 30, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) &\
                (p_range[p]+10 > out_arr[4]/100.) & (out_arr[4]/100. >= p_range[p]-10) )
        
        fad_array[3,p] = np.nansum(foo) / np.nansum(foo_all)
        fad_quantiles[3,:] = np.nanquantile(out_arr[4,foo_all]/100,(0.1,0.25,0.5,0.75,0.9))
  
        
    #############                                                                                                                                                                             
    #save iagos flight altitude distributions
    filename='iagos_flight_altitude_distributions'
    print('iagos_flight_altitude_distributions')                                                                                                                                                                  
    if server == 0:                                                                                                                                                                           
        save_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                                                                                                                                                                                                              
    if server == 1:                                                                                                                                                                           
        save_stats_file = ('/homedata/kwolf/40_era_iagos/'+filename)                                                                                                                     
    print('saved to: '+str(save_stats_file))                                                                                                                                                  
    np.savez(save_stats_file,fad_array,p_range,fad_quantiles) 
    
    ###################################
    
    #--get the flight latitude distribution from iagos
    #-- for the entire region, for us, the atlantic, and the eu
    #--for lat bins of 5 deg
    
    lat_range = np.arange(20,90,5)
    
    flatd_array = np.zeros((4,len(lat_range))) # 1+3 regions; all, us, na, eu
    
    print('#all')
    cords = [-105, 30, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > lat_range[p]-2.5) & (out_arr[1,:] <= lat_range[p]+2.5))
        flatd_array[0,p] = np.nansum(foo) / np.nansum(foo_all)
    
    print('#us')
    cords = [-105, -60, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > lat_range[p]-2.5) & (out_arr[1,:] <= lat_range[p]+2.5))
        flatd_array[1,p] = np.nansum(foo) / np.nansum(foo_all)
        
    print('#na')
    cords = [-60, -5, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > lat_range[p]-2.5) & (out_arr[1,:] <= lat_range[p]+2.5))
        flatd_array[2,p] = np.nansum(foo) / np.nansum(foo_all)
        
    print('#eu')
    cords = [-5, 30, 32, 68]
    for p in np.arange(0,len(p_range)-1):
        foo_all =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]))
        foo =  ((out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) &\
                (out_arr[1,:] > lat_range[p]-2.5) & (out_arr[1,:] <= lat_range[p]+2.5))
        flatd_array[3,p] = np.nansum(foo) / np.nansum(foo_all)
    
  
        
    #############                                                                                                                                                                             
    #save iagos flight altitude distributions
    filename='iagos_flight_latitude_distributions'
    print('iagos_flight_latitude_distributions')                                                                                                                                                                  
    if server == 0:                                                                                                                                                                           
        save_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                                                                                                                                                                                                              
    if server == 1:                                                                                                                                                                           
        save_stats_file = ('/homedata/kwolf/40_era_iagos/'+filename)                                                                                                                     
    print('saved to: '+str(save_stats_file))                                                                                                                                                  
    np.savez(save_stats_file,flatd_array,lat_range)
    
#%%
################################################################################
#use only values on pressure levels 200  and 250
cords = [-110, 30, 30, 70]
foo =  (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1]) & (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3]) & (out_arr[4,:]/100 <= 250 + 25) & (out_arr[4,:]/100 > 175 + 12.5) 
out_arr = out_arr[:,foo]
pres_out = pres_out[foo]
t_out = t_out[foo]
r_out = r_out[foo]
r_out_ice = r_out_ice[foo]
wspd_out = wspd_out[foo]
wndr_out = wndr_out[foo]

#--cloud fraction data; same resolution as 1h data
cc_out = cc_out[foo]

#--smoothed data
out_arr_smooth = out_arr_smooth[:,foo]
pres_out_smooth = pres_out_smooth[foo]
t_out_smooth = t_out_smooth[foo]
r_out_smooth = r_out_smooth[foo]
r_out_ice_smooth = r_out_ice_smooth[foo]
wspd_out_smooth = wspd_out_smooth[foo]
wndr_out_smooth = wndr_out_smooth[foo]

#--ensemble spread
out_arr_3h_spread = out_arr_3h_spread[:,foo]
pres_out_3h_spread = pres_out_3h_spread[foo]
t_out_3h_spread = t_out_3h_spread[foo]
r_out_3h_spread = r_out_3h_spread[foo]
wspd_out_3h_spread = wspd_out_3h_spread[foo]
wndr_out_3h_spread = wndr_out_3h_spread[foo]


#& (r_out_ice_3h >= 5) # iagos is in pa
out_arr_3h = out_arr_3h[:,foo]
pres_out_3h = pres_out_3h[foo]
t_out_3h = t_out_3h[foo]
r_out_3h = r_out_3h[foo]
r_out_ice_3h = r_out_ice_3h[foo]
wspd_out_3h = wspd_out_3h[foo]
wndr_out_3h = wndr_out_3h[foo]

out_arr_6h = out_arr_6h[:,foo]
pres_out_6h = pres_out_6h[foo]
t_out_6h = t_out_6h[foo]
r_out_6h = r_out_6h[foo]
r_out_ice_6h = r_out_ice_6h[foo]
wspd_out_6h = wspd_out_6h[foo]
wndr_out_6h = wndr_out_6h[foo]


#%%
print('Shapes of the input data after filtering for -110 30 30 70')
outfile2.write('Shapes of the input data after filtering for -110 30 30 70 \n')
print('=============================')
outfile2.write('=============================')
print(out_arr.shape)
print(out_arr_smooth.shape)
print(out_arr_3h_spread.shape)
print(out_arr_3h.shape)
print(out_arr_6h.shape)
print(cc_out.shape)

outfile2.write(str(out_arr.shape)+'\n')
outfile2.write(str(out_arr_smooth.shape)+'\n')
outfile2.write(str(out_arr_3h_spread.shape)+'\n')
outfile2.write(str(out_arr_3h.shape)+'\n')
outfile2.write(str(out_arr_6h.shape)+'\n')
outfile2.write(str(cc_out.shape)+'\n')

outfile2.write('\n')

#%%

print('Apply correction')

#--read the cdf file
if server == 0:
    File = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/cdf_distributions.npz'
if server ==1:
    File = '/homedata/kwolf/40_era_iagos/cdf_distributions.npz'

dummy = np.load(File,allow_pickle=True)


cdf_iagos_r = np.asarray(dummy['arr_0'])  #cdf of r iagos size: alt,[0,180]
cdf_era_r = np.asarray(dummy['arr_1'])  #cdf of r era

cdf_iagos_t = np.asarray(dummy['arr_2'])  #cdf of t iagos size: alt,[190,273,0.2]
cdf_era_t = np.asarray(dummy['arr_3'])  #cdf of t era

cdf_iagos_r_2d = np.asarray(dummy['arr_4'])  #cdf of r iagos size: alt,[0,180]
cdf_era_r_2d = np.asarray(dummy['arr_5'])  #cdf of r era

r_2d_tbins = np.asarray(dummy['arr_6'])  # variable temperature bins for 10% percentiles

r_2d_tlat_bins = np.asarray(dummy['arr_7'])  # variable latitude bins deoending on pressure level [8 levels, 4 temps]



#--initialize empty array with size of the input
r_out_ice_cor = np.zeros((r_out_ice.shape))
r_out_ice_cor2d = np.zeros((r_out_ice.shape))
t_out_cor = np.zeros((t_out.shape))

#--set to nan
r_out_ice_cor[:] = np.nan
r_out_ice_cor2d[:] = np.nan
t_out_cor[:] = np.nan



#%%

outfile2.write('###################### \n')
outfile2.write('Results of the correction\n')
outfile2.write(' \n')

outfile2.write('1D Temprature and rh correction \n')
for pl in np.arange(0,len(pres)-1):
#for pl in np.arange(17,18):
    print('Processing p level : '+str(pres[pl+1]))
    print('with p brakets:'+str(pres_brak[pl])+' and '+str(pres_brak[pl+1]))
    
    outfile2.write('Processing p level : '+str(pres[pl+1])+'\n')
    outfile2.write('with p brakets:'+str(pres_brak[pl])+' and '+str(pres_brak[pl+1])+'\n')
    
    
    #--loop over the regions
    for r in np.arange(0,r_2d_tlat_bins.shape[1]-1):
        outfile2.write('Region: '+str(r)+'\n')
        outfile2.write('Temperature correction \n')
        
    
        altfoo = (out_arr[4,:]/100 < pres_brak[pl]) & (out_arr[4,:]/100 > pres_brak[pl+1]) & (out_arr[1,:] > r_2d_tlat_bins[pl,r]) & (out_arr[1,:] < r_2d_tlat_bins[pl,r+1])
        
        if (np.nansum(altfoo) !=0):
            #1d correction of temp
            ######################
            score = np.interp(t_out[altfoo],np.linspace(190.,273.,415),cdf_era_t[pl,r,:],left=0,right=1)
            scoreiagos = np.interp(t_out[altfoo],np.linspace(190.,273.,415),cdf_iagos_t[pl,r,:],left=0,right=1)
            t_out_cor[altfoo] = np.interp(score,cdf_iagos_t[pl,r,:],np.linspace(190,273,415),left=190,right=273)
            
            
            
            print('Genreal warning: ')
            print('recovering old temp, where correction provides non-physical values')
            print('')
            
            foo3 = ((score < 0.02) | (score > 0.98) | (scoreiagos < 0.02) | (scoreiagos > 0.98))
            outfile2.write('Number of corrected temperature values repacled by origional due to score: '+str(np.nansum(foo3))+' '+'Ratio: %6.3f' % (np.nansum(foo3)/np.nansum(scoreiagos))+'\n')
            t_out_cor[altfoo][foo3] = t_out[altfoo][foo3]
            
            
            
            footer = ((np.abs(t_out_cor - t_out) > 5) | ((t_out_cor < 100)) | (np.isnan(t_out_cor)))
            outfile2.write('Number of corrected temperature values repacled by origional due to nan or outside of bound: '+str(np.nansum(footer))+' '+'Ratio: %6.3f' %(np.nansum(footer)/np.nansum(scoreiagos))+'\n')
            t_out_cor[footer] = t_out[footer]
            
            #1d correction of rh
            ####################
            score = np.interp(r_out_ice[altfoo],np.arange(0,180,1),cdf_era_r[pl,r,:],left=0,right=1)
            r_out_ice_cor[altfoo] = np.interp(score,cdf_iagos_r[pl,r,:],np.arange(0,180,1),left=0,right=180)

        else:
            outfile2.write('There are no values to correct \n')
        
        outfile2.write(' \n')

outfile2.write('\n')
outfile2.write('2d rel hum correction\n')
outfile2.write('=====================\n')
outfile2.write('\n')

#--2d correction of rh
for pl in np.arange(0,len(pres)-1):
    print('Processing p level : '+str(pres[pl+1]))
    print('with p brakets:'+str(pres_brak[pl])+' and '+str(pres_brak[pl+1]))
    
    outfile2.write('Processing p level : '+str(pres[pl+1])+'\n')
    outfile2.write('with p brakets:'+str(pres_brak[pl])+' and '+str(pres_brak[pl+1])+'\n')
    
    #--loop over the regions
    for r in np.arange(0,r_2d_tlat_bins.shape[1]-1):
        #--loop over temperature
        for tl in np.arange(0,r_2d_tbins.shape[2]-1):
            outfile2.write('Region: '+str(r)+'\n')
            outfile2.write('Temperature bin: '+str(tl)+'\n')
        
            altfoo = (out_arr[4,:]/100 < pres_brak[pl]) & (out_arr[4,:]/100 > pres_brak[pl+1]) & (out_arr[1,:] > r_2d_tlat_bins[pl,r]) \
                & (out_arr[1,:] < r_2d_tlat_bins[pl,r+1]) & (out_arr[7,:] >= r_2d_tbins[pl,r,tl]) & (out_arr[7,:] < r_2d_tbins[pl,r,tl+1])  #--filter for pressure altitude, longitude, and tempreature bin

        
            tempocdfera = cdf_era_r_2d[pl,r,tl,:] #--use this one for the correction
        
            tempocdfiagos = cdf_iagos_r_2d[pl,r,tl,:] #--use this one for the correction


            score = np.interp(r_out_ice[altfoo],np.arange(0,180,1),tempocdfera,left=0,right=1)
            scoreiagos2 = np.interp(r_out_ice[altfoo],np.arange(0,180,1),tempocdfiagos,left=0,right=1)

            r_out_ice_cor2d[altfoo] = np.interp(score,tempocdfiagos,np.arange(0,180,1),left=0,right=180)

            if np.all(np.isnan(r_out_ice_cor2d[altfoo])):
                print('All values are nan')
                outfile2.write('All values are nan \n')
            else:
                print('There are some nan values')
                print('original values with nan: ',np.nansum(np.isnan(r_out_ice[altfoo])))
                print('cor2d values  with nan: ',np.nansum(np.isnan(r_out_ice_cor2d[altfoo])))
                print('ratio: ',np.nansum(~np.isnan(r_out_ice[altfoo])) / np.nansum(~np.isnan(r_out_ice_cor2d[altfoo])))
                
                outfile2.write('There are some values\n')
                outfile2.write('original values: %10.0f \n' %(np.nansum(~np.isnan(r_out_ice[altfoo]))))
                outfile2.write('cor2d values: %10.0f \n' %(np.nansum(~np.isnan(r_out_ice_cor2d[altfoo]))))
                outfile2.write('ratio: %6.4f \n' %(np.nansum(~np.isnan(r_out_ice[altfoo])) / np.nansum(~np.isnan(r_out_ice_cor2d[altfoo]))))
        
            #--do not correct if era5 relaitve humidity is in a range where cdf is above 0.95
            score_rem = ((score < 0.02) | (score > 0.98) | (scoreiagos2 < 0.02) | (scoreiagos2 > 0.98))
            r_out_ice_cor2d[altfoo][score_rem] = r_out_ice_cor[altfoo][score_rem]
            outfile2.write('Number of QM-2d-corrected RH values repacled by origional due to score: '+str(np.nansum(score_rem))+' '+'Ratio: %6.3f' % (np.nansum(score_rem)/len(scoreiagos2))+'\n')


        print('Uncorrected hum: ',r_out[altfoo])
        print('1D corrected hum: ',r_out_ice_cor[altfoo])
        print('2D corrected hum: ', r_out_ice_cor2d[altfoo])

    outfile2.write('\n')



#%%
#--Teoh et al 2022 correction
r_out_ice_teoh = np.zeros((r_out.shape))
r_out_ice_teoh[:] = np.nan

#--parameters
a = 0.9779
b = 1.635

lindex = (r_out_ice*0.01 / a < 1)
r_out_ice_teoh[lindex] = r_out_ice[lindex] / a
lindex = (r_out_ice*0.01 / a >= 1)
r_out_ice_teoh[lindex] = (r_out_ice[lindex]*0.01 / a)**b*100
lindex = (r_out_ice_teoh*0.01 >= 1.65)
r_out_ice_teoh[lindex] = 1.65*100


#%%
print('Before filtering find the differences between iagos and era5 for the different regions')

outfile2.write('\n')
outfile2.write('This is from the data belonging to the plot with col1 Temp, col2 Temp diff, col3 rh, and col 4 rh diff for the corrections')


def calc_pdfs_t(cords, iagos_in, era_in, alt_min, alt_max, min_bin, max_bin, steps_bin):
    altfoo = (iagos_in[4,:]/100 < alt_min) & (iagos_in[4,:]/100 > alt_max)
    iagos_in = iagos_in[:,altfoo]
    era_in = era_in[altfoo]
    foo = (iagos_in[0,:] > cords[0]) & (iagos_in[0,:] < cords[1])  #Lon
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    foo = (iagos_in[1,:] > cords[2]) & (iagos_in[1,:] < cords[3])  # Lat
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    xxx,yyy = my_histogram(era_in.flatten(),min_bin,max_bin,steps_bin)
    print_mean = np.nanmean(era_in)
    print_median = np.nanmedian(era_in) 
    return(xxx, yyy, print_mean, print_median)

def calc_diff_pdfs_t(cords, iagos_in, era_in, alt_min, alt_max, min_bin, max_bin, steps_bin):
    altfoo = (iagos_in[4,:]/100 < alt_min) & (iagos_in[4,:]/100 > alt_max)
    iagos_in = iagos_in[:,altfoo]
    era_in = era_in[altfoo]
    foo = (iagos_in[0,:] > cords[0]) & (iagos_in[0,:] < cords[1])  #Lon
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    foo = (iagos_in[1,:] > cords[2]) & (iagos_in[1,:] < cords[3])  # Lat
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    xxx,yyy = my_histogram((era_in - iagos_in[7,:]).flatten(),min_bin,max_bin,steps_bin)
    print_mean = np.nanmean(era_in - iagos_in[7,:])
    print_median = np.nanmedian(era_in - iagos_in[7,:]) 
    return(xxx, yyy, print_mean, print_median)


def calc_pdfs_r(cords, iagos_in, era_in, alt_min, alt_max, min_bin, max_bin, steps_bin):
    altfoo = (iagos_in[4,:]/100 < alt_min) & (iagos_in[4,:]/100 > alt_max)
    iagos_in = iagos_in[:,altfoo]
    era_in = era_in[altfoo]
    foo = (iagos_in[0,:] > cords[0]) & (iagos_in[0,:] < cords[1])  #Lon
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    foo = (iagos_in[1,:] > cords[2]) & (iagos_in[1,:] < cords[3])  # Lat
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    xxx,yyy = my_histogram(era_in.flatten(),min_bin,max_bin,steps_bin)
    print_mean = np.nanmean(era_in)
    print_median = np.nanmedian(era_in) 
    return(xxx, yyy, print_mean, print_median)

def calc_diff_pdfs_r(cords, iagos_in, era_in, alt_min, alt_max, min_bin, max_bin, steps_bin):
    altfoo = (iagos_in[4,:]/100 < alt_min) & (iagos_in[4,:]/100 > alt_max)
    iagos_in = iagos_in[:,altfoo]
    era_in = era_in[altfoo]
    foo = (iagos_in[0,:] > cords[0]) & (iagos_in[0,:] < cords[1])  #Lon
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    foo = (iagos_in[1,:] > cords[2]) & (iagos_in[1,:] < cords[3])  # Lat
    iagos_in = iagos_in[:,foo]
    era_in = era_in[foo]
    xxx,yyy = my_histogram(era_in - (iagos_in[13,:]).flatten(),min_bin,max_bin,steps_bin)
    print_mean = np.nanmean(era_in - iagos_in[13,:])
    print_median = np.nanmedian(era_in - iagos_in[13,:]) 
    return(xxx, yyy, print_mean, print_median)

outfile2.write('==================\n')
outfile2.write('Temperature \n')

#--plot iagos measurement distribution
F,x=plt.subplots(3,4,figsize=(20,10),squeeze=False)
#########################
x1=x[0,0].plot(0,0)
cords = [-110, 30, 30, 70] #--all available

#--labels for legend
x[0,0].plot(0,0,color='k',label='IAGOS')
x[0,0].plot(0,0,color='r',label='ERA')
x[0,0].plot(0,0,color='b',label='ERA QM')
x[0,0].plot(0,0,color='orange',label='ERA T22')


xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, t_out, 212.5, 187.5, 190, 250, 1)
print('===')
print('Full domain ERA 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='red',zorder=3)
x[0,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, t_out_cor, 212.5, 187.5, 190, 250, 1)
print('===')
print('Full domain ERA cor 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='blue',zorder=3)
x[0,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)


xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, out_arr[7,:], 212.5, 187.5, 190, 250, 1)
print('===')
print('Full domain IAGOS 200hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain IAGOS 200hpa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='k',zorder=3)
x[0,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='k',zorder=3)


x[0,0].fill_between((print_mean+0.5,print_mean+0.5,print_mean-0.5,print_mean-0.5),(0,0.2,0.2,0),color='gray',alpha=0.3)
x[0,0].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[0,0].set_xlim(200,240)
x[0,0].set_ylim(0.,0.18)
x[0,0].set_xticklabels([])
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_ylabel('Level 200 hPa \n PDF',fontsize = 20)
x[0,0].text(233,0.15,'(a)',fontsize=20)
x[0,0].legend(fontsize=12,shadow=True,loc='upper left')

x2=x[1,0].plot(0,0)
cords = [-105, 30, 40, 60] # all available
xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, t_out, 237.5, 212.5, 190, 250, 1)
print('===')
print('Full domain ERA 225 hPA')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[1,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, t_out_cor, 237.5, 212.5, 190, 250, 1)
print('===')
print('Full domain ERA cor 225hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[1,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, out_arr[7,:], 237.5, 212.5, 190, 250, 1)
print('===')
print('Full domain IAGOS 225 hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain IAGOS 225hpa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='IAGOS all',color='k',zorder=3)
x[1,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='k',zorder=3)


x[1,0].fill_between((print_mean+0.5,print_mean+0.5,print_mean-0.5,print_mean-0.5),(0,0.2,0.2,0),color='gray',alpha=0.3)
x[1,0].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[1,0].set_xlim(200,240)
x[1,0].set_ylim(0.,0.18)
x[1,0].set_xticklabels([])
x[1,0].tick_params(labelsize=20)
x[1,0].xaxis.set_tick_params(width=2,length=5)
x[1,0].yaxis.set_tick_params(width=2,length=5)
x[1,0].spines['top'].set_linewidth(1.5)
x[1,0].spines['left'].set_linewidth(1.5)
x[1,0].spines['right'].set_linewidth(1.5)
x[1,0].spines['bottom'].set_linewidth(1.5)
x[1,0].set_ylabel('Level 225 hPa \n PDF',fontsize = 20)
x[1,0].text(233,0.15,'(e)',fontsize=20)




#--

x3=x[2,0].plot(0,0)
cords = [-105, 30, 40, 60] # all available
xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, t_out, 275, 237.5, 190, 250, 1)
print('===')
print('Full domain ERA 250 hPA')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 250 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[2,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, t_out_cor, 275, 237.5, 190, 250, 1)
print('===')
print('Full domain ERA cor 250')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 250 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[2,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_t(cords, out_arr, out_arr[7,:], 275, 237.5, 190, 250, 1)
print('===')
print('Full domain IAGOS 250 hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain IAGOS 250\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='IAGOS all',color='k',zorder=3)
x[2,0].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='k',zorder=3)


x[2,0].fill_between((print_mean+0.5,print_mean+0.5,print_mean-0.5,print_mean-0.5),(0,0.2,0.2,0),color='gray',alpha=0.3)

x[2,0].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[2,0].set_xlim(200,240)
x[2,0].set_ylim(0.,0.18)
x[2,0].tick_params(labelsize=20)
x[2,0].xaxis.set_tick_params(width=2,length=5)
x[2,0].yaxis.set_tick_params(width=2,length=5)
x[2,0].spines['top'].set_linewidth(1.5)
x[2,0].spines['left'].set_linewidth(1.5)
x[2,0].spines['right'].set_linewidth(1.5)
x[2,0].spines['bottom'].set_linewidth(1.5)
x[2,0].set_ylabel('Level 250 hPa \n PDF',fontsize = 20)
x[2,0].text(233,0.15,'(i)',fontsize=20)
x[2,0].set_xlabel(r'T [K]',fontsize = 20)




outfile2.write('===\n')
outfile2.write('Diferences\n')
outfile2.write('===\n')


#########################
x2=x[0,1].plot(0,0)
cords = [-105, 30, 40, 60] # all available

x[0,1].fill_between((0.5,0.5,-0.5,-0.5),(0,0.2,0.2,0),color='gray',alpha=0.3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_t(cords, out_arr, t_out, 212.5, 187.5, -5, 5, 0.2)
print('===')
print('Full domain ERA 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[0,1].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_t(cords, out_arr, t_out_cor, 212.5, 187.5, -5, 5, 0.2)
print('===')
print('Full domain ERA cor 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[0,1].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)


x[0,1].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[0,1].set_xlim(-5,5)
x[0,1].set_ylim(0.,0.18)
x[0,1].set_xticklabels([])
x[0,1].set_yticklabels([])
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].text(3.2,0.15,'(b)',fontsize=20)






#########################
x2=x[1,1].plot(0,0)
cords = [-105, 30, 40, 60] # all available

x[1,1].fill_between((0.5,0.5,-0.5,-0.5),(0,0.2,0.2,0),color='gray',alpha=0.3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_t(cords, out_arr, t_out, 237.5, 212.5, -5, 5, 0.2)
print('===')
print('Full domain ERA 225')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[1,1].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_t(cords, out_arr, t_out_cor, 237.5, 212.5, -5, 5, 0.2)
print('===')
print('Full domain ERA cor 225')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[1,1].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

x[1,1].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[1,1].set_xlim(-5,5)
x[1,1].set_ylim(0.,0.18)
x[1,1].set_xticklabels([])
x[1,1].set_yticklabels([])
x[1,1].tick_params(labelsize=20)
x[1,1].xaxis.set_tick_params(width=2,length=5)
x[1,1].yaxis.set_tick_params(width=2,length=5)
x[1,1].spines['top'].set_linewidth(1.5)
x[1,1].spines['left'].set_linewidth(1.5)
x[1,1].spines['right'].set_linewidth(1.5)
x[1,1].spines['bottom'].set_linewidth(1.5)
x[1,1].text(3.2,0.15,'(f)',fontsize=20)


#########################
x2=x[2,1].plot(0,0)

x[2,1].fill_between((0.5,0.5,-0.5,-0.5),(0,0.2,0.2,0),color='gray',alpha=0.3)

cords = [-105, 30, 40, 60] # all available
xxx,yyy,print_mean,print_median = calc_diff_pdfs_t(cords, out_arr, t_out, 275, 237.5, -5, 5, 0.2)
print('===')
print('Full domain ERA 225')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[2,1].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_t(cords, out_arr, t_out_cor, 275, 237.5, -5, 5, 0.2)
print('===')
print('Full domain ERA cor 225')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[2,1].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

x[2,1].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[2,1].set_xlim(-5,5)
x[2,1].set_ylim(0.,0.18)
x[2,1].set_yticklabels([])
x[2,1].tick_params(labelsize=20)
x[2,1].xaxis.set_tick_params(width=2,length=5)
x[2,1].yaxis.set_tick_params(width=2,length=5)
x[2,1].spines['top'].set_linewidth(1.5)
x[2,1].spines['left'].set_linewidth(1.5)
x[2,1].spines['right'].set_linewidth(1.5)
x[2,1].spines['bottom'].set_linewidth(1.5)
x[2,1].text(3.2,0.15,'(j)',fontsize=20)
x[2,1].set_xlabel(r'$\Delta$T [K]',fontsize = 20)


outfile2.write('===\n')
outfile2.write('relative humidity')
outfile2.write('===\n')


#########################
x1=x[0,2].plot(0,0)
cords = [-110, 30, 30, 70] # all available
xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice, 212.5, 187.5, 0, 180, 5)
print('===')
print('Full domain ERA 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[0,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice_cor2d, 212.5, 187.5, 0, 180, 5)
print('===')
print('Full domain ERA cor 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[0,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice_teoh, 212.5, 187.5, 0, 180, 5)
print('===')
print('Full domain ERA teoh 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA teoh 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='orange',zorder=3)
x[0,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='orange',zorder=3)



xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, out_arr[13,:], 212.5, 187.5, 0, 180, 5)
print('===')
print('Full domain IAGOS 200hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain IAGOS 200hpa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='IAGOS all',color='k',zorder=3)
x[0,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='k',zorder=3)


x[0,2].fill_between((print_mean+10,print_mean+10,print_mean-10,print_mean-10),(0,0.2,0.2,0),color='gray',alpha=0.3)
x[0,2].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[0,2].set_xlim(0,180)
x[0,2].set_ylim(0.,0.18)
x[0,2].set_xticklabels([])
x[0,2].set_yticklabels([])
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].text(155,0.15,'(c)',fontsize=20)


#--

x2=x[1,2].plot(0,0)
cords = [-110, 30, 30, 70] # all available
xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice, 237.5, 212.5, 0, 180, 5)
print('===')
print('Full domain ERA 225 hPA')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[1,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice_cor2d, 237.5, 212.5, 0, 180, 5)
print('===')
print('Full domain ERA cor 225hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA core 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[1,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)


xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice_teoh,237.5, 212.5, 0, 180, 5)
print('===')
print('Full domain ERA teoh 225')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA teoh hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='orange',zorder=3)
x[1,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='orange',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, out_arr[13,:], 237.5, 212.5, 0, 180, 5)

x[1,2].fill_between((print_mean+10,print_mean+10,print_mean-10,print_mean-10),(0,0.2,0.2,0),color='gray',alpha=0.3)

print('===')
print('Full domain IAGOS 225 hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain IAGOS 225hpa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='IAGOS all',color='k',zorder=3)
x[1,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='k',zorder=3)

x[1,2].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[1,2].set_xlim(0,180)
x[1,2].set_ylim(0.,0.18)
x[1,2].set_xticklabels([])
x[1,2].set_yticklabels([])
x[1,2].tick_params(labelsize=20)
x[1,2].xaxis.set_tick_params(width=2,length=5)
x[1,2].yaxis.set_tick_params(width=2,length=5)
x[1,2].spines['top'].set_linewidth(1.5)
x[1,2].spines['left'].set_linewidth(1.5)
x[1,2].spines['right'].set_linewidth(1.5)
x[1,2].spines['bottom'].set_linewidth(1.5)
x[1,2].text(155,0.15,'(g)',fontsize=20)



#--

x3=x[2,2].plot(0,0)
cords = [-110, 30, 30, 70] # all available
xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice, 275, 237.5,  0, 180, 5)
print('===')
print('Full domain ERA 250 hPA')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 250 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[2,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice_cor2d, 275, 237.5,  0, 180, 5)
print('===')
print('Full domain ERA cor 250')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA cor 250 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[2,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, r_out_ice_teoh, 275, 237.5,  0, 180, 5)
print('===')
print('Full domain ERA teoh 250')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA teoh 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='orange',zorder=3)
x[2,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='orange',zorder=3)

xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr, out_arr[13,:], 275, 237.5,  0, 180, 5)

x[2,2].fill_between((print_mean+10,print_mean+10,print_mean-10,print_mean-10),(0,0.2,0.2,0),color='gray',alpha=0.3)

print('===')
print('Full domain IAGOS 250 hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain IAGOS 250\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,2].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='IAGOS all',color='k',zorder=3)
x[2,2].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='k',zorder=3)


x[2,2].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[2,2].set_xlim(0,180)
x[2,2].set_ylim(0.,0.18)
x[2,2].tick_params(labelsize=20)
x[2,2].set_yticklabels([])
x[2,2].xaxis.set_tick_params(width=2,length=5)
x[2,2].yaxis.set_tick_params(width=2,length=5)
x[2,2].spines['top'].set_linewidth(1.5)
x[2,2].spines['left'].set_linewidth(1.5)
x[2,2].spines['right'].set_linewidth(1.5)
x[2,2].spines['bottom'].set_linewidth(1.5)
x[2,2].text(155,0.15,'(k)',fontsize=20)
x[2,2].set_xlabel(r'rH$_i$ [%]',fontsize = 20)



#########################
x1=x[0,3].plot(0,0)
cords = [-110, 30, 30, 70] # all available
xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice, 212.5, 187.5, -40,40,2)
print('===')
print('Full domain ERA 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[0,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice_cor2d, 212.5, 187.5, -40,40,2)
print('===')
print('Full domain ERA cor 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[0,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice_teoh,  212.5, 187.5, -40,40,2)
print('===')
print('Full domain ERA teoh 200hPa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA teoh 200 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[0,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='orange',zorder=3)
x[0,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='orange',zorder=3)




x[0,3].fill_between((0+10,0+10,0-10,0-10),(0,0.2,0.2,0),color='gray',alpha=0.3)
x[0,3].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[0,3].set_xlim(-50,50)
x[0,3].set_ylim(0.,0.18)
x[0,3].set_xticklabels([])
x[0,3].set_yticklabels([])
x[0,3].tick_params(labelsize=20)
x[0,3].xaxis.set_tick_params(width=2,length=5)
x[0,3].yaxis.set_tick_params(width=2,length=5)
x[0,3].spines['top'].set_linewidth(1.5)
x[0,3].spines['left'].set_linewidth(1.5)
x[0,3].spines['right'].set_linewidth(1.5)
x[0,3].spines['bottom'].set_linewidth(1.5)
x[0,3].text(35,0.15,'(d)',fontsize=20)


#--

x2=x[1,3].plot(0,0)
cords = [-105, 30, 40, 60] # all available
xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice, 237.5, 212.5, -40,40,2)
print('===')
print('Full domain ERA 225 hPA')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[1,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice_cor2d, 237.5, 212.5, -40,40,2)
print('===')
print('Full domain ERA cor 225hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[1,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice_teoh, 237.5, 212.5, -40,40,2)
print('===')
print('Full domain ERA teoh 225hpa')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA teoh 225 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[1,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='orange',zorder=3)
x[1,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='orange',zorder=3)


x[1,3].fill_between((0+10,0+10,0-10,0-10),(0,0.2,0.2,0),color='gray',alpha=0.3)
x[1,3].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[1,3].set_xlim(-50,50)
x[1,3].set_ylim(0.,0.18)
x[1,3].set_xticklabels([])
x[1,3].set_yticklabels([])
x[1,3].tick_params(labelsize=20)
x[1,3].xaxis.set_tick_params(width=2,length=5)
x[1,3].yaxis.set_tick_params(width=2,length=5)
x[1,3].spines['top'].set_linewidth(1.5)
x[1,3].spines['left'].set_linewidth(1.5)
x[1,3].spines['right'].set_linewidth(1.5)
x[1,3].spines['bottom'].set_linewidth(1.5)
x[1,3].text(35,0.15,'(h)',fontsize=20)




#--

x3=x[2,3].plot(0,0)
cords = [-105, 30, 40, 60] # all available
xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice, 275, 237.5, -40,40,2)
print('===')
print('Full domain ERA 250 hPA')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 250 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='red',zorder=3)
x[2,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='red',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice_cor2d, 275, 237.5, -40,40,2)
print('===')
print('Full domain ERA cor 250')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA 250 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,3].plot(xxx[:-1], yyy, alpha=1,linewidth=2,label='All',color='blue',zorder=3)
x[2,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='blue',zorder=3)

xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr, r_out_ice_teoh,  275, 237.5, -40,40,2)
print('===')
print('Full domain ERA teoh 250')
print('Mean: ',print_mean)
print('Median: ',print_median)
outfile2.write('===\n')
outfile2.write('Full domain ERA teoh 250 hPa\n')
outfile2.write('Mean: '+str(print_mean)+'\n')
outfile2.write('Median: '+str(print_median)+'\n')
outfile2.write('\n')
x[2,3].plot((print_mean,print_mean),(0,0.20),linestyle='dashed',color='orange',zorder=3)



x[2,3].fill_between((0+10,0+10,0-10,0-10),(0,0.2,0.2,0),color='gray',alpha=0.3)
x[2,3].plot((0.0,0.0),(0,0.5),linestyle='dashed',color='k')
x[2,3].set_xlim(-50,50)
x[2,3].set_ylim(0.,0.18)
#x[2,3].set_xticklabels([])
x[2,3].set_yticklabels([])
x[2,3].tick_params(labelsize=20)
x[2,3].xaxis.set_tick_params(width=2,length=5)
x[2,3].yaxis.set_tick_params(width=2,length=5)
x[2,3].spines['top'].set_linewidth(1.5)
x[2,3].spines['left'].set_linewidth(1.5)
x[2,3].spines['right'].set_linewidth(1.5)
x[2,3].spines['bottom'].set_linewidth(1.5)
x[2,3].text(35,0.15,'(l)',fontsize=20)
x[2,3].set_xlabel(r'$\Delta$rH$_i$ [%]',fontsize = 20)


filename = 'pdf_temp_rh.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    #plt.close()
    plt.show()


#%%
#--separtaion with cloud cover from ERA5


outfile2.write('\n')
outfile2.write('Separating RH and RH diff for cloud fraction from ERA5 \n')
outfile2.write('DELTA RH \n')
outfile2.write('=======================================================\n')
outfile2.write('\n')


F,x=plt.subplots(5,2,figsize=(15,10),squeeze=False)
#########################

for i in np.arange(0,5):
    x1=x[i,1].plot(0,0)
    cld_f = ((cc_out >= 0.2*i) & (cc_out < 0.2*i+0.2))
    print('Cloud fraction braket from %3.1f < c <= %3.1f \n' % (0.2*i, 0.2*i+0.2 ))
    outfile2.write('Cloud fraction braket from %3.1f < c <= %3.1f \n' % (0.2*i, 0.2*i+0.2 ))
    
    #--get the fraction of in cloud measurements from iagos
    #--where larger than 0 and not non
    flg = (~np.isnan(out_arr[9,:]) & (out_arr[9,:] > 0) & (cc_out >= 0.2*i) & (cc_out < 0.2*i+0.2))
    flgcld = (~np.isnan(out_arr[9,:]) & (out_arr[9,:] > 0) & (out_arr[9,:] >= 0.015) & (cc_out >= 0.2*i) & (cc_out < 0.2*i+0.2))
    
    print('===')
    print('275 to 187.5 for 250 to 200 hPa')
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 275, 187.5, -80, 80,2)
    print('Mean ERA uncor: ',print_mean)
    print('Median ERA uncor: ',print_median)
    x[i,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='orange',zorder=3,linestyle='solid')
    x[i,1].plot((print_mean,print_mean),(0,1.0),linestyle='solid',color='orange',zorder=3,alpha=1,label='ERA')
    outfile2.write('=====\n')
    outfile2.write('275 to 187.5 for 250 to 200 hP a\n')
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 275, 187.5, -80, 80,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    x[i,1].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='blue',zorder=3,linestyle='solid')
    x[i,1].plot((print_mean,print_mean),(0,1.0),linestyle='solid',color='blue',zorder=3,alpha=1,label='ERA QM')
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)
    
    
    print('===')
    print('275 to 237.5 for 250 hPa')
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 275, 237.5, -80, 80,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('=====\n')
    outfile2.write('275 to 237.5 for 250 hPa \n')
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 275, 237.5, -80, 80,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)
    
    
    print('===')
    print('237.5 to 212.5 for 225 hPa')    
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 237.5, 212.5, -80, 80,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('=====\n')
    outfile2.write('237.5 to 212.5 for 225 hPa \n')
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 237.5, 212.5, -80, 80,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)
    
    
    print('===')
    print('212.5 to 187.5 for 200 hPa')
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 212.5, 187.5, -80, 80,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('=====\n')
    outfile2.write('212.5 to 187.5 for 200 hPa \n')
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_diff_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 212.5, 187.5, -80, 80,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)
    
    #--plot a zero line
    x[i,1].plot((0,0),(0,1),linestyle='dashed',color='k',linewidth=2)

    x[i,1].set_xlim(-80,80)
    x[i,1].set_ylim(0,0.15)
    x[i,1].tick_params(labelsize=20)
    x[i,1].xaxis.set_tick_params(width=2,length=5)
    x[i,1].yaxis.set_tick_params(width=2,length=5)
    x[i,1].spines['top'].set_linewidth(1.5)
    x[i,1].spines['left'].set_linewidth(1.5)
    x[i,1].spines['right'].set_linewidth(1.5)
    x[i,1].spines['bottom'].set_linewidth(1.5)
    x[i,1].text(-75,0.12,'(%s)' %  alpha_string[i*2+1],fontsize=18)
    x[i,1].text(85,0.12,'%3.1f >= c < %3.1f' %  (0.2*i, 0.2*i+0.2),fontsize=18)
    x[i,1].text(85,0.08,'N: %5.1f %%' %  (np.nansum(cld_f) / len(cc_out) * 100),fontsize=18)
    x[i,1].text(85,0.04,'In-cloud: %5.1f %%' %  (np.nansum(flgcld) / np.nansum(flg) * 100),fontsize=18)
    x[i,1].grid()
    
    
    if i != 4:
        x[i,1].set_xticklabels([])
    if i==4:
        x[i,1].set_xlabel(r'$\Delta rH_i$ [%]',fontsize = 20)
    #x[i,1].set_yticklabels([])
    


outfile2.write('\n')
outfile2.write('Separating RH and RH diff for cloud fraction from ERA5 \n')
outfile2.write('Absolute RH \n')
outfile2.write('=======================================================\n')
outfile2.write('\n')


for i in np.arange(0,5):
    x1=x[i,0].plot(0,0)
    cld_f = ((cc_out >= 0.2*i) & (cc_out < 0.2*i+0.2))
    print('Cloud fraction braket from %3.1f < c <= %3.1f \n' % (0.2*i, 0.2*i+0.2 ))
    outfile2.write('Cloud fraction braket from %3.1f < c <= %3.1f \n' % (0.2*i, 0.2*i+0.2 ))
    
    print('===')
    print('275 to 187.5 for 250 to 200 hPa')
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], out_arr[13,cld_f], 275, 187.5, 0, 140,2)
    x[i,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='k',linestyle='solid',label='IAGOS')
    x[i,0].plot((print_mean,print_mean),(0,1.0),linestyle='solid',color='k', alpha=1)
    outfile2.write('=====\n')
    outfile2.write('275 to 187.5 for 250 to 200 hP a\n')
    outfile2.write('Mean IAGOS: %6.4f \n' % print_mean)
    outfile2.write('Median IAGOS: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 275, 187.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    x[i,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='orange',zorder=3,linestyle='solid',label='ERA')
    x[i,0].plot((print_mean,print_mean),(0,1.0),linestyle='solid',color='orange',zorder=3)
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 275, 187.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    x[i,0].plot(xxx[:-1], yyy, alpha=1,linewidth=2,color='blue',zorder=3,linestyle='solid',label='ERA QM')
    x[i,0].plot((print_mean,print_mean),(0,1.0),linestyle='solid',color='blue',zorder=3)
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)
    
    
    print('===')
    print('275 to 237.5 for 250 hPa')
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], out_arr[13,cld_f], 275, 237.5, 0, 140,2)
    outfile2.write('=====\n')
    outfile2.write('275 to 237.5 for 250 hPa \n')
    outfile2.write('Mean IAGOS: %6.4f \n' % print_mean)
    outfile2.write('Median IAGOS: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 275, 237.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 275, 237.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)
    
    
    print('===')
    print('237.5 to 212.5 for 225 hPa')   
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], out_arr[13,cld_f], 273.5, 212.5, 0, 140,2)
    outfile2.write('=====\n')
    outfile2.write('237.5 to 212.5 for 225 hPa \n')
    outfile2.write('Mean IAGOS: %6.4f \n' % print_mean)
    outfile2.write('Median IAGOS: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 273.5, 212.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 273.5, 212.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)
    
    
    print('===')
    print('212.5 to 187.5 for 200 hPa')
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], out_arr[13,cld_f], 212.5, 187.5, 0, 140,2)
    outfile2.write('=====\n')
    outfile2.write('212.5 to 187.5 for 200 hPa \n')
    outfile2.write('Mean IAGOS: %6.4f \n' % print_mean)
    outfile2.write('Median IAGOS: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice[cld_f], 212.5, 187.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA uncor: %6.4f \n' % print_mean)
    outfile2.write('Median ERA uncor: %6.4f \n' % print_median)
    
    xxx,yyy,print_mean,print_median = calc_pdfs_r(cords, out_arr[:,cld_f], r_out_ice_cor2d[cld_f], 212.5, 187.5, 0, 140,2)
    print('Mean: ',print_mean)
    print('Median: ',print_median)
    outfile2.write('Mean ERA cor2d: %6.4f \n' % print_mean)
    outfile2.write('Median ERA cor2d: %6.4f \n' % print_median)


    x[i,0].set_xlim(0,140)
    x[i,0].set_ylim(0,0.5)
    x[i,0].tick_params(labelsize=20)
    x[i,0].xaxis.set_tick_params(width=2,length=5)
    x[i,0].yaxis.set_tick_params(width=2,length=5)
    x[i,0].spines['top'].set_linewidth(1.5)
    x[i,0].spines['left'].set_linewidth(1.5)
    x[i,0].spines['right'].set_linewidth(1.5)
    x[i,0].spines['bottom'].set_linewidth(1.5)
    x[i,0].text(2,0.4,'(%s)' %  (alpha_string[i*2]),fontsize=18)
    x[i,0].grid()
    
    if i != 4:
        x[i,0].set_xticklabels([])
    if i==4:
        x[i,0].set_xlabel(r'$rH_i$ [%]',fontsize = 20)
    x[i,0].set_ylabel(r'PDF',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=15,loc='upper right')

filename = 'era_iags_rh_as_fct_of_cc.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()



#%%


#--calculate the mean for each month
import calendar
if server == 0:
    ayears = np.arange(2017,2022,1)
if server == 1:
    ayears = np.arange(2010,2022,1)
amonths = np.arange(1,13,1)

n_meas_month = np.zeros((0))#--number of meas points per month

day_mean_t_iagos = np.zeros((0))
day_mean_t_era = np.zeros((0))
day_mean_t_era_cor = np.zeros((0))


day_mean_r_iagos = np.zeros((0))
day_mean_r_era = np.zeros((0))
day_mean_r_era_cor2d = np.zeros((0))

for y in np.arange(0,len(ayears)):
    print(ayears[y])
    for m in np.arange(0,len(amonths)):
        print(amonths[m])
    
        print(ayears[y], amonths[m])
        fool = (out_arr[15,:] == ayears[y]) & (out_arr[16,:] == amonths[m])
        #print(np.nanmean(t_out[fool]))
        daily_mean = np.nanmean(out_arr[7,fool])
        if np.isnan(daily_mean) == False:
            day_mean_t_iagos = np.append(day_mean_t_iagos,daily_mean)
        else:
            day_mean_t_iagos = np.append(day_mean_t_iagos,0)
        
        daily_mean = np.nanmean(t_out[fool])
        if np.isnan(daily_mean) == False:
            day_mean_t_era = np.append(day_mean_t_era,daily_mean)
        else:
            day_mean_t_era = np.append(day_mean_t_era,0)
        
        daily_mean = np.nanmean(t_out_cor[fool])
        if np.isnan(daily_mean) == False:   
            day_mean_t_era_cor = np.append(day_mean_t_era_cor,daily_mean)
        else:
            day_mean_t_era_cor = np.append(day_mean_t_era_cor,0)

        daily_mean = np.nanmean(out_arr[13,fool])
        if np.isnan(daily_mean) == False:
            day_mean_r_iagos = np.append(day_mean_r_iagos,daily_mean)
        else:
            day_mean_r_iagos = np.append(day_mean_r_iagos,0)
        
        daily_mean = np.nanmean(r_out_ice[fool])
        if np.isnan(daily_mean) == False:
            day_mean_r_era = np.append(day_mean_r_era,daily_mean)
        else:
            day_mean_r_era = np.append(day_mean_r_era,00)
        
        
        daily_mean = np.nanmean(r_out_ice_cor2d[fool])
        if np.isnan(daily_mean) == False:
            day_mean_r_era_cor2d = np.append(day_mean_r_era_cor2d,daily_mean)
        else:
            day_mean_r_era_cor2d = np.append(day_mean_r_era_cor2d,0)
            
        daily_mean = np.nansum(r_out_ice_cor2d[fool])
        if np.isnan(daily_mean) == False:
            n_meas_month = np.append(n_meas_month,daily_mean)
        else:
            n_meas_month = np.append(n_meas_month,0)
            


#%%           

day_mean_t_iagos_bak = day_mean_t_iagos
day_mean_t_era_bak = day_mean_t_era
day_mean_t_era_cor_bak = day_mean_t_era_cor

day_mean_r_iagos_bak = day_mean_r_iagos
day_mean_r_era_bak = day_mean_r_era
#
day_mean_r_era_cor2d_bak = day_mean_r_era_cor2d


#--make it a timestamp with year month day
dt=[]
for y in np.arange(0,len(ayears)):
    print(ayears[y])
    for m in np.arange(0,len(amonths)):
        print(amonths[m])
        dummy = np.datetime64(str(f'{ayears[y]:04.0f}')+'-'+str(f'{amonths[m]:02.0f}')+'-'+str(f'{1:02.0f}')+'')
        dt.append(dummy)
#--convert list to np array
dt = np.asarray(dt)

#%%

min_time_x = np.datetime64(str(f'{2017:04.0f}')+'-'+str(f'{4:02.0f}')+'-'+str(f'{1:02.0f}')+'')
max_time_x = np.datetime64(str(f'{2021:04.0f}')+'-'+str(f'{4:02.0f}')+'-'+str(f'{1:02.0f}')+'')

min_time_x = np.datetime64(str(f'{2015:04.0f}')+'-'+str(f'{1:02.0f}')+'-'+str(f'{1:02.0f}')+'')
max_time_x = np.datetime64(str(f'{2021:04.0f}')+'-'+str(f'{12:02.0f}')+'-'+str(f'{31:02.0f}')+'')

#--same as above but t and rh in one single plot

F,x=plt.subplots(5,1,figsize=(10,25),squeeze=False,gridspec_kw={'height_ratios':[1,1,1,1,0.5]})
x1=x[0,0].plot(dt,day_mean_t_iagos,label='IAGOS',color='k',marker='o')
x[0,0].plot(dt,day_mean_t_era,label='ERA',color='red',marker='o')
x[0,0].plot(dt,day_mean_t_era_cor,label='ERA QM',color='blue',marker='o')
x[0,0].set_xlim(min_time_x,max_time_x)
x[0,0].set_ylim(200,230)
x[0,0].tick_params(labelsize=20)
x[0,0].set_xticklabels([])
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_ylabel('Mean \n temperature [K]',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='lower left')
x[0,0].set_title('Monthly mean values',fontsize=20)
x[0,0].grid()
x[0,0].text(min_time_x +10, 227,'(a)',fontsize=20)
  
x1=x[1,0].plot(dt,day_mean_t_era - day_mean_t_iagos,label='ERA - IAGOS',color='red',marker='o')
x[1,0].plot(dt,day_mean_t_era_cor - day_mean_t_iagos,label='ERA QM - IAGOS',color='blue',marker='o')
x[1,0].plot((min_time_x,max_time_x),(0,0),linestyle='dashed',color='gray')
x[1,0].set_xlim(min_time_x,max_time_x)
x[1,0].set_ylim(-2.5,2.5)
x[1,0].set_xticklabels([])
x[1,0].tick_params(labelsize=20)
x[1,0].xaxis.set_tick_params(width=2,length=5)
x[1,0].yaxis.set_tick_params(width=2,length=5)
x[1,0].spines['top'].set_linewidth(1.5)
x[1,0].spines['left'].set_linewidth(1.5)
x[1,0].spines['right'].set_linewidth(1.5)
x[1,0].spines['bottom'].set_linewidth(1.5)
x[1,0].set_ylabel('Mean abs. temp. \n diff. ERA-IAGOS [K]',fontsize = 20)
x[1,0].grid()
x[1,0].text(min_time_x +10,2.0,'(b)',fontsize=20)
x[1,0].fill((min_time_x,max_time_x,max_time_x,min_time_x), [-0.5, -0.5, 0.5, 0.5], color = 'gray', alpha = 0.5)

    

x1=x[2,0].plot(dt,day_mean_r_iagos,label='IAGOS',color='k',marker='o')
x[2,0].plot(dt,day_mean_r_era,label='ERA',color='red',marker='o')
x[2,0].plot(dt,day_mean_r_era_cor2d,label='ERA QM',color='blue',marker='o')
x[2,0].set_xlim(min_time_x,max_time_x)
x[2,0].set_ylim(0,100)
x[2,0].set_xticklabels([])
x[2,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[2,0].yaxis.set_tick_params(width=2,length=5)
x[2,0].spines['top'].set_linewidth(1.5)
x[2,0].spines['left'].set_linewidth(1.5)
x[2,0].spines['right'].set_linewidth(1.5)
x[2,0].spines['bottom'].set_linewidth(1.5)
x[2,0].set_ylabel('Mean rel. hum. \n ice [%]',fontsize = 20)
x[2,0].grid()
x[2,0].text(min_time_x +10,90,'(c)',fontsize=20)
  

x1=x[3,0].plot(dt,day_mean_r_era - day_mean_r_iagos,label='ERA - IAGOS',color='red',marker='o')
x[3,0].plot(dt,day_mean_r_era_cor2d - day_mean_r_iagos,label='ERA QM - IAGOS',color='blue',marker='o')
x[3,0].plot((min_time_x,max_time_x),(0,0),linestyle='dashed',color='gray')


x[3,0].set_xlim(min_time_x,max_time_x)
x[3,0].set_ylim(-20,20)
x[3,0].set_xticklabels([])
x[3,0].tick_params(labelsize=20)
x[3,0].xaxis.set_tick_params(width=2,length=5)
x[3,0].yaxis.set_tick_params(width=2,length=5)
x[3,0].spines['top'].set_linewidth(1.5)
x[3,0].spines['left'].set_linewidth(1.5)
x[3,0].spines['right'].set_linewidth(1.5)
x[3,0].spines['bottom'].set_linewidth(1.5)
x[3,0].set_ylabel('Mean. rel. hum. ice diff. \n  ERA-IAGOS [%]',fontsize = 20)
x[3,0].grid()
x[3,0].text(min_time_x +10,16,'(d)',fontsize=20)
x[3,0].fill((min_time_x,max_time_x,max_time_x,min_time_x), [-10, -10, 10, 10], color = 'lightgray', alpha = 0.5)
x[3,0].fill((min_time_x,max_time_x,max_time_x,min_time_x), [-5, -5, 5, 5], color = 'gray', alpha = 0.5)


x1=x[4,0].plot(dt,n_meas_month/2000000 ,label='ERA - IAGOS',color='red',marker='o')
x[4,0].set_xlim(min_time_x,max_time_x)
x[4,0].set_ylim(0,12)
x[4,0].tick_params(labelsize=20)
x[4,0].xaxis.set_tick_params(width=2,length=5)
x[4,0].yaxis.set_tick_params(width=2,length=5)
x[4,0].spines['top'].set_linewidth(1.5)
x[4,0].spines['left'].set_linewidth(1.5)
x[4,0].spines['right'].set_linewidth(1.5)
x[4,0].spines['bottom'].set_linewidth(1.5)
x[4,0].set_xlabel('Datetime',fontsize = 20)
x[4,0].set_ylabel('Number of \n samples [*2e6]',fontsize = 20)
x[4,0].grid()
x[4,0].text(min_time_x +10,10,'(e)',fontsize=20)
    
filename = 'day_mean_era_minus_iagos_and_difference_t_and_rh.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()


#%%


#--calculate the mean of t and r for each month and all years.
ayears = np.arange(2015,2022)
amonths = np.arange(1,13)

t_means_iagos = np.zeros((len(ayears),len(amonths)))
t_means_era = np.zeros((len(ayears),len(amonths)))
t_means_era_cor = np.zeros((len(ayears),len(amonths)))

r_means_iagos = np.zeros((len(ayears),len(amonths)))
r_means_era = np.zeros((len(ayears),len(amonths)))
r_means_era_cor2d = np.zeros((len(ayears),len(amonths)))

for y in np.arange(0,len(ayears)):
    for m in np.arange(0,len(amonths)):
        foo = ((out_arr[15,:] == ayears[y]) & (out_arr[16,:] == amonths[m]))
        t_means_iagos[y,m] = np.nanmean(out_arr[7,foo])
        t_means_era[y,m] = np.nanmean(t_out[foo])
        t_means_era_cor[y,m] = np.nanmean(t_out_cor[foo])
        
        r_means_iagos[y,m] = np.nanmean(out_arr[13,foo])
        r_means_era[y,m] = np.nanmean(r_out_ice[foo])
        r_means_era_cor2d[y,m] = np.nanmean(r_out_ice_cor2d[foo])
        
#%%

#-- same as above but always using ERA5 as a reference

F,x=plt.subplots(len(ayears),1,figsize=(15,len(ayears)*5),squeeze=False) 
x1=x[0,0].plot(0,0)

mycolors = ['k','crimson','blue','red','green','orange','steelblue','darkred','olive','deepskyblue','palegreen','darkorange']
for y in np.arange(0,len(ayears)):
    x[y,0].plot(np.arange(1,13),np.nanmean(r_means_era[year_ind,:],axis=0) - r_means_iagos[y,:],linestyle='solid',marker='o',label='IAGOS '+str(ayears[y]),color='k')
    x[y,0].plot(np.arange(1,13),np.nanmean(r_means_era[year_ind,:],axis=0) - r_means_era[y,:],linestyle='solid',marker='o',label='ERA '+str(ayears[y]),color='red')
    x[y,0].plot(np.arange(1,13),np.nanmean(r_means_era[year_ind,:],axis=0) - r_means_era_cor2d[y,:],linestyle='solid',marker='x',label='ERA cor '+str(ayears[y]),color='blue')
    x[y,0].plot((-1,18),(0,0),linestyle='dashed',color='gray')
    x[y,0].text(0.2,22,'('+alpha_string[y]+ ') '+str(ayears[y]),fontsize=25)
    
    x[y,0].fill([0, 15, 15, 0], [-10, -10, 10, 10], color = 'lightgray', alpha = 0.5)
    x[y,0].fill([0, 15, 15, 0], [-5, -5, 5, 5], color = 'darkgray', alpha = 0.5)
    
    #--calculate mean deviation of the entire year
    year_mean_dev_iagos = np.nanmean(np.nanmean(r_means_era[year_ind,:],axis=0) - r_means_iagos[y,:])
    year_mean_dev_era = np.nanmean(np.nanmean(r_means_era[year_ind,:],axis=0) - r_means_era[y,:])
    year_mean_dev_era_cor2d = np.nanmean(np.nanmean(r_means_era[year_ind,:],axis=0) - r_means_era_cor2d[y,:])
    
    x[y,0].text(1.9,22,'ERA5      : %4.1f' %(year_mean_dev_era),fontsize=20)
    x[y,0].text(1.9,17,'ERA5 QM: %4.1f' %(year_mean_dev_era_cor2d),fontsize=20)
    x[y,0].text(1.9,12,'IAGOS    : %4.1f' %(year_mean_dev_iagos),fontsize=20)
    x[y,0].set_xlim(0,13)
    x[y,0].set_xticks(np.arange(0, 13, 1))
    x[y,0].set_ylim(-30,30)
    x[y,0].tick_params(labelsize=20)
    x[y,0].xaxis.set_tick_params(width=2,length=5)
    x[y,0].yaxis.set_tick_params(width=2,length=5)
    x[y,0].spines['top'].set_linewidth(1.5)
    x[y,0].spines['left'].set_linewidth(1.5)
    x[y,0].spines['right'].set_linewidth(1.5)
    x[y,0].spines['bottom'].set_linewidth(1.5)
    x[y,0].grid()
    if y == 0:
        x[y,0].set_title('Monthly mean (2018-2021) - Monthly mean of year',fontsize = 20)
    if y == len(ayears)-1:
        x[y,0].set_xlabel('Month',fontsize = 20)
    x[y,0].set_ylabel('ERA - X (legend) $\Delta RH $',fontsize = 20)
    x[y,0].legend(shadow=True,fontsize=18,loc='upper right')

filename = 'inconsistency_rh_iagos_era_ref_era.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    #F.show()
    plt.close()

#%%


#%%
#--calculate the regression over 12 month sections
from scipy import stats

for i in np.arange(0,4):
    print('i: ',i)
    diff_y = day_mean_r_era[12*i:12*i+12] - day_mean_r_iagos[12*i:12*i+12]
    print(diff_y)
    res = stats.linregress(np.arange(0,len(diff_y)),diff_y)
    print(res)
    print('')


#%%
#--calculate the temperature and rel hum statistical stuff and safe

print('below statistic only for plevels 250 to 200 hPa')
#--print for each pressure level
for p in np.arange(2,5):
    print('Pressure level: ',pres[p])
    print('1h 3h 6h timestep')
    print('For Temperature')
    print('RMSE:    %6.3f %6.3f %6.3f' % ((rmse_t_era_iagos_hour[p,0]),(rmse_t_era_iagos_hour[p,1]),(rmse_t_era_iagos_hour[p,2])))
    print('MAE:     %6.3f %6.3f %6.3f' % ((mae_t_era_iagos_hour[p,0]),(mae_t_era_iagos_hour[p,1]),(mae_t_era_iagos_hour[p,2])))
    print('rsquare: %6.3f %6.3f %6.3f' % ((rsquare_t_era_iagos_hour[p,0]),(rsquare_t_era_iagos_hour[p,1]),(rsquare_t_era_iagos_hour[p,2])))
    print('MSE:     %6.3f %6.3f %6.3f' % ((mse_t_era_iagos_hour[p,0]),(mse_t_era_iagos_hour[p,1]),(mse_t_era_iagos_hour[p,2])))
    print('MD:      %6.3f %6.3f %6.3f' % ((md_t_era_iagos_hour[p,0]),(md_t_era_iagos_hour[p,1]),(md_t_era_iagos_hour[p,2])))
    print('')
    print('For rh ice')
    print('RMSE:    %6.3f %6.3f %6.3f' % ((rmse_r_era_iagos_hour[p,0]),(rmse_r_era_iagos_hour[p,1]),(rmse_r_era_iagos_hour[p,2])))
    print('MAE:     %6.3f %6.3f %6.3f' % ((mae_r_era_iagos_hour[p,0]),(mae_r_era_iagos_hour[p,1]),(mae_r_era_iagos_hour[p,2])))
    print('rsquare: %6.3f %6.3f %6.3f' % ((rsquare_r_era_iagos_hour[p,0]),(rsquare_r_era_iagos_hour[p,1]),(rsquare_r_era_iagos_hour[p,2])))
    print('MSE:     %6.3f %6.3f %6.3f' % ((mse_r_era_iagos_hour[p,0]),(mse_r_era_iagos_hour[p,1]),(mse_r_era_iagos_hour[p,2])))
    print('MD:      %6.3f %6.3f %6.3f' % ((md_r_era_iagos_hour[p,0]),(md_r_era_iagos_hour[p,1]),(md_r_era_iagos_hour[p,2])))
    print('')
print('------------------')
    

outfile2.write('Below statistic only for plevels 250 to 200 hPa')
outfile2.write('/n')
    

for p in np.arange(2,5):
    outfile2.write('Pressure level: '+str(pres[p])+'\n')
    outfile2.write('1h 3h (%) 6h (%) timestep \n')
    outfile2.write('For Temperature \n')
    outfile2.write('RMSE:    %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((rmse_t_era_iagos_hour[p,0]),   (rmse_t_era_iagos_hour[p,1]),   (rmse_t_era_iagos_hour[p,1])/(rmse_t_era_iagos_hour[p,0]),      (rmse_t_era_iagos_hour[p,2]),   (rmse_t_era_iagos_hour[p,2])/(rmse_t_era_iagos_hour[p,0])))
    outfile2.write('MAE:     %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((mae_t_era_iagos_hour[p,0]),    (mae_t_era_iagos_hour[p,1]),    (mae_t_era_iagos_hour[p,1])/(mae_t_era_iagos_hour[p,0]),        (mae_t_era_iagos_hour[p,2]),    (mae_t_era_iagos_hour[p,2])/(mae_t_era_iagos_hour[p,0])))
    outfile2.write('rsquare: %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((rsquare_t_era_iagos_hour[p,0]),(rsquare_t_era_iagos_hour[p,1]),(rsquare_t_era_iagos_hour[p,1])/(rsquare_t_era_iagos_hour[p,0]),(rsquare_t_era_iagos_hour[p,2]),(rsquare_t_era_iagos_hour[p,2])/(rsquare_t_era_iagos_hour[p,0])))
    outfile2.write('MSE:     %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((mse_t_era_iagos_hour[p,0]),    (mse_t_era_iagos_hour[p,1]),    (mse_t_era_iagos_hour[p,1])/(mse_t_era_iagos_hour[p,0]),        (mse_t_era_iagos_hour[p,2]),    (mse_t_era_iagos_hour[p,2])/(mse_t_era_iagos_hour[p,0])))
    outfile2.write('MD:      %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((md_t_era_iagos_hour[p,0]),     (md_t_era_iagos_hour[p,1]),     (md_t_era_iagos_hour[p,1])/(md_t_era_iagos_hour[p,0]),          (md_t_era_iagos_hour[p,2]),     (md_t_era_iagos_hour[p,2])/(md_t_era_iagos_hour[p,0])))
    outfile2.write(' \n')
    outfile2.write('For rh ice \n')
    outfile2.write('RMSE:    %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((rmse_r_era_iagos_hour[p,0]), (rmse_r_era_iagos_hour[p,1]), (rmse_r_era_iagos_hour[p,1])/(rmse_r_era_iagos_hour[p,0]), (rmse_r_era_iagos_hour[p,2]), (rmse_r_era_iagos_hour[p,2])/(rmse_r_era_iagos_hour[p,0])))
    outfile2.write('MAE:     %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((mae_r_era_iagos_hour[p,0]), (mae_r_era_iagos_hour[p,1]), (mae_r_era_iagos_hour[p,1])/(mae_r_era_iagos_hour[p,0]), (mae_r_era_iagos_hour[p,2]),(mae_r_era_iagos_hour[p,2])/(mae_r_era_iagos_hour[p,0])))
    outfile2.write('rsquare: %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((rsquare_r_era_iagos_hour[p,0]), (rsquare_r_era_iagos_hour[p,1]), (rsquare_r_era_iagos_hour[p,1])/(rsquare_r_era_iagos_hour[p,0]),(rsquare_r_era_iagos_hour[p,2]),(rsquare_r_era_iagos_hour[p,2])/(rsquare_r_era_iagos_hour[p,0])))
    outfile2.write('MSE:     %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((mse_r_era_iagos_hour[p,0]), (mse_r_era_iagos_hour[p,1]), (mse_r_era_iagos_hour[p,1])/(mse_r_era_iagos_hour[p,0]), (mse_r_era_iagos_hour[p,2]), (mse_r_era_iagos_hour[p,2])/(mse_r_era_iagos_hour[p,0])))
    outfile2.write('MD:      %6.3f %6.3f %6.3f %6.3f %6.3f \n' % ((md_r_era_iagos_hour[p,0]), (md_r_era_iagos_hour[p,1]), (md_r_era_iagos_hour[p,1])/(md_r_era_iagos_hour[p,0]), (md_r_era_iagos_hour[p,2]), (md_r_era_iagos_hour[p,2])/(md_r_era_iagos_hour[p,0])))
    outfile2.write(' \n')
outfile2.write('------------------ \n')


#%%
#--calculating 5 metrics to describe the difference between the original data, cor2d, and theo correction

#--RMSE
rmse_t_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
rmse_r_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
for p in np.arange(0,len(pres)-1):
    foo = (pres_out == pres[p])
    rmse_t_era_iagos_cor[p,0] = rmse(t_out[foo], out_arr[7,foo])
    rmse_t_era_iagos_cor[p,1] = rmse(t_out_cor[foo], out_arr[7,foo])
    
    rmse_r_era_iagos_cor[p,0] = rmse(r_out_ice[foo], out_arr[13,foo])
    rmse_r_era_iagos_cor[p,1] = rmse(r_out_ice_cor2d[foo], out_arr[13,foo])
    rmse_r_era_iagos_cor[p,2] = rmse(r_out_ice_teoh[foo], out_arr[13,foo])


#--MAE
mae_t_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
mae_r_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
for p in np.arange(0,len(pres)-1):
    foo = (pres_out == pres[p])
    mae_t_era_iagos_cor[p,0] = mae(t_out[foo],out_arr[7,foo])
    mae_t_era_iagos_cor[p,1] = mae(t_out_cor[foo],out_arr[7,foo])
    
    mae_r_era_iagos_cor[p,0] = mae(r_out_ice[foo],out_arr[13,foo])
    mae_r_era_iagos_cor[p,1] = mae(r_out_ice_cor2d[foo], out_arr[13,foo])
    mae_r_era_iagos_cor[p,2] = mae(r_out_ice_teoh[foo],out_arr[13,foo])
    

#--rsquare
rsquare_t_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
rsquare_r_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
for p in np.arange(0,len(pres)-1):
    foo = (pres_out == pres[p])
    rsquare_t_era_iagos_cor[p,0] = rSquare(t_out[foo], out_arr[7,foo])
    rsquare_t_era_iagos_cor[p,1] = rSquare(t_out_cor[foo],out_arr[7,foo])
    
    rsquare_r_era_iagos_cor[p,0] = rSquare(r_out_ice[foo],out_arr[13,foo])
    rsquare_r_era_iagos_cor[p,1] = rSquare(r_out_ice_cor2d[foo],out_arr[13,foo])
    rsquare_r_era_iagos_cor[p,2] = rSquare(r_out_ice_teoh[foo],out_arr[13,foo])


#--mse
mse_t_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
mse_r_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
for p in np.arange(0,len(pres)-1):
    foo = (pres_out == pres[p])
    mse_t_era_iagos_cor[p,0] = mse(t_out[foo], out_arr[7,foo])
    mse_t_era_iagos_cor[p,1] = mse(t_out_cor[foo],out_arr[7,foo])
    
    mse_r_era_iagos_cor[p,0] = mse(r_out_ice[foo],out_arr[13,foo])
    mse_r_era_iagos_cor[p,1] = mse(r_out_ice_cor2d[foo],out_arr[13,foo])
    mse_r_era_iagos_cor[p,2] = mse(r_out_ice_teoh[foo],out_arr[13,foo])
    

#--md
md_t_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
md_r_era_iagos_cor = np.zeros((len(pres),3))   #plevel,timeres
for p in np.arange(0,len(pres)-1):
    foo = (pres_out == pres[p])
    md_t_era_iagos_cor[p,0] = md(t_out[foo], out_arr[7,foo])
    md_t_era_iagos_cor[p,1] = md(t_out_cor[foo],out_arr[7,foo])
    
    md_r_era_iagos_cor[p,0] = md(r_out_ice[foo],out_arr[13,foo])
    md_r_era_iagos_cor[p,1] = md(r_out_ice_cor2d[foo],out_arr[13,foo])
    md_r_era_iagos_cor[p,2] = md(r_out_ice_teoh[foo],out_arr[13,foo])


#%%

F,x=plt.subplots(2,5,figsize=(25,10),squeeze=False)
#--rmse
#--------------------------------
x1=x[0,0].scatter(-3,0,color='red')
x[0,0].bar(1-0.5,rmse_t_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[0,0].bar(1,rmse_t_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[0,0].bar(1+0.5,rmse_t_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[0,0].bar(3-0.5,rmse_t_era_iagos_cor[2,1],0.5,0,color='blue')
x[0,0].bar(3,rmse_t_era_iagos_cor[3,1],0.5,0,color='orange')
x[0,0].bar(3+0.5,rmse_t_era_iagos_cor[4,1],0.5,0,color='green')

x[0,0].bar(6-0.5,rmse_t_era_iagos_cor[2,2],0.5,0,color='blue')
x[0,0].bar(6,rmse_t_era_iagos_cor[3,2],0.5,0,color='orange')
x[0,0].bar(6+0.5,rmse_t_era_iagos_cor[4,2],0.5,0,color='green')

x[0,0].set_xticks(np.arange(1,5,2))
x[0,0].set_xticklabels(['Org','QM'])
x[0,0].set_xlim(0,7)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('(a) RMSE [K]',fontsize=20)


x6=x[1,0].scatter(-3,0,color='red')
x[1,0].bar(1-0.5,rmse_r_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[1,0].bar(1,rmse_r_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[1,0].bar(1+0.5,rmse_r_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[1,0].bar(3-0.5,rmse_r_era_iagos_cor[2,1],0.5,0,color='blue')
x[1,0].bar(3,rmse_r_era_iagos_cor[3,1],0.5,0,color='orange')
x[1,0].bar(3+0.5,rmse_r_era_iagos_cor[4,1],0.5,0,color='green')

x[1,0].bar(5-0.5,rmse_r_era_iagos_cor[2,2],0.5,0,color='blue')
x[1,0].bar(5,rmse_r_era_iagos_cor[3,2],0.5,0,color='orange')
x[1,0].bar(5+0.5,rmse_r_era_iagos_cor[4,2],0.5,0,color='green')

x[1,0].set_xticks(np.arange(1,8,2))
x[1,0].set_xticklabels(['Org','QM','T22',''])
x[1,0].set_xlim(0,7)
x[1,0].tick_params(labelsize=20)
x[1,0].xaxis.set_tick_params(width=2,length=5)
x[1,0].yaxis.set_tick_params(width=2,length=5)
x[1,0].spines['top'].set_linewidth(1.5)
x[1,0].spines['left'].set_linewidth(1.5)
x[1,0].spines['right'].set_linewidth(1.5)
x[1,0].spines['bottom'].set_linewidth(1.5)
x[1,0].set_title('(f) RMSE [%]',fontsize=20)


#--mae
#--------------------------------
x1=x[0,1].scatter(-3,0,color='red')
x[0,1].bar(1-0.5,mae_t_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[0,1].bar(1,mae_t_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[0,1].bar(1+0.5,mae_t_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[0,1].bar(3-0.5,mae_t_era_iagos_cor[2,1],0.5,0,color='blue')
x[0,1].bar(3,mae_t_era_iagos_cor[3,1],0.5,0,color='orange')
x[0,1].bar(3+0.5,mae_t_era_iagos_cor[4,1],0.5,0,color='green')

x[0,1].bar(6-0.5,mae_t_era_iagos_cor[2,2],0.5,0,color='blue')
x[0,1].bar(6,mae_t_era_iagos_cor[3,2],0.5,0,color='orange')
x[0,1].bar(6+0.5,mae_t_era_iagos_cor[4,2],0.5,0,color='green')

x[0,1].set_xticks(np.arange(1,5,2))
x[0,1].set_xticklabels(['Org','QM'])
x[0,1].set_xlim(0,7)
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].set_title('(b) MAE [K]',fontsize=20)

x6=x[1,1].scatter(-3,0,color='red')
x[1,1].bar(1-0.5,mae_r_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[1,1].bar(1,mae_r_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[1,1].bar(1+0.5,mae_r_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[1,1].bar(3-0.5,mae_r_era_iagos_cor[2,1],0.5,0,color='blue')
x[1,1].bar(3,mae_r_era_iagos_cor[3,1],0.5,0,color='orange')
x[1,1].bar(3+0.5,mae_r_era_iagos_cor[4,1],0.5,0,color='green')

x[1,1].bar(5-0.5,mae_r_era_iagos_cor[2,2],0.5,0,color='blue')
x[1,1].bar(5,mae_r_era_iagos_cor[3,2],0.5,0,color='orange')
x[1,1].bar(5+0.5,mae_r_era_iagos_cor[4,2],0.5,0,color='green')

x[1,1].set_xticks(np.arange(1,8,2))
x[1,1].set_xticklabels(['Org','QM','T22',''])
x[1,1].set_xlim(0,7)
x[1,1].tick_params(labelsize=20)
x[1,1].xaxis.set_tick_params(width=2,length=5)
x[1,1].yaxis.set_tick_params(width=2,length=5)
x[1,1].spines['top'].set_linewidth(1.5)
x[1,1].spines['left'].set_linewidth(1.5)
x[1,1].spines['right'].set_linewidth(1.5)
x[1,1].spines['bottom'].set_linewidth(1.5)
x[1,1].set_title('(g) MAE [%]',fontsize=20)

#--rsquare
#--------------------------------
x1=x[0,2].scatter(-3,0,color='red')
x[0,2].bar(1-0.5,rsquare_t_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[0,2].bar(1,rsquare_t_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[0,2].bar(1+0.5,rsquare_t_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[0,2].bar(3-0.5,rsquare_t_era_iagos_cor[2,1],0.5,0,color='blue')
x[0,2].bar(3,rsquare_t_era_iagos_cor[3,1],0.5,0,color='orange')
x[0,2].bar(3+0.5,rsquare_t_era_iagos_cor[4,1],0.5,0,color='green')

x[0,2].bar(5-0.5,rsquare_t_era_iagos_cor[2,2],0.5,0,color='blue')
x[0,2].bar(5,rsquare_t_era_iagos_cor[3,2],0.5,0,color='orange')
x[0,2].bar(5+0.5,rsquare_t_era_iagos_cor[4,2],0.5,0,color='green')

x[0,2].set_xticks(np.arange(1,5,2))
x[0,2].set_xticklabels(['Org','QM'])
x[0,2].set_xlim(0,7)
x[0,2].set_ylim(0,1)
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].set_title('(c) r square',fontsize=20)

x6=x[1,2].scatter(-3,0,color='red')
x[1,2].bar(1-0.5,rsquare_r_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[1,2].bar(1,rsquare_r_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[1,2].bar(1+0.5,rsquare_r_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[1,2].bar(3-0.5,rsquare_r_era_iagos_cor[2,1],0.5,0,color='blue')
x[1,2].bar(3,rsquare_r_era_iagos_cor[3,1],0.5,0,color='orange')
x[1,2].bar(3+0.5,rsquare_r_era_iagos_cor[4,1],0.5,0,color='green')

x[1,2].bar(5-0.5,rsquare_r_era_iagos_cor[2,2],0.5,0,color='blue')
x[1,2].bar(5,rsquare_r_era_iagos_cor[3,2],0.5,0,color='orange')
x[1,2].bar(5+0.5,rsquare_r_era_iagos_cor[4,2],0.5,0,color='green')

x[1,2].set_xticks(np.arange(1,8,2))
x[1,2].set_xticklabels(['Org','QM','T22',''])
x[1,2].set_xlim(0,7)
x[1,2].set_ylim(0,1)
x[1,2].tick_params(labelsize=20)
x[1,2].xaxis.set_tick_params(width=2,length=5)
x[1,2].yaxis.set_tick_params(width=2,length=5)
x[1,2].spines['top'].set_linewidth(1.5)
x[1,2].spines['left'].set_linewidth(1.5)
x[1,2].spines['right'].set_linewidth(1.5)
x[1,2].spines['bottom'].set_linewidth(1.5)
x[1,2].set_title('(h) r square',fontsize=20)


#--mse
#--------------------------------
x1=x[0,3].scatter(-3,0,color='red')
x[0,3].bar(1-0.5,mse_t_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[0,3].bar(1,mse_t_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[0,3].bar(1+0.5,mse_t_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[0,3].bar(3-0.5,mse_t_era_iagos_cor[2,1],0.5,0,color='blue')
x[0,3].bar(3,mse_t_era_iagos_cor[3,1],0.5,0,color='orange')
x[0,3].bar(3+0.5,mse_t_era_iagos_cor[4,1],0.5,0,color='green')

x[0,3].bar(5-0.5,mse_t_era_iagos_cor[2,2],0.5,0,color='blue')
x[0,3].bar(5,mse_t_era_iagos_cor[3,2],0.5,0,color='orange')
x[0,3].bar(5+0.5,mse_t_era_iagos_cor[4,2],0.5,0,color='green')

x[0,3].set_xticks(np.arange(1,5,2))
x[0,3].set_xticklabels(['Org','QM'])
x[0,3].set_xlim(0,7)
x[0,3].tick_params(labelsize=20)
x[0,3].xaxis.set_tick_params(width=2,length=5)
x[0,3].yaxis.set_tick_params(width=2,length=5)
x[0,3].spines['top'].set_linewidth(1.5)
x[0,3].spines['left'].set_linewidth(1.5)
x[0,3].spines['right'].set_linewidth(1.5)
x[0,3].spines['bottom'].set_linewidth(1.5)
x[0,3].set_title(r'(d) MSE [K$^2$]',fontsize=20)

x6=x[1,3].scatter(-3,0,color='red')
x[1,3].bar(1-0.5,mse_r_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[1,3].bar(1,mse_r_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[1,3].bar(1+0.5,mse_r_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[1,3].bar(3-0.5,mse_r_era_iagos_cor[2,1],0.5,0,color='blue')
x[1,3].bar(3,mse_r_era_iagos_cor[3,1],0.5,0,color='orange')
x[1,3].bar(3+0.5,mse_r_era_iagos_cor[4,1],0.5,0,color='green')

x[1,3].bar(5-0.5,mse_r_era_iagos_cor[2,2],0.5,0,color='blue')
x[1,3].bar(5,mse_r_era_iagos_cor[3,2],0.5,0,color='orange')
x[1,3].bar(5+0.5,mse_r_era_iagos_cor[4,2],0.5,0,color='green')

x[1,3].set_xticks(np.arange(1,8,2))
x[1,3].set_xticklabels(['Org','QM','T22',''])
x[1,3].set_xlim(0,7)
x[1,3].tick_params(labelsize=20)
x[1,3].xaxis.set_tick_params(width=2,length=5)
x[1,3].yaxis.set_tick_params(width=2,length=5)
x[1,3].spines['top'].set_linewidth(1.5)
x[1,3].spines['left'].set_linewidth(1.5)
x[1,3].spines['right'].set_linewidth(1.5)
x[1,3].spines['bottom'].set_linewidth(1.5)
x[1,3].set_title(r'(i) MSE [%$^2$]',fontsize=20)


#--md
#--------------------------------
x1=x[0,4].scatter(-3,0,color='red')
x[0,4].bar(1-0.5,md_t_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[0,4].bar(1,md_t_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[0,4].bar(1+0.5,md_t_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[0,4].bar(3-0.5,md_t_era_iagos_cor[2,1],0.5,0,color='blue')
x[0,4].bar(3,md_t_era_iagos_cor[3,1],0.5,0,color='orange')
x[0,4].bar(3+0.5,md_t_era_iagos_cor[4,1],0.5,0,color='green')

x[0,4].bar(5-0.5,md_t_era_iagos_cor[2,2],0.5,0,color='blue')
x[0,4].bar(5,md_t_era_iagos_cor[3,2],0.5,0,color='orange')
x[0,4].bar(5+0.5,md_t_era_iagos_cor[4,2],0.5,0,color='green')

x[0,4].plot((0,10),(0,0),linestyle='solid',color='gray')

x[0,4].set_xticks(np.arange(1,5,2))
x[0,4].set_xticklabels(['Org','QM'])
x[0,4].set_xlim(0,7)
x[0,4].set_ylim(-1.5,1.5)
x[0,4].tick_params(labelsize=20)
x[0,4].xaxis.set_tick_params(width=2,length=5)
x[0,4].yaxis.set_tick_params(width=2,length=5)
x[0,4].spines['top'].set_linewidth(1.5)
x[0,4].spines['left'].set_linewidth(1.5)
x[0,4].spines['right'].set_linewidth(1.5)
x[0,4].spines['bottom'].set_linewidth(1.5)
x[0,4].set_title('(e) MD [K]',fontsize=20)
x[0,4].legend(shadow=True,fontsize=20)

x6=x[1,4].scatter(-3,0,color='red')
x[1,4].bar(1-0.5,md_r_era_iagos_cor[2,0],0.5,0,label='%3d hPa' % pres[ind_250],color='blue')
x[1,4].bar(1,md_r_era_iagos_cor[3,0],0.5,0,label='%3d hPa' % pres[ind_225],color='orange')
x[1,4].bar(1+0.5,md_r_era_iagos_cor[4,0],0.5,0,label='%3d hPa' % pres[ind_200],color='green')

x[1,4].bar(3-0.5,md_r_era_iagos_cor[2,1],0.5,0,color='blue')
x[1,4].bar(3,md_r_era_iagos_cor[3,1],0.5,0,color='orange')
x[1,4].bar(3+0.5,md_r_era_iagos_cor[4,1],0.5,0,color='green')

x[1,4].bar(5-0.5,md_r_era_iagos_cor[2,2],0.5,0,color='blue')
x[1,4].bar(5,md_r_era_iagos_cor[3,2],0.5,0,color='orange')
x[1,4].bar(5+0.5,md_r_era_iagos_cor[4,2],0.5,0,color='green')

x[1,4].plot((0,10),(0,0),linestyle='solid',color='gray')

x[1,4].set_xticks(np.arange(1,8,2))
x[1,4].set_xticklabels(['Org','QM','T22',''])
x[1,4].set_xlim(0,7)
x[1,4].set_ylim(-5.5,5.5)
x[1,4].tick_params(labelsize=20)
x[1,4].xaxis.set_tick_params(width=2,length=5)
x[1,4].yaxis.set_tick_params(width=2,length=5)
x[1,4].spines['top'].set_linewidth(1.5)
x[1,4].spines['left'].set_linewidth(1.5)
x[1,4].spines['right'].set_linewidth(1.5)
x[1,4].spines['bottom'].set_linewidth(1.5)
x[1,4].set_title('(j) MD [%]',fontsize=20)

filename = 'metrics_plot_corrections.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()



#%%
#--calculate the temperature and rel hum statistical stuff and safe

print('below statistic only for plevels 250 to 200 hPa')
#--print for each pressure level
for p in np.arange(2,5):
    print('Pressure level: ',pres[p])
    print('original era, cor2d, and teoh')
    print('For Temperature')
    print('RMSE:    %6.3f %6.3f %6.3f' % ((rmse_t_era_iagos_cor[p,0]),(rmse_t_era_iagos_cor[p,1]),(rmse_t_era_iagos_cor[p,2])))
    print('MAE:     %6.3f %6.3f %6.3f' % ((mae_t_era_iagos_cor[p,0]),(mae_t_era_iagos_cor[p,1]),(mae_t_era_iagos_cor[p,2])))
    print('rsquare: %6.3f %6.3f %6.3f' % ((rsquare_t_era_iagos_cor[p,0]),(rsquare_t_era_iagos_cor[p,1]),(rsquare_t_era_iagos_cor[p,2])))
    print('MSE:     %6.3f %6.3f %6.3f' % ((mse_t_era_iagos_cor[p,0]),(mse_t_era_iagos_cor[p,1]),(mse_t_era_iagos_cor[p,2])))
    print('MD:      %6.3f %6.3f %6.3f' % ((md_t_era_iagos_cor[p,0]),(md_t_era_iagos_cor[p,1]),(md_t_era_iagos_cor[p,2])))
    print('')
    print('For rh ice')
    print('RMSE:    %6.3f %6.3f %6.3f' % ((rmse_r_era_iagos_cor[p,0]),(rmse_r_era_iagos_cor[p,1]),(rmse_r_era_iagos_cor[p,2])))
    print('MAE:     %6.3f %6.3f %6.3f' % ((mae_r_era_iagos_cor[p,0]),(mae_r_era_iagos_cor[p,1]),(mae_r_era_iagos_cor[p,2])))
    print('rsquare: %6.3f %6.3f %6.3f' % ((rsquare_r_era_iagos_cor[p,0]),(rsquare_r_era_iagos_cor[p,1]),(rsquare_r_era_iagos_cor[p,2])))
    print('MSE:     %6.3f %6.3f %6.3f' % ((mse_r_era_iagos_cor[p,0]),(mse_r_era_iagos_cor[p,1]),(mse_r_era_iagos_cor[p,2])))
    print('MD:      %6.3f %6.3f %6.3f' % ((md_r_era_iagos_cor[p,0]),(md_r_era_iagos_cor[p,1]),(md_r_era_iagos_cor[p,2])))
    print('')
print('------------------')
    

outfile2.write('Below statistic only for plevels 250 to 200 hPa')
outfile2.write('/n')
    

for p in np.arange(2,5):
    outfile2.write('Pressure level: '+str(pres[p])+'\n')
    outfile2.write('original era, cor2d, and teoh \n')
    outfile2.write('For Temperature \n')
    outfile2.write('RMSE:    %6.3f %6.3f %6.3f \n' % ((rmse_t_era_iagos_cor[p,0]),(rmse_t_era_iagos_cor[p,1]),(rmse_t_era_iagos_cor[p,2])))
    outfile2.write('MAE:     %6.3f %6.3f %6.3f \n' % ((mae_t_era_iagos_cor[p,0]),(mae_t_era_iagos_cor[p,1]),(mae_t_era_iagos_cor[p,2])))
    outfile2.write('rsquare: %6.3f %6.3f %6.3f \n' % ((rsquare_t_era_iagos_cor[p,0]),(rsquare_t_era_iagos_cor[p,1]),(rsquare_t_era_iagos_cor[p,2])))
    outfile2.write('MSE:     %6.3f %6.3f %6.3f \n' % ((mse_t_era_iagos_cor[p,0]),(mse_t_era_iagos_cor[p,1]),(mse_t_era_iagos_cor[p,2])))
    outfile2.write('MD:      %6.3f %6.3f %6.3f \n' % ((md_t_era_iagos_cor[p,0]),(md_t_era_iagos_cor[p,1]),(md_t_era_iagos_cor[p,2])))
    outfile2.write(' \n')
    outfile2.write('For rh ice \n')
    outfile2.write('RMSE:    %6.3f %6.3f %6.3f \n' % ((rmse_r_era_iagos_cor[p,0]),(rmse_r_era_iagos_cor[p,1]),(rmse_r_era_iagos_cor[p,2])))
    outfile2.write('MAE:     %6.3f %6.3f %6.3f \n' % ((mae_r_era_iagos_cor[p,0]),(mae_r_era_iagos_cor[p,1]),(mae_r_era_iagos_cor[p,2])))
    outfile2.write('rsquare: %6.3f %6.3f %6.3f \n' % ((rsquare_r_era_iagos_cor[p,0]),(rsquare_r_era_iagos_cor[p,1]),(rsquare_r_era_iagos_cor[p,2])))
    outfile2.write('MSE:     %6.3f %6.3f %6.3f \n' % ((mse_r_era_iagos_cor[p,0]),(mse_r_era_iagos_cor[p,1]),(mse_r_era_iagos_cor[p,2])))
    outfile2.write('MD:      %6.3f %6.3f %6.3f \n' % ((md_r_era_iagos_cor[p,0]),(md_r_era_iagos_cor[p,1]),(md_r_era_iagos_cor[p,2])))
    outfile2.write(' \n')
outfile2.write('------------------ \n')



#--hist_2d
F,x=plt.subplots(1,3,figsize=(25,6),squeeze=False,gridspec_kw={'width_ratios':[1,1,1.2]})
x[0,0].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,0].hist2d(out_arr[13,:],r_out_ice[:],range=[[0,160],[0,160]],bins=80, cmap=plt.cm.jet,norm = LogNorm(),vmin=0.1,vmax=20000)

x[0,0].plot((0,180),(100,100),linewidth=2,linestyle='dashed',color='k')
x[0,0].plot((0,180),(0,180),linewidth=2,linestyle='dashed',color='k')
x[0,0].plot((100,100),(0,180),linewidth=2,linestyle='dashed',color='k')
x[0,0].set_xlim(0,170)
x[0,0].set_ylim(0,170)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_xlabel('RH$_i$ IAGOS [%]',fontsize = 20)
x[0,0].set_ylabel('RH$_i$ [%]',fontsize = 20)
x[0,0].set_title('(a) ERA5',fontsize=20)

x[0,1].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,1].hist2d(out_arr[13,:],r_out_ice_cor2d[:],range=[[0,160],[0,160]],bins=80, cmap=plt.cm.jet,norm = LogNorm(),vmin=0.1,vmax=20000)

x[0,1].plot((0,180),(100,100),linewidth=2,linestyle='dashed',color='k')
x[0,1].plot((0,180),(0,180),linewidth=2,linestyle='dashed',color='k')
x[0,1].plot((100,100),(0,180),linewidth=2,linestyle='dashed',color='k')
x[0,1].set_xlim(0,170)
x[0,1].set_ylim(0,170)
x[0,1].set_yticklabels([])
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].set_xlabel('RH$_i$ IAGOS [%]',fontsize = 20)
x[0,1].set_title('(b) ERA5 QM',fontsize=20)

x[0,2].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,2].hist2d(out_arr[13,:],r_out_ice_teoh[:],range=[[0,160],[0,160]],bins=80, cmap=plt.cm.jet,norm = LogNorm(),vmin=0.1,vmax=20000)

x[0,2].plot((0,180),(100,100),linewidth=2,linestyle='dashed',color='k')
x[0,2].plot((0,180),(0,180),linewidth=2,linestyle='dashed',color='k')
x[0,2].plot((100,100),(0,180),linewidth=2,linestyle='dashed',color='k')
x[0,2].set_xlim(0,170)
x[0,2].set_ylim(0,170)
x[0,2].set_yticklabels([])
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].set_xlabel('RH$_i$ IAGOS [%]',fontsize = 20)
x[0,2].set_title('(c) ERA5 T22',fontsize=20)
cbar = plt.colorbar(x1[3],ax=x[0,2])
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Frequency of occurence',size=20)

filename = 'r_era_to_iagos_one-on-one_hist2d.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #plt.show()
    
#--r_out is the original with respect to liquid water
#--r_out_ice_cor is corrected with respect to ice
#--converting corrected ice profiles to liquid water because that is how the SA works
r_out_liq_cor = rh_ice_to_rh_liquid(r_out_ice_cor[:]/100, t_out[:])*100  #--temp and rh indepedent
r_out_liq_cor2d = rh_ice_to_rh_liquid(r_out_ice_cor2d[:]/100, t_out[:])*100 #--rh is temp denependen
r_out_liq_teoh = rh_ice_to_rh_liquid(r_out_ice_teoh[:]/100, t_out[:])*100 #--rh is temp denependen




#--Actuall analysis of the data
#--implement contrail criterion here
eta=0.3
rhi_crit = 1


#%%
######################################
##starting with uncorrected analysis##
######################################
print('######################################')
print('##starting with uncorrected analysis##')
print('######################################')


#-- test impact of measurment uncertainitnies
Tbias = +0 # negative adds a negative bias; makes meas colder / dryer
rhbias = -0/100 #-- no units between 0 an 1

cf_iagos_uncor = np.zeros((len(out_arr[0,:])))
cf_era_uncor = np.zeros((len(out_arr[0,:])))

crit_temp_profile = CritTemp_rasp(out_arr[7,:]+Tbias,out_arr[4,:],out_arr[11,:]/100+rhbias,eta) # temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
print(crit_temp_profile.shape)
pot_layers_index1 = np.where(((out_arr[7,:]+Tbias) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100+rhbias) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:]+Tbias - 273.15) <= -38))  #diagramgroup 1
pot_layers_index2 = np.where(((out_arr[7,:]+Tbias) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100+rhbias) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:]+Tbias - 273.15) <= -38))  #diagramgroup 2
pot_layers_index3 = np.where(((out_arr[7,:]+Tbias) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100+rhbias) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr[7,:]+Tbias - 273.15) <= -38))  #diagramgroup 3
cf_iagos_uncor[pot_layers_index2] = 2   #--Reservoir
cf_iagos_uncor[pot_layers_index3] = 3   #--non-persistent 
cf_iagos_uncor[pot_layers_index1] = 1   #--persistent



#--smoothed iagos
#################
cf_iagos_smooth_uncor = np.zeros((len(out_arr_smooth[0,:])))
crit_temp_profile = CritTemp_rasp(out_arr_smooth[7,:]+Tbias,out_arr_smooth[4,:],out_arr_smooth[11,:]/100+rhbias,eta) # temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((out_arr_smooth[7,:]+Tbias) < crit_temp_profile[0,:]) & ((out_arr_smooth[11,:]/100+rhbias) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr_smooth[7,:]+Tbias - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((out_arr_smooth[7,:]+Tbias) < crit_temp_profile[0,:]) & ((out_arr_smooth[11,:]/100+rhbias) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr_smooth[7,:]+Tbias - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((out_arr_smooth[7,:]+Tbias) < crit_temp_profile[0,:]) & ((out_arr_smooth[11,:]/100+rhbias) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr_smooth[7,:]+Tbias - 273.15) <= -38))  #--diagramgroup 3
cf_iagos_smooth_uncor[pot_layers_index2] = 2   #--Reservoir
cf_iagos_smooth_uncor[pot_layers_index3] = 3   #--non-persistent 
cf_iagos_smooth_uncor[pot_layers_index1] = 1   #--persistent

pres_arr = pres
crit_temp_profile = CritTemp_rasp(t_out,pres_out*100,r_out/100,eta) # temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out - 273.15) <= -38))  #diagramgroup 1
pot_layers_index2 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out - 273.15) <= -38))  #diagramgroup 2
pot_layers_index3 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((t_out - 273.15) <= -38))  #diagramgroup 3
cf_era_uncor[pot_layers_index2] = 2   #--Reservoir
cf_era_uncor[pot_layers_index3] = 3   #--non-persistent 
cf_era_uncor[pot_layers_index1] = 1   #--persistent


outfile2.write('\n')                                                                                                                                                                    
outfile2.write('------\n')                                                                                                                                                              
outfile2.write('Values below / above threshold uncorrected \n.')                                                                                                              
#--number of data points above or below the critical temperature                                                                                                                        
n_below_t_crit = np.nansum((t_out) < crit_temp_profile[0,:])                                                                                                                        
n_above_t_crit = np.nansum((t_out) >= crit_temp_profile[0,:])                                                                                                                       
                                                                                                                                                                                        
outfile2.write('T below crit: %09.4f \n' % (n_below_t_crit / len(t_out)))                                                                                        
outfile2.write('T above crit: %09.4f \n' % (n_above_t_crit / len(t_out)))                                                                                                                                                                          
#--number of data points above or below the critical humidity                                                                                                                           
n_below_rh_crit = np.nansum(r_out/100 < crit_temp_profile[1,:])                                                                                                               
n_above_rh_crit = np.nansum(r_out/100 >= crit_temp_profile[1,:])                                                                                                              
outfile2.write('rh below crit: %09.4f \n' % (n_below_rh_crit / len(t_out)))                                                    
outfile2.write('rh above crit: %09.4f \n' % (n_above_rh_crit / len(t_out)))                                                           
outfile2.write('-------\n')                                                                                                                                                             
outfile2.write('\n')                                                                                                                                                                    
                                                                                                                                                                                        

print('For IAGOS')
print('=========')
print('NPC ',np.nansum((cf_iagos_uncor == 3)) / np.nansum((cf_iagos_uncor >=0))*100)
print('PC ',np.nansum((cf_iagos_uncor == 1)) / np.nansum((cf_iagos_uncor >=0))*100)
print('R ',np.nansum((cf_iagos_uncor == 2)) / np.nansum((cf_iagos_uncor >=0))*100)

npc_iagos = np.nansum((cf_iagos_uncor == 3)) / np.nansum((cf_iagos_uncor >=0))*100
pc_iagos = np.nansum((cf_iagos_uncor == 1)) / np.nansum((cf_iagos_uncor >=0))*100
r_iagos = np.nansum((cf_iagos_uncor == 2)) / np.nansum((cf_iagos_uncor >=0))*100

print('For ERA')
print('=========')
print('NPC ',np.nansum((cf_era_uncor == 3)) / np.nansum((cf_era_uncor >=0))*100)
print('PC ',np.nansum((cf_era_uncor == 1)) / np.nansum((cf_era_uncor >=0))*100)
print('R ',np.nansum((cf_era_uncor == 2)) / np.nansum((cf_era_uncor >=0))*100)
print('')
print('NPC')

npc_era = np.nansum((cf_era_uncor == 3)) / np.nansum((cf_era_uncor >=0))*100
pc_era = np.nansum((cf_era_uncor == 1)) / np.nansum((cf_era_uncor >=0))*100
r_era = np.nansum((cf_era_uncor == 2)) / np.nansum((cf_era_uncor >=0))*100

outfile2.write('\n')
outfile2.write('\n')
outfile2.write('Classification uncorrected\n')
outfile2.write('===========================\n')
outfile2.write('\n')
outfile2.write('For IAGOS \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_iagos_uncor == 3)) / np.nansum((cf_iagos_uncor >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_iagos_uncor == 1)) / np.nansum((cf_iagos_uncor >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_iagos_uncor == 2)) / np.nansum((cf_iagos_uncor >=0))*100)+' \n')
outfile2.write('None '+str(np.nansum((cf_iagos_uncor == 0)) / np.nansum((cf_iagos_uncor >=0))*100)+' \n')
outfile2.write('For ERA \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_era_uncor == 3)) / np.nansum((cf_era_uncor >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_era_uncor == 1)) / np.nansum((cf_era_uncor >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_era_uncor == 2)) / np.nansum((cf_era_uncor >=0))*100)+' \n')
outfile2.write('None '+str(np.nansum((cf_era_uncor == 0)) / np.nansum((cf_era_uncor >=0))*100)+' \n')
outfile2.write('')


out1 = np.asarray([npc_iagos,pc_iagos,r_iagos,npc_era,pc_era,r_era])


F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)


print('')
print('NPC')
print('==')
total = np.nansum(((cf_iagos_uncor >= 0) | (cf_era_uncor >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor == 3) & (cf_era_uncor == 3)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor == 3) & (cf_era_uncor != 3)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor != 3) & (cf_era_uncor == 3)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor != 3) & (cf_era_uncor != 3)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))


outfile2.write('\n')
outfile2.write('NPC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out2 = np.asarray([true_positive,true_negative, false_negative, false_positive]) 

x[0,0].bar(1,true_positive,0.5,0,color='blue')
x[0,0].bar(1,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(1,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(1,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('PC')
print('==')
total = np.nansum(((cf_iagos_uncor >= 0) | (cf_era_uncor >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor == 1) & (cf_era_uncor == 1)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor == 1) & (cf_era_uncor != 1)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor != 1) & (cf_era_uncor == 1)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor != 1) & (cf_era_uncor != 1)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)


print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('PC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out3 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(3,true_positive,0.5,0,color='blue')
x[0,0].bar(3,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(3,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(3,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('R')
print('=')
total = np.nansum(((cf_iagos_uncor >= 0) | (cf_era_uncor >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor == 2) & (cf_era_uncor == 2)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor == 2) & (cf_era_uncor != 2)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor != 2) & (cf_era_uncor == 2)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor != 2) & (cf_era_uncor != 2)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)


print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('R\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out4 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(5,true_positive,0.5,0,color='blue')
x[0,0].bar(5,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(5,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(5,false_negative,0.5,true_negative+true_positive+false_positive,color='red')
x[0,0].set_xlim(0,6)
x[0,0].set_xticklabels(['','NPC','','PC','','R'])
x[0,0].set_ylim(0,110)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('ERA5 1h extraction',fontsize = 20)
x[0,0].set_xlabel('Contrail Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')

filename = "classification_quality_uncorrected.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#############
#--save the stats
if server == 0:
    save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/stats_file_uncorrected.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)
if server == 1:
    save_stats_file = '/homedata/kwolf/40_era_iagos/stats_file_uncorrected.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)


#%%
#########################################
##starting with uncorrected analysis#####
##3 Hour exraction
#########################################
print('####################################################')
print('##starting with uncorrected analysis 3h extraction##')
print('####################################################')


cf_iagos_uncor_3h = np.zeros((len(out_arr_3h[0,:])))
cf_era_uncor_3h = np.zeros((len(out_arr_3h[0,:])))

crit_temp_profile = CritTemp_rasp(out_arr_3h[7,:],out_arr_3h[4,:],out_arr_3h[11,:]/100,eta) # temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((out_arr_3h[7,:]) < crit_temp_profile[0,:]) & ((out_arr_3h[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr_3h[7,:] - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((out_arr_3h[7,:]) < crit_temp_profile[0,:]) & ((out_arr_3h[11,:]/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr_3h[7,:] - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((out_arr_3h[7,:]) < crit_temp_profile[0,:]) & ((out_arr_3h[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr_3h[7,:] - 273.15) <= -38))  #--diagramgroup 3
cf_iagos_uncor_3h[pot_layers_index2] = 2   #--Reservoir
cf_iagos_uncor_3h[pot_layers_index3] = 3   #--non-persistent 
cf_iagos_uncor_3h[pot_layers_index1] = 1   #--persistent


pres_arr = pres
crit_temp_profile = CritTemp_rasp(t_out_3h,pres_out_3h*100,r_out_3h/100,eta) # temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((t_out_3h) < crit_temp_profile[0,:]) & ((r_out_3h/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_3h - 273.15) <= -38))  #diagramgroup 1
pot_layers_index2 = np.where(((t_out_3h) < crit_temp_profile[0,:]) & ((r_out_3h/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_3h - 273.15) <= -38))  #diagramgroup 2
pot_layers_index3 = np.where(((t_out_3h) < crit_temp_profile[0,:]) & ((r_out_3h/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((t_out_3h - 273.15) <= -38))  #diagramgroup 3
cf_era_uncor_3h[pot_layers_index2] = 2   #--Reservoir
cf_era_uncor_3h[pot_layers_index3] = 3   #--non-persistent 
cf_era_uncor_3h[pot_layers_index1] = 1   #--persistent


print('For IAGOS')
print('=========')
print('NPC ',np.nansum((cf_iagos_uncor_3h == 3)) / np.nansum((cf_iagos_uncor_3h >=0))*100)
print('PC ',np.nansum((cf_iagos_uncor_3h == 1)) / np.nansum((cf_iagos_uncor_3h >=0))*100)
print('R ',np.nansum((cf_iagos_uncor_3h == 2)) / np.nansum((cf_iagos_uncor_3h >=0))*100)

npc_iagos = np.nansum((cf_iagos_uncor_3h == 3)) / np.nansum((cf_iagos_uncor_3h >=0))*100
pc_iagos = np.nansum((cf_iagos_uncor_3h == 1)) / np.nansum((cf_iagos_uncor_3h >=0))*100
r_iagos = np.nansum((cf_iagos_uncor_3h == 2)) / np.nansum((cf_iagos_uncor_3h >=0))*100

print('For ERA')
print('=========')
print('NPC ',np.nansum((cf_era_uncor_3h == 3)) / np.nansum((cf_era_uncor_3h >=0))*100)
print('PC ',np.nansum((cf_era_uncor_3h == 1)) / np.nansum((cf_era_uncor_3h >=0))*100)
print('R ',np.nansum((cf_era_uncor_3h == 2)) / np.nansum((cf_era_uncor_3h >=0))*100)
print('')
print('NPC')

npc_era = np.nansum((cf_era_uncor_3h == 3)) / np.nansum((cf_era_uncor_3h >=0))*100
pc_era = np.nansum((cf_era_uncor_3h == 1)) / np.nansum((cf_era_uncor_3h >=0))*100
r_era = np.nansum((cf_era_uncor_3h == 2)) / np.nansum((cf_era_uncor_3h >=0))*100

outfile2.write('\n')
outfile2.write('\n')
outfile2.write('Classification 3h extraction uncorrected\n')
outfile2.write('===========================\n')
outfile2.write('\n')
outfile2.write('For IAGOS \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_iagos_uncor_3h == 3)) / np.nansum((cf_iagos_uncor_3h >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_iagos_uncor_3h == 1)) / np.nansum((cf_iagos_uncor_3h >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_iagos_uncor_3h == 2)) / np.nansum((cf_iagos_uncor_3h >=0))*100)+' \n')
outfile2.write('For ERA \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_era_uncor_3h == 3)) / np.nansum((cf_era_uncor_3h >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_era_uncor_3h == 1)) / np.nansum((cf_era_uncor_3h >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_era_uncor_3h == 2)) / np.nansum((cf_era_uncor_3h >=0))*100)+' \n')
outfile2.write('')


out1 = np.asarray([npc_iagos,pc_iagos,r_iagos,npc_era,pc_era,r_era])


F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)


print('')
print('NPC')
print('==')
total = np.nansum(((cf_iagos_uncor_3h >= 0) | (cf_era_uncor_3h >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor_3h == 3) & (cf_era_uncor_3h == 3)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor_3h == 3) & (cf_era_uncor_3h != 3)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor_3h != 3) & (cf_era_uncor_3h == 3)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor_3h != 3) & (cf_era_uncor_3h != 3)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))


outfile2.write('\n')
outfile2.write('NPC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out2 = np.asarray([true_positive,true_negative, false_negative, false_positive]) 

x[0,0].bar(1,true_positive,0.5,0,color='blue')
x[0,0].bar(1,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(1,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(1,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('PC')
print('==')
total = np.nansum(((cf_iagos_uncor_3h >= 0) | (cf_era_uncor_3h >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor_3h == 1) & (cf_era_uncor_3h == 1)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor_3h == 1) & (cf_era_uncor_3h != 1)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor_3h != 1) & (cf_era_uncor_3h == 1)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor_3h != 1) & (cf_era_uncor_3h != 1)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('PC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out3 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(3,true_positive,0.5,0,color='blue')
x[0,0].bar(3,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(3,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(3,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('R')
print('=')
total = np.nansum(((cf_iagos_uncor_3h >= 0) | (cf_era_uncor_3h >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor_3h == 2) & (cf_era_uncor_3h == 2)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor_3h == 2) & (cf_era_uncor_3h != 2)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor_3h != 2) & (cf_era_uncor_3h == 2)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor_3h != 2) & (cf_era_uncor_3h != 2)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('R\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out4 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(5,true_positive,0.5,0,color='blue')
x[0,0].bar(5,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(5,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(5,false_negative,0.5,true_negative+true_positive+false_positive,color='red')
x[0,0].set_xlim(0,6)
x[0,0].set_xticklabels(['','NPC','','PC','','R'])
x[0,0].set_ylim(0,110)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('ERA5 3h extraction',fontsize = 20)
x[0,0].set_xlabel('Contrail Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')

filename = "classification_quality_uncorrected_3h.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#############
#save the stats
if server == 0:
    save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/stats_file_uncorrected_3h.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)
if server == 1:
    save_stats_file = '/homedata/kwolf/40_era_iagos/stats_file_uncorrected_3h.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)


#%%
#########################################
##starting with uncorrected analysis#####
##6 Hour exraction
#########################################
print('#######################################################')
print('##starting with uncorrected analysis 6hour extraction##')
print('#######################################################')


cf_iagos_uncor_6h = np.zeros((len(out_arr_6h[0,:])))
cf_era_uncor_6h = np.zeros((len(out_arr_6h[0,:])))

crit_temp_profile = CritTemp_rasp(out_arr_6h[7,:],out_arr_6h[4,:],out_arr_6h[11,:]/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((out_arr_6h[7,:]) < crit_temp_profile[0,:]) & ((out_arr_6h[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr_6h[7,:] - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((out_arr_6h[7,:]) < crit_temp_profile[0,:]) & ((out_arr_6h[11,:]/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr_6h[7,:] - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((out_arr_6h[7,:]) < crit_temp_profile[0,:]) & ((out_arr_6h[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr_6h[7,:] - 273.15) <= -38))  #--diagramgroup 3
cf_iagos_uncor_6h[pot_layers_index2] = 2   #-- Reservoir
cf_iagos_uncor_6h[pot_layers_index3] = 3   #-- non-persistent 
cf_iagos_uncor_6h[pot_layers_index1] = 1     #--persistent


pres_arr = pres
crit_temp_profile = CritTemp_rasp(t_out_6h,pres_out_6h*100,r_out_6h/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((t_out_6h) < crit_temp_profile[0,:]) & ((r_out_6h/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_6h - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((t_out_6h) < crit_temp_profile[0,:]) & ((r_out_6h/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_6h - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((t_out_6h) < crit_temp_profile[0,:]) & ((r_out_6h/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((t_out_6h - 273.15) <= -38))  #--diagramgroup 3
cf_era_uncor_6h[pot_layers_index2] = 2   #-- Reservoir
cf_era_uncor_6h[pot_layers_index3] = 3   #-- non-persistent 
cf_era_uncor_6h[pot_layers_index1] = 1     #--persistent


print('For IAGOS')
print('=========')
print('NPC ',np.nansum((cf_iagos_uncor_6h == 3)) / np.nansum((cf_iagos_uncor_6h >=0))*100)
print('PC ',np.nansum((cf_iagos_uncor_6h == 1)) / np.nansum((cf_iagos_uncor_6h >=0))*100)
print('R ',np.nansum((cf_iagos_uncor_6h == 2)) / np.nansum((cf_iagos_uncor_6h >=0))*100)

npc_iagos = np.nansum((cf_iagos_uncor_6h == 3)) / np.nansum((cf_iagos_uncor_6h >=0))*100
pc_iagos = np.nansum((cf_iagos_uncor_6h == 1)) / np.nansum((cf_iagos_uncor_6h >=0))*100
r_iagos = np.nansum((cf_iagos_uncor_6h == 2)) / np.nansum((cf_iagos_uncor_6h >=0))*100

print('For ERA')
print('=========')
print('NPC ',np.nansum((cf_era_uncor_6h == 3)) / np.nansum((cf_era_uncor_6h >=0))*100)
print('PC ',np.nansum((cf_era_uncor_6h == 1)) / np.nansum((cf_era_uncor_6h >=0))*100)
print('R ',np.nansum((cf_era_uncor_6h == 2)) / np.nansum((cf_era_uncor_6h >=0))*100)
print('')
print('NPC')

npc_era = np.nansum((cf_era_uncor_6h == 3)) / np.nansum((cf_era_uncor_6h >=0))*100
pc_era = np.nansum((cf_era_uncor_6h == 1)) / np.nansum((cf_era_uncor_6h >=0))*100
r_era = np.nansum((cf_era_uncor_6h == 2)) / np.nansum((cf_era_uncor_6h >=0))*100

outfile2.write('\n')
outfile2.write('\n')
outfile2.write('Classification 6h extraction uncorrected\n')
outfile2.write('===========================\n')
outfile2.write('\n')
outfile2.write('For IAGOS \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_iagos_uncor_6h == 3)) / np.nansum((cf_iagos_uncor_6h >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_iagos_uncor_6h == 1)) / np.nansum((cf_iagos_uncor_6h >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_iagos_uncor_6h == 2)) / np.nansum((cf_iagos_uncor_6h >=0))*100)+' \n')
outfile2.write('For ERA \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_era_uncor_6h == 3)) / np.nansum((cf_era_uncor_6h >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_era_uncor_6h == 1)) / np.nansum((cf_era_uncor_6h >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_era_uncor_6h == 2)) / np.nansum((cf_era_uncor_6h >=0))*100)+' \n')
outfile2.write('')


out1 = np.asarray([npc_iagos,pc_iagos,r_iagos,npc_era,pc_era,r_era])


F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)


print('')
print('NPC')
print('==')
total = np.nansum(((cf_iagos_uncor_6h >= 0) | (cf_era_uncor_6h >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor_6h == 3) & (cf_era_uncor_6h == 3)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor_6h == 3) & (cf_era_uncor_6h != 3)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor_6h != 3) & (cf_era_uncor_6h == 3)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor_6h != 3) & (cf_era_uncor_6h != 3)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))


outfile2.write('\n')
outfile2.write('NPC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out2 = np.asarray([true_positive,true_negative, false_negative, false_positive]) 

x[0,0].bar(1,true_positive,0.5,0,color='blue')
x[0,0].bar(1,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(1,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(1,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('PC')
print('==')
total = np.nansum(((cf_iagos_uncor_6h >= 0) | (cf_era_uncor_6h >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor_6h == 1) & (cf_era_uncor_6h == 1)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor_6h == 1) & (cf_era_uncor_6h != 1)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor_6h != 1) & (cf_era_uncor_6h == 1)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor_6h != 1) & (cf_era_uncor_6h != 1)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('PC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out3 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(3,true_positive,0.5,0,color='blue')
x[0,0].bar(3,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(3,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(3,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('R')
print('=')
total = np.nansum(((cf_iagos_uncor_6h >= 0) | (cf_era_uncor_6h >= 0)))
true_positive   = np.nansum(((cf_iagos_uncor_6h == 2) & (cf_era_uncor_6h == 2)))/ total * 100
false_negative = np.nansum(((cf_iagos_uncor_6h == 2) & (cf_era_uncor_6h != 2)))/ total * 100
false_positive = np.nansum(((cf_iagos_uncor_6h != 2) & (cf_era_uncor_6h == 2)))/ total * 100
true_negative  = np.nansum(((cf_iagos_uncor_6h != 2) & (cf_era_uncor_6h != 2)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('R\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out4 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(5,true_positive,0.5,0,color='blue')
x[0,0].bar(5,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(5,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(5,false_negative,0.5,true_negative+true_positive+false_positive,color='red')
x[0,0].set_xlim(0,6)
x[0,0].set_xticklabels(['','NPC','','PC','','R'])
x[0,0].set_ylim(0,110)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('ERA5 6h extraction',fontsize = 20)
x[0,0].set_xlabel('Contrail Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')

filename = "classification_quality_uncorrected_6h.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#############
#save the stats
if server == 0:
    save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/stats_file_uncorrected_6h.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)
if server == 1:
    save_stats_file = '/homedata/kwolf/40_era_iagos/stats_file_uncorrected_6h.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)


#%%
#####################################################
##starting with corrected analysis TEMPERATURE ONLY##
#####################################################


cf_iagos_cor_t = np.zeros((len(out_arr[0,:])))
cf_era_cor_t = np.zeros((len(out_arr[0,:])))

crit_temp_profile = CritTemp_rasp(out_arr[7,:],out_arr[4,:],out_arr[11,:]/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 3
cf_iagos_cor_t[pot_layers_index2] = 2   #-- Reservoir
cf_iagos_cor_t[pot_layers_index3] = 3   #-- non-persistent 
cf_iagos_cor_t[pot_layers_index1] = 1     #--persistent


pres_arr = pres
crit_temp_profile = CritTemp_rasp(t_out_cor,pres_out*100,r_out/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((t_out_cor) < crit_temp_profile[0,:]) & ((r_out/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_cor - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((t_out_cor) < crit_temp_profile[0,:]) & ((r_out/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_cor - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((t_out_cor) < crit_temp_profile[0,:]) & ((r_out/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((t_out_cor - 273.15) <= -38))  #--diagramgroup 3
cf_era_cor_t[pot_layers_index2] = 2   #-- Reservoir
cf_era_cor_t[pot_layers_index3] = 3   #-- non-persistent 
cf_era_cor_t[pot_layers_index1] = 1     #--persistent

print('For IAGOS')
print('=========')
print('NPC ',np.nansum((cf_iagos_cor_t == 3)) / np.nansum((cf_iagos_cor_t >=0))*100)
print('PC ',np.nansum((cf_iagos_cor_t == 1)) / np.nansum((cf_iagos_cor_t >=0))*100)
print('R ',np.nansum((cf_iagos_cor_t == 2)) / np.nansum((cf_iagos_cor_t >=0))*100)

npc_iagos = np.nansum((cf_iagos_cor_t == 3)) / np.nansum((cf_iagos_cor_t >=0))*100
pc_iagos = np.nansum((cf_iagos_cor_t == 1)) / np.nansum((cf_iagos_cor_t >=0))*100
r_iagos = np.nansum((cf_iagos_cor_t == 2)) / np.nansum((cf_iagos_cor_t >=0))*100

print('For ERA')
print('=========')
print('NPC ',np.nansum((cf_era_cor_t == 3)) / np.nansum((cf_era_cor_t >=0))*100)
print('PC ',np.nansum((cf_era_cor_t == 1)) / np.nansum((cf_era_cor_t >=0))*100)
print('R ',np.nansum((cf_era_cor_t == 2)) / np.nansum((cf_era_cor_t >=0))*100)
print('')
print('NPC')

npc_era = np.nansum((cf_era_cor_t == 3)) / np.nansum((cf_era_cor_t >=0))*100
pc_era = np.nansum((cf_era_cor_t == 1)) / np.nansum((cf_era_cor_t >=0))*100
r_era = np.nansum((cf_era_cor_t == 2)) / np.nansum((cf_era_cor_t >=0))*100

outfile2.write('\n')
outfile2.write('Classification T - corrected\n')
outfile2.write('===========================\n')
outfile2.write('\n')
outfile2.write('For IAGOS \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_iagos_cor_t == 3)) / np.nansum((cf_iagos_cor_t >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_iagos_cor_t == 1)) / np.nansum((cf_iagos_cor_t >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_iagos_cor_t == 2)) / np.nansum((cf_iagos_cor_t >=0))*100)+' \n')
outfile2.write('None '+str(np.nansum((cf_iagos_cor_t == 0)) / np.nansum((cf_iagos_cor_t >=0))*100)+' \n')
outfile2.write('For ERA \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_era_cor_t == 3)) / np.nansum((cf_era_cor_t >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_era_cor_t == 1)) / np.nansum((cf_era_cor_t >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_era_cor_t == 2)) / np.nansum((cf_era_cor_t >=0))*100)+' \n')
outfile2.write('None '+str(np.nansum((cf_era_cor_t == 0)) / np.nansum((cf_era_cor_t >=0))*100)+' \n')
outfile2.write('')

out1 = np.asarray([npc_iagos,pc_iagos,r_iagos,npc_era,pc_era,r_era])


F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)


print('')
print('NPC')
print('==')
total = np.nansum(((cf_iagos_cor_t >= 0) | (cf_era_cor_t >= 0)))
true_positive   = np.nansum(((cf_iagos_cor_t == 3) & (cf_era_cor_t == 3)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor_t == 3) & (cf_era_cor_t != 3)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor_t != 3) & (cf_era_cor_t == 3)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor_t != 3) & (cf_era_cor_t != 3)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))

outfile2.write('\n')
outfile2.write('NPC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out2 = np.asarray([true_positive,true_negative, false_negative, false_positive]) 

x[0,0].bar(1,true_positive,0.5,0,color='blue')
x[0,0].bar(1,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(1,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(1,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('PC')
print('==')
total = np.nansum(((cf_iagos_cor_t >= 0) | (cf_era_cor_t >= 0)))
true_positive   = np.nansum(((cf_iagos_cor_t == 1) & (cf_era_cor_t == 1)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor_t == 1) & (cf_era_cor_t != 1)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor_t != 1) & (cf_era_cor_t == 1)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor_t != 1) & (cf_era_cor_t != 1)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))

outfile2.write('\n')
outfile2.write('NP\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out3 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(3,true_positive,0.5,0,color='blue')
x[0,0].bar(3,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(3,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(3,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('R')
print('=')
total = np.nansum(((cf_iagos_cor_t >= 0) | (cf_era_cor_t >= 0)))
true_positive   = np.nansum(((cf_iagos_cor_t == 2) & (cf_era_cor_t == 2)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor_t == 2) & (cf_era_cor_t != 2)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor_t != 2) & (cf_era_cor_t == 2)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor_t != 2) & (cf_era_cor_t != 2)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('R\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out4 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(5,true_positive,0.5,0,color='blue')
x[0,0].bar(5,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(5,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(5,false_negative,0.5,true_negative+true_positive+false_positive,color='red')
x[0,0].set_xlim(0,6)
x[0,0].set_xticklabels(['','NPC','','PC','','R'])
x[0,0].set_ylim(0,110)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
#--x[0,0].set_title('10-11km',fontsize = 20)
x[0,0].set_xlabel('Contrail Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
    
filename = "classification_quality_corrected_t.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#############
#save the stats
if server == 0:
    save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/stats_file_corrected_t.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)
if server == 1:
    save_stats_file = '/homedata/kwolf/40_era_iagos/stats_file_corrected_t.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)


#%%

##############################################
##starting with corrected analysis - rh only 2d##
##############################################


cf_iagos_cor_r = np.zeros((len(out_arr[0,:])))
cf_era_cor_r = np.zeros((len(out_arr[0,:])))

crit_temp_profile = CritTemp_rasp(out_arr[7,:],out_arr[4,:],out_arr[11,:]/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 3
cf_iagos_cor_r[pot_layers_index2] = 2   #-- Reservoir
cf_iagos_cor_r[pot_layers_index3] = 3   #-- non-persistent 
cf_iagos_cor_r[pot_layers_index1] = 1     #--persistent


pres_arr = pres
crit_temp_profile = CritTemp_rasp(t_out,pres_out*100,r_out_liq_cor2d/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out_liq_cor2d/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out_liq_cor2d/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out_liq_cor2d/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((t_out_cor - 273.15) <= -38))  #--diagramgroup 3
cf_era_cor_r[pot_layers_index2] = 2   #-- Reservoir
cf_era_cor_r[pot_layers_index3] = 3   #-- non-persistent 
cf_era_cor_r[pot_layers_index1] = 1     #--persistent

print('For IAGOS')
print('=========')
print('NPC ',np.nansum((cf_iagos_cor_r == 3)) / np.nansum((cf_iagos_cor_r >=0))*100)
print('PC ',np.nansum((cf_iagos_cor_r == 1)) / np.nansum((cf_iagos_cor_r >=0))*100)
print('R ',np.nansum((cf_iagos_cor_r == 2)) / np.nansum((cf_iagos_cor_r >=0))*100)

npc_iagos = np.nansum((cf_iagos_cor_r == 3)) / np.nansum((cf_iagos_cor_r >=0))*100
pc_iagos = np.nansum((cf_iagos_cor_r == 1)) / np.nansum((cf_iagos_cor_r >=0))*100
r_iagos = np.nansum((cf_iagos_cor_r == 2)) / np.nansum((cf_iagos_cor_r >=0))*100

print('For ERA')
print('=========')
print('NPC ',np.nansum((cf_era_cor_r == 3)) / np.nansum((cf_era_cor_r >=0))*100)
print('PC ',np.nansum((cf_era_cor_r == 1)) / np.nansum((cf_era_cor_r >=0))*100)
print('R ',np.nansum((cf_era_cor_r == 2)) / np.nansum((cf_era_cor_r >=0))*100)
print('')
print('NPC')

npc_era = np.nansum((cf_era_cor_r == 3)) / np.nansum((cf_era_cor_r >=0))*100
pc_era = np.nansum((cf_era_cor_r == 1)) / np.nansum((cf_era_cor_r >=0))*100
r_era = np.nansum((cf_era_cor_r == 2)) / np.nansum((cf_era_cor_r >=0))*100

outfile2.write('\n')
outfile2.write('Classification rh- corrected with 2d correction\n')
outfile2.write('===========================\n')
outfile2.write('\n')
outfile2.write('For IAGOS \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_iagos_cor_r == 3)) / np.nansum((cf_iagos_cor_r >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_iagos_cor_r == 1)) / np.nansum((cf_iagos_cor_r >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_iagos_cor_r == 2)) / np.nansum((cf_iagos_cor_r >=0))*100)+' \n')
outfile2.write('None '+str(np.nansum((cf_iagos_cor_r == 0)) / np.nansum((cf_iagos_cor_r >=0))*100)+' \n')
outfile2.write('For ERA \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_era_cor_r == 3)) / np.nansum((cf_era_cor_r >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_era_cor_r == 1)) / np.nansum((cf_era_cor_r >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_era_cor_r == 2)) / np.nansum((cf_era_cor_r >=0))*100)+' \n')
outfile2.write('None '+str(np.nansum((cf_era_cor_r == 0)) / np.nansum((cf_era_cor_r >=0))*100)+' \n')
outfile2.write('')

out1 = np.asarray([npc_iagos,pc_iagos,r_iagos,npc_era,pc_era,r_era])


F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)


print('')
print('NPC')
print('==')
total = np.nansum(((cf_iagos_cor_r >= 0) | (cf_era_cor_r >= 0)))
true_positive   = np.nansum(((cf_iagos_cor_r == 3) & (cf_era_cor_r == 3)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor_r == 3) & (cf_era_cor_r != 3)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor_r != 3) & (cf_era_cor_r == 3)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor_r != 3) & (cf_era_cor_r != 3)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))

outfile2.write('\n')
outfile2.write('NPC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out2 = np.asarray([true_positive,true_negative, false_negative, false_positive]) 

x[0,0].bar(1,true_positive,0.5,0,color='blue')
x[0,0].bar(1,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(1,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(1,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('PC')
print('==')
total = np.nansum(((cf_iagos_cor_r >= 0) | (cf_era_cor_r >= 0)))
true_positive   = np.nansum(((cf_iagos_cor_r == 1) & (cf_era_cor_r == 1)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor_r == 1) & (cf_era_cor_r != 1)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor_r != 1) & (cf_era_cor_r == 1)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor_r != 1) & (cf_era_cor_r != 1)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))

outfile2.write('\n')
outfile2.write('NP\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out3 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(3,true_positive,0.5,0,color='blue')
x[0,0].bar(3,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(3,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(3,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('R')
print('=')
total = np.nansum(((cf_iagos_cor_r >= 0) | (cf_era_cor_r >= 0)))
true_positive   = np.nansum(((cf_iagos_cor_r == 2) & (cf_era_cor_r == 2)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor_r == 2) & (cf_era_cor_r != 2)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor_r != 2) & (cf_era_cor_r == 2)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor_r != 2) & (cf_era_cor_r != 2)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('R\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')


#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')
out4 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(5,true_positive,0.5,0,color='blue')
x[0,0].bar(5,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(5,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(5,false_negative,0.5,true_negative+true_positive+false_positive,color='red')
x[0,0].set_xlim(0,6)
x[0,0].set_xticklabels(['','NPC','','PC','','R'])
x[0,0].set_ylim(0,110)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_xlabel('Contrail Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
    
filename = "classification_quality_corrected_r.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#############
#save the stats
if server == 0:
    save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/stats_file_corrected_r.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)
if server == 1:
    save_stats_file = '/homedata/kwolf/40_era_iagos/stats_file_corrected_r.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)



#%%

######################################
##starting with 2d corrected analysis##
######################################


cf_iagos_cor2d = np.zeros((len(out_arr[0,:])))
cf_era_cor2d = np.zeros((len(out_arr[0,:])))

crit_temp_profile = CritTemp_rasp(out_arr[7,:],out_arr[4,:],out_arr[11,:]/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 3
cf_iagos_cor2d[pot_layers_index2] = 2   #-- Reservoir
cf_iagos_cor2d[pot_layers_index3] = 3   #-- non-persistent 
cf_iagos_cor2d[pot_layers_index1] = 1     #--persistent


pres_arr = pres
crit_temp_profile = CritTemp_rasp(t_out_cor,pres_out*100,r_out_liq_cor2d/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((t_out_cor) < crit_temp_profile[0,:]) & ((r_out_liq_cor2d/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_cor - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((t_out_cor) < crit_temp_profile[0,:]) & ((r_out_liq_cor2d/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out_cor - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((t_out_cor) < crit_temp_profile[0,:]) & ((r_out_liq_cor2d/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((t_out_cor - 273.15) <= -38))  #--diagramgroup 3
cf_era_cor2d[pot_layers_index2] = 2   #-- Reservoir
cf_era_cor2d[pot_layers_index3] = 3   #-- non-persistent 
cf_era_cor2d[pot_layers_index1] = 1     #--persistent


outfile2.write('\n')
outfile2.write('------\n')
outfile2.write('Values below / above thresholds after the 2D correction.\n')
#--number of data points above or below the critical temperature                                                                                                                        
n_below_t_crit = np.nansum((t_out_cor) < crit_temp_profile[0,:])                                                                                                     
n_above_t_crit = np.nansum((t_out_cor) >= crit_temp_profile[0,:])                                                                                                      
                                                                                                                                                                                        
outfile2.write('T below crit: %09.4f \n' % (n_below_t_crit / len(t_out)))                                             
outfile2.write('T above crit: %09.4f \n' % (n_above_t_crit / len(t_out)))                                                                                                                               
#--number of data points above or below the critical humidity                                                                                                                           
n_below_rh_crit = np.nansum(r_out_liq_cor2d/100 < crit_temp_profile[1,:])                                                                                        
n_above_rh_crit = np.nansum(r_out_liq_cor2d/100 >= crit_temp_profile[1,:])                                                                               
outfile2.write('rh below crit: %09.4f \n' % (n_below_rh_crit / len(t_out)))                                                                 
outfile2.write('rh above crit: %09.4f \n' % (n_above_rh_crit / len(t_out)))
outfile2.write('-------\n')
outfile2.write('\n')

print('For IAGOS')
print('=========')
print('NPC ',np.nansum((cf_iagos_cor2d == 3)) / np.nansum((cf_iagos_cor2d >=0))*100)
print('PC ',np.nansum((cf_iagos_cor2d == 1)) / np.nansum((cf_iagos_cor2d >=0))*100)
print('R ',np.nansum((cf_iagos_cor2d == 2)) / np.nansum((cf_iagos_cor2d >=0))*100)

npc_iagos = np.nansum((cf_iagos_cor2d == 3)) / np.nansum((cf_iagos_cor2d >=0))*100
pc_iagos = np.nansum((cf_iagos_cor2d == 1)) / np.nansum((cf_iagos_cor2d >=0))*100
r_iagos = np.nansum((cf_iagos_cor2d == 2)) / np.nansum((cf_iagos_cor2d >=0))*100

print('For ERA')
print('=========')
print('NPC ',np.nansum((cf_era_cor2d == 3)) / np.nansum((cf_era_cor2d >=0))*100)
print('PC ',np.nansum((cf_era_cor2d == 1)) / np.nansum((cf_era_cor2d >=0))*100)
print('R ',np.nansum((cf_era_cor2d == 2)) / np.nansum((cf_era_cor2d >=0))*100)
print('')
print('NPC')

npc_era = np.nansum((cf_era_cor2d == 3)) / np.nansum((cf_era_cor2d >=0))*100
pc_era = np.nansum((cf_era_cor2d == 1)) / np.nansum((cf_era_cor2d >=0))*100
r_era = np.nansum((cf_era_cor2d == 2)) / np.nansum((cf_era_cor2d >=0))*100

outfile2.write('\n')
outfile2.write('\n')
outfile2.write('Classification 2d corrected\n')
outfile2.write('===========================\n')
outfile2.write('\n')
outfile2.write('For IAGOS \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_iagos_cor2d == 3)) / np.nansum((cf_iagos_cor2d >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_iagos_cor2d == 1)) / np.nansum((cf_iagos_cor2d >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_iagos_cor2d == 2)) / np.nansum((cf_iagos_cor2d >=0))*100)+' \n')
outfile2.write('None '+str(np.nansum((cf_iagos_cor2d == 0)) / np.nansum((cf_iagos_cor2d >=0))*100)+' \n')
outfile2.write('For ERA \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_era_cor2d == 3)) / np.nansum((cf_era_cor2d >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_era_cor2d == 1)) / np.nansum((cf_era_cor2d >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_era_cor2d == 2)) / np.nansum((cf_era_cor2d >=0))*100)+' \n')
outfile2.write('Non '+str(np.nansum((cf_era_cor2d == 0)) / np.nansum((cf_era_cor2d >=0))*100)+' \n')
outfile2.write('')

out1 = np.asarray([npc_iagos,pc_iagos,r_iagos,npc_era,pc_era,r_era])


F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)


print('')
print('NPC')
print('==')
total = np.nansum(((cf_iagos_cor2d >= 0) | (cf_era_cor2d >= 0)))
true_positive   = np.nansum(((cf_iagos_cor2d == 3) & (cf_era_cor2d == 3)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor2d == 3) & (cf_era_cor2d != 3)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor2d != 3) & (cf_era_cor2d == 3)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor2d != 3) & (cf_era_cor2d != 3)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('NPC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out2 = np.asarray([true_positive,true_negative, false_negative, false_positive]) 

x[0,0].bar(1,true_positive,0.5,0,color='blue')
x[0,0].bar(1,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(1,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(1,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('PC')
print('==')
total = np.nansum(((cf_iagos_cor2d >= 0) | (cf_era_cor2d >= 0)))
true_positive   = np.nansum(((cf_iagos_cor2d == 1) & (cf_era_cor2d == 1)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor2d == 1) & (cf_era_cor2d != 1)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor2d != 1) & (cf_era_cor2d == 1)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor2d != 1) & (cf_era_cor2d != 1)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('PC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out3 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(3,true_positive,0.5,0,color='blue')
x[0,0].bar(3,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(3,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(3,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('R')
print('=')
total = np.nansum(((cf_iagos_cor2d >= 0) | (cf_era_cor2d >= 0)))
true_positive   = np.nansum(((cf_iagos_cor2d == 2) & (cf_era_cor2d == 2)))/ total * 100
false_negative = np.nansum(((cf_iagos_cor2d == 2) & (cf_era_cor2d != 2)))/ total * 100
false_positive = np.nansum(((cf_iagos_cor2d != 2) & (cf_era_cor2d == 2)))/ total * 100
true_negative  = np.nansum(((cf_iagos_cor2d != 2) & (cf_era_cor2d != 2)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('R\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')
#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out4 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(5,true_positive,0.5,0,color='blue')
x[0,0].bar(5,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(5,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(5,false_negative,0.5,true_negative+true_positive+false_positive,color='red')
x[0,0].set_xlim(0,6)
x[0,0].set_xticklabels(['','NPC','','PC','','R'])
x[0,0].set_ylim(0,110)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_xlabel('Contrail Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
    
filename = "classification_quality_corrected_2d.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#############
#save the stats
if server == 0:
    save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/stats_file_corrected_2d.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)
if server == 1:
    save_stats_file = '/homedata/kwolf/40_era_iagos/stats_file_corrected_2d.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)


#%%

######################################
##starting with teoh corrected analysis##
######################################


cf_iagos_teoh = np.zeros((len(out_arr[0,:])))
cf_era_teoh = np.zeros((len(out_arr[0,:])))

crit_temp_profile = CritTemp_rasp(out_arr[7,:],out_arr[4,:],out_arr[11,:]/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((out_arr[7,:]) < crit_temp_profile[0,:]) & ((out_arr[11,:]/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((out_arr[7,:] - 273.15) <= -38))  #--diagramgroup 3
cf_iagos_teoh[pot_layers_index2] = 2   #-- Reservoir
cf_iagos_teoh[pot_layers_index3] = 3   #-- non-persistent 
cf_iagos_teoh[pot_layers_index1] = 1     #--persistent


pres_arr = pres
crit_temp_profile = CritTemp_rasp(t_out,pres_out*100,r_out_liq_teoh/100,eta) #-- temperature in k, pressure in pa, and rel hum in 0-1
crit_temp_profile = crit_temp_profile[:,0,:].T
pot_layers_index1 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out_liq_teoh/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out - 273.15) <= -38))  #--diagramgroup 1
pot_layers_index2 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out_liq_teoh/100) <= crit_temp_profile[1,:])  & (crit_temp_profile[2,:] >= rhi_crit) & ((t_out - 273.15) <= -38))  #--diagramgroup 2
pot_layers_index3 = np.where(((t_out) < crit_temp_profile[0,:]) & ((r_out_liq_teoh/100) > crit_temp_profile[1,:])  & (crit_temp_profile[2,:] <= rhi_crit)& ((t_out_cor - 273.15) <= -38))  #--diagramgroup 3
cf_era_teoh[pot_layers_index2] = 2   #-- Reservoir
cf_era_teoh[pot_layers_index3] = 3   #-- non-persistent 
cf_era_teoh[pot_layers_index1] = 1     #--persistent

print('For IAGOS')
print('=========')
print('NPC ',np.nansum((cf_iagos_teoh == 3)) / np.nansum((cf_iagos_teoh >=0))*100)
print('PC ',np.nansum((cf_iagos_teoh == 1)) / np.nansum((cf_iagos_teoh >=0))*100)
print('R ',np.nansum((cf_iagos_teoh == 2)) / np.nansum((cf_iagos_teoh >=0))*100)

npc_iagos = np.nansum((cf_iagos_teoh == 3)) / np.nansum((cf_iagos_teoh >=0))*100
pc_iagos = np.nansum((cf_iagos_teoh == 1)) / np.nansum((cf_iagos_teoh >=0))*100
r_iagos = np.nansum((cf_iagos_teoh == 2)) / np.nansum((cf_iagos_teoh >=0))*100

print('For ERA')
print('=========')
print('NPC ',np.nansum((cf_era_teoh == 3)) / np.nansum((cf_era_teoh >=0))*100)
print('PC ',np.nansum((cf_era_teoh == 1)) / np.nansum((cf_era_teoh >=0))*100)
print('R ',np.nansum((cf_era_teoh == 2)) / np.nansum((cf_era_teoh >=0))*100)
print('')
print('NPC')

npc_era = np.nansum((cf_era_teoh == 3)) / np.nansum((cf_era_teoh >=0))*100
pc_era = np.nansum((cf_era_teoh == 1)) / np.nansum((cf_era_teoh >=0))*100
r_era = np.nansum((cf_era_teoh == 2)) / np.nansum((cf_era_teoh >=0))*100

outfile2.write('\n')
outfile2.write('\n')
outfile2.write('Classification 2d corrected teoh\n')
outfile2.write('===========================\n')
outfile2.write('\n')
outfile2.write('For IAGOS \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_iagos_teoh == 3)) / np.nansum((cf_iagos_teoh >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_iagos_teoh == 1)) / np.nansum((cf_iagos_teoh >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_iagos_teoh == 2)) / np.nansum((cf_iagos_teoh >=0))*100)+' \n')
outfile2.write('Non '+str(np.nansum((cf_iagos_teoh == 0)) / np.nansum((cf_iagos_teoh >=0))*100)+' \n')
outfile2.write('For ERA \n')
outfile2.write('=========\n')
outfile2.write('NPC '+str(np.nansum((cf_era_teoh == 3)) / np.nansum((cf_era_teoh >=0))*100)+' \n')
outfile2.write('PC '+str(np.nansum((cf_era_teoh == 1)) / np.nansum((cf_era_teoh >=0))*100)+' \n')
outfile2.write('R '+str(np.nansum((cf_era_teoh == 2)) / np.nansum((cf_era_teoh >=0))*100)+' \n')
outfile2.write('Non '+str(np.nansum((cf_era_teoh == 0)) / np.nansum((cf_era_teoh >=0))*100)+' \n')
outfile2.write('')

out1 = np.asarray([npc_iagos,pc_iagos,r_iagos,npc_era,pc_era,r_era])


F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)


print('')
print('NPC')
print('==')
total = np.nansum(((cf_iagos_cor2d >= 0) | (cf_era_cor2d >= 0)))
true_positive   = np.nansum(((cf_iagos_teoh == 3) & (cf_era_teoh == 3)))/ total * 100
false_negative = np.nansum(((cf_iagos_teoh == 3) & (cf_era_teoh != 3)))/ total * 100
false_positive = np.nansum(((cf_iagos_teoh != 3) & (cf_era_teoh == 3)))/ total * 100
true_negative  = np.nansum(((cf_iagos_teoh != 3) & (cf_era_teoh != 3)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))

print('Hitrate: ',hit_rate)
print('False alarm: ',false_alarm)

outfile2.write('\n')
outfile2.write('NPC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out2 = np.asarray([true_positive,true_negative, false_negative, false_positive]) 

x[0,0].bar(1,true_positive,0.5,0,color='blue')
x[0,0].bar(1,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(1,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(1,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('PC')
print('==')
total = np.nansum(((cf_iagos_teoh >= 0) | (cf_era_teoh >= 0)))
true_positive   = np.nansum(((cf_iagos_teoh == 1) & (cf_era_teoh == 1)))/ total * 100
false_negative = np.nansum(((cf_iagos_teoh == 1) & (cf_era_teoh != 1)))/ total * 100
false_positive = np.nansum(((cf_iagos_teoh != 1) & (cf_era_teoh == 1)))/ total * 100
true_negative  = np.nansum(((cf_iagos_teoh != 1) & (cf_era_teoh != 1)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('PC\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out3 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(3,true_positive,0.5,0,color='blue')
x[0,0].bar(3,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(3,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(3,false_negative,0.5,true_negative+true_positive+false_positive,color='red')

print('')
print('R')
print('=')
total = np.nansum(((cf_iagos_cor2d >= 0) | (cf_era_cor2d >= 0)))
true_positive   = np.nansum(((cf_iagos_teoh == 2) & (cf_era_teoh == 2)))/ total * 100
false_negative = np.nansum(((cf_iagos_teoh == 2) & (cf_era_teoh != 2)))/ total * 100
false_positive = np.nansum(((cf_iagos_teoh != 2) & (cf_era_teoh == 2)))/ total * 100
true_negative  = np.nansum(((cf_iagos_teoh != 2) & (cf_era_teoh != 2)))/ total * 100
hit_rate = true_positive / (true_positive + false_positive)
false_alarm = false_negative / (false_negative + true_negative)

print(true_positive)
print(false_positive)
print(false_negative)
print(true_negative)
print((true_positive + false_positive + false_negative + true_negative))
outfile2.write('\n')
outfile2.write('R\n')
outfile2.write('==\n')
outfile2.write('\n')
outfile2.write('TP: '+str(true_positive)+'\n')
outfile2.write('TN: '+str(true_negative)+'\n')
outfile2.write('FP: '+str(false_positive)+'\n')
outfile2.write('FN: '+str(false_negative)+'\n')
outfile2.write('Hitrate: '+str(hit_rate)+'\n')
outfile2.write('False alarm: '+str(false_alarm)+'\n')

#--accuracy
accu = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
print(('Accuracy: '+str(accu)+'\n'))
outfile2.write('Accuracy: '+str(accu)+'\n')
#--positive predictive value
ppv = (true_positive) / (false_positive + true_positive)   
outfile2.write('Positive predictive value: '+str(ppv)+'\n')
print('Positive predictive value: '+str(ppv)+'\n')
#--negative predictive value
npv = (true_negative) / (false_negative + true_negative)   
outfile2.write('Negative predictive value: '+str(npv)+'\n')
print('Negative predictive value: '+str(npv)+'\n')

out4 = np.asarray([true_positive,true_negative, false_negative, false_positive])

x[0,0].bar(5,true_positive,0.5,0,color='blue')
x[0,0].bar(5,true_negative,0.5,true_positive,color='orange')
x[0,0].bar(5,false_positive,0.5,true_negative+true_positive,color='green')
x[0,0].bar(5,false_negative,0.5,true_negative+true_positive+false_positive,color='red')
x[0,0].set_xlim(0,6)
x[0,0].set_xticklabels(['','NPC','','PC','','R'])
x[0,0].set_ylim(0,110)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_xlabel('Contrail Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
    
filename = "classification_quality_corrected_teoh.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#############
#save the stats
if server == 0:
    save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/stats_file_corrected_teoh.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)
if server == 1:
    save_stats_file = '/homedata/kwolf/40_era_iagos/stats_file_corrected_teoh.npz'
    np.savez(save_stats_file,out1,out2,out3,out4)

#%%
#--what is the transition from one to the other region
#--for example going from NPC to PC or No contrail after the correction
#--using era5 org as the reference

outfile2.write('\n')
outfile2.write('Tracking the history of the categorization\n')
outfile2.write('#################################\n')
outfile2.write('\n')

print('Transition from uncor to cor_t')
outfile2.write('Transition from uncor to cor_t\n')
#--from ERA npc to ERA cor2d pc
NPC_NPC = np.nansum((cf_era_uncor == 3) & (cf_era_cor_t == 3)) / np.nansum(cf_era_cor_t == 3) * 100
print('NPC to NPC',NPC_NPC)
outfile2.write('NPC to NPC %6.2f \n' % (NPC_NPC))
PC_NPC = np.nansum((cf_era_uncor == 1) & (cf_era_cor_t == 3)) / np.nansum(cf_era_cor_t == 3) * 100
print('PC to NPC',PC_NPC)
outfile2.write('PC to NPC %6.2f \n' % (PC_NPC))
R_NPC = np.nansum((cf_era_uncor == 2) & (cf_era_cor_t == 3)) / np.nansum(cf_era_cor_t == 3) * 100
print('R to NPC',R_NPC)
outfile2.write('R to NPC %6.2f \n' % (R_NPC))
NOC_NPC = np.nansum((cf_era_uncor == 0) & (cf_era_cor_t == 3)) / np.nansum(cf_era_cor_t == 3) * 100
print('NoC to NPC',NOC_NPC)
outfile2.write('NoC to NPC %6.2f \n' % (NOC_NPC))

print('')
outfile2.write('\n')
NPC_PC = np.nansum((cf_era_uncor == 3) & (cf_era_cor_t == 1)) / np.nansum(cf_era_cor_t == 1) * 100
print('NPC to PC',NPC_PC)
outfile2.write('NPC to PC %6.2f \n' % (NPC_PC))
PC_PC = np.nansum((cf_era_uncor == 1) & (cf_era_cor_t == 1)) / np.nansum(cf_era_cor_t == 1) * 100
print('PC to PC',PC_PC)
outfile2.write('PC to PC %6.2f \n' % (PC_PC))
R_PC = np.nansum((cf_era_uncor == 2) & (cf_era_cor_t == 1)) / np.nansum(cf_era_cor_t == 1) * 100
print('R to PC',R_PC)
outfile2.write('R to PC %6.2f \n' % (R_PC))
NOC_PC = np.nansum((cf_era_uncor == 0) & (cf_era_cor_t == 1)) / np.nansum(cf_era_cor_t == 1) * 100
print('NoC to PC',NOC_PC)
outfile2.write('NoC to PC %6.2f \n' % (NOC_PC))

print('')
outfile2.write('\n')
NPC_R = np.nansum((cf_era_uncor == 3) & (cf_era_cor_t == 2)) / np.nansum(cf_era_cor_t == 2) * 100
print('NPC to R',NPC_R)
outfile2.write('NPC to R %6.2f \n' % (NPC_R))
PC_R = np.nansum((cf_era_uncor == 1) & (cf_era_cor_t == 2)) / np.nansum(cf_era_cor_t == 2) * 100
print('PC to R',PC_R)
outfile2.write('PC to R %6.2f \n' % (PC_R))
R_R = np.nansum((cf_era_uncor == 2) & (cf_era_cor_t == 2)) / np.nansum(cf_era_cor_t == 2) * 100
print('R to R',R_R)
outfile2.write('R to R %6.2f \n' % (R_R))
NOC_R = np.nansum((cf_era_uncor == 0) & (cf_era_cor_t == 2)) / np.nansum(cf_era_cor_t == 2) * 100
print('NoC to R',NOC_R)
outfile2.write('NoC to R %6.2f \n' % (NOC_R))

print('')
outfile2.write('\n')
NPC_NOC = np.nansum((cf_era_uncor == 3) & (cf_era_cor_t == 0)) / np.nansum(cf_era_cor_t == 0) * 100
print('NPC to NoC',NPC_NOC)
outfile2.write('NPC to NoC %6.2f \n' % (NPC_NOC))
PC_NOC = np.nansum((cf_era_uncor == 1) & (cf_era_cor_t == 0)) / np.nansum(cf_era_cor_t == 0) * 100
print('PC to NoC',PC_NOC)
outfile2.write('PC to NoC %6.2f \n' % (PC_NOC))
R_NOC = np.nansum((cf_era_uncor == 2) & (cf_era_cor_t == 0)) / np.nansum(cf_era_cor_t == 0) * 100
print('R to NoC',R_NOC)
outfile2.write('R to NoC %6.2f \n' % (R_NOC))
NOC_NOC = np.nansum((cf_era_uncor == 0) & (cf_era_cor_t == 0)) / np.nansum(cf_era_cor_t == 0) * 100
print('NoC to NoC',NOC_NOC)
outfile2.write('NoC to NoC %6.2f \n' % (NOC_NOC))


F,x=plt.subplots(1,4,figsize=(25,6),squeeze=False)
x1=x[0,0].plot(0,0,label='NPC',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='PC',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='R',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='NoC',color='red',linewidth=4)

i=0
x[0,0].bar(i*2+2,NPC_NPC,0.5,0,color='blue')
x[0,0].bar(i*2+2,PC_NPC,0.5,NPC_NPC,color='orange')
x[0,0].bar(i*2+2,R_NPC,0.5,NPC_NPC+PC_NPC,color='green')
x[0,0].bar(i*2+2,NOC_NPC,0.5,NPC_NPC+PC_NPC+R_NPC,color='red')

i=1
x[0,0].bar(i*2+2,NPC_PC,0.5,0,color='blue')
x[0,0].bar(i*2+2,PC_PC,0.5,NPC_PC,color='orange')
x[0,0].bar(i*2+2,R_PC,0.5,NPC_PC+PC_PC,color='green')
x[0,0].bar(i*2+2,NOC_PC,0.5,NPC_PC+PC_PC+R_PC,color='red')

i=2
x[0,0].bar(i*2+2,NPC_R,0.5,0,color='blue')
x[0,0].bar(i*2+2,PC_R,0.5,NPC_R,color='orange')
x[0,0].bar(i*2+2,R_R,0.5,NPC_R+PC_R,color='green')
x[0,0].bar(i*2+2,NOC_R,0.5,NPC_R+PC_R+R_R,color='red')

i=3
x[0,0].bar(i*2+2,NPC_NOC,0.5,0,color='blue')
x[0,0].bar(i*2+2,PC_NOC,0.5,NPC_NOC,color='orange')
x[0,0].bar(i*2+2,R_NOC,0.5,NPC_NOC+PC_NOC,color='green')
x[0,0].bar(i*2+2,NOC_NOC,0.5,NPC_NOC+PC_NOC+R_NOC,color='red')


x[0,0].set_xlim(0,10)
x[0,0].set_xticklabels(['','NPC','PC','R','NoC'])
x[0,0].set_ylim(0,105)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('(a) T only-correction',fontsize = 20)
x[0,0].set_xlabel('New contrail category',fontsize = 20)
x[0,0].set_ylabel('Fraction',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
 

#--what is the transition from one to the other region
#--for example going from NPC to PC or No contrail after the correction
#-- using era5 org as the reference

outfile2.write('\n')
print('Transition from uncor to cor_rh')
outfile2.write('Transition from uncor to cor_rh\n')
#--from ERA npc to ERA cor2d pc
NPC_NPC = np.nansum((cf_era_uncor == 3) & (cf_era_cor_r == 3)) / np.nansum(cf_era_cor_r == 3) * 100
print('NPC to NPC',NPC_NPC)
outfile2.write('NPC to NPC %6.2f \n' % (NPC_NPC))
PC_NPC = np.nansum((cf_era_uncor == 1) & (cf_era_cor_r == 3)) / np.nansum(cf_era_cor_r == 3) * 100
print('PC to NPC',PC_NPC)
outfile2.write('PC to NPC %6.2f \n' % (PC_NPC))
R_NPC = np.nansum((cf_era_uncor == 2) & (cf_era_cor_r == 3)) / np.nansum(cf_era_cor_r == 3) * 100
print('R to NPC',R_NPC)
outfile2.write('R to NPC %6.2f \n' % (R_NPC))
NOC_NPC = np.nansum((cf_era_uncor == 0) & (cf_era_cor_r == 3)) / np.nansum(cf_era_cor_r == 3) * 100
print('NoC to NPC',NOC_NPC)
outfile2.write('NoC to NPC %6.2f \n' % (NOC_NPC))

print('')
outfile2.write('\n')
NPC_PC = np.nansum((cf_era_uncor == 3) & (cf_era_cor_r == 1)) / np.nansum(cf_era_cor_r == 1) * 100
print('NPC to PC',NPC_PC)
outfile2.write('NPC to PC %6.2f \n' % (NPC_PC))
PC_PC = np.nansum((cf_era_uncor == 1) & (cf_era_cor_r == 1)) / np.nansum(cf_era_cor_r == 1) * 100
print('PC to PC',PC_PC)
outfile2.write('PC to PC %6.2f \n' % (PC_PC))
R_PC = np.nansum((cf_era_uncor == 2) & (cf_era_cor_r == 1)) / np.nansum(cf_era_cor_r == 1) * 100
print('R to PC',R_PC)
outfile2.write('R to PC %6.2f \n' % (R_PC))
NOC_PC = np.nansum((cf_era_uncor == 0) & (cf_era_cor_r == 1)) / np.nansum(cf_era_cor_r == 1) * 100
print('NoC to PC',NOC_PC)
outfile2.write('NoC to PC %6.2f \n' % (NOC_PC))

print('')
outfile2.write('\n')
NPC_R = np.nansum((cf_era_uncor == 3) & (cf_era_cor_r == 2)) / np.nansum(cf_era_cor_r == 2) * 100
print('NPC to R',NPC_R)
outfile2.write('NPC to R %6.2f \n' % (NPC_R))
PC_R = np.nansum((cf_era_uncor == 1) & (cf_era_cor_r == 2)) / np.nansum(cf_era_cor_r == 2) * 100
print('PC to R',PC_R)
outfile2.write('PC to R %6.2f \n' % (PC_R))
R_R = np.nansum((cf_era_uncor == 2) & (cf_era_cor_r == 2)) / np.nansum(cf_era_cor_r == 2) * 100
print('R to R',R_R)
outfile2.write('R to R %6.2f \n' % (R_R))
NOC_R = np.nansum((cf_era_uncor == 0) & (cf_era_cor_r == 2)) / np.nansum(cf_era_cor_r == 2) * 100
print('NoC to R',NOC_R)
outfile2.write('NoC to R %6.2f \n' % (NOC_R))

print('')
outfile2.write('\n')
NPC_NOC = np.nansum((cf_era_uncor == 3) & (cf_era_cor_r == 0)) / np.nansum(cf_era_cor_r == 0) * 100
print('NPC to NoC',NPC_NOC)
outfile2.write('NPC to NoC %6.2f \n' % (NPC_NOC))
PC_NOC = np.nansum((cf_era_uncor == 1) & (cf_era_cor_r == 0)) / np.nansum(cf_era_cor_r == 0) * 100
print('PC to NoC',PC_NOC)
outfile2.write('PC to NoC %6.2f \n' % (PC_NOC))
R_NOC = np.nansum((cf_era_uncor == 2) & (cf_era_cor_r == 0)) / np.nansum(cf_era_cor_r == 0) * 100
print('R to NoC',R_NOC)
outfile2.write('R to NoC %6.2f \n' % (R_NOC))
NOC_NOC = np.nansum((cf_era_uncor == 0) & (cf_era_cor_r == 0)) / np.nansum(cf_era_cor_r == 0) * 100
print('NoC to NoC',NOC_NOC)
outfile2.write('NoC to NoC %6.2f \n' % (NOC_NOC))


i=0
x[0,1].bar(i*2+2,NPC_NPC,0.5,0,color='blue')
x[0,1].bar(i*2+2,PC_NPC,0.5,NPC_NPC,color='orange')
x[0,1].bar(i*2+2,R_NPC,0.5,NPC_NPC+PC_NPC,color='green')
x[0,1].bar(i*2+2,NOC_NPC,0.5,NPC_NPC+PC_NPC+R_NPC,color='red')

i=1
x[0,1].bar(i*2+2,NPC_PC,0.5,0,color='blue')
x[0,1].bar(i*2+2,PC_PC,0.5,NPC_PC,color='orange')
x[0,1].bar(i*2+2,R_PC,0.5,NPC_PC+PC_PC,color='green')
x[0,1].bar(i*2+2,NOC_PC,0.5,NPC_PC+PC_PC+R_PC,color='red')

i=2
x[0,1].bar(i*2+2,NPC_R,0.5,0,color='blue')
x[0,1].bar(i*2+2,PC_R,0.5,NPC_R,color='orange')
x[0,1].bar(i*2+2,R_R,0.5,NPC_R+PC_R,color='green')
x[0,1].bar(i*2+2,NOC_R,0.5,NPC_R+PC_R+R_R,color='red')

i=3
x[0,1].bar(i*2+2,NPC_NOC,0.5,0,color='blue')
x[0,1].bar(i*2+2,PC_NOC,0.5,NPC_NOC,color='orange')
x[0,1].bar(i*2+2,R_NOC,0.5,NPC_NOC+PC_NOC,color='green')
x[0,1].bar(i*2+2,NOC_NOC,0.5,NPC_NOC+PC_NOC+R_NOC,color='red')


x[0,1].set_xlim(0,10)
x[0,1].set_xticklabels(['','NPC','PC','R','NoC'])
x[0,1].set_yticklabels([])
x[0,1].set_ylim(0,105)
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].set_title('(b) r only-correction',fontsize = 20)
x[0,1].set_xlabel('New contrail category',fontsize = 20)

#--what is the transition from one to the other region
#--for example going from NPC to PC or No contrail after the correction
#-- using era5 org as the reference

outfile2.write('\n')
print('Transition from uncor to cor2d')
outfile2.write('Transition from uncor to cor_2d\n')
#--from ERA npc to ERA cor2d pc
NPC_NPC = np.nansum((cf_era_uncor == 3) & (cf_era_cor2d == 3)) / np.nansum(cf_era_cor2d == 3) * 100
print('NPC to NPC',NPC_NPC)
outfile2.write('NPC to NPC %6.2f \n' % (NPC_NPC))
PC_NPC = np.nansum((cf_era_uncor == 1) & (cf_era_cor2d == 3)) / np.nansum(cf_era_cor2d == 3) * 100
print('PC to NPC',PC_NPC)
outfile2.write('PC to NPC %6.2f \n' % (PC_NPC))
R_NPC = np.nansum((cf_era_uncor == 2) & (cf_era_cor2d == 3)) / np.nansum(cf_era_cor2d == 3) * 100
print('R to NPC',R_NPC)
outfile2.write('R to NPC %6.2f \n' % (R_NPC))
NOC_NPC = np.nansum((cf_era_uncor == 0) & (cf_era_cor2d == 3)) / np.nansum(cf_era_cor2d == 3) * 100
print('NoC to NPC',NOC_NPC)
outfile2.write('NoC to NPC %6.2f \n' % (NOC_NPC))

print('')
outfile2.write('\n')
NPC_PC = np.nansum((cf_era_uncor == 3) & (cf_era_cor2d == 1)) / np.nansum(cf_era_cor2d == 1) * 100
print('NPC to PC',NPC_PC)
outfile2.write('NPC to PC %6.2f \n' % (NPC_PC))
PC_PC = np.nansum((cf_era_uncor == 1) & (cf_era_cor2d == 1)) / np.nansum(cf_era_cor2d == 1) * 100
print('PC to PC',PC_PC)
outfile2.write('PC to PC %6.2f \n' % (PC_PC))
R_PC = np.nansum((cf_era_uncor == 2) & (cf_era_cor2d == 1)) / np.nansum(cf_era_cor2d == 1) * 100
print('R to PC',R_PC)
outfile2.write('R to PC %6.2f \n' % (R_PC))
NOC_PC = np.nansum((cf_era_uncor == 0) & (cf_era_cor2d == 1)) / np.nansum(cf_era_cor2d == 1) * 100
print('NoC to PC',NOC_PC)
outfile2.write('NoC to PC %6.2f \n' % (NOC_PC))

print('')
NPC_R = np.nansum((cf_era_uncor == 3) & (cf_era_cor2d == 2)) / np.nansum(cf_era_cor2d == 2) * 100
print('NPC to R',NPC_R)
outfile2.write('NPC to R %6.2f \n' % (NPC_R))
PC_R = np.nansum((cf_era_uncor == 1) & (cf_era_cor2d == 2)) / np.nansum(cf_era_cor2d == 2) * 100
print('PC to R',PC_R)
outfile2.write('PC to R %6.2f \n' % (PC_R))
R_R = np.nansum((cf_era_uncor == 2) & (cf_era_cor2d == 2)) / np.nansum(cf_era_cor2d == 2) * 100
print('R to R',R_R)
outfile2.write('R to R %6.2f \n' % (R_R))
NOC_R = np.nansum((cf_era_uncor == 0) & (cf_era_cor2d == 2)) / np.nansum(cf_era_cor2d == 2) * 100
print('NoC to R',NOC_R)
outfile2.write('NoC to R %6.2f \n' % (NOC_R))

print('')
outfile2.write('\n')
NPC_NOC = np.nansum((cf_era_uncor == 3) & (cf_era_cor2d == 0)) / np.nansum(cf_era_cor2d == 0) * 100
print('NPC to NoC',NPC_NOC)
outfile2.write('NPC to NoC %6.2f \n' % (NPC_NOC))
PC_NOC = np.nansum((cf_era_uncor == 1) & (cf_era_cor2d == 0)) / np.nansum(cf_era_cor2d == 0) * 100
print('PC to NoC',PC_NOC)
outfile2.write('PC to NoC %6.2f \n' % (PC_NOC))
R_NOC = np.nansum((cf_era_uncor == 2) & (cf_era_cor2d == 0)) / np.nansum(cf_era_cor2d == 0) * 100
print('R to NoC',R_NOC)
outfile2.write('R to NoC %6.2f \n' % (R_NOC))
NOC_NOC = np.nansum((cf_era_uncor == 0) & (cf_era_cor2d == 0)) / np.nansum(cf_era_cor2d == 0) * 100
print('NoC to NoC',NOC_NOC)
outfile2.write('NoC to NoC %6.2f \n' % (NOC_NOC))



i=0
x[0,2].bar(i*2+2,NPC_NPC,0.5,0,color='blue')
x[0,2].bar(i*2+2,PC_NPC,0.5,NPC_NPC,color='orange')
x[0,2].bar(i*2+2,R_NPC,0.5,NPC_NPC+PC_NPC,color='green')
x[0,2].bar(i*2+2,NOC_NPC,0.5,NPC_NPC+PC_NPC+R_NPC,color='red')

i=1
x[0,2].bar(i*2+2,NPC_PC,0.5,0,color='blue')
x[0,2].bar(i*2+2,PC_PC,0.5,NPC_PC,color='orange')
x[0,2].bar(i*2+2,R_PC,0.5,NPC_PC+PC_PC,color='green')
x[0,2].bar(i*2+2,NOC_PC,0.5,NPC_PC+PC_PC+R_PC,color='red')

i=2
x[0,2].bar(i*2+2,NPC_R,0.5,0,color='blue')
x[0,2].bar(i*2+2,PC_R,0.5,NPC_R,color='orange')
x[0,2].bar(i*2+2,R_R,0.5,NPC_R+PC_R,color='green')
x[0,2].bar(i*2+2,NOC_R,0.5,NPC_R+PC_R+R_R,color='red')

i=3
x[0,2].bar(i*2+2,NPC_NOC,0.5,0,color='blue')
x[0,2].bar(i*2+2,PC_NOC,0.5,NPC_NOC,color='orange')
x[0,2].bar(i*2+2,R_NOC,0.5,NPC_NOC+PC_NOC,color='green')
x[0,2].bar(i*2+2,NOC_NOC,0.5,NPC_NOC+PC_NOC+R_NOC,color='red')


x[0,2].set_xlim(0,10)
x[0,2].set_xticklabels(['','NPC','PC','R','NoC'])
x[0,2].set_yticklabels([])
x[0,2].set_ylim(0,105)
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].set_title('(c) QM-correction',fontsize = 20)
x[0,2].set_xlabel('New contrail category',fontsize = 20)


#--what is the transition from one to the other region
#--for example going from NPC to PC or No contrail after the correction
#-- using era5 org as the reference

outfile2.write('\n')
print('Transition from uncor to teoh')
outfile2.write('Transition from uncor to cor_teoh\n')
#--from ERA npc to ERA cor2d pc
NPC_NPC = np.nansum((cf_era_uncor == 3) & (cf_era_teoh == 3)) / np.nansum(cf_era_teoh == 3) * 100
print('NPC to NPC',NPC_NPC)
outfile2.write('NPC to NPC %6.2f \n' % (NPC_NPC))
PC_NPC = np.nansum((cf_era_uncor == 1) & (cf_era_teoh == 3)) / np.nansum(cf_era_teoh == 3) * 100
print('PC to NPC',PC_NPC)
outfile2.write('PC to NPC %6.2f \n' % (PC_NPC))
R_NPC = np.nansum((cf_era_uncor == 2) & (cf_era_teoh == 3)) / np.nansum(cf_era_teoh == 3) * 100
print('R to NPC',R_NPC)
outfile2.write('R to NPC %6.2f \n' % (R_NPC))
NOC_NPC = np.nansum((cf_era_uncor == 0) & (cf_era_teoh == 3)) / np.nansum(cf_era_teoh == 3) * 100
print('NoC to NPC',NOC_NPC)
outfile2.write('NoC to NPC %6.2f \n' % (NOC_NPC))

print('')
outfile2.write('\n')
NPC_PC = np.nansum((cf_era_uncor == 3) & (cf_era_teoh == 1)) / np.nansum(cf_era_teoh == 1) * 100
print('NPC to PC',NPC_PC)
outfile2.write('NPC to PC %6.2f \n' % (NPC_PC))
PC_PC = np.nansum((cf_era_uncor == 1) & (cf_era_teoh == 1)) / np.nansum(cf_era_teoh == 1) * 100
print('PC to PC',PC_PC)
outfile2.write('PC to PC %6.2f \n' % (PC_PC))
R_PC = np.nansum((cf_era_uncor == 2) & (cf_era_teoh == 1)) / np.nansum(cf_era_teoh == 1) * 100
print('R to PC',R_PC)
outfile2.write('R to PC %6.2f \n' % (R_PC))
NOC_PC = np.nansum((cf_era_uncor == 0) & (cf_era_teoh == 1)) / np.nansum(cf_era_teoh == 1) * 100
print('NoC to PC',NOC_PC)
outfile2.write('NoC to PC %6.2f \n' % (NOC_PC))

print('')
outfile2.write('\n')
NPC_R = np.nansum((cf_era_uncor == 3) & (cf_era_teoh == 2)) / np.nansum(cf_era_teoh == 2) * 100
print('NPC to R',NPC_R)
outfile2.write('NPC to R %6.2f \n' % (NPC_R))
PC_R = np.nansum((cf_era_uncor == 1) & (cf_era_teoh == 2)) / np.nansum(cf_era_teoh == 2) * 100
print('PC to R',PC_R)
outfile2.write('PC to R %6.2f \n' % (PC_R))
R_R = np.nansum((cf_era_uncor == 2) & (cf_era_teoh == 2)) / np.nansum(cf_era_teoh == 2) * 100
print('R to R',R_R)
outfile2.write('R to R %6.2f \n' % (R_R))
NOC_R = np.nansum((cf_era_uncor == 0) & (cf_era_teoh == 2)) / np.nansum(cf_era_teoh == 2) * 100
print('NoC to R',NOC_R)
outfile2.write('NoC to R %6.2f \n' % (NOC_R))

print('')
outfile2.write('\n')
NPC_NOC = np.nansum((cf_era_uncor == 3) & (cf_era_teoh == 0)) / np.nansum(cf_era_teoh == 0) * 100
print('NPC to NoC',NPC_NOC)
outfile2.write('NPC to NoC %6.2f \n' % (NPC_NOC))
PC_NOC = np.nansum((cf_era_uncor == 1) & (cf_era_teoh == 0)) / np.nansum(cf_era_teoh == 0) * 100
print('PC to NoC',PC_NOC)
outfile2.write('PC to NoC %6.2f \n' % (PC_NOC))
R_NOC = np.nansum((cf_era_uncor == 2) & (cf_era_teoh == 0)) / np.nansum(cf_era_teoh == 0) * 100
print('R to NoC',R_NOC)
outfile2.write('R to NoC %6.2f \n' % (R_NOC))
NOC_NOC = np.nansum((cf_era_uncor == 0) & (cf_era_teoh == 0)) / np.nansum(cf_era_teoh == 0) * 100
print('NoC to NoC',NOC_NOC)
outfile2.write('NoC to NoC %6.2f \n' % (NOC_NOC))

i=0
x[0,3].bar(i*2+2,NPC_NPC,0.5,0,color='blue')
x[0,3].bar(i*2+2,PC_NPC,0.5,NPC_NPC,color='orange')
x[0,3].bar(i*2+2,R_NPC,0.5,NPC_NPC+PC_NPC,color='green')
x[0,3].bar(i*2+2,NOC_NPC,0.5,NPC_NPC+PC_NPC+R_NPC,color='red')

i=1
x[0,3].bar(i*2+2,NPC_PC,0.5,0,color='blue')
x[0,3].bar(i*2+2,PC_PC,0.5,NPC_PC,color='orange')
x[0,3].bar(i*2+2,R_PC,0.5,NPC_PC+PC_PC,color='green')
x[0,3].bar(i*2+2,NOC_PC,0.5,NPC_PC+PC_PC+R_PC,color='red')

i=2
x[0,3].bar(i*2+2,NPC_R,0.5,0,color='blue')
x[0,3].bar(i*2+2,PC_R,0.5,NPC_R,color='orange')
x[0,3].bar(i*2+2,R_R,0.5,NPC_R+PC_R,color='green')
x[0,3].bar(i*2+2,NOC_R,0.5,NPC_R+PC_R+R_R,color='red')

i=3
x[0,3].bar(i*2+2,NPC_NOC,0.5,0,color='blue')
x[0,3].bar(i*2+2,PC_NOC,0.5,NPC_NOC,color='orange')
x[0,3].bar(i*2+2,R_NOC,0.5,NPC_NOC+PC_NOC,color='green')
x[0,3].bar(i*2+2,NOC_NOC,0.5,NPC_NOC+PC_NOC+R_NOC,color='red')


x[0,3].set_xlim(0,10)
x[0,3].set_xticklabels(['','NPC','PC','R','NoC'])
x[0,3].set_yticklabels([])
x[0,3].set_ylim(0,105)
x[0,3].tick_params(labelsize=20)
x[0,3].xaxis.set_tick_params(width=2,length=5)
x[0,3].yaxis.set_tick_params(width=2,length=5)
x[0,3].spines['top'].set_linewidth(1.5)
x[0,3].spines['left'].set_linewidth(1.5)
x[0,3].spines['right'].set_linewidth(1.5)
x[0,3].spines['bottom'].set_linewidth(1.5)
x[0,3].set_title('(d) T22-correction',fontsize = 20)
x[0,3].set_xlabel('New contrail category',fontsize = 20)
 

filename = "history_of_categories_due_to_correction.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()

#%%
#--plot all npc, pc, and r in separate plots

detarr = np.zeros((6,3,4))
files_to_load=['stats_file_uncorrected.npz','stats_file_corrected_t.npz','stats_file_corrected_r.npz','stats_file_corrected_2d.npz','stats_file_corrected_2d.npz','stats_file_corrected_teoh.npz']
if server == 0:
    fpath = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
if server == 1:
    fpath = '/homedata/kwolf/40_era_iagos/'
for i in np.arange(0,6):
    f = fpath+files_to_load[i]
    dummy = np.load(f,allow_pickle=True)
    detarr[i,0,:] = np.asarray(dummy['arr_1'])  #--uncorrected
    detarr[i,1,:] = np.asarray(dummy['arr_2'])  #--uncorrected
    detarr[i,2,:] = np.asarray(dummy['arr_3'])  #--uncorrected

#%%
F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)

for i in np.arange(0,5):
    x[0,0].bar(i*2+2,detarr[i,0,0],0.5,0,color='blue')
    x[0,0].bar(i*2+2,detarr[i,0,1],0.5,detarr[i,0,0],color='orange')
    x[0,0].bar(i*2+2,detarr[i,0,2],0.5,detarr[i,0,0]+detarr[i,0,1],color='green')
    x[0,0].bar(i*2+2,detarr[i,0,3],0.5,detarr[i,0,0]+detarr[i,0,1]+detarr[i,0,2],color='red')


x[0,0].set_xlim(0,12)
x[0,0].set_xticklabels(['','No Cor','T cor','r cor','T-r cor','T-r2d cor'])
x[0,0].set_ylim(30,105)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('NPC',fontsize = 20)
x[0,0].set_xlabel('Correction Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
    
filename = "classification_npc.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#%%

F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)

for i in np.arange(0,5):
    x[0,0].bar(i*2+2,detarr[i,1,0],0.5,0,color='blue')
    x[0,0].bar(i*2+2,detarr[i,1,1],0.5,detarr[i,1,0],color='orange')
    x[0,0].bar(i*2+2,detarr[i,1,2],0.5,detarr[i,1,0]+detarr[i,1,1],color='green')
    x[0,0].bar(i*2+2,detarr[i,1,3],0.5,detarr[i,1,0]+detarr[i,1,1]+detarr[i,1,2],color='red')


x[0,0].set_xlim(0,12)
x[0,0].set_xticklabels(['','No Cor','T cor','r cor','T-r cor','T-r2d cor'])
x[0,0].set_ylim(0,105)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('PC',fontsize = 20)
x[0,0].set_xlabel('Correction Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
    
filename = "classification_pc.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()


#%%
#--similar to above one but a compbined plot for NPC and PC
##

detarr_reduced = detarr[np.asarray([0,4,5]),:,:] #-- selecting onlyt the original, cor2d, and the teoh correction

F,x=plt.subplots(1,3,figsize=(25,6),squeeze=False)
#--for first plot with npc
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)

for i in np.arange(0,3):
    x[0,0].bar(i*2+2,detarr_reduced[i,0,0],0.5,0,color='blue')
    x[0,0].bar(i*2+2,detarr_reduced[i,0,1],0.5,detarr_reduced[i,0,0],color='orange')
    x[0,0].bar(i*2+2,detarr_reduced[i,0,2],0.5,detarr_reduced[i,0,0]+detarr_reduced[i,0,1],color='green')
    x[0,0].bar(i*2+2,detarr_reduced[i,0,3],0.5,detarr_reduced[i,0,0]+detarr_reduced[i,0,1]+detarr_reduced[i,0,2],color='red')
    
    col = ['blue','orange','green','red']
    ypos=np.asarray([18,60,88,95])
    for s in np.arange(0,4):
        x[0,0].text(i*2+0.6,ypos[s],str(f'{detarr_reduced[i,0,s]:4.1f}'),fontsize=25,color=col[s])


x[0,0].set_xlim(0,8)
x[0,0].set_xticklabels(['','ERA','ERA QM','ERA T22'])
x[0,0].set_ylim(0,105)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('(a) Non-persistent contrails',fontsize = 20)
x[0,0].set_xlabel('Correction Type',fontsize = 20)
x[0,0].set_ylabel('Fraction in each class',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')

#--second plot  with PC   
for i in np.arange(0,3):
    x[0,1].bar(i*2+2,detarr_reduced[i,1,0],0.5,0,color='blue')
    x[0,1].bar(i*2+2,detarr_reduced[i,1,1],0.5,detarr_reduced[i,1,0],color='orange')
    x[0,1].bar(i*2+2,detarr_reduced[i,1,2],0.5,detarr_reduced[i,1,0]+detarr_reduced[i,1,1],color='green')
    x[0,1].bar(i*2+2,detarr_reduced[i,1,3],0.5,detarr_reduced[i,1,0]+detarr_reduced[i,1,1]+detarr_reduced[i,1,2],color='red')
    
    col = ['blue','orange','green','red']
    ypos=np.asarray([18,60,88,95])
    for s in np.arange(0,4):
        x[0,1].text(i*2+0.6,ypos[s],str(f'{detarr_reduced[i,1,s]:4.1f}'),fontsize=25,color=col[s])


x[0,1].set_xlim(0,8)
x[0,1].set_xticklabels(['','ERA','ERA QM','ERA T22'])
x[0,1].set_ylim(0,105)
x[0,1].set_yticklabels([])
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].set_title('(b) Persistent contrails',fontsize = 20)
x[0,1].set_xlabel('Correction Type',fontsize = 20)

#--third plot with reservoire
for i in np.arange(0,3):
    x[0,2].bar(i*2+2,detarr_reduced[i,2,0],0.5,0,color='blue')
    x[0,2].bar(i*2+2,detarr_reduced[i,2,1],0.5,detarr_reduced[i,2,0],color='orange')
    x[0,2].bar(i*2+2,detarr_reduced[i,2,2],0.5,detarr_reduced[i,2,0]+detarr_reduced[i,2,1],color='green')
    x[0,2].bar(i*2+2,detarr_reduced[i,2,3],0.5,detarr_reduced[i,2,0]+detarr_reduced[i,2,1]+detarr_reduced[i,2,2],color='red')
    
    col = ['blue','orange','green','red']
    ypos=np.asarray([18,60,88,95])
    for s in np.arange(0,4):
        x[0,2].text(i*2+0.6,ypos[s],str(f'{detarr_reduced[i,2,s]:4.1f}'),fontsize=25,color=col[s])


x[0,2].set_xlim(0,8)
x[0,2].set_xticklabels(['','ERA','ERA QM','ERA T22'])
x[0,2].set_ylim(0,105)
x[0,2].set_yticklabels([])
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].set_title('(c) Reservoir conditions',fontsize = 20)
x[0,2].set_xlabel('Correction Type',fontsize = 20)
    
filename = "classification_npc_pc_r.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()

#%%

F,x=plt.subplots(1,1,figsize=(8,8),squeeze=False)
x1=x[0,0].plot(0,0,label='TP',color='blue',linewidth=4)
x1=x[0,0].plot(0,0,label='TN',color='orange',linewidth=4)
x1=x[0,0].plot(0,0,label='FN',color='green',linewidth=4)
x1=x[0,0].plot(0,0,label='FP',color='red',linewidth=4)

for i in np.arange(0,5):
    x[0,0].bar(i*2+2,detarr[i,2,0],0.5,0,color='blue')
    x[0,0].bar(i*2+2,detarr[i,2,1],0.5,detarr[i,2,0],color='orange')
    x[0,0].bar(i*2+2,detarr[i,2,2],0.5,detarr[i,2,0]+detarr[i,2,1],color='green')
    x[0,0].bar(i*2+2,detarr[i,2,3],0.5,detarr[i,2,0]+detarr[i,2,1]+detarr[i,2,2],color='red')


x[0,0].set_xlim(0,12)
x[0,0].set_xticklabels(['','No Cor','T cor','r cor','T-r cor','T-r2d cor'])
x[0,0].set_ylim(0,102)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('R',fontsize = 20)
x[0,0].set_xlabel('Correction Type',fontsize = 20)
x[0,0].set_ylabel('Detection',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=18,loc='center right')
    
filename = "classification_r.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
print('####################################################')
print('####################################################')
print('##get length of aircraft is flying in issr regions##')
print('####################################################')
print('####################################################')

outfile2.write('####################################################\n')
outfile2.write('####################################################\n')
outfile2.write('##get length of aircraft is flying in issr regions##\n')
outfile2.write('####################################################\n')
outfile2.write('####################################################\n')


dist = np.zeros((out_arr.shape[1]))

#--calculate the length of the sections for each flight
for i in np.arange(0,len(dist)-2):
    dist[i] = gcc.distance_between_points((out_arr[0,i],out_arr[1,i]),(out_arr[0,i+1],out_arr[1,i+1]), unit='kilometers') #--coordiante touple (lon,lat)



#%%

##################
###IAGOS LENGTH###
##################

#--how many consecutive days with persistent contrails
con_meas_p_iagos = []
con_meas_np_iagos = []
con_meas_r_iagos = []

#--length of section
con_meas_p_iagos_length = []
con_meas_np_iagos_length = []
con_meas_r_iagos_length = []

counter = 1
temp_len = 0
for i in np.arange(0,len(cf_iagos_uncor[:])):
    if cf_iagos_uncor[i] == 1:
        temp_len = temp_len + dist[i]   #--adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_p_iagos_length.append(temp_len)  #-- add to the length list
        con_meas_p_iagos.append(counter)
        temp_len = 0
        counter = 1
        
counter = 1
temp_len = 0
for i in np.arange(0,len(cf_iagos_uncor[:])):
    if (cf_iagos_uncor[i] == 2) | (cf_iagos_uncor[i] == 1):
        temp_len = temp_len + dist[i]   #--adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_np_iagos_length.append(temp_len)  #-- add to the length list
        con_meas_np_iagos.append(counter)
        temp_len = 0
        counter = 1
        
counter = 1
temp_len = 0
for i in np.arange(0,len(cf_iagos_uncor[:])):
    if cf_iagos_uncor[i] == 3:
        temp_len = temp_len + dist[i]   #--adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_r_iagos_length.append(temp_len)  #-- add to the length list
        con_meas_r_iagos.append(counter)
        temp_len = 0
        counter = 1
 
con_meas_p_iagos = np.asarray(con_meas_p_iagos)
con_meas_np_iagos = np.asarray(con_meas_np_iagos)
con_meas_r_iagos = np.asarray(con_meas_r_iagos)

#--length of section
con_meas_p_iagos_length = np.asarray(con_meas_p_iagos_length)
con_meas_np_iagos_length = np.asarray(con_meas_np_iagos_length)
con_meas_r_iagos_length = np.asarray(con_meas_r_iagos_length)



#%%
###########################
###IAGOS LENGTH smoothed###
###########################

#--how many consecutive days with persistent contrails
con_meas_p_iagos_smooth = []
con_meas_np_iagos_smooth = []
con_meas_r_iagos_smooth = []

#--length of section
con_meas_p_iagos_length_smooth = []
con_meas_np_iagos_length_smooth = []
con_meas_r_iagos_length_smooth = []

counter = 1
temp_len = 0
for i in np.arange(0,len(cf_iagos_smooth_uncor[:])-1):
    if cf_iagos_smooth_uncor[i] == 1:
        temp_len = temp_len + dist[i]   #--adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_p_iagos_length_smooth.append(temp_len)  #--add to the length list
        con_meas_p_iagos_smooth.append(counter)
        temp_len = 0
        counter = 1
        
counter = 1
temp_len = 0
for i in np.arange(0,len(cf_iagos_smooth_uncor[:])-1):
    if (cf_iagos_smooth_uncor[i] == 2) | (cf_iagos_smooth_uncor[i] == 1):
        temp_len = temp_len + dist[i]   #-- adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_np_iagos_length_smooth.append(temp_len)  #-- add to the length list
        con_meas_np_iagos_smooth.append(counter)
        temp_len = 0
        counter = 1
        
counter = 1
temp_len = 0
for i in np.arange(0,len(cf_iagos_smooth_uncor[:])-1):
    if cf_iagos_smooth_uncor[i] == 3:
        temp_len = temp_len + dist[i]   #-- adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_r_iagos_length_smooth.append(temp_len)  #-- add to the length list
        con_meas_r_iagos_smooth.append(counter)
        temp_len = 0
        counter = 1
 
con_meas_p_iagos_smooth = np.asarray(con_meas_p_iagos_smooth)
con_meas_np_iagos_smooth = np.asarray(con_meas_np_iagos_smooth)
con_meas_r_iagos_smooth = np.asarray(con_meas_r_iagos_smooth)

#length of section
con_meas_p_iagos_length_smooth = np.asarray(con_meas_p_iagos_length_smooth)
con_meas_np_iagos_length_smooth = np.asarray(con_meas_np_iagos_length_smooth)
con_meas_r_iagos_length_smooth = np.asarray(con_meas_r_iagos_length_smooth)
        
#%%

################
###ERA LENGTH###
################

#--how many consecutive days with persistent contrails
con_meas_p_era = []
con_meas_np_era = []
con_meas_r_era = []

#--length of section
con_meas_p_era_length = []
con_meas_np_era_length = []
con_meas_r_era_length = []

counter = 1
temp_len = 0
for i in np.arange(0,len(cf_era_uncor[:])):
    if cf_era_uncor[i] == 1:
        temp_len = temp_len + dist[i]   #-- adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_p_era_length.append(temp_len)  #-- add to the length list
        con_meas_p_era.append(counter)
        temp_len = 0
        counter = 1
        
counter = 1
temp_len = 0
for i in np.arange(0,len(cf_era_uncor[:])):
    if (cf_era_uncor[i] == 2) | (cf_era_uncor[i] == 1):
        temp_len = temp_len + dist[i]   #-- adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_np_era_length.append(temp_len)  #-- add to the length list
        con_meas_np_era.append(counter)
        temp_len = 0
        counter = 1
        
counter = 1
temp_len = 0
for i in np.arange(0,len(cf_era_uncor[:])):
    if cf_era_uncor[i] == 3:
        temp_len = temp_len + dist[i]   #-- adding the length of the current section to the previously added ones
        counter = counter + 1
    else:
        con_meas_r_era_length.append(temp_len)  #-- add to the length list
        con_meas_r_era.append(counter)
        temp_len = 0
        counter = 1

con_meas_p_era = np.asarray(con_meas_p_era)
con_meas_np_era = np.asarray(con_meas_np_era)
con_meas_r_era = np.asarray(con_meas_r_era)

#--length of section
con_meas_p_era_length = np.asarray(con_meas_p_era_length)
con_meas_np_era_length = np.asarray(con_meas_np_era_length)
con_meas_r_era_length = np.asarray(con_meas_r_era_length)


#%%

#--percentiles to print
percenti = np.asarray([0.05,0.1,0.25,0.5,0.75,0.9])

#--IAGOS
print('')
print('Quantiles of length for the three different contrial formation areas from IAGOS')
print('R')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_r_iagos_length,p)))
print('P')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_p_iagos_length,p)))
print('NP')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_np_iagos_length,p)))

#--IAGOS smooth
print('')
print('Quantiles of length for the three different contrial formation areas from IAGOS smooth')
print('R')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_r_iagos_length_smooth,p)))
print('P')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_p_iagos_length_smooth,p)))
print('NP')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_np_iagos_length_smooth,p)))
    
#--ERA
print('')
print('Quantiles of length for the three different contrial formation areas from ERA')
print('R')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_r_era_length,p)))
print('P')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_p_era_length,p)))
print('NP')
print('==')
for p in percenti:
    print(str(p*100)+'% '+str(np.nanquantile(con_meas_np_era_length,p)))
    
    
#--write this into the diagnose file

outfile2.write('\n')
outfile2.write('Length of traversals of aircrfat though eiterh of the contrail formation regions\n')
#--IAGOS
outfile2.write("\n")
outfile2.write("Quantiles of length for the three different contrial formation areas from IAGOS\n")
outfile2.write("R\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_r_iagos_length,p))+'\n')
outfile2.write("P\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_p_iagos_length,p))+'\n')
outfile2.write("NP\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_np_iagos_length,p))+'\n')

#--IAGOS smooth
outfile2.write("\n")
outfile2.write("Quantiles of length for the three different contrial formation areas from IAGOS smooth\n")
outfile2.write("R\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_r_iagos_length_smooth,p))+'\n')
outfile2.write("P\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_p_iagos_length_smooth,p))+'\n')
outfile2.write("NP\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_np_iagos_length_smooth,p))+'\n')
    
#--ERA
outfile2.write("\n")
outfile2.write("Quantiles of length for the three different contrial formation areas from ERA\n")
outfile2.write("R\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_r_era_length,p))+'\n')
outfile2.write("P\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_p_era_length,p))+'\n')
outfile2.write("NP\n")
outfile2.write("==\n")
for p in percenti:
    outfile2.write(str(p*100)+'% '+str(np.nanquantile(con_meas_np_era_length,p))+'\n')


#%%

#--percentages
plotpercentages = np.asarray([0.25,0.5,0.75])


F,x=plt.subplots(2,1,figsize=(8,16),squeeze=False)
x1=x[0,0].plot(0,0)
x[0,0].plot(0,0,color='k',label='IAGOS')
x[0,0].plot(0,0,color='b',label='IAGOS smoothed')
x[0,0].plot(0,0,color='r',label='ERA')
#--Plot for iagos
xxx24,yyy24 = my_histogram(con_meas_np_iagos_length,0.,500.,20.)
x[0,0].plot(xxx24[:-1], yyy24[:],linestyle='solid',color='k',linewidth=2,marker='o')
#--Plot for iagos smooth
xxx24,yyy24 = my_histogram(con_meas_np_iagos_length_smooth,0.,500.,20.)
x[0,0].plot(xxx24[:-1], yyy24[:],linestyle='solid',color='b',linewidth=2,marker='o')
#--plot for ERA
xxx24,yyy24 = my_histogram(con_meas_np_era_length,0.,500.,20.)
x[0,0].plot(xxx24[:-1], yyy24[:],linestyle='solid',color='r',linewidth=2,marker='o')
x[0,0].set_xlim(5,500)
x[0,0].set_ylim(0.001,1)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].grid()
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_ylabel('Normalized PDF',fontsize = 20)
x[0,0].set_xlabel('Crossing length $L$ [km]',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=20,loc='upper right')
x[0,0].set_title('(a) Non-persistent contrails',fontsize=20)

x2=x[1,0].plot(0,0)
x[1,0].plot(0,0,color='k',label='IAGOS')
x[1,0].plot(0,0,color='r',label='ERA')
x[1,0].plot(0,0,color='k',label='Persistent',linestyle='solid')
x[1,0].plot(0,0,color='k',label='Non-Persistent',linestyle='dashed')
#--Plot for iagos
xxx24,yyy24 = my_histogram(con_meas_p_iagos_length,0.,500.,20.)
x[1,0].plot(xxx24[:-1], yyy24[:],linestyle='solid',color='k',linewidth=2,marker='o')
#--Plot for iagos smooth
xxx24,yyy24 = my_histogram(con_meas_p_iagos_length_smooth,0.,500.,20.)
x[1,0].plot(xxx24[:-1], yyy24[:],linestyle='solid',color='b',linewidth=2,marker='o')
#--plot for ERA
xxx24,yyy24 = my_histogram(con_meas_p_era_length,0.,500.,20.)
x[1,0].plot(xxx24[:-1], yyy24[:],linestyle='solid',color='red',linewidth=2,marker='o')
x[1,0].set_xlim(5,500)
x[1,0].set_yscale('log')
x[1,0].set_ylim(0.001,1)
x[1,0].tick_params(labelsize=20)
x[1,0].xaxis.set_tick_params(width=2,length=5)
x[1,0].yaxis.set_tick_params(width=2,length=5)
x[1,0].grid()
x[1,0].spines['top'].set_linewidth(1.5)
x[1,0].spines['left'].set_linewidth(1.5)
x[1,0].spines['right'].set_linewidth(1.5)
x[1,0].spines['bottom'].set_linewidth(1.5)
x[1,0].set_xlabel('Crossing length $L$ [km]',fontsize = 20)
x[1,0].set_title('(b) Persistent contrails',fontsize=20)
x[1,0].set_ylabel('Normalized PDF',fontsize = 20)

filename = 'region_size_iagos_res_1km.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()
    
#%%

##same as above but for cdf!!!

#--percentages
plotpercentages = np.asarray([0.25,0.5,0.75])
F,x=plt.subplots(2,1,figsize=(8,18),squeeze=False)
x1=x[0,0].plot(0,0)
x[0,0].plot(0,0,color='k',label='IAGOS')
x[0,0].plot(0,0,color='b',label='IAGOS smooth')
x[0,0].plot(0,0,color='r',label='ERA')
#--Plot for iagos
xxx24,yyy24 = cum_sum(con_meas_np_iagos_length)
x[0,0].plot(yyy24, xxx24[:],linestyle='solid',color='k',linewidth=2)
#--indicate 25, 50, 75 percentile
for f in [0.1,0.25,0.5,0.75,0.9]:
    val = np.interp(f,xxx24,yyy24,left=None,right=None)
    x[0,0].scatter(val,0,s=30,c='k',marker='o')
    x[0,0].plot((val,val),(f,0),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
    x[0,0].plot((0,val),(f,f),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
#--Plot for iagos smooth
xxx24,yyy24 = cum_sum(con_meas_np_iagos_length_smooth)
x[0,0].plot(yyy24, xxx24[:],linestyle='solid',color='b',linewidth=2)
for f in [0.1,0.25,0.5,0.75,0.9]:
    val = np.interp(f,xxx24,yyy24,left=None,right=None)
    x[0,0].scatter(val,0,s=30,c='b',marker='o')
    x[0,0].plot((val,val),(f,0),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
    x[0,0].plot((0,val),(f,f),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
#--plot for ERA
xxx24,yyy24 = cum_sum(con_meas_np_era_length)
x[0,0].plot(yyy24, xxx24[:],linestyle='solid',color='r',linewidth=2)
for f in [0.1,0.25,0.5,0.75,0.9]:
    val = np.interp(f,xxx24,yyy24,left=None,right=None)
    x[0,0].scatter(val,0,s=30,c='r',marker='o')
    x[0,0].plot((val,val),(f,0),linewidth=1,color='r',alpha=0.6,linestyle='dashed')
    x[0,0].plot((0,val),(f,f),linewidth=1,color='r',alpha=0.6,linestyle='dashed')
    
x[0,0].set_xlim(0.1,10000)
x[0,0].set_xscale('log')
x[0,0].set_ylim(0.,1.05)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_ylabel('Probability',fontsize = 20)
x[0,0].set_xlabel('Crossing length $L$ [km]',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=20,loc='upper left')
x[0,0].set_title('(a) Non-persistent contrails',fontsize=20)

x2=x[1,0].plot(0,0)
x[1,0].plot(0,0,color='k',label='IAGOS')
x[1,0].plot(0,0,color='r',label='ERA')
x[1,0].plot(0,0,color='k',label='Persistent',linestyle='solid')
x[1,0].plot(0,0,color='k',label='Non-Persistent',linestyle='dashed')
#--Plot for iagos
xxx24,yyy24 = cum_sum(con_meas_p_iagos_length)
x[1,0].plot(yyy24, xxx24,linestyle='solid',color='k',linewidth=2)
#--indicate 25, 50, 75 percentile
for f in [0.1,0.25,0.5,0.75,0.9]:
    val = np.interp(f,xxx24,yyy24,left=None,right=None)
    x[1,0].scatter(val,0,s=30,c='k',marker='o')
    x[1,0].plot((val,val),(f,0),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
    x[1,0].plot((0,val),(f,f),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
#--Plot for iagos smooth
xxx24,yyy24 = cum_sum(con_meas_p_iagos_length_smooth)
x[1,0].plot(yyy24, xxx24,linestyle='solid',color='b',linewidth=2)
for f in [0.1,0.25,0.5,0.75,0.9]:
    val = np.interp(f,xxx24,yyy24,left=None,right=None)
    x[1,0].scatter(val,0,s=30,c='b',marker='o')
    x[1,0].plot((val,val),(f,0),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
    x[1,0].plot((0,val),(f,f),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
#--plot for ERA
xxx24,yyy24 = cum_sum(con_meas_p_era_length)
x[1,0].plot(yyy24, xxx24,linestyle='solid',color='red',linewidth=2)
for f in [0.1,0.25,0.5,0.75,0.9]:
    val = np.interp(f,xxx24,yyy24,left=None,right=None)
    x[1,0].scatter(val,0,s=30,c='r',marker='o')
    x[1,0].plot((val,val),(f,0),linewidth=1,color='r',alpha=0.6,linestyle='dashed')
    x[1,0].plot((0,val),(f,f),linewidth=1,color='r',alpha=0.6,linestyle='dashed')
x[1,0].set_xlim(0.1,10000)
x[1,0].set_xscale('log')
x[1,0].set_ylim(0.,1.05)
x[1,0].tick_params(labelsize=20)
x[1,0].xaxis.set_tick_params(width=2,length=5)
x[1,0].yaxis.set_tick_params(width=2,length=5)
x[1,0].spines['top'].set_linewidth(1.5)
x[1,0].spines['left'].set_linewidth(1.5)
x[1,0].spines['right'].set_linewidth(1.5)
x[1,0].spines['bottom'].set_linewidth(1.5)
x[1,0].set_xlabel('Crossing length $L$ [km]',fontsize = 20)
x[1,0].set_title('(b) Persistent contrails',fontsize=20)
x[1,0].set_ylabel('Normalized PDF',fontsize = 20)

filename = 'region_size_iagos_res_1km_cdf.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()

#%%


outfile2.write('\n')
outfile2.write('######################## \n')
outfile2.write('Fraction (0-1) of samples in each combined category per pressure level \n')
outfile2.write('First IAGOS -- ERA cor2d!!! \n')
outfile2.write('######################## \n')

outfile2.write('P-level 250 hPa \n')
outfile2.write('** \n')

F,x=plt.subplots(1,3,figsize=(15,5),squeeze=False)

scal_fact = 40
#--s in scatter is the number of points which is then squared.
#--double s means 4 times the area
lo_count = 0

x1=x[0,0].plot((0,0))
lo_all = (pres_out == 250)
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 3) & (pres_out == 250))
outfile2.write('NoC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='r',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 1) & (pres_out == 250))
outfile2.write('NoC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='steelblue',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 0) & (pres_out == 250))
outfile2.write('NoC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)

lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 3) & (pres_out == 250))
outfile2.write('NPC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 1) & (pres_out == 250))
outfile2.write('NPC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='purple',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 0) & (pres_out == 250))
outfile2.write('NPC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='orange',marker='o',alpha=0.5)

lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 3) & (pres_out == 250))
outfile2.write('PC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='darkolivegreen',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 1) & (pres_out == 250))
outfile2.write('PC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 0) & (pres_out == 250))
outfile2.write('PC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
lo_count = lo_count + np.nansum(lo)
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,0].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='b',marker='o',alpha=0.5)


#--total fraction dies not summ up to 100% percent becuase we also do have non contrail condistions
#--lo-all are all observations (inlc no contrails) per pressure level
print(lo_count / np.nansum(lo_all))


x[0,0].plot([0,0],[-100,100],color='k',linestyle='dashed')
x[0,0].plot([-100,100],[0,0],color='k',linestyle='dashed')

x[0,0].set_ylim(-50,50)
x[0,0].set_xlim(-5,5)
x[0,0].tick_params(labelsize=15)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('(a) 250 hPa',fontsize = 20)
x[0,0].set_ylabel('$\Delta RH$ ERA5 - IAGOS [%]',fontsize = 20)
x[0,0].set_xlabel('$\Delta T$ ERA5 - IAGOS [K]',fontsize = 20)

outfile2.write('\n')
outfile2.write('P-level 225 hPa \n')
outfile2.write('** \n')

lo_all = (pres_out == 225)   

x1=x[0,1].plot((0,0))
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 3) & (pres_out == 225))
outfile2.write('NoC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='r',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 1) & (pres_out == 225))
outfile2.write('NoC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='steelblue',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 0) & (pres_out == 225))
outfile2.write('NoC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)

lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 3) & (pres_out == 225))
outfile2.write('NPC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 1) & (pres_out == 225))
outfile2.write('NPC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='purple',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 0) & (pres_out == 225))
outfile2.write('NPC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='orange',marker='o',alpha=0.5)

lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 3) & (pres_out == 225))
outfile2.write('PC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='darkolivegreen',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 1) & (pres_out == 225))
outfile2.write('PC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 0) & (pres_out == 225))
outfile2.write('PC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,1].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='b',marker='o',alpha=0.5)

x[0,1].plot([0,0],[-100,100],color='k',linestyle='dashed')
x[0,1].plot([-100,100],[0,0],color='k',linestyle='dashed')

x[0,1].set_ylim(-50,50)
x[0,1].set_xlim(-5,5)
x[0,1].tick_params(labelsize=15)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].set_title('(b) 225 hPa',fontsize = 20)
x[0,1].set_xlabel('$\Delta T$ ERA5 - IAGOS [K]',fontsize = 20)


outfile2.write('\n')
outfile2.write('P-level 200 hPa \n')
outfile2.write('** \n')

lo_all = (pres_out == 200)  

x1=x[0,2].plot((0,0))
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 3) & (pres_out == 200))
outfile2.write('NoC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='r',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 1) & (pres_out == 200))
outfile2.write('NoC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='steelblue',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 0) & (cf_era_uncor == 0) & (pres_out == 200))
outfile2.write('NoC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)

lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 3) & (pres_out == 200))
outfile2.write('NPC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 1) & (pres_out == 200))
outfile2.write('NPC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='purple',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 3) & (cf_era_uncor == 0) & (pres_out == 200))
outfile2.write('NPC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='orange',marker='o',alpha=0.5)

lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 3) & (pres_out == 200))
outfile2.write('PC - NPC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='darkolivegreen',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 1) & (pres_out == 200))
outfile2.write('PC - PC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='k',marker='o',alpha=0.5)
lo = ((cf_iagos_uncor == 1) & (cf_era_uncor == 0) & (pres_out == 200))
outfile2.write('PC - NoC %7.5f \n' % (np.nansum(lo) / np.nansum(lo_all)))
s_in = np.nansum(lo) / np.nansum(lo_all) * 100 *scal_fact #--conversion to percent and then 10 as a scaling factor
x[0,2].scatter(np.nanmean(t_out_cor[lo] - out_arr[7,lo]), np.nanmean(r_out_ice_cor2d[lo] - out_arr[13,lo]) ,s=s_in,color='b',marker='o',alpha=0.5)

x[0,2].plot([0,0],[-100,100],color='k',linestyle='dashed')
x[0,2].plot([-100,100],[0,0],color='k',linestyle='dashed')

x[0,2].set_ylim(-50,50)
x[0,2].set_xlim(-5,5)
x[0,2].tick_params(labelsize=15)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].set_title('(c) 200 hPa',fontsize = 20)
x[0,2].set_xlabel('$\Delta T$ ERA5 - IAGOS [K]',fontsize = 20)

#1%
s_in = 0.01 * 100 *scal_fact
x[0,2].scatter([-100],[-100],color='k',label='1%',marker='o',s=s_in)
#10%
s_in = 0.1 * 100 *scal_fact
x[0,2].scatter([-100],[-100],color='k',label='10%',marker='o',s=s_in)
#50%
s_in = 0.5 * 100 *scal_fact
x[0,2].scatter([-100],[-100],color='k',label='50%',marker='o',s=s_in)


x[0,2].legend(shadow=True,fontsize=20,bbox_to_anchor=(1., 0.5),loc='center left')



filename = 't_diff_r_diff_mean_2dplot_cor2d.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()

#%%

#--append the confusion matrix plot

F,x=plt.subplots(1,1,figsize=(10,8),squeeze=False)

x[0,0].plot((1,1),(0,3),color='k') #--vertical lines
x[0,0].plot((2,2),(0,3),color='k') #--vertical lines

x[0,0].plot((0,3),(1,1),color='k') #--horizontal lines
x[0,0].plot((0,3),(2,2),color='k') #--horizontal lines

x[0,0].scatter(0.5,2.5,s=3000,color='k',alpha=0.8) #--IAGOS No / ERA No matrix(0,0)
x[0,0].scatter(1.5,2.5,s=3000,color='red',alpha=0.8) #--IAGOS No / ERA NPC matrix(0,1)
x[0,0].scatter(2.5,2.5,s=3000,color='steelblue',alpha=0.8) #--IAGOS No / ERA NR matrix(0,2)
x[0,0].scatter(0.5,1.5,s=3000,color='orange',alpha=0.8) #--IAGOS NPC / ERA No matrix(1,0)
x[0,0].scatter(1.5,1.5,s=3000,color='k',alpha=0.8) #--IAGOS NPC / ERA NPC matrix(1,1)
x[0,0].scatter(2.5,1.5,s=3000,color='purple',alpha=0.8) #--IAGOS NPC / ERA PC matrix(1,2)
x[0,0].scatter(0.5,0.5,s=3000,color='b',alpha=0.8) #--IAGOS PC / ERA Np matrix(2,0)
x[0,0].scatter(1.5,0.5,s=3000,color='darkolivegreen',alpha=0.8) #--IAGOS PC / ERA NPC matrix(2,1)
x[0,0].scatter(2.5,0.5,s=3000,color='k',alpha=0.8) #--IAGOS PC / ERA PC matrix(2,2)

x[0,0].set_xlim(0,3)
x[0,0].set_ylim(0,3)
x[0,0].set_xticklabels(['','No C','','NPC','','PC'])
x[0,0].set_yticklabels(['','PC','','NPC','','No C'])
x[0,0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,labelsize=30)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('ERA5',fontsize = 30)
x[0,0].set_ylabel('IAGOS',fontsize = 30)

filename = "addon_confusion_matrix.png"
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()



if server == 0:
    load_path='/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    F.show()
if server == 1:
    load_path='/homedata/kwolf/40_era_iagos/'
to_load = [load_path+'t_diff_r_diff_mean_2dplot_cor2d.png',load_path+'addon_confusion_matrix.png']

images = [Image.open(x) for x in to_load]
widths, heights = zip(*(i.size for i in images))

total_width = round(max(widths)*1.45) 
max_height =round( max(heights)*0.7)

new_im = Image.new('RGB', (total_width, max_height),(255, 255, 255))

images[1] = images[1].resize((round(widths[1]*0.65),round(heights[1]*0.65)), Image.ANTIALIAS)

x_offset = 0
new_im.paste(images[0], (x_offset,0))
x_offset += images[0].size[0]
new_im.paste(images[1], (x_offset-0,0))

    
new_im.save(load_path+'t_diff_r_diff_mean_2dplot_cor2d_annotated.png')



outfile2.close()


print('#################################')
print('Safed statistics to:')
if server == 0:
    print('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/')
if server == 1:
    print('/homedata/kwolf/40_era_iagos/')
print('#################################')

print('Done.')



#%%
F,x=plt.subplots(3,3,figsize=(21,21),squeeze=False)

foo =  ((cf_iagos_uncor == 3) & (cf_era_uncor == 3))
x[0,0].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,0].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[0,0].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[0,0].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[0,0].set_xlim(-10,10)
x[0,0].set_ylim(-80,80)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('ERA NPC',fontsize=20)
x[0,0].set_ylabel('$\Delta RH_{ice}$ ERA5 - IAGOS [%]',fontsize = 20)

foo =  ((cf_iagos_uncor == 3) & (cf_era_uncor == 1))
x[0,1].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,1].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[0,1].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[0,1].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[0,1].set_xlim(-10,10)
x[0,1].set_ylim(-80,80)
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].set_title('ERA PC',fontsize=20)

foo =  ((cf_iagos_uncor == 3) & (cf_era_uncor == 0))
x[0,2].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,2].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[0,2].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[0,2].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[0,2].set_xlim(-10,10)
x[0,2].set_ylim(-80,80)
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].set_title('ERA R',fontsize=20)


foo =  ((cf_iagos_uncor == 1) & (cf_era_uncor == 3))
x[1,0].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[1,0].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[1,0].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[1,0].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[1,0].set_xlim(-10,10)
x[1,0].set_ylim(-80,80)
x[1,0].tick_params(labelsize=20)
x[1,0].xaxis.set_tick_params(width=2,length=5)
x[1,0].yaxis.set_tick_params(width=2,length=5)
x[1,0].spines['top'].set_linewidth(1.5)
x[1,0].spines['left'].set_linewidth(1.5)
x[1,0].spines['right'].set_linewidth(1.5)
x[1,0].spines['bottom'].set_linewidth(1.5)
x[1,0].set_ylabel('$\Delta RH_{ice}$ ERA5 - IAGOS [%]',fontsize = 20)

foo =  ((cf_iagos_uncor == 1) & (cf_era_uncor == 1))
x[1,1].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[1,1].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[1,1].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[1,1].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[1,1].set_xlim(-10,10)
x[1,1].set_ylim(-80,80)
x[1,1].tick_params(labelsize=20)
x[1,1].xaxis.set_tick_params(width=2,length=5)
x[1,1].yaxis.set_tick_params(width=2,length=5)
x[1,1].spines['top'].set_linewidth(1.5)
x[1,1].spines['left'].set_linewidth(1.5)
x[1,1].spines['right'].set_linewidth(1.5)
x[1,1].spines['bottom'].set_linewidth(1.5)

foo =  ((cf_iagos_uncor == 1) & (cf_era_uncor == 0))
x[1,2].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[1,2].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[1,2].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[1,2].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[1,2].set_xlim(-10,10)
x[1,2].set_ylim(-80,80)
x[1,2].tick_params(labelsize=20)
x[1,2].xaxis.set_tick_params(width=2,length=5)
x[1,2].yaxis.set_tick_params(width=2,length=5)
x[1,2].spines['top'].set_linewidth(1.5)
x[1,2].spines['left'].set_linewidth(1.5)
x[1,2].spines['right'].set_linewidth(1.5)
x[1,2].spines['bottom'].set_linewidth(1.5)

foo =  ((cf_iagos_uncor == 0) & (cf_era_uncor == 3))
x[2,0].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[2,0].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[2,0].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[2,0].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[2,0].set_xlim(-10,10)
x[2,0].set_ylim(-80,80)
x[2,0].tick_params(labelsize=20)
x[2,0].xaxis.set_tick_params(width=2,length=5)
x[2,0].yaxis.set_tick_params(width=2,length=5)
x[2,0].spines['top'].set_linewidth(1.5)
x[2,0].spines['left'].set_linewidth(1.5)
x[2,0].spines['right'].set_linewidth(1.5)
x[2,0].spines['bottom'].set_linewidth(1.5)
x[2,0].set_ylabel('$\Delta RH_{ice}$ ERA5 - IAGOS [%]',fontsize = 20)
x[2,0].set_xlabel('$\Delta$T ERA5 - IAGOS [K]',fontsize = 20)

foo =  ((cf_iagos_uncor == 0) & (cf_era_uncor == 1))
x[2,1].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x2=x[2,1].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[2,1].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[2,1].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[2,1].set_xlim(-10,10)
x[2,1].set_ylim(-80,80)
x[2,1].tick_params(labelsize=20)
x[2,1].xaxis.set_tick_params(width=2,length=5)
x[2,1].yaxis.set_tick_params(width=2,length=5)
x[2,1].spines['top'].set_linewidth(1.5)
x[2,1].spines['left'].set_linewidth(1.5)
x[2,1].spines['right'].set_linewidth(1.5)
x[2,1].spines['bottom'].set_linewidth(1.5)
x[2,1].set_xlabel('$\Delta$T ERA5 - IAGOS [K]',fontsize = 20)

foo =  ((cf_iagos_uncor == 0) & (cf_era_uncor == 0))
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']

x1=x[2,2].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]), bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[2,2].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[2,2].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[2,2].set_xlim(-10,10)
x[2,2].set_ylim(-80,80)
x[2,2].tick_params(labelsize=20)
x[2,2].xaxis.set_tick_params(width=2,length=5)
x[2,2].yaxis.set_tick_params(width=2,length=5)
x[2,2].spines['top'].set_linewidth(1.5)
x[2,2].spines['left'].set_linewidth(1.5)
x[2,2].spines['right'].set_linewidth(1.5)
x[2,2].spines['bottom'].set_linewidth(1.5)
cax = F.add_axes([0.95, 0.1, 0.025, 0.8])
cbar = F.colorbar(x1[3],cax=cax)
cbar.ax.tick_params(labelsize=25)

filename = 'NPC_PC_R_multi_devs_cor.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()


#%%
F,x=plt.subplots(3,3,figsize=(21,21),squeeze=False)

foo =  ((cf_iagos_uncor == 3) & (cf_era_cor2d == 3))
x[0,0].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,0].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[0,0].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[0,0].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[0,0].set_xlim(-10,10)
x[0,0].set_ylim(-80,80)
x[0,0].tick_params(labelsize=20)
x[0,0].xaxis.set_tick_params(width=2,length=5)
x[0,0].yaxis.set_tick_params(width=2,length=5)
x[0,0].spines['top'].set_linewidth(1.5)
x[0,0].spines['left'].set_linewidth(1.5)
x[0,0].spines['right'].set_linewidth(1.5)
x[0,0].spines['bottom'].set_linewidth(1.5)
x[0,0].set_title('ERA NPC',fontsize=20)
x[0,0].set_ylabel('$\Delta RH_{ice}$ ERA5 - IAGOS [%]',fontsize = 20)

foo =  ((cf_iagos_uncor == 3) & (cf_era_cor2d == 1))
x[0,1].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,1].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[0,1].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[0,1].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[0,1].set_xlim(-10,10)
x[0,1].set_ylim(-80,80)
x[0,1].tick_params(labelsize=20)
x[0,1].xaxis.set_tick_params(width=2,length=5)
x[0,1].yaxis.set_tick_params(width=2,length=5)
x[0,1].spines['top'].set_linewidth(1.5)
x[0,1].spines['left'].set_linewidth(1.5)
x[0,1].spines['right'].set_linewidth(1.5)
x[0,1].spines['bottom'].set_linewidth(1.5)
x[0,1].set_title('ERA PC',fontsize=20)

foo =  ((cf_iagos_uncor == 3) & (cf_era_cor2d == 0))
x[0,2].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[0,2].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[0,2].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[0,2].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[0,2].set_xlim(-10,10)
x[0,2].set_ylim(-80,80)
x[0,2].tick_params(labelsize=20)
x[0,2].xaxis.set_tick_params(width=2,length=5)
x[0,2].yaxis.set_tick_params(width=2,length=5)
x[0,2].spines['top'].set_linewidth(1.5)
x[0,2].spines['left'].set_linewidth(1.5)
x[0,2].spines['right'].set_linewidth(1.5)
x[0,2].spines['bottom'].set_linewidth(1.5)
x[0,2].set_title('ERA R',fontsize=20)


foo =  ((cf_iagos_uncor == 1) & (cf_era_cor2d == 3))
x[1,0].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[1,0].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[1,0].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[1,0].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[1,0].set_xlim(-10,10)
x[1,0].set_ylim(-80,80)
x[1,0].tick_params(labelsize=20)
x[1,0].xaxis.set_tick_params(width=2,length=5)
x[1,0].yaxis.set_tick_params(width=2,length=5)
x[1,0].spines['top'].set_linewidth(1.5)
x[1,0].spines['left'].set_linewidth(1.5)
x[1,0].spines['right'].set_linewidth(1.5)
x[1,0].spines['bottom'].set_linewidth(1.5)
x[1,0].set_ylabel('$\Delta RH_{ice}$ ERA5 - IAGOS [%]',fontsize = 20)

foo =  ((cf_iagos_uncor == 1) & (cf_era_cor2d == 1))
x[1,1].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[1,1].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[1,1].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[1,1].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[1,1].set_xlim(-10,10)
x[1,1].set_ylim(-80,80)
x[1,1].tick_params(labelsize=20)
x[1,1].xaxis.set_tick_params(width=2,length=5)
x[1,1].yaxis.set_tick_params(width=2,length=5)
x[1,1].spines['top'].set_linewidth(1.5)
x[1,1].spines['left'].set_linewidth(1.5)
x[1,1].spines['right'].set_linewidth(1.5)
x[1,1].spines['bottom'].set_linewidth(1.5)

foo =  ((cf_iagos_uncor == 1) & (cf_era_cor2d == 0))
x[1,2].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[1,2].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[1,2].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[1,2].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[1,2].set_xlim(-10,10)
x[1,2].set_ylim(-80,80)
x[1,2].tick_params(labelsize=20)
x[1,2].xaxis.set_tick_params(width=2,length=5)
x[1,2].yaxis.set_tick_params(width=2,length=5)
x[1,2].spines['top'].set_linewidth(1.5)
x[1,2].spines['left'].set_linewidth(1.5)
x[1,2].spines['right'].set_linewidth(1.5)
x[1,2].spines['bottom'].set_linewidth(1.5)


foo =  ((cf_iagos_uncor == 0) & (cf_era_cor2d == 3))
x[2,0].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x1=x[2,0].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[2,0].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[2,0].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[2,0].set_xlim(-10,10)
x[2,0].set_ylim(-80,80)
x[2,0].tick_params(labelsize=20)
x[2,0].xaxis.set_tick_params(width=2,length=5)
x[2,0].yaxis.set_tick_params(width=2,length=5)
x[2,0].spines['top'].set_linewidth(1.5)
x[2,0].spines['left'].set_linewidth(1.5)
x[2,0].spines['right'].set_linewidth(1.5)
x[2,0].spines['bottom'].set_linewidth(1.5)
x[2,0].set_ylabel('$\Delta RH_{ice}$ ERA5 - IAGOS [%]',fontsize = 20)
x[2,0].set_xlabel('$\Delta$T ERA5 - IAGOS [K]',fontsize = 20)

foo =  ((cf_iagos_uncor == 0) & (cf_era_cor2d == 1))
x[2,1].plot(0,0)
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
x2=x[2,1].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]),bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[2,1].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[2,1].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[2,1].set_xlim(-10,10)
x[2,1].set_ylim(-80,80)
x[2,1].tick_params(labelsize=20)
x[2,1].xaxis.set_tick_params(width=2,length=5)
x[2,1].yaxis.set_tick_params(width=2,length=5)
x[2,1].spines['top'].set_linewidth(1.5)
x[2,1].spines['left'].set_linewidth(1.5)
x[2,1].spines['right'].set_linewidth(1.5)
x[2,1].spines['bottom'].set_linewidth(1.5)
x[2,1].set_xlabel('$\Delta$T ERA5 - IAGOS [K]',fontsize = 20)

foo =  ((cf_iagos_uncor == 0) & (cf_era_cor2d == 0))
mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']

x1=x[2,2].hist2d((t_out_cor[foo] - out_arr[7,foo]) ,(r_out_ice_cor[foo] - out_arr[13,foo]), bins=80,range=[[-10,10],[-80,80]], cmap=plt.cm.jet,norm = LogNorm(vmin=0.1,vmax=1000))

x[2,2].plot((0,0),(-80,80),linewidth=2,linestyle='dashed',color='k')
x[2,2].plot((-10,10),(0,0),linewidth=2,linestyle='dashed',color='k')
x[2,2].set_xlim(-10,10)
x[2,2].set_ylim(-80,80)
x[2,2].tick_params(labelsize=20)
x[2,2].xaxis.set_tick_params(width=2,length=5)
x[2,2].yaxis.set_tick_params(width=2,length=5)
x[2,2].spines['top'].set_linewidth(1.5)
x[2,2].spines['left'].set_linewidth(1.5)
x[2,2].spines['right'].set_linewidth(1.5)
x[2,2].spines['bottom'].set_linewidth(1.5)
x[2,2].set_xlabel('$\Delta$T ERA5 - IAGOS [K]',fontsize = 20)

cax = F.add_axes([0.95, 0.1, 0.025, 0.8])
cbar = F.colorbar(x1[3],cax=cax)
cbar.ax.tick_params(labelsize=25)
cbar.set_label('Frequency of occurence',size=30)

filename = 'NPC_PC_R_multi_devs_cor.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()


#--end of code
