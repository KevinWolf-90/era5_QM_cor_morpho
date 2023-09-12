#!/usr/bin/env python3rm 
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
import netCDF4 as nc
#from datetime import datetime
#from time import mktime
import copy
import sys
import os
from pathlib import Path
import pandas as pd
import random

import xarray as xr

#statistical imports
from scipy import stats

#--get string of alphabet
import string
alpha_string = list(string.ascii_lowercase)

import os.path
from matplotlib.colors import LogNorm


#--used to calculate my own histograms
def my_histogram(value,xmin,xmax,step):
        dummy = copy.deepcopy(value)
        dummy = dummy.reshape(dummy[:].size)
        yyy, xxx = np.histogram(dummy[:], bins=np.linspace(xmin, xmax, int((xmax-xmin)/step)))
        y_total = np.nansum(yyy)
        yyy = np.divide(yyy,y_total)
        return(xxx,yyy)


#--random noise generator
def set_rand(inarr):
    randarr = np.random.rand(len(inarr))*0.01
    outarr = inarr + randarr
    return(outarr)


def cor_low_high(inp,lowb,highb):
    if inp[0] == inp[1]:
        loc = np.where(inp ==lowb)
        n = len(loc[0])
        if n != 0:
            low = lowb
            high= inp[loc[0][-1]+1]
            replace = np.zeros((n))
            for i in np.arange(0,n):
                replace[i] = low + (high-low) /n * (i+1)
            inp[inp==low] = replace
        
    if inp[-1] == inp[-2]:
        loc = np.where(inp == highb)
        n = len(loc[0])
        if n!=0:
            low = inp[loc[0][0]-1]
            high= highb
            replace = np.zeros((n))
            for i in np.arange(0,n):
                replace[i] = low + (high-low) /n * (i+1)
            inp[inp==high] = replace

    return(np.asarray(inp))


#--disable warning. keep output clean
import warnings
warnings.filterwarnings("ignore")
print('#################################')
print('Filter warnings are switched off!')
print('#################################')
time.sleep(1)


#--run on server or local
#server = 0 # 0 if local, 1 if on server
server = 1

    
    
#%%
#--write diagnose ouput file
filename = 'diagnose_cdf_build.txt'

if server == 0:
    filename_diag = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename
if server == 1:
    filename_diag = '/homedata/kwolf/40_era_iagos/'+filename



outfile2 = open(filename_diag ,'w')
outfile2.write('Diagnose CDF build \n')
outfile2.write('========\n')

#%%

#--plotting on server or local
if server == 1:  # to not use Xwindow
    if any('SPYDER' in name for name in os.environ):
        print('Activated plotting on screen')
    else:
        print('Deactivated plotting on screen for terminal and batch')
        matplotlib.use('Agg')


#--path to my routines
if server == 0:
    sys.path.insert(1, '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines')
if server == 1:
    sys.path.insert(1, '/homedata/kwolf/40_era_iagos/00_code')


#--load the picles files with the timeseries 
filedummz = '*_era_1h.npz'
#filedummz = '*2018_era_1h.npz'
print(filedummz)
if server == 0:
    path = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'
    files = Path(path).glob(filedummz)
if server == 1:    
    path='/homedata/kwolf/40_era_iagos/'
    files = Path(path).glob(filedummz)

#--define empty arrays that will be appended with the data from each npz file
out_arr = np.zeros((25,0))
pres_out = np.zeros((0))
t_out = np.zeros((0))
r_out = np.zeros((0))
r_out_ice = np.zeros((0))
wspd_out = np.zeros((0))
wndr_out = np.zeros((0))

for f in files:
    print('###############################')
    print('Read the extracted data from:!')
    print(f)
    print('###############################')
    time.sleep(1)

    dummy = np.load(f,allow_pickle=True)
    out_arr =  np.append(out_arr, np.asarray(dummy['arr_0'],dtype=np.float64),axis=1)#iagos out arr
    pres = np.asarray(dummy['arr_1'])  #era press
    
    pres_out = np.append(pres_out, np.asarray(dummy['arr_2'],dtype=np.float64)) #era p
    t_out = np.append(t_out, np.asarray(dummy['arr_3'],dtype=np.float64)) #era temp
    r_out = np.append(r_out, np.asarray(dummy['arr_4'],dtype=np.float64)) #era relhum
    r_out_ice = np.append(r_out_ice, np.asarray(dummy['arr_5'],dtype=np.float64)) #era relhumice
    wspd_out = np.append(wspd_out, np.asarray(dummy['arr_6'],dtype=np.float64)) #era wspd
    wndr_out = np.append(wndr_out, np.asarray(dummy['arr_7'],dtype=np.float64)) #era wndir


#--remove flights from certain areas
cords = [-115, 35, 30, 70] # all available do not go beyond these otherwise edges are used
print('###################')
print('Selected domain is!')
print('###################')
print('Lon: '+str(cords[0])+' '+str(cords[1]))
print('Lat: '+str(cords[2])+' '+str(cords[3]))
time.sleep(3)

foo = (out_arr[0,:] > cords[0]) & (out_arr[0,:] < cords[1])  #Lon
out_arr = out_arr[:,foo]
pres_out = pres_out[foo]
t_out = t_out[foo]
r_out = r_out[foo]
r_out_ice = r_out_ice[foo]
wspd_out = wspd_out[foo]
wndr_out = wndr_out[foo]


foo = (out_arr[1,:] > cords[2]) & (out_arr[1,:] < cords[3])  # Lat
out_arr = out_arr[:,foo]
pres_out = pres_out[foo]
t_out = t_out[foo]
r_out = r_out[foo]
r_out_ice = r_out_ice[foo]
wspd_out = wspd_out[foo]
wndr_out = wndr_out[foo]

#%%


print('There are '+str(len(out_arr[0,:]))+' measurements.')
print('')
print('Min presure in iagos: ',np.nanmin(out_arr[4,:]/100))
print('Max presure in iagos: ',np.nanmax(out_arr[4,:]/100))
print('Min presure in era: ',np.nanmin(pres_out))
print('Max presure in era: ',np.nanmax(pres_out))
print('')





#%%

#-- convert string list to numpy float array
pres = np.asarray(pres,dtype=np.float64)
print('Pressure array: ',pres)

#<--calculate the difference in pressure levels and the brakets-->
pres_diff = np.diff(pres)
pres_brak = np.append(pres[1:] - np.diff(pres[0:])/2,pres[-1])
print('Pressure brackets: ',pres_brak)


#%%

print('Calculate the cdf for each level from era')
print('and interpolate on a 1%-rid resolution')

r_2d_tlat_bins = np.zeros((8,3)) # for 8 p-levels, 3 different boudnary temperatures
r_2d_tbins = np.zeros((8,2,6))  #for 8 p-levels, 2 bins, 6 boundary temperatures


cdf_iagos_arr_t = np.zeros((8,2,len(np.linspace(190,273,415))))  #for 8 p-levels, 2 lat regions, temp from 190 to 273K in 0.2 K steps for temp CDF
cdf_iagos_arr_r = np.zeros((8,2,len(np.arange(0,180,1)))) #8 p-levels, 2 lat regions, RHi from  0% to 180 % with 1% steps
cdf_iagos_arr_r_2d = np.zeros((8,2,5,len(np.arange(0,180,1))))  #8 p-levels, 2 lat regions, 6 temps, RHi from  0% to 180 % with 1% steps

cdf_era_arr_t = np.zeros((8,2,len(np.linspace(190,273,415))))  #for 8 p-levels, 2 lat regions, temp from 190 to 273K in 0.2 K steps for temp CDF
cdf_era_arr_r = np.zeros((8,2,len(np.arange(0,180,1)))) #8 p-levels, 2 lat regions, RHi from  0% to 180 % with 1% steps
cdf_era_arr_r_2d = np.zeros((8,2,5,len(np.arange(0,180,1))))  #8 p-levels, 2 lat regions, 6 temps, RHi from  0% to 180 % with 1% steps


#%%

count_total = 0
count_nan = 0

for pl in np.arange(0,len(pres)-1):

    print('Processing p level: '+str(pres[pl])+' hPa')
    outfile2.write('\n')
    p_filter = ((pres_out <= pres_brak[pl]) & (pres_out > pres_brak[pl+1]))
    
    #--get the percentiles
    prctls = np.nanquantile(out_arr[1,p_filter], (0.5))   #--for two intervals, divide measurments into half
    prctls = np.append(np.append(np.asarray([30]),prctls),np.asarray([70]))    #--these are not percentiles, these are the latitude boundaries!!!
    print('persentilces: ',prctls)
    r_2d_tlat_bins[pl,:] = prctls
    
    
    for rl in np.arange(0,len(prctls)-1):  #-loop over the three regions
        outfile2.write('\n')
        print('rl: ',rl)
        
        altfoo = (pres_out[:] <= pres_brak[pl]) & (pres_out[:] > pres_brak[pl+1]) & (out_arr[1,:] >= prctls[rl]) & (out_arr[1,:] < prctls[rl+1])
        
        print('')
        print('prctls: ',prctls[rl],prctls[rl+1])
        print('')
 
        #get p-level and region specific values
        dataera_t = t_out[altfoo]
        dataera_r = r_out_ice[altfoo]
        dataiagos_t = out_arr[7,altfoo]
        dataiagos_r = out_arr[13,altfoo]
        
        #######################
        ##temperature
        #######################
        # sort the data:
        data_sortediagos = np.sort(dataiagos_t)
        data_sortedera = np.sort(dataera_t)
        
        piagos =  np.linspace(0, 1, len(data_sortediagos), endpoint=False)
        pera =  np.linspace(0, 1, len(data_sortedera), endpoint=False)
        #--set distribution to nan if there is no data in one of the groups; interpolate on 1% resolution
        if ((len(data_sortedera) !=0) & (len(data_sortediagos) !=0)):
            pera_int = np.interp(np.arange(190.,273.,0.2), data_sortedera, pera,left=0,right=1)
            pera_int = cor_low_high(pera_int,0,1)
            cdf_era_arr_t[pl,rl,:] = pera_int
            
            piagos_int = np.interp(np.arange(190.,273.,0.2), data_sortediagos, piagos,left=0,right=1)
            piagos_int = cor_low_high(piagos_int,0,1)
            cdf_iagos_arr_t[pl,rl,:] = piagos_int
        else:
            cdf_iagos_arr_t[pl,rl,:] = np.nan 
        
        

        #add random noise to avoid zeros
        dataera_r_dummy = set_rand(dataera_r)
        dataiagos_r_dummy = set_rand(dataiagos_r)
        ########################################
        ##Relative humidity (no temp dependence)
        ########################################
        # sort the data:
        data_sortedera = np.sort(dataera_r_dummy)
        data_sortediagos = np.sort(dataiagos_r_dummy)
        # calculate the proportional values of samples
        pera = np.linspace(0, 1, len(data_sortedera), endpoint=False)
        piagos = np.linspace(0, 1, len(data_sortediagos), endpoint=False)
        #interpolate on 1% resolution
        if len(data_sortedera) !=0:
            pera_int = np.interp(np.arange(0,180,1), data_sortedera, pera,left=0,right=1)
            pera_int = cor_low_high(pera_int,0,1)
            #print(pera_int)
            cdf_era_arr_r[pl,rl,:] = pera_int
        else:
            cdf_era_arr_r[pl,rl,:] = np.nan  #set function to nan, when there are no values
        if len(data_sortediagos) !=0:
            piagos_int = np.interp(np.arange(0,180,1), data_sortediagos, piagos,left=0,right=1)
            piagos_int = cor_low_high(piagos_int,0,1)
            #print(piagos_int)
            cdf_iagos_arr_r[pl,rl,:] = piagos_int
        else:
            cdf_iagos_arr_r[pl,rl,:] = np.nan  #set function to nan, when there are no values
            

        #######################
        ##Relative humidity (temp dep)
        #######################
        if ((len(data_sortedera) !=0) & (len(data_sortediagos) !=0)):
            tquart = np.arange(0,1.2,0.2) #--20% quantiles for 6 boundary temps and 5 bins
            r_2d_tbins[pl,rl,:] = np.quantile(np.append(np.append(out_arr[7,altfoo],190),273),tquart) # add boundary temps (190, 270) as they have to be covered to have the full range.
            print(r_2d_tbins[pl,rl,:])
       
        #for each temperature bin
        for tl in np.arange(0,len(r_2d_tbins[pl,rl,:])-1):  
            print('T range: from',r_2d_tbins[pl,rl,tl] ,' to: ',r_2d_tbins[pl,rl,tl+1])
            tempfoo = (dataiagos_t >= r_2d_tbins[pl,rl,tl]) & (dataiagos_t < r_2d_tbins[pl,rl,tl+1])# variable steps
            dataiagos_r_dummy = dataiagos_r[tempfoo]  #already selected for p-level and region
            dataera_r_dummy = dataera_r[tempfoo]  #already selected for p-level and region
            
            #print(r_2d_tbins[pl,rl])
            print('Nr of observatiosn in bin t bin: ', np.nansum(tempfoo))
            
            #add random noise
            dataiagos_r_dummy = set_rand(dataiagos_r_dummy)
            dataera_r_dummy = set_rand(dataera_r_dummy)
            # sort the data:
            data_sortediagos = np.sort(dataiagos_r_dummy)
            data_sortedera = np.sort(dataera_r_dummy)
            # calculate the proportional values of samples
            piagos = np.linspace(0, 1, len(dataiagos_r_dummy), endpoint=False)
            pera = np.linspace(0, 1, len(dataera_r_dummy), endpoint=False)
            
            #interpolate on 1% resolution
            #if len(data_sortedera) !=0:
            #if len(data_sortedera) > (out_arr[0,:].shape[0]/84):  # in each of the 167 distribution at least a minimum
            if  ((len(np.unique(data_sortedera)) > 90) & (len(np.unique(data_sortediagos)) > 90)): #--have atl east 90 unique points in the sample
                piagos_int = np.interp(np.arange(0.,180.,1.), data_sortediagos, piagos,left=0,right=1)
                piagos_int = cor_low_high(piagos_int,0,1)
                cdf_iagos_arr_r_2d[pl,rl,tl,:] = piagos_int
                pera_int = np.interp(np.arange(0.,180.,1.), data_sortedera, pera,left=0,right=1)
                pera_int = cor_low_high(pera_int,0,1)
                cdf_era_arr_r_2d[pl,rl,tl,:] = pera_int
                
                outfile2.write('Plevel: %5.0f, Region: %2.0f, Tbin: %6.2f - %6.2f, Obs: %8.0f \n' % (pres[pl],prctls[rl], r_2d_tbins[pl,rl,tl], r_2d_tbins[pl,rl,tl+1], np.nansum(tempfoo)))
                #some preliminary plot for test
                
                plt.plot(np.arange(0.,180,1.),cdf_iagos_arr_r_2d[pl,rl,tl,:],color='k')
                plt.scatter(data_sortediagos,piagos,s=2,c='r',marker='^')
                
                plt.plot(np.arange(0.,180,1.),cdf_era_arr_r_2d[pl,rl,tl,:],color='g')
                plt.scatter(data_sortedera,pera,s=2,c='b',marker='^')
                plt.title(str(pres[pl])+' '+str(prctls[rl])+' '+str(tl))
                plt.xlim(0,180)
                plt.ylim(0,1)
                plt.show()            
            else:
                cdf_era_arr_r_2d[pl,rl,tl,:] = np.nan
                cdf_iagos_arr_r_2d[pl,rl,tl,:] = np.nan
                outfile2.write('Plevel: %5.0f, Region: %2.0f, Tbin: %6.2f - %6.2f, Obs: %8.0f, Is nan \n' % (pres[pl],prctls[rl], r_2d_tbins[pl,rl,tl], r_2d_tbins[pl,rl,tl+1], np.nansum(tempfoo)))
                count_nan += 1# counter fo non nan cdfs
                

            #time.sleep(2)
            #print('')
            
            count_total +=1


print('######')
print('Total: ',count_total)
print('Non nan: ',count_nan)
print('######')


#%%
#%%
#%%
#--close diagnostic file
outfile2.close()

#%%

#--plot cdf distributions.
F,x=plt.subplots(len(pres)-2,1,figsize=(15,(len(pres)-2)*6),squeeze=False)

my_list = ['brown','red','darkorange','green','steelblue','purple']
mycolor= list(reversed(my_list))


for kt in np.arange(0,len(pres)-2):
    x1=x[kt,0].plot(0,0)
    #x[kt,0].plot(np.arange(0,180,1), cdf_era_arr_r[kt,0,:], alpha=1,label='ERA R1',c='r',linewidth=2)
    x[kt,0].plot(np.arange(0,180,1), cdf_era_arr_r[kt,1,:], alpha=1,c='k',linewidth=2,label='ERA',linestyle='solid')
    #x[kt,0].plot(np.arange(0,180,1), cdf_iagos_arr_r[kt,0,:], alpha=1,label='IAGOS R2',c='k',linewidth=2)
    x[kt,0].plot(np.arange(0,180,1), cdf_iagos_arr_r[kt,1,:], alpha=1,c='k',linewidth=2,linestyle='dashed',label='IAGOS')
    
    
    for tt in  np.arange(0,5):
        x[kt,0].plot(np.arange(0,180,1), cdf_era_arr_r_2d[kt,1,tt,:], alpha=0.8,c=mycolor[tt],linewidth=1,label='T ('+str((20*tt)+20)+'%)')
        #x[kt,0].plot(np.arange(0,180,1), cdf_era_arr_r_2d[kt,1,tt,:], alpha=0.5,c=mycolor[tt],linewidth=1,linestyle='dashed')
        x[kt,0].plot(np.arange(0,180,1), cdf_iagos_arr_r_2d[kt,1,tt,:], alpha=0.8,c=mycolor[tt],linewidth=1,linestyle='dashed')
        #x[kt,0].plot(np.arange(0,180,1), cdf_iagos_arr_r_2d[kt,1,tt,:], alpha=0.5,c='k',linewidth=1,linestyle='dashed')
    
    if kt < len(pres)-3:
        x[kt,0].set_xticklabels([])
    x[kt,0].plot((100,100),(-1,1.1),color='k',linewidth=2,linestyle='dashed')
    x[kt,0].set_xlim(0,140)
    x[kt,0].set_ylim(-0.1,1.1)
    x[kt,0].tick_params(labelsize=20)
    x[kt,0].xaxis.set_tick_params(width=2,length=5)
    x[kt,0].yaxis.set_tick_params(width=2,length=5)
    x[kt,0].spines['top'].set_linewidth(1.5)
    x[kt,0].spines['left'].set_linewidth(1.5)
    x[kt,0].spines['right'].set_linewidth(1.5)
    x[kt,0].spines['bottom'].set_linewidth(1.5)
    x[kt,0].set_title('p-level: '+str(pres[kt])+' hPa ',fontsize = 20)
    x[kt,0].text(2,0.95,'('+alpha_string[kt]+')',fontsize=20)
    
    x[kt,0].set_ylabel('Probability distribution',fontsize = 20)
x[0,0].legend(shadow=True,fontsize=20,loc='lower right')
x[len(pres)-3,0].set_xlabel('Relative humidity w.r.t ice [%]',fontsize = 20)
filename = 'cdf_layers_r_era_iagos.png'
if server == 0:
    F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
    F.show()
if server == 1:
    F.savefig('/homedata/kwolf/40_era_iagos/'+filename,bbox_inches='tight')
    plt.close()
    #F.show()


#%%

filename='cdf_distributions'
#############
#save the stats
print('#save the stats')
if server == 0:
    save_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)
if server == 1:
    save_stats_file = ('/homedata/kwolf/40_era_iagos/'+filename)
print('saved to: '+str(save_stats_file))

#--write distribtuions to npz file
np.savez(save_stats_file,cdf_iagos_arr_r,cdf_era_arr_r,cdf_iagos_arr_t,cdf_era_arr_t,cdf_iagos_arr_r_2d,cdf_era_arr_r_2d,r_2d_tbins,r_2d_tlat_bins)

