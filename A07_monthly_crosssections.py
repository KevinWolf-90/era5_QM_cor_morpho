#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:33:51 2022

@author: kwolf
"""

#--create monthly cross sections
#--with lon, lat, and pressure projected in the 2d space


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# --3d scatterplot
from mpl_toolkits.mplot3d import axes3d
import copy
import netCDF4 as nc
from pathlib import Path
import time
import string
from matplotlib.ticker import MaxNLocator
# statistical imports
from scipy import stats
import sys
import os
import xarray as xr
import os.path
import warnings

#--used to convert longitude from -180 180 to 0 360
def convert_lon(lonIn):
    convLon = (lonIn + 360.) % 360.
    return convLon

def my_histogram(value, xmin, xmax, step):
    # used to calculate my own histograms
    dummy = copy.deepcopy(value)
    dummy = dummy.reshape(dummy[:].size)
    yyy, xxx = np.histogram(dummy[:], bins=np.linspace(
        xmin, xmax, int((xmax-xmin)/step)))
    y_total = np.nansum(yyy)
    yyy = np.divide(yyy, y_total)

    return(xxx, yyy)


#--disable warning. keep output clean
warnings.filterwarnings("ignore")
print('#################################')
print('Filter warnings are switched off!')
print('#################################')
time.sleep(1)


#server = 0  # 0 if local, 1 if on server
server = 1

if server == 1:  # to not use Xwindow
    if any('SPYDER' in name for name in os.environ):
        print('Activated plotting on screen')
    else:
        print('Deactivated plotting on screen for terminal and batch')
        matplotlib.use('Agg')

# --import my routines
if server == 0:
    sys.path.insert(1, '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines')
if server == 1:
    sys.path.insert(1, '/homedata/kwolf/40_era_iagos/00_code')


from CritTemp_rasp import CritTemp_rasp
from rh_liquid_to_rh_ice_ecmwf import rh_liquid_to_rh_ice_ecmwf
from rh_ice_to_rh_liquid import rh_ice_to_rh_liquid


#--which years
processYear = [2015,2016,2017,2018,2019,2020,2021]
#--which months
processMonth = list(np.arange(1,13))
#--number of years to process
nYears = len(processYear)
nMonths = len(processMonth)

geoBoundaries = np.asarray([-110,30,30,70]) #--lon min, lon max, lat min, lat max
print('Selected boundaries in normal space: Min Lon, Max Lon, Min Lat, Max Lat: ',geoBoundaries)


#--fuel and model properties
Q = 43e6 #--specific combustion energy; values are for JetA1
EI = 1.25 #--water vapor emission index
eta = 0.35 #--aircraft-engine-efficiency
rhi_crit = 0.95 #--crit threshold for ice supersaturation

startTime = time.time()

# %%
# --read original era data
#-- just read one single month to get the shape of the data
for yearCounter in np.arange(0,1):
    for monthCounter in np.arange(0,1):

        if server == 0:
            file_era2 = '/home/kwolf/Documents/00_CLIMAVIATION/03_ERA5_netcdf_025/'+str(f'{processYear[yearCounter]:04.0f}')+ \
                '_'+str(f'{processMonth[monthCounter]:02.0f}')+'_era5_1hour_t_r_u_v_180W_180E_30N_70N.grib'
                
        if server == 1:
            file_era2 = '/scratchx/kwolf/ERA5/'+str(f'{processYear[yearCounter]:04.0f}')+'/'+str(f'{processYear[yearCounter]:04.0f}')+ \
                '_'+str(f'{processMonth[monthCounter]:02.0f}')+'_era5_1hour_t_r_u_v_180W_180E_30N_70N.grib'
            

        print('Read: ', file_era2)
        xr_era2 = xr.open_dataset(file_era2)
        print(xr_era2)
        #--for my repository
        levels_era2 = list(xr_era2['isobaricInhPa'].values)
        lons_era2 = xr_era2.longitude.values
        lats_era2 = xr_era2.latitude.values
        times_era2 = xr_era2.time.values

        # --select the region of interest
        lat_ind2 = (lats_era2 > geoBoundaries[2]) & (lats_era2 < geoBoundaries[3])
        lon_ind2 = ((lons_era2 > geoBoundaries[0]) & (lons_era2 <= geoBoundaries[1]))
        # --cut to the selected region
        lats_era2 = lats_era2[lat_ind2]
        lons_era2 = lons_era2[lon_ind2]
        #--get the number of lons and lats
        nLats = round(len(lats_era2)/2)
        nLons = round(len(lons_era2)/2)
        nLevels = len(levels_era2)

        #--close dataset to free memory
        xr_era2.close()


#--try to grab all the information and calculate at the end
#--do not know the number of times because month have different lengths
era_T_month_mean_dummy = np.zeros((nYears,nMonths,nLevels,nLats,nLons))    # reduce the number of lons and lats by half; intotal 2^2
era_rh_month_mean_dummy = np.zeros((nYears,nMonths,nLevels,nLats,nLons))
era_u_month_mean_dummy = np.zeros((nYears,nMonths,nLevels,nLats,nLons))
era_v_month_mean_dummy = np.zeros((nYears,nMonths,nLevels,nLats,nLons))
era_wspd_month_mean_dummy = np.zeros((nYears,nMonths,nLevels,nLats,nLons))
PC_flag_monthDummy = np.zeros((nYears,nMonths,nLevels,nLats,nLons))

PCAllDummy = np.zeros((nYears,nMonths,nLevels,nLats)) #--fraction of points that are P-contrail full domain
PCUsDummy = np.zeros((nYears,nMonths,nLevels,nLats)) #--fraction of points that are P-contrail in us / watlantic domain
PCAtlanticDummy = np.zeros((nYears,nMonths,nLevels,nLats)) #--fraction of points that are P-contrail in atlantic domain
PCEuropeDummy = np.zeros((nYears,nMonths,nLevels,nLats)) #--fraction of points that are P-contrail in europe / eatlatnic domain


#%%
#--load iagos flight altitude distributions
#-- first dimension is the region: all, us, NA, eu ; second is the altitude
filename='iagos_flight_altitude_distributions.npz'                                                                  
#############                                                                                 
#--read the stats                                                                                                                                       
if server == 0:                                                                               
    saved_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
if server == 1:                                                                               
    saved_stats_file = ('/homedata/kwolf/40_era_iagos/'+filename) 
print('reading: ',saved_stats_file)
print('')
dummy = np.load(saved_stats_file,allow_pickle=True)
iagos_fad = np.asarray(dummy['arr_0'])
iagos_fad_alt = np.asarray(dummy['arr_1'])
iagos_fad_quantiles = np.asarray(dummy['arr_2'])

#--load iagos flight altitude distributions
#-- first dimension is the region: all, us, NA, eu ; second is the altitude
filename='iagos_flight_latitude_distributions.npz'                                                                  
#############                                                                                 
#--read the stats                                                                                                                                       
if server == 0:                                                                               
    saved_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
if server == 1:                                                                               
    saved_stats_file = ('/homedata/kwolf/40_era_iagos/'+filename) 
print('reading: ',saved_stats_file)
print('')
dummy = np.load(saved_stats_file,allow_pickle=True)
iagos_flatd = np.asarray(dummy['arr_0'])
iagos_flatd_lat = np.asarray(dummy['arr_1'])


#%%

for monthCounter in np.arange(0,nMonths):
    for yearCounter in np.arange(0,nYears):
    
        if server == 0:
            file_era2 = '/home/kwolf/Documents/00_CLIMAVIATION/03_ERA5_netcdf_025/'+str(f'{processYear[yearCounter]:04.0f}')+ \
                '_'+str(f'{processMonth[monthCounter]:02.0f}')+'_era5_1hour_t_r_u_v_180W_180E_30N_70N.grib'
        if server == 1:
            file_era2 = '/scratchx/kwolf/ERA5/'+str(f'{processYear[yearCounter]:04.0f}')+'/'+str(f'{processYear[yearCounter]:04.0f}')+ \
                '_'+str(f'{processMonth[monthCounter]:02.0f}')+'_era5_1hour_t_r_u_v_180W_180E_30N_70N.grib'
            
        print('Read: ', file_era2)
        xr_era2 = xr.open_dataset(file_era2)
        levels_era2 = list(xr_era2['isobaricInhPa'].values)
        lons_era2 = xr_era2.longitude.values
        lats_era2 = xr_era2.latitude.values
        times_era2 = xr_era2.time.values


        # --select the region of interest
        lat_ind2 = (lats_era2 > geoBoundaries[2]) & (lats_era2 < geoBoundaries[3])
        lon_ind2 = ((lons_era2 > geoBoundaries[0]) & (lons_era2 <= geoBoundaries[1]))
        # --cut to the selected region
        lats_era2 = lats_era2[lat_ind2]
        lons_era2 = lons_era2[lon_ind2]

        #--reduce the number of lats and lons in the data
        lats_era2 = lats_era2[::2]
        lons_era2 = lons_era2[::2]

        times_era2 = times_era2[0::6]   #--just reading every 6th timestep

        #--get the number of times in each file
        nDays = len(times_era2)

        era_t = xr_era2.t.sel(time=times_era2, isobaricInhPa=np.asarray( 
            [350,300, 250, 225, 200, 175,150], dtype=np.float64), longitude=lons_era2, latitude=lats_era2)
        era_r = xr_era2.r.sel(time=times_era2, isobaricInhPa=np.asarray(
            [350,300, 250, 225, 200, 175,150], dtype=np.float64), longitude=lons_era2, latitude=lats_era2)

        era_u = xr_era2.u.sel(time=times_era2, isobaricInhPa=np.asarray(
            [350,300, 250, 225, 200, 175,150], dtype=np.float64), longitude=lons_era2, latitude=lats_era2)
        era_v = xr_era2.v.sel(time=times_era2, isobaricInhPa=np.asarray(
            [350,300, 250, 225, 200, 175,150], dtype=np.float64), longitude=lons_era2, latitude=lats_era2)
        
        # --convert to numpy array
        era_t = np.asarray(era_t)
        era_r = np.asarray(era_r)
        #--convert to liquid
        era_rLiquid = rh_ice_to_rh_liquid(era_r/100.,era_t)
        #-- read windspeed
        era_u = np.asarray(era_u)
        era_v = np.asarray(era_v)
        #--calculate windspeed
        era_wspd = np.sqrt(era_u**2 + era_v**2)
        
        #--close file and free memonry
        xr_era2.close()

        
        
        #--make the monthly mean
        era_T_month_mean_dummy[yearCounter,monthCounter,:,:,:] = np.nanmean(era_t,axis=(0))
        era_rh_month_mean_dummy[yearCounter,monthCounter,:,:,:] = np.nanmean(era_r,axis=(0))
        era_u_month_mean_dummy[yearCounter,monthCounter,:,:,:] = np.nanmean(era_u,axis=(0))
        era_v_month_mean_dummy[yearCounter,monthCounter,:,:,:] = np.nanmean(era_v,axis=(0))
        era_wspd_month_mean_dummy[yearCounter,monthCounter,:,:,:] = np.nanmean(era_wspd,axis=(0))
        
        
        
        
        PC_flag = np.zeros((len(times_era2), nLevels, nLats, nLons )) ##--array to store the PC contrail flaggs size: time, levels,lon,lat
        #--expand the pressure column to all dimensions
        levels_era2Expanded = np.zeros((era_t.shape[1],era_t.shape[2],era_t.shape[3]))
        levels_era2Array = np.asarray(levels_era2) #--conversion from list to array required
        levels_era2Expanded[:,:,:] = levels_era2Array[:,None,None]           
        crit_temp_profile = CritTemp_rasp(era_t,levels_era2Expanded*100.,era_rLiquid,eta,Q,Ein=EI) # temperature in k, pressure in pa, and rel hum in 0-1
        crit_temp_profile = crit_temp_profile[:,:,:,:,0,:].T
        pot_layers_index1 = np.where((era_t[:,:,:,:] < crit_temp_profile[0,:,:,:,:]) & (era_rLiquid[:,:,:,:] > crit_temp_profile[1,:,:,:,:])  & (crit_temp_profile[2,:,:,:,:] >= rhi_crit) & (era_t[:,:,:,:] <= (-38+273.15)))  #--diagramgroup 1
        PC_flag[pot_layers_index1] = 1
        
        print('Print raw PC Flag: ',PC_flag.shape)
       
        PC_flag_monthDummy[yearCounter, monthCounter,:,:,:] = np.nansum(PC_flag[:,:,:,:],axis=(0)) / (nDays)
 
        #--full domain
        lon_ind4 = ((lons_era2 > -105) & (lons_era2 <= 30))
        PCAllDummy[yearCounter,monthCounter,:,:] = np.nansum(PC_flag[:,:,:,lon_ind4],axis=(0,3)) / (nDays * np.nansum(lon_ind4))
        #--us domain
        lon_ind4 = ((lons_era2 > -105) & (lons_era2 <= -65))
        PCUsDummy[yearCounter,monthCounter,:,:] = np.nansum(PC_flag[:,:,:,lon_ind4],axis=(0,3)) / (nDays * np.nansum(lon_ind4))
        #--atlantic domain
        lon_ind4 = ((lons_era2 > -65) & (lons_era2 <= -5))
        PCAtlanticDummy[yearCounter,monthCounter,:,:] = np.nansum(PC_flag[:,:,:,lon_ind4],axis=(0,3)) / (nDays * np.nansum(lon_ind4))
        #--eu domain
        lon_ind4 = ((lons_era2 > -5) & (lons_era2 <= 30))
        PCEuropeDummy[yearCounter,monthCounter,:,:] = np.nansum(PC_flag[:,:,:,lon_ind4],axis=(0,3)) / (nDays * np.nansum(lon_ind4))

        
        
        print('Year counter: ',yearCounter) 
        print('month counter: ',monthCounter) 
        

#--after reading all the data you can make a mean over all the years
era_T_month_mean = np.nanmean(era_T_month_mean_dummy,axis=(0))
era_rh_month_mean = np.nanmean(era_rh_month_mean_dummy,axis=(0))
era_u_month_mean = np.nanmean(era_u_month_mean_dummy,axis=(0))
era_v_month_mean = np.nanmean(era_v_month_mean_dummy,axis=(0))
era_wspd_month_mean = np.nanmean(era_wspd_month_mean_dummy,axis=(0))

PC_flag_month = np.nansum(PC_flag,axis=(0)) / nYears

PCAllDummy = np.nansum(PCAllDummy,axis=(0)) / nYears 
PCUsDummy = np.nansum(PCUsDummy,axis=(0)) / nYears 
PCAtlanticDummy = np.nansum(PCAtlanticDummy,axis=(0)) / nYears 
PCEuropeDummy = np.nansum(PCEuropeDummy,axis=(0)) / nYears 

print('shape of PCUsDummy: ', PCUsDummy.shape)

endTime = time.time()

print('Required duration: ',endTime-startTime)

#%%

this_section = 1

if this_section == 1:
    yrangeplot = [0,0.4]
    
    F,x=plt.subplots(2,2,figsize=(12,8),squeeze=False)
    #--plot for DJF
    x[0,0].plot(0,0)
    my_x_labels=['','US','','Atlantic','','Europe', '', 'Full']
    x[0,0].set_xticks(np.arange(1,9))
    x[0,0].set_xticklabels([])
    
    
    #--us
    x[0,0].bar(2-0.4,np.nansum(PCUsDummy[np.array([11,0,1]),2,:],axis=(0,1)) / (3 * nLats) ,0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,0].bar(2-0.2,np.nansum(PCUsDummy[np.array([11,0,1]),3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,0].bar(2,np.nansum(PCUsDummy[np.array([11,0,1]),4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,0].bar(2+0.2,np.nansum(PCUsDummy[np.array([11,0,1]),5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--atlantic
    x[0,0].bar(4-0.4,np.nansum(PCAtlanticDummy[np.array([11,0,1]),2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,0].bar(4-0.2,np.nansum(PCAtlanticDummy[np.array([11,0,1]),3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,0].bar(4,np.nansum(PCAtlanticDummy[np.array([11,0,1]),4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,0].bar(4+0.2,np.nansum(PCAtlanticDummy[np.array([11,0,1]),5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--eu
    x[0,0].bar(6-0.4,np.nansum(PCEuropeDummy[np.array([11,0,1]),2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,0].bar(6-0.2,np.nansum(PCEuropeDummy[np.array([11,0,1]),3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,0].bar(6,np.nansum(PCEuropeDummy[np.array([11,0,1]),4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,0].bar(6+0.2,np.nansum(PCEuropeDummy[np.array([11,0,1]),5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--all
    x[0,0].bar(8-0.4,np.nansum(PCAllDummy[np.array([11,0,1]),2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,0].bar(8-0.2,np.nansum(PCAllDummy[np.array([11,0,1]),3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,0].bar(8,np.nansum(PCAllDummy[np.array([11,0,1]),4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,0].bar(8+0.2,np.nansum(PCAllDummy[np.array([11,0,1]),5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    
    #--make nice horizontal lines for orientation
    for f in np.arange(0,1,0.05):
        x[0,0].plot((0,10),(f,f),linewidth=1,linestyle='dotted',color='k')
    
    x[0,0].set_xlim(1,9)
    x[0,0].set_ylim(yrangeplot[0],yrangeplot[1])
    x[0,0].tick_params(labelsize=18)
    x[0,0].xaxis.set_tick_params(width=2,length=5)
    x[0,0].yaxis.set_tick_params(width=2,length=5)
    x[0,0].spines['top'].set_linewidth(1.5)
    x[0,0].spines['left'].set_linewidth(1.5)
    x[0,0].spines['right'].set_linewidth(1.5)
    x[0,0].spines['bottom'].set_linewidth(1.5)
    x[0,0].set_ylabel('Occurence [0-1]',fontsize = 18)
    x[0,0].text(1.0,yrangeplot[1]*0.9,'(a) DJF',fontsize = 18)
    
    #--plot for MAM
    x[0,1].plot(0,0)
    my_x_labels=['','US','','Atlantic','','Europe','','Full']
    x[0,1].set_xticks(np.arange(1,9))
    x[0,1].set_xticklabels([])
    
    #--us
    x[0,1].bar(2-0.4,np.nansum(PCUsDummy[2:5,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,1].bar(2-0.2,np.nansum(PCUsDummy[2:5,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,1].bar(2,np.nansum(PCUsDummy[2:5,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,1].bar(2+0.2,np.nansum(PCUsDummy[2:5,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--atlantic
    x[0,1].bar(4-0.4,np.nansum(PCAtlanticDummy[2:5,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,1].bar(4-0.2,np.nansum(PCAtlanticDummy[2:5,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,1].bar(4,np.nansum(PCAtlanticDummy[2:5,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,1].bar(4+0.2,np.nansum(PCAtlanticDummy[2:5,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--eu
    x[0,1].bar(6-0.4,np.nansum(PCEuropeDummy[2:5,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,1].bar(6-0.2,np.nansum(PCEuropeDummy[2:5,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,1].bar(6,np.nansum(PCEuropeDummy[2:5,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,1].bar(6+0.2,np.nansum(PCEuropeDummy[2:5,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    
    #--all
    x[0,1].bar(8-0.4,np.nansum(PCAllDummy[2:5,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[0,1].bar(8-0.2,np.nansum(PCAllDummy[2:5,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[0,1].bar(8,np.nansum(PCAllDummy[2:5,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[0,1].bar(8+0.2,np.nansum(PCAllDummy[2:5,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    
    #--make nice horizontal lines for orientation
    for f in np.arange(0,1,0.05):
        x[0,1].plot((0,10),(f,f),linewidth=1,linestyle='dotted',color='k')
    
    x[0,1].set_xlim(1,9)
    x[0,1].set_ylim(yrangeplot[0],yrangeplot[1])
    x[0,1].tick_params(labelsize=18)
    x[0,1].xaxis.set_tick_params(width=2,length=5)
    x[0,1].yaxis.set_tick_params(width=2,length=5)
    x[0,1].spines['top'].set_linewidth(1.5)
    x[0,1].spines['left'].set_linewidth(1.5)
    x[0,1].spines['right'].set_linewidth(1.5)
    x[0,1].spines['bottom'].set_linewidth(1.5)
    x[0,1].text(1.0,yrangeplot[1]*0.9,'(b) MAM',fontsize = 18)
    
    #--plot for JJA
    x[1,0].plot(0,0)
    my_x_labels=['','US','','Atlantic','','Europe', '', 'Full']
    x[1,0].set_xticks(np.arange(1,9))
    x[1,0].set_xticklabels(my_x_labels)
    
    #--us
    x[1,0].bar(2-0.4,np.nansum(PCUsDummy[5:8,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,0].bar(2-0.2,np.nansum(PCUsDummy[5:8,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,0].bar(2,np.nansum(PCUsDummy[5:8,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,0].bar(2+0.2,np.nansum(PCUsDummy[5:8,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--atlantic
    x[1,0].bar(4-0.4,np.nansum(PCAtlanticDummy[5:8,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,0].bar(4-0.2,np.nansum(PCAtlanticDummy[5:8,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,0].bar(4,np.nansum(PCAtlanticDummy[5:8,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,0].bar(4+0.2,np.nansum(PCAtlanticDummy[5:8,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--eu
    x[1,0].bar(6-0.4,np.nansum(PCEuropeDummy[5:8,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,0].bar(6-0.2,np.nansum(PCEuropeDummy[5:8,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,0].bar(6,np.nansum(PCEuropeDummy[5:8,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,0].bar(6+0.2,np.nansum(PCEuropeDummy[5:8,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--all
    x[1,0].bar(8-0.4,np.nansum(PCAllDummy[5:8,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,0].bar(8-0.2,np.nansum(PCAllDummy[5:8,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,0].bar(8,np.nansum(PCAllDummy[5:8,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,0].bar(8+0.2,np.nansum(PCAllDummy[5:8,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    
    #--make nice horizontal lines for orientation
    for f in np.arange(0,1,0.05):
        x[1,0].plot((0,10),(f,f),linewidth=1,linestyle='dotted',color='k')
    
    x[1,0].set_xlim(1,9)
    x[1,0].set_ylim(yrangeplot[0],yrangeplot[1])
    x[1,0].tick_params(labelsize=18)
    x[1,0].xaxis.set_tick_params(width=2,length=5)
    x[1,0].yaxis.set_tick_params(width=2,length=5)
    x[1,0].spines['top'].set_linewidth(1.5)
    x[1,0].spines['left'].set_linewidth(1.5)
    x[1,0].spines['right'].set_linewidth(1.5)
    x[1,0].spines['bottom'].set_linewidth(1.5)
    x[1,0].set_ylabel('Occurence [0-1]',fontsize = 18)
    x[1,0].text(1.0,yrangeplot[1]*0.9,'(c) JJA',fontsize = 18)
    
    #--plot for SON
    x[1,1].plot(0,0)
    my_x_labels=['','US','','Atlantic','','Europe','','Full']
    x[1,1].set_xticks(np.arange(1,9))
    x[1,1].set_xticklabels( (my_x_labels) )
    
    #--us
    x[1,1].bar(2-0.4,np.nansum(PCUsDummy[8:11,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,1].bar(2-0.2,np.nansum(PCUsDummy[8:11,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,1].bar(2,np.nansum(PCUsDummy[8:11,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,1].bar(2+0.2,np.nansum(PCUsDummy[8:11,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--atlantic
    x[1,1].bar(4-0.4,np.nansum(PCAtlanticDummy[8:11,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,1].bar(4-0.2,np.nansum(PCAtlanticDummy[8:11,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,1].bar(4,np.nansum(PCAtlanticDummy[8:11,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,1].bar(4+0.2,np.nansum(PCAtlanticDummy[8:11,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--eu
    x[1,1].bar(6-0.4,np.nansum(PCEuropeDummy[8:11,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,1].bar(6-0.2,np.nansum(PCEuropeDummy[8:11,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,1].bar(6,np.nansum(PCEuropeDummy[8:11,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,1].bar(6+0.2,np.nansum(PCEuropeDummy[8:11,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    #--all
    x[1,1].bar(8-0.4,np.nansum(PCAllDummy[8:11,2],axis=(0,1)) / (3 * nLats),0.2,color='blue',zorder=3) #plot spring for plevel 2, 250
    x[1,1].bar(8-0.2,np.nansum(PCAllDummy[8:11,3],axis=(0,1)) / (3 * nLats),0.2,color='orange',zorder=3) #plot spring for plevel 3, 225
    x[1,1].bar(8,np.nansum(PCAllDummy[8:11,4],axis=(0,1)) / (3 * nLats),0.2,color='green',zorder=3) #plot spring for plevel 4, 200
    x[1,1].bar(8+0.2,np.nansum(PCAllDummy[8:11,5],axis=(0,1)) / (3 * nLats),0.2,color='red',zorder=3) #plot spring for plevel 5, 175
    
    #--make nice horizontal lines for orientation
    for f in np.arange(0,1,0.05):
        x[1,1].plot((0,10),(f,f),linewidth=1,linestyle='dotted',color='k')
    
    x[1,1].set_xlim(1,9)
    x[1,1].set_ylim(yrangeplot[0],yrangeplot[1])
    x[1,1].tick_params(labelsize=18)
    x[1,1].xaxis.set_tick_params(width=2,length=5)
    x[1,1].yaxis.set_tick_params(width=2,length=5)
    x[1,1].spines['top'].set_linewidth(1.5)
    x[1,1].spines['left'].set_linewidth(1.5)
    x[1,1].spines['right'].set_linewidth(1.5)
    x[1,1].spines['bottom'].set_linewidth(1.5)
    x[1,1].text(1.0,yrangeplot[1]*0.9,'(d) SON',fontsize = 18)
    
    
    #--dummy for the legend
    x[0,0].plot((0,0),(0,0),color='blue',linewidth=4,label='250 hpa')
    x[0,0].plot((0,0),(0,0),color='orange',linewidth=4,label='225 hpa')
    x[0,0].plot((0,0),(0,0),color='green',linewidth=4,label='200 hpa')
    x[0,0].plot((0,0),(0,0),color='red',linewidth=4,label='175 hpa')
    x[0,0].legend(fontsize=18,loc='upper right')
    
    filename = 'Poccur_per_season_per_region_per_level.png'
    if server == 0:
        F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
        F.show()
    if server == 1:
        F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
        plt.close()



    
#%%



#--make this plot for the entire area but also for the individual regions
#--loop over the areas instead of creating multiple plots

this_section = 1
if this_section == 1:
    #--labels
    region_labels=['all','us','atlantic','eu']
    
    #--loop over regions
    for regi in np.arange(0,4):
        
        T_levels = np.arange(-80+273,-20+273,2)
        r_levels = np.arange(0,140,10)
        wspd_levels = np.arange(0,75,5)
        Pc_levels = np.arange(0,0.5,0.05)
        
        if regi == 0:    
            #--full domain
            lon_ind4 = ((lons_era2 > -105) & (lons_era2 <= 30))
            #--have to make thhis stupid step to keep the era_CO... variable and not to replace it everywhere.
            era_CO_month_mean_region = PCAllDummy
        if regi == 1:    
            #--us domain
            lon_ind4 = ((lons_era2 > -105) & (lons_era2 <= -65))
            era_CO_month_mean_region = PCUsDummy
        if regi == 2:    
            #--atlantic domain
            lon_ind4 = ((lons_era2 > -65) & (lons_era2 <= -5)) 
            era_CO_month_mean_region = PCAtlanticDummy
        if regi == 3:    
            #--eu domain
            lon_ind4 = ((lons_era2 > -5) & (lons_era2 <= 30))
            era_CO_month_mean_region = PCEuropeDummy
        
        era_T_month_mean_region = np.nanmean(era_T_month_mean[:,:,:,lon_ind4],axis=(3))
        era_rh_month_mean_region = np.nanmean(era_rh_month_mean[:,:,:,lon_ind4],axis=(3))
        era_u_month_mean_region = np.nanmean(era_u_month_mean[:,:,:,lon_ind4],axis=(3))
        era_v_month_mean_region = np.nanmean(era_v_month_mean[:,:,:,lon_ind4],axis=(3))
        era_wspd_month_mean_region = np.nanmean(era_wspd_month_mean[:,:,:,lon_ind4],axis=(3))
            
             
            
        
        F,x=plt.subplots(4,5,figsize=(25,20),squeeze=False,gridspec_kw={'width_ratios': [1,1,1,1, 0.5]})
        x[0,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,0].contourf(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[np.array([11,0,1]),:,:],axis=(0)),levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,0].contour(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[np.array([11,0,1]),:,:],axis=(0)),levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[0,0].set_xlim(30,60)
        x[0,0].set_ylim(350,150)
        x[0,0].set_xticklabels([])
        x[0,0].tick_params(labelsize=20)
        x[0,0].xaxis.set_tick_params(width=2,length=5)
        x[0,0].yaxis.set_tick_params(width=2,length=5)
        x[0,0].spines['top'].set_linewidth(1.5)
        x[0,0].spines['left'].set_linewidth(1.5)
        x[0,0].spines['right'].set_linewidth(1.5)
        x[0,0].spines['bottom'].set_linewidth(1.5)
        x[0,0].set_ylabel('DJF \n Pressure [hPa]',fontsize = 20)
        x[0,0].set_title('Zonal mean temperature [K] \n',fontsize=20)
        
        x[0,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,1].contourf(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[np.array([11,0,1]),:,:],axis=(0)),levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,1].contour(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[np.array([11,0,1]),:,:],axis=(0)),levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[0,1].set_xlim(30,60)
        x[0,1].set_ylim(350,150)
        x[0,1].set_xticklabels([])
        x[0,1].set_yticklabels([])
        x[0,1].tick_params(labelsize=20)
        x[0,1].xaxis.set_tick_params(width=2,length=5)
        x[0,1].yaxis.set_tick_params(width=2,length=5)
        x[0,1].spines['top'].set_linewidth(1.5)
        x[0,1].spines['left'].set_linewidth(1.5)
        x[0,1].spines['right'].set_linewidth(1.5)
        x[0,1].spines['bottom'].set_linewidth(1.5)
        x[0,1].set_title('Zonal mean rel. humidity [%] \n',fontsize=20)
        
        x[0,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,2].contourf(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[np.array([11,0,1]),:,:],axis=(0)),levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,2].contour(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[np.array([11,0,1]),:,:],axis=(0)),levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[0,2].set_xlim(30,60)
        x[0,2].set_ylim(350,150)
        x[0,2].set_xticklabels([])
        x[0,2].set_yticklabels([])
        x[0,2].tick_params(labelsize=20)
        x[0,2].xaxis.set_tick_params(width=2,length=5)
        x[0,2].yaxis.set_tick_params(width=2,length=5)
        x[0,2].spines['top'].set_linewidth(1.5)
        x[0,2].spines['left'].set_linewidth(1.5)
        x[0,2].spines['right'].set_linewidth(1.5)
        x[0,2].spines['bottom'].set_linewidth(1.5)
        x[0,2].set_title('Zonal mean windspeed [m s$^{-1}$] \n',fontsize=20)
        
        x[0,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,3].contourf(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[np.array([11,0,1]),:,:],axis=(0)) / 3,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,3].contour(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[np.array([11,0,1]),:,:],axis=(0)) / 3,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[0,3].set_xlim(30,60)
        x[0,3].set_ylim(350,150)
        x[0,3].set_xticklabels([])
        x[0,3].set_yticklabels([])
        x[0,3].tick_params(labelsize=20)
        x[0,3].xaxis.set_tick_params(width=2,length=5)
        x[0,3].yaxis.set_tick_params(width=2,length=5)
        x[0,3].spines['top'].set_linewidth(1.5)
        x[0,3].spines['left'].set_linewidth(1.5)
        x[0,3].spines['right'].set_linewidth(1.5)
        x[0,3].spines['bottom'].set_linewidth(1.5)
        x[0,3].set_title('Zonal mean PC occurnce [0-1] \n',fontsize=20)
        
        x[0,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,4].plot(iagos_fad[0,:],iagos_fad_alt[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[0,4].set_xlim(0.,0.4)
        x[0,4].set_ylim(350,150)
        x[0,4].set_xticklabels([])
        x[0,4].set_yticklabels([])
        x[0,4].tick_params(labelsize=20)
        x[0,4].xaxis.set_tick_params(width=2,length=5)
        x[0,4].yaxis.set_tick_params(width=2,length=5)
        x[0,4].spines['top'].set_linewidth(1.5)
        x[0,4].spines['left'].set_linewidth(1.5)
        x[0,4].spines['right'].set_linewidth(1.5)
        x[0,4].spines['bottom'].set_linewidth(1.5)
        
        x[1,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,0].contourf(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[2:5,:,:],axis=(0)),levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,0].contour(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[2:5,:,:],axis=(0)),levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[1,0].set_xlim(30,60)
        x[1,0].set_ylim(350,150)
        x[1,0].set_xticklabels([])
        x[1,0].tick_params(labelsize=20)
        x[1,0].xaxis.set_tick_params(width=2,length=5)
        x[1,0].yaxis.set_tick_params(width=2,length=5)
        x[1,0].spines['top'].set_linewidth(1.5)
        x[1,0].spines['left'].set_linewidth(1.5)
        x[1,0].spines['right'].set_linewidth(1.5)
        x[1,0].spines['bottom'].set_linewidth(1.5)
        x[1,0].set_ylabel('MAM \n Pressure [hPa]',fontsize = 20)
        
        x[1,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,1].contourf(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[2:5,:,:],axis=(0)),levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,1].contour(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[2:5,:,:],axis=(0)),levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[1,1].set_xlim(30,60)
        x[1,1].set_ylim(350,150)
        x[1,1].set_xticklabels([])
        x[1,1].set_yticklabels([])
        x[1,1].tick_params(labelsize=20)
        x[1,1].xaxis.set_tick_params(width=2,length=5)
        x[1,1].yaxis.set_tick_params(width=2,length=5)
        x[1,1].spines['top'].set_linewidth(1.5)
        x[1,1].spines['left'].set_linewidth(1.5)
        x[1,1].spines['right'].set_linewidth(1.5)
        x[1,1].spines['bottom'].set_linewidth(1.5)

        
        x[1,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,2].contourf(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[2:5,:,:],axis=(0)),levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,2].contour(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[2:5,:,:],axis=(0)),levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[1,2].set_xlim(30,60)
        x[1,2].set_ylim(350,150)
        x[1,2].set_xticklabels([])
        x[1,2].set_yticklabels([])
        x[1,2].tick_params(labelsize=20)
        x[1,2].xaxis.set_tick_params(width=2,length=5)
        x[1,2].yaxis.set_tick_params(width=2,length=5)
        x[1,2].spines['top'].set_linewidth(1.5)
        x[1,2].spines['left'].set_linewidth(1.5)
        x[1,2].spines['right'].set_linewidth(1.5)
        x[1,2].spines['bottom'].set_linewidth(1.5)

        
        x[1,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,3].contourf(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[2:5,:,:],axis=(0)) / 3,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,3].contour(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[2:5,:,:],axis=(0)) / 3,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[1,3].set_xlim(30,60)
        x[1,3].set_ylim(350,150)
        x[1,3].set_xticklabels([])
        x[1,3].set_yticklabels([])
        x[1,3].tick_params(labelsize=20)
        x[1,3].xaxis.set_tick_params(width=2,length=5)
        x[1,3].yaxis.set_tick_params(width=2,length=5)
        x[1,3].spines['top'].set_linewidth(1.5)
        x[1,3].spines['left'].set_linewidth(1.5)
        x[1,3].spines['right'].set_linewidth(1.5)
        x[1,3].spines['bottom'].set_linewidth(1.5)

        
        x[1,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,4].plot(iagos_fad[0,:],iagos_fad_alt[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[1,4].set_xlim(0.,0.4)
        x[1,4].set_ylim(350,150)
        x[1,4].set_xticklabels([])
        x[1,4].set_yticklabels([])
        x[1,4].tick_params(labelsize=20)
        x[1,4].xaxis.set_tick_params(width=2,length=5)
        x[1,4].yaxis.set_tick_params(width=2,length=5)
        x[1,4].spines['top'].set_linewidth(1.5)
        x[1,4].spines['left'].set_linewidth(1.5)
        x[1,4].spines['right'].set_linewidth(1.5)
        x[1,4].spines['bottom'].set_linewidth(1.5)

        
        x[2,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,0].contourf(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[5:8,:,:],axis=(0)),levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,0].contour(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[5:8,:,:],axis=(0)),levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[2,0].set_xlim(30,60)
        x[2,0].set_ylim(350,150)
        x[2,0].set_xticklabels([])
        x[2,0].tick_params(labelsize=20)
        x[2,0].xaxis.set_tick_params(width=2,length=5)
        x[2,0].yaxis.set_tick_params(width=2,length=5)
        x[2,0].spines['top'].set_linewidth(1.5)
        x[2,0].spines['left'].set_linewidth(1.5)
        x[2,0].spines['right'].set_linewidth(1.5)
        x[2,0].spines['bottom'].set_linewidth(1.5)
        x[2,0].set_ylabel('JJA \n Pressure [hPa]',fontsize = 20)
        
        x[2,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,1].contourf(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[5:8,:,:],axis=(0)),levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,1].contour(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[5:8,:,:],axis=(0)),levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[2,1].set_xlim(30,60)
        x[2,1].set_ylim(350,150)
        x[2,1].set_xticklabels([])
        x[2,1].set_yticklabels([])
        x[2,1].tick_params(labelsize=20)
        x[2,1].xaxis.set_tick_params(width=2,length=5)
        x[2,1].yaxis.set_tick_params(width=2,length=5)
        x[2,1].spines['top'].set_linewidth(1.5)
        x[2,1].spines['left'].set_linewidth(1.5)
        x[2,1].spines['right'].set_linewidth(1.5)
        x[2,1].spines['bottom'].set_linewidth(1.5)

        
        x[2,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,2].contourf(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[5:8,:,:],axis=(0)),levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,2].contour(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[5:8,:,:],axis=(0)),levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[2,2].set_xlim(30,60)
        x[2,2].set_ylim(350,150)
        x[2,2].set_xticklabels([])
        x[2,2].set_yticklabels([])
        x[2,2].tick_params(labelsize=20)
        x[2,2].xaxis.set_tick_params(width=2,length=5)
        x[2,2].yaxis.set_tick_params(width=2,length=5)
        x[2,2].spines['top'].set_linewidth(1.5)
        x[2,2].spines['left'].set_linewidth(1.5)
        x[2,2].spines['right'].set_linewidth(1.5)
        x[2,2].spines['bottom'].set_linewidth(1.5)

        
        x[2,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,3].contourf(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[5:8,:,:],axis=(0)) / 3,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,3].contour(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[5:8,:,:],axis=(0)) / 3,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[2,3].set_xlim(30,60)
        x[2,3].set_ylim(350,150)
        x[2,3].set_xticklabels([])
        x[2,3].set_yticklabels([])
        x[2,3].tick_params(labelsize=20)
        x[2,3].xaxis.set_tick_params(width=2,length=5)
        x[2,3].yaxis.set_tick_params(width=2,length=5)
        x[2,3].spines['top'].set_linewidth(1.5)
        x[2,3].spines['left'].set_linewidth(1.5)
        x[2,3].spines['right'].set_linewidth(1.5)
        x[2,3].spines['bottom'].set_linewidth(1.5)

        
        x[2,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,4].plot(iagos_fad[0,:],iagos_fad_alt[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[2,4].set_xlim(0.,0.4)
        x[2,4].set_ylim(350,150)
        x[2,4].set_xticklabels([])
        x[2,4].set_yticklabels([])
        x[2,4].tick_params(labelsize=20)
        x[2,4].xaxis.set_tick_params(width=2,length=5)
        x[2,4].yaxis.set_tick_params(width=2,length=5)
        x[2,4].spines['top'].set_linewidth(1.5)
        x[2,4].spines['left'].set_linewidth(1.5)
        x[2,4].spines['right'].set_linewidth(1.5)
        x[2,4].spines['bottom'].set_linewidth(1.5)

        
        x[3,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,0].contourf(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[8:11,:,:],axis=(0)),levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,0].contour(lats_era2,levels_era2,np.nanmean(era_T_month_mean_region[8:11,:,:],axis=(0)),levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[3,0].set_xlim(30,60)
        x[3,0].set_ylim(350,150)
        x[3,0].tick_params(labelsize=20)
        x[3,0].xaxis.set_tick_params(width=2,length=5)
        x[3,0].yaxis.set_tick_params(width=2,length=5)
        x[3,0].spines['top'].set_linewidth(1.5)
        x[3,0].spines['left'].set_linewidth(1.5)
        x[3,0].spines['right'].set_linewidth(1.5)
        x[3,0].spines['bottom'].set_linewidth(1.5)
        x[3,0].set_ylabel('SON \n Pressure [hPa]',fontsize = 20)
        x[3,0].set_xlabel('Latitude [$^\circ$]',fontsize = 20)
        
        x[3,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,1].contourf(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[8:11,:,:],axis=(0)),levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,1].contour(lats_era2,levels_era2,np.nanmean(era_rh_month_mean_region[8:11,:,:],axis=(0)),levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[3,1].set_xlim(30,60)
        x[3,1].set_ylim(350,150)
        x[3,1].set_yticklabels([])
        x[3,1].tick_params(labelsize=20)
        x[3,1].xaxis.set_tick_params(width=2,length=5)
        x[3,1].yaxis.set_tick_params(width=2,length=5)
        x[3,1].spines['top'].set_linewidth(1.5)
        x[3,1].spines['left'].set_linewidth(1.5)
        x[3,1].spines['right'].set_linewidth(1.5)
        x[3,1].spines['bottom'].set_linewidth(1.5)
        x[3,1].set_xlabel('Latitude [$^\circ$]',fontsize = 20)
        
        
        x[3,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,2].contourf(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[8:11,:,:],axis=(0)),levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,2].contour(lats_era2,levels_era2,np.nanmean(era_wspd_month_mean_region[8:11,:,:],axis=(0)),levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[3,2].set_xlim(30,60)
        x[3,2].set_ylim(350,150)
        x[3,2].set_yticklabels([])
        x[3,2].tick_params(labelsize=20)
        x[3,2].xaxis.set_tick_params(width=2,length=5)
        x[3,2].yaxis.set_tick_params(width=2,length=5)
        x[3,2].spines['top'].set_linewidth(1.5)
        x[3,2].spines['left'].set_linewidth(1.5)
        x[3,2].spines['right'].set_linewidth(1.5)
        x[3,2].spines['bottom'].set_linewidth(1.5)
        x[3,2].set_xlabel('Latitude [$^\circ$]',fontsize = 20)
        
        x[3,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,3].contourf(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[8:11,:,:],axis=(0)) / 3,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,3].contour(lats_era2,levels_era2,np.nansum(era_CO_month_mean_region[8:11,:,:],axis=(0)) / 3,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[3,3].set_xlim(30,60)
        x[3,3].set_ylim(350,150)
        x[3,3].set_yticklabels([])
        x[3,3].tick_params(labelsize=20)
        x[3,3].xaxis.set_tick_params(width=2,length=5)
        x[3,3].yaxis.set_tick_params(width=2,length=5)
        x[3,3].spines['top'].set_linewidth(1.5)
        x[3,3].spines['left'].set_linewidth(1.5)
        x[3,3].spines['right'].set_linewidth(1.5)
        x[3,3].spines['bottom'].set_linewidth(1.5)
        x[3,3].set_xlabel('Latitude [$^\circ$]',fontsize = 20)
        
        x[3,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,4].plot(iagos_fad[0,:],iagos_fad_alt[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[3,4].set_xlim(0.,0.4)
        x[3,4].set_ylim(350,150)
        x[3,4].set_yticklabels([])
        x[3,4].tick_params(labelsize=20)
        x[3,4].xaxis.set_tick_params(width=2,length=5)
        x[3,4].yaxis.set_tick_params(width=2,length=5)
        x[3,4].spines['top'].set_linewidth(1.5)
        x[3,4].spines['left'].set_linewidth(1.5)
        x[3,4].spines['right'].set_linewidth(1.5)
        x[3,4].spines['bottom'].set_linewidth(1.5)
        x[3,4].set_xlabel('FAD [0-1]',fontsize = 20)
        
        #--plot the a-p labels
        panel_labels = list(string.ascii_lowercase)
        i = 0
        for z in np.arange(0,4):
            for y in np.arange(0,5):
                if (y!=4):
                    x[z,y].text(30,145,'('+str(panel_labels[i])+')',fontsize=20)
                if (y==4):
                    x[z,y].text(0,145,'('+str(panel_labels[i])+')',fontsize=20)
                i = i+1
        
        
        filename = 'multi_lat_p_for_seasons_'+str(region_labels[regi])+'.png'
        if server == 0:
            F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
            F.show()
        if server == 1:
            F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
            plt.close()
            #F.show()

        
        F,x=plt.subplots(4,5,figsize=(25,20),squeeze=False,gridspec_kw={'width_ratios': [1,1,1,1, 0.5]})
        x[0,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,0].contourf(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,2,:].T,levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,0].contour(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,2,:].T,levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[0,0].set_xlim(1,12)
        x[0,0].set_ylim(30,60)
        x[0,0].set_xticklabels([])
        x[0,0].tick_params(labelsize=20)
        x[0,0].xaxis.set_tick_params(width=2,length=5)
        x[0,0].yaxis.set_tick_params(width=2,length=5)
        x[0,0].spines['top'].set_linewidth(1.5)
        x[0,0].spines['left'].set_linewidth(1.5)
        x[0,0].spines['right'].set_linewidth(1.5)
        x[0,0].spines['bottom'].set_linewidth(1.5)
        x[0,0].set_ylabel('250 hPa \n Latitude [$\circ$]',fontsize = 20)
        x[0,0].set_title('Monthly mean \n temperature [K] \n',fontsize=20)
        
        x[0,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,1].contourf(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,2,:].T,levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,1].contour(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,2,:].T,levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[0,1].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[0,1].set_xlim(1,12)
        x[0,1].set_ylim(30,60)
        x[0,1].set_xticklabels([])
        x[0,1].set_yticklabels([])
        x[0,1].tick_params(labelsize=20)
        x[0,1].xaxis.set_tick_params(width=2,length=5)
        x[0,1].yaxis.set_tick_params(width=2,length=5)
        x[0,1].spines['top'].set_linewidth(1.5)
        x[0,1].spines['left'].set_linewidth(1.5)
        x[0,1].spines['right'].set_linewidth(1.5)
        x[0,1].spines['bottom'].set_linewidth(1.5)
        x[0,1].set_title('Monthly mean \n rel. humidity [%] \n',fontsize=20)
        
        x[0,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,2].contourf(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,2,:].T,levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,2].contour(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,2,:].T,levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[0,2].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[0,2].set_xlim(1,12)
        x[0,2].set_ylim(30,60)
        x[0,2].set_xticklabels([])
        x[0,2].set_yticklabels([])
        x[0,2].tick_params(labelsize=20)
        x[0,2].xaxis.set_tick_params(width=2,length=5)
        x[0,2].yaxis.set_tick_params(width=2,length=5)
        x[0,2].spines['top'].set_linewidth(1.5)
        x[0,2].spines['left'].set_linewidth(1.5)
        x[0,2].spines['right'].set_linewidth(1.5)
        x[0,2].spines['bottom'].set_linewidth(1.5)
        x[0,2].set_title('Monthly mean \n wind speed [m s$^{-1}$] \n',fontsize=20)
        
        x[0,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,3].contourf(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,2,:].T,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[0,3].contour(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,2,:].T,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[0,3].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[0,3].set_xlim(1,12)
        x[0,3].set_ylim(30,60)
        x[0,3].set_xticklabels([])
        x[0,3].set_yticklabels([])
        x[0,3].tick_params(labelsize=20)
        x[0,3].xaxis.set_tick_params(width=2,length=5)
        x[0,3].yaxis.set_tick_params(width=2,length=5)
        x[0,3].spines['top'].set_linewidth(1.5)
        x[0,3].spines['left'].set_linewidth(1.5)
        x[0,3].spines['right'].set_linewidth(1.5)
        x[0,3].spines['bottom'].set_linewidth(1.5)
        x[0,3].set_title('Monthly mean \n PC occurence [0-1] \n',fontsize=20)
        
        x[0,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[0,4].plot(iagos_flatd[0,:],iagos_flatd_lat[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[0,4].set_xlim(0.,0.5)
        x[0,4].set_ylim(30,60)
        x[0,4].set_xticklabels([])
        x[0,4].set_yticklabels([])
        x[0,4].tick_params(labelsize=20)
        x[0,4].xaxis.set_tick_params(width=2,length=5)
        x[0,4].yaxis.set_tick_params(width=2,length=5)
        x[0,4].spines['top'].set_linewidth(1.5)
        x[0,4].spines['left'].set_linewidth(1.5)
        x[0,4].spines['right'].set_linewidth(1.5)
        x[0,4].spines['bottom'].set_linewidth(1.5)
        
        x[1,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,0].contourf(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,3,:].T,levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,0].contour(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,3,:].T,levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[1,0].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[1,0].set_xlim(1,12)
        x[1,0].set_ylim(30,60)
        x[1,0].set_xticklabels([])
        x[1,0].tick_params(labelsize=20)
        x[1,0].xaxis.set_tick_params(width=2,length=5)
        x[1,0].yaxis.set_tick_params(width=2,length=5)
        x[1,0].spines['top'].set_linewidth(1.5)
        x[1,0].spines['left'].set_linewidth(1.5)
        x[1,0].spines['right'].set_linewidth(1.5)
        x[1,0].spines['bottom'].set_linewidth(1.5)
        x[1,0].set_ylabel('225 hPa \n Latitude [$\circ$]',fontsize = 20)
        
        x[1,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,1].contourf(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,3,:].T,levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,1].contour(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,3,:].T,levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[1,1].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[1,1].set_xlim(1,12)
        x[1,1].set_ylim(30,60)
        x[1,1].set_xticklabels([])
        x[1,1].set_yticklabels([])
        x[1,1].tick_params(labelsize=20)
        x[1,1].xaxis.set_tick_params(width=2,length=5)
        x[1,1].yaxis.set_tick_params(width=2,length=5)
        x[1,1].spines['top'].set_linewidth(1.5)
        x[1,1].spines['left'].set_linewidth(1.5)
        x[1,1].spines['right'].set_linewidth(1.5)
        x[1,1].spines['bottom'].set_linewidth(1.5)

        
        x[1,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,2].contourf(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,3,:].T,levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,2].contour(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,3,:].T,levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[1,2].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[1,2].set_xlim(1,12)
        x[1,2].set_ylim(30,60)
        x[1,2].set_xticklabels([])
        x[1,2].set_yticklabels([])
        x[1,2].tick_params(labelsize=20)
        x[1,2].xaxis.set_tick_params(width=2,length=5)
        x[1,2].yaxis.set_tick_params(width=2,length=5)
        x[1,2].spines['top'].set_linewidth(1.5)
        x[1,2].spines['left'].set_linewidth(1.5)
        x[1,2].spines['right'].set_linewidth(1.5)
        x[1,2].spines['bottom'].set_linewidth(1.5)
        
        x[1,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,3].contourf(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,3,:].T,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[1,3].contour(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,3,:].T,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[1,3].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[1,3].set_xlim(1,12)
        x[1,3].set_ylim(30,60)
        x[1,3].set_xticklabels([])
        x[1,3].set_yticklabels([])
        x[1,3].tick_params(labelsize=20)
        x[1,3].xaxis.set_tick_params(width=2,length=5)
        x[1,3].yaxis.set_tick_params(width=2,length=5)
        x[1,3].spines['top'].set_linewidth(1.5)
        x[1,3].spines['left'].set_linewidth(1.5)
        x[1,3].spines['right'].set_linewidth(1.5)
        x[1,3].spines['bottom'].set_linewidth(1.5)

        
        x[1,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[1,4].plot(iagos_flatd[0,:],iagos_flatd_lat[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[1,4].set_xlim(0.,0.5)
        x[1,4].set_ylim(30,60)
        x[1,4].set_xticklabels([])
        x[1,4].set_yticklabels([])
        x[1,4].tick_params(labelsize=20)
        x[1,4].xaxis.set_tick_params(width=2,length=5)
        x[1,4].yaxis.set_tick_params(width=2,length=5)
        x[1,4].spines['top'].set_linewidth(1.5)
        x[1,4].spines['left'].set_linewidth(1.5)
        x[1,4].spines['right'].set_linewidth(1.5)
        x[1,4].spines['bottom'].set_linewidth(1.5)

        
        x[2,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,0].contourf(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,4,:].T,levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,0].contour(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,4,:].T,levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[2,0].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[2,0].set_xlim(1,12)
        x[2,0].set_ylim(30,60)
        x[2,0].set_xticklabels([])
        x[2,0].tick_params(labelsize=20)
        x[2,0].xaxis.set_tick_params(width=2,length=5)
        x[2,0].yaxis.set_tick_params(width=2,length=5)
        x[2,0].spines['top'].set_linewidth(1.5)
        x[2,0].spines['left'].set_linewidth(1.5)
        x[2,0].spines['right'].set_linewidth(1.5)
        x[2,0].spines['bottom'].set_linewidth(1.5)
        x[2,0].set_ylabel('200 hPa \n Latitude [$\circ$]',fontsize = 20)

        
        x[2,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,1].contourf(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,4,:].T,levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,1].contour(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,4,:].T,levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[2,1].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[2,1].set_xlim(1,12)
        x[2,1].set_ylim(30,60)
        x[2,1].set_xticklabels([])
        x[2,1].set_yticklabels([])
        x[2,1].tick_params(labelsize=20)
        x[2,1].xaxis.set_tick_params(width=2,length=5)
        x[2,1].yaxis.set_tick_params(width=2,length=5)
        x[2,1].spines['top'].set_linewidth(1.5)
        x[2,1].spines['left'].set_linewidth(1.5)
        x[2,1].spines['right'].set_linewidth(1.5)
        x[2,1].spines['bottom'].set_linewidth(1.5)

        
        x[2,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,2].contourf(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,4,:].T,levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,2].contour(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,4,:].T,levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[2,2].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[2,2].set_xlim(1,12)
        x[2,2].set_ylim(30,60)
        x[2,2].set_xticklabels([])
        x[2,2].set_yticklabels([])
        x[2,2].tick_params(labelsize=20)
        x[2,2].xaxis.set_tick_params(width=2,length=5)
        x[2,2].yaxis.set_tick_params(width=2,length=5)
        x[2,2].spines['top'].set_linewidth(1.5)
        x[2,2].spines['left'].set_linewidth(1.5)
        x[2,2].spines['right'].set_linewidth(1.5)
        x[2,2].spines['bottom'].set_linewidth(1.5)

        
        x[2,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,3].contourf(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,4,:].T,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[2,3].contour(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,4,:].T,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[1,3].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[2,3].set_xlim(1,12)
        x[2,3].set_ylim(30,60)
        x[2,3].set_xticklabels([])
        x[2,3].set_yticklabels([])
        x[2,3].tick_params(labelsize=20)
        x[2,3].xaxis.set_tick_params(width=2,length=5)
        x[2,3].yaxis.set_tick_params(width=2,length=5)
        x[2,3].spines['top'].set_linewidth(1.5)
        x[2,3].spines['left'].set_linewidth(1.5)
        x[2,3].spines['right'].set_linewidth(1.5)
        x[2,3].spines['bottom'].set_linewidth(1.5)

        
        x[2,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[2,4].plot(iagos_flatd[0,:],iagos_flatd_lat[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[2,4].set_xlim(0.,0.5)
        x[2,4].set_ylim(30,60)
        x[2,4].set_xticklabels([])
        x[2,4].set_yticklabels([])
        x[2,4].tick_params(labelsize=20)
        x[2,4].xaxis.set_tick_params(width=2,length=5)
        x[2,4].yaxis.set_tick_params(width=2,length=5)
        x[2,4].spines['top'].set_linewidth(1.5)
        x[2,4].spines['left'].set_linewidth(1.5)
        x[2,4].spines['right'].set_linewidth(1.5)
        x[2,4].spines['bottom'].set_linewidth(1.5)

        
        x[3,0].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,0].contourf(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,5,:].T,levels=T_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,0].contour(np.arange(0,nMonths)+1,lats_era2,era_T_month_mean_region[:,5,:].T,levels=T_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[3,0].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[3,0].set_xlim(1,12)
        x[3,0].set_ylim(30,60)
        x[3,0].tick_params(labelsize=20)
        x[3,0].xaxis.set_tick_params(width=2,length=5)
        x[3,0].yaxis.set_tick_params(width=2,length=5)
        x[3,0].spines['top'].set_linewidth(1.5)
        x[3,0].spines['left'].set_linewidth(1.5)
        x[3,0].spines['right'].set_linewidth(1.5)
        x[3,0].spines['bottom'].set_linewidth(1.5)
        x[3,0].set_ylabel('175 hPa \n Latitude [$\circ$]',fontsize = 20)
        x[3,0].set_xlabel('Month',fontsize = 20)
        
        x[3,1].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,1].contourf(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,5,:].T,levels=r_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,1].contour(np.arange(0,nMonths)+1,lats_era2,era_rh_month_mean_region[:,5,:].T,levels=r_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[3,1].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[3,1].set_xlim(1,12)
        x[3,1].set_ylim(30,60)
        x[3,1].set_yticklabels([])
        x[3,1].tick_params(labelsize=20)
        x[3,1].xaxis.set_tick_params(width=2,length=5)
        x[3,1].yaxis.set_tick_params(width=2,length=5)
        x[3,1].spines['top'].set_linewidth(1.5)
        x[3,1].spines['left'].set_linewidth(1.5)
        x[3,1].spines['right'].set_linewidth(1.5)
        x[3,1].spines['bottom'].set_linewidth(1.5)
        x[3,1].set_xlabel('Month',fontsize = 20)
        
        
        x[3,2].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,2].contourf(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,5,:].T,levels=wspd_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,2].contour(np.arange(0,nMonths)+1,lats_era2,era_wspd_month_mean_region[:,5,:].T,levels=wspd_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%1.0f')
        x[3,2].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[3,2].set_xlim(1,12)
        x[3,2].set_ylim(30,60)
        x[3,2].set_yticklabels([])
        x[3,2].tick_params(labelsize=20)
        x[3,2].xaxis.set_tick_params(width=2,length=5)
        x[3,2].yaxis.set_tick_params(width=2,length=5)
        x[3,2].spines['top'].set_linewidth(1.5)
        x[3,2].spines['left'].set_linewidth(1.5)
        x[3,2].spines['right'].set_linewidth(1.5)
        x[3,2].spines['bottom'].set_linewidth(1.5)
        x[3,2].set_xlabel('Month',fontsize = 20)
        
        x[3,3].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,3].contourf(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,5,:].T ,levels=Pc_levels,cmap='Greys')  #for plevel2
        cf42 = x[3,3].contour(np.arange(0,nMonths)+1,lats_era2,era_CO_month_mean_region[:,5,:].T,levels=Pc_levels,colors='k',linewidth=2)
        clt=plt.clabel(cf42, fontsize=15, inline=1,fmt = '%3.2f')
        x[3,3].xaxis.set_major_locator(MaxNLocator(integer=True))
        x[3,3].set_xlim(1,12)
        x[3,3].set_ylim(30,60)
        x[3,3].set_yticklabels([])
        x[3,3].tick_params(labelsize=20)
        x[3,3].xaxis.set_tick_params(width=2,length=5)
        x[3,3].yaxis.set_tick_params(width=2,length=5)
        x[3,3].spines['top'].set_linewidth(1.5)
        x[3,3].spines['left'].set_linewidth(1.5)
        x[3,3].spines['right'].set_linewidth(1.5)
        x[3,3].spines['bottom'].set_linewidth(1.5)
        x[3,3].set_xlabel('Month',fontsize = 20)
        
        x[3,4].plot(0,0)
        mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
        x1=x[3,4].plot(iagos_flatd[0,:],iagos_flatd_lat[:],linestyle='solid',linewidth=2,color='k',marker='o')
        x[3,4].set_xlim(0.,0.5)
        x[3,4].set_ylim(30,60)
        x[3,4].set_yticklabels([])
        x[3,4].tick_params(labelsize=20)
        x[3,4].xaxis.set_tick_params(width=2,length=5)
        x[3,4].yaxis.set_tick_params(width=2,length=5)
        x[3,4].spines['top'].set_linewidth(1.5)
        x[3,4].spines['left'].set_linewidth(1.5)
        x[3,4].spines['right'].set_linewidth(1.5)
        x[3,4].spines['bottom'].set_linewidth(1.5)
        x[3,4].set_xlabel('FLATD [0-1]',fontsize = 20)
               
        #--plot the a-p labels
        panel_labels = list(string.ascii_lowercase)
        i = 0
        for z in np.arange(0,4):
            for y in np.arange(0,5):
                if (y!=4):
                    x[z,y].text(1,61,'('+str(panel_labels[i])+')',fontsize=20)
                if (y==4):
                    x[z,y].text(0.01,61,'('+str(panel_labels[i])+')',fontsize=20)
                i = i+1
        
        filename = 'multi_month_lat_'+str(region_labels[regi])+'.png'
        if server == 0:
            F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
            F.show()
        if server == 1:
            F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
            #plt.close()
            F.show()

#%%


this_section = 0
if this_section == 1:


    T_levels = np.arange(-80+273,-25+273,2)
    r_levels = np.arange(0,140,5)
    wspd_levels = np.arange(0,50,2)
    Pc_levels = np.arange(0,1,0.05)
    
    F,x=plt.subplots(4,3,figsize=(25,20),squeeze=False)
    
    #--all
    lon_ind4 = ((lons_era2 > -105) & (lons_era2 <= 30))
    era_T_month_mean_region = np.nanmean(era_T_month_mean[:,:,:,lon_ind4],axis=(3))
    era_rh_month_mean_region = np.nanmean(era_rh_month_mean[:,:,:,lon_ind4],axis=(3))
    era_u_month_mean_region = np.nanmean(era_u_month_mean[:,:,:,lon_ind4],axis=(3))
    era_v_month_mean_region = np.nanmean(era_v_month_mean[:,:,:,lon_ind4],axis=(3))
    era_wspd_month_mean_region = np.nanmean(era_wspd_month_mean[:,:,:,lon_ind4],axis=(3))
    
    #--us
    lon_ind4 = ((lons_era2 > -105) & (lons_era2 <= -65))
    era_T_month_mean_region_us = np.nanmean(era_T_month_mean[:,:,:,lon_ind4],axis=(3))
    era_rh_month_mean_region_us = np.nanmean(era_rh_month_mean[:,:,:,lon_ind4],axis=(3))
    era_u_month_mean_region_us = np.nanmean(era_u_month_mean[:,:,:,lon_ind4],axis=(3))
    era_v_month_mean_region_us = np.nanmean(era_v_month_mean[:,:,:,lon_ind4],axis=(3))
    era_wspd_month_mean_region_us = np.nanmean(era_wspd_month_mean[:,:,:,lon_ind4],axis=(3))
    
    #--atlantic
    lon_ind4 = ((lons_era2 > -65) & (lons_era2 <= -5))
    era_T_month_mean_region_atlantic = np.nanmean(era_T_month_mean[:,:,:,lon_ind4],axis=(3))
    era_rh_month_mean_region_atlantic = np.nanmean(era_rh_month_mean[:,:,:,lon_ind4],axis=(3))
    era_u_month_mean_region_atlantic = np.nanmean(era_u_month_mean[:,:,:,lon_ind4],axis=(3))
    era_v_month_mean_region_atlantic = np.nanmean(era_v_month_mean[:,:,:,lon_ind4],axis=(3))
    era_wspd_month_mean_region_atlantic = np.nanmean(era_wspd_month_mean[:,:,:,lon_ind4],axis=(3))
    
    #--eu
    lon_ind4 = ((lons_era2 > -5) & (lons_era2 <= 30))
    era_T_month_mean_region_eu = np.nanmean(era_T_month_mean[:,:,:,lon_ind4],axis=(3))
    era_rh_month_mean_region_eu = np.nanmean(era_rh_month_mean[:,:,:,lon_ind4],axis=(3))
    era_u_month_mean_region_eu = np.nanmean(era_u_month_mean[:,:,:,lon_ind4],axis=(3))
    era_v_month_mean_region_eu = np.nanmean(era_v_month_mean[:,:,:,lon_ind4],axis=(3))
    era_wspd_month_mean_region_eu = np.nanmean(era_wspd_month_mean[:,:,:,lon_ind4],axis=(3))
    
    x1=x[0,0].plot()
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_T,yyy_all_T = my_histogram(era_T_month_mean_region[np.array([0,1,11]),2,:].flatten(),205,230,1.0)
    xxx_us_T,yyy_us_T = my_histogram(era_T_month_mean_region_us[np.array([0,1,11]),2,:].flatten(),205,230,1.0)
    xxx_atlantic_T,yyy_atlantic_T = my_histogram(era_T_month_mean_region_atlantic[np.array([0,1,11]),2,:].flatten(),205,230,1.0)
    xxx_eu_T,yyy_eu_T = my_histogram(era_T_month_mean_region_eu[np.array([0,1,11]),2,:].flatten(),205,230,1.0)    
    x[0,0].plot(xxx_all_T[:-1], yyy_all_T, alpha=1,label='all',c='k',linewidth=2)
    x[0,0].plot(xxx_us_T[:-1], yyy_us_T, alpha=1,label='us',c='red',linewidth=2)
    x[0,0].plot(xxx_atlantic_T[:-1], yyy_atlantic_T, alpha=1,label='atlantic',c='green',linewidth=2)
    x[0,0].plot(xxx_eu_T[:-1], yyy_eu_T, alpha=1,label='eu',c='blue',linewidth=2)
    x[0,0].set_xlim(205,230)
    x[0,0].set_ylim(0,0.4)
    
    x[0,0].tick_params(labelsize=20)
    x[0,0].xaxis.set_tick_params(width=2,length=5)
    x[0,0].yaxis.set_tick_params(width=2,length=5)
    x[0,0].spines['top'].set_linewidth(1.5)
    x[0,0].spines['left'].set_linewidth(1.5)
    x[0,0].spines['right'].set_linewidth(1.5)
    x[0,0].spines['bottom'].set_linewidth(1.5)
    x[0,0].set_ylabel('PDF [0-1]',fontsize = 20)
    
    x[0,1].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_rh,yyy_all_rh = my_histogram(era_rh_month_mean_region[np.array([0,1,11]),2,:].flatten(),0,110,2.5)
    xxx_us_rh,yyy_us_rh = my_histogram(era_rh_month_mean_region_us[np.array([0,1,11]),2,:].flatten(),0,110,2.5)
    xxx_atlantic_rh,yyy_atlantic_rh = my_histogram(era_rh_month_mean_region_atlantic[np.array([0,1,11]),2,:].flatten(),0,110,2.5)
    xxx_eu_rh,yyy_eu_rh = my_histogram(era_rh_month_mean_region_eu[np.array([0,1,11]),2,:].flatten(),0,110,2.5)    
    x[0,1].plot(xxx_all_rh[:-1], yyy_all_rh, alpha=1,label='all',c='k',linewidth=2)
    x[0,1].plot(xxx_us_rh[:-1], yyy_us_rh, alpha=1,label='us',c='red',linewidth=2)
    x[0,1].plot(xxx_atlantic_rh[:-1], yyy_atlantic_rh, alpha=1,label='atlantic',c='green',linewidth=2)
    x[0,1].plot(xxx_eu_rh[:-1], yyy_eu_rh, alpha=1,label='eu',c='blue',linewidth=2)
    x[0,1].set_xlim(0,110)
    x[0,1].set_ylim(0,0.4)
    
    x[0,1].tick_params(labelsize=20)
    x[0,1].xaxis.set_tick_params(width=2,length=5)
    x[0,1].yaxis.set_tick_params(width=2,length=5)
    x[0,1].spines['top'].set_linewidth(1.5)
    x[0,1].spines['left'].set_linewidth(1.5)
    x[0,1].spines['right'].set_linewidth(1.5)
    x[0,1].spines['bottom'].set_linewidth(1.5)
    
    x[0,2].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']

    xxx_all_wspd,yyy_all_wspd = my_histogram(era_wspd_month_mean_region[np.array([0,1,11]),2,:].flatten(),0,70,1.0)
    xxx_us_wspd,yyy_us_wspd = my_histogram(era_wspd_month_mean_region_us[np.array([0,1,11]),2,:].flatten(),0,70,1.0)
    xxx_atlantic_wspd,yyy_atlantic_wspd = my_histogram(era_wspd_month_mean_region_atlantic[np.array([0,1,11]),2,:].flatten(),0,70,1.0)
    xxx_eu_wspd,yyy_eu_wspd = my_histogram(era_wspd_month_mean_region_eu[np.array([0,1,11]),2,:].flatten(),0,70,1.0)    
    x[0,2].plot(xxx_all_wspd[:-1], yyy_all_wspd, alpha=1,label='all',c='k',linewidth=2)
    x[0,2].plot(xxx_us_wspd[:-1], yyy_us_wspd, alpha=1,label='us',c='red',linewidth=2)
    x[0,2].plot(xxx_atlantic_wspd[:-1], yyy_atlantic_wspd, alpha=1,label='atlantic',c='green',linewidth=2)
    x[0,2].plot(xxx_eu_wspd[:-1], yyy_eu_wspd, alpha=1,label='eu',c='blue',linewidth=2)
    x[0,2].set_xlim(0,60)
    x[0,2].set_ylim(0,0.4)    

    x[0,2].tick_params(labelsize=20)
    x[0,2].xaxis.set_tick_params(width=2,length=5)
    x[0,2].yaxis.set_tick_params(width=2,length=5)
    x[0,2].spines['top'].set_linewidth(1.5)
    x[0,2].spines['left'].set_linewidth(1.5)
    x[0,2].spines['right'].set_linewidth(1.5)
    x[0,2].spines['bottom'].set_linewidth(1.5)
    
    
    x1=x[1,0].plot()
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_T,yyy_all_T = my_histogram(era_T_month_mean_region[np.array([0,1,11]),3,:].flatten(),205,230,1.0)
    xxx_us_T,yyy_us_T = my_histogram(era_T_month_mean_region_us[np.array([0,1,11]),3,:].flatten(),205,230,1.0)
    xxx_atlantic_T,yyy_atlantic_T = my_histogram(era_T_month_mean_region_atlantic[np.array([0,1,11]),3,:].flatten(),205,230,1.0)
    xxx_eu_T,yyy_eu_T = my_histogram(era_T_month_mean_region_eu[np.array([0,1,11]),3,:].flatten(),205,230,1.0)    
    x[1,0].plot(xxx_all_T[:-1], yyy_all_T, alpha=1,label='all',c='k',linewidth=2)
    x[1,0].plot(xxx_us_T[:-1], yyy_us_T, alpha=1,label='us',c='red',linewidth=2)
    x[1,0].plot(xxx_atlantic_T[:-1], yyy_atlantic_T, alpha=1,label='atlantic',c='green',linewidth=2)
    x[1,0].plot(xxx_eu_T[:-1], yyy_eu_T, alpha=1,label='eu',c='blue',linewidth=2)
    x[1,0].set_xlim(205,230)
    x[1,0].set_ylim(0,0.4)
    
    x[1,0].tick_params(labelsize=20)
    x[1,0].xaxis.set_tick_params(width=2,length=5)
    x[1,0].yaxis.set_tick_params(width=2,length=5)
    x[1,0].spines['top'].set_linewidth(1.5)
    x[1,0].spines['left'].set_linewidth(1.5)
    x[1,0].spines['right'].set_linewidth(1.5)
    x[1,0].spines['bottom'].set_linewidth(1.5)
    x[1,0].set_ylabel('PDF [0-1]',fontsize = 20)
    
    x[1,1].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_rh,yyy_all_rh = my_histogram(era_rh_month_mean_region[np.array([0,1,11]),3,:].flatten(),0,110,2.5)
    xxx_us_rh,yyy_us_rh = my_histogram(era_rh_month_mean_region_us[np.array([0,1,11]),3,:].flatten(),0,110,2.5)
    xxx_atlantic_rh,yyy_atlantic_rh = my_histogram(era_rh_month_mean_region_atlantic[np.array([0,1,11]),3,:].flatten(),0,110,2.5)
    xxx_eu_rh,yyy_eu_rh = my_histogram(era_rh_month_mean_region_eu[np.array([0,1,11]),3,:].flatten(),0,110,2.5)    
    x[1,1].plot(xxx_all_rh[:-1], yyy_all_rh, alpha=1,label='all',c='k',linewidth=2)
    x[1,1].plot(xxx_us_rh[:-1], yyy_us_rh, alpha=1,label='us',c='red',linewidth=2)
    x[1,1].plot(xxx_atlantic_rh[:-1], yyy_atlantic_rh, alpha=1,label='atlantic',c='green',linewidth=2)
    x[1,1].plot(xxx_eu_rh[:-1], yyy_eu_rh, alpha=1,label='eu',c='blue',linewidth=2)
    x[1,1].set_xlim(0,110)
    x[1,1].set_ylim(0,0.4)
    
    x[1,1].tick_params(labelsize=20)
    x[1,1].xaxis.set_tick_params(width=2,length=5)
    x[1,1].yaxis.set_tick_params(width=2,length=5)
    x[1,1].spines['top'].set_linewidth(1.5)
    x[1,1].spines['left'].set_linewidth(1.5)
    x[1,1].spines['right'].set_linewidth(1.5)
    x[1,1].spines['bottom'].set_linewidth(1.5)
    
    x[1,2].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']

    xxx_all_wspd,yyy_all_wspd = my_histogram(era_wspd_month_mean_region[np.array([0,1,11]),3,:].flatten(),0,70,1.0)
    xxx_us_wspd,yyy_us_wspd = my_histogram(era_wspd_month_mean_region_us[np.array([0,1,11]),3,:].flatten(),0,70,1.0)
    xxx_atlantic_wspd,yyy_atlantic_wspd = my_histogram(era_wspd_month_mean_region_atlantic[np.array([0,1,11]),3,:].flatten(),0,70,1.0)
    xxx_eu_wspd,yyy_eu_wspd = my_histogram(era_wspd_month_mean_region_eu[np.array([0,1,11]),3,:].flatten(),0,70,1.0)    
    x[1,2].plot(xxx_all_wspd[:-1], yyy_all_wspd, alpha=1,label='all',c='k',linewidth=2)
    x[1,2].plot(xxx_us_wspd[:-1], yyy_us_wspd, alpha=1,label='us',c='red',linewidth=2)
    x[1,2].plot(xxx_atlantic_wspd[:-1], yyy_atlantic_wspd, alpha=1,label='atlantic',c='green',linewidth=2)
    x[1,2].plot(xxx_eu_wspd[:-1], yyy_eu_wspd, alpha=1,label='eu',c='blue',linewidth=2)
    x[1,2].set_xlim(0,60)
    x[1,2].set_ylim(0,0.4)    

    x[1,2].tick_params(labelsize=20)
    x[1,2].xaxis.set_tick_params(width=2,length=5)
    x[1,2].yaxis.set_tick_params(width=2,length=5)
    x[1,2].spines['top'].set_linewidth(1.5)
    x[1,2].spines['left'].set_linewidth(1.5)
    x[1,2].spines['right'].set_linewidth(1.5)
    x[1,2].spines['bottom'].set_linewidth(1.5)
    
    
    
    x1=x[2,0].plot()
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_T,yyy_all_T = my_histogram(era_T_month_mean_region[np.array([0,1,11]),4,:].flatten(),205,230,1.0)
    xxx_us_T,yyy_us_T = my_histogram(era_T_month_mean_region_us[np.array([0,1,11]),4,:].flatten(),205,230,1.0)
    xxx_atlantic_T,yyy_atlantic_T = my_histogram(era_T_month_mean_region_atlantic[np.array([0,1,11]),4,:].flatten(),205,230,1.0)
    xxx_eu_T,yyy_eu_T = my_histogram(era_T_month_mean_region_eu[np.array([0,1,11]),4,:].flatten(),205,230,1.0)    
    x[2,0].plot(xxx_all_T[:-1], yyy_all_T, alpha=1,label='all',c='k',linewidth=2)
    x[2,0].plot(xxx_us_T[:-1], yyy_us_T, alpha=1,label='us',c='red',linewidth=2)
    x[2,0].plot(xxx_atlantic_T[:-1], yyy_atlantic_T, alpha=1,label='atlantic',c='green',linewidth=2)
    x[2,0].plot(xxx_eu_T[:-1], yyy_eu_T, alpha=1,label='eu',c='blue',linewidth=2)
    x[2,0].set_xlim(205,230)
    x[2,0].set_ylim(0,0.4)
    
    x[2,0].tick_params(labelsize=20)
    x[2,0].xaxis.set_tick_params(width=2,length=5)
    x[2,0].yaxis.set_tick_params(width=2,length=5)
    x[2,0].spines['top'].set_linewidth(1.5)
    x[2,0].spines['left'].set_linewidth(1.5)
    x[2,0].spines['right'].set_linewidth(1.5)
    x[2,0].spines['bottom'].set_linewidth(1.5)
    x[2,0].set_ylabel('Pressure [hPa]',fontsize = 20)
    
    x[2,1].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_rh,yyy_all_rh = my_histogram(era_rh_month_mean_region[np.array([0,1,11]),4,:].flatten(),0,110,2.5)
    xxx_us_rh,yyy_us_rh = my_histogram(era_rh_month_mean_region_us[np.array([0,1,11]),4,:].flatten(),0,110,2.5)
    xxx_atlantic_rh,yyy_atlantic_rh = my_histogram(era_rh_month_mean_region_atlantic[np.array([0,1,11]),4,:].flatten(),0,110,2.5)
    xxx_eu_rh,yyy_eu_rh = my_histogram(era_rh_month_mean_region_eu[np.array([0,1,11]),4,:].flatten(),0,110,2.5)    
    x[2,1].plot(xxx_all_rh[:-1], yyy_all_rh, alpha=1,label='all',c='k',linewidth=2)
    x[2,1].plot(xxx_us_rh[:-1], yyy_us_rh, alpha=1,label='us',c='red',linewidth=2)
    x[2,1].plot(xxx_atlantic_rh[:-1], yyy_atlantic_rh, alpha=1,label='atlantic',c='green',linewidth=2)
    x[2,1].plot(xxx_eu_rh[:-1], yyy_eu_rh, alpha=1,label='eu',c='blue',linewidth=2)
    x[2,1].set_xlim(0,110)
    x[2,1].set_ylim(0,0.4)
    
    x[2,1].tick_params(labelsize=20)
    x[2,1].xaxis.set_tick_params(width=2,length=5)
    x[2,1].yaxis.set_tick_params(width=2,length=5)
    x[2,1].spines['top'].set_linewidth(1.5)
    x[2,1].spines['left'].set_linewidth(1.5)
    x[2,1].spines['right'].set_linewidth(1.5)
    x[2,1].spines['bottom'].set_linewidth(1.5)
    
    x[2,2].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']

    xxx_all_wspd,yyy_all_wspd = my_histogram(era_wspd_month_mean_region[np.array([0,1,11]),4,:].flatten(),0,70,1.0)
    xxx_us_wspd,yyy_us_wspd = my_histogram(era_wspd_month_mean_region_us[np.array([0,1,11]),4,:].flatten(),0,70,1.0)
    xxx_atlantic_wspd,yyy_atlantic_wspd = my_histogram(era_wspd_month_mean_region_atlantic[np.array([0,1,11]),4,:].flatten(),0,70,1.0)
    xxx_eu_wspd,yyy_eu_wspd = my_histogram(era_wspd_month_mean_region_eu[np.array([0,1,11]),4,:].flatten(),0,70,1.0)    
    x[2,2].plot(xxx_all_wspd[:-1], yyy_all_wspd, alpha=1,label='all',c='k',linewidth=2)
    x[2,2].plot(xxx_us_wspd[:-1], yyy_us_wspd, alpha=1,label='us',c='red',linewidth=2)
    x[2,2].plot(xxx_atlantic_wspd[:-1], yyy_atlantic_wspd, alpha=1,label='atlantic',c='green',linewidth=2)
    x[2,2].plot(xxx_eu_wspd[:-1], yyy_eu_wspd, alpha=1,label='eu',c='blue',linewidth=2)
    x[2,2].set_xlim(0,70)
    x[2,2].set_ylim(0,0.4)    

    x[2,2].tick_params(labelsize=20)
    x[2,2].xaxis.set_tick_params(width=2,length=5)
    x[2,2].yaxis.set_tick_params(width=2,length=5)
    x[2,2].spines['top'].set_linewidth(1.5)
    x[2,2].spines['left'].set_linewidth(1.5)
    x[2,2].spines['right'].set_linewidth(1.5)
    x[2,2].spines['bottom'].set_linewidth(1.5)
    
    x1=x[3,0].plot()
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_T,yyy_all_T = my_histogram(era_T_month_mean_region[np.array([0,1,11]),5,:].flatten(),205,230,1.0)
    xxx_us_T,yyy_us_T = my_histogram(era_T_month_mean_region_us[np.array([0,1,11]),5,:].flatten(),205,230,1.0)
    xxx_atlantic_T,yyy_atlantic_T = my_histogram(era_T_month_mean_region_atlantic[np.array([0,1,11]),5,:].flatten(),205,230,1.0)
    xxx_eu_T,yyy_eu_T = my_histogram(era_T_month_mean_region_eu[np.array([0,1,11]),5,:].flatten(),205,230,1.0)    
    x[3,0].plot(xxx_all_T[:-1], yyy_all_T, alpha=1,label='all',c='k',linewidth=2)
    x[3,0].plot(xxx_us_T[:-1], yyy_us_T, alpha=1,label='us',c='red',linewidth=2)
    x[3,0].plot(xxx_atlantic_T[:-1], yyy_atlantic_T, alpha=1,label='atlantic',c='green',linewidth=2)
    x[3,0].plot(xxx_eu_T[:-1], yyy_eu_T, alpha=1,label='eu',c='blue',linewidth=2)
    x[3,0].set_xlim(205,230)
    x[3,0].set_ylim(0,0.4)
    
    x[3,0].tick_params(labelsize=20)
    x[3,0].xaxis.set_tick_params(width=2,length=5)
    x[3,0].yaxis.set_tick_params(width=2,length=5)
    x[3,0].spines['top'].set_linewidth(1.5)
    x[3,0].spines['left'].set_linewidth(1.5)
    x[3,0].spines['right'].set_linewidth(1.5)
    x[3,0].spines['bottom'].set_linewidth(1.5)
    x[3,0].set_ylabel('PDF [0-1]',fontsize = 20)
    x[3,0].set_xlabel('Tempetaure [K]',fontsize = 20)
    
    x[3,1].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    
    xxx_all_rh,yyy_all_rh = my_histogram(era_rh_month_mean_region[np.array([0,1,11]),5,:].flatten(),0,110,2.5)
    xxx_us_rh,yyy_us_rh = my_histogram(era_rh_month_mean_region_us[np.array([0,1,11]),5,:].flatten(),0,110,2.5)
    xxx_atlantic_rh,yyy_atlantic_rh = my_histogram(era_rh_month_mean_region_atlantic[np.array([0,1,11]),5,:].flatten(),0,110,2.5)
    xxx_eu_rh,yyy_eu_rh = my_histogram(era_rh_month_mean_region_eu[np.array([0,1,11]),5,:].flatten(),0,110,2.5)    
    x[3,1].plot(xxx_all_rh[:-1], yyy_all_rh, alpha=1,label='all',c='k',linewidth=2)
    x[3,1].plot(xxx_us_rh[:-1], yyy_us_rh, alpha=1,label='us',c='red',linewidth=2)
    x[3,1].plot(xxx_atlantic_rh[:-1], yyy_atlantic_rh, alpha=1,label='atlantic',c='green',linewidth=2)
    x[3,1].plot(xxx_eu_rh[:-1], yyy_eu_rh, alpha=1,label='eu',c='blue',linewidth=2)
    x[3,1].set_xlim(0,110)
    x[3,1].set_ylim(0,0.4)
    
    x[3,1].tick_params(labelsize=20)
    x[3,1].xaxis.set_tick_params(width=2,length=5)
    x[3,1].yaxis.set_tick_params(width=2,length=5)
    x[3,1].spines['top'].set_linewidth(1.5)
    x[3,1].spines['left'].set_linewidth(1.5)
    x[3,1].spines['right'].set_linewidth(1.5)
    x[3,1].spines['bottom'].set_linewidth(1.5)
    x[3,1].set_xlabel('rel hum ice [%]',fontsize = 20)
    
    x[3,2].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']

    xxx_all_wspd,yyy_all_wspd = my_histogram(era_wspd_month_mean_region[np.array([0,1,11]),5,:].flatten(),0,70,1.0)
    xxx_us_wspd,yyy_us_wspd = my_histogram(era_wspd_month_mean_region_us[np.array([0,1,11]),5,:].flatten(),0,70,1.0)
    xxx_atlantic_wspd,yyy_atlantic_wspd = my_histogram(era_wspd_month_mean_region_atlantic[np.array([0,1,11]),5,:].flatten(),0,70,1.0)
    xxx_eu_wspd,yyy_eu_wspd = my_histogram(era_wspd_month_mean_region_eu[np.array([0,1,11]),5,:].flatten(),0,70,1.0)    
    x[3,2].plot(xxx_all_wspd[:-1], yyy_all_wspd, alpha=1,label='all',c='k',linewidth=2)
    x[3,2].plot(xxx_us_wspd[:-1], yyy_us_wspd, alpha=1,label='us',c='red',linewidth=2)
    x[3,2].plot(xxx_atlantic_wspd[:-1], yyy_atlantic_wspd, alpha=1,label='atlantic',c='green',linewidth=2)
    x[3,2].plot(xxx_eu_wspd[:-1], yyy_eu_wspd, alpha=1,label='eu',c='blue',linewidth=2)
    x[3,2].set_xlim(0,70)
    x[3,2].set_ylim(0,0.4)    

    x[3,2].tick_params(labelsize=20)
    x[3,2].xaxis.set_tick_params(width=2,length=5)
    x[3,2].yaxis.set_tick_params(width=2,length=5)
    x[3,2].spines['top'].set_linewidth(1.5)
    x[3,2].spines['left'].set_linewidth(1.5)
    x[3,2].spines['right'].set_linewidth(1.5)
    x[3,2].spines['bottom'].set_linewidth(1.5)
    x[3,2].set_xlabel('Windspeed [m s$^{-1}$]',fontsize = 20)
    
    
    filename = 'pdfs.png'
    if server == 0:
        F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
        F.show()
    if server == 1:
        F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
        plt.close()
        
 %--end of code
