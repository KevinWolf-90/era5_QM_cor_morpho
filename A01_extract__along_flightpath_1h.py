#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:39:57 2022

@author: kwolf
"""

#Programto read IAGOS data
#--since 2023 02 03 this version reads the latest IAGOS flight
#--bases on a routine from Olivier Boucher

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
import netCDF4 as nc
import copy
import sys
import os
from pathlib import Path
import pandas as pd
from itertools import chain
import xarray as xr
from datetime import datetime,timedelta
from calendar import monthrange
import warnings
import glob

#--import gaussian filter to smooth the iagos observations
from scipy.ndimage import gaussian_filter

import os.path
from matplotlib.colors import LogNorm

#--select server or local computation
#server = 0 # 0 if local, 1 if on server
server = 1


#--load my exrenal routines
if server == 0:
    sys.path.insert(1, '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines')
if server == 1:
    sys.path.insert(1, '/homedata/kwolf/40_era_iagos/00_code')

from rh_ice_to_rh_liquid_era import rh_ice_to_rh_liquid_era
from rh_liquid_to_rh_ice_ecmwf import rh_liquid_to_rh_ice_ecmwf

#--for histograms
def my_histogram(value,xmin,xmax,step):
        #used to calculate my own histograms
        dummy = copy.deepcopy(value)
        dummy = dummy.reshape(dummy[:].size)
        yyy, xxx = np.histogram(dummy[:], bins=np.linspace(xmin, xmax, int((xmax-xmin)/step)))
        y_total = np.nansum(yyy)
        yyy = np.divide(yyy,y_total)
        return(xxx,yyy)

#--find nearest item in items to pivot
def nearest(items, pivot):
   minitems=min(items, key=lambda x: abs(x - pivot))
   return minitems, items.index(minitems)

#--find nearest hour to datetime
def hour_rounder(t):
   # Rounds to nearest hour by adding a timedelta hour if minute >= 30
   return t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)


#--disable warning. keep output clean
#warnings.filterwarnings("ignore")
print('#################################')
print('Filter warnings are switched off!')
print('#################################')
time.sleep(1)


#--no plotting when run on server
if server == 1:  # to not use Xwindow
    matplotlib.use('Agg')

#--define the years to extract
years = np.arange(2010,2022,1)
#years = np.arange(2015,2017,1)
#years = np.asarray([2018])
months = np.arange(1,13,1)
#months = np.arange(1,3,1)


for year in years:

    #--print a diagfile for each year
    diagf = open('/homedata/kwolf/40_era_iagos/'+str(f'{year:04.0f}')+'_diag_extraction_1h.log','w')
    diagf.write('Diagnose file for extracting ERA5 data along flight track \n')
    diagf.write('Year: '+str(f'{year:04.0f}'+'\n'))

    out_arr = np.zeros((25,0))  #for iagos 
    out_arr_smooth = np.zeros((25,0))  #for iagos smoothed
    t_out = np.zeros((0))
    r_out = np.zeros((0))
    u_out = np.zeros((0))
    v_out = np.zeros((0))
    w_out = np.zeros((0))
    pres_out = np.zeros((0))

    for month in months:
        #--get the number of days in the month
        ndays = monthrange(year,month)[1]
        print('Processing Year: ',str(f'{year:04.0f}'))    
        print('Processing Month: ',str(f'{month:02.0f}'))    
        path_iagos='/bdd/IAGOS/'
        files_iagos_all=glob.glob(path_iagos+str(year)+str(month).zfill(2)+'/*.nc4')
        #--clean datasets by removing duplicates with different versions
        files_iagos_seeds=sorted(list(set([file[0:55] for file in files_iagos_all])))
        #--create new list
        files_iagos=[]
        for seed in files_iagos_seeds:
            #--we only keep the latest version of each file if duplicates
            file_iagos=[file for file in files_iagos_all if file[0:55]==seed][-1]
            #--append to list
            files_iagos.append(file_iagos)
        print('We found ',len(files_iagos),' independent files for ',year,' and month ',month)
        
        #--sort files in order otherwise it is arbitary
        files = sorted(files_iagos)
        files_cleared=[]
        #--remove files from the last day of the month
        
        for file in files:
             if file[41:43]<str(ndays):
                  files_cleared.append(file)
        #--variable handover
        files = files_cleared
        
        #-diagnose file
        diagf.write('Number of files in month: '+str(f'{month:02.0f}')+': '+str(len(files))+'\n')

        print(files_cleared)

        for i in files:
        #for i in files[0:20]:
            print('Read file: ',str(i))
            f = nc.Dataset(str(i), 'r')
            #extract data from opend file
            #get keys in dictionary
            variable_keys = f.variables.keys()
            
            #time = np.array(f['time'])
            time_dummy = f['UTC_time']  # read the time; time in SOD
            #time = netCDF4.num2date(time_dummy, time_dummy.units, time_dummy.calendar) # concert the calender  (time from 1800 ) to date
            lon_iagos = f['lon']
            lat_iagos = f['lat']
            air_press_AC = f['air_press_AC']
            
            time_dummy = np.asarray(time_dummy)
            lon_iagos = np.asarray(lon_iagos)
            lat_iagos = np.asarray(lat_iagos)
            
            air_press_AC = np.asarray(air_press_AC)
            
            if 'gps_alt_AC' in variable_keys:
                gps_alt = f['gps_alt_AC']
                gps_alt = np.asarray(gps_alt)
                print('Use GPS alt')
            else:
                gps_alt = f['baro_alt_AC']
                gps_alt = np.asarray(gps_alt)
                print('Use baro alt')
            
            
            if 'air_temp_P1' in variable_keys:
                air_temp_P1 = f['air_temp_P1']
                air_temp_P1_flag = f['air_temp_P1_validity_flag']
                air_temp_P1_error = f['air_temp_P1_error']
                air_temp_P1 = np.asarray(air_temp_P1)
                air_temp_P1_flag = np.asarray(air_temp_P1_flag)
                air_temp_P1_error = np.asarray(air_temp_P1_error)
                
            if 'air_temp_AC' in variable_keys:
                air_temp_AC = f['air_temp_AC']    
                air_temp_AC = np.asarray(air_temp_AC)     #and convert from deg C in K  
                air_temp_AC_flag = f['air_temp_AC_validity_flag']    
                air_temp_AC_flag = np.asarray(air_temp_AC_flag)     #and convert from deg C in K  
        
            if 'cloud_P1' in variable_keys:
                cloud_P1 = f['cloud_P1']  # number concentration of liquid water cloud particles cm-3
                cloud_P1_flag = f['cloud_P1_validity_flag']
                cloud_P1 = np.asarray(cloud_P1)
                cloud_P1_flag = np.asarray(cloud_P1_flag)
        
            if 'RHL_P1' in variable_keys:
                RHL_P1 = f['RHL_P1'] # rel hum over liquid water
                RHL_P1_flag = f['RHL_P1_validity_flag']
                RHL_P1 = np.asarray(RHL_P1)
                RHL_P1_flag = np.asarray(RHL_P1_flag)
        
            if 'RHI_P1' in variable_keys:
                RHI_P1 = f['RHI_P1'] # rel hum over liquid water
                RHI_P1_flag = f['RHI_P1_validity_flag']
                RHI_P1_error = f['RHI_P1_error']
                RHI_P1 = np.asarray(RHI_P1)
                RHI_P1_flag = np.asarray(RHI_P1_flag)
                RHI_P1_error = np.asarray(RHI_P1_error)
        
            if 'wind_dir_AC' in variable_keys:
                wind_dir = f['wind_dir_AC'] # winddirection from aircraft
                wind_dir = np.asarray(wind_dir)
        
            if 'wind_speed_AC' in variable_keys:
                wind_spd = f['wind_speed_AC'] # windspeed from aircraft
                wind_spd = np.asarray(wind_spd)
        
            if 'arrival_coord' in variable_keys:
                ggggg = f['arrival_coord'] # rel hum over liquid water
                
            par_array=np.zeros((out_arr.shape[0],len(time_dummy))) # for lon lat plus params
            
            par_array[0,:] = lon_iagos #for min max finding
            par_array[1,:] = lat_iagos
            par_array[2,:] = time_dummy    # units in seconds
            par_array[3,:] = gps_alt
            par_array[4,:] = air_press_AC
            par_array[5,:] = air_temp_AC
            par_array[6,:] = air_temp_AC_flag
            
            if 'air_temp_P1' in variable_keys:    
                par_array[7,:] = air_temp_P1   # name only in 2016 file
                par_array[8,:] = air_temp_P1_flag   # name only in 2016 file
                par_array[21,:] = air_temp_P1_error   # name only in 2016 file
                print('There is P1 temp')
            else:
                par_array[7,:] = np.nan
                par_array[8,:] = np.nan
                par_array[21,:] = np.nan
                print('No P1 temp. Replace by nan')
            
            if 'cloud_P1' in variable_keys:    
                par_array[9,:] = cloud_P1   # name only in 2016 file
                par_array[10,:] = cloud_P1_flag   # name only in 2016 file
                print('There is P1 cloud')
            else:
                par_array[9,:] = np.nan
                par_array[10,:] = np.nan
                print('No P1 cloud. Replace by nan')
           
            if 'RHL_P1' in variable_keys:    
                par_array[11,:] = RHL_P1   # name only in 2016 file
                par_array[12,:] = RHL_P1_flag   # name only in 2016 file
                print('There is P1 rhl')
            else:
                par_array[11,:] = np.nan
                par_array[12,:] = np.nan
                print('No P1 rhl. Replace by nan')
            
            if 'RHI_P1' in variable_keys:    
                par_array[13,:] = RHI_P1   # name only in 2016 file
                par_array[14,:] = RHI_P1_flag   # name only in 2016 file
                par_array[22,:] = RHI_P1_error   # name only in 2016 file
                print('There is P1 rhi')
            else:
                par_array[13,:] = np.nan
                par_array[14,:] = np.nan
                par_array[22,:] = np.nan
                print('No P1 rhi. Replace by nan')
               
            if np.any(par_array[14,:] > 2):
                diagf.write('This file has a ic flag larger than 2\n')
                diagf.write(str(i)+'\n')
                diagf.write('')

 
            if 'wind_dir_AC' in variable_keys:    
                par_array[19,:] = wind_dir   # name only in 2016 file
                print('There is ac wind dir')
            else:
                par_array[19,:] = np.nan
                print('No wind dir. Replace by nan')
                
            if 'wind_speed_AC' in variable_keys:    
                par_array[20,:] = wind_spd   # name only in 2016 file
                print('There is ac wind spd')
            else:
                par_array[20,:] = np.nan
                print('No wspd. Replace by nan')
        
            par_array[15,:] = int(str(i)[-29:-25])  # year
            par_array[16,:] = int(str(i)[-25:-23]) # month
            par_array[17,:] = int(str(i)[-23:-21]) # day
            par_array[18,:] = par_array[17,:]*24*3600 +par_array[2,:]
           
            par_array[23,:] = int(str(i)[-29:-13]) # full flight number
            par_array[24,:] = int(str(i)[-9])+int(str(i)[-7])+int(str(i)[-5]) # full flight number
            
            #--limiting to pressure and 30N to 70N
            #--sample only above 300 hpa and between 30 and 70 N, neglect last days
            index = ((par_array[4,:] >= 10000) & (par_array[4,:] < 30000) & (par_array[1,:] > 30.5) & (par_array[1,:] < 69.5))
            par_array = par_array[:,index] 
            print('------------------------')
            print(par_array.shape)
            
            if par_array.shape[1] !=0:
                
                print('#--input files')
                file_iagos = i 
                xr_iagos=xr.open_dataset(file_iagos)
                
                print('read iagos')
                #--read iagos
                dep_airport_iagos=xr_iagos.departure_airport.split(',')[0]
                arr_airport_iagos=xr_iagos.arrival_airport.split(',')[0]
                dep_time_iagos=datetime.strptime(xr_iagos.departure_UTC_time,"%Y-%m-%dT%H:%M:%SZ")
                arr_time_iagos=datetime.strptime(xr_iagos.arrival_UTC_time,"%Y-%m-%dT%H:%M:%SZ")
                flightid_iagos=xr_iagos.platform.split(',')[3].lstrip().rstrip()
                lats_iagos=xr_iagos['lat']
                lons_iagos=xr_iagos['lon']
                times_iagos=xr_iagos['UTC_time']
                pressures_iagos=xr_iagos['air_press_AC']
     
                indices=np.where((pressures_iagos.values>=10000.) & (pressures_iagos.values<30000.) & (lats_iagos.values>30.5) & (lats_iagos.values<69.5))
                lats_iagos=lats_iagos.values[indices]
                lons_iagos=lons_iagos.values[indices]
                times_iagos=times_iagos.values[indices]
                pressures_iagos=pressures_iagos.values[indices]/100. #--Pa => hPa
                #
                #--sample 1 in nbskip points
                #print('#--sample 1 in nbskip points')
                nbskip=1
                lats_iagos=lats_iagos[::nbskip]
                lons_iagos=lons_iagos[::nbskip]
                times_iagos=times_iagos[::nbskip]
                pressures_iagos=pressures_iagos[::nbskip]
                print('we select',len(lats_iagos),'points on the IAGOS trajectory above 300 hPa')
                #
                xr_iagos.close() 
                
                #file_era='/projsu/cmip-work/oboucher/ERA5/u.2021.GLOBAL.nc'
                file_era='/scratchx/kwolf/ERA5/'+str(f'{par_array[15,0]:04.0f}')+'/'+str(f'{par_array[15,0]:04.0f}')+'_'+str(f'{par_array[16,0]:02.0f}')+'_era5_1hour_t_r_u_v_180W_180E_30N_70N.grib'
                xr_era=xr.open_dataset(file_era)
                #levels_era=list(xr_era['level'].values)
                levels_era=list(xr_era['isobaricInhPa'].values) # in my files it is called different
                lons_era=xr_era.longitude.values
                #print('longitudes ERA=',np.min(lons_era),np.max(lons_era))
                #
              
                print('#--find closest pressure levels')
                levels_iagos_closest=[]
                for pressure in pressures_iagos:
                  pressure_rounded,pressure_index=nearest(levels_era,pressure)
                  levels_iagos_closest.append(pressure_rounded)
                levels_iagos_unique=sorted(list(set(levels_iagos_closest)))
                levels_iagos_closest=np.array(levels_iagos_closest)
                #print('levels=',levels_iagos_unique)
                #
                print('#--find closest times')
                times_iagos_closest=[]
                for time_iagos in times_iagos:
                  times_iagos_closest.append(hour_rounder(pd.to_datetime(time_iagos)))
                times_iagos_unique=sorted(list(set(times_iagos_closest)))
                times_iagos_closest=np.array(times_iagos_closest)
                #print('times=',times_iagos_unique)
                #
                imax=1
                print('we do',imax,'retrievals for the same selection of levels and times')

                start_time = time.time()
                xr_era_loaded=xr_era.sel(time=times_iagos_unique,isobaricInhPa=levels_iagos_unique,method='nearest').load()
                end_time = time.time()
                print('era loaded',end_time-start_time)
                #
                #--method 1
                start_time = time.time()
                for i in range(0,imax):
                  lons_z = xr.DataArray(lons_iagos, dims="z")
                  lats_z = xr.DataArray(lats_iagos, dims="z")
                  times_z = xr.DataArray(times_iagos_closest, dims="z")
                  levels_z = xr.DataArray(levels_iagos_closest, dims="z")
                  
                  #--extracting the individual variables along the track
                  t_era = xr_era_loaded.sel(latitude=lats_z,longitude=lons_z,time=times_z,isobaricInhPa=levels_z,method='nearest')['t'].values
                  r_era = xr_era_loaded.sel(latitude=lats_z,longitude=lons_z,time=times_z,isobaricInhPa=levels_z,method='nearest')['r'].values
                  u_era = xr_era_loaded.sel(latitude=lats_z,longitude=lons_z,time=times_z,isobaricInhPa=levels_z,method='nearest')['u'].values
                  v_era = xr_era_loaded.sel(latitude=lats_z,longitude=lons_z,time=times_z,isobaricInhPa=levels_z,method='nearest')['v'].values
                  #w_era = xr_era_loaded.sel(latitude=lats_z,longitude=lons_z,time=times_z,isobaricInhPa=levels_z,method='nearest')['w'].values
                  pres_era = levels_z.values
                
                xr_era_loaded.close()
                end_time = time.time()
                print('sel - loaded - sel')
                print('method 1=',end_time-start_time)
                
                par_array_smooth = np.zeros((par_array.shape))

                for co in np.asarray([5,7,9,11,13]): #--loop over each parameter
                    par_array_smooth[co,:] = gaussian_filter(par_array[co,:], 5.8, mode='nearest')  # calculation see green notebook page -57-

                #--handover of par_array that contains the iagos data
                out_arr = np.append(out_arr,par_array,axis=(1))
                out_arr_smooth = np.append(out_arr_smooth,par_array_smooth,axis=(1))
                pres_out =  np.append(pres_out,pres_era,axis=(0))
                t_out =  np.append(t_out,t_era,axis=(0))
                r_out =  np.append(r_out,r_era,axis=(0))
                u_out =  np.append(u_out,u_era,axis=(0))
                v_out =  np.append(v_out,v_era,axis=(0))

    print('Shape of out_arr before any filtering: ', out_arr.shape)
    print('Shape of t_out before any filtering: ', t_out.shape)

    
    #print('----------------')
    len_org = out_arr.shape[1]
    #-diagnose file
    diagf.write('Samples before filtering: '+str(len_org)+'\n')
 
    pres = np.asarray(levels_era)
    #pres = np.asarray(['100','125','150','200','225','250','300','350','400'])
    
    #print('#--remove temperatures flagged as 9999')
    foo = (out_arr[7,:] > -9500) &  (out_arr[7,:] < 9500)
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--remove temperatures flagged as 9999\n')
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n')


    #print('#--filter for the iagos flags of tempature')
    #--only keep, where temp flag is good (0) or at least limited (2)
    foo  = ((out_arr[8,:] > -1) & (out_arr[8,:] <= 2))# | (out_arr[8,:] == 2))
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--filter for the iagos flags of tempature\n')
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n') 

    #print('#--only keep, where hum flag is good(0) or at least limited (2)')
    foo  = ((out_arr[14,:] > -1) & (out_arr[14,:] <= 2))
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--only keep, where flag is good)\n') 
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n') 
   
    #--correct for doubling kelvin conversion
    foo = out_arr[5,:] > 450
    out_arr[5,foo] = out_arr[5,foo] - 273.15
    out_arr_smooth[5,foo] = out_arr_smooth[5,foo] - 273.15
    foo = out_arr[7,:] > 450
    out_arr[7,foo] = out_arr[7,foo] - 273.15
    out_arr_smooth[7,foo] = out_arr_smooth[7,foo] - 273.15
    #correct for missing kelvin conversion
    foo = out_arr[5,:] < 100
    out_arr[5,foo] = out_arr[5,foo] + 273.15
    out_arr_smooth[5,foo] = out_arr_smooth[5,foo] + 273.15
    foo = out_arr[7,:] < 100
    out_arr[7,foo] = out_arr[7,foo] + 273.15
    out_arr_smooth[7,foo] = out_arr_smooth[7,foo] + 273.15
    
    print('#--keep where abs diff between P1 and AC is smaller than 5 degree')
    foo = np.abs(out_arr[5,:] - out_arr[7,:]) < 5
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--keep where abs diff between P1 and AC is smaller than 5 degree\n') 
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n') 
    
    #print('#remove p1 hum flagged as 9999')
    foo = (out_arr[11,:] > -9500) & (out_arr[11,:] < 9500)
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--remove p1 hum flagged as 9999\n') 
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n')
    
    #print('#--keep where rhice smaller than 170%')
    foo =  ((out_arr[11,:] >= 0) & (out_arr[11,:] < 170))
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--keep where rhlarger larger than 0 and smaller than 100%\n')
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n')
    
    #print('#--keep where rhice smaller than 170%')
    foo =  ((out_arr[13,:] >= 0) & (out_arr[13,:] < 170))
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--keep where rhice larger than 0 and smaller than 170%\n')
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n')

    #print('#--keep only positive values rh liquid measurements')
    foo = (out_arr[11,:] > 0)
    out_arr = out_arr[:,foo]
    out_arr_smooth = out_arr_smooth[:,foo]
    t_out = t_out[foo]
    r_out = r_out[foo]
    u_out = u_out[foo]
    v_out = v_out[foo]
    pres_out = pres_out[foo]
    diagf.write('#--keep only positive values rh liquid measurements\n')
    diagf.write('Remaining: '+str(out_arr.shape[1])+' of '+str(len_org)+' ('+str(f'{out_arr.shape[1]/len_org*100:6.2f}')+'%)\n')
   
    #convert rh to %
    out_arr[11,:] = out_arr[11,:]*100
    out_arr[13,:] = out_arr[13,:]*100
    out_arr_smooth[11,:] = out_arr_smooth[11,:]*100
    out_arr_smooth[13,:] = out_arr_smooth[13,:]*100
    
    #convert rh to over liquid and to
    r_out = rh_ice_to_rh_liquid_era(r_out/100,t_out)*100
    r_out_ice = rh_liquid_to_rh_ice_ecmwf(r_out[:]/100,t_out[:])*100
    
    #calculate windspeed and winddirection
    wspd_out = np.sqrt(u_out**2 + v_out**2)
    wndr_out = np.mod(180+np.rad2deg(np.arctan2(u_out, v_out)),360)
    
    del out_arr_smooth
    out_arr_smooth = copy.deepcopy(out_arr)

    #--smooth at the end after the filter
    for co in np.asarray([5,7,9,11,13]): #--loop over each parameter
        out_arr_smooth[co,:] = gaussian_filter(out_arr[co,:], 3, mode='nearest')  # calculation see manuskript
   
    print('#################################')
    print('File safed to:')
    filename='along_track_data_'+str(f'{year:04.0f}')+'_era_1h'
    if server == 0:
        save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename+'.npz'
        print(save_stats_file)
        np.savez(save_stats_file,out_arr,pres,pres_out,t_out,r_out,r_out_ice,wspd_out,wndr_out)
    if server == 1:
        save_stats_file = '/homedata/kwolf/40_era_iagos/'+filename+'.npz'
        print(save_stats_file)
        np.savez(save_stats_file,out_arr,pres,pres_out,t_out,r_out,r_out_ice,wspd_out,wndr_out)
        
    diagf.write('File safed to: '+save_stats_file+'/'+filename+'npz\n')
    print('#################################')
    
    print('#################################')
    print('File safed to:')
    filename='along_track_data_'+str(f'{year:04.0f}')+'_era_1h_smooth'
    if server == 0:
        save_stats_file = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename+'.npz'
        print(save_stats_file)
        np.savez(save_stats_file,out_arr_smooth,pres,pres_out,t_out,r_out,r_out_ice,wspd_out,wndr_out)
    if server == 1:
        save_stats_file = '/homedata/kwolf/40_era_iagos/'+filename+'.npz'
        print(save_stats_file)
        np.savez(save_stats_file,out_arr_smooth,pres,pres_out,t_out,r_out,r_out_ice,wspd_out,wndr_out)
        
    diagf.write('File safed to: '+save_stats_file+'/'+filename+'npz\n')
    print('#################################')
    
    diagf.close()
print('Done.')

