#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:57:04 2022

@author: kwolf
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import copy
import datetime
from matplotlib.colors import LogNorm

import secrets  #--gernerate 'better distributed' random numbers

import netCDF4 as nc
from pathlib import Path
import time
from itertools import groupby
#--statistical imports
from scipy import stats, ndimage
from skimage.measure import label, regionprops  # --for jakes routines
import sys
import os
import xarray as xr
import os.path
import warnings


def my_histogram(value, xmin, xmax, step):
    # used to calculate my own histograms
    dummy = copy.deepcopy(value)
    dummy = dummy.reshape(dummy[:].size)
    yyy, xxx = np.histogram(dummy[:], bins=np.linspace(
        xmin, xmax, int((xmax-xmin)/step)))
    y_total = np.nansum(yyy)
    yyy = np.divide(yyy, y_total)

    return(xxx, yyy)

def cum_sum(values_in):
    #--incoming data sorted
    data_cum_sorted = np.sort(values_in)
    # calculate the proportional values of samples
    p = np.linspace(0, 1, len(data_cum_sorted), endpoint=False)
    return p,data_cum_sorted

# disable warning. keep output clean
warnings.filterwarnings("ignore")
print('#################################')
print('Filter warnings are switched off!')
print('#################################')
time.sleep(1)



#server = 0  # 0 if local, 1 if on server
server = 1

#--calc and safe stat files or plot
safe_stats = 0 #--ploting of  section 1
#safe_stats = 1 #--calc and safe stats




if server == 1:  # to not use Xwindow
    if any('SPYDER' in name for name in os.environ):
        print('Activated plotting on screen')
    else:
        print('Deactivated plotting on screen for terminal and batch')
        matplotlib.use('Agg')


#--import my routines
if server == 0:
    sys.path.insert(
        1, '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines')
if server == 1:
    sys.path.insert(1, '/homedata/kwolf/40_era_iagos/00_code')




#--import external routines for rh and Schmidt-Appleman-criterion
from CritTemp_rasp import CritTemp_rasp
from rh_liquid_to_rh_ice_ecmwf import rh_liquid_to_rh_ice_ecmwf
from rh_ice_to_rh_liquid_era import rh_ice_to_rh_liquid_era



if safe_stats == 1:
    

    #--write diagnose ouput file
    filename = 'diagnose_A06_individual_blobs_processing.txt'
    
    if server == 0:
        filename_diag = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename
    if server == 1:
        filename_diag = '/homedata/kwolf/41_era_statistics/'+filename
    
    
    
    outfile2 = open(filename_diag ,'w')
    outfile2.write('Diagnose \n')
    outfile2.write('========\n')
    
    
    
   
    
    processYear = [2015,2016,2017,2018,2019,2020,2021]
    processMonth = list(np.arange(1,13))
    
    #--number of years to processi
    nYears = len(processYear)
    nMonths = 12 #len(processMonth)
    
    outfile2.write('Processing years: '+str(processYear)+'\n')
    outfile2.write('Processing months: '+str(processMonth)+'\n')
    
    
    geoBoundaries = np.asarray([-110,30,30,70]) #--lon min, lon max, lat min, lat max
    print('Selected boundaries in normal space: Min Lon, Max Lon, Min Lat, Max Lat: ',geoBoundaries)
    outfile2.write('Selected boundaries in normal space: Min Lon, Max Lon, Min Lat, Max Lat: '+str(geoBoundaries)+'\n')
    
    #--fuel and model properties
    Q = 43e6 #--specific combustion energy; values are for JetA1
    EI = 1.25
    eta = 0.35
    rhi_crit = 0.95
    

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
            #--get the number of lons and lats
            nLats = len(lats_era2)
            nLons = len(lons_era2)
            nLevels = len(levels_era2)
    
            #--close dataset to free memory
            xr_era2.close()
    
    
    #--main region
    PC_overlap_monthDummy = np.zeros((nYears,nMonths,nLevels))
    PC_frac_monthDummy = np.zeros((nYears,nMonths,nLevels))
    #--sub-regions region
    PC_overlap_monthDummy_US = np.zeros((nYears,nMonths,nLevels))
    PC_overlap_monthDummy_AT = np.zeros((nYears,nMonths,nLevels))
    PC_overlap_monthDummy_EU = np.zeros((nYears,nMonths,nLevels))
    
    PC_frac_monthDummy_US = np.zeros((nYears,nMonths,nLevels))
    PC_frac_monthDummy_AT = np.zeros((nYears,nMonths,nLevels))
    PC_frac_monthDummy_EU = np.zeros((nYears,nMonths,nLevels))
    
    #--arrays to store the properties of the individual feature
    #--values stored from the ndimage function
    dlon = np.zeros((0))
    dlat = np.zeros((0))
    dpl = np.zeros((0))
    pstart = np.zeros((0))
    pstop = np.zeros((0))
    area = np.zeros((0))
    aspect = np.zeros((0)) #--aspect ratio of cloud from regionprops
    orientation = np.zeros((0)) #--orientation of the cloud
    major_ax_len = np.zeros((0)) #--safe the length of the major axis
    #--for each extracted feature, i do store the month and pressure level
    #--this allows to separate later on
    month_ind = np.zeros((0))
    pres_ind = np.zeros((0))
    
    edge_flag = np.zeros((0))#--mark the ones that hit the boundary of the 
    
    #--arrays to store the temporal decorrelation
    #--how often PC appear in time for one single pixel
    con_t_pc_300 = np.zeros((0,2)) #300 hPa level # [consecutive t, month of year]
    con_t_pc_250 = np.zeros((0,2)) #250 hPa level
    con_t_pc_225 = np.zeros((0,2)) #250 hPa level
    con_t_pc_200 = np.zeros((0,2)) #200 hPa level
    con_t_pc_175 = np.zeros((0,2)) #175 hPa level
    
    #--counter for the total number of processed clouds over all year, months, and pressure levels
    total_cloud_counter = 0 #-- total of pieces that were detected in the first place
    cloud_v1_cnt = 0 #--count the number of clouds that pass the larger than one pixel count
    cloud_v2_cnt = 0 #--number of clouds that are large eneought and are not at the boundary
    cloud_v3_cnt = 0 #--number of clouds that finally passed the processing and are included in the caluclations
    
    
    print('Going into the month and year loop')
    outfile2.write('Going into the month and year loop\n')
    outfile2.write('\n')
    
    #   #%%
    
    for yearCounter in np.arange(0,len(processYear)):
        for monthCounter in np.arange(0,len(processMonth)):
        
            if server == 0:
                file_era2 = '/home/kwolf/Documents/00_CLIMAVIATION/03_ERA5_netcdf_025/'+str(f'{processYear[yearCounter]:04.0f}')+ \
                    '_'+str(f'{processMonth[monthCounter]:02.0f}')+'_era5_1hour_t_r_u_v_180W_180E_30N_70N.grib'
            if server == 1:
                file_era2 = '/scratchx/kwolf/ERA5/'+str(f'{processYear[yearCounter]:04.0f}')+'/'+str(f'{processYear[yearCounter]:04.0f}')+ \
                    '_'+str(f'{processMonth[monthCounter]:02.0f}')+'_era5_1hour_t_r_u_v_180W_180E_30N_70N.grib'
                
            print('Read: ', file_era2)
            xr_era2 = xr.open_dataset(file_era2)
            outfile2.write('\n')
            outfile2.write('Reading: '+file_era2+'\n')
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
            
            #--generate 10*4 random numers
            loops = 0
            n = []
            while (len(n) < 10):   #--draw 12 random days and 4 steps per day
                new_add = secrets.choice(np.arange(1,28)) #--only use the first 28 days; no issues with non-februarys
                if new_add not in n:  #--only unique days
                    #print('append')
                    n.append(new_add)
                loops +=1
            n.sort()  #--sort in accending order
            n = np.asarray(n)  #-final indices to use to use
            n  = n-1 #--first of month with index 0
            #--for each n day get the 0, 6,12, 18 hour step
            new_n = []
            for i in n:
                new_n.append(i*24+0)
                new_n.append(i*24+6)
                new_n.append(i*24+12)
                new_n.append(i*24+18)
              
            n = new_n
            times_era2 = times_era2[n]
            
            print('Using only: '+str(len(times_era2))+' timesteps')
            outfile2.write('Using only: '+str(len(times_era2))+' timesteps\n')
            
    
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
            era_rLiquid = rh_ice_to_rh_liquid_era(era_r/100.,era_t)
            #-- read windspeed
            era_u = np.asarray(era_u)
            era_v = np.asarray(era_v)
            #--calculate windspeed
            era_wspd = np.sqrt(era_u**2 + era_v**2)
            
            #--close file and free memonry
            xr_era2.close()
       
            
            
            #--array to store flag for ISSR; store where rhive >100%
            ISSR_flag = np.zeros((len(times_era2), nLevels, nLats, nLons ))
            #--safe where SAc is fullfilled
            SAc_flag = np.zeros((len(times_era2), nLevels, nLats, nLons ))
            #--array to store the PC contrail flaggs size: time, levels,lon,lat
            PC_flag = np.zeros((len(times_era2), nLevels, nLats, nLons ))
            #--expand the pressure column to all dimensions
            levels_era2Expanded = np.zeros((era_t.shape[1],era_t.shape[2],era_t.shape[3]))
            levels_era2Array = np.asarray(levels_era2) #--conversion from list to array required
            levels_era2Expanded[:,:,:] = levels_era2Array[:,None,None]           
            crit_temp_profile = CritTemp_rasp(era_t,levels_era2Expanded*100.,era_rLiquid,eta,Q,Ein=EI) # temperature in k, pressure in pa, and rel hum in 0-1
            crit_temp_profile = crit_temp_profile[:,:,:,:,0,:].T
    
            #--SAc AND ISSR
            pot_layers_index1 = np.where((era_t[:,:,:,:] < crit_temp_profile[0,:,:,:,:]) & (era_rLiquid[:,:,:,:] > crit_temp_profile[1,:,:,:,:])  & (crit_temp_profile[2,:,:,:,:] >= rhi_crit) & (era_t[:,:,:,:] <= (-38+273.15)))  #diagramgroup 1
            PC_flag[pot_layers_index1] = 1
    
            #-- only SAc
            pot_layers_index2 = np.where((era_t[:,:,:,:] < crit_temp_profile[0,:,:,:,:]) & (era_rLiquid[:,:,:,:] > crit_temp_profile[1,:,:,:,:])  & (era_t[:,:,:,:] <= (-38+273.15)))
            SAc_flag[pot_layers_index2] = 1
    
            #--ISSR flag
            pot_layers_index3 = np.where((crit_temp_profile[2,:,:,:,:] >= rhi_crit)) 
            ISSR_flag[pot_layers_index3] = 1
         
            start_time_tloop = datetime.datetime.now()
            for tx in np.arange(0,era_t.shape[0]-1):   #--loop over time
                for px in np.arange(2,5): #--loop over pressure levels 250, 225, and 200 hPa, these are of interest
                    start_time = datetime.datetime.now()
                    
                    #--handover of input variable
                    a = PC_flag[tx,px,:,:]
                    #--generate a structure that represents the connectivity between pixels
                    str_3d = ndimage.generate_binary_structure(2,1) #- switched to 2d structure to work on each p level separately
                    labels, numobjects = ndimage.label(a,structure=str_3d)
                    print('Labels: ',labels.shape)
                    
                    
                    total_cloud_counter = total_cloud_counter + numobjects #--add the number of identified clouds to the total sum
                    
                    #--Now find their bounding boxes (This will be a tuple of slice objects)
                    #--You can use each one to directly index your data. 
                    #--E.g. a[slices[0]] gives you the original data within the bounding box of the
                    #--first object.
                    print('printing labels: ',labels)
                    slices = ndimage.find_objects(labels)
                    
                    print('Number of individual objects:')
                    print(numobjects)
                    print('')
                               
                    #--loop over features
                    #--get the size for each unique feature
                    clouds = regionprops(labels)
                    
                    for n in np.arange(0,numobjects):
                        print('n: ',n)
                        lat_start_index = slices[n][0].start
                        lat_stop_index = slices[n][0].stop
                        lon_start_index = slices[n][1].start
                        lon_stop_index = slices[n][1].stop
                        

                        #--only objects larger than a single gridbox
                        if ((np.abs(lat_start_index - lat_stop_index) > 1) & (np.abs(lon_start_index - lon_stop_index) > 1)):
                            cloud_v1_cnt +=1
                        
                            dlon_km = 0 #--have to provide at least 0 value otherwise the code below is not working
                            dlat_km = 0
                            
                            if len(clouds) >= 1:
                            
                                area_dummy = np.nansum(labels[:,:] == n)*361 #--count the number of pixels ; assume a 19*19km per pixel = 361 km^2
                           
                                #--get the aspect ratio using region props
                                aspect_add = clouds[n].eccentricity #--just pick the fist one. as there should be only one cloud in the area that is assigned to an individual object boundary box
                                
                                orientation_add = clouds[n].orientation
                                orientation_add = orientation_add*180/np.pi + 90
                                orientation_add = np.abs(orientation_add)
                                #--reproject the anglae
                                if ((0 < orientation_add) & (orientation_add <= 90)):
                                    orientation_add = orientation_add
                                if ((90 < orientation_add) & (orientation_add <= 190)):
                                    orientation_add = -(180 - orientation_add)
                                if ((180 < orientation_add) & (orientation_add <= 270)):
                                    orientation_add = orientation_add - 180
                                if ((270 < orientation_add) & (orientation_add <= 360)):
                                    orientation_add = -(360-orientation_add)
                                #--set orientation to nan, when the shape is almost circular
                                #--because then the detection is not working; determined a threshold of 0.95
                                if aspect_add >= 0.95:
                                    orientation_add = -9999
                                
                            
                                major_ax_len_add = clouds[n].major_axis_length * 19 * 2 # approx 19 km per gridbox or pixel AND TIMES 2. because other half of the axis
                
                                
                                if ((~np.isnan(area_dummy)) & (~np.isnan(aspect_add)) & (~np.isnan(orientation_add)) & (~np.isnan(major_ax_len_add))):
                                    
                                    cloud_v3_cnt +=1 #-- increase the counter if the clouds is finally included in the processing
                                    
                                    area = np.append(area,area_dummy)
                                    aspect = np.append(aspect,aspect_add)
                                    orientation = np.append(orientation,orientation_add)
                                    major_ax_len = np.append(major_ax_len,major_ax_len_add)
                                    month_ind = np.append(month_ind,monthCounter+1)
                                    pres_ind = np.append(pres_ind,levels_era2[px])
                                    #--flag for boundary interaction
                                    #--use | or, have to touch either of the sides, not all at the same time
                                    if (lon_start_index == 0) | (lon_stop_index == nLons-1) | (lat_start_index == 0) | (lat_stop_index == nLats-1):
                                        edge_flag = np.append(edge_flag, 1)
                                    else:
                                        edge_flag = np.append(edge_flag, 0)
            
                                    print('Aspect ratio: ',aspect_add)
                                    print('Orientation deg: ',orientation_add)
                                    print('Area : ',area_dummy)
                                    print('self calc area  : ', np.pi*(major_ax_len_add/2)**2)
                                    print('major_ax_len : ',major_ax_len_add)
                                    print(edge_flag[-1])
                    
                            else:
                                print('Object too small',lon_start_index,lon_stop_index,lat_start_index,lat_stop_index)

        
            print('Print PC Flag Shape for each month file: ',PC_flag.shape)
           
            #--first calculate the overlapp; than you can average
            #--get the mask with the overlap
            #-- layer 0 of the mask is between layer 0 and 1 of the PC flag
            overlapMask = np.multiply(PC_flag[:,0:-1,:,:],PC_flag[:,1:,:,:])
            
            print('Shape of the overlap mask:', overlapMask.shape)
            
m           #--here calculate the fraction for each level; hence same number of levels as PC_flag but only one value per level
            #--no lon and lat, get mean over the entire domain
            PC_overlapDummy = np.zeros((PC_flag.shape[0],PC_flag.shape[1]))
            #overlapMask is where two adjacent layers are 1 or 0 at the same time; binary multiplication
            PC_overlapDummy[:,0] = 0 
            PC_overlapDummy[:,1:7] = np.nansum(overlapMask[:,0:6,:,:],axis=(2,3)) / np.nansum(PC_flag[:,1:7,:,:],axis=(2,3))
            #--now you can calculate the mean values
            #--for each year, month, and pressure level
            #--so I have to make the mean over time steps, lats, and lons
            PC_overlap_monthDummy[yearCounter, monthCounter,:] = np.nanmean(PC_overlapDummy,axis=(0))
            #--persistent contrail fraction per year, month, and layer
            PC_frac_monthDummy[yearCounter, monthCounter,:] = np.nanmean(np.nansum(PC_flag,axis=(2,3)) / (nLons * nLats),axis=(0))
            

            #--US section
            lon_ind4 = ((lons_era2 > -105) & (lons_era2 <= -65)) 
            PC_overlapDummy = np.zeros((PC_flag.shape[0],PC_flag.shape[1]))
            overlapMask_foo = overlapMask[:,:,:,lon_ind4]
            PC_flag_foo = PC_flag[:,:,:,lon_ind4]
            PC_overlapDummy[:,0] = 0 
            PC_overlapDummy[:,1:7] = np.nansum(overlapMask_foo[:,0:6,:,:],axis=(2,3)) / np.nansum(PC_flag_foo[:,1:7,:,:],axis=(2,3))
            PC_overlap_monthDummy_US[yearCounter, monthCounter,:] = np.nanmean(PC_overlapDummy,axis=(0))
            PC_frac_monthDummy_US[yearCounter, monthCounter,:] = np.nanmean(np.nansum(PC_flag_foo,axis=(2,3)) / (nLats * np.nansum(lon_ind4)),axis=(0))
            
            #--AT section
            lon_ind4 = ((lons_era2 > -65) & (lons_era2 <= -5)) 
            PC_overlapDummy = np.zeros((PC_flag.shape[0],PC_flag.shape[1]))
            overlapMask_foo = overlapMask[:,:,:,lon_ind4]
            PC_flag_foo = PC_flag[:,:,:,lon_ind4]
            PC_overlapDummy[:,0] = 0
            PC_overlapDummy[:,1:7] = np.nansum(overlapMask_foo[:,0:6,:,:],axis=(2,3)) / np.nansum(PC_flag_foo[:,1:7,:,:],axis=(2,3))
            PC_overlap_monthDummy_AT[yearCounter, monthCounter,:] = np.nanmean(PC_overlapDummy,axis=(0))
            PC_frac_monthDummy_AT[yearCounter, monthCounter,:] = np.nanmean(np.nansum(PC_flag_foo,axis=(2,3)) / (nLats * np.nansum(lon_ind4)),axis=(0))
            
            #--EU section
            lon_ind4 = ((lons_era2 > -5) & (lons_era2 <= 30)) 
            PC_overlapDummy = np.zeros((PC_flag.shape[0],PC_flag.shape[1]))
            overlapMask_foo = overlapMask[:,:,:,lon_ind4]
            PC_flag_foo = PC_flag[:,:,:,lon_ind4]
            PC_overlapDummy[:,0] = 0
            PC_overlapDummy[:,1:7] = np.nansum(overlapMask_foo[:,0:6,:,:],axis=(2,3)) / np.nansum(PC_flag_foo[:,1:7,:,:],axis=(2,3))
            PC_overlap_monthDummy_EU[yearCounter, monthCounter,:] = np.nanmean(PC_overlapDummy,axis=(0))
            PC_frac_monthDummy_EU[yearCounter, monthCounter,:] = np.nanmean(np.nansum(PC_flag_foo,axis=(2,3)) / (nLats * np.nansum(lon_ind4)),axis=(0))
   
                   
            
            #--still within the month counter
            #===================================
            #--get the number times one pixel is flagged as PC in time (temporal decorrelation
            #--numpy arrays where results are stored, are defined at the beginning
            #--plot happens after the the year loop
            #--tried with less loops but actually slower
            
            time_size = PC_flag.shape[0]
            p_size = PC_flag.shape[1]
            lat_size = PC_flag.shape[2]
            lon_size = PC_flag.shape[3]
            
            start_grouping_method1 = datetime.datetime.now()
            for px in np.arange(1,len(levels_era2)-2):
                groups = []
                uniquekeys = []
                for latx in np.arange(0,lat_size,2):
                    for lonx in np.arange(0,lon_size,2):
                        for k, g in groupby(PC_flag[:,px,latx,lonx]):
                            groups.append(list(g))      # Store group iterator as a list
                            uniquekeys.append(k)
                for group in groups:
                    if (group[0] == 1):  # get only these, where the flag for pc is one. otherwise you get all the non-pc in between; this works, i have varyfied
                        #print(group)
                        if (levels_era2[px] == 300):
                            con_t_pc_300 = np.append(con_t_pc_300,np.array([[len(group),monthCounter+1]]),axis=0)
                        if (levels_era2[px] == 250):
                            con_t_pc_250 = np.append(con_t_pc_250,np.array([[len(group),monthCounter+1]]),axis=0)
                        if (levels_era2[px] == 225):
                            con_t_pc_225 = np.append(con_t_pc_225,np.array([[len(group),monthCounter+1]]),axis=0)
                        if (levels_era2[px] == 200):
                            con_t_pc_200 = np.append(con_t_pc_200,np.array([[len(group),monthCounter+1]]),axis=0)
                        if (levels_era2[px] == 175):
                            con_t_pc_175 = np.append(con_t_pc_175,np.array([[len(group),monthCounter+1]]),axis=0)
            end_grouping_method1 = datetime.datetime.now()
            
            print('time for the grouping section:', end_grouping_method1 - start_grouping_method1)        
            outfile2.write('time for the grouping section: %10.2f \n' %( (end_grouping_method1 - start_grouping_method1).total_seconds()))        
            
            
    
            print('month counter: ',monthCounter) 
        print('Year counter: ',yearCounter) 
    
    
    
        
    # #%%
    

    print('area of aspect list: '+str(area.shape))
    print('aspect of aspect list: '+str(aspect.shape))
    print('orientation of aspect list: '+str(orientation.shape))
    print('major_ax_len of aspect list: '+str(major_ax_len.shape))
    
    outfile2.write('\n')
    outfile2.write('Total number of identified objects: %6.1f \n' %(area.shape) )
    
    
    # #%%
    
    #--safe the calcualted statistics so it can be plotted later without running all the coude again
                                                                                                  
    filename='pc_frac_overlap'                                                                  
    #############                                                                                 
    #save the stats                                                                               
    print('pc_frac_overlap')                                                                      
    if server == 0:                                                                               
        save_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
    if server == 1:                                                                               
        save_stats_file = ('/homedata/kwolf/41_era_statistics/'+filename)                              
    print('saved to: '+str(save_stats_file))                                                      
    np.savez(save_stats_file,PC_overlap_monthDummy, PC_overlap_monthDummy_US, PC_overlap_monthDummy_AT, PC_overlap_monthDummy_EU, PC_frac_monthDummy, PC_frac_monthDummy_US, PC_frac_monthDummy_AT, PC_frac_monthDummy_EU)
    #--these are arrays with size year,month,level
                                                                                              
    filename='orientation_aspec_area_etc'                                                                  
    #############                                                                                 
    #save the stats                                                                               
    print('pc_frac_overlap')                                                                      
    if server == 0:                                                                               
        save_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
    if server == 1:                                                                               
        save_stats_file = ('/homedata/kwolf/41_era_statistics/'+filename)                              
    print('saved to: '+str(save_stats_file))                                                      
    np.savez(save_stats_file,pres_ind, month_ind, area, aspect, orientation, major_ax_len, edge_flag)  #-- format of lists better array  with one dimension
    
    filename='temporal_decorrelation'                                                                  
    #############                                                                                 
    #save the stats                                                                               
    print('pc_frac_overlap')                                                                      
    if server == 0:                                                                               
        save_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
    if server == 1:                                                                               
        save_stats_file = ('/homedata/kwolf/41_era_statistics/'+filename)                              
    print('saved to: '+str(save_stats_file))                                                      
    np.savez(save_stats_file,con_t_pc_300, con_t_pc_250, con_t_pc_225, con_t_pc_200, con_t_pc_175) #-- array with format x,2
            
    outfile2.close()


#%%
#--if not calculation is on then you have to read the data from here.

if safe_stats == 0:
    
    levels_era2 = np.array([350, 300, 250, 225, 200, 175, 150])
    
    #### write diagnose ouput file
    filename = 'diagnose_A06_individual_blobs_diagnosis.txt'
    
    if server == 0:
        filename_diag = '/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename
    if server == 1:
        filename_diag = '/homedata/kwolf/41_era_statistics/'+filename
    
    
    
    outfile2 = open(filename_diag ,'w')
    outfile2.write('Diagnose and analysis to plots \n')
    outfile2.write('========\n')
    
    
    
    filename='pc_frac_overlap.npz'                                                                  
    #############                                                                                 
    #read the stats                                                                                                                                            
    if server == 0:                                                                               
        saved_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
    if server == 1:                                                                               
        saved_stats_file = ('/homedata/kwolf/41_era_statistics/'+filename) 
    print('reading: ',saved_stats_file)
    print('')
    dummy = np.load(saved_stats_file,allow_pickle=True)
    PC_overlap_monthDummy = np.asarray(dummy['arr_0'])
    PC_overlap_monthDummy_US = np.asarray(dummy['arr_1'])
    PC_overlap_monthDummy_AT = np.asarray(dummy['arr_2'])
    PC_overlap_monthDummy_EU = np.asarray(dummy['arr_3'])
    PC_frac_monthDummy = np.asarray(dummy['arr_4'])
    PC_frac_monthDummy_US = np.asarray(dummy['arr_5'])
    PC_frac_monthDummy_AT = np.asarray(dummy['arr_6'])
    PC_frac_monthDummy_EU = np.asarray(dummy['arr_7'])
    
    
    

    
    
    #%%
    
    filename='orientation_aspec_area_etc.npz'                                                                  
    #############                                                                                 
    #read the stats                                                                                                                                                
    if server == 0:                                                                               
        saved_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
    if server == 1:                                                                               
        saved_stats_file = ('/homedata/kwolf/41_era_statistics/'+filename) 
    print('reading: ',saved_stats_file)
    print('')
    dummy = np.load(saved_stats_file,allow_pickle=True)
    pres_ind = np.asarray(dummy['arr_0'])
    month_ind = np.asarray(dummy['arr_1'])
    area = np.asarray(dummy['arr_2'])
    aspect = np.asarray(dummy['arr_3'])
    orientation = np.asarray(dummy['arr_4'])
    major_ax_len = np.asarray(dummy['arr_5'])
    edge_flag = np.asarray(dummy['arr_6'])
    
    filename='temporal_decorrelation.npz'                                                                  
    #############                                                                                 
    #read the stats                                                                                                                                       
    if server == 0:                                                                               
        saved_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
    if server == 1:                                                                               
        saved_stats_file = ('/homedata/kwolf/41_era_statistics/'+filename) 
    print('reading: ',saved_stats_file)
    print('')
    dummy = np.load(saved_stats_file,allow_pickle=True)
    con_t_pc_300 = np.asarray(dummy['arr_0'])
    con_t_pc_250 = np.asarray(dummy['arr_1'])
    con_t_pc_225 = np.asarray(dummy['arr_2'])
    con_t_pc_200 = np.asarray(dummy['arr_3'])
    con_t_pc_175 = np.asarray(dummy['arr_4'])
    
    #%%
    
    #--load iagos flight altitude distributions
    #-- first dimension is the region: all, us, NA, eu ; second is the altitude
    filename='iagos_flight_altitude_distributions.npz'                                                                  
    #############                                                                                 
    #read the stats                                                                                                                                       
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
    #--first dimension is the region: all, us, NA, eu ; second is the altitude
    filename='iagos_flight_latitude_distributions.npz'                                                                  
    #############                                                                                 
    #read the stats                                                                                                                                       
    if server == 0:                                                                               
        saved_stats_file = ('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename)                                                                                  
    if server == 1:                                                                               
        saved_stats_file = ('/homedata/kwolf/40_era_iagos/'+filename) 
    print('reading: ',saved_stats_file)
    print('')
    dummy = np.load(saved_stats_file,allow_pickle=True)
    iagos_flatd = np.asarray(dummy['arr_0'])
    iagos_flatd_lat = np.asarray(dummy['arr_1'])

    
    #--plot area hist
    level_colors=['k','blue','red','green','orange','steelblue']
    F, x = plt.subplots(2, 2, figsize=(25,12), squeeze=False)
    x1 = x[0,0].plot(0,0)
    
    #--filter for levels
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200))
    xxxera_l, yyyera_l = cum_sum(area)
    x[0,0].plot(yyyera_l, xxxera_l, alpha=1, c='k', linewidth=2)  # 350hpa
    for f in [0.1,0.25,0.5,0.75,0.9]:
        val = np.interp(f,xxxera_l,yyyera_l,left=None,right=None)
        x[0,0].scatter(val,0,s=30,c='k',marker='o')
        x[0,0].plot((val,val),(f,0),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
        x[0,0].plot((0,val),(f,f),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
    
    #--filter for levels and edge contact; keep only meas, that do not hit the boundary
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200) & (edge_flag ==0))
    xxxera_l, yyyera_l = cum_sum(area[pfoo])
    x[0,0].plot(yyyera_l, xxxera_l, alpha=1, c='b', linewidth=2,linestyle='solid')  # 350hpa
    for f in [0.1,0.25,0.5,0.75,0.9]:
        val = np.interp(f,xxxera_l,yyyera_l,left=None,right=None)
        x[0,0].scatter(val,0,s=30,c='b',marker='o')
        x[0,0].plot((val,val),(f,0),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
        x[0,0].plot((0,val),(f,f),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
        
    x[0,0].set_xlim(100, 1e7)
    x[0,0].set_ylim(0,1)
    x[0,0].set_xscale('log')
    x[0,0].tick_params(labelsize=20)
    x[0,0].xaxis.set_tick_params(width=2, length=5)
    x[0,0].yaxis.set_tick_params(width=2, length=5)
    x[0,0].spines['top'].set_linewidth(1.5)
    x[0,0].spines['left'].set_linewidth(1.5)
    x[0,0].spines['right'].set_linewidth(1.5)
    x[0,0].spines['bottom'].set_linewidth(1.5)
    x[0,0].set_xlabel('Area [km$^2$]', fontsize=20)
    x[0,0].set_ylabel('Probability', fontsize=20)
    x[0,0].text(110,0.9,'(a)',fontsize=20)
    
    outfile2.write('\n')    
    outfile2.write('p level, 10, 25, mean, median, 75, 90  area in km2 \n')
    outfile2.write('all regions \n')
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = (pres_ind == levels_era2[ps])
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(area[pfoo],0.1),\
                                                                 np.nanquantile(area[pfoo],0.25), np.nanmean(area[pfoo]), \
                                                                     np.nanmedian(area[pfoo]), np.nanquantile(area[pfoo],0.75), \
                                                                         np.nanquantile(area[pfoo],0.90)))
    outfile2.write('Remove the ones that hit the boundary \n')
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = ((pres_ind == levels_era2[ps]) & (edge_flag ==0))
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(area[pfoo],0.1),\
                                                                 np.nanquantile(area[pfoo],0.25), np.nanmean(area[pfoo]), \
                                                                     np.nanmedian(area[pfoo]), np.nanquantile(area[pfoo],0.75), \
                                                                         np.nanquantile(area[pfoo],0.90)))
        
    
    #--plot length hist
    x1 = x[0,1].plot(0,0)
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200))
    xxxera_l, yyyera_l = cum_sum(major_ax_len)
    x[0,1].plot(yyyera_l, xxxera_l, alpha=1, c='k', linewidth=2)  # 350hpa
    for f in [0.1,0.25,0.5,0.75,0.9]:
        val = np.interp(f,xxxera_l,yyyera_l,left=None,right=None)
        x[0,1].scatter(val,0,s=30,c='k',marker='o')
        x[0,1].plot((val,val),(f,0),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
        x[0,1].plot((0,val),(f,f),linewidth=1,color='k',alpha=0.6,linestyle='dashed')
    #--filter for levels and edge contact; keep only meas, that do not hit the boundary
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200) & (edge_flag ==0))
    xxxera_l, yyyera_l = cum_sum(major_ax_len[pfoo])
    x[0,1].plot(yyyera_l, xxxera_l, alpha=1, c='b', linewidth=2,linestyle='solid')  # 350hpa
    for f in [0.1,0.25,0.5,0.75,0.9]:
        val = np.interp(f,xxxera_l,yyyera_l,left=None,right=None)
        x[0,1].scatter(val,0,s=30,c='b',marker='o')
        x[0,1].plot((val,val),(f,0),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
        x[0,1].plot((0,val),(f,f),linewidth=1,color='b',alpha=0.6,linestyle='dashed')
    
    x[0,1].set_xlim(10, 100000)
    x[0,1].set_ylim(0,1)
    x[0,1].set_xscale('log')
    x[0,1].tick_params(labelsize=20)
    x[0,1].xaxis.set_tick_params(width=2, length=5)
    x[0,1].yaxis.set_tick_params(width=2, length=5)
    x[0,1].spines['top'].set_linewidth(1.5)
    x[0,1].spines['left'].set_linewidth(1.5)
    x[0,1].spines['right'].set_linewidth(1.5)
    x[0,1].spines['bottom'].set_linewidth(1.5)
    x[0,1].set_xlabel('Major axis length [km]', fontsize=20)
    x[0,1].set_ylabel('Probability', fontsize=20)
    x[0,1].text(12,0.9,'(b)',fontsize=20)
    x[0,1].plot((0,0),(0,0),label='all',linestyle='solid',color='k')
    x[0,1].plot((0,0),(0,0),label='edge filter',linestyle='solid',color='blue')
    x[0,1].legend(shadow=True, fontsize=20,loc='center right')
    
    
    
    outfile2.write('\n')    
    outfile2.write('p level, 10, 25, mean, median, 75, 90  major axis length in km \n')
    outfile2.write('all regions \n')
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = (pres_ind == levels_era2[ps])
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(major_ax_len[pfoo],0.1),\
                                                                 np.nanquantile(major_ax_len[pfoo],0.25), np.nanmean(major_ax_len[pfoo]), \
                                                                     np.nanmedian(major_ax_len[pfoo]), np.nanquantile(major_ax_len[pfoo],0.75), \
                                                                         np.nanquantile(major_ax_len[pfoo],0.90)))
    outfile2.write('Remove the ones that hit the boundary \n')
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = ((pres_ind == levels_era2[ps]) & (edge_flag ==0))
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(major_ax_len[pfoo],0.1),\
                                                                 np.nanquantile(major_ax_len[pfoo],0.25), np.nanmean(major_ax_len[pfoo]), \
                                                                     np.nanmedian(major_ax_len[pfoo]), np.nanquantile(major_ax_len[pfoo],0.75), \
                                                                         np.nanquantile(major_ax_len[pfoo],0.90)))
        
    
    #--plot aspect ratio hist
    x1 = x[1,0].plot(0,0)
    #--filter for levels
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200))
    xxxera_l, yyyera_l = my_histogram(aspect[pfoo],0.1,1,0.1)
    x[1,0].scatter(xxxera_l[:-1]+0.015, yyyera_l, alpha=0.6, label=levels_era2[ps], c='k',s=240)  # 350hpa
    #--filter for levels and edge contact; keep only meas, that do not hit the boundary
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200) & (edge_flag ==0))
    xxxera_l, yyyera_l = my_histogram(aspect[pfoo],0.1,1,0.1)
    x[1,0].scatter(xxxera_l[:-1]-0.015, yyyera_l, alpha=0.6, c='b',s=240)  # 350hpa 

        
    x[1,0].set_xlim(-0.05, 1)
    x[1,0].tick_params(labelsize=20)
    x[1,0].xaxis.set_tick_params(width=2, length=5)
    x[1,0].yaxis.set_tick_params(width=2, length=5)
    x[1,0].spines['top'].set_linewidth(1.5)
    x[1,0].spines['left'].set_linewidth(1.5)
    x[1,0].spines['right'].set_linewidth(1.5)
    x[1,0].spines['bottom'].set_linewidth(1.5)
    x[1,0].set_xlabel('Aspect ratio', fontsize=20)
    x[1,0].set_ylabel('PDF', fontsize=20)
    x[1,0].text(-0.03,0.6,'(c)',fontsize=20)
   
 
    outfile2.write('\n')     
    outfile2.write('p level, 10, 25, mean, median, 75, 90  aspect ratio \n')
    outfile2.write('all regions \n')
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = (pres_ind == levels_era2[ps])
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(aspect[pfoo],0.1),\
                                                                 np.nanquantile(aspect[pfoo],0.25), np.nanmean(aspect[pfoo]), \
                                                                     np.nanmedian(aspect[pfoo]), np.nanquantile(aspect[pfoo],0.75), \
                                                                         np.nanquantile(aspect[pfoo],0.90)))
    outfile2.write('Remove the ones that hit the boundary \n')       
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = ((pres_ind == levels_era2[ps]) & (edge_flag ==0))
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(aspect[pfoo],0.1),\
                                                                 np.nanquantile(aspect[pfoo],0.25), np.nanmean(aspect[pfoo]), \
                                                                     np.nanmedian(aspect[pfoo]), np.nanquantile(aspect[pfoo],0.75), \
                                                                         np.nanquantile(aspect[pfoo],0.90)))

    #--remove -9999 / 9999
    rm_flag = (orientation == -9999)
    orientation[rm_flag] = np.nan

    ##--use absolute values of orietnation
    orientation = np.abs(orientation) 

    #--plot orientation hist
    x1 = x[1,1].plot(0,0)
    #--filter for levels
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200) & (~np.isnan(orientation)))
    xxxera_l, yyyera_l = my_histogram(orientation[pfoo],0,90,15)
    x[1,1].scatter(xxxera_l[:-1]+1.5, yyyera_l, alpha=0.6, c='k',s=240)  # 350hpa
    #--filter for levels and edge contact; keep only meas, that do not hit the boundary
    pfoo = ((pres_ind <= 250) & (pres_ind >= 200) & (edge_flag == 0) & (~np.isnan(orientation)))
    xxxera_l, yyyera_l = my_histogram(orientation[pfoo],0,90,15)
    x[1,1].scatter(xxxera_l[:-1]-1.5, yyyera_l, alpha=0.6, c='b',s=240)  # 350hpa
    x[1,1].set_xlim(-5, 90)
    x[1,1].set_ylim(0,0.5)
    x[1,1].tick_params(labelsize=20)
    x[1,1].xaxis.set_tick_params(width=2, length=5)
    x[1,1].yaxis.set_tick_params(width=2, length=5)
    x[1,1].spines['top'].set_linewidth(1.5)
    x[1,1].spines['left'].set_linewidth(1.5)
    x[1,1].spines['right'].set_linewidth(1.5)
    x[1,1].spines['bottom'].set_linewidth(1.5)
    x[1,1].set_xlabel('Orientation [$^\circ$]', fontsize=20)
    x[1,1].set_ylabel('PDF', fontsize=20)
    x[1,1].text(-3,0.45,'(d)',fontsize=20)
    x[1,1].scatter(-10,-100,label='all',s=240,c='red',alpha=0.6)
    x[1,1].scatter(-10,-10,label='edge filter',s=240,c='red',marker='^',alpha=0.6)
    
    outfile2.write('\n')
    outfile2.write('p level, 10, 25, mean, median, 75, 90  orientation \n')
    outfile2.write('all regions \n')
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = ((pres_ind == levels_era2[ps]) & (orientation >= 0))  #ecldue where flagged as -9999
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(orientation[pfoo],0.1),\
                                                                 np.nanquantile(orientation[pfoo],0.25), np.nanmean(orientation[pfoo]), \
                                                                     np.nanmedian(orientation[pfoo]), np.nanquantile(orientation[pfoo],0.75), \
                                                                         np.nanquantile(orientation[pfoo],0.90)))
    outfile2.write('Remove the ones that hit the boundary \n')
    for ps in np.arange(2,len(levels_era2)-2):
        pfoo = ((pres_ind == levels_era2[ps]) & (orientation >= 0) & (edge_flag ==0))  #ecldue where flagged as -9999
        outfile2.write('%4.1f %06.2f %06.2f %06.2f %06.2f %06.2f %06.2f \n' % (np.float(levels_era2[ps]), np.nanquantile(orientation[pfoo],0.1),\
                                                                 np.nanquantile(orientation[pfoo],0.25), np.nanmean(orientation[pfoo]), \
                                                                     np.nanmedian(orientation[pfoo]), np.nanquantile(orientation[pfoo],0.75), \
                                                                         np.nanquantile(orientation[pfoo],0.90)))
            
    
    filename = '3d_orientation_aspect_major_ax_length_area_edge_filtering_cdf.png'
    if server == 0:
        F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
        F.show()
    if server == 1:
        F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
        plt.close()  
        #F.show()
   
    filt = ((~np.isnan(orientation)))# & (orientation != 45))

    F,x=plt.subplots(1,3,figsize=(25,6),squeeze=False,gridspec_kw={'width_ratios':[1,1,1.2]})
    x[0,0].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    x1=x[0,0].hist2d(orientation[filt],major_ax_len[filt],range=[[0,90],[0,5000]],bins=[12,40], cmap=plt.cm.jet,norm = LogNorm(),vmin=1,vmax=10000)
    
    x[0,0].plot((0,180),(100,100),linewidth=2,linestyle='dashed',color='k')
    x[0,0].plot((0,180),(0,180),linewidth=2,linestyle='dashed',color='k')
    x[0,0].plot((100,100),(0,180),linewidth=2,linestyle='dashed',color='k')
    x[0,0].set_xlim(0,90)
    x[0,0].set_ylim(0,5000)
    x[0,0].tick_params(labelsize=20)
    x[0,0].xaxis.set_tick_params(width=2,length=5)
    x[0,0].yaxis.set_tick_params(width=2,length=5)
    x[0,0].spines['top'].set_linewidth(1.5)
    x[0,0].spines['left'].set_linewidth(1.5)
    x[0,0].spines['right'].set_linewidth(1.5)
    x[0,0].spines['bottom'].set_linewidth(1.5)
    x[0,0].set_xlabel('Orientation [$^\circ$]',fontsize = 20)
    x[0,0].set_ylabel('Major axis length [km]',fontsize = 20)
    x[0,0].text(5,4500,'(a)',fontsize=20)
    
    x[0,1].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    x1=x[0,1].hist2d(aspect[filt],major_ax_len[filt],range=[[0,1],[0,5000]],bins=[11,40], cmap=plt.cm.jet,norm = LogNorm(),vmin=1,vmax=10000)
    
    x[0,1].plot((0,180),(100,100),linewidth=2,linestyle='dashed',color='k')
    x[0,1].plot((0,180),(0,180),linewidth=2,linestyle='dashed',color='k')
    x[0,1].plot((100,100),(0,180),linewidth=2,linestyle='dashed',color='k')
    x[0,1].set_xlim(0,1)
    x[0,1].set_ylim(0,5000)
    x[0,1].tick_params(labelsize=20)
    x[0,1].xaxis.set_tick_params(width=2,length=5)
    x[0,1].yaxis.set_tick_params(width=2,length=5)
    x[0,1].spines['top'].set_linewidth(1.5)
    x[0,1].spines['left'].set_linewidth(1.5)
    x[0,1].spines['right'].set_linewidth(1.5)
    x[0,1].spines['bottom'].set_linewidth(1.5)
    x[0,1].set_xlabel('Aspect ratio [0-1]',fontsize = 20)
    x[0,1].set_yticklabels([])
    x[0,1].text(0.05,4500,'(b)',fontsize=20)
    
    x[0,2].plot(0,0)
    mycolor = ['red','green','blue','orange','steelblue','k','dimgray','lime']
    x1=x[0,2].hist2d(orientation[filt],aspect[filt],range=[[0,90],[0,1]],bins=[12,20], cmap=plt.cm.jet,norm = LogNorm(),vmin=1,vmax=10000)
    
    x[0,2].set_xlim(0,90)
    x[0,2].set_ylim(0,1)
    x[0,2].tick_params(labelsize=20)
    x[0,2].xaxis.set_tick_params(width=2,length=5)
    x[0,2].yaxis.set_tick_params(width=2,length=5)
    x[0,2].spines['top'].set_linewidth(1.5)
    x[0,2].spines['left'].set_linewidth(1.5)
    x[0,2].spines['right'].set_linewidth(1.5)
    x[0,2].spines['bottom'].set_linewidth(1.5)
    x[0,2].set_xlabel('Orientation [$^\circ$]',fontsize = 20)
    x[0,2].set_ylabel('Aspect ratio [0-1]',fontsize = 20)
    x[0,2].text(5,0.9,'(c)',fontsize=20)
    cbar = plt.colorbar(x1[3],ax=x[0,2])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Frequency of occurence',size=20)

    filename = 'relations_orientation_aspect_major_ax_length_area.png'
    if server == 0:
        F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
        F.show()
    if server == 1:
        F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
        plt.close()  
        #F.show()


    #%%

            
    #--get the mean fraction over all years
    #-- gives me the monlyt resolved vertical occurence of PC over the entire domain
    
    PC_frac_month = np.nanmean(PC_frac_monthDummy,axis=(0))
    PC_frac_month_US = np.nanmean(PC_frac_monthDummy_US,axis=(0))
    PC_frac_month_AT = np.nanmean(PC_frac_monthDummy_AT,axis=(0))
    PC_frac_month_EU = np.nanmean(PC_frac_monthDummy_EU,axis=(0))
    
    PC_overlap_month = np.nanmean(PC_overlap_monthDummy,axis=(0))
    PC_overlap_month_US = np.nanmean(PC_overlap_monthDummy_US,axis=(0))
    PC_overlap_month_AT = np.nanmean(PC_overlap_monthDummy_AT,axis=(0))
    PC_overlap_month_EU = np.nanmean(PC_overlap_monthDummy_EU,axis=(0))
    
    

    
    
    #--make a diagnostic plot
    F,x = plt.subplots(1,4,figsize=(25,8),squeeze=False)
    x[0,0].plot(np.nanmean(PC_overlap_month[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,0].plot(np.nanmean(PC_overlap_month[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,0].plot(np.nanmean(PC_overlap_month[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,0].plot(np.nanmean(PC_overlap_month[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,0].set_xlim(0,1)
    x[0,0].set_ylim(350,125)
    x[0,0].tick_params(labelsize=20)
    x[0,0].xaxis.set_tick_params(width=2, length=5)
    x[0,0].yaxis.set_tick_params(width=2, length=5)
    x[0,0].spines['top'].set_linewidth(1.5)
    x[0,0].spines['left'].set_linewidth(1.5)
    x[0,0].spines['right'].set_linewidth(1.5)
    x[0,0].spines['bottom'].set_linewidth(1.5)
    x[0,0].set_xlabel('Fractional overlap [0-1]', fontsize=20)
    x[0,0].set_ylabel('Pressure [hPa]', fontsize=20)
    x[0,0].set_title('(a) All', fontsize=20)
    x[0,0].grid()
    
    x[0,0].fill_between((0,1,1,0),(iagos_fad_quantiles[0,0], iagos_fad_quantiles[0,0],\
                                   iagos_fad_quantiles[0,4], iagos_fad_quantiles[0,4]), color='lightgray', alpha=1)
    x[0,0].fill_between((0,1,1,0),(iagos_fad_quantiles[0,1], iagos_fad_quantiles[0,1],\
                                   iagos_fad_quantiles[0,3], iagos_fad_quantiles[0,3]), color='darkgray', alpha=1)
    x[0,0].fill_between((0,1,1,0),(iagos_fad_quantiles[0,2]-1, iagos_fad_quantiles[0,2]-1,\
                                   iagos_fad_quantiles[0,2]+1, iagos_fad_quantiles[0,2]+1), color='dimgray', alpha=0.8)
    
    x[0,1].plot(np.nanmean(PC_overlap_month_US[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,1].plot(np.nanmean(PC_overlap_month_US[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,1].plot(np.nanmean(PC_overlap_month_US[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,1].plot(np.nanmean(PC_overlap_month_US[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,1].set_xlim(0,1)
    x[0,1].set_ylim(350,125)
    x[0,1].set_yticklabels([])
    x[0,1].tick_params(labelsize=20)
    x[0,1].xaxis.set_tick_params(width=2, length=5)
    x[0,1].yaxis.set_tick_params(width=2, length=5)
    x[0,1].spines['top'].set_linewidth(1.5)
    x[0,1].spines['left'].set_linewidth(1.5)
    x[0,1].spines['right'].set_linewidth(1.5)
    x[0,1].spines['bottom'].set_linewidth(1.5)
    x[0,1].set_xlabel('Fractional overlap [0-1]', fontsize=20)
    x[0,1].set_title('(b) US', fontsize=20)
    x[0,1].grid()

    x[0,1].fill_between((0,1,1,0),(iagos_fad_quantiles[1,0], iagos_fad_quantiles[1,0],\
                                   iagos_fad_quantiles[1,4], iagos_fad_quantiles[1,4]), color='lightgray', alpha=1)
    x[0,1].fill_between((0,1,1,0),(iagos_fad_quantiles[1,1], iagos_fad_quantiles[1,1],\
                                   iagos_fad_quantiles[1,3], iagos_fad_quantiles[1,3]), color='darkgray', alpha=1)
    x[0,1].fill_between((0,1,1,0),(iagos_fad_quantiles[1,2]-1, iagos_fad_quantiles[1,2]-1,\
                                   iagos_fad_quantiles[1,2]+1, iagos_fad_quantiles[1,2]+1), color='dimgray', alpha=0.8)
    
    x[0,2].plot(np.nanmean(PC_overlap_month_AT[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,2].plot(np.nanmean(PC_overlap_month_AT[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,2].plot(np.nanmean(PC_overlap_month_AT[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,2].plot(np.nanmean(PC_overlap_month_AT[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,2].set_xlim(0,1)
    x[0,2].set_ylim(350,125)
    x[0,2].set_yticklabels([])
    x[0,2].tick_params(labelsize=20)
    x[0,2].xaxis.set_tick_params(width=2, length=5)
    x[0,2].yaxis.set_tick_params(width=2, length=5)
    x[0,2].spines['top'].set_linewidth(1.5)
    x[0,2].spines['left'].set_linewidth(1.5)
    x[0,2].spines['right'].set_linewidth(1.5)
    x[0,2].spines['bottom'].set_linewidth(1.5)
    x[0,2].set_xlabel('Fractional overlap [0-1]', fontsize=20)
    x[0,2].set_title('(c) NA', fontsize=20)
    x[0,2].grid()
    
        
    x[0,2].fill_between((0,1,1,0),(iagos_fad_quantiles[2,0], iagos_fad_quantiles[2,0],\
                                   iagos_fad_quantiles[2,4], iagos_fad_quantiles[2,4]), color='lightgray', alpha=1)
    x[0,2].fill_between((0,1,1,0),(iagos_fad_quantiles[2,1], iagos_fad_quantiles[2,1],\
                                   iagos_fad_quantiles[2,3], iagos_fad_quantiles[2,3]), color='darkgray', alpha=1)
    x[0,2].fill_between((0,1,1,0),(iagos_fad_quantiles[2,2]-1, iagos_fad_quantiles[2,2]-1,\
                                   iagos_fad_quantiles[2,2]+1, iagos_fad_quantiles[2,2]+1), color='dimgray', alpha=0.8)

    
    x[0,3].plot(np.nanmean(PC_overlap_month_EU[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,3].plot(np.nanmean(PC_overlap_month_EU[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,3].plot(np.nanmean(PC_overlap_month_EU[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,3].plot(np.nanmean(PC_overlap_month_EU[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,3].set_xlim(0,1)
    x[0,3].set_ylim(350,125)
    x[0,3].set_yticklabels([])
    x[0,3].tick_params(labelsize=20)
    x[0,3].xaxis.set_tick_params(width=2, length=5)
    x[0,3].yaxis.set_tick_params(width=2, length=5)
    x[0,3].spines['top'].set_linewidth(1.5)
    x[0,3].spines['left'].set_linewidth(1.5)
    x[0,3].spines['right'].set_linewidth(1.5)
    x[0,3].spines['bottom'].set_linewidth(1.5)
    x[0,3].set_xlabel('Fractional overlap [0-1]', fontsize=20)
    x[0,3].set_title('(d) EU', fontsize=20)
    x[0,3].grid()
    
    x[0,3].fill_between((0,1,1,0),(iagos_fad_quantiles[3,0], iagos_fad_quantiles[3,0],\
                                   iagos_fad_quantiles[3,4], iagos_fad_quantiles[3,4]), color='lightgray', alpha=1)
    x[0,3].fill_between((0,1,1,0),(iagos_fad_quantiles[3,1], iagos_fad_quantiles[3,1],\
                                   iagos_fad_quantiles[3,3], iagos_fad_quantiles[3,3]), color='darkgray', alpha=1)
    x[0,3].fill_between((0,1,1,0),(iagos_fad_quantiles[3,2]-1, iagos_fad_quantiles[3,2]-1,\
                                   iagos_fad_quantiles[3,2]+1, iagos_fad_quantiles[3,2]+1), color='dimgray', alpha=0.8)
    
    #--dummy legend
    x[0,0].plot((0,0),color='blue',label='DJF')
    x[0,0].plot((0,0),color='green',label='MAM')
    x[0,0].plot((0,0),color='red',label='JJA')
    x[0,0].plot((0,0),color='orange',label='SON')
    x[0,0].legend(shadow=True,fontsize=20,loc='lower right')
    
    
    outfile2.write(' \n')
    outfile2.write('PC overlap separated for season and us, north atlantic, and europe \n')
    outfile2.write(' \n')
    outfile2.write('All regions combined for DJF, MAM, JJA, SON \n')
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_overlap_month[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_overlap_month[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_overlap_month[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_overlap_month[np.array([8,9,10]),ps],axis=(0))))
        
    outfile2.write('\n')    
    outfile2.write('US for DJF, MAM, JJA, SON \n')    
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_overlap_month_US[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_overlap_month_US[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_overlap_month_US[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_overlap_month_US[np.array([8,9,10]),ps],axis=(0))))
    outfile2.write('\n')    
    outfile2.write('Atlantic for DJF, MAM, JJA, SON \n')    
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_overlap_month_AT[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_overlap_month_AT[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_overlap_month_AT[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_overlap_month_AT[np.array([8,9,10]),ps],axis=(0))))
    outfile2.write('\n')    
    outfile2.write('EU for DJF, MAM, JJA, SON \n')    
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_overlap_month_EU[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_overlap_month_EU[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_overlap_month_EU[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_overlap_month_EU[np.array([8,9,10]),ps],axis=(0))))

    
    
    filename = 'vertical_overlap_season.png'
    if server == 0:
        F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
        F.show()
    if server == 1:
        F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
        #F.show()
        plt.close()


    F,x = plt.subplots(1,4,figsize=(25,8),squeeze=False)
    x[0,0].plot(np.nanmean(PC_frac_month[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,0].plot(np.nanmean(PC_frac_month[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,0].plot(np.nanmean(PC_frac_month[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,0].plot(np.nanmean(PC_frac_month[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,0].set_xlim(0,0.4)
    x[0,0].set_ylim(350,125)
    x[0,0].tick_params(labelsize=20)
    x[0,0].xaxis.set_tick_params(width=2, length=5)
    x[0,0].yaxis.set_tick_params(width=2, length=5)
    x[0,0].spines['top'].set_linewidth(1.5)
    x[0,0].spines['left'].set_linewidth(1.5)
    x[0,0].spines['right'].set_linewidth(1.5)
    x[0,0].spines['bottom'].set_linewidth(1.5)
    x[0,0].set_xlabel('Frequency of occurence [0-1]', fontsize=20)
    x[0,0].set_ylabel('Pressure [hPa]', fontsize=20)
    x[0,0].set_title('(a) All', fontsize=20)
    #x[0,0].text(0.05,135,'(a)',fontsize=20)
    x[0,0].grid()
    
    x[0,0].fill_between((0,1,1,0),(iagos_fad_quantiles[0,0], iagos_fad_quantiles[0,0],\
                                   iagos_fad_quantiles[0,4], iagos_fad_quantiles[0,4]), color='lightgray', alpha=1)
    x[0,0].fill_between((0,1,1,0),(iagos_fad_quantiles[0,1], iagos_fad_quantiles[0,1],\
                                   iagos_fad_quantiles[0,3], iagos_fad_quantiles[0,3]), color='darkgray', alpha=1)
    x[0,0].fill_between((0,1,1,0),(iagos_fad_quantiles[0,2]-1, iagos_fad_quantiles[0,2]-1,\
                                   iagos_fad_quantiles[0,2]+1, iagos_fad_quantiles[0,2]+1), color='dimgray', alpha=0.8)
    
    
    x[0,1].plot(np.nanmean(PC_frac_month_US[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,1].plot(np.nanmean(PC_frac_month_US[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,1].plot(np.nanmean(PC_frac_month_US[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,1].plot(np.nanmean(PC_frac_month_US[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,1].set_xlim(0,0.4)
    x[0,1].set_ylim(350,125)
    x[0,1].set_yticklabels([])
    x[0,1].tick_params(labelsize=20)
    x[0,1].xaxis.set_tick_params(width=2, length=5)
    x[0,1].yaxis.set_tick_params(width=2, length=5)
    x[0,1].spines['top'].set_linewidth(1.5)
    x[0,1].spines['left'].set_linewidth(1.5)
    x[0,1].spines['right'].set_linewidth(1.5)
    x[0,1].spines['bottom'].set_linewidth(1.5)
    x[0,1].set_xlabel('Frequency of occurence [0-1]', fontsize=20)
    x[0,1].set_title('(b) US', fontsize=20)
    x[0,1].grid()
    
    x[0,1].fill_between((0,1,1,0),(iagos_fad_quantiles[1,0], iagos_fad_quantiles[1,0],\
                                   iagos_fad_quantiles[1,4], iagos_fad_quantiles[1,4]), color='lightgray', alpha=1)
    x[0,1].fill_between((0,1,1,0),(iagos_fad_quantiles[1,1], iagos_fad_quantiles[1,1],\
                                   iagos_fad_quantiles[1,3], iagos_fad_quantiles[1,3]), color='darkgray', alpha=1)
    x[0,1].fill_between((0,1,1,0),(iagos_fad_quantiles[1,2]-1, iagos_fad_quantiles[1,2]-1,\
                                   iagos_fad_quantiles[1,2]+1, iagos_fad_quantiles[1,2]+1), color='dimgray', alpha=0.8)

    
    x[0,2].plot(np.nanmean(PC_frac_month_AT[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,2].plot(np.nanmean(PC_frac_month_AT[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,2].plot(np.nanmean(PC_frac_month_AT[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,2].plot(np.nanmean(PC_frac_month_AT[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,2].set_xlim(0,0.4)
    x[0,2].set_ylim(350,125)
    x[0,2].set_yticklabels([])
    x[0,2].tick_params(labelsize=20)
    x[0,2].xaxis.set_tick_params(width=2, length=5)
    x[0,2].yaxis.set_tick_params(width=2, length=5)
    x[0,2].spines['top'].set_linewidth(1.5)
    x[0,2].spines['left'].set_linewidth(1.5)
    x[0,2].spines['right'].set_linewidth(1.5)
    x[0,2].spines['bottom'].set_linewidth(1.5)
    x[0,2].set_xlabel('Frequency of occurence [0-1]', fontsize=20)
    x[0,2].set_title('(c) NA', fontsize=20)
    x[0,2].grid()
    
    x[0,2].fill_between((0,1,1,0),(iagos_fad_quantiles[2,0], iagos_fad_quantiles[2,0],\
                                   iagos_fad_quantiles[2,4], iagos_fad_quantiles[2,4]), color='lightgray', alpha=1)
    x[0,2].fill_between((0,1,1,0),(iagos_fad_quantiles[2,1], iagos_fad_quantiles[2,1],\
                                   iagos_fad_quantiles[2,3], iagos_fad_quantiles[2,3]), color='darkgray', alpha=1)
    x[0,2].fill_between((0,1,1,0),(iagos_fad_quantiles[2,2]-1, iagos_fad_quantiles[2,2]-1,\
                                   iagos_fad_quantiles[2,2]+1, iagos_fad_quantiles[2,2]+1), color='dimgray', alpha=0.8)

    
    x[0,3].plot(np.nanmean(PC_frac_month_EU[np.array([11,0,1]),:],axis=(0)),levels_era2,color='blue',linewidth=2,marker='o')
    x[0,3].plot(np.nanmean(PC_frac_month_EU[np.array([2,3,4]),:],axis=(0)),levels_era2,color='green',linewidth=2,marker='o')
    x[0,3].plot(np.nanmean(PC_frac_month_EU[np.array([5,6,7]),:],axis=(0)),levels_era2,color='red',linewidth=2,marker='o')
    x[0,3].plot(np.nanmean(PC_frac_month_EU[np.array([8,9,10]),:],axis=(0)),levels_era2,color='orange',linewidth=2,marker='o')
    x[0,3].set_xlim(0,0.4)
    x[0,3].set_ylim(350,125)
    x[0,3].set_yticklabels([])
    x[0,3].tick_params(labelsize=20)
    x[0,3].xaxis.set_tick_params(width=2, length=5)
    x[0,3].yaxis.set_tick_params(width=2, length=5)
    x[0,3].spines['top'].set_linewidth(1.5)
    x[0,3].spines['left'].set_linewidth(1.5)
    x[0,3].spines['right'].set_linewidth(1.5)
    x[0,3].spines['bottom'].set_linewidth(1.5)
    x[0,3].set_xlabel('Frequency of occurence [0-1]', fontsize=20)
    x[0,3].set_title('(d) EU', fontsize=20)
    x[0,3].grid()

    x[0,3].fill_between((0,1,1,0),(iagos_fad_quantiles[3,0], iagos_fad_quantiles[3,0],\
                                   iagos_fad_quantiles[3,4], iagos_fad_quantiles[3,4]), color='lightgray', alpha=1)
    x[0,3].fill_between((0,1,1,0),(iagos_fad_quantiles[3,1], iagos_fad_quantiles[3,1],\
                                   iagos_fad_quantiles[3,3], iagos_fad_quantiles[3,3]), color='darkgray', alpha=1)
    x[0,3].fill_between((0,1,1,0),(iagos_fad_quantiles[3,2]-1, iagos_fad_quantiles[3,2]-1,\
                                   iagos_fad_quantiles[3,2]+1, iagos_fad_quantiles[3,2]+1), color='dimgray', alpha=0.8)
    
    
    
    outfile2.write(' \n')
    outfile2.write('PC frac occurence separated for season and us, north atlantic, and europe \n')
    outfile2.write(' \n')
    outfile2.write('All regions combined for DJF, MAM, JJA, SON \n')
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_frac_month[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_frac_month[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_frac_month[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_frac_month[np.array([8,9,10]),ps],axis=(0))))
        
    outfile2.write('\n')    
    outfile2.write('US for DJF, MAM, JJA, SON \n')    
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_frac_month_US[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_frac_month_US[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_frac_month_US[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_frac_month_US[np.array([8,9,10]),ps],axis=(0))))
    outfile2.write('\n')    
    outfile2.write('Atlantic for DJF, MAM, JJA, SON \n')    
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_frac_month_AT[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_frac_month_AT[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_frac_month_AT[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_frac_month_AT[np.array([8,9,10]),ps],axis=(0))))
    outfile2.write('\n')    
    outfile2.write('EU for DJF, MAM, JJA, SON \n')    
    for ps in np.arange(1,len(levels_era2)-1):
        outfile2.write('%3.0f %4.2f %4.2f %4.2f %4.2f \n' % (levels_era2[ps],\
                                                          np.nanmean(PC_frac_month_EU[np.array([11,0,1]),ps],axis=(0)),\
                                                              np.nanmean(PC_frac_month_EU[np.array([2,3,4]),ps],axis=(0)),\
                                                                  np.nanmean(PC_frac_month_EU[np.array([5,6,7]),ps],axis=(0)), \
                                                                      np.nanmean(PC_frac_month_EU[np.array([8,9,10]),ps],axis=(0))))
    
    
    #--dummy legend
    x[0,0].plot((0,0),color='blue',label='DJF')
    x[0,0].plot((0,0),color='green',label='MAM')
    x[0,0].plot((0,0),color='red',label='JJA')
    x[0,0].plot((0,0),color='orange',label='SON')
    x[0,0].legend(shadow=True,fontsize=20,loc='upper right')
    
    filename = 'vertical_occurence_season.png'
    if server == 0:
        F.savefig('/home/kwolf/Documents/00_CLIMAVIATION/01_python_code/my_routines/dummy/'+filename,bbox_inches='tight')
        F.show()
    if server == 1:
        F.savefig('/homedata/kwolf/41_era_statistics/plots/'+filename,bbox_inches='tight')
        plt.close()    
        #F.show()


#%%

    #--close outputfile
    outfile2.close()
    
    
#--end of code
