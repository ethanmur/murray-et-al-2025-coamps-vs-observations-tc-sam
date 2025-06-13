# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:12:28 2024

@author: etmu9498
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs          # For plotting maps
import cartopy.feature as cfeature  # For plotting maps
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns


# same as plots created above, but make two subplots: one with tdr mapped out,
# another with flight tracks and nhc ( + model?) 6h tracks.
def two_subplots(tcname='Sam', tcyear=2021, tdrfield='reflect'):
    # load data file: 
    path_nhc=basepath + "data/best-tracks/"
    coamps_bt_path = basepath + "data/best-tracks/coamps/"
    
    # old path
    # path_xy = "/Users/etmu9498/research/data/tc-radar/"
    # tdr_name = 'tc_radar_v3k_2020_2022_xy_rel_merge_ships.nc'
    # new path: 
    path_xy = basepath + "data/tc-radar/"
    tdr_name = 'tc_radar_v3k_2020_2022_xy_rel_merge_ships_sam.nc'

    #os.chdir(path_xy)
    tdr = xr.open_dataset(path_xy + tdr_name)    

    ymin, ymax = 11, 22.5
    xmin, xmax = -62, -48 
    mission_list = ['210926H1', '210927H1', '210929H2']
    ncol=1
    loc='lower left'
    lw=1.2
    file_nhc='bal182021.dat'
    coamps_bt_names = ["TRK_COAMPS_CTCX_3_2021092612_18L",
                        "TRK_COAMPS_CTCX_3_2021092712_18L",
                        "TRK_COAMPS_CTCX_3_2021092818_18L",
                        ]
    
    # get plot colors
    cmap = plt.cm.get_cmap( "rainbow") # "YlOrRd")
    colors = []
    for i in range( 6):
        colors.append( cmap(i/5))
    colori = 0
    colors = ['mediumblue', 'firebrick', 'skyblue']

    # new: make one figure outside of loop
    proj = ccrs.PlateCarree()
    
    fig = plt.figure(figsize=(11,3.5))    
    ax0 = fig.add_subplot(121, projection=proj)
    ax1 = fig.add_subplot(122, projection=proj)
    sz = 14.
    xloc, yloc = .85, .875
    helper_fns.change_font_sizes(sz, sz)  


    # create lines subplot here    
    ax = ax0
    # Bound and fill in nice lines
    MAP_BOUNDS = [xmin, xmax, ymin, ymax]
    ax.set_extent(MAP_BOUNDS)    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)                 
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.tick_params(labelsize=sz)
    ax.set_ylabel("Latitude ($\degree$)")
    ax.set_xlabel("Longitude ($\degree$)")
    ax.set_title("TC " + tcname + ", Tracks")
    
    crlorder=2.
    nhcorder=0.
    coampsorder=1.
    
    txt = plt.text( xloc, yloc, '(b)', fontsize=sz, transform=ax.transAxes)
    txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

    
    # only need 4 markers for H1 storms
    clats, clons, ids = [], [], []
    
    # cycle through all swath or merged cases
    ncount = 0
    for casei, casename in enumerate(tdr['storm_name'].values):
        year = tdr['merge_year'].values[casei]
        missionid = tdr['mission_ID'].values[casei]
    
        if missionid in mission_list:
            ids.append(missionid)
            
            hri = tdr['merge_hour'].values[casei]
            mini = tdr['merge_min'].values[casei]
            medtime = str(np.round(hri + mini/60., 2))
             
            pass_lat, pass_lon = tdr.merged_latitudes[casei, :, :].values, tdr.merged_longitudes[casei, :, :].values
           
           
            # new lat / lon finder: use 2 km center instead
            # results: dots are much closer to actual radar centers now!
            clats.append(tdr['tc_center_latitudes'].values[casei, 4])
            clons.append(tdr['tc_center_longitudes'].values[casei, 4])
    
            
            # add crl flight tracks here!
            os.chdir(basepath + "/data/crl/")
            crl = xr.open_dataset("P3_20" + missionid + "_highres.nc")
            
            ax.plot(crl.Lon, crl.Lat, c=colors[colori], label=missionid, linewidth=lw, zorder=crlorder)
            colori+=1
       
    # 9/6/24 new: load and plot nhc best tracks for both storms
    # use a helper fn to load nhc track data
    nhc_dates, nhc_times, nhc_lats, nhc_lons = helper_fns.return_nhc_lat_lons(path_nhc, file_nhc)
    ax.plot(nhc_lons, nhc_lats, c='k', linewidth='1.2', zorder=nhcorder)
    ax.scatter(nhc_lons, nhc_lats, c='k', s=18, label='NHC 6H Best Track', zorder=nhcorder)
    

    # 9/18/24 new: plot coamps tracks beneath nhc and flights!
    for ci, coamps_bt_name in enumerate(coamps_bt_names):
        ctrack_times, ctrack_lats, ctrack_lons = helper_fns.return_coamps_lat_lons(coamps_bt_path, coamps_bt_name)
        if ci==0:
            ax.plot(ctrack_lons, ctrack_lats, c='orange', linewidth=.8, zorder=coampsorder, label='COAMPS-TC Forecasts')
        else:
            ax.plot(ctrack_lons, ctrack_lats, c='orange', linewidth=.8,zorder=coampsorder)
    
    
    # repeat plot for tdr depiction of sam!
    ax = ax1    
    # Bound and fill in nice lines
    MAP_BOUNDS = [xmin, xmax, ymin, ymax]
    ax.set_extent(MAP_BOUNDS)    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)                 
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.tick_params(labelsize=sz)
    plt.ylabel("Latitude ($\degree$)")
    plt.xlabel("Longitude ($\degree$)")
    plt.title("TC " + tcname + ", TDR Data")    
    
    # only need 4 markers for H1 storms
    clats, clons, ids = [], [], []
    
    txt = plt.text( xloc, yloc, '(c)', fontsize=sz, transform=ax.transAxes)
    txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

    # cycle through all swath or merged cases
    ncount = 0
    for casei, casename in enumerate(tdr['storm_name'].values):
        year = tdr['merge_year'].values[casei]
        missionid = tdr['mission_ID'].values[casei]
        if missionid in mission_list:
            ids.append(missionid)
            
            hri = tdr['merge_hour'].values[casei]
            mini = tdr['merge_min'].values[casei]
            medtime = str(np.round(hri + mini/60., 2))
    
            if tdrfield=='reflect':
                pass_r = tdr.merged_reflectivity[casei, :, :, 4].values        
                ref_color_map = plt.cm.get_cmap( "RdYlBu").reversed()
                levels = np.arange(0, 46,5)
                rlabel='2 km TDR Reflectivity (dBz)'
                
            elif tdrfield=='windspeed':        
                #pass_r = tdr.merged_wind_speed[casei, :, :, 1].values        
                pass_r = tdr.merged_wind_speed[casei, :, :, 4].values        
                ref_color_map = 'plasma'
                levels = np.arange(0, 80,5)
                rlabel='2 km TDR Wind Speed (m/s)'

            pass_lat, pass_lon = tdr.merged_latitudes[casei, :, :].values, tdr.merged_longitudes[casei, :, :].values
            # pull reflectivity, winds
            plt.contourf( pass_lon, pass_lat, pass_r, cmap = ref_color_map, levels=levels)
            if ncount == 0:
                plt.colorbar(label=rlabel)
            ncount += 1

            # new lat / lon finder: use 2 km center instead
            # results: dots are much closer to actual radar centers now!
            clats.append(tdr['tc_center_latitudes'].values[casei, 4])
            clons.append(tdr['tc_center_longitudes'].values[casei, 4])
    
                       
    os.chdir(basepath + "figures/")
    plt.savefig('figure1_track.png', dpi=300., bbox_inches='tight')
    