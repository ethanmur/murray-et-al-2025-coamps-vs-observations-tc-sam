# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:13:02 2024

@author: etmu9498
"""



# import...
import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns


# 
def return_coamps_intensity(path_coamps = basepath + "data/best-tracks/coamps/",
                           file_c = "TRK_COAMPS_CTCX_3_2021081800_07L"):    
    # descriptions of Coamps best track columns
    col_names = ['Basin', 'TC Number', 'Run Time', '03', 'Model', 'Lead Time', 'Lat', 'Lon', 'Intensity (Kt)', 'Pressure (hPa)', 
                 'XX', 'Wind Radius Threshold (Kt)', 'Quadrant ID', 'NE Wind Radius (NM)', 'SE Wind Radius (NM)', 'SW Wind Radius (NM)',
                 'NW Wind Radius (NM)', 'h', 'i', 'RMW (NM)']
        
    # load coamps track data
    track_c = pd.read_fwf( path_coamps + file_c, header=None, names = col_names)
    # remove pointless columns
    track_c = track_c.drop(track_c.columns[17:19], axis=1)
    track_c = track_c.drop(track_c.columns[3], axis=1)
    track_c = track_c.drop(track_c.columns[9], axis=1)

    # turn lats / lons into a readable format
    vmaxstr, pminstr = track_c['Intensity (Kt)'].values, track_c['Pressure (hPa)'].values
    vmax, pmin = [], []
    for vi, vstr in enumerate(vmaxstr):
        vstr=vstr[:len(vstr)-1] # cut off trailing ','
        pstr = pminstr[vi][:len(vstr)-1]
        vmax.append(float(vstr))
        pmin.append(float(pstr))
    
    # append float columns back to pandas dataframe
    track_c['Vmax'] = vmax
    track_c['Pmin'] = pmin
    # drop rows where there are duplicate lead times (from the three different wind radii parameters)
    leads = track_c['Lead Time'].values
    # get inds of unique and repeat lead times (from 3 wind radii)
    uniquei = np.unique(leads, return_index=True)[1]
    repeati = np.setdiff1d(np.arange(0, len(leads), 1), uniquei)

    # dataframe holding rows with only unique lead times
    tracks_c_unique = track_c.drop(repeati)

    # cycle through string timesteps and convert to floats
    lt_floats = []
    for lt in tracks_c_unique['Lead Time'].values:
        lt_floats.append(float(lt[0:3]))
    tracks_c_unique['Lead Time Float'] = lt_floats

    return tracks_c_unique['Lead Time Float'].values, tracks_c_unique['Vmax'].values, tracks_c_unique['Pmin'].values


# setup for making an intensity plot from ships environmental data!
def intensity_ships(tcname='Sam', year='2021'):
    os.chdir( basepath + "/data/ships/")
    file1 = open('2021_5day.txt', 'r') 
    Lines = file1.readlines()
    
    coamps_bt_path = basepath + "data/best-tracks/coamps/"

    # TC Sam information
    fl_list = [ 'P3_20210926H1_highres.nc', 'P3_20210927H1_highres.nc', 
                'P3_20210929H2_highres.nc']
    i0 = 5
    i1 = 34
    yvmax = [60, 140]
    strname='SAM'
    coamps_bt_names = ["TRK_COAMPS_CTCX_3_2021092612_18L",
                        "TRK_COAMPS_CTCX_3_2021092712_18L",
                        "TRK_COAMPS_CTCX_3_2021092818_18L",
                        ]
    startdates = [datetime.datetime(2021, 9, 26, 12, 0),
                    datetime.datetime(2021, 9, 27, 12, 0),
                    datetime.datetime(2021, 9, 28, 18, 0)]

    # convenient x axis variables for plotting
    time_since_start = []
    datetimes = [] # save datetime objects here
    date_only = [] # save MM/DD strings for nicer date representations below
    
    # useful y axis variables
    vmax = []
    psurf = []
    startdate = 0
    starttime = 0
    
    # go through all the lines to find indices 
    header_inds = []
    for ind in range( len( Lines)):
        # get the heading lines, and look for this TC's cases!
        if 'HEAD' and strname  in Lines[ ind]:
            # only keep 2021 cases
            if Lines[ ind][ 6 : 8] == str( year[ 2:4]):
                header_inds.append( ind)
                
    # search through all headers and append useful plotting info like current time, date, etc
    for headeri, headerval in enumerate( header_inds):    
        # add times to the list! increments of 6 hours
        if headeri == 0:
            time_since_start.append( 0)
            # append starting dates and times
            for i in range( headerval,  len( Lines) ):
                if 'HEAD' in Lines[ i]:
                    startdate = Lines[i][6:12]
                    starttime = Lines[i][13:15]
                    break
        else:
            # otherwise, find the most recent time and add 6 hours!
            time_since_start.append( time_since_start[-1] + 6)
    
        # add datetime objects!
        for i in range( headerval,  len( Lines) ):
            if 'HEAD' in Lines[ i]:
                month = int( Lines[i][8:10] )
                day = int( Lines[i][10:12] )
                hours = int( Lines[i][13:15] )
                datetime_orig = datetime.datetime( int( year), month, day, hours, 0, 0)
                datetimes.append( datetime_orig) # datetime_orig.strftime( "%m/%d %H %M %S"))
                date_only.append( str( datetime_orig.strftime( "%m/%d")))
                break
        # search for vmax!
        for i in range( headerval,  len( Lines) ):
            if 'VMAX' in Lines[i]:
                vmax.append( int( Lines[i][12 : 15]) ) # the last 3 vals for vmax at 0 hours
                break
        # repeat for pressure
        for i in range( headerval,  len( Lines) ):
            if 'MSLP' in Lines[i]:
                psurf.append( int( Lines[i][11:15] )  ) # the last 3 vals for vmax at 0 hours
                break
                
    
    # trim down datetimes and vmaxes to these limits for more concise plotting
    datetimes = datetimes[ i0:i1]
    date_only = date_only[ i0:i1]
    vmax = vmax[ i0:i1]
    psurf = psurf[ i0:i1]
    
    
    # make the figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5)) 
    fs=12
    helper_fns.change_font_sizes( fs, fs)
    lw = 2
    
    # add nice axis labels and add the wind speed plot!
    # ax.set_title( "Intensity of TC " + tcname + " during Flight Periods", fontsize= fs)
    ax.set_xlabel( "Date and Time (2021, UTC)")
    ax.set_ylabel( "$V_{Max}$ (kt)")
    plt.xticks(rotation=60, ha="right")
    ax.plot( datetimes, vmax, c='k', linewidth=lw, label='NHC Best Track')
    
    xloc, yloc = .92, .885
    txt = plt.text( xloc, yloc, '(a)', fontsize=fs, transform=ax.transAxes)
    txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

    # get plot colors
    cmap = plt.cm.get_cmap( "rainbow") # "YlOrRd")
    colors = []
    for i in range( 6):
        colors.append( cmap(i/5))
    ax.set_ylim( yvmax)
    ax.set_title("TC Sam, Intensity", fontsize=fs+3)
    colors = ['mediumblue', 'firebrick', 'skyblue']
    
    # 9/18/24 new: add coamps intensity forecasts to plot as well!
    # also add a single scatter point to represent the starting intensity!
    for ci, coamps_bt_name in enumerate(coamps_bt_names):
        # pull starting datetime for this case
        startdate = startdates[ci]
        # find ind where startdate matches nhc datetime list
        startind = np.where(np.array(datetimes)==startdate)[0][0]

        # trim coamps vmax and pmin values to fit nhc time axis!
        nhcdates = np.array(datetimes)[startind:]
        ctrack_times, vmax, pmin = return_coamps_intensity(coamps_bt_path, coamps_bt_name)
        vmax = vmax[:len(nhcdates)]
        
        if ci==0:
            ax.plot(nhcdates, vmax, c='orange', linewidth=.8, label='COAMPS-TC forecasts')
        else:
            ax.plot(nhcdates, vmax, c='orange', linewidth=.8)
        ax.scatter(nhcdates[0], vmax[0], c='orange', s=20, zorder=20)



    
    #############
    # plot vertical lines on main plot signifying locations of P-3 flights! at median time
    #############
    # add each fl case- taken from spatial plot above
    
    # find the median times and convert them to datetimes
    for fli, flval in enumerate( fl_list):
        # load current dataset
        os.chdir( "/Users/etmu9498/all-research/nrl-summer-2024-local/data-crl/")
        crl_data = xr.open_dataset( flval, decode_times=False)
        
        # make vertical bars representing timespans of flights!
        start, end = crl_data.time[0].values, crl_data.time[-1].values
        remainder_min_start, remainder_min_end = ( start - math.floor( start) ) * 60, ( end - math.floor( end) ) * 60
        remainder_sec_start, remainder_sec_end = ( remainder_min_start - math.floor( remainder_min_start)) * 60, ( remainder_min_end - math.floor( remainder_min_end)) * 60
        # get the median time from hours (decimal) into datetime format for plotting!
        month = int( flval[7:9] )
        startday = int( flval[9:11] )
        starthours, endhours = math.floor( start), math.floor( end) # round down the median hour val
        # wraparound case
        if endhours > 23:
            endhours = endhours - 24
            endday = startday + 1
        else:
            endday = startday
        startminutes, endminutes = math.floor( remainder_min_start), math.floor( remainder_min_end)
        startseconds, endseconds = round( remainder_sec_start), round( remainder_sec_end)

        # account for overlap second cases (maybe repeat for minutes?)        
        if startseconds==60:
            startminutes+=1
            startseconds=0
        if endseconds==60:
            endminutes+=1
            endseconds=0
        startdt = datetime.datetime( int( year), month, startday, starthours, startminutes, startseconds)
        enddt = datetime.datetime( int( year), month, endday, endhours, endminutes, endseconds)
        ax.axvspan( startdt, enddt, color= colors[fli], alpha=.5, label=flval )
    
        
    # change the datetime format on the x axis to just dates
    myFmt = DateFormatter('%m/%d')
    ax.xaxis.set_major_formatter(myFmt)
    
    xmin = datetime.datetime(2021, 9, 22, 12, 0)
    xmax = datetime.datetime(2021, 10, 1, 6, 0)
    plt.xlim([xmin, xmax])
        
    os.chdir(basepath + "figures/")
    plt.savefig("figure1_intensity.png", dpi=300., bbox_inches='tight')
