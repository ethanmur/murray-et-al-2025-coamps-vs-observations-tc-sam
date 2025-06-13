# import...
import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import scipy
import scipy.stats as stats
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['figure.dpi'] = 300
from scipy import interpolate

basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns



            
    
    
# plot composites saved in local nrl-summer-2024 folder! add anomalies too.
# a new figure for each variable and each day.
def plot_composites(fig, ax0, ax1, ax2, tcname='20210929H2', correct=False, 
                    coamps_curtain=True, var='radial', testype='profile', manual_center=False,
                    xaxis='dist', coamps_rmw=0., tdr_rmw=0.):
    if var=='radial':
        vartit ='Radial Wind (m/s)'
        coamps_var='radwnd'
    elif var=='tangential':
        vartit='Tangential Wind (m/s)'
        coamps_var='tanwnd'
    elif var=='vertical':
        vartit='Vertical Wind (m/s)'
        coamps_var='wwwind'
        
        
    # for a given TC name, find the matching coamps dataset and metadata associated with it
    crlname='P3_' + tcname
    # pull metadata here        
        
    metadata = helper_fns.obs_to_single_model_run(crlname)
    coamps_folder, coamps_name, crlfolder, nhcpath, nhcfile = metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], 
    coamps_bt_path, coamps_bt_name, nhcdate, nhci, li = metadata[5], metadata[6], metadata[7], metadata[8], metadata[9]
    
    #crlname+="_highres.nc"

    # cycle through variables and create subplots

    # rename the crl dataset for loading next!
    crlinput = crlname + '_highres.nc'

    if testype=='profile':
        spath = basepath + "data/tdr-profile-composites/"
    elif testype=='level3':
        spath = basepath + "data/tdr-level3-composites/"


    # load tdr composite data for this day    
    savetit = tcname + "-" + var + '.png'
    savestart=tcname + '-'                      
    os.chdir(spath)  
    x, y = np.load(savestart + 'x.npy'), np.load(savestart + 'y.npy')
    binned_tdr = np.load(savestart + var + '.npy')

    # clear tdr data below 450m for 9/29 data -> too noisy
    if tcname=='20210929H2' and var != 'vertical':
        binned_tdr[:,:3] = np.nan

    # using selected metadata, use helper fn to trim data to crl limits (along tdr
    # path too) and return flattened data.
    rc, hc, varc  = helper_fns.flatten_tdr(coamps_folder, coamps_name, crlfolder, crlinput, nhcpath, nhcfile,
                                                    coamps_bt_path, coamps_bt_name, nhcdate, nhci, li, var=coamps_var,
                                                    correct_crl_data=correct, dtype='coamps', coamps_curtain=coamps_curtain,
                                                    test_curtains=False, manual_center=manual_center)

    print('COAMPS data flattened.')
    
    # complete the coamps compositing
    binned_coamps_all = scipy.stats.binned_statistic_2d(rc, hc, varc, statistic='mean', bins=[x,y])
    # only keep stat output
    binned_coamps = binned_coamps_all.statistic
    
    # calculate anomaly between the two: easy because they are binned the same way!!
    anom = binned_coamps - binned_tdr

    print('COAMPS data composited.')

    # variable presets 
    if var=='radial':
        color_map = 'bwr'
        levels = np.linspace(-15,15,14) # 13 before
        ticks=[-15., -7.5, 0., 7.5, 15.]
        ex = 'both'
        anomlevels = np.arange(-10,10,2.)
        anomticks=[-10., 0., 10.]
    elif var=='tangential':
        color_map = 'plasma'
        levels = np.arange(0,70,5)
        ex = 'max'
        ticks=[0., 20, 40., 60.]
        anomlevels = np.arange(-20,20,4.)
        anomticks=[-20., 0., 20.]
    elif var=='vertical':
        color_map = 'bwr'
        levels = np.arange(-4,4,.25)
        ex = 'both'
        ticks=[-4., -2., 0., 2., 4.]
        anomlevels = np.arange(-2,2,.25)
        anomticks=[-2., 0., 2.]
        

    # if vt case, find and plot rmws at each height level! for model and obs
    if var=='tangential':

        print(tcname)
        print(len(y))
        print(np.shape(binned_coamps))
        print(np.shape(binned_tdr))
        

        # cycle through binned heights
        coamps_maxhs = []
        tdr_maxhs = []
        for hti, ht in enumerate(y):
            if hti==len(y)-1:
                continue

            # check for nans for both fields
            if len(np.where(~np.isnan(binned_coamps[:,hti]))[0])==0 or len(np.where(~np.isnan(binned_tdr[:,hti]))[0])==0:
                coamps_maxhs.append(np.nan)
                tdr_maxhs.append(np.nan)
                continue

            coamps_maxi = np.nanargmax(binned_coamps[:,hti])
            tdr_maxi = np.nanargmax(binned_tdr[:,hti])
            # coamps_maxhs.append(binned_coamps[coamps_maxi,hti])
            # tdr_maxhs.append(binned_tdr[tdr_maxi,hti])
            coamps_maxhs.append(x[coamps_maxi])
            tdr_maxhs.append(x[tdr_maxi])

            print(f'Height ({ht} m, {hti} ind), ' +
                  f'TDR ({binned_tdr[tdr_maxi,hti]:.1f} m/s, {x[tdr_maxi]:.2f} km, {tdr_maxi} ind),' +
                  f'Coamps ({binned_coamps[coamps_maxi,hti]:.1f} m/s, {x[coamps_maxi]:.2f} km, {coamps_maxi} ind)')


    # actually create the plot here!
    fs = 12
    lw = 1.5
    xtxt, ytxt = .7, 1.2
    helper_fns.change_font_sizes(fs,fs)


    # new: if xaxis='rmw', divide distance by rmw to create a normalized r* axis!
    if xaxis=='rmw':
        coampsx = x / coamps_rmw
        tdrx = x / tdr_rmw

        # also need to re bin coamps coamps data to reflect rmw axis??
        # interpolate coamps data to match tdr scale

        # create interpolation function for each height level
        binned_coamps_interp = np.full_like(binned_tdr, np.nan)
        for hti in range(binned_coamps.shape[1]):
            # get valid data points (no nans)
            valid_mask = ~np.isnan(binned_coamps[:, hti])

            if len(np.where(valid_mask)[0]) > 0:
                f_interp = interpolate.interp1d(coampsx[:-1][valid_mask], binned_coamps[valid_mask, hti], 
                                                kind='linear', bounds_error=False, fill_value=np.nan)
                binned_coamps_interp[:, hti] = f_interp(tdrx[:-1])
            else:
                binned_coamps_interp[:, hti] = np.nan

        # update variables to use interpolated data
        #binned_coamps = binned_coamps_interp
        #coampsx = tdrx
        anom = binned_coamps_interp - binned_tdr

    else:
        coampsx = x
        tdrx = x


    f = ax0.contourf( coampsx[:-1], y[:-1]/1000., binned_coamps.transpose(), cmap=color_map, levels=levels, extend=ex)    
    if var=='tangential':
        ax0.plot(coamps_maxhs, y[:-1]/1000, c='k', lw=1.6)

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(f, cax=cax, ticks=ticks,orientation='vertical', label=vartit)


    f = ax1.contourf( tdrx[:-1], y[:-1]/1000., binned_tdr.transpose(), cmap=color_map, levels=levels, extend=ex)    
    if var=='tangential':
        ax1.plot(tdr_maxhs, y[:-1]/1000, c='k', lw=1.6)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(f, cax=cax, ticks=ticks,orientation='vertical', label=vartit)


    f = ax2.contourf( tdrx[:-1], y[:-1]/1000., anom.transpose(), cmap='BrBG', levels=anomlevels, extend='both')    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(f, cax=cax, ticks=anomticks,orientation='vertical', label=vartit)

    # make left axes smaller to fit colorbar label!
    plt.subplots_adjust(wspace=.6)


    for ax in [ax0, ax1, ax2]:
        # ax.set_ylim([.3,6.5])  
        ax.set_ylim([0.,6.5])  
        # ax.set_ylim([0.,14])  

        if xaxis=='rmw':
            ax.set_xlabel('r*')
            ax.set_xlim([0,5.])
            ax.axvline(x=1.,c='k',lw=.8)
            ax.set_xticks([0,1,2,3,4,5])
        else:
            ax.set_xlabel('Radial Distance (km)')
            ax.set_xlim([0,150])
    ax0.set_ylabel( 'Height (km)')

    return fig, ax0, ax1, ax2