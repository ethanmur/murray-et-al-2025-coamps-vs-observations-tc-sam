
import os
import xarray as xr
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import metpy.calc as mpcalc
from metpy.units import units
import cartopy.crs as ccrs          # For plotting maps
import cartopy.feature as cfeature  # For plotting maps
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable


basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns


def interp_binned_smooth(coamps_composite, coampsh, hinterp):
    coamps_interp = np.empty((np.shape(coamps_composite)[0], len(hinterp)))    
    for xi in range(np.shape(coamps_composite)[0]):
        f = interpolate.interp1d(coampsh, coamps_composite[xi,:], fill_value='extrapolate')
        coamps_interp[xi,:] = f(hinterp)        
    return coamps_interp

# find composites of crl and coamps data, and return them for plotting later / next!

def prepare_data(coamps_folder, coamps_name, crlfolder, crlname, nhcpath, nhcfile, coamps_bt_path, 
                   coamps_bt_name, nhcdate, nhci, ti, var='pottmp', correct_crl_data=False,
                   correct_motion=True, manual_center=False):

    
    # load coamps track lat lons for all leadtimes
    track_times, track_lats, track_lons = helper_fns.return_coamps_lat_lons(coamps_bt_path, coamps_bt_name)
    # load nhc track data
    nhc_dates, nhc_times, nhc_lats, nhc_lons = helper_fns.return_nhc_lat_lons(nhcpath, nhcfile)
    
    # load coamps-tc data in .nc format
    os.chdir(coamps_folder)
    coamps_data = xr.open_dataset(coamps_name)
    sigma = np.flip(coamps_data.z_sig.values) # flip to plot surface levels first
    # leadtimes = coamps_data.leadtime.values
    # leadtime = leadtimes[ti]
    leadtime = coamps_data.leadtime.values
    
    os.chdir(crlfolder)
    crl_data = xr.open_dataset(crlname)

    lat, lon = coamps_data.lat.values, coamps_data.lon.values

    # Pull track center corresponding to selected leadtime, and find radial distances
    if manual_center:
        clat, clon = helper_fns.manual_centers(crlname[:-11])                       
    else:
        time_match_i = np.where(leadtime==track_times)[0]
        clat, clon = track_lats[time_match_i], track_lons[time_match_i]
    
    cx, cy = helper_fns.return_coamps_xy(coamps_data, clat, clon, ti)
    r = (cx**2 + cy**2)**.5 # simple radius calculation

    # find crl distances next
    ox, oy = crl_data.x_2m.values, crl_data.y_2m.values
    r_o = (ox**2 + oy**2)**.5 # simple radius calculation
    h = crl_data.height.values


    # Note: only do this for one variable

    # use these bins for both crl and coamps analysis
    binr = np.arange(0,300,3)
    binh = np.arange(0,3500,200)

    # pull crl variable here
    # variable options: ['T', 'pottmp', 'thetae', 'thetav', 'wvapor', 'relhum']
    # some special cases when crl label doesn't match coamps label
    
    # 8/27 new wind speed composites: do this before thermo composites! no flattening yet
    if var=='wndspd':
        nanarray = np.empty((len(binr)-1, len(binh)-1))
        nanarray[:] = np.nan
        binned_crl = {
            'statistic': nanarray,
            'x_edge': binr,
            'y_edge': binh,
            } 
    elif var=='radwnd':
        nanarray = np.empty((len(binr)-1, len(binh)-1))
        nanarray[:] = np.nan
        binned_crl = {
            'statistic': nanarray,
            'x_edge': binr,
            'y_edge': binh,
            } 
    elif var=='wwwind':
        nanarray = np.empty((len(binr)-1, len(binh)-1))
        nanarray[:] = np.nan
        binned_crl = {
            'statistic': nanarray,
            'x_edge': binr,
            'y_edge': binh,
            } 
        
    # thermodynamic variables:
    else:
        if var=='pottmp':
            crlvar = helper_fns.select_crl_var('theta', crl_data, correct_crl_data=correct_crl_data, fname=crlname[3:-11])
        elif var=='wvapor':
            crlvar = helper_fns.select_crl_var('wv', crl_data, correct_crl_data=correct_crl_data, fname=crlname[3:-11])
        elif var=='relhum':
            crlvar = helper_fns.select_crl_var('rh', crl_data, correct_crl_data=correct_crl_data, fname=crlname[3:-11])
        else:
            crlvar = helper_fns.select_crl_var(var, crl_data, correct_crl_data=correct_crl_data, fname=crlname[3:-11])
    
        # flatten variable
        rflat = np.tile(r_o, len(h)) #np.repeat(r, len(h))  
        hflat = np.repeat(h, len(r_o))
        varflat = crlvar.flatten()   
    
        # remove nan cases from all datasets! need to cycle through all three arrays
        dnan = np.where(~np.isnan(rflat))[0]
        dist_flat = rflat[dnan]
        hflat = hflat[dnan]
        varflat = varflat[dnan]
        # repeat for heights
        hnan = np.where(~np.isnan(hflat))[0]
        rflat = rflat[hnan]
        hflat = hflat[hnan]
        varflat = varflat[hnan]
        # finally, repeat for Temps
        tnan = np.where(~np.isnan(varflat))[0]
        rflat = rflat[tnan]
        hflat = hflat[tnan]
        varflat = varflat[tnan]
    
    
        # complete the crl compositing
        binned_crl = scipy.stats.binned_statistic_2d(rflat, hflat, varflat, statistic='mean', bins=(binr, binh))



    # part 2: coamps composite
        
    # variables to calculate manually
    if var=='T':
        r_cp = .286
        p0 = 1000 * 10**5
        theta = np.flip( coamps_data['pottmp'].values, axis=0)
        p =  np.flip(coamps_data.ttlprs.values, axis=0) * 10**5 # hpa to bars
        datavar = theta / (p0 / p)**r_cp - 273.        

    elif var=='thetae':
        # first, calculate T from theta
        r_cp = .286
        p0 = 1000 * 10**5
        theta = np.flip( coamps_data['pottmp'].values, axis=0)
        p =  np.flip(coamps_data.ttlprs.values, axis=0) * 10**5 # hpa to bars
        T = theta / (p0 / p)**r_cp * units.K
        p =  np.flip(coamps_data.ttlprs.values, axis=0) * units.hPa # redefine for metpy calcs below
        # find the coamps dewpoint
        spechum = np.flip(coamps_data.wvapor.values)
        mr = spechum/(1-spechum) # kg / kg        
        e =  mpcalc.vapor_pressure( p, mr)
        td = mpcalc.dewpoint(e)
        # finally, calculate theta e!
        datavar = mpcalc.equivalent_potential_temperature(p, T, td).magnitude
   
    elif var=='thetav':
        # first, calculate T from theta
        r_cp = .286
        p0 = 1000 * 10**5
        theta = np.flip( coamps_data['pottmp'].values, axis=0)
        p =  np.flip(coamps_data.ttlprs.values) * 10**5 # hpa to bars
        T = theta / (p0 / p)**r_cp * units.K
        p =  np.flip(coamps_data.ttlprs.values) * units.hPa # redefine for metpy calcs below
        # find the coamps dewpoint
        spechum = np.flip(coamps_data.wvapor.values)
        mr = spechum/(1-spechum) # kg / kg        
        datavar = mpcalc.virtual_potential_temperature(p, T, mr).magnitude
    elif var == 'pottmp':
        datavar = np.flip( coamps_data[var].values, axis=0) # flip to match sigma flip above        
    elif var == 'relhum':
        datavar = np.flip( coamps_data[var].values, axis=0) # flip to match sigma flip above        
        datavar[datavar>100.] = 100.        
    elif var == 'wvapor':
        datavar = np.flip( coamps_data[var].values, axis=0) # flip to match sigma flip above        
    
    # array shapes
    # build 1d height, 2d distance arrays into 3d matrices to match var of interest!
    l0, l1 = np.shape(r)[0], np.shape(r)[1]    
    allhs = np.tile(sigma, l0*l1).reshape((l0,l1,len(sigma)))
    allhs = np.moveaxis(allhs, [2], [0])
    allrs =  np.dstack([r]*len(sigma))
    allrs = np.moveaxis(allrs, [2], [0])        
    # collapse height and r values to 1d for histogram calculation
    rflat, hflat, varflat = allrs.flatten(), allhs.flatten(), datavar.flatten()    
    # complete the compositing and make a simple theta plot!

    binh = sigma
    binned_coamps = scipy.stats.binned_statistic_2d(rflat, hflat, varflat, statistic='mean', bins=(binr, binh))
  
    return binned_crl, binned_coamps





# given a coamps field with uneven sigma levels, convert to even crl height scales!
def change_heights(coamps_stat, coamps_x_edge, coamps_y_edge, crl_stat, crl_x_edge, crl_y_edge):
    step = crl_y_edge[1] - crl_y_edge[0] 
    maxh = crl_y_edge[-1] # make the max higher than the crl array to include as much coamps data as possible
    newh = np.arange(0,maxh,step) # new height array, based on crl array
    trimbin = coamps_stat
    lowis = np.where(coamps_y_edge <= maxh)[0] # remove coamps data above threshold
    trimbin = trimbin[:, lowis]
    trimy = coamps_y_edge[lowis]
    binned_interp = interp_binned_smooth(trimbin, trimy, newh)    
    return binned_interp


def plot(fig, ax0, ax1, ax2, coamps_name, coamps_folder, li, coamps_matrix, crl_matrix, x_edge, y_edge,
         var='pottmp', ytop=3., 
         namecases=[], correct_crl_data=False, coamps_curtain=False,
         xaxis='dist', coamps_rmw=0., tdr_rmw=0., date=''):
    # annoying, but need to reload coamps dataset to pull leadtime
    os.chdir(coamps_folder)
    full_coamps = xr.open_dataset(coamps_name)
    leadtime = full_coamps.leadtime.values
    
    if var=='wvapor':
        coamps_matrix = coamps_matrix * 1000.
    x, y = x_edge[:-1], y_edge[:-1] / 1000.
    
    
    # choose correct colormaps, etc
    nbins = 21
    fs = 12
    if var=='T':
        cmap = mpl.colormaps["RdYlBu"].reversed()
        varlabel = 'T (C)'
        xticks = [5., 10., 15., 20., 25., 30.]
        vmin, vmax = 5, 30
        width = 4.
        levels = np.linspace(vmin,vmax,21)
    elif var=='thetae':
        vmin,vmax = 330., 370. # 280
        levels = np.linspace(vmin,vmax,21)
        cmap = mpl.colormaps['RdYlBu'].reversed()
        varlabel = r'$\theta_{e}$ (K)'
        xticks = [330., 340., 350., 360., 370.]
        width=10.
    elif var=='thetav':
        cmap = mpl.colormaps['RdYlBu'].reversed()
        varlabel = r"$\theta_{v}$ (K)"
        levels = np.linspace(295.,325.,nbins)
        xticks = [295., 300., 305, 310, 315., 320., 325.]
        width=10. 
    elif var == 'pottmp':
        cmap = mpl.colormaps["RdYlBu"].reversed()
        varlabel=r'$\theta$ (K)'
        levels = np.linspace(295.,325.,nbins)
        xticks = [295., 300., 305, 310, 315., 320., 325.]
        width=10. 
    elif var == 'relhum':
        cmap = "Blues"
        varlabel='RH (%)'
        levels = np.linspace(50,100.,nbins)
        xticks = [50., 75., 100.]        
        width=20. 
    elif var == 'wvapor':
        cmap = 'viridis'
        levels = np.linspace(5.,25.,21)
        xticks = [5., 8., 11., 14., 17., 20., 23.]
        varlabel = 'WVMR (g/kg)'
        width=4. 
    aticks = [-width, 0., width]


    anom = coamps_matrix.transpose() - crl_matrix.transpose()

    # new: if xaxis='rmw', divide distance by rmw to create a normalized r* axis!
    if xaxis=='rmw':
        coampsx = x / coamps_rmw
        crlx = x / tdr_rmw

        # also need to re bin coamps coamps data to reflect rmw axis??
        # interpolate coamps data to match tdr scale

        # create interpolation function for each height level
        binned_coamps_interp = np.full_like(crl_matrix, np.nan)
        for hti in range(coamps_matrix.shape[1]):
            # get valid data points (no nans)
            valid_mask = ~np.isnan(coamps_matrix[:, hti])
            f_interp = interpolate.interp1d(coampsx[valid_mask], coamps_matrix[valid_mask, hti], 
                                            kind='linear', bounds_error=False, fill_value=np.nan)
            binned_coamps_interp[:, hti] = f_interp(crlx)

        # update variables to use interpolated data
        #binned_coamps = binned_coamps_interp
        #coampsx = tdrx
        anom = binned_coamps_interp.transpose() - crl_matrix.transpose()

    else:
        coampsx = x
        crlx = x

    # finally, make the plot here! both composite fields and anomaly field adjacent
    helper_fns.change_font_sizes(fs,fs)
    f = ax0.contourf(coampsx, y, coamps_matrix.transpose(), cmap=cmap, levels=levels, extend='both')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = plt.colorbar(f, label='COAMPS-TC ' +varlabel, ticks=xticks, cax=cax)
    
    ax0.set_ylabel('Height (km)')
    
    # crl plot
    f = ax1.contourf(crlx, y, crl_matrix.transpose(), cmap=cmap, levels=levels, extend='both')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(f, label='CRL ' +varlabel, ticks=xticks, cax=cax)
    


    # anomaly plot
    alevels = np.linspace(-width,width,20)
    f = ax2.contourf(crlx, y, anom, cmap='bwr', levels=alevels, extend='both')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(f, label='COAMPS-TC - CRL ' + varlabel, ticks=aticks, cax=cax)
    

    for ax in [ax0, ax1, ax2]:
        ax.set_ylim([0,ytop])
        ax.set_facecolor('k')
        if xaxis=='rmw':
            ax.set_xlabel('r*')
            if date=='0927':
                ax.set_xlim([0,15.])
            else:
                ax.set_xlim([0,10.])
            ax.axvline(x=1.,c='k',lw=.8)
        else:
            ax.set_xlabel('Radial Distance (km)')
            ax.set_xlim([0,300])