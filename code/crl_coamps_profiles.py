# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:58:30 2024

@author: etmu9498
"""



import os
import sys
import xarray as xr
import numpy as np
import scipy
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT
import cartopy.crs as ccrs          # For plotting maps
import cartopy.feature as cfeature  # For plotting maps
import matplotlib as mpl
basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns
        
        
# the same as the figure above, but for no dropsonde case! since dropsondes are compared
# to coamps points, not curtains, this comparison is a little unfair. so taken out of analysis.
def plot(manual_center=True):

    if manual_center:
        temppath = basepath + "data/corrected-anomalies-manual-center"        

    # variables to test
    for var, figlabel in zip(['theta', 'wv'], ['7', '8']):
        # make a figure containing all three subplots here!
        fs=12
        helper_fns.change_font_sizes(fs,fs)                
        fig = plt.figure(figsize=(12,4))
        subplot_list=[131, 132, 133]
        subplot_labels = ['(a)', '(b)', '(c)']
        
        if var=='theta' or var=='thetav' or var=='thetae':
            xloc, yloc = .08, .91
        elif var=='wv' or var=='T':
            xloc, yloc = .85, .91

        # sam cases to test
        for fi, fname in enumerate(['20210926H1', '20210927H1', '20210929H2']):
            cmap, varlabel, vmin, vmax, xticks, width = helper_fns.get_simple_presets(var)

            if var=='theta':
                vmin=290.
                vmax=320.

            # old code using .npy arrays
            # repeat for coamps - drops cases
            os.chdir(temppath + "/coamps-crl/")
            # coamps_matrix = np.load(var + "_" + fname + "_" + 'coampscols.npy')
            # obs_matrix = np.load(var + "_" + fname + "_" + 'crlcols.npy')
            # anom_matrix = np.load(var + "_" + fname + "_" + 'anom.npy')
            # times = np.load(var + "_" + fname + "_" + 't.npy')
            # h = np.load(var + "_" + fname + "_" + 'height.npy')
            # h = h / 1000.

            # new code using .nc files
            ncdata = xr.open_dataset(fname + '_anomalies.nc')
            coamps_matrix = ncdata[var + "_" + fname + "_" + 'coampscols'].values
            obs_matrix = ncdata[var + "_" + fname + "_" + 'crlcols'].values
            anom_matrix = ncdata[var + "_" + fname + "_" + 'anom'].values
            times = ncdata.time.values
            h = ncdata.height.values / 1000.

            # 3/11/25: new matrices have already been transposed?!?
            # 8/16/24 new: transpose obs matrix for code compatability!
            # obs_matrix = obs_matrix.transpose()
            # coamps_matrix = coamps_matrix.transpose()
            # anom_matrix = anom_matrix.transpose()
            
            # optional
            # remove crl data above a certain height! eyewall signal contaminating the 
            # mean above this height
            tallinds = np.where(h > 2.500)[0]
            obs_matrix[:, tallinds] = np.nan
            
            # only for water vapor, remove bottom 100m of data to account for turning errors in p-3
            if var=='wv':
                lowinds = np.where(h < .100)[0]
                obs_matrix[:, lowinds] = np.nan
            
            # use these crl eye limits
            # or use for drop plot creation: would include outlier warm anomaly sondes!
            if fname=='20210926H1':
                eyelims = [(22.55,22.693), (23.84, 23.91), (24.855,24.965)]
            elif fname=='20210927H1':
                eyelims = [(22.04, 22.10), (23.28,23.40), (24.48,24.59)]
            elif fname=='20210929H2':
                eyelims = [(21.54,21.68), (22.75,22.92), (23.97, 24.09), (25.65,25.86)]
            
            # calculate the temp anomaly OUTSIDE the given tc eyes!! pull all available data
            # outside of eye limits. only need to do this once for all tc eyes
            # cycle through eye limits: find time inds for eye regions
            eye_inds = np.array([]) # list of all valid eye indices for this day
            for eyei, lims in enumerate(eyelims):                    
                pass_inds = np.intersect1d(np.where(times>lims[0])[0], np.where(times<lims[1])[0])
                eye_inds = np.concatenate((eye_inds, pass_inds))
            
            out_inds = np.setdiff1d(np.arange(0,len(times),1), eye_inds)
            eye_inds = eye_inds.astype('int')            
            out_inds = out_inds.astype('int')

            # take the mean theta values outside of the eye here!
            outmean = np.nanmean(anom_matrix[out_inds, :])
            eyemean = np.nanmean(anom_matrix[eye_inds, :])
            anoms = anom_matrix[out_inds, :] # coamps_matrix[pass_inds, :] - obs_matrix[pass_inds, :]
                
            # finally, plot mean values here! outside of eye
            mean = np.round(np.nanmean(anoms), 2)
            outmean = np.round(outmean, 2)
            tit = fname + ', All Profiles Inside and Outside Eye' # : ' + str(outmean) + ' K'
            
            subplot = subplot_list[fi]
            plt.subplot(subplot)
            lw = .8
            ax1 = plt.gca()
            if subplot==131:
                ax1.set_ylabel("Height (km)")
            ax1.set_ylim([0,3.25])
            #ax1.set_xlim([295,325])
            ax1.set_xlim([vmin,vmax])
            ax1.set_xlabel(varlabel)

            ax1.plot(np.nanmean(obs_matrix[out_inds, :], axis=0), h, c='k', linewidth=lw*2, label='CRL, Not Eye')
            ax1.plot(np.nanmean(coamps_matrix[out_inds, :], axis=0), h, c='b', linewidth=lw*2, label='COAMPS, Not Eye') #  linestyle='--',
            ax1.set_title(fname)
            
            # finally, plot mean values here! inside of eye
            
            # remove lower data for 9/26: error points present!
            if fi==0 and var=='theta':
                data = np.nanmean(obs_matrix[eye_inds, :], axis=0)[75:]
                hplot = h[75:]
                print('remove value case\n\n\n\n')
            else:
                data = np.nanmean(obs_matrix[eye_inds, :], axis=0)
                hplot = h
            
            ax1.plot( data, hplot, c='k', alpha=.55, linewidth=lw*2, label='CRL, Eye')
            ax1.plot(np.nanmean(coamps_matrix[eye_inds, :], axis=0), h, c='skyblue', linewidth=lw*2, label='COAMPS, Eye') #  linestyle='--',
            
            subplot_label = subplot_labels[fi]
            txt = plt.text( xloc, yloc, subplot_label, fontsize=fs, transform=ax1.transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

            if fi==0 and var=='theta':
                ax1.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(0.025,.87), framealpha=1.)
            elif fi==0 and var=='wv':
                ax1.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(.425,.87), framealpha=1.)
               
        # save figure! only after adding dropsonde cases
        os.chdir(basepath + "figures/")
        savename = 'figure' + figlabel + '_' + var + '.png'    
        plt.savefig( savename, dpi=300., bbox_inches='tight')
