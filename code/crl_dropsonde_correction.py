import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import pandas as pd
import scipy
import sys

basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns



def plot_corrections():    
    # make the plot here
    fs=14
    lw=1.2
    helper_fns.change_font_sizes(fs,fs)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=.6, hspace=.3)

    # do this analysis for both water vapor and temperature
    vars = ['T', 'wv']
    tcnames = ["P3_20210926H1", "P3_20210927H1", "P3_20210929H2"]
    subplot_labels = [('(a)', '(b)', '(c)'), ('(d)', '(e)', '(f)')]
    xloc, yloc = .08, .88
    for vari, var in enumerate(vars):
        # pull plotting defaults from helper function
        color_map, levels, ex, ticks, vartitle = comparison_plot_defaults(var)
        
        
        # do this analysis for all three days
        for datei, tcname in enumerate(tcnames):
            metadata = helper_fns.obs_to_single_model_run(tcname)
            coamps_folder, coamps_name, crlfolder, nhcpath, nhcfile = metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], 
            coamps_bt_path, coamps_bt_name, nhcdate, nhci, li = metadata[5], metadata[6], metadata[7], metadata[8], metadata[9]

            # plot setup
    
            # access the correct subplot panel here
            ax = axs[vari,datei]
            start, end = levels[0], levels[-1]
            ax.set_xlim([start, end])
            ax.set_ylim([start, end])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if vari==0:
                ax.set_title('Flight ' + tcname[3:])
            ax.set_xlabel('Dropsonde ' + vartitle)
            ax.set_ylabel('CRL ' + vartitle)

            
            # add subplot label
            subplot_label = subplot_labels[vari][datei]
            txt = plt.text( xloc, yloc, subplot_label, fontsize=fs, transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

            # pull crl and dropsonde data
            crldata, dropdata = pull_data(var, tcname)
            
                
            # create 2D bin of data here
            bincount=40 # 50
            bins = np.linspace(start,end,bincount)
            hist2d = np.histogram2d(crldata, dropdata, bins=bins)[0]
            # remove nan data
            hist2d[hist2d==0.] = np.nan

            # normalize histogram data here!
            hist2d=hist2d / np.nansum(hist2d)

            # extra: pull stats for data comparisons
            var_mean, var_low, var_high, out_low, out_high, corrected_mean = extra_stats(crldata, dropdata, bins)
                        
            # create and save plot!
            vmin=10**-3 # 10**-4
            vmax=5*10**-2 # 10**-1
            f = ax.pcolormesh(bins,bins,hist2d,cmap=color_map, norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(f, cax=cax)
            # add one to one line
            ax.plot( np.arange(start,end), np.arange(start,end), c='k', linewidth=lw)            

            # add correction line!
            if vari==0:
                ax.plot( np.arange(start,end), np.arange(start,end) - corrected_mean, c='forestgreen', linewidth=lw)            
                # also add text showing shift values!
                shift_label = f'{corrected_mean:.2f}' + " C"
                txt = plt.text( .65, .07, shift_label, fontsize=fs, transform=ax.transAxes)
                #txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))
            elif vari==1:
                ax.plot( np.arange(start,end), np.arange(start,end) - corrected_mean, c='forestgreen', linewidth=lw)            
                # also add text showing shift values!
                shift_label = f'{corrected_mean:.2f}' + " g/kg"
                txt = plt.text( .55, .07, shift_label, fontsize=fs, transform=ax.transAxes)
                #txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))
    

 
    savename="figure3_crl_vs_dropsondes.png"
    os.chdir(basepath + "/figures/")
    plt.savefig(savename, dpi=300., bbox_inches='tight')


def pull_data_npy(var, date):
    os.chdir(basepath + "data/all-anomalies/drops-crl/")
    # build name of dataset to pull here!
    crlname = var + '_' + date[3:] + '_crlcols.npy'
    dropname = var + '_' + date[3:] + '_dropcols.npy'
    crldata = np.load(crlname).flatten()
    dropdata = np.load(dropname).flatten()
    
    
    return crldata, dropdata



def pull_data(var, date):
    os.chdir(basepath + "data/all-anomalies/drops-crl/")

    # new: load netcdf file for this date!
    data = xr.open_dataset(date[3:] + '_anomalies.nc')

    crlname = var + '_' + date[3:] + '_crlcols'
    dropname = var + '_' + date[3:] + '_dropcols'
    crldata = data[crlname].values.flatten()
    dropdata =  data[dropname].values.flatten()
    return crldata, dropdata


def extra_stats(allvar1, allvar2, bins, print_stats=True):
    # 7/8/24 new: for each coamps T, find the mean T val for the crl, and error metrics: IQR perhaps?
    var_mean, var_low, var_high, out_low, out_high = [], [], [], [], []
    bin_anom = []
    all_anoms = []
    var_std = []
    
    # cycle through bins first
    for vari in range(len(bins)-1):
        bin_i = bins[vari]
        bin_i_plus_1 = bins[vari+1]
        
        # pull inds where coamps data fall within bins
        coampsi = np.intersect1d( np.where(allvar1 > bin_i), np.where(allvar1 <= bin_i_plus_1))

        # calculate mean, std crl temp at this valid bin
        var_mean.append(np.nanmean(allvar2[coampsi]))
        var_low.append(np.nanpercentile(allvar2[coampsi], 25))
        var_high.append(np.nanpercentile(allvar2[coampsi], 75))
        out_low.append(np.nanpercentile(allvar2[coampsi], 5))
        out_high.append(np.nanpercentile(allvar2[coampsi], 95))
        
        # var_std = 
        
        # calculate the mean anomaly between the expected coamps val and the mean crl val
        # used for corrections below!
        center_bin = (bin_i_plus_1 + bin_i)/2 # centered value to compare to obs 2
        anom = np.nanmean(allvar2[coampsi]) - center_bin
        bin_anom.append(anom)
          
        # method 2: do var1 - var2 for all cases within bins. save temporarily
        all_anoms += (allvar2[coampsi] - allvar1[coampsi]).tolist()
                        
        
    # calculate the "corrected mean" aka the single value used to shift
    # dist up or down. weight every bin equally? not sure if that's a good idea
    # tbh haha, but its a good try for now
    # corrected_mean = np.nanmean(np.array(bin_anom))
    # new way: average across all points, not a mean of means!
    corrected_mean = np.nanmean(np.array(all_anoms))   
    corrected_std = np.nanstd(np.array(all_anoms))   
    
    if print_stats:
        print('Mean var Anomaly = ' + str(corrected_mean))
        print('Corrected Mean = ' + str(corrected_mean))
        print('STD var Anomaly = ' + str(corrected_std))
    
    return var_mean, var_low, var_high, out_low, out_high, corrected_mean
    
    
def comparison_plot_defaults(varname='wvmr'):
    if varname=='wv':             
        levels = np.arange(0,30,.25)
        ticks=[0, 10, 20, 30]
        vartitle='WVMR (g/kg)'
    elif varname=='T':             
        vartitle='T (C)'
        levels = np.linspace(0,35,14)
        ticks=[0, 10, 20, 30]
    ex = 'both'
    color_map = mpl.colormaps['RdYlBu'].reversed() 
    return color_map, levels, ex, ticks, vartitle