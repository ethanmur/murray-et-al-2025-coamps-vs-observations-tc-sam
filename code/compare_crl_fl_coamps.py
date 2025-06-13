import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import xarray as xr
import matplotlib.colors as mcol
import matplotlib.cm as cm
import cartopy.crs as ccrs          # For plotting maps
import cartopy.feature as cfeature  # For plotting maps
from metpy.units import units
import metpy.calc as mpcalc

basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns



# New function: plot both T and wvmr differences. Also, use different crl and drops
# datasets (more recently updated). are results generally the same as before?
def plot(manual_center=False, crlname='P3_20210927H1'):
    crl_path = basepath + "data/crl/"    
    tdr_names = ['P3_20210927H1_highres.nc']
    all_xlims = [[22.6, 24.]]

    metadata = helper_fns.obs_to_single_model_run(crlname)
    coamps_folder, coamps_name, crlfolder, nhcpath, nhcfile = metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], 
    coamps_bt_path, coamps_bt_name, nhcdate, nhci, li = metadata[5], metadata[6], metadata[7], metadata[8], metadata[9]


    for tdri, tdr_name in enumerate(tdr_names):        
        os.chdir(crl_path)
        data = xr.open_dataset(tdr_name)
        xaxis = data.time.values
        i0 = np.nanargmin(np.abs(data.time.values - all_xlims[tdri][0]))
        i1 = np.nanargmin(np.abs(data.time.values - all_xlims[tdri][1]))
        ox, oy = data.x_2m.values, data.y_2m.values

        # load coamps-tc data in .nc format
        os.chdir(coamps_folder)
        coamps = xr.open_dataset(coamps_name)
    
    
        coampsh = coamps.z_sig.values
        # leadtimes = coamps.leadtime.values
        # leadtime = leadtimes[li]
        leadtime = coamps.leadtime.values

        # new step: find respective xy axis for coamps data given lat lons and center fix!
        
        # optional: use manually inputed center, or one from coamps best track file
        if manual_center:
            
        
            clat, clon = helper_fns.manual_centers(crlname)                       
        else:
            ctrack_times, ctrack_lats, ctrack_lons = helper_fns.return_coamps_lat_lons(coamps_bt_path, coamps_bt_name)
            time_match_i = np.where(leadtime==ctrack_times)[0]
            clat, clon = ctrack_lats[time_match_i],  ctrack_lons[time_match_i]
        
        cx, cy = helper_fns.return_coamps_xy(coamps, clat, clon, li)

        # first, make coamps xy plot here!
        fig = plt.figure(figsize=(6,6))
                
        spechum = coamps.wvapor.values * 1000.
        cmap = 'viridis'
        varlabel = 'WVMR (g/kg)'
        vmin, vmax = 10, 20
        xticks = [0., 5., 10., 15., 20., 25.]
        width=2.5 # +- endpoints for anomaly plot
        
        leveli = -10
        plt.pcolormesh(cx, cy, spechum[leveli,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label=varlabel, ticks=xticks)
        plt.xlabel('X (Km)')
        plt.ylabel('Y (Km)')
        plt.xlim([-200,200])
        plt.ylim([-200,200])
        
        lw = 2.
        sz=60.
        fs=14
        plt.plot(ox[i0:i1],oy[i0:i1],c='k',linewidth=lw)
        plt.scatter(ox[i0],oy[i0],c='k',s=sz, marker='o')
        plt.scatter(ox[i1],oy[i1],c='k',s=sz, marker='*')
        # tit="TC Sam Inner Core, " + all_titles[tdri]
        # plt.title(tit)

        print(f'Start time: {data.time.values[i0:i1] [0]:.3f} UTC')
        print(f'End time: {data.time.values[i0:i1] [-1]:.3f} UTC')


        plt.title('Part of 20210927H1 Flight Path')
        ax=plt.gca()
        txt = plt.text( .07, .93, '(a)', fontsize=fs, transform=ax.transAxes)
        txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

        os.chdir(basepath + "figures/")
        plt.savefig('figure2_xy.png', dpi=300.)
        
        
        # return a list of xy location inds using a helper function
        valid_i, close_y, close_x = helper_fns.compare_coamps_obs_horiz(oy, ox, cy, cx, testtype='xy')  

        
        # next step: make curtain plots
    
        fig = plt.figure(figsize=(12, 12))
        helper_fns.change_font_sizes(fs,fs)
        plt.subplots_adjust(hspace=.25)
    
        # repeat plots for T and wvmr
        subplots=[(323, 321, 325),(324, 322, 326)]
        varnames=['T', 'wvmr']
        xlims = all_xlims[tdri]
        ylims = [0, 3200]
        # subplot_labels = [('(c)', '(b)', '(d)'), ('(f)', '(e)', '(g)')]
        subplot_labels = [('(d)', '(b)', '(f)'), ('(e)', '(c)', '(g)')]
        xtxt, ytxt = .8, 1.15
        xloc, yloc = .045, .85
        for vari, varname in enumerate(varnames):
            
            # crl T plot: show data coverage
            plt.subplot(subplots[vari][0])
            if varname=='T':
                min = 5
                max = 35
                map = plt.cm.get_cmap( "RdYlBu").reversed()
                crlvar = data.T.transpose()
                flvar = data.fl_T
                varlabel = "T (Degrees C)"
                # coamps T
                r_cp = .286
                p0 = 1000 * 10**5
                theta = coamps.pottmp.values
                p = coamps.ttlprs.values * 10**5 # hpa to bars
                coamps_var = theta / (p0 / p)**r_cp - 273. 
                
            elif varname=='wvmr':
                min = 5
                max = 20
                map = 'viridis' # plt.cm.get_cmap( "RdYlBu").reversed()
                crlvar = data.WVMR.transpose()
                flvar = data.fl_wv
                varlabel = 'WVMR (g/kg)'
                spechum = coamps.wvapor.values
                coamps_var = 1000 * (spechum/(1-spechum))
                
            plt.pcolormesh( xaxis, data.height.values, crlvar, vmin = min, vmax = max, cmap = map)
            plt.colorbar(label=varlabel)
            ax = plt.gca()
            ax.set_facecolor('k')
            plt.xlim(xlims)
            plt.ylim(ylims)
            if varname=='T':
                plt.ylabel("Height (m)")
            txt = plt.text( xloc, yloc, subplot_labels[vari][0], fontsize=fs, transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

            # plot (1) and (2) text boxes representing
            if varname=='T':
                txt = plt.text(.85, .1, '(1)', fontsize=fs, transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))
                txt = plt.text(.37, .15, '(2)', fontsize=fs, transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))
                txt = plt.text(.6, .15, '(2)', fontsize=fs, transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

        
            # fl, sonde T plot: show how coverage is much worse
        
            # add some heights above and below original heights to extend the flight level profile
            nudge = 50.
            fl_ht_orig = data.fl_ht.values
            fl_low = fl_ht_orig - nudge
            fl_high = fl_ht_orig + 1.
            fl_hts = np.vstack((fl_high, fl_ht_orig, fl_low))
        
            # repeat for fl vals and x axis to get into correct shape
            fl_xs = np.vstack((xaxis, xaxis, xaxis))
            fl_ts = np.vstack((flvar, flvar, flvar))

            # Add subplot title
            plt.title('CRL')
            

        
            plt.subplot(subplots[vari][1])
            plt.pcolormesh( fl_xs, fl_hts, fl_ts, vmin = min, vmax = max, cmap = map)
            plt.colorbar(label=varlabel)
            ax=plt.gca()
            if varname=='T':
                plt.ylabel("Height (m)")
            ax = plt.gca()
            ax.set_facecolor('k')
            plt.xlim(xlims)
            plt.ylim(ylims)
        
            # add text box denoting subplot letter
            ax=plt.gca()
            txt = plt.text( xloc, yloc, subplot_labels[vari][1], fontsize=fs, transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))
        
            # add dropsonde data to plot using tc-drops database (new analysis)

            # old path
            # os.chdir("/Users/etmu9498/research/data/tc-obs/")
            # drops = xr.open_dataset("tc-obs.nc")
            # new path
            os.chdir(basepath + "data/tc-drops/")
            drops = xr.open_dataset("tc-drops.nc")
            
            # find inds of dropsondes matching current flight id
            bnames=drops.flight_id.values[0]
            names=[]
            for bname in bnames:
                names.append(bname.decode("utf-8"))
            names=np.array(names)
            goodis = np.where(names==tdr_name[3:13])[0]
    
            # repeat these steps for all soundings
            for sondecount, sondei in enumerate(goodis):   
                day = drops.day.values[0][sondei]
                hours = drops.hour.values[0][sondei]
                # account for wraparound hours by checking dates
                if str(int(day)) != tdr_name[9:11]:
                    hours=hours+24.
                mins = drops.minute.values[0][sondei]
                secs = drops.second.values[0][sondei]
                time = float( hours) + float( mins) / 60 + float( secs) / 3600
                
                # once again, make a fancy time array with slightly nudged values
                z = drops.height.values[:, sondei]
                # # remove nan values from z
                z[z==-999.]=np.nan
                # only keep non nan values
                zi = np.where(~np.isnan(z))[0]
                z=z[zi]
                tcenter = time * np.ones(len(z))
                t_low = tcenter - .0025
                t_high = tcenter + .0025
                sonde_xs = np.vstack((t_low, tcenter, t_high))
    
                if varname=='T':
                    sondeval=drops.temperature.values[:,sondei]
                elif varname=='wvmr':
                    sondeval=drops.wvmr.values[:,sondei] * 1000.  
                sondeval[sondeval==-999.]=np.nan
                sondeval=sondeval[zi]    
                
                # stack sonde z and ts to get into same format
                sonde_hts = np.vstack((z, z, z))
                sonde_ts = np.vstack((sondeval, sondeval, sondeval))
                plt.pcolormesh( sonde_xs, sonde_hts, sonde_ts, vmin = min, vmax = max, cmap = map)
    
            plt.title('Dropsonde and Flight Level')

    
    
    
            # coamps T / wv plot
            plt.subplot(subplots[vari][2])
            plt.xlabel('Time (Hours, UTC)')

            # cycle through valid indices to build a coamps curtain, matching the crl x axis    
            new_var = []
            for vi, valid_pair in enumerate(valid_i):
                lati, loni = valid_pair[0] - 1, valid_pair[1] - 1
                # pull column moisture data at correct lat lon regions!
                new_col = coamps_var[:,loni,lati]
                if len(new_var) == 0:
                    new_var = new_col
                else:
                    new_var = np.vstack((new_var, new_col))
            
           # final step: interpolate coamps data to match crl shape vertically 
            coamps_curtain = helper_fns.interp_coamps_smooth(crlvar, new_var, data, coamps)
            ht = data.height.values
            
            plt.pcolormesh( xaxis[i0:i1], data.height.values, coamps_curtain[:,i0:i1], vmin=min, vmax=max, cmap=map)
            plt.colorbar(label=varlabel)
            ax = plt.gca()
            plt.xlim(xlims)
            plt.ylim(ylims)
            if varname=='T':
                plt.ylabel("Height (m)")
            txt = plt.text( xloc, yloc, subplot_labels[vari][2], fontsize=fs, transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

            plt.title('COAMPS-TC')
            
                
    
        os.chdir(basepath + "figures/")
        plt.savefig('figure2_cross_sections.png', dpi=300.)
