import numpy as np
import os
import matplotlib.pyplot as plt
import sys

basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import composite_thermo
import composite_winds
import helper_fns_2024 as helper_fns

def make_figs_9_10_11(manual_center=True, xaxis='dist'):
    correct=True
    coamps_curtain=True
    
    vars = ['wvapor', 'pottmp', 'thetae']
    figlabels = ['9', '10', '11']
    for var, figlabel in zip(vars, figlabels):
        # only make a new figure for new variables
        fig, axs = plt.subplots(3,3,figsize=(12,12))
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=.6, hspace=.3)
        crlflights = ["P3_20210926H1", "P3_20210927H1", "P3_20210929H2"]
        coamps_rmw = [20., 20., 35.]
        tdr_rmw = [10., 30., 30.]
        
        for crli, crlname in enumerate(crlflights):
            # metadata for single tc cases, aka match obs to closest coamps dataset.            
            metadata = helper_fns.obs_to_single_model_run(crlname)
            coamps_folder, coamps_name, crlfolder, nhcpath, nhcfile = metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], 
            coamps_bt_path, coamps_bt_name, nhcdate, nhci, li = metadata[5], metadata[6], metadata[7], metadata[8], metadata[9]
                
            crlname+='_highres.nc'
            crl, coamps = composite_thermo.prepare_data(coamps_folder, coamps_name, crlfolder, crlname, nhcpath, nhcfile,
                                                            coamps_bt_path, coamps_bt_name, nhcdate, nhci, li, var=var,
                                                            correct_crl_data=correct, manual_center=manual_center,
                                                            )
            # convert coamps data to crl y scale
            coamps_corrected = composite_thermo.change_heights(coamps.statistic, coamps.x_edge, coamps.y_edge, crl.statistic, crl.x_edge, crl.y_edge)
                
            # plot the results! overall composites and anomalies
            composite_thermo.plot(fig, axs[crli,0], axs[crli,1], axs[crli,2], coamps_name, coamps_folder, li, 
                                  coamps_corrected, crl.statistic, crl.x_edge, crl.y_edge, var=var, 
                                  correct_crl_data=correct, coamps_curtain=coamps_curtain,
                                  xaxis=xaxis, coamps_rmw=coamps_rmw[crli], tdr_rmw=tdr_rmw[crli],
                                  date=nhcdate)

            
        # add labels here
        fs = 12
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
        axlist = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2],
                  axs[2,0], axs[2,1], axs[2,2]]
        for si, subplot_label in enumerate(subplot_labels):
            xloc, yloc = .82, .9
            ax = axlist[si]
            txt = plt.text( xloc, yloc, subplot_label, fontsize=fs, transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

        # save final figures here!
        savetit = 'figure' + figlabel + '_' + var + '.png'
        spath = basepath + "figures/"
        os.chdir(spath)
        plt.savefig(savetit, dpi=300., bbox_inches='tight')    



def make_figs_4_5_6(manual_center=False, xaxis='dist'):
    #plot colored composites for all cases
    variables = ['radial', 'tangential', 'vertical']     
    testtypes = ['profile', 'profile', 'level3']   
    figlabels = ['4', '5', '6']
    # variables = ['tangential', 'vertical']     
    # testtypes = ['profile', 'level3']   
    # figlabels = ['5', '6']
    variables = ['vertical']     
    testtypes = ['profile']   
    figlabels = ['6']
    
    
    
    for var, testtype, figlabel in zip(variables, testtypes, figlabels):
        # only make a new figure for new variables
        fig, axs = plt.subplots(3,3,figsize=(12,12))
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=.6, hspace=.3)
        crlflights = ["20210926H1", "20210927H1", "20210929H2"]
        coamps_rmw = [20., 20., 35.]
        tdr_rmw = [10., 30., 30.]
    
        for crli, crlname in enumerate(crlflights):
            composite_winds.plot_composites(fig, axs[crli, 0], axs[crli, 1], axs[crli, 2], 
                                            tcname=crlname, var=var, coamps_curtain=True,
                                            testype=testtype, manual_center=manual_center, 
                                            xaxis=xaxis, coamps_rmw=coamps_rmw[crli], tdr_rmw=tdr_rmw[crli])
    
        # add labels here
        fs = 12
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
        axlist = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2],
                  axs[2,0], axs[2,1], axs[2,2]]
        for si, subplot_label in enumerate(subplot_labels):
            xloc, yloc = .82, .9
            ax = axlist[si]
            txt = plt.text( xloc, yloc, subplot_label, fontsize=fs, transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))
        axs[0,0].set_title('COAMPS-TC')
        axs[0,1].set_title('TDR')
        axs[0,2].set_title('COAMPS-TC - TDR Anomaly')

        #save final figure here!
        savetit = 'figure' + figlabel + '_' + var + '.png'
        spath = basepath + "figures/"
        os.chdir(spath)
        plt.savefig(savetit, dpi=300., bbox_inches='tight')    
