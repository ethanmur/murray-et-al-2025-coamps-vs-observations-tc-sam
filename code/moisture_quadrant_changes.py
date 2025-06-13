# the same as v1, but able to cycle through variables, dates, etc more efficiently!
# create more plots to better understand data anomalies

# setup
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import matplotlib.patches as mpatches

basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
os.chdir(basepath + "code/")
import helper_fns_2024 as helper_fns


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta)


def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)


def find_shear_dists_one_case(x, y, sheardir):
    r, theta = cart2pol(np.array(x), np.array(y))
    theta2 = theta + np.radians(sheardir)
    xnew, ynew = pol2cart(r, theta2)
    return xnew, ynew
    


# correct: correct crl T data using dropsondes
def single_legs_plot(correct=True, fname='20210927H1', var='wv', quad='ur-dl', 
                     manual_center=False):
    
    if correct:
        correction_label = 'corrected'
    else:
        correction_label = 'all'

    # use flight name and selected shear quadrants to trim data
    if fname == "20210926H1":
        sheardir = 64.  # shear direction angle from southerly. aka 0 points from south to north, 90 points from west to east
        # UL-DR tests
        if quad=='ul-dr':
            range1 = [875, 1575]
            range2 = [5050, 6150]
            quad1, quad2 = 'DR', 'UL'
        # UR-DL tests
        if quad=='ur-dl':
            range1 = [2300, 3050]
            range2 = [3050, 3775]
            quad1, quad2 = 'DL', 'UR'
    elif fname == "20210927H1":
        sheardir = 44.  
        if quad=='ul-ur':
            # original ranges: 1-> ul, 2->ur
            range1 = [0,1050]
            range2 = [3250,4100]
            quad1, quad2 = 'UL', 'UR'
        elif quad=='ur-dl':
            # new ranges: 1-> dl, 2->ur
            range1 = [2450, 3250]
            range2 = [3250, 4100]
            quad1, quad2 = 'DL', 'UR'
    elif fname == "20210929H2":
        sheardir = 57.
        if quad=='ur-dl':
            # UR-DL tests
            range1 = [7550, 8425]
            range2 =[8625, 9400]
            quad1, quad2 = 'UR', 'DL'
        elif quad=='ul-dr':
            # UL-DR tests
            range1 = [2150, 2900]
            range2 =[3000, 3800]
            quad1, quad2 = 'UL', 'DR'

    cmap, varlabel, vmin, vmax, xticks, width = helper_fns.get_simple_presets(var)

    # load original crl dataset and make lat lon plots to figure out shear quadrant
    # being sampled!
    os.chdir(basepath + 'data/crl/')
    crl = xr.open_dataset('P3_' + fname + '_highres.nc')


    # load pre saved data matrices / anomalies here
    if manual_center:
        anomaly_path = basepath + "data/corrected-anomalies-manual-center/coamps-crl/"
    os.chdir(anomaly_path)
    ncdata = xr.open_dataset(fname + '_anomalies.nc')
    coamps_matrix = ncdata[var + "_" + fname + "_" + 'coampscols'].values.transpose()
    crl_matrix = ncdata[var + "_" + fname + "_" + 'crlcols'].values.transpose()
    h = ncdata.height.values / 1000.

    # coamps_matrix = np.load(var + "_" + fname + "_" + 'coampscols.npy')
    # crl_matrix = np.load(var + "_" + fname + "_" + 'crlcols.npy')
    # anom_matrix = np.load(var + "_" + fname + "_" + 'anom.npy')
    # times = np.load(var + "_" + fname + "_" + 't.npy')
    # h = np.load(var + "_" + fname + "_" + 'height.npy')
    # h = h / 1000.

    # making the figure
    fs = 14
    helper_fns.change_font_sizes(fs, fs)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=.4, hspace=.25)
    helper_fns.change_font_sizes(fs, fs)

    axs[0, 0].set_ylabel("Height (km)")
    axs[1, 0].set_ylabel("Height (km)")
    axs[2, 0].set_ylabel("Height (km)")
    axs[2, 0].set_xlabel('Radial Distance (km)')
    axs[2, 1].set_xlabel('Radial Distance (km)')
    axs[1, 2].set_xlabel('Radial Distance (km)')

    #axs[2, 2].axis('off')  # turn off corner subplot

    # add storm name as title
    plt.text(x=.2, y=1.2, s=fname, fontsize=fs + 4, transform=axs[0, 1].transAxes)

    axs[1, 0].set_facecolor('k')
    axs[1, 1].set_facecolor('k')
    axs[1, 2].set_facecolor('k')
    axs[2, 0].set_facecolor('k')
    axs[2, 1].set_facecolor('k')

    axs[0, 0].set_title('COAMPS, ' + quad1)
    axs[0, 1].set_title('COAMPS, ' + quad2)
    axs[0, 2].set_title('COAMPS, ' + quad1 + ' - ' + quad2 + ' Anomaly')

    axs[1, 0].set_title('CRL, ' + quad1)
    axs[1, 1].set_title('CRL, ' + quad2)
    axs[1, 2].set_title('CRL, ' + quad1 + ' - ' + quad2 + ' Anomaly')

    axs[2, 0].set_title('COAMPS - CRL, ' + quad1)
    axs[2, 1].set_title('COAMPS - CRL, ' + quad2)

    

    # 10/4/24 new: add a lil p-3 flight map in the bottom right corner of the figure!
    correction_label = 'corrected'
    if fname == "20210926H1":
        sheardir = 64.
    elif fname == "20210927H1":
        sheardir = 44.  
    elif fname == "20210929H2":
        sheardir = 57.
      
  

    for axi in range(np.shape(axs)[0]):
        for axj in range(np.shape(axs)[1]):
            if axi!=2 or axj!=2:
                axs[axi, axj].set_ylim([0, 3.250])

    # crl plots
    r = (crl.x_2m**2 + crl.y_2m**2)**.5
    p = axs[1, 0].pcolormesh(r[range1[0]: range1[1]], h, crl_matrix[:,
                             range1[0]: range1[1]], vmin=vmin, vmax=vmax, cmap=cmap)
    p2 = axs[1, 1].pcolormesh(r[range2[0]: range2[1]], h, crl_matrix[:,
                              range2[0]: range2[1]], vmin=vmin, vmax=vmax, cmap=cmap)

    # add colorbars nicely!
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p, cax=cax, orientation='vertical', label=varlabel)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p2, cax=cax, orientation='vertical', label=varlabel)

    # coamps tc plots
    p = axs[0, 0].pcolormesh(r[range1[0]: range1[1]], h, coamps_matrix[:,
                             range1[0]: range1[1]], vmin=vmin, vmax=vmax, cmap=cmap)
    p2 = axs[0, 1].pcolormesh(r[range2[0]: range2[1]], h, coamps_matrix[:,
                              range2[0]: range2[1]], vmin=vmin, vmax=vmax, cmap=cmap)

    # add colorbars nicely!
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p, cax=cax, orientation='vertical', label=varlabel)
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p2, cax=cax, orientation='vertical', label=varlabel)

    # coamps - coamps anomalies, crl - crl anomalies
    # challenging: need to interpolate datasets to a common grid, then subtract
    # because they can be different sizes!

    # find furthest radius between two datasets

    # maybe limit all plots to the largest x value?

    # do this for crl datasets first
    bigr1, bigr2 = np.nanmax(r[range1[0]:range1[1]]), np.nanmax(
        r[range2[0]:range2[1]])
    bigr = np.max([bigr1, bigr2])
    # round value to fit nicely with interpolated axis
    bigend = np.ceil(bigr)
    # incrementing by 250m (? closest to crl xy resolution) intervals, create a new distance array!
    rinterp = np.arange(0., bigend, .250)

    crlnew1 = np.empty([len(h), len(rinterp)])
    crlnew2 = np.empty([len(h), len(rinterp)])
    crlnew1[:] = np.nan
    crlnew2[:] = np.nan
    crl1 = crl_matrix[:, range1[0]: range1[1]]
    crl2 = crl_matrix[:, range2[0]: range2[1]]

    # cycle through and inerpolate dataset 1
    # go through dataset row by row
    for rowi in range(len(h)):
        crlrow = crl1[rowi, :]
        # interpolation code taken from auto-analysis/auto_coamps_crl_drops_curtain.py
        f = interpolate.interp1d(
            r[range1[0]:range1[1]], crlrow, fill_value='extrapolate')
        crlnew1[rowi, :] = f(rinterp)

    # repeat for dataset 2
    for rowi in range(len(h)):
        crlrow = crl2[rowi, :]
        f = interpolate.interp1d(
            r[range2[0]:range2[1]], crlrow, fill_value='extrapolate')
        crlnew2[rowi, :] = f(rinterp)

    # for shorter dataset, get rid of fake values at extended radial distances!
    if bigr1 <= bigend:
        highris = np.where(bigr1 <= rinterp)[0]  # indices for large r values
        # turn these values into nans!
        crlnew1[:, highris] = np.nan
    if bigr2 <= bigend:
        highris = np.where(bigr2 <= rinterp)[0]  # indices for large r values
        crlnew2[:, highris] = np.nan

    # repeat for coamps
    # rinterp, bigr, etc all stay the same from before!
    coampsnew1 = np.empty([len(h), len(rinterp)])
    coampsnew2 = np.empty([len(h), len(rinterp)])
    coampsnew1[:] = np.nan
    coampsnew2[:] = np.nan
    crl1 = coamps_matrix[:, range1[0]: range1[1]]
    crl2 = coamps_matrix[:, range2[0]: range2[1]]

    # cycle through and inerpolate dataset 1
    # go through dataset row by row
    for rowi in range(len(h)):
        crlrow = crl1[rowi, :]
        # interpolation code taken from auto-analysis/auto_coamps_crl_drops_curtain.py
        f = interpolate.interp1d(
            r[range1[0]:range1[1]], crlrow, fill_value='extrapolate')
        coampsnew1[rowi, :] = f(rinterp)
    # repeat for dataset 2
    for rowi in range(len(h)):
        crlrow = crl2[rowi, :]
        f = interpolate.interp1d(
            r[range2[0]:range2[1]], crlrow, fill_value='extrapolate')
        coampsnew2[rowi, :] = f(rinterp)

    if bigr1 <= bigend:
        highris = np.where(bigr1 <= rinterp)[0]
        coampsnew1[:, highris] = np.nan
    if bigr2 <= bigend:
        highris = np.where(bigr2 <= rinterp)[0]
        coampsnew2[:, highris] = np.nan

    # finally, make the plots here!
    step = 6
    anom1 = crlnew1 - crlnew2
    anom2 = coampsnew1 - coampsnew2
    p = axs[1, 2].pcolormesh(rinterp, h, anom1, vmin=-
                             step, vmax=step, cmap='bwr')
    p2 = axs[0, 2].pcolormesh(
        rinterp, h, anom2, vmin=-step, vmax=step, cmap='bwr')

    # add colorbars nicely!
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p, cax=cax, orientation='vertical',
                 label=varlabel[:-6] + ' Anomaly ' + varlabel[-6:])
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p2, cax=cax, orientation='vertical',
                 label=varlabel[:-6] + ' Anomaly ' + varlabel[-6:])

    # coamps - crl anomalies: easier because datasets are from same region, have
    # same shapes!
    step = 6
    anom1 = coamps_matrix[:, range1[0]: range1[1]] - \
        crl_matrix[:, range1[0]: range1[1]]
    anom2 = coamps_matrix[:, range2[0]: range2[1]] - \
        crl_matrix[:, range2[0]: range2[1]]
    p = axs[2, 0].pcolormesh(r[range1[0]: range1[1]],
                             h, anom1, vmin=-step, vmax=step, cmap='bwr')
    p2 = axs[2, 1].pcolormesh(r[range2[0]: range2[1]],
                              h, anom2, vmin=-step, vmax=step, cmap='bwr')

    # add colorbars nicely!
    divider = make_axes_locatable(axs[2, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p, cax=cax, orientation='vertical',
                 label=varlabel[:-6] + ' Anomaly ' + varlabel[-6:])
    divider = make_axes_locatable(axs[2, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p2, cax=cax, orientation='vertical',
                 label=varlabel[:-6] + ' Anomaly ' + varlabel[-6:])



    # repeat for shear corrected distances!
    ax = axs[2,2]
    lw = 4.
    shearx, sheary = find_shear_dists_one_case(
        crl.x_2m.values, crl.y_2m.values, sheardir)
    ax.plot(shearx, sheary, c='k', linewidth=lw)
    ax.plot(shearx[range1[0]:range1[1]],
             sheary[range1[0]:range1[1]], c='r', label='Leg 1', linewidth=lw)
    ax.plot(shearx[range2[0]:range2[1]],
             sheary[range2[0]:range2[1]], c='c', label='Leg 2', linewidth=lw)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    
    # add a vertical arrow here to denote environmental shear direction
    arrow_mag = .2
    shear_dir = 90    
    x_tail = 0.15
    y_tail = 0.15
    arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_tail, .45),
                         mutation_scale=40,
                         transform=ax.transAxes, color='k')
    ax.add_patch(arrow)
    
    
    # add subplot labels here
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    axlist = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2],
              axs[2,0], axs[2,1], axs[2,2]]
    for si, subplot_label in enumerate(subplot_labels):
        if si==8:
            xloc, yloc = .07, .85
        else:
            xloc, yloc = .8, .85
            
        ax = axlist[si]
        txt = plt.text( xloc, yloc, subplot_label, fontsize=fs, transform=ax.transAxes)
        txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))


    savepath = basepath + "figures/"
    os.chdir(savepath)
    plt.savefig('figure12.png', dpi=300., bbox_inches='tight')


