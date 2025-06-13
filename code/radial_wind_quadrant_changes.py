# setup
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import sys
import os
from scipy.interpolate import RegularGridInterpolator
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


# helper function to pull tdr data along a coamps flight curtain!
# code taken from "/code/secondary-circulation/tdr_level3_make_composites.py"
def pull_coamps_wind_curtain(coamps_folder, coamps_name, crlfolder, crlname, coamps_bt_path, 
                   coamps_bt_name, ti, var='radial', test_curtains=False, manual_center=False):
    
    # load coamps track lat lons for all leadtimes
    track_times, track_lats, track_lons = helper_fns.return_coamps_lat_lons(coamps_bt_path, coamps_bt_name)
    
    # load coamps-tc data in .nc format
    os.chdir(coamps_folder)
    coamps_data = xr.open_dataset(coamps_name)
    sigma = coamps_data.z_sig.values
    # leadtimes = coamps_data.leadtime.values
    # leadtime = leadtimes[ti]
    leadtime = coamps_data.leadtime.values

    # load crl data to figure out track sampled by p-3!
    os.chdir(crlfolder)
    crl_data = xr.open_dataset(crlname)

    # also load tdr dataset to get size for comparisons
    # all vars, dates have the same size, so just load one example
    os.chdir(basepath + "data/tdr-profile-composites/")
    tdrxname, tdryname = "20210927H1-x.npy", "20210927H1-y.npy"
    tdrx, tdry = np.load(tdrxname), np.load(tdryname)

    # Pull track center corresponding to selected leadtime, and find radial distances
    time_match_i = np.where(leadtime==track_times)[0]
    lat, lon = coamps_data.lat.values, coamps_data.lon.values
    
    if manual_center:
        clat, clon = helper_fns.manual_centers(crlname[:-11])                               
    else: 
        clat, clon = track_lats[time_match_i], track_lons[time_match_i]   
    cx, cy = helper_fns.return_coamps_xy(coamps_data, clat, clon, ti)
    r = (cx**2 + cy**2)**.5 # simple radius calculation

    # find crl distances next
    ox, oy = crl_data.x_2m.values, crl_data.y_2m.values
    r_o = (ox**2 + oy**2)**.5 # simple radius calculation
    

    # part 2: coamps composite
    if var=='vertical':             
        # load special coamps w file data here! also use special sigma levels
        os.chdir(basepath + "data/coamps-tc-w/")
        data_w = xr.open_dataset(coamps_name[:-3] + "_verticalwind.nc")
        sigma = data_w.z_sig.values 
        # datavar = data_w.wwwind.values[ti,:,:,:]
        datavar = data_w.wwwind.values
    
        color_map = 'bwr'
        levels = np.arange(-4,4,.25)
        ex = 'both'
        ticks=[-4., -2., 0., 2., 4.] 

    elif var=='radial':             
        # load special coamps w file data here! also use special sigma levels
        os.chdir(basepath + "data/coamps-tc-radwnd/")
        data_vr = xr.open_dataset(coamps_name[:-3] + "_radwnd.nc")       
        #sigma = np.flip(coamps_data.z_sig.values) 
        # datavar = data_vr.radwnd.values[ti,:,:,:]
        datavar = data_vr.radwnd.values
    
        color_map = 'bwr'
        levels = np.linspace(-15,15,14) # 13 before
        ticks=[-15., -7.5, 0., 7.5, 15.]
        ex = 'both'
    elif var=='tangential':             
        # load special coamps w file data here! also use special sigma levels
        os.chdir(basepath + "data/coamps-tc-tanwnd/")
        data_vt = xr.open_dataset(coamps_name[:-3] + "_tanwnd.nc")       
        #sigma = np.flip(coamps_data.z_sig.values) 
        # datavar = data_vt.tanwnd.values[ti,:,:,:]
        datavar = data_vt.tanwnd.values
   
        color_map = 'plasma'
        levels = np.arange(0,70,5)
        ex = 'max'
        ticks=[0., 20, 40., 60.]
  
    # 8/7/24 new: coamps_curtain=True case
    # basically, create a curtain of coamps data mimicking crl sampling, and return
    # all values from that curtain! instead of all uninterpolated coamps data.
    # code taken from auto_coamps_crl_drops_curtain.py
    
    # find closest coamps xy location indices to crl flight... for curtain analysis
    valid_i, close_y, close_x = helper_fns.compare_coamps_obs_horiz(oy, ox, cy, cx, testtype='xy')   
    new_var = []
    for vi, valid_pair in enumerate(valid_i):
        lati, loni = valid_pair[0] - 1, valid_pair[1] - 1
        # pull column data at correct lat lon regions!
        new_col = datavar[:,loni,lati]
        if len(new_var) == 0:
            new_var = new_col
        else:
            new_var = np.vstack((new_var, new_col))
    tdrvar = np.empty((len(tdrx), len(tdry))) # temp defined tdr var just to get its size! 

    # final step: interpolate coamps data to match tdr shape vertically
    # smooth the data too to avoid jumps between height levels     
    # using the closest local height inds determined above, cycle through crl 
    # rows and cols, saving closest coamps var in crl sized array        
    coamps_curtain = np.empty((np.shape(new_var)[0], len(tdry)))

    for xi in range(np.shape(new_var)[0]):
        f = interpolate.interp1d(sigma, new_var[xi,:], fill_value='extrapolate')
        coamps_curtain[xi,:] = f(tdry)           

    return r_o, tdry, coamps_curtain



def pull_tdr_profile(profilename, var='radial'):    
    # load profile data
    os.chdir("/Users/etmu9498/all-research/tc-rainbands/data/tdr-all-profiles-xy")
    profile = xr.open_dataset(profilename)
    
    z = profile.height.values
    r = profile.radius.values
    
    # REFLECTIVITY, Wind_Speed, Vertical_wind, Tangential_Wind, Radial_wind
    if var=='radial':
        tdr_curtain = profile.Radial_wind.values[:,0,:,0]
    elif var=='vertical':
        tdr_curtain = profile.Vertical_wind.values[:,0,:,0]
    elif var=='tangential':
        tdr_curtain = profile.Tangential_Wind.values[:,0,:,0]
    return tdr_curtain, z, r


# helper function to regrid coamps data to 
def regrid_coamps_tests(coampsvar, coampsr, coampsh, tdrvar, tdrr, tdrh): 
    target_r = np.linspace(coampsr.min(), coampsr.max(), 167)  # Target grid with 167 points
    target_h = np.linspace(coampsh.min(), coampsh.max(), 121)  # Target grid with 121 points
    target_r_grid, target_h_grid = np.meshgrid(target_r, target_h)
    
    # Combine the target grid into a single array of points
    target_points = np.array([target_r_grid.ravel(), target_h_grid.ravel()]).T 
    
    # Create the interpolator
    f = RegularGridInterpolator((coampsr, coampsh), coampsvar.T)
    
    # Interpolate the data to the target grid
    regrid_data = f(target_points)
        
    # Reshape the interpolated data to match the target grid shape
    regrid_data = regrid_data.reshape(target_r_grid.shape)  # Shape (121, 167)  
    return regrid_data, tdrr, tdrh




# helper function to regrid coamps data to 
def regrid_coamps(coampsvar, coampsr, coampsh, tdrvar, tdrr, tdrh):    
    target_r = np.linspace(coampsr.min(), coampsr.max(), 167)  # Target grid with 167 points
    target_h = np.linspace(coampsh.min(), coampsh.max(), 121)  # Target grid with 121 points
    target_r_grid, target_h_grid = np.meshgrid(target_r, target_h)
    
    # Combine the target grid into a single array of points
    target_points = np.array([target_r_grid.ravel(), target_h_grid.ravel()]).T 
    
    # Create the interpolator
    f = RegularGridInterpolator((coampsr, coampsh), coampsvar.T)
    
    # Interpolate the data to the target grid
    regrid_data = f(target_points)
    
    # Reshape the interpolated data to match the target grid shape
    regrid_data = regrid_data.reshape(target_r_grid.shape)  # Shape (121, 167)

    return regrid_data, tdrr, tdrh


# same as v1 code, but use tdr profile data for composite, not level 3 swath! probably more
# detail in abl

# correct: correct crl T data using dropsondes
def single_legs_plot(correct=True, fname='20210927H1', quad='ur-dl', 
                     manual_center=False, windvar='radial'):
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
            # range1 = [2450, 3250]
            # range2 = [3250, 4100]
            quad1, quad2 = 'DL', 'UR'
            range1 = [2450, 3250]
            range2 = [3265, 4080] # limit ranges to get rid of overlap cases
    
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

    # pull plotting info
    cmap, varlabel, vmin, vmax, xticks, width = helper_fns.get_simple_presets(windvar)
    if windvar=='radial':
        alevels = np.linspace(-10., 10.1, 20)
        aticks = np.array([-10, -5, 0, 5, 10])
    if windvar=='tangential':
        alevels = np.linspace(-20., 20.1, 40)
        aticks = np.array([-20, -10, 0, 10, 20])
    if windvar=='vertical':
        alevels = np.linspace(-2., 2.1, 20)
        aticks = np.array([-2, -1, 0, 1, 2])
    alabel = 'Wind Anomaly (m/s)'


    # load original crl dataset and make lat lon plots to figure out shear quadrant
    # being sampled!
    basepathlocal = "/Users/etmu9498/all-research/nrl-summer-2024-local/"
    os.chdir(basepathlocal + 'data-crl/')
    crl = xr.open_dataset('P3_' + fname + '_highres.nc')

    var='wv'

    # metadata for coamps data analysis
    crlname = 'P3_' + fname
    metadata = helper_fns.obs_to_single_model_run(crlname)
    coamps_folder, coamps_name, crlfolder, nhcpath, nhcfile = metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], 
    coamps_bt_path, coamps_bt_name, nhcdate, nhci, li = metadata[5], metadata[6], metadata[7], metadata[8], metadata[9]
    crlname = 'P3_' + fname + '_highres.nc'
    time, h, coamps_matrix = pull_coamps_wind_curtain(coamps_folder, coamps_name, crlfolder, crlname, coamps_bt_path, 
                       coamps_bt_name, li, var=windvar, test_curtains=True, manual_center=False)
    coamps_matrix=coamps_matrix.transpose()
    h = h/1000.

    # pull level 2 tdr profile data here!
    # 210927H1_2320_vert_inbound
    tdr_inbound, tdrh, tdrr = pull_tdr_profile('210927H1_2320_vert_inbound.nc', var=windvar)
    tdr_outbound, tdrh, tdrr = pull_tdr_profile('210927H1_2320_vert_outbound.nc', var=windvar)



    # making the figure
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fs = 14
    helper_fns.change_font_sizes(fs, fs)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=.4, hspace=.25)
    axs[0, 0].set_ylabel("Height (km)")
    axs[1, 0].set_ylabel("Height (km)")

    # axs[2, 0].set_ylabel("Height (km)")
    # axs[2, 0].set_xlabel('Radial Distance (km)')
    # axs[2, 1].set_xlabel('Radial Distance (km)')
    # axs[2, 0].axis('off')
    # axs[2, 1].axis('off')

    # add storm name as title
    # plt.text(x=.2, y=1.2, s=fname, fontsize=fs + 4, transform=axs[0, 1].transAxes)
   

    axs[0, 0].set_title('COAMPS, ' + quad1)
    axs[0, 1].set_title('COAMPS, ' + quad2)

    axs[1, 0].set_title('TDR, ' + quad1)
    axs[1, 1].set_title('TDR, ' + quad2)
    axs[1, 0].set_xlabel('Radial Distance (km)')
    axs[1, 1].set_xlabel('Radial Distance (km)')
          
    for axi in range(np.shape(axs)[0]):
        for axj in range(np.shape(axs)[1]):
            if axi!=2 or axj!=2:
                axs[axi, axj].set_ylim([0, 6.]) # 3.250])

    # tdr plots
    r = (crl.x_2m**2 + crl.y_2m**2)**.5
    levels = np.linspace(vmin,vmax,20)
    p = axs[1, 0].contourf(tdrr, tdrh, tdr_inbound.transpose(), levels=levels, cmap=cmap, extend='both')
    p2 = axs[1, 1].contourf(tdrr, tdrh, tdr_outbound.transpose(), levels=levels, cmap=cmap, extend='both')

    # add colorbars nicely!
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p, cax=cax, orientation='vertical', label=varlabel, ticks=xticks)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p2, cax=cax, orientation='vertical', label=varlabel, ticks=xticks)


    # coamps tc plots
    cmap, varlabel, vmin, vmax, xticks, width = helper_fns.get_simple_presets(windvar)
    levels = np.linspace(vmin,vmax,20)
    p = axs[0, 0].contourf(r[range1[0]: range1[1]], h, coamps_matrix[:,
                             range1[0]: range1[1]], levels=levels, cmap=cmap, extend='both')
    p2 = axs[0, 1].contourf(r[range2[0]: range2[1]], h, coamps_matrix[:,
                              range2[0]: range2[1]], levels=levels, cmap=cmap, extend='both')

    # add colorbars nicely!
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p, cax=cax, orientation='vertical', label=varlabel, ticks=xticks)
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p2, cax=cax, orientation='vertical', label=varlabel, ticks=xticks)

    
    # add subplot labels here
    subplot_labels = ['(a)', '(b)', '(c)', '(d)'] # , '(e)', '(f)']
    axlist = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]] #,
              # axs[2,0], axs[2,1]]
    for si, subplot_label in enumerate(subplot_labels):
        if si==8:
            xloc, yloc = .07, .85
        else:
            xloc, yloc = .8, .85
            
        ax = axlist[si]
        txt = plt.text( xloc, yloc, subplot_label, fontsize=fs, transform=ax.transAxes)
        txt.set_bbox(dict(facecolor='w', alpha=1.0, edgecolor='k', linewidth=1.5))

    basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
    savepath = basepath + "figures/"
    os.chdir(savepath)
    plt.savefig('figure13.png', dpi=300., bbox_inches='tight')

