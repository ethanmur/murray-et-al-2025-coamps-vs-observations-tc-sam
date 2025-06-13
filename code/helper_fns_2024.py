import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from metpy.units import units
import metpy.calc as mpcalc
import warnings
import os
import xarray as xr
import numpy as np
import scipy
from scipy import interpolate
import pandas as pd
from geopy import distance


def change_font_sizes(small=14, medium=18 ):
    """this quick function updates the sizes of fonts in plots, decreasing code clutter.
    it only applies the font changes if called before figure / axis creation.

    param small: change font sizes to this smaller value.
    param medium: change font sizes to this larger value.
    return: none
    """
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize



def radwnd2d(nx, ny, ic, jc, u, v):
    """
    convert u and v winds to radial and tangential fields.
    Args:
        nx (int): Grid point indices in x-direction.
        ny (int): Grid point indices in y-direction.
        ic (int): Storm center x-index.
        jc (int): Storm center y-index.
        u (np.ndarray): 2D array of u-component wind.
        v (np.ndarray): 2D array of v-component wind.
    Returns:n
        np.ndarray: 2D array of radial wind (u_rad).
        np.ndarray: 2D array of tangential wind (u_tan).
    """
    pi = np.pi
    u_rad = np.empty(np.shape(u))
    u_tan = np.empty(np.shape(u))

    # cycle through ny indices    
    for j in range(len(ny)):
        jrel = j - jc
        # cycle through nx indices    
        for i in range(len(nx)):
            irel = i - ic
            
            # Goal: Compute Azimuth Angle (in radians)
            # edge cases: between quadrants
            if irel >= 0 and jrel == 0:
                az = 0.0
            elif irel < 0 and jrel == 0:
                az = pi
            elif irel == 0 and jrel > 0:
                az = pi / 2.0
            elif irel == 0 and jrel < 0:
                az = 3.0 * pi / 2.0
                
            # grid point falls within quadrant
            else:
                # relative fraction between i and j (after center correction)
                # -> used to determine azimuth
                frac = jrel / irel

                if irel > 0 and jrel > 0:  # QUAD 1
                    az = np.arctan(np.abs(frac))
                elif irel < 0 and jrel > 0:  # QUAD 2
                    az = pi - np.arctan(np.abs(frac))
                elif irel < 0 and jrel < 0:  # QUAD 3
                    az = pi + np.arctan(np.abs(frac))
                elif irel > 0 and jrel < 0:  # QUAD 4
                    az = 2.0 * pi - np.arctan(np.abs(frac))
            
            # Compute radial & tangential wind
            u_rad[i, j] = u[i, j] * np.cos(az) + v[i, j] * np.sin(az)  # positive outward
            u_tan[i, j] = -u[i, j] * np.sin(az) + v[i, j] * np.cos(az)  # positive ccw
    
    return u_rad, u_tan


# compare_coamps_obs: given a crl (or other obs) flight track / lat lon points,
#     and coamps lat lon grid, find index of closest lat lons for each obs.
# inputs: olats, olons. Observational latitudes and longitudes. must be same length.
#     clats, clons. coamps grid of lat lons.
# return: valid_i. an array of tuples the length of olats. each tuple specifies:
#     (valid_i_lat, valid_i_lon) in coamps lat lon grids.
#     close_lats, close_lons. the actual latitude and longitude values that are
#     closest to the provided observation locations.
def compare_coamps_obs_horiz(olats, olons, clats, clons, testtype='latlon'):
    # save valid index pairs, lats / lons here
    valid_i = []
    close_lats = []
    close_lons = []
        
    # only need to view one column / row version of lats + lons. They repeat frequently
    clats, clons = clats[0,:], clons[:,0]
    
    # IMPORTANT NOTE: only do this conversion if working with lat lons, not distances!!
    if testtype=='latlon':
        # turn coamps lats into negative coords
        clons = clons - 360.
    elif testtype=='xy':
        pass
    
    # cycle through all lat lon pairs
    for obs_i, olon in enumerate(olons):
        # current lat lon pair
        olat = olats[obs_i]
        lati = np.nanargmin((clats-olat)**2)
        loni = np.nanargmin((clons-olon)**2)
        
        # append valid inds to return list as a tuple
        valid_i.append( (lati,loni))
        # also append actual lat lon values
        close_lats.append(clats[lati])
        close_lons.append(clons[loni])
    return valid_i, close_lats, close_lons



basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"


# metadata for a single flight -> a single model run
# input: crl name. drop the "_hires" and "_processed.nc" postfixes, too complicated!

# given a crl dataset, select a coamps dataset, flight, and time indices here
# variable specifications:
# coamps_folder, crl_folder -> folders holding model and obs data
# coamps_name, crlname -> dataset names
# nhcdate -> the date used to pull the nhc best track information. 
#       sometime different than the dataset nammes due to wraparound times (0 UTC cases)
# nhci -> at the selected nhc best track date, the index of data to pull. Goes up
#       in 6h increments, so nhci=2 -> 12 utc
# li -> leadtime index, aka the time in the coamps best track time series to pull
#       center info. li=3 -> 18 UTC.
# nhcpath, nhcfile -> name and path of best track dataset for obs
# coamps_bt_path, coamps_bt_file -> name and path of best track dataset for model
def obs_to_single_model_run(crlname):
    # always the same
    coamps_folder = basepath + "data/coamps/"
    crlfolder = basepath + "data/crl/"
    
    # tc sam
    if crlname=="P3_20210926H1": 
        coamps_name='CTCX.al182021.2021092612.nc'
        nhcdate='0927'
        nhci=0 
        li=2 # 0926 12utc to 0927 0utc
        tcname='sam'
        nhcfile="bal182021.dat"
        coamps_bt_name = "TRK_COAMPS_CTCX_3_2021092612_18L"    
    elif crlname=="P3_20210927H1":
        coamps_name='CTCX.al182021.2021092712.nc'
        nhcdate='0928'
        nhci=0 
        li=2
        tcname='sam'
        nhcfile="bal182021.dat"
        coamps_bt_name = "TRK_COAMPS_CTCX_3_2021092712_18L"
    elif crlname=="P3_20210929H2":
        coamps_name='CTCX.al182021.2021092818.nc'
        nhcdate='0930'
        nhci=0 
        li=5 # 0928 18utc to 0930 0utc
        tcname='sam'
        nhcfile="bal182021.dat"
        coamps_bt_name = "TRK_COAMPS_CTCX_3_2021092818_18L"

    nhcpath=basepath + "data/best-tracks/"
    coamps_bt_path = basepath + "data/best-tracks/coamps/"

    return [coamps_folder, coamps_name, crlfolder, nhcpath, nhcfile, coamps_bt_path, coamps_bt_name, nhcdate, nhci, li]


def manual_centers(crlname):
    if crlname=='P3_20210926H1':
        clat, clon = 14.8, -51.225
    elif crlname=='P3_20210927H1':
        clat, clon = 16.54, -52.76
    elif crlname=='P3_20210929H2':
        clat, clon = 19.94, -58.1
    return clat, clon



def return_coamps_lat_lons(path_coamps = basepath + "/data/best-tracks/coamps/",
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
    lats_raw, lons_raw = track_c['Lat'].values, track_c['Lon'].values
    lats, lons = [], []
    for li, lat in enumerate(lats_raw):
        lon = lons_raw[li]
        # account for lons > 100 W
        lats.append(float(lat[0:3]) / 10.)
        if len(lon) == 5:
            lons.append(-float(lon[0:3]) / 10.)
        elif len(lon) == 6:
            lons.append(-float(lon[0:4]) / 10.)

    # append float columns back to pandas dataframe
    track_c['Lat Float'] = lats
    track_c['Lon Float'] = lons

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

    return tracks_c_unique['Lead Time Float'].values, tracks_c_unique['Lat Float'].values, tracks_c_unique['Lon Float'].values

        
def return_nhc_lat_lons(path_nhc= basepath + "/data/best-tracks/", file_nhc = "bal182021.dat"):
    track_nhc = pd.read_fwf( path_nhc + file_nhc, header=None)
    track_nhc = track_nhc.drop(track_nhc.columns[35:41], axis=1)
    
    # turn lats / lons into a readable format
    lats_raw, lons_raw = track_nhc[6].values, track_nhc[7].values
    lats, lons = [], []
    for li, lat in enumerate(lats_raw):
        lon = lons_raw[li]
        lats.append(float(lat[0:3]) / 10.)        
        # account for cases >= 100 degrees lon (4 decimals needed)    
        if len(lon) == 5:
            lons.append(-float(lon[0:3]) / 10.)
        elif len(lon) == 6:
            lons.append(-float(lon[0:4]) / 10.)

    # also figure out dates and times. 
    dates, times = [], []
    dts = track_nhc[2].values
    for datetime in dts:
        dates.append(datetime[4:8])
        times.append(float(datetime[8:10]))
    
    # get inds of unique and repeat datetimes (from 3 wind radii)
    uniquei = np.unique(dts, return_index=True)[1]    
    return np.array(dates)[uniquei], np.array(times)[uniquei], np.array(lats)[uniquei], np.array(lons)[uniquei]
    
    
# return x and y distance fields from the tc center, when provided with coamps data and a lat / lon center fix
# x and y distances can be used to calculate overall radial distance from the TC center.
def return_coamps_xy(data, clat, clon, ti):
    lats, lons = data.lat.values, data.lon.values
    
    # lat lon values repeat across rows and columns. only keep single rows / cols to speed up analysis.
    rowlats, rowlons = lats[0,:], lons[:,0]
    
    # turn coamps lats into negative coords
    rowlons = rowlons - 360.

    # find ind of nearest lat lon to center fix. 
    lati = np.nanargmin((rowlats-clat)**2)
    loni = np.nanargmin((rowlons-clon)**2)
    
    # make - side of distance array. append to + side
    xlim = -4 * loni # furthest x distance
    xminus = np.arange(xlim,0.,4.)
    xlim = 4 * ( len(rowlons) - loni )
    xplus = np.arange(0,xlim,4.)
    x = np.concatenate((xminus, xplus))
    
    # repeat for y vals with lat column
    ylim = -4 * lati # furthest x distance
    yminus = np.arange(ylim,0.,4.)
    ylim = 4 * ( len(rowlats) - lati )
    yplus = np.arange(0,ylim,4.)
    y = np.concatenate((yminus,yplus))
    
    fully = np.tile(y, (226,1))
    
    # fullx = np.repeat(x, 226, axis=1)
    fullx = np.tile(x, (226,1))
    fullx = fullx.transpose()
    
    return fullx, fully


def flatten_tdr(coamps_folder, coamps_name, crlfolder, crlname, nhcpath, nhcfile, coamps_bt_path, 
                   coamps_bt_name, nhcdate, nhci, ti, var='pottmp', correct_crl_data=False, dtype='crl', 
                   coamps_curtain=False, smooth_coamps=True, test_curtains=False, manual_center=False):
        
    # load coamps track lat lons for all leadtimes
    track_times, track_lats, track_lons = return_coamps_lat_lons(coamps_bt_path, coamps_bt_name)
    # load nhc track data
    nhc_dates, nhc_times, nhc_lats, nhc_lons = return_nhc_lat_lons(nhcpath, nhcfile)
    
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
    lat, lon = coamps_data.lat.values[:,:], coamps_data.lon.values[:,:]
    
    if manual_center:
        clat, clon = manual_centers(crlname[:-11])                               
    else: 
        clat, clon = track_lats[time_match_i], track_lons[time_match_i]   
    cx, cy = return_coamps_xy(coamps_data, clat, clon, ti)
    r = (cx**2 + cy**2)**.5 # simple radius calculation

    # find crl distances next
    ox, oy = crl_data.x_2m.values, crl_data.y_2m.values
    r_o = (ox**2 + oy**2)**.5 # simple radius calculation
    

    # part 2: coamps composite
    if var=='wwwind':             
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

    elif var=='radwnd':             
        # load special coamps w file data here! also use special sigma levels
        os.chdir(basepath + "data/coamps-tc-radwnd/")
        data_vr = xr.open_dataset(coamps_name[:-3] + "_radwnd.nc")       
        #sigma = np.flip(coamps_data.z_sig.values) 
        
        #datavar = data_vr.radwnd.values[ti,:,:,:]
        datavar = data_vr.radwnd.values
    
        color_map = 'bwr'
        levels = np.linspace(-15,15,14) # 13 before
        ticks=[-15., -7.5, 0., 7.5, 15.]
        ex = 'both'
    elif var=='tanwnd':             
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
    if coamps_curtain:
        print('curtain case')
        
        # find closest coamps xy location indices to crl flight... for curtain analysis
        valid_i, close_y, close_x = compare_coamps_obs_horiz(oy, ox, cy, cx, testtype='xy')   
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
          
        h = tdry        
        rflat = np.tile(r_o, len(h)) 
        hflat = np.repeat(h, len(r_o))
        varflat = coamps_curtain.flatten('F')   

    # sample full coamps region, not just along p-3 flight path
    else:
        # array shapes
        # build 1d height, 2d distance arrays into 3d matrices to match var of interest!
        l0, l1 = np.shape(r)[0], np.shape(r)[1]    
        allhs = np.tile(sigma, l0*l1).reshape((l0,l1,len(sigma)))
        allhs = np.moveaxis(allhs, [2], [0])
        allrs =  np.dstack([r]*len(sigma))
        allrs = np.moveaxis(allrs, [2], [0])        
        # collapse height and r values to 1d for histogram calculation
        rflat, hflat, varflat = allrs.flatten(), allhs.flatten(), datavar.flatten()    

    return rflat, hflat, varflat





# given a tc best track lat / lon field, calculate
# we need to calculate this manually because speed and headings aren't provided
# in the coamps best track data.
# inputs: pt1, pt2: the starting and ending tc center points in (lat, lon) format
#         timestep, the time change between both points (in hours)
# return: tcspeed, the speed of the tc (m/s)
#         tcdir, the direction of TC motion ()
#         u, v, the zonal and meridional speed components of the tc motion 
def find_tc_motion(pt1, pt2, timestep):
    
    
    # pull out lats and lons
    # c -> first point
    # o -> second point
    olat, olon = pt2[0], pt2[1]
    clat, clon = pt1[0], pt1[1]
    
    x, y = [], []
    
    # current lat lon pair
    # hold either lats or lons constant to get xy dists
    coords_x = np.ma.masked_invalid((clat,olon))
    coords_y = np.ma.masked_invalid((olat,clon))
            
    # find and save distance to observation!
    
    dist_tot = distance.geodesic(pt1, pt2).km
    
    # find x and y components of distances here
    # account for - distances too
    if olon > clon:        
        x = distance.geodesic(coords_x, pt1).km
    else:
        x = -distance.geodesic(coords_x, pt1).km
    if olat > clat:
        y = distance.geodesic(coords_y, pt1).km
    else:
        y = -distance.geodesic(coords_y, pt1).km

    
    # calculate tc heading angles here. tc motion to the east -> 0 degrees
    # edge cases
    if x>=0 and y==0:        
        tcdir = 0.
    elif x<0 and y==0:
        tcdir = 180.
    elif x==0 and y>0:
        tcdir = 90.
    elif x==0 and y<0:
        tcdir = 270.
    else:
        # within quadrant cases
        # note that the y/x or x/y term changes! depending on quadrant.
        # draw out the angles to see why
        if x>0 and y>0:        
            tcdir = np.degrees(np.arctan(np.abs(y/x)))
        elif x<0 and y>0:
            tcdir = 90. + np.degrees(np.arctan(np.abs(x/y)))
        elif x<0 and y<0:
            tcdir = 180.  + np.degrees(np.arctan(np.abs(y/x)))
        elif x>0 and y<0:
            tcdir = 270. + np.degrees(np.arctan(np.abs(x/y)))

    # find speeds from total distances, u, v components
    # distances in m
    tot_m, x_m, y_m = dist_tot * 1000., x * 1000., y * 1000.
    # time elapsed in seconds
    step_s = timestep * 3600.
    
    # calculate the speeds here!
    tcspeed = tot_m / step_s
    u = x_m / step_s
    v = y_m / step_s

    return tcspeed, tcdir, u, v



# find pressure p2 at different heights throughout the column using the hypsometric
# equation (wallace and hobbes eqn 3.26)
def find_pressure(crl_data):
    # local variables used to calculate column pressure
    temp = crl_data.T.values
    psurf = crl_data.fl_psurf.values
    crlh = crl_data.height.values

    # hold pressure values
    new_matrix=np.empty(np.shape(temp))

    # cycle through pressure columns    
    for xi in range(len(temp)):
                
        # get the surface pressure at the closest timestep
        p1 = psurf[ xi]
        # surface height = 0m
        z1 = 0.
        r_g0 = 29.3 # gas constant / grav constant
        H = r_g0 * (temp[xi,:] + 273.15) # scale height (for one T / p column)
    
        p2 = p1 * np.exp(-crlh / H)
        new_matrix[ xi, :] = p2
    
    return new_matrix



# provide pressure optionally to save calculation time!
def find_rh( crl_data, p=[], correct_crl_data=[]):
    if len(p)==0:
        p = find_pressure( crl_data)
    else:
        pass

    if len(correct_crl_data)==0:
        temp = crl_data.T.values * units.degC
    elif len(correct_crl_data)==1:
        temp = (crl_data.T.values - correct_crl_data[0]) * units.degC
    mr = crl_data.WVMR.values / 1000
    
    # save rh values here
    new_matrix=np.empty([np.size(p, 0), np.size(p, 1)])
    warnings.filterwarnings("ignore")
    
    # do this for every height row
    for rowi in range( np.shape(p)[1]):
        # find the relative humidity for this height row
        rh_row = mpcalc.relative_humidity_from_mixing_ratio( p[:, rowi] * units.hPa, temp[:, rowi], mr[:, rowi]).to('percent')
        
        # add the new height row!
        new_matrix[:, rowi] = rh_row
        
    # ensure values stop at 100% rh
    new_matrix[new_matrix > 100.] = 100.
        
    warnings.filterwarnings("default")
    return new_matrix



def find_thetav(crl_data, p=[], correct_crl_data=[]):
    # setup
    if len(p)==0:
        p = find_pressure( crl_data)
    else:
        pass
    p = p * units.hPa

    if len(correct_crl_data)==0:
        temp = crl_data.T.values * units.degC
    elif len(correct_crl_data)==1:
        temp = (crl_data.T.values - correct_crl_data[0]) * units.degC
    
    mr = crl_data.WVMR.values / 1000 # kg/kg
    # calculate potential temperature!!! woo! use metpy for accuracy
    thetav = mpcalc.virtual_potential_temperature(p, temp, mr)

    return thetav




# provide pressure optionally to save calculation time!
def find_theta(crl_data, p=[], correct_crl_data=[]):
    # setup
    if len(p)==0:
        p = find_pressure( crl_data)
    else:
        pass
    
    if len(correct_crl_data)==0:
        temp = crl_data.T.values
    elif len(correct_crl_data)==1:
        temp = crl_data.T.values - correct_crl_data[0]

    # calculate potential temperature!!! woo!!
    p0 = 1000
    r_cp = .286
    theta = (temp + 273.15 ) * ( p0 / p)**( r_cp)
    return theta



def find_thetae(crl_data, p=[], correct_crl_data=[]):
    # setup
    if len(p)==0:
        p = find_pressure( crl_data)
    else:
        pass
    p = p * units.hPa
    
    if len(correct_crl_data)==0:
        T = crl_data.T.values * units.degC
    elif len(correct_crl_data)==1:
        T = (crl_data.T.values - correct_crl_data[0]) * units.degC
    
    mr = crl_data.WVMR.values / 1000 # g/g (unitless)
    e =  mpcalc.vapor_pressure( p, mr)
    td = mpcalc.dewpoint(e)
    # finally, calculate theta e!
    thetae = mpcalc.equivalent_potential_temperature(p, T, td)
    return thetae


# 7/8/24 new: option to correct crl data by a set amount!
def select_crl_var(var, crl, correct_crl_data=False, fname=''):
    if var=='wv':
        crl_var = crl.WVMR.values.transpose()
        
        
    elif var=='T':
        if correct_crl_data:            
            # use helper function to find correction amount
            allvar1, allvar2, bins = prepare_correction(fname=fname, var='T')
            correction = correct_crl(allvar1, allvar2, bins)            
            #print('correction = ' + str(np.round(correction, 3)))
            
            crl_var = crl.T.values.transpose() - correction
        else:
            crl_var = crl.T.values.transpose()            
    
    elif var=='rh':
        if correct_crl_data:            
            # use helper function to find correction amount
            allvar1, allvar2, bins = prepare_correction(fname=fname, var='T')
            correction = correct_crl(allvar1, allvar2, bins)
            #print('correction = ' + str(np.round(correction, 3)))

            p = find_pressure(crl)
            rh = find_rh(crl, p, correct_crl_data=[correction])
            crl_var = rh.transpose()
        else:
            p = find_pressure(crl)
            rh = find_rh(crl, p, correct_crl_data=[])
            crl_var = rh.transpose()
            
    elif var=='theta':
        if correct_crl_data:            
            # use helper function to find correction amount
            allvar1, allvar2, bins = prepare_correction(fname=fname, var='T')
            correction = correct_crl(allvar1, allvar2, bins)
            #print('correction = ' + str(np.round(correction, 3)))
           
            # use temperature correction here!
            p = find_pressure(crl)
            theta = find_theta(crl, p, correct_crl_data=[correction])
            crl_var = theta.transpose()
        else:
            p = find_pressure(crl)
            theta = find_theta(crl, p, correct_crl_data=[])
            crl_var = theta.transpose()
        
    elif var=='thetae':
        if correct_crl_data:            
            # use helper function to find correction amount
            allvar1, allvar2, bins = prepare_correction(fname=fname, var='T')
            correction = correct_crl(allvar1, allvar2, bins)
            #print('correction = ' + str(np.round(correction, 3)))

            p = find_pressure(crl)
            theta = find_thetae(crl, p, correct_crl_data=[correction])
            crl_var = theta.magnitude.transpose()
        else:
            p = find_pressure(crl)
            theta = find_thetae(crl, p, correct_crl_data=[])
            crl_var = theta.magnitude.transpose()

    elif var=='thetav':
        if correct_crl_data:            
            # use helper function to find correction amount
            allvar1, allvar2, bins = prepare_correction(fname=fname, var='T')
            correction = correct_crl(allvar1, allvar2, bins)
            #print('correction = ' + str(np.round(correction, 3)))

            p = find_pressure(crl)
            thetav = find_thetav(crl, p, correct_crl_data=[correction])
            crl_var = thetav.magnitude.transpose()
        else:
            p = find_pressure(crl)
            theta = find_thetav(crl, p, correct_crl_data=[])
            crl_var = theta.magnitude.transpose()

    return crl_var

    
# actual vertical interpolation of coamps columns!
def interp_coamps_smooth(crl_var, coamps_var, crl, coamps):
    
    coamps_interp = np.empty(np.shape(crl_var))    
    crlh = crl.height.values
    coampsh = coamps.z_sig.values
    coamps_hinds = np.empty(len(crlh))
    
    # go through all crl heights
    # find the index in the coamps height column closest to the current crlh
    # and save it locally.
    for hti, val in enumerate(crlh):
        hi = np.nanargmin(np.abs(coampsh-val))
        coamps_hinds[hti] = hi
    coamps_hinds = np.array(coamps_hinds).astype(int)        

    # using the closest local height inds determined above, cycle through crl 
    # rows and cols, saving closest coamps var in crl sized array        
    for xi, val in enumerate(crl.time.values):
        f = interpolate.interp1d(coampsh, coamps_var[xi,:], fill_value='extrapolate')
        coamps_interp[:,xi] = f(crlh)
    return coamps_interp


# same as the function above, but with no coamps data processing and slightly different var limits + colormaps
def get_simple_presets(var):
    # also set up labels, scale, etc for below
    if var=='wv':
        cmap = 'viridis'
        varlabel = 'WVMR (g/kg)'
        vmin, vmax = 5, 25
        xticks = [5., 10., 15., 20., 25.]
        
        #vmin, vmax = 0, 25
        #xticks = [0., 5., 10., 15., 20., 25.]
        
        width=2.5 # +- endpoints for anomaly plot
    elif var=='T':
        cmap = mpl.colormaps["RdYlBu"].reversed()
        varlabel = 'T (C)'
        vmin, vmax = 10, 30
        xticks = [10., 15., 20., 25, 30.]
        # vmin, vmax = 0, 25
        # xticks = [5., 10., 15., 20.]
        width = 4.
    elif var=='theta':
        cmap = mpl.colormaps["RdYlBu"].reversed()
        varlabel = 'Theta (K)'       
        vmin,vmax = 295., 320.
        levels = np.linspace(vmin,vmax,24)        
        xticks = [280., 300., 320.]
        width=10.
    elif var=='thetae':
        vmin,vmax = 330., 380. # 280
        levels = np.linspace(vmin,vmax,21)
        cmap = mpl.colormaps['RdYlBu'].reversed()
        varlabel = 'Theta E (K)'
        xticks = [330., 355., 380.]
        width=10.
    elif var=='thetav':
        vmin,vmax = 290., 320.
        levels = np.linspace(vmin,vmax,24)
        xticks = [280., 300., 320.]
        cmap = mpl.colormaps['RdYlBu'].reversed()
        varlabel = 'Theta V (K)'
        width=10.
    elif var == 'rh':
        cmap = "Blues"
        varlabel='RH (%)'
        xticks = [0., 25., 50., 75., 100.]
        width = 10
        vmin, vmax = 0, 100
    elif var == 'radial':
        cmap = "bwr"
        varlabel='Radial Wind (m/s)'
        xticks = [-20., -10., 0., 10., 20.]
        width = 10
        vmin, vmax = -20, 20
    elif var == 'vertical':
        cmap = "bwr"
        varlabel='Vertical Wind (m/s)'
        xticks = [-2., -1., 0., 1., 2.]
        width = 1
        vmin, vmax = -2, 2
    elif var == 'tangential':
        cmap = "plasma"
        varlabel='Tangential Wind (m/s)'
        xticks = [0., 20., 40., 60., 80.]
        width = 10
        vmin, vmax = 0, 80
                
    return cmap, varlabel, vmin, vmax, xticks, width




# use this wrapper function to load and preprocess the correct histogram data before
# crl correction.
def prepare_correction_npy(testtype='drop-crl', fname='20210929H2', var='T', nbins=50):
    # only need to pull specified max and min values for array here
    cmap, varlabel, vmin, vmax, xticks, width = get_simple_presets(var)
    
    
    # just using coamps-crl correction for now, but can just as easily correct using sondes, too!
    if testtype=='drop-crl':
        # load the relevant crl, drop, and anomaly datasets
        os.chdir(basepath + "data/all-anomalies/drops-crl/")
        var1_matrix = np.load(var + "_" + fname + "_" + 'dropcols.npy')
        var2_matrix = np.load(var + "_" + fname + "_" + 'crlcols.npy')
        xlab = "Sonde " + varlabel
        ylab = "CRL " + varlabel
        
        anom_matrix = np.load(var + "_" + fname + "_" + 'anom.npy')
        times = np.load(var + "_" + fname + "_" + 't.npy')
        h = np.load(var + "_" + fname + "_" + 'height.npy')
    elif testtype=='coamps-drop':
        # repeat for coamps - drops cases
        os.chdir(basepath + "data/all-anomalies/coamps-drops/")
        var1_matrix = np.load(var + "_" + fname + "_" + 'dropcols.npy')
        var2_matrix = np.load(var + "_" + fname + "_" + 'coampscols.npy')
        xlab = "Sonde " + varlabel
        ylab = "COAMPS " + varlabel
        
        anom_matrix = np.load(var + "_" + fname + "_" + 'anom.npy')
        times = np.load(var + "_" + fname + "_" + 't.npy')
        h = np.load(var + "_" + fname + "_" + 'height.npy')
    elif testtype=='coamps-crl':
        # repeat for coamps - drops cases
        os.chdir(basepath + "data/all-anomalies/coamps-crl/")
        var1_matrix = np.load(var + "_" + fname + "_" + 'coampscols.npy')
        var2_matrix = np.load(var + "_" + fname + "_" + 'crlcols.npy')
        xlab = "COAMPS " + varlabel
        ylab = "CRL " + varlabel
        
        anom_matrix = np.load(var + "_" + fname + "_" + 'anom.npy')
        times = np.load(var + "_" + fname + "_" + 't.npy')
        h = np.load(var + "_" + fname + "_" + 'height.npy')
    
    # flatten dropsonde, coamps, crl matrices for one to one plot here
    allvar1, allvar2, allanom = var1_matrix.flatten(), var2_matrix.flatten(), anom_matrix.flatten()
    
    
    if testtype=='coamps-crl':
        allvar1, allvar2 = allvar1[::100], allvar2[::100]
        
    bins = np.linspace(vmin,vmax,nbins)
    return allvar1, allvar2, bins


# use this wrapper function to load and preprocess the correct histogram data before
# crl correction.
def prepare_correction(testtype='drop-crl', fname='20210929H2', var='T', nbins=50):
    # only need to pull specified max and min values for array here
    cmap, varlabel, vmin, vmax, xticks, width = get_simple_presets(var)
    
    
    # just using coamps-crl correction for now, but can just as easily correct using sondes, too!
    if testtype=='drop-crl':
        # load the relevant crl, drop, and anomaly datasets
        os.chdir(basepath + "data/all-anomalies/drops-crl/")
        ncdata = xr.open_dataset(fname + '_anomalies.nc')
        var1_matrix = ncdata[var + "_" + fname + "_" + 'dropcols'].values
        var2_matrix = ncdata[var + "_" + fname + "_" + 'crlcols'].values
        anom_matrix = ncdata[var + "_" + fname + "_" + 'anom'].values
   
    # elif testtype=='coamps-crl':
    #     # repeat for coamps - drops cases
    #     os.chdir(basepath + "data/all-anomalies/coamps-crl/") 
    #     ncdata = xr.open_dataset(fname + '_anomalies.nc')
    #     var1_matrix = ncdata[var + "_" + fname + "_" + 'coampscols'].values
    #     var2_matrix = ncdata[var + "_" + fname + "_" + 'crlcols'].values
    #     anom_matrix = ncdata[var + "_" + fname + "_" + 'anom'].values

    # flatten dropsonde, coamps, crl matrices for one to one plot here
    allvar1, allvar2, allanom = var1_matrix.flatten(), var2_matrix.flatten(), anom_matrix.flatten()
    
    
    # if testtype=='coamps-crl':
    #     allvar1, allvar2 = allvar1[::100], allvar2[::100]
        
    bins = np.linspace(vmin,vmax,nbins)
    return allvar1, allvar2, bins



def correct_crl(allvar1, allvar2, bins):
    hist2d = np.histogram2d(allvar2, allvar1, bins=bins)[0]
    
    # for each coamps T, find the mean T val for the crl, and anomaly metrics
    var_mean = [] 
    bin_anom = []
    all_anoms = []
    # cycle through bins first
    for vari in range(len(bins)-1):
        bin_i = bins[vari]
        bin_i_plus_1 = bins[vari+1]
        
        # pull inds where coamps data fall within bins
        coampsi = np.intersect1d( np.where(allvar1 > bin_i), np.where(allvar1 <= bin_i_plus_1))

        # calculate mean crl temp at this valid bin
        var_mean.append(np.nanmean(allvar2[coampsi]))
        
        # correction value 1:
        # calculate the mean anomaly between the expected coamps val and the mean crl val
        # used for corrections below!
        center_bin = (bin_i_plus_1 + bin_i)/2 # centered value to compare to obs 2
        anom = np.nanmean(allvar2[coampsi]) - center_bin
        bin_anom.append(anom)
        
        # method 2: do var1 - var2 for all cases within bins. save temporarily
        all_anoms += (allvar2[coampsi] - allvar1[coampsi]).tolist()
                        
    # method 1
    # calculate the "corrected mean" aka the single value used to shift
    # dist up or down. weight every bin equally? not sure if that's a good idea
    # tbh haha, but its a good try for now
    corrected_mean = np.nanmean(np.array(bin_anom))

    # method 2: 
    # average across all points, not a mean of means!
    corrected_mean = np.nanmean(np.array(all_anoms))
    
    return corrected_mean
