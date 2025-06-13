# use this wrapper file to automatically run tests on input model and obs data.
# all code is found in /code-paper-v1, and all figures created are sent to /figures-paper-v1.

#%%

# paths used to pull data and helper scripts
import sys
basepath = "/Users/etmu9498/all-research/nrl-summer-2024/mwr-paper-supporting-material/"
sys.path.append(basepath + "code/")
import sam_intensity
import sam_track
import compare_crl_fl_coamps
import crl_dropsonde_correction
import crl_coamps_profiles
import helper_plots
import moisture_quadrant_changes
import radial_wind_quadrant_changes
mc = True # use a manual center for coamps-tc xy distance calculation?


# #Figure 1: TC Sam CRL, FL, Drop, and coamps curtain
# compare_crl_fl_coamps.plot(manual_center=mc)
# print('Figure 1 saved.')           
   
# #Figure 2: crl T correction via dropsondes
# crl_dropsonde_correction.plot_corrections()
# print('Figure 2 saved.')           

# #Figure 3: sam tracks and intensity
# sam_intensity.intensity_ships(tcname='Sam')
# sam_track.two_subplots(tcname='Sam', tdrfield='windspeed')
# print('Figure 3 saved.')           

#Figures 4, 5, and 6: create and save TDR composites
# helper_plots.make_figs_4_5_6(manual_center=mc, xaxis='rmw')
# print('Figures 4-6 saved.')           

# # Figures 7 and 8: create vertically averaged columns of eye vs outside eye T, etc
# crl_coamps_profiles.plot(manual_center=mc)
# print('Figures 7-8 saved.')           

# Figures 9, 10, and 11: thermodynamic composites, model vs obs
helper_plots.make_figs_9_10_11(manual_center=mc, xaxis='rmw')
print('Figures 9-11 saved.')           

# # Figure 12: quadrant and model obs differences, two radial passes, crl
# moisture_quadrant_changes.single_legs_plot(correct=True, fname='20210927H1', var='wv', quad='ur-dl',
#                             manual_center=mc)
# print('Figure 12 saved.')           

# # Figure 13: quadrant differences in radial wind
# radial_wind_quadrant_changes.single_legs_plot(correct=True, fname='20210927H1', quad='ur-dl',
#                             manual_center=mc, windvar='radial')
# print('Figure 13 saved.')           


# %%
