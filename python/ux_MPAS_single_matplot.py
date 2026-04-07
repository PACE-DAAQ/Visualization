#!/usr/bin/env python
# coding: utf-8

import os
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import cartopy.crs as ccrs
import cartopy.feature as cft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uxarray as ux
import itertools
rgas = 287.
rv = 461.6
cp = 7. * rgas / 2.
rcv = rgas/(cp - rgas)


# 
init_from = "20241021"
first_date = "2024102200"
final_date = "2024102200"
date_int = 1
plotvarname = 'temp'
plotlevel = 12
colormap = 'RdYlBu_r'
mpastype = '163842.output'
regional = False
quality = 300

# Specify mpas output folders and file template
outfile_path = "/glade/campaign/ncar/nmmm0072/rkumar/mpas_fcst"
outfile_tmpl = f"{init_from}/{mpastype}.%Y-%m-%d_%H.00.00.nc"

# Specify the static/invariant fields file
init_file = "/glade/derecho/scratch/lacey/MPAS_DEVEL/GOCART2G_v8.3.1/initialization_gocart2G/x1.163842.static_chems.nc"

# Specify figure saving folder
savefig_path = "/glade/work/swei/projects/mmm.pace_aod/plots/mpasfcst"
savefig_tmpl = f"{mpastype}.%Y%m%d%H.from_{init_from}.png"

dates = pd.date_range(
    pd.to_datetime(first_date, format="%Y%m%d%H"),
    pd.to_datetime(final_date, format="%Y%m%d%H"),
    freq=f'{date_int}h',
)

if not os.path.exists(savefig_path):
    os.makedirs(savefig_path)

projection = ccrs.PlateCarree()

def set_size(w,h, ax=None, l=None, r=None, t=None, b=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    if not l:
       l = ax.figure.subplotpars.left
    else:
       ax.figure.subplots_adjust(left=l)
    if not r:
       r = ax.figure.subplotpars.right
    else:
       ax.figure.subplots_adjust(right=r)
    if not t:
       t = ax.figure.subplotpars.top
    else:
       ax.figure.subplots_adjust(top=t)
    if not b:
       b = ax.figure.subplotpars.bottom
    else:
       ax.figure.subplots_adjust(bottom=b)

    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def theta_to_temp(ds):
    if 'theta' in ds.data_vars:
        exist_theta = True
    if 'qv' in ds.data_vars and 'rho' in ds.data_vars:
        exist_qv_rho = True
    if exist_theta:
        if exist_qv_rho:
            qv = np.where(ds['qv'] < 0, 0., ds['qv'])
            theta_m = (1 + rv / rgas * qv) * ds['theta']
            if 'exner' in ds.data_vars:
                exner = ds['exner']
            else:
                exner = ((rgas / 100000.) * (ds['rho'] * theta_m)) ** rcv
            temperature = theta_m * exner
    else:
        raise NameError('theta does not exits')
    return temperature

def get_minmax(da):
    return np.nanmin(da), np.nanmax(da), np.unravel_index(np.nanargmin(da), da.shape), np.unravel_index(np.nanargmax(da), da.shape)


for date in dates:
    out_file = f"{outfile_path}/{date.strftime(outfile_tmpl)}"
    out_ds = ux.open_dataset(init_file, out_file)
    out_ds['temp'] = theta_to_temp(out_ds)
    out_ds['temp'] = out_ds['temp'].assign_attrs(
         {'units': 'K', 'long_name':'air temperature'}
    )
    
    get_minmax(out_ds['temp'])
    
    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    set_size(8, 5, l=0.1, r=0.9, t=0.9, b=0.1)

    lonData = out_ds.uxgrid.face_lon
    latData = out_ds.uxgrid.face_lat
    lonData = ((lonData + 180) % 360) - 180
   
    if regional: 
        maxlon = lonData.data.max()
        minlon = lonData.data.min()
        maxlat = latData.data.max()
        minlat = latData.data.min()
    
        ax.set_extent((minlon, maxlon, minlat, maxlat), crs=projection)
        ax.add_feature(cft.STATES.with_scale('50m'), edgecolor='grey', linewidth=0.5)
    else:  # Global
        ax.set_global()

    ax.add_feature(cft.BORDERS.with_scale('50m'), color='grey', linestyle='--', linewidth=0.5)
    
    if out_ds[plotvarname].ndim == 2:
        plot_var = (out_ds[plotvarname][0]).to_raster(ax=ax)
    elif out_ds[plotvarname].ndim == 3:
        plot_var = (out_ds[plotvarname][0][:, plotlevel]).to_raster(ax=ax)
    colorbar_label = out_ds[plotvarname].long_name
    
    img = ax.imshow(plot_var, cmap=colormap, origin="lower", extent=ax.get_xlim() + ax.get_ylim())
    # ax.set_title(dates[0])
    ax.set_title(f'{dates[0]}\nMax={np.nanmax(plot_var):.2f}, Min={np.nanmin(plot_var):.2f}', loc='left')
    cbar = fig.colorbar(img, ax=ax, fraction=0.03, label=colorbar_label)
    ax.coastlines()

    figname = f"{savefig_path}/{date.strftime(savefig_tmpl)}"
    fig.savefig(figname, dpi=quality)


#minval = 999999.
#min_mem, max_mem = 0, 0
#maxval = -999999.
#for n in range(1, ens_size + 1):
#    tmpfile = f"member{n}/{mpastype}.%Y-%m-%d_%H.00.00.nc"
#    out_file = f"{outfile_path}/{dates[0].strftime(outfile_tmpl)}"
#    out_ds = ux.open_dataset(init_file, out_file)
#    out_ds['temp'] = theta_to_temp(out_ds)
#    out_ds['temp'] = out_ds['temp'].assign_attrs({'units': 'K',
#                                                  'long_name':'air temperature'})
#
#    tmpmin, tmpmax, minloc, maxloc = get_minmax(out_ds[plotvarname])
#    if tmpmin < minval:
#        minval = tmpmin
#        min_mem = n
#        min_lev = minloc[2]
#    if tmpmax > maxval:
#        maxval = tmpmax
#        max_mem = n
#        max_lev = maxloc[2]
#
#print(minval, min_mem, min_lev, maxval, max_mem, max_lev)
#
#Parallel(n_jobs=-1)(delayed(process_date)(date) for date in tqdm(dates))

