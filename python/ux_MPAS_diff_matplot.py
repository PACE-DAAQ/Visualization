#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[16]:


first_date = "2024050218"
final_date = "2024050218"
date_int = 1
plotvarname = 'theta'
plotlevel = 50
colormap = 'RdBu'
member = 48

outfile_path = "/glade/derecho/scratch/swei/hydrosat_tmp/MPAS-DART/superobhyd15km"
prefile_tmpl = f"%Y%m%d%H/preassim_member_{member:04d}.nc"
pstfile_tmpl = f"%Y%m%d%H/postassim_member_{member:04d}.nc"
init_file = "/glade/work/swei/projects/hydrosat/mpas.configs/15km/conus.init.nc"
savefig_path = "/glade/work/swei/projects/hydrosat/plots/baseline"

dates = pd.date_range(
    pd.to_datetime(first_date, format="%Y%m%d%H"),
    pd.to_datetime(final_date, format="%Y%m%d%H"),
    freq=f'{date_int}h',
)

if not os.path.exists(savefig_path):
    os.makedirs(savefig_path)

projection = ccrs.PlateCarree()


# In[17]:


pre_file = f"{outfile_path}/{dates[0].strftime(prefile_tmpl)}"
pst_file = f"{outfile_path}/{dates[0].strftime(pstfile_tmpl)}"
pre_ds = ux.open_dataset(init_file, pre_file)
pst_ds = ux.open_dataset(init_file, pst_file)

lonData = pre_ds.uxgrid.face_lon
latData = pre_ds.uxgrid.face_lat
lonData = ((lonData + 180) % 360) - 180

maxlon = lonData.data.max()
minlon = lonData.data.min()
maxlat = latData.data.max()
minlat = latData.data.min()
print((minlon, maxlon, minlat, maxlat))


# In[19]:


fig = plt.figure()
ax = plt.subplot(projection=projection)

# ax.set_global()
ax.set_extent((minlon, maxlon, minlat, maxlat), crs=projection)
ax.add_feature(cft.BORDERS.with_scale('50m'), color='grey', linestyle='--', linewidth=0.5)
ax.add_feature(cft.STATES.with_scale('50m'), edgecolor='grey', linewidth=0.5)

if pre_ds[plotvarname].ndim == 2:
    plot_var = (pre_ds[plotvarname][0] - pst_ds[plotvarname][0]).to_raster(ax=ax)
elif pre_ds[plotvarname].ndim == 3:
    plot_var = (pre_ds[plotvarname][0][:, plotlevel] - pst_ds[plotvarname][0][:, plotlevel]).to_raster(ax=ax)
colorbar_label = pre_ds[plotvarname].long_name

img = ax.imshow(plot_var, cmap=colormap, origin="lower", extent=ax.get_xlim() + ax.get_ylim())
ax.set_title(f'Max={np.nanmax(plot_var):.2f}, Min={np.nanmin(plot_var):.2f}')
cbar = fig.colorbar(img, ax=ax, fraction=0.03, label=colorbar_label)
ax.coastlines()


# In[66]:


def process_date(date):
    data_file = f"{outfile_path}/{date.strftime(outfile_tmpl)}"
    fig_file = f"{savefig_path}/{date.strftime(fig_name_tmpl)}"

    ds_i = ux.open_dataset(init_file, data_file)

    lonData = ds_i.uxgrid.face_lon
    latData = ds_i.uxgrid.face_lat
    lonData = ((lonData + 180) % 360) - 180

    maxlon = lonData.data.max()
    minlon = lonData.data.min()
    maxlat = latData.data.max()
    minlat = latData.data.min()

    in_domain = (
        (all_orbit_df['lon'] <= maxlon)&
        (all_orbit_df['lon'] >= minlon)&
        (all_orbit_df['lat'] <= maxlat)&
        (all_orbit_df['lat'] >= minlat)
    )

    orbits_in_domain = all_orbit_df.loc[in_domain]

    ctt = ds_i['ctt'][0].plot.polygons(
        rasterize=True,
        projection=projection,
        cmap='binary',
        features={"borders": "50m", "coastline": "50m", "states": "50m"},
        title=f'{date}',
        clim=(200, 320),
        clabel='Cloud Top Temperature (K)',
    ).opts(
        fontsize={'title': 12, 'labels': 12, 'xticks': 12, 'yticks': 12},
    )

    return


# In[ ]:


Parallel(n_jobs=-1)(delayed(process_date)(date) for date in tqdm(dates))


# In[ ]:




