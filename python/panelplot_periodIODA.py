#!/usr/bin/env python3
import os
import cartopy.crs as ccrs
from daaq.utils.date_tools import get_dates
from daaq.utils.plot_tools import make_panels, setupax_2dmap
from daaq.obs_utils.ioda_tools import IODAFile

ioda_path = "/glade/campaign/ncar/nmmm0072/Data/obs_pandac"
obs_name_chidx_dict = {
    'oci_uaa_pace_aod': [3],
    # 'spexone_remotap_pace_aod': [7],
    # 'spexone_fmapol_pace_aod': [22],
    # 'harp2_fmapol_pace_aod': [1],
    # 'modis_terra_aod': [0],
    'modis_aqua_aod': [0],
    'viirs_aod_dt_npp': [0], 
    # 'viirs_aod_dt_n20': [0],
    # 'viirs_aod_db_npp': [0], 
    'viirs_aod_db_npp-thinned0p8': [0], 
    # 'viirs_aod_db_n20': [0],
}
start_date = "2024103000"
final_date = "2024103018"
date_interval = 6
dates = get_dates(start_date, final_date, date_interval)

plotgrp = 'ObsValue'
plotvar = 'aerosolOpticalDepth'

figsavedir = '/glade/work/swei/projects/mmm.pace_aod/test/panels/'
savefig = True
quality = 600

area_corner = None
proj = ccrs.PlateCarree()

data_name_dict = {
    'oci_uaa_pace_aod': 'OCI UAA AOD on PACE',
    'spexone_remotap_pace_aod': 'SPEXone RemoTAP AOD on PACE',
    'spexone_fmapol_pace_aod': 'SPEXone FastMAPOL AOD on PACE',
    'harp2_fmapol_pace_aod': 'HARP2 FastMAPOL AOD on PACE',
    'modis_aqua_aod': 'MODIS AOD on Aqua',
    'modis_terra_aod': 'MODIS AOD on Terra',
    'viirs_aod_dt_n20': 'VIIRS DT AOD on NOAA-20',
    'viirs_aod_db_n20': 'VIIRS DB AOD on NOAA-20',
    'viirs_aod_dt_npp': 'VIIRS DT AOD on Suomi-NPP',
    'viirs_aod_db_npp': 'VIIRS DB AOD on Suomi-NPP',
    'viirs_aod_db_npp-thinned0p8': 'VIIRS DB AOD on Suomi-NPP (80% thinned)',
    'aeronet_l15_aod': 'AERONET Level 1.5 AOD',
}

if not os.path.exists(figsavedir):
    os.makedirs(figsavedir)

prods_ch_meta = {}
plot_obs_dfs = {}
for prod, selchidx in obs_name_chidx_dict.items():
    print(f'Load {prod}')
    obs_files = []
    for date in dates:
        cdate_str = date.strftime("%Y%m%d%H")
        obs_file = f"{ioda_path}/{prod}/{cdate_str}/{prod}_obs_{cdate_str}.h5"
        if os.path.exists(obs_file):
            obs_files.append(obs_file)
    plot_obs_dfs[prod], prods_ch_meta[prod] = IODAFile.concat(obs_files, channel_index=selchidx)

fig, axes = make_panels(
    obs_name_chidx_dict.keys(),
    ncols=2,   # Change it as needed
    aspect=0.5,
    hspace=0.1,
    wspace=0.1,
    panel_size=(8,4),
    projection=proj,
)

prod_stats = {}
for pidx, (ax, (prod, selchidx)) in enumerate(zip(axes, obs_name_chidx_dict.items())):
    prod_varname = f'{plotgrp}_{plotvar}_{selchidx[0]}'
    preqc_varname = f'PreQC_{plotvar}_{selchidx[0]}'
    plotdf = plot_obs_dfs[prod].dropna().loc[plot_obs_dfs[prod][preqc_varname] == 0]
    if 'sensorCentralWavelength' not in prods_ch_meta[prod]:
        ax_title = f'{data_name_dict[prod]} at 0.550 um'

    else:
        ax_title = f'{data_name_dict[prod]} at {prods_ch_meta[prod]['sensorCentralWavelength'][selchidx[0]]:.3f} um'

    _, _, gl = setupax_2dmap(ax=ax, cornerlatlon=None, gl_lbsize=10)
    sc = ax.scatter(
        plotdf['longitude'],
        plotdf['latitude'],
        c=plotdf[prod_varname],
        s=1,
        cmap='jet',
        vmin=0.,
        vmax=1.0,
        edgecolors="None",
    )
    ax.set_title(ax_title, fontweight='bold')
    
    # Remove the gridlines labels
    # gl.bottom_labels = False
    # gl.left_labels = False

# fig.suptitle(f'{plotvar} {plotgrp}')
fig.colorbar(sc, ax=axes, pad=0.025, shrink=0.8, label=plotvar)

if savefig:
    figfile = f'{figsavedir}/{plotgrp}_{plotvar}.{start_date}_{final_date}.png'
    fig.savefig(figfile, dpi=quality, bbox_inches='tight')
    print(f'Saving figure -> {figfile}')




