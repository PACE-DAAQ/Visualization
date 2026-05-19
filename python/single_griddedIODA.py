#!/usr/bin/env python3
import os
import xarray as xr
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from daaq.utils.date_tools import get_dates
from daaq.obs_utils.ioda_tools import IODAFile
from daaq.obs_utils.obs_tools import bin_obsdf_to_grid, aggregate_total, compute_stats
from daaq.obs_utils.hofx_tools import load_hofx_cycles_single, load_hofx_cycles_paired

sdate = 2024110100
edate = 2024113018
date_interval = 6
dates = get_dates(sdate, edate, date_interval)
bkg = 'MERRA-2'
expname = 'metens_1.daout'
fsave = 1
quality = 600

hofx_path = '/glade/derecho/scratch/junpark/pandac/junpark_3denvar_OIE60km_WarmStart_AOD_30_PACE_OCI_FromOCT15_v2_NOAH/CyclingDA/'

# template acceptable strings: datestr, product, and stagetag (if paired files)
file_template = '{datestr}/dbOut/obsout_da_{product}.h5'
outfile_tmpl = '{savedir}/{exp}.{product}.griddedIODA.{resolution}.nc'

obs_name_chidx_dict = {
    'oci_uaa_pace_aod': [3],
    # 'spexone_remotap_pace_aod': [7],
    # 'spexone_fmapol_pace_aod': [22],
    # 'harp2_fmapol_pace_aod': [1],
    #'modis_terra_aod': [0],
    #'modis_aqua_aod': [0],
    # 'viirs_aod_dt_npp': [0], 
    # 'viirs_aod_dt_n20': [0],
    # 'viirs_aod_db_npp': [0], 
    #'viirs_aod_db_npp-thinned0p8': [0], 
    # 'viirs_aod_db_n20': [0],
}
varname = 'aerosolOpticalDepth'

savedir = f'/glade/work/swei/projects/mmm.pace_aod/exp_data'
if not os.path.exists(savedir):
    os.makedirs(savedir)

obs_dfs = {}
bkg_files = {}
ana_files = {}
cycles = {}
for prod, selchidx in obs_name_chidx_dict.items():
    print(f'Load {prod}')
    obs_dfs[prod], cycles[prod] = load_hofx_cycles_single(
        dates=dates,
        hofx_path=hofx_path,
        file_template=file_template,
        product=prod,
        varname=varname,
        channel_index=selchidx,
        bkg_group='hofx0',
        ana_group='hofx1',
    )
    print(f'  obs files cnts= {len(cycles[prod])}')

latlon_res = 1
lat_edges = np.arange(-90., 90 + latlon_res, latlon_res)
lon_edges = np.arange(-180., 180 + latlon_res, latlon_res)

for prod in obs_dfs.keys():
    qccols = [col for col in obs_dfs[prod].columns if 'EffectiveQC' in col]
    per_cycle = [bin_obsdf_to_grid(g, lat_edges, lon_edges, qc_pass_cols=qccols) for c, g in obs_dfs[prod].groupby('cycle')]
    ds = xr.concat(per_cycle, dim=pd.Index(cycles[prod], name='time'))
    ncoutfile = outfile_tmpl.format(savedir=savedir, exp=expname, product=prod, resolution=f"{str(latlon_res).replace('.', 'p')}deg")
    ds.to_netcdf(ncoutfile)
