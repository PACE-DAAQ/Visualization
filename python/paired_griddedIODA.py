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
expname = 'test.metens_1'
fsave = 1
quality = 600

hofx_path = '/glade/derecho/scratch/junpark/pandac/MPASJEDI_HOFX3D_AODCRTM_PACE_REVISED_MODIS'
hofx_file_tmpl = 'OUT_{datestr}/{stagetag}_da_{product}.h5'
oci_file_tmpl = 'OUT_{datestr}/{stagetag}_{product}_obs_{datestr}.h5'
outfile_tmpl = '{savedir}/{exp}.{product}.griddedIODA.nc'

stage_tag_dict = {
    'bkg': 'obsout',
    'ana': 'ana_obsout',
}

obs_name_chidx_dict = {
    'oci_uaa_pace_aod': [3],
    # 'spexone_remotap_pace_aod': [7],
    # 'spexone_fmapol_pace_aod': [22],
    # 'harp2_fmapol_pace_aod': [1],
    'modis_terra_aod': [0],
    'modis_aqua_aod': [0],
    # 'viirs_aod_dt_npp': [0], 
    # 'viirs_aod_dt_n20': [0],
    # 'viirs_aod_db_npp': [0], 
    'viirs_aod_db_npp-thinned0p8': [0], 
    # 'viirs_aod_db_n20': [0],
}
plotgrp = 'ObsValue'
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
    if 'oci' in prod:
        filetmpl = oci_file_tmpl
    else:
        filetmpl = hofx_file_tmpl

    obs_dfs[prod], cycles[prod] = load_hofx_cycles_paired(
        dates=dates,
        hofx_path=hofx_path,
        file_template=filetmpl,
        product=prod,
        varname=varname,
        channel_index=selchidx,
        bkg_stage_tag=stage_tag_dict['bkg'],
        ana_stage_tag=stage_tag_dict['ana'],
    )
    print(f'  obs files cnts= {len(cycles[prod])}')

latlon_res = 1
lat_edges = np.arange(-90., 90 + latlon_res, latlon_res)
lon_edges = np.arange(-180., 180 + latlon_res, latlon_res)

for prod in obs_dfs.keys():
    qccols = [col for col in obs_dfs[prod].columns if 'EffectiveQC' in col]
    per_cycle = [bin_obsdf_to_grid(g, lat_edges, lon_edges, qc_pass_cols=qccols) for c, g in obs_dfs[prod].groupby('cycle')]
    ds = xr.concat(per_cycle, dim=pd.Index(cycles[prod], name='time'))
    ncoutfile = outfile_tmpl.format(savedir=savedir, exp=expname, product=prod)
    ds.to_netcdf(ncoutfile)
