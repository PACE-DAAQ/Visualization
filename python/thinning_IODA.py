#!/usr/bin/env python3
import os
import numpy as np
from daaq.utils.date_tools import get_dates
from daaq.obs_utils.ioda_tools import thinIoda

start_date = "2024101800"
final_date = "2024113018"
date_int = 6
dates = get_dates(start_date, final_date, date_int)

obspath = "/glade/campaign/ncar/nmmm0072/Data/obs_pandac"
obstype = "viirs_aod_db_npp"

thin_ratio = 0.8
new_obstype = f"{obstype}-thinned{str(thin_ratio).replace('.', 'p')}"

for cdate in dates:
    print(" ", flush=True)
    print(f"Processing {cdate}", flush=True)
    
    cdate_str = cdate.strftime("%Y%m%d%H")
    iodafile = f"{obspath}/{obstype}/{cdate_str}/{obstype}_obs_{cdate_str}.h5"
    if not os.path.exists(iodafile):
        print(f"  Skip {iodafile}, no such file", flush=True)
        continue

    outiodapath = f"{obspath}/{new_obstype}/{cdate_str}"
    if not os.path.exists(outiodapath):
        os.makedirs(outiodapath)

    outiodafile = f"{outiodapath}/{new_obstype}_obs_{cdate_str}.h5"
    thinIoda(iodafile, outiodafile, thin_ratio)
