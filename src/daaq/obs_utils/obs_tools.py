# obs_utils/obs_tools.py
import numpy as np
import xarray as xr
from datetime import datetime
import geopandas as gpd
import cartopy.crs as ccrs
import pandas as pd
from scipy.spatial import cKDTree
from haversine import haversine_vector, Unit

def dataframes_matchup(
    df_left,
    df_right,
    lat_col='lat',
    lon_col='lon',
    time_col='datetime',
    dist_km=25,
    time_min=30,
    left_cols=None,
    right_cols=None,
    left_suffix='_left',
    right_suffix='_right',
    spatial_prefilter=True,
    prefilter_buffer=1.5,
    prefilter_tree=None,
    verbose=True,
):
    """Vectorized spatiotemporal matchup between two dataframes.

    Pipeline:
      1. Temporal prefilter: drop df_l rows outside df_r's time range (± time_min).
      2. Spatial prefilter:  drop df_l rows with no df_r neighbor within dist_km * buffer.
      3. Hour bucketing:     for each hour in df_l, query a KDTree built only from
                             df_r rows in nearby hours, then apply exact distance
                             and time filters.
    """

    # --- Validate ---
    for name, df in [('df_left', df_left), ('df_right', df_right)]:
        missing = {lat_col, lon_col, time_col} - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    required = {lat_col, lon_col, time_col}
    left_cols = list(df_left.columns) if left_cols is None else list(left_cols)
    right_cols = list(df_right.columns) if right_cols is None else list(right_cols)

    # Preserve column order while ensuring required columns are present
    def _with_required(cols):
        extras = [c for c in required if c not in cols]
        return cols + extras

    df_l = df_left[_with_required(left_cols)].copy()
    df_r = df_right[_with_required(right_cols)].copy()

    df_l[time_col] = pd.to_datetime(df_l[time_col])
    df_r[time_col] = pd.to_datetime(df_r[time_col])

    # Drop rows with missing time — they can never match
    df_l = df_l.dropna(subset=[time_col])
    df_r = df_r.dropna(subset=[time_col])

    # Overlap handling (used for both early-exit and final output)
    overlap = (set(left_cols) & set(right_cols)) - required
    rename_left = {c: f'{c}{left_suffix}' for c in overlap}
    rename_right = {c: f'{c}{right_suffix}' for c in overlap}

    if df_l.empty or df_r.empty:
        return _empty_result(left_cols, right_cols, rename_left, rename_right)

    # --- Temporal prefilter: restrict df_l to df_r's time range (± time_min) ---
    time_delta = pd.Timedelta(minutes=time_min)
    r_tmin = df_r[time_col].min() - time_delta
    r_tmax = df_r[time_col].max() + time_delta

    n_before_time = len(df_l)
    df_l = df_l[(df_l[time_col] >= r_tmin) & (df_l[time_col] <= r_tmax)]

    if verbose:
        print(f"  [matchup] Temporal prefilter: {n_before_time:,} → {len(df_l):,} "
              f"left rows ({len(df_l) / max(n_before_time, 1):.1%} retained, "
              f"window {r_tmin} to {r_tmax})")

    if df_l.empty:
        return _empty_result(left_cols, right_cols, rename_left, rename_right)

    # --- Spatial prefilter ---
    if spatial_prefilter:
        n_before = len(df_l)

        if prefilter_tree is None:
            right_coords_all = np.radians(df_r[[lat_col, lon_col]].to_numpy())
            prefilter_tree = cKDTree(right_coords_all)

        left_coords_all = np.radians(df_l[[lat_col, lon_col]].to_numpy())
        prefilter_rad = (dist_km * prefilter_buffer) / 6371.0

        nearest_dist, _ = prefilter_tree.query(left_coords_all, k=1)
        df_l = df_l.loc[nearest_dist <= prefilter_rad].copy()

        if verbose:
            print(f"  [matchup] Spatial prefilter: {n_before:,} → {len(df_l):,} "
                  f"left rows ({len(df_l) / max(n_before, 1):.1%} retained)")

    if df_l.empty:
        return _empty_result(left_cols, right_cols, rename_left, rename_right)

    # --- Reset indices so positional indexing lines up ---
    df_l = df_l.reset_index(drop=True)
    df_r = df_r.reset_index(drop=True)

    # --- Time bucketing ---
    df_l['_hour'] = df_l[time_col].dt.floor('h')
    df_r['_hour'] = df_r[time_col].dt.floor('h')

    dist_rad = dist_km / 6371.0

    # Pre-extract numpy views for speed
    l_lat = df_l[lat_col].to_numpy()
    l_lon = df_l[lon_col].to_numpy()
    l_time = df_l[time_col].to_numpy()
    r_lat = df_r[lat_col].to_numpy()
    r_lon = df_r[lon_col].to_numpy()
    r_time = df_r[time_col].to_numpy()

    r_by_hour = df_r.groupby('_hour').indices  # dict of hour -> int array

    # How many hour buckets on each side we need to sweep to cover ±time_min
    one_hour = pd.Timedelta(hours=1)
    n_hours = int(np.ceil(time_min / 60))

    left_idx_all = []
    right_idx_all = []
    dist_all = []
    tdiff_all = []

    for hour, l_subidx in df_l.groupby('_hour', sort=False).indices.items():
        # Gather right indices from all hour buckets that could hold a match
        hour_keys = [hour + i * one_hour for i in range(-n_hours, n_hours + 1)]
        r_lists = [r_by_hour[h] for h in hour_keys if h in r_by_hour]
        if not r_lists:
            continue
        r_subidx = np.concatenate(r_lists)

        # Build tree for this window and find candidate pairs within dist_km
        right_rad = np.radians(np.c_[r_lat[r_subidx], r_lon[r_subidx]])
        tree = cKDTree(right_rad)
        left_rad = np.radians(np.c_[l_lat[l_subidx], l_lon[l_subidx]])
        neighbors = tree.query_ball_point(left_rad, dist_rad)

        # Flatten neighbor lists into index pairs
        counts = [len(n) for n in neighbors]
        total = sum(counts)
        if total == 0:
            continue
        left_pairs = np.repeat(l_subidx, counts)
        right_pairs_local = np.fromiter(
            (j for sub in neighbors for j in sub), dtype=np.int64, count=total
        )
        right_pairs = r_subidx[right_pairs_local]

        # Exact distance (vectorized over all candidate pairs at once)
        dists = haversine_vector(
            np.c_[l_lat[left_pairs], l_lon[left_pairs]],
            np.c_[r_lat[right_pairs], r_lon[right_pairs]],
            unit=Unit.KILOMETERS,
        )

        # Exact time filter
        tdiffs = np.abs(
            (l_time[left_pairs] - r_time[right_pairs])
            .astype('timedelta64[s]').astype(np.float64) / 60
        )
        valid = tdiffs <= time_min

        if valid.any():
            left_idx_all.append(left_pairs[valid])
            right_idx_all.append(right_pairs[valid])
            dist_all.append(dists[valid])
            tdiff_all.append(tdiffs[valid])

    if not left_idx_all:
        return _empty_result(left_cols, right_cols, rename_left, rename_right)

    left_idx = np.concatenate(left_idx_all)
    right_idx = np.concatenate(right_idx_all)
    dist_arr = np.concatenate(dist_all)
    tdiff_arr = np.concatenate(tdiff_all)

    # --- Build output ---
    left_out = df_l.iloc[left_idx][left_cols].rename(columns=rename_left).reset_index(drop=True)
    right_out = df_r.iloc[right_idx][right_cols].rename(columns=rename_right).reset_index(drop=True)

    result = pd.concat([left_out, right_out], axis=1)
    result['distance_km'] = dist_arr
    result['time_diff_min'] = tdiff_arr

    return result


def _empty_result(left_cols, right_cols, rename_left, rename_right):
    left_out = [rename_left.get(c, c) for c in left_cols]
    right_out = [rename_right.get(c, c) for c in right_cols]
    return pd.DataFrame(columns=left_out + right_out + ['distance_km', 'time_diff_min'])

def select_obs_in_regions(obs_df, regions, projection=ccrs.PlateCarree()):
    regions = regions.set_crs("EPSG:4326", allow_override=True)
    regions_proj = regions.to_crs(projection)

    region_geom = regions_proj.geometry.union_all()
    
    obs_gdf = gpd.GeoDataFrame(
        obs_df,
        geometry=gpd.points_from_xy(obs_df['longitude'], obs_df['latitude']),
        crs="EPSG:4326"
    )

    obs_proj = obs_gdf.to_crs(projection)

    # --- Spatial mask ---
    mask = obs_proj.geometry.within(region_geom)

    return obs_df.loc[mask.values]

def modisaod_to_dataset(modisgranule):
    from pyhdf.SD import SD, SDC
    hdf = SD(modisgranule, SDC.READ)

    # All of MODIS AOD data have a singular reference time - good practice to get from attribute
    modis_time_key = 'Scan_Start_Time'
    try:
        modis_time_attribute = hdf.select(modis_time_key).attributes().get('units')
        if modis_time_attribute is None:
            print("'units' attribute is not present in {modis_time_key}.")
            modis_ref_time = datetime(1993, 1, 1, 0, 0, 0)
        else:
            # Extract the date and time part
            datetime_str = modis_time_attribute.split('since ')[1].rsplit(' ', 1)[0]

            # Convert to a datetime object
            modis_ref_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        # Catch and print any errors
        print(f"An error occurred: {e}")
    #  Get variables
    modis_time = hdf.select(modis_time_key)[:].ravel()
    cnts = len(modis_time)

    land_sea_flag = hdf.select('Land_sea_Flag')[:].ravel()
    aod = hdf.select('AOD_550_Dark_Target_Deep_Blue_Combined')[:].ravel() * 1e-3
    unc_land = hdf.select('Deep_Blue_Aerosol_Optical_Depth_550_Land_Estimated_Uncertainty')[:].ravel() * 1e-3
    over_land = np.logical_not(land_sea_flag == 0)
    
    data_dict = {
        'lat': (['Location'], hdf.select('Latitude')[:].ravel()),
        'lon': (['Location'], hdf.select('Longitude')[:].ravel()),
        'aod': (['Location'], aod),
        'land_sea_flag': (['Location'], land_sea_flag),
        'QC_flag': (['Location'], hdf.select('Land_Ocean_Quality_Flag')[:].ravel()),
        'sol_zen': (['Location'], hdf.select('Solar_Zenith')[:].ravel()),
        'sen_zen': (['Location'], hdf.select('Sensor_Zenith')[:].ravel()),
        'uncertainty': (['Location'], np.where(over_land, unc_land, np.add(0.05, np.multiply(0.15, aod)))),
        'obs_time': (['Location'], (modis_time + modis_ref_time.timestamp()).astype('datetime64[s]')),
    }

    coords_dict = {'Location': np.arange(cnts)}
    return xr.Dataset(data_dict, coords=coords_dict)