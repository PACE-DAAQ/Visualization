# obs_utils/obs_tools.py
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import xarray as xr
from datetime import datetime
import geopandas as gpd
import cartopy.crs as ccrs
import pandas as pd
from scipy.spatial import cKDTree
from haversine import haversine_vector, Unit
 
from daaq.obs_utils.ioda_tools import IODAFile

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
# 
# Functions to grid the IODA dataframe into lat/lon xr.dataset
# 
def _is_binnable(series: pd.Series) -> bool:
    """A column is binnable if it's numeric and not a QC-like flag."""
    # pd.api.types handles both numpy and pandas extension dtypes
    if not pd.api.types.is_numeric_dtype(series):
        return False
    # Booleans are technically numeric but we'd rather treat them as flags
    if pd.api.types.is_bool_dtype(series):
        return False
    return True
 
 
def _looks_like_qc(name: str) -> bool:
    QC_GROUPS = ('PreQC', 'EffectiveQC')
    return any(name.startswith(g) or name == g for g in QC_GROUPS)
 
 
def bin_obsdf_to_grid(
    df: pd.DataFrame,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    *,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    variables: Optional[Sequence[str]] = None,
    skip_cols: Sequence[str] = (),
    qc_pass_cols: Optional[Sequence[str]] = None,
    qc_pass_value: int = 0,
) -> xr.Dataset:
    """
    Bin per-observation values onto a lat/lon grid, storing only additive
    statistics (sum, sum of squares, count) so results can be safely combined
    across cycles via simple addition.
 
    Means and standard deviations are *not* stored — use `compute_stats(ds)`
    on the result (or any time-reduced version of it) to derive them.
 
    QC-like columns are skipped automatically; pass them explicitly via
    `qc_pass_cols` to also store per-cell pass *counts* (additive). Convert
    those to pass rates in post-processing as `pass_count / count`.
 
    Parameters
    ----------
    df : DataFrame
        Per-observation rows, must contain `lat_col` and `lon_col`.
    lat_edges, lon_edges : 1D arrays
        Cell edges, monotonic increasing. Lengths NLAT+1 and NLON+1.
        Longitudes are normalized to match `lon_edges`'s range.
    variables : sequence of str or None
        Columns to bin. If None, all numeric, non-QC, non-coordinate columns.
        An explicit empty list means "bin no variables, just count/QC".
    skip_cols : sequence of str
        Extra columns to exclude.
    qc_pass_cols : sequence of str or None
        QC columns to bin as pass counts.
    qc_pass_value : int
        Flag value treated as "pass". Default 0.
 
    Returns
    -------
    xr.Dataset
        Dimensions ('lat', 'lon'). Always contains `count`. For each binned
        variable, contains `{var}_sum` and `{var}_sumsq`. For each QC column,
        contains `{qc_col}_pass_count`.
    """
    lat_edges = np.asarray(lat_edges, dtype=float)
    lon_edges = np.asarray(lon_edges, dtype=float)
    nlat = len(lat_edges) - 1
    nlon = len(lon_edges) - 1
    if nlat < 1 or nlon < 1:
        raise ValueError("lat_edges and lon_edges must each have length >= 2.")
 
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()
 
    # Normalize longitudes into the edge range.
    lon_min = lon_edges[0]
    lon = ((lon - lon_min) % 360.0) + lon_min
 
    in_range = (
        (lat >= lat_edges[0]) & (lat <= lat_edges[-1]) &
        (lon >= lon_edges[0]) & (lon <= lon_edges[-1])
    )
    if not in_range.all():
        n_drop = int((~in_range).sum())
        if n_drop:
            print(f'[bin] dropping {n_drop} obs outside grid')
 
    df_in = df.loc[in_range]
    lat = lat[in_range]
    lon = lon[in_range]
 
    ilat = np.clip(np.digitize(lat, lat_edges) - 1, 0, nlat - 1)
    ilon = np.clip(np.digitize(lon, lon_edges) - 1, 0, nlon - 1)
    flat_idx = ilat * nlon + ilon
    n_cells = nlat * nlon
 
    def _bincount_2d(idx, weights=None):
        return np.bincount(idx, weights=weights, minlength=n_cells).reshape(nlat, nlon)
 
    n_bin = _bincount_2d(flat_idx).astype(np.int64)
 
    # Decide which columns to bin.
    auto_skip = {lat_col, lon_col, 'dateTime'}
    auto_skip.update(skip_cols)
    qc_pass_cols = list(qc_pass_cols or [])
 
    if variables is None:
        candidates = [
            c for c in df_in.columns
            if c not in auto_skip
            and c not in qc_pass_cols
            and _is_binnable(df_in[c])
            and not _looks_like_qc(c)
        ]
    else:
        # An explicit empty list means "bin no variables, just count/QC".
        candidates = list(variables)
 
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
 
    data_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {
        'count': (('lat', 'lon'), n_bin),
    }

    for col in candidates:
        chidx = col.split('_')[-1]
        qc_col = next((q for q in qc_pass_cols if q.endswith(f'_{chidx}')), None)

        vals_full = df_in[col].to_numpy(dtype=float)
        valid = np.isfinite(vals_full)
        total_idx = flat_idx[valid]
        total_count = _bincount_2d(total_idx)

        if qc_col and qc_col in df_in.columns:
            mask = valid & (df_in[qc_col].to_numpy() == qc_pass_value)
        else:
            mask = valid

        idx = flat_idx[mask]
        vals = vals_full[mask]
        pass_count = _bincount_2d(idx)

        data_vars[f'{col}_sum']       = (('lat', 'lon'), _bincount_2d(idx, weights=vals))
        data_vars[f'{col}_sumsq']     = (('lat', 'lon'), _bincount_2d(idx, weights=vals * vals))
        data_vars[f'{col}_count']     = (('lat', 'lon'), pass_count)
        data_vars[f'{col}_total']     = (('lat', 'lon'), total_count)
        with np.errstate(divide='ignore', invalid='ignore'):
            data_vars[f'{col}_pass_rate'] = (('lat', 'lon'), np.where(total_count > 0, pass_count / total_count, np.nan))
 
    return xr.Dataset(
        data_vars,
        coords={'lat': lat_centers, 'lon': lon_centers},
        attrs={'lat_edges': lat_edges, 'lon_edges': lon_edges},
    )

# ---------------------------------------------------------------------------
# Time aggregation and statistics
# ---------------------------------------------------------------------------
 
# Variables in a binned/aggregated dataset that are *additive* across time.
# Anything else (e.g. derived means/stds) should not be stored, only computed.
_ADDITIVE_SUFFIXES = ('_sum', '_sumsq', '_pass_count')
 
 
def _is_additive(name: str) -> bool:
    return name == 'count' or any(name.endswith(s) for s in _ADDITIVE_SUFFIXES)
 
 
def aggregate_total(ds: xr.Dataset, dim: str = 'time') -> xr.Dataset:
    """
    Reduce a per-cycle gridded dataset along `dim` by summing additive vars.
 
    Counts, sums, sums-of-squares, and pass-counts are simply added across
    time. The result has the same spatial dims with `time` collapsed.
 
    Parameters
    ----------
    ds : xr.Dataset
        Output of `aggregate_ioda_cycles` (or any compatible per-cycle stack).
    dim : str
        Dimension to sum over. Default 'time'.
 
    Returns
    -------
    xr.Dataset
        Same data variables as input, with `dim` collapsed.
    """
    if dim not in ds.dims:
        raise ValueError(f"Dimension '{dim}' not found in dataset (dims={list(ds.dims)}).")
 
    out_vars = {}
    for name, da in ds.data_vars.items():
        if _is_additive(name) and dim in da.dims:
            out_vars[name] = da.sum(dim=dim, skipna=True)
        elif dim in da.dims:
            # Non-additive variable along time — drop with a warning rather
            # than silently producing garbage.
            import warnings
            warnings.warn(
                f"Dropping non-additive variable '{name}' during aggregate_total; "
                f"recompute it from sums/counts after reduction.",
                stacklevel=2,
            )
        else:
            out_vars[name] = da
 
    out = xr.Dataset(out_vars, attrs=dict(ds.attrs))
    out.attrs[f'reduced_{dim}'] = 1
    return out
 
 
def compute_stats(
    ds: xr.Dataset,
    *,
    variables: Optional[Sequence[str]] = None,
    ddof: int = 0,
    pass_rate: bool = True,
) -> xr.Dataset:
    """
    Derive mean / variance / std (and optionally QC pass rate) from the
    additive sums in a binned dataset.
 
    Works on either a per-cycle dataset (with `time` dim) or a time-reduced
    one. The math is identical: `mean = sum/count`, `var = sumsq/count - mean^2`.
 
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing `count` and `{var}_sum`/`{var}_sumsq` pairs.
    variables : sequence of str or None
        Variable base names (without suffix) to compute stats for. If None,
        every variable with both `_sum` and `_sumsq` is included.
    ddof : int
        Delta degrees of freedom for variance: divisor is `count - ddof`.
        Default 0 (population variance).
    pass_rate : bool
        If True, also emit `{qc_col}_pass_rate = pass_count / count` for any
        `_pass_count` variables present.
 
    Returns
    -------
    xr.Dataset
        For each variable: `{var}_mean`, `{var}_var`, `{var}_std`. Plus the
        original `count`, plus optional `{qc_col}_pass_rate`.
    """
    if 'count' not in ds.data_vars:
        raise KeyError("Dataset must contain 'count'.")
    count = ds['count'].astype(float)
 
    # Discover variables if not specified.
    if variables is None:
        bases = set()
        for name in ds.data_vars:
            if name.endswith('_sum'):
                base = name[:-len('_sum')]
                if f'{base}_sumsq' in ds.data_vars:
                    bases.add(base)
        variables = sorted(bases)
 
    out = xr.Dataset(coords=ds.coords, attrs=dict(ds.attrs))
    out['count'] = ds['count']
 
    safe_count = count.where(count > 0)
 
    for base in variables:
        sum_name   = f'{base}_sum'
        sumsq_name = f'{base}_sumsq'
        if sum_name not in ds.data_vars or sumsq_name not in ds.data_vars:
            continue
 
        s  = ds[sum_name]
        ss = ds[sumsq_name]
 
        mean = s / safe_count
        # Population variance: E[X^2] - E[X]^2
        var_pop = ss / safe_count - mean * mean
        # Numerical noise can produce small negatives; clip.
        var_pop = var_pop.where(var_pop >= 0, 0.0)
 
        if ddof != 0:
            denom = (safe_count - ddof).where(safe_count - ddof > 0)
            var = var_pop * safe_count / denom
        else:
            var = var_pop
 
        out[f'{base}_mean'] = mean
        out[f'{base}_var']  = var
        out[f'{base}_std']  = np.sqrt(var)
 
    if pass_rate:
        for name in ds.data_vars:
            if name.endswith('_pass_count'):
                base = name[:-len('_pass_count')]
                out[f'{base}_pass_rate'] = ds[name] / safe_count
 
    return out