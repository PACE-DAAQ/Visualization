# obs_utils/obs_tools.py
import numpy as np
import xarray as xr
from datetime import datetime
import geopandas as gpd
import cartopy.crs as ccrs

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