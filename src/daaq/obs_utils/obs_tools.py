# obs_utils/obs_tools.py

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