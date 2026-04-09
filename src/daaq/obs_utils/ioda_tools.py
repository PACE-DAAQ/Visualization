# obs_utils/ioda_tools.py
import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd

def readAeronetIoda(iodafile, select_wavelengths):
    aeronet_aod_wvl = [340., 380., 440., 500., 675, 870., 1020., 1640.]
    groups = xr.open_groups(iodafile)
    groups_list = groups.keys()
    metads = groups['/MetaData']
    obsvds = groups['/ObsValue'].assign_coords(Channel=aeronet_aod_wvl)
    df = pd.DataFrame()
    for wvl in select_wavelengths:
        col_name = f'{int(wvl)}nm'
        df[col_name] = obsvds['aerosolOpticalDepth'].sel(Channel=wvl)
    df['station'] = metads['stationIdentification']
    df['dateTime'] = metads['dateTime']
    df['latitude'] = metads['latitude']
    df['longitude'] = metads['longitude']
    df = df.dropna()
    df['Angstrom_440_870nm'] = -np.log(df['440nm'] / df['870nm']) / np.log(440. / 870.)
    df['AOD550nm'] = df['500nm'] * (550/500) ** (df['Angstrom_440_870nm'])

    # if '/hofx' in groups_list:
    #     hofxds = groups['/hofx'].assign_coords(Channel=aeronet_aod_wvl)
    #     df['Angstrom_440_870nm'] = -np.log(df['440nm'] / df['870nm']) / np.log(440. / 870.)
    #     df['AOD550nm'] = df['500nm'] * (550/500) ** (df['Angstrom_440_870nm'])
    #     df['hofx550'] = hofxds['aerosolOpticalDepth'].sel(Channel=550)
    return df

def readAodProdIoda(iodafile):
    groups = xr.open_groups(iodafile)
    groups_list = groups.keys()
    metads = groups['/MetaData']
    if 'pace_aod' in iodafile:
        tmpwvl = metads['sensorCentralWavelength'].values * 1e3
    else:
        tmpwvl = [550.]
    obsvds = groups['/ObsValue'].assign_coords(Channel=tmpwvl)
    preqcds = groups['/PreQC'].assign_coords(Channel=tmpwvl)
    df = pd.DataFrame()
    df['dateTime'] = metads['dateTime']
    df['latitude'] = metads['latitude']
    df['longitude'] = metads['longitude']
    df['AOD550nm'] = obsvds['aerosolOpticalDepth'].sel(Channel=550)
    df['AOD550_preqc'] = preqcds['aerosolOpticalDepth'].sel(Channel=550)
    
    if '/hofx' in groups_list:
        hofxds = groups['/hofx'].assign_coords(Channel=tmpwvl)
        df['hofx550'] = hofxds['aerosolOpticalDepth'].sel(Channel=550)
    return df

def cropIoda(iodafile, output, poly_file):
    '''
        output: path of cropped IODA file
        polygon: cropping area
    '''

    def isinside(lat_arr, lon_arr, poly_file):
        from shapely.geometry import Point, Polygon
        df = pd.read_csv(poly_file)
        minlat = np.floor(df['Lat'].min()).astype(np.int32)
        maxlat = np.ceil(df['Lat'].max()).astype(np.int32)
        minlon = np.floor(df['Lon'].min()).astype(np.int32)
        maxlon = np.ceil(df['Lon'].max()).astype(np.int32)
        print(f'min/max lat, min/max lon: {minlat}/{maxlat}, {minlon}/{maxlon}')

        polygon_coords = list(zip(df['Lat'].values, df['Lon'].values))
        # Create a shapely Polygon object
        polygon = Polygon(polygon_coords)
        out_mask = np.zeros_like(lat_arr, dtype=bool)
        near_mask = ((lat_arr > minlat) & (lat_arr < maxlat) &
                    (lon_arr > minlon) & (lon_arr < maxlon))

        for i, (plat, plon) in enumerate(zip(lat_arr, lon_arr)):
            if near_mask[i]:
                point = Point(plat, plon)
                out_mask[i] = polygon.contains(point)
        del(df)

        return out_mask


    src = nc.Dataset(iodafile, 'r')
    if src.dimensions['Location'].size == 0:
        raise Exception('no obs available')

    lat = src.groups['MetaData'].variables['latitude'][:].ravel()
    lon = src.groups['MetaData'].variables['longitude'][:].ravel()

    mask = isinside(lat, lon, poly_file)

    if np.count_nonzero(mask) == 0:
        raise Exception('no obs available in the target area')
    else:
        print(f'{np.count_nonzero(mask)} obs in the target area')

    dst = nc.Dataset(output, 'w')
    dst.setncatts(src.__dict__)

    for name, dimension in src.dimensions.items():
        if name == 'Location':
            dst.createDimension(name, np.count_nonzero(mask))
        else:
            dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    for name, variable in src.variables.items():
        print(f'Processing {variable}')
        # Define the variable in the new file
        dst_var = dst.createVariable(name, variable.datatype, variable.dimensions)
        # Copy variable attributes
        dst_var.setncatts(variable.__dict__)
        if 'Location' in variable.dimensions:
            indices = [slice(None)] * variable.ndim
            dim_index = variable.dimensions.index('Location')
            indices[dim_index] = mask
            dst_var[:] = variable[tuple(indices)]
        else:
            dst_var[:] = variable[:]

    for grp, group in src.groups.items():
        dst_grp = dst.createGroup(grp)
        for var, variable in src.groups[grp].variables.items():
            print(f'Processing {grp} / {var}')
            fill_value = variable.getncattr('_FillValue') if '_FillValue' in variable.ncattrs() else None

            if fill_value is not None:
                try:
                    dst_var = dst_grp.createVariable(var, variable.datatype, variable.dimensions,
                                                        fill_value=variable.datatype.type(fill_value))
                except Exception as e:
                    print(f"Could not set _FillValue during variable creation: {e}")
                    dst_var = dst_grp.createVariable(var, variable.datatype, variable.dimensions)
            else:
                dst_var = dst_grp.createVariable(var, variable.datatype, variable.dimensions)

            # Then copy remaining attributes
            for attr in variable.ncattrs():
                if attr != '_FillValue':
                    dst_var.setncattr(attr, variable.getncattr(attr))

            if 'Location' in variable.dimensions:
                indices = [slice(None)] * variable.ndim
                dim_index = variable.dimensions.index('Location')
                indices[dim_index] = mask
                dst_var[:] = variable[tuple(indices)]
            else:
                dst_var[:] = variable[:]
    src.close()
    dst.close()
    return

def thinIoda(inIoda, outIoda, thinning_ratio, **kwargs):
    src = nc.Dataset(inIoda, 'r')
    if src.dimensions['Location'].size == 0:
        raise Exception('no obs available')

    lat = src.groups['MetaData'].variables['latitude'][:].ravel()
    lon = src.groups['MetaData'].variables['longitude'][:].ravel()

    mask = np.random.uniform(size=len(lon)) > thinning_ratio
        
    if np.count_nonzero(mask) == 0:
        raise Exception('no obs available in the target area')
    else:
        print(f'{np.count_nonzero(mask)} obs in the target area')

    dst = nc.Dataset(outIoda, 'w')
    dst.setncatts(src.__dict__)

    for name, dimension in src.dimensions.items():
        if name == 'Location':
            dst.createDimension(name, np.count_nonzero(mask))
        else:
            dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    for name, variable in src.variables.items():
        print(f'Processing {variable}')
        # Define the variable in the new file
        dst_var = dst.createVariable(name, variable.datatype, variable.dimensions)
        # Copy variable attributes
        dst_var.setncatts(variable.__dict__)
        if 'Location' in variable.dimensions:
            indices = [slice(None)] * variable.ndim
            dim_index = variable.dimensions.index('Location')
            indices[dim_index] = mask
            dst_var[:] = variable[tuple(indices)]
        else:
            dst_var[:] = variable[:]

    for grp, group in src.groups.items():
        dst_grp = dst.createGroup(grp)
        for var, variable in src.groups[grp].variables.items():
            print(f'Processing {grp} / {var}')
            fill_value = variable.getncattr('_FillValue') if '_FillValue' in variable.ncattrs() else None

            if fill_value is not None:
                try:
                    dst_var = dst_grp.createVariable(var, variable.datatype, variable.dimensions,
                                                        fill_value=variable.datatype.type(fill_value))
                except Exception as e:
                    print(f"Could not set _FillValue during variable creation: {e}")
                    dst_var = dst_grp.createVariable(var, variable.datatype, variable.dimensions)
            else:
                dst_var = dst_grp.createVariable(var, variable.datatype, variable.dimensions)

            # Then copy remaining attributes
            for attr in variable.ncattrs():
                if attr != '_FillValue':
                    dst_var.setncattr(attr, variable.getncattr(attr))

            if 'Location' in variable.dimensions:
                indices = [slice(None)] * variable.ndim
                dim_index = variable.dimensions.index('Location')
                indices[dim_index] = mask
                dst_var[:] = variable[tuple(indices)]
            else:
                dst_var[:] = variable[:]
    src.close()
    dst.close()
    return