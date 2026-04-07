# mpas_utils/ux_mpas.py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cft

rgas = 287.
rv = 461.6
cp = 7. * rgas / 2.
rcv = rgas/(cp - rgas)

def diag_mpasds(ds, delz):
    attrs_dict = {
        'temp': {'units': 'K', 'long_name': 'air temperature'},
        'lwc': {'units': 'g/m^3', 'long_name': 'liquid water content'},
        'lwp': {'units': 'g/m^2', 'long_name': 'liquid water path'},
        'iwc': {'units': 'g/m^3', 'long_name': 'ice water content'},
        'iwp': {'units': 'g/m^2', 'long_name': 'ice water path'},
        'rain': {'units': 'mm', 'long_name': 'accumulated precipitation'},
        'qt': {'units': 'kg/kg', 'long_name': 'total mixing ratio'},
        'pwv': {'units': 'kg/m^2', 'long_name': 'precipitable water vapor'},
        'ctt': {'units': 'K', 'long_name': 'cloud top temperature'}
    }

    exist_theta = False
    exist_qv_rho = False
    # Add temp def theta_to_temp(ds):
    if 'theta' in ds.data_vars:
        exist_theta = True
    if 'qv' in ds.data_vars and 'rho' in ds.data_vars:
        exist_qv_rho = True
        ds['pwv'] = (ds['qv'] * ds['rho'] * delz).sum(dim='nVertLevels')
    if exist_theta:
        if exist_qv_rho:
            qv = np.where(ds['qv'] < 0, 0., ds['qv'])
            theta_m = (1 + rv / rgas * qv) * ds['theta']
            exner = ((rgas / 100000.) * (ds['rho'] * theta_m)) ** rcv
            temperature = theta_m * exner
        ds['temp'] = temperature

    wmr_list = ['qc', 'qr', 'qs', 'qi', 'qg']
    liq_list = ['qc', 'qr']
    ice_list = ['qs', 'qi', 'qg']
    rain_list = ['rainc', 'rainnc']
    
    if all(mr in ds.data_vars for mr in wmr_list):
        tmpvar = 0.
        for mr in wmr_list:
            tmpvar += ds[mr]
        ds['qt'] = tmpvar

    if all(mr in ds.data_vars for mr in liq_list):
        tmpvar = 0.
        # Calculate liquid water content in g/m3 and convert LWP in g/m2
        for mr in liq_list:
            tmpvar += (ds[mr] * ds['rho']) * 1000.
        ds['lwc'] = tmpvar
        ds['lwp'] = (tmpvar * delz).sum(dim='nVertLevels')

    if all(mr in ds.data_vars for mr in ice_list):
        tmpvar = 0.
        # Calculate ice water content in g/m3 first and convert IWP in g/m2
        for mr in ice_list:
            tmpvar += (ds[mr] * ds['rho']) * 1000.
        ds['iwc'] = tmpvar
        ds['iwp'] = (tmpvar * delz).sum(dim='nVertLevels')

    if all(var in ds.data_vars for var in rain_list):
        ds['rain'] = ds['rainc'] + ds['rainnc']

    if 'olrtoa' in ds.data_vars:
        print('Creating ctt')
        # constants
        aa = 1.228
        bb = -1.106e-3  # K−1
        # Planck constant
        sigma = 5.670374419e-8  # W⋅m−2⋅K−4

        # flux equivalent brightness temperature
        Tf = (abs(ds['olrtoa']) / sigma) ** (1.0 / 4)
        ds['ctt'] = (((aa**2 + 4 * bb * Tf) ** (1.0 / 2)) - aa) / (2 * bb)

    for var, attr in attrs_dict.items():
        if var in ds.data_vars:
            ds[var] = ds[var].assign_attrs(attr)

    return ds


def plot_mpas2d(da, ax=None, regions=None, colormap=None,
                projection=ccrs.PlateCarree(), **kwargs):
    '''
      da : 2D ux.UxDataArray
      ax : existing cartopy axis (optional)
    '''

    if ax is not None:
        plot_panels = True
    else:
        plot_panels = False
        
    plotdiff = kwargs.get('plotdiff', False)
    varmin = kwargs.get('vmin', None)
    varmax = kwargs.get('vmax', None)
    figsize = kwargs.get('figsize', (5, 4))
    extent = kwargs.get('extent', None)
    axes_pos = kwargs.get('axes_pos', [0.05, 0.1, 0.85, 0.8])
    latlonlb = kwargs.get('latlonlb', False)
    gridline = kwargs.get('gridline', False)
    title = kwargs.get('title', None)

    plot_cbar = kwargs.get('colorbar', False)
    user_cb_opt = kwargs.get('cb_opt', {})
    default_cb_opt = dict(
        orientation="vertical",
        extend='max',
        fraction=0.03,
        pad=0.04,
        label='',
    )
    cb_opt = default_cb_opt | user_cb_opt

    cmap = plt.get_cmap(colormap)

    if plotdiff:
        if cb_opt['extend'] == 'max':
            cb_opt['extend'] = 'both'
    else:
        if da.name in ['refl10cm_max', 'lwc', 'lwp', 'iwc', 'iwp', 'rain', 'rainc', 'rainnc']:
            cmap.set_under('white')
            if da.name.startswith('rain'):
                varmin = 0.05

    if varmax is None:
        varmax = da.quantile(.99).data
    if varmin is None:
        varmin = da.quantile(.01).data
        if varmin == 0.:
            cmap.set_under('white')
            varmin = (varmax - varmin)/100.

    # -------------------------
    # Create axes only if needed
    # -------------------------
    if not plot_panels:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw=dict(projection=projection),
            gridspec_kw=dict(
                left=axes_pos[0],
                right=axes_pos[0]+axes_pos[2],
                bottom=axes_pos[1],
                top=axes_pos[1]+axes_pos[3]
            )
        )
    else:
        fig = ax.figure

    # -------------------------
    # Extent
    # -------------------------
    if extent is None:
        lonData = ((da.uxgrid.face_lon + 180) % 360) - 180
        latData = da.uxgrid.face_lat
        extent = (
            lonData.min().item(),
            lonData.max().item(),
            latData.min().item(),
            latData.max().item()
        )

    ax.set_extent(extent, crs=projection)
    ax.add_feature(cft.BORDERS.with_scale('50m'), color='grey', linestyle='--', linewidth=0.5)
    ax.add_feature(cft.STATES.with_scale('50m'), edgecolor='grey', linewidth=0.5)
    ax.coastlines()

    plotdata = da.to_raster(ax=ax)

    img = ax.imshow(
        plotdata,
        vmin=varmin,
        vmax=varmax,
        cmap=cmap,
        origin="lower",
        extent=ax.get_xlim() + ax.get_ylim()
    )

    if regions is not None:
        ax.add_geometries(regions.geometry, crs=projection,
                          edgecolor='magenta', facecolor='none')

    # colorbar tied to this axis
    if plot_cbar:
        fig.colorbar(img, ax=ax, **cb_opt)

    if latlonlb:
        gl = ax.gridlines(draw_labels=True, dms=True,
                          x_inline=False, y_inline=False)
        gl.right_labels = False
        gl.top_labels = False
        gl.xlines = gridline
        gl.ylines = gridline

    ax.set_title(title, loc='right')

    if da.name == 'rain':
        substr = f'Min={np.nanmin(plotdata):.2f}, Max={np.nanmax(plotdata):.2f},\nTotal={np.nansum(plotdata):.2f}'
    else:
        substr = f'Min={np.nanmin(plotdata):.2f}, Max={np.nanmax(plotdata):.2f}'
    ax.annotate(substr, (0.5, -0.06),
                ha='center', va='center',
                fontsize=12, xycoords='axes fraction')

    return ax, img

def select_mpascells_in_regions(da, regions, projection=ccrs.PlateCarree()):
    regions = regions.set_crs("EPSG:4326", allow_override=True)
    regions_proj = regions.to_crs(projection)

    region_geom = regions_proj.geometry.union_all()

    cells = da.uxgrid.to_geodataframe(engine='geopandas')
    cells = cells.set_crs("EPSG:4326", allow_override=True)
    cells_proj = cells.to_crs(projection)
    cells_proj["centroid"] = cells_proj.geometry.centroid

    mask = cells_proj["centroid"].within(region_geom)
    cells_in_region = cells_proj[mask]
    
    cell_indices = cells_in_region.index.values

    return da.isel(n_face=cell_indices)