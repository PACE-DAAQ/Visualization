# utils/date_tools.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcrs
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import stats
from daaq.utils.colormap import white_gist_earth

def plt_hist2d(dataframe, x, y, axis, save, savename, **kwargs):
    x_data = dataframe[x]
    y_data = dataframe[y]
    xlbstr = kwargs.get('xlb', x)
    ylbstr = kwargs.get('ylb', y)
    in_log = kwargs.get('in_log', False)

    vmin = np.percentile(x_data, 1.)
    vmax = np.percentile(x_data, 99.)
    valmin = kwargs.get('vmin', vmin)
    valmax = kwargs.get('vmax', vmax)

    show_stats = kwargs.get('stats', False)
    show_regline = kwargs.get('reg_line', False)
    figsize = kwargs.get('figsize', (4, 4))
    quality = kwargs.get('dpi', 300)
    
    hist2d, x_edge, y_edge = np.histogram2d(
        x_data,
        y_data,
        bins=axis,
    )

    cnlvs = np.linspace(0, hist2d.max(), 256)
    clrnorm = mpcrs.BoundaryNorm(cnlvs, len(cnlvs), extend='max')

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position([0.15, 0.12, 0.8, 0.8])
    cn = ax.contourf(axis[:-1], axis[:-1], hist2d.swapaxes(0,1),
                     levels=cnlvs, norm=clrnorm, cmap=white_gist_earth,
                     extend='max',)
    plt.plot(
        [valmin, valmax],
        [valmin, valmax],
        color='gray',
        linewidth=2,
        linestyle='--'
    )
    plt.xlim(valmin, valmax)
    plt.ylim(valmin, valmax)

    if in_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_aspect('equal')

    plt.grid(alpha=0.5)
    plt.xlabel(xlbstr, fontsize=12)
    plt.ylabel(ylbstr, fontsize=12)

    # Calculate statistics and place it on the top-left corner
    slope, intercept, r, p, se = stats.linregress(x_data, y_data)
    r_squared = r ** 2
    bias = np.mean(y_data) - np.mean(x_data)
    rbias = bias/np.mean(x_data)
    ssize = len(x_data)
    rmse = np.sqrt(np.mean((y_data - x_data) ** 2))
    stats_dict = {
        'Counts': str("%.0f" % ssize),
        'Absolute Bias': str("%.3f" % bias),
        'Relative Bias': str("%.3f" % rbias),
        'RMSE': str("%.3f" % rmse),
        # 'R': str("%.3f" % r),
        'R$^{2}$': str("%.3f" % r_squared),
    }
    regstats_dict = {
        'Intercept': str("%.3f" % intercept),
        'Slope': str("%.3f" % slope),
    }
    
    x_pos = 0.012
    y_pos = 1.02
    if show_stats:
        for key in stats_dict.keys():
            stat_str = '%s= %s' % (key, stats_dict[key])
            y_pos = y_pos - 0.05
            ax.annotate(stat_str, (x_pos, y_pos), ha='left', va='center', 
                        fontsize=12, xycoords='axes fraction')
    if show_regline:
        for key in regstats_dict.keys():
            stat_str = '%s= %s' % (key, regstats_dict[key])
            y_pos = y_pos - 0.05
            ax.annotate(stat_str, (x_pos, y_pos), ha='left', va='center', 
                        fontsize=12, xycoords='axes fraction')
        plt.plot(axis, intercept + slope * axis, 'k')
    
    #cb = plt.colorbar(cn, orientation='horizontal', fraction=0.03, aspect=30, 
    #                  pad=0.12, extend='max', ticks=cnlvs[::50])
    #cb.ax.minorticks_off()
    #cb.ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0),
    #                       useMathText=True)

    if save:
        plt.savefig(savename, dpi=quality)
        plt.close(fig)
    return

def setupax_2dmap(cornerlatlon=None, projection=ccrs.PlateCarree(), lbsize=None):
    if not lbsize: lbsize=20

    fig=plt.figure()
    ax=plt.subplot(projection=projection)
    ax.coastlines(resolution='110m')

    if cornerlatlon is None:
        ax.set_global()
    else:
        minlat=cornerlatlon[0]; maxlat=cornerlatlon[1]
        minlon=cornerlatlon[2]; maxlon=cornerlatlon[3]
        ax.set_extent((minlon, maxlon, minlat, maxlat), crs=projection)

    gl=ax.gridlines(draw_labels=True,dms=True,x_inline=False, y_inline=False)
    gl.right_labels=False
    gl.top_labels=False
    # gl.xformatter=LongitudeFormatter(degree_symbol=u'\u00B0 ')
    # gl.yformatter=LatitudeFormatter(degree_symbol=u'\u00B0 ')
    gl.xlabel_style={'size':lbsize}
    gl.ylabel_style={'size':lbsize}

    return fig, ax, gl