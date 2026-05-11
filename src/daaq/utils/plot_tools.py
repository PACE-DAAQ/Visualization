# utils/plot_tools.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpcrs
import cartopy.crs as ccrs
from scipy import stats
from daaq.utils.colormap import white_gist_earth
from typing import Dict, List, Optional, Sequence, Union

# --- Colorbar defaults ---
cbar_defaults = {
    'show': 'auto',
    'orientation': 'vertical',
    'label': None,
    'shrink': 0.8,
    'pad': 0.04,
    'extend': 'max',
}

# --- Font defaults ---
font_defaults = {
    'weight': 'normal',  # global default for all text below
    'label_weight': None,      # None = inherit 'weight'
    'title_weight': None,
    'stats_weight': None,
    'tick_weight': None,      # x/y tick labels
    'cbar_label_weight': None,
    'cbar_tick_weight': None,
    'label_size': 12,
    'title_size': 13,
    'stats_size': 10,
    'tick_size': 10,
}

def make_panels(
    items,
    ncols=None,
    nrows=None,
    panel_size=(4, 4),
    aspect=None,
    sharex=False,
    sharey=False,
    wspace=0.3,
    hspace=0.3,
    projection=None,
):
    """
    Create a figure with one subplot per item in `items`.
 
    Parameters
    ----------
    items : list
        The list you'll iterate over to fill each panel.
    ncols : int or None
        Number of columns. Inferred from nrows or items length if None.
    nrows : int or None
        Number of rows. Inferred from ncols or items length if None.
    panel_size : (float, float)
        (width, height) of each individual panel in inches.
        Height is overridden if `aspect` is set.
    aspect : float or None
        Height/width ratio of each panel. E.g. 0.5 → landscape, 2 → portrait.
        When set, panel height = panel_size[0] * aspect.
    sharex : bool
        Share x-axis across all panels.
    sharey : bool
        Share y-axis across all panels.
    wspace : float
        Horizontal space between panels (fraction of panel width).
    hspace : float
        Vertical space between panels (fraction of panel height).
    projection : cartopy.crs.Projection or None
        If provided, all panels are created as cartopy GeoAxes with this
        projection. Not compatible with sharex/sharey.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        Flat list, same length as `items`. Any extra axes are hidden.
 
    Examples
    --------
    keys = ["A", "B", "C", "D", "E"]
    fig, axes = make_panels(keys, ncols=3, panel_size=(4, 3))
    for ax, key in zip(axes, keys):
        ax.set_title(key)
        ax.plot([1, 2, 3])
    plt.show()
    """
    n = len(items)
    if n == 0:
        raise ValueError("`items` must not be empty.")
 
    if projection is not None and (sharex or sharey):
        import warnings
        warnings.warn(
            "sharex/sharey ignored when projection is set; "
            "use a consistent cornerlatlon across panels instead.",
            stacklevel=2,
        )
        sharex = sharey = False

    # --- resolve grid dimensions ---
    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
 
    if nrows * ncols < n:
        raise ValueError(
            f"Grid {nrows}×{ncols} = {nrows*ncols} cells, but {n} items were given."
        )
 
    # --- resolve panel dimensions ---
    pw = panel_size[0]
    ph = panel_size[1] if aspect is None else pw * aspect
 
    fig_w = pw * ncols
    fig_h = ph * nrows

    subplot_kw = {'projection': projection} if projection is not None else {}
 
    fig, ax_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,          # always 2-D array
        subplot_kw=subplot_kw,
    )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
 
    # flatten to a 1-D list for easy iteration
    axes_flat = ax_grid.flatten().tolist()
 
    # hide extra (empty) panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)
 
    return fig, axes_flat[:n]

def plt_hist2d(
    dataframe, x, y, axis,
    ax=None,
    save=False, savename=None,
    **kwargs,
):
    """
    2D density scatter with 1:1 line, regression, and statistics.

    Parameters
    ----------
    dataframe : pd.DataFrame
    x, y : str
        Column names for x and y axes.
    axis : array-like
        Bin edges for the 2D histogram (shared for both axes).
    ax : matplotlib.axes.Axes or None
        If None, creates its own figure. If given, draws on it.
    save : bool
        Save the figure (only when ax is None).
    savename : str or None
        Output filename when saving.

    kwargs
    ------
    xlb, ylb : str
        Axis labels (default: column names).
    vmin, vmax : float
        View range (default: 1st/99th percentile of x).
    in_log : bool
        Log-scale both axes.
    stats : bool
        Annotate statistics on the plot.
    reg_line : bool
        Add linear regression line and annotate slope/intercept.
    color_log : bool
        Use log color scale for density (default False).
    title : str
        Axis title.
    figsize : tuple
        Figure size when standalone (default (4, 4)).
    dpi : int
        Save DPI (default 300).
    stats_loc : {'upper left', 'upper right', 'lower left', 'lower right'}
        Where to place stats annotations (default 'upper left').
    normalize : {'counts', 'density', 'probability', 'column'}
        How to normalize the 2D histogram (default 'counts').
    cbar_dict : dict or None
        Colorbar configuration. Set to None or an empty dict to disable.
        Supported keys:
            show : bool or 'auto'
                Whether to draw a colorbar. 'auto' draws one only in standalone
                mode. Default 'auto'.
            orientation : {'vertical', 'horizontal'}
                Colorbar orientation (default 'vertical').
            label : str or None
                Override the default colorbar label. If None, uses the label
                derived from `normalize`.
            shrink : float
                Shrink factor (default 0.8).
            pad : float
                Padding between axes and colorbar (default 0.04).
            extend : {'neither', 'both', 'min', 'max'}
                Which ends to extend (default 'max').

    Returns
    -------
    ax : matplotlib axis
    mesh : QuadMesh
        The pcolormesh object, useful for attaching a colorbar.
    stats_out : dict
        Dictionary of computed statistics.
    """

    # --- Parse kwargs ---
    xlbstr = kwargs.get('xlb', None)
    ylbstr = kwargs.get('ylb', None)
    in_log = kwargs.get('in_log', False)
    show_stats = kwargs.get('stats', False)
    show_regline = kwargs.get('reg_line', False)
    show_extrastat = kwargs.get('extrastats', False)
    ax_title = kwargs.get('ax_title', None)
    figsize = kwargs.get('figsize', (4, 4))
    quality = kwargs.get('dpi', 300)
    stats_loc = kwargs.get('stats_loc', 'upper left')
    normalize = kwargs.get('normalize', 'counts')

    # --- Color range control ---
    color_log = kwargs.get('color_log', None)
    color_vmin = kwargs.get('color_vmin', None)
    color_vmax = kwargs.get('color_vmax', None)

    cbar_dict = {**cbar_defaults, **(kwargs.get('cbar_dict') or {})}
    font_dict = {**font_defaults, **(kwargs.get('font_dict') or {})}

    # --- Drop NaNs / Infs ---
    sub = dataframe[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 2:
        raise ValueError(f"Not enough valid points to plot ({len(sub)} after NaN drop).")

    x_data = sub[x].to_numpy()
    y_data = sub[y].to_numpy()

    # --- View range ---
    valmin = kwargs.get('ax_min', np.percentile(x_data, 1))
    valmax = kwargs.get('ax_max', np.percentile(x_data, 99))

    # --- Ax handling ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.92)
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    # Resolve 'auto' to a concrete boolean
    show_cbar = cbar_dict['show']
    if show_cbar == 'auto':
        show_cbar = standalone

    # --- 2D histogram with chosen normalization ---
    if normalize == 'density':
        hist2d, x_edge, y_edge = np.histogram2d(
            x_data, y_data, bins=axis, density=True,
        )
        default_cbar_label = 'Density'
    elif normalize == 'probability':
        hist2d, x_edge, y_edge = np.histogram2d(x_data, y_data, bins=axis)
        total = hist2d.sum()
        if total > 0:
            hist2d = hist2d / total
        default_cbar_label = 'Probability'
    elif normalize == 'column':
        hist2d, x_edge, y_edge = np.histogram2d(x_data, y_data, bins=axis)
        col_sums = hist2d.sum(axis=1, keepdims=True)
        col_sums[col_sums == 0] = 1
        hist2d = hist2d / col_sums
        default_cbar_label = 'P(y | x)'
    else:  # 'counts'
        hist2d, x_edge, y_edge = np.histogram2d(x_data, y_data, bins=axis)
        default_cbar_label = 'Counts'

    if hist2d.max() == 0:
        raise ValueError("Histogram is empty — check data range vs bin edges.")

    # --- Build color normalization ---
    clrnorm, hist2d_plot = _build_color_norm(
        hist2d, normalize, color_log=color_log,
        vmin=color_vmin, vmax=color_vmax,
    )

    # --- pcolormesh (correct for discrete bins, faster than contourf) ---
    mesh = ax.pcolormesh(
        x_edge, y_edge, hist2d_plot.T,
        norm=clrnorm, cmap=white_gist_earth, shading='flat',
    )

    # --- 1:1 reference line ---
    ax.plot([valmin, valmax], [valmin, valmax],
            color='gray', linewidth=1.5, linestyle='--', zorder=3)

    ax.set_xlim(valmin, valmax)
    ax.set_ylim(valmin, valmax)
    if in_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    if xlbstr:
        ax.set_xlabel(xlbstr, fontsize=font_dict['label_size'], fontweight=_font_weight(font_dict, 'label_weight'))
    if ylbstr:
        ax.set_ylabel(ylbstr, fontsize=font_dict['label_size'], fontweight=_font_weight(font_dict, 'label_weight'))
    if ax_title:
        ax.set_title(ax_title, fontsize=font_dict['title_size'], fontweight=_font_weight(font_dict, 'title_weight'))

    # --- Tick labels ---
    ax.tick_params(axis='both', labelsize=font_dict['tick_size'])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight(_font_weight(font_dict, 'tick_weight'))

    # --- Statistics ---
    slope, intercept, r, p, se = stats.linregress(x_data, y_data)
    mean_x = np.mean(x_data)
    bias = np.mean(y_data) - mean_x
    rbias = bias / mean_x if abs(mean_x) > 1e-12 else np.nan
    rmse = np.sqrt(np.mean((y_data - x_data) ** 2))

    stats_out = {
        'count': len(x_data),
        'bias': bias,
        'rbias': rbias,
        'rmse': rmse,
        'r_squared': r ** 2,
        'slope': slope,
        'intercept': intercept,
        'p_value': p,
        'std_err': se,
    }

    # --- Stats annotations ---
    display_stats = {
        'Counts': f"{stats_out['count']:,}",
        'Bias': f"{stats_out['bias']:.3f}",
        'RMSE': f"{stats_out['rmse']:.3f}",
        'R$^2$': f"{stats_out['r_squared']:.3f}",
    }
    if show_extrastat:
        display_stats['Rel. Bias'] = f"{stats_out['rbias']:.3f}"

    # --- Regression line (clipped to view) ---
    if show_regline:
        xline = np.array([valmin, valmax])
        ax.plot(xline, intercept + slope * xline, 'k-', linewidth=1.5, zorder=3)
        display_stats.update({
            'Intercept': f"{stats_out['intercept']:.3f}",
            'Slope': f"{stats_out['slope']:.3f}",
        })

    _annotate_stats(
        ax, display_stats if show_stats else {},
        fontsize=font_dict['stats_size'],
        fontweight=_font_weight(font_dict, 'stats_weight'),
        loc=stats_loc,
    )

    # --- Colorbar ---
    if show_cbar:
        add_colorbar(
            fig, ax, mesh,
            cbar_dict=cbar_dict,
            font_dict=font_dict,
            default_label=default_cbar_label,
            cbar_label_weight=_font_weight(font_dict, 'cbar_label_weight'),
            cbar_tick_weight=_font_weight(font_dict, 'cbar_tick_weight'),
        )

    # --- Save ---
    if save and standalone:
        fig.savefig(savename, dpi=quality, bbox_inches='tight')
        plt.close(fig)

    return ax, mesh, stats_out

def _annotate_stats(ax, display_stats, loc='upper left',
                    fontsize=12, fontweight='normal'):
    """Place stats text block at one of the four corners, inside the axes."""
    if not display_stats:
        return
    
    pad = 0.03
    line_h = 0.05
    positions = {
        'upper left':  (pad,       1 - pad, 'left',  'top'),
        'upper right': (1 - pad,   1 - pad, 'right', 'top'),
        'lower left':  (pad,       pad,     'left',  'bottom'),
        'lower right': (1 - pad,   pad,     'right', 'bottom'),
    }
    x0, y0, ha, va = positions[loc]
    direction = -1 if va == 'top' else 1

    for i, (k, v) in enumerate(display_stats.items()):
        y = y0 + direction * i * line_h
        ax.annotate(
            f'{k} = {v}', (x0, y),
            xycoords='axes fraction',
            ha=ha, va=va,
            fontsize=fontsize,
            fontweight=fontweight,
        )

def _build_color_norm(hist2d, normalize, color_log, vmin=None, vmax=None):
    """
    Build an appropriate color normalization for the given histogram type.

    Parameters
    ----------
    hist2d : np.ndarray
        The 2D histogram array.
    normalize : {'counts', 'density', 'probability', 'column'}
        The normalization type applied to the histogram.
    color_log : bool
        Whether to use log color scale.
    vmin, vmax : float or None
        User overrides for the color scale range.

    Returns
    -------
    (norm, hist2d_for_plot)
        A matplotlib Normalize instance and the histogram array to plot
        (masked if log scale is used).
    """
    # Per-mode sensible defaults
    mode_defaults = {
        'counts': {
            'log': True,
            'vmin': None,   # auto — smallest nonzero value
            'vmax': None,   # auto — hist.max()
        },
        'density': {
            'log': True,
            'vmin': None,
            'vmax': None,
        },
        'probability': {
            'log': True,
            'vmin': None,
            'vmax': None,
        },
        'column': {
            'log': False,   # bounded [0, 1] — linear reads better
            'vmin': 0.0,
            'vmax': 1.0,
        },
    }
    defaults = mode_defaults.get(normalize, mode_defaults['counts'])

    # User's color_log overrides the mode default
    use_log = color_log if color_log is not None else defaults['log']

    # Resolve vmin/vmax — explicit user values win over mode defaults
    nonzero = hist2d[hist2d > 0]
    has_nonzero = nonzero.size > 0

    if vmax is None:
        vmax = defaults['vmax'] if defaults['vmax'] is not None else (
            hist2d.max() if has_nonzero else 1.0
        )

    if vmin is None:
        if defaults['vmin'] is not None:
            vmin = defaults['vmin']
        elif use_log:
            vmin = nonzero.min() if has_nonzero else vmax / 1000.0
        else:
            vmin = 0.0

    # Guard against degenerate ranges
    if vmin >= vmax:
        vmin = vmax / 10.0 if use_log else 0.0
        if vmin >= vmax:  # still bad (vmax == 0)
            vmax = vmin + 1e-10

    # Build norm and plot array
    if use_log:
        hist2d_plot = np.ma.masked_where(hist2d <= 0, hist2d)
        # LogNorm requires strictly positive vmin
        if vmin <= 0:
            vmin = nonzero.min() if has_nonzero else 1e-10
        norm = mpcrs.LogNorm(vmin=vmin, vmax=vmax)
    else:
        hist2d_plot = hist2d
        norm = mpcrs.Normalize(vmin=vmin, vmax=vmax)

    return norm, hist2d_plot

def _font_weight(font_dict, key):
    """Resolve a per-element weight, falling back to global 'weight'."""
    return font_dict[key] if font_dict[key] is not None else font_dict['weight']

def add_colorbar(fig, ax, mesh, cbar_dict=None, font_dict=None, default_label='',
                 cbar_label_weight='normal', cbar_tick_weight='normal'):
    """
    Attach a styled colorbar to a matplotlib mappable.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure.
    ax : matplotlib.axes.Axes
        Axes the colorbar is associated with.
    mesh : ScalarMappable
        The mappable returned by pcolormesh, imshow, contourf, etc.
    cbar_dict : dict or None
        Keys: 'label', 'orientation', 'shrink', 'pad', 'extend'.
        'label' may be None or '' to fall back to `default_label`.
    font_dict : dict or None
        Keys: 'label_size', 'tick_size'.
    default_label : str, optional
        Used when cbar_dict['label'] is falsy.
    cbar_label_weight, cbar_tick_weight : str, optional
        Font weights for the label and tick labels (e.g. 'normal', 'bold').

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    cbar_dict = {**cbar_defaults, **(cbar_dict or {})}
    font_dict = {**font_defaults, **(font_dict or {})}
    label = cbar_dict['label'] or default_label

    cbar_obj = fig.colorbar(
        mesh, ax=ax,
        orientation=cbar_dict['orientation'],
        shrink=cbar_dict['shrink'],
        pad=cbar_dict['pad'],
        extend=cbar_dict['extend'],
    )
    cbar_obj.set_label(
        label,
        fontsize=font_dict['label_size'],
        fontweight=cbar_label_weight,
    )
    cbar_obj.ax.tick_params(labelsize=font_dict['tick_size'])
    for lbl in cbar_obj.ax.get_xticklabels() + cbar_obj.ax.get_yticklabels():
        lbl.set_fontweight(cbar_tick_weight)

    return cbar_obj

def setupax_2dmap(
        ax=None,
        cornerlatlon=None,
        projection=ccrs.PlateCarree(),
        *,
        figsize: tuple = (8, 4),
        show_gridlines: bool = True,
        gl_lbsize: int = 20,
        # **kwargs,
    ):
    """
    Set up a cartopy GeoAxes with coastlines, extent, and gridline labels.

    Parameters
    ----------
    ax : cartopy GeoAxes or None
        If None, a new figure and axes are created.
    cornerlatlon : (minlat, maxlat, minlon, maxlon) or None
        Map extent. If None, the map is set to global.
    projection : cartopy.crs.Projection
        Used both as the axes projection (when ax is None) and as the CRS
        for `set_extent`. Default: PlateCarree.
    lbsize : float
        Font size for gridline labels. Default 20.
    
    kwargs
    ------
    figsize: default (8,4)
    show_gridlines: default True
    gl_lbsize: default 20

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : cartopy GeoAxes
    gl : cartopy Gridliner
    """

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={'projection': projection},
        )
    else:
        fig = ax.figure

    ax.coastlines(resolution='110m')

    if cornerlatlon is None:
        ax.set_global()
    else:
        minlat, maxlat, minlon, maxlon = cornerlatlon
        ax.set_extent((minlon, maxlon, minlat, maxlat), crs=projection)

    gl = None
    if show_gridlines:
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=2)
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {'size':gl_lbsize}
        gl.ylabel_style = {'size':gl_lbsize}

    return fig, ax, gl

def plt_boxplot_by_cycle(
    df, cycle_col, value_col, ax=None, save=False, savename=None, **kwargs
):
    """
    Box plot of `value_col` grouped by `cycle_col`, with cycles on the x-axis.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format frame: one row per observation, with a cycle column
        (datetime-like or convertible via pd.to_datetime) and a value column.
    cycle_col : str
        Column holding cycle timestamps (e.g. '2024010100', '2024010106', ...).
    value_col : str
        Column holding the values to box-plot.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw into. If None, a new figure/axis is created.
    save : bool
        If True, save the figure to `savename` (only when ax is None).
    savename : str
        Output path when save=True.

    Keyword Arguments
    -----------------
    figsize : tuple, default (8, 4)
    dpi : int, default 300
    ylb : str, default value_col
    xlb : str, default 'Cycle'
    title : str, default None
    date_fmt : str, default '%Y-%m-%d %HZ'
    tick_rotation : float, default 30
    max_xticks : int, default 10
        Upper bound on number of x-tick labels shown. Ticks are thinned to
        an even stride so labels don't overlap.
    showfliers : bool, default True
    widths : float, default 0.6
    box_color : str, default 'tab:blue'
    mean_color : str, default 'red'
    show_n : bool, default False
    grid : bool, default True

    Returns
    -------
    ax : matplotlib.axes.Axes
    artists : dict
    """
    # --- options ---
    figsize       = kwargs.get('figsize', (8, 4))
    quality       = kwargs.get('dpi', 300)
    ylbstr        = kwargs.get('ylb', value_col)
    xlbstr        = kwargs.get('xlb', 'Cycle')
    title         = kwargs.get('title', None)
    date_fmt      = kwargs.get('date_fmt', '%Y-%m-%d %HZ')
    tick_rotation = kwargs.get('tick_rotation', 30)
    max_xticks    = kwargs.get('max_xticks', 10)
    showfliers    = kwargs.get('showfliers', True)
    widths        = kwargs.get('widths', 0.6)
    box_color     = kwargs.get('box_color', 'tab:blue')
    mean_color    = kwargs.get('mean_color', 'red')
    show_n        = kwargs.get('show_n', False)
    grid          = kwargs.get('grid', True)

    # --- axis setup ---
    if ax is None:
        standalone = True
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_position([0.12, 0.18, 0.83, 0.72])
    else:
        standalone = False
        fig = ax.figure

    # --- group by cycle, preserve chronological order ---
    work = df[[cycle_col, value_col]].copy()
    work[cycle_col] = pd.to_datetime(work[cycle_col])
    work = work.dropna(subset=[value_col])

    grouped = work.groupby(cycle_col, sort=True)[value_col]
    cycles = list(grouped.groups.keys())
    data = [grouped.get_group(c).values for c in cycles]

    if not data:
        raise ValueError("No data to plot after dropping NaNs.")

    # --- box plot ---
    positions = np.arange(len(cycles))
    artists = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        showmeans=True,
        showfliers=showfliers,
        meanprops=dict(marker='D', markerfacecolor=mean_color,
                       markeredgecolor=mean_color, markersize=5),
        medianprops=dict(color='black', linewidth=1.2),
        boxprops=dict(color=box_color, linewidth=1.2),
        whiskerprops=dict(color=box_color, linewidth=1.0),
        capprops=dict(color=box_color, linewidth=1.0),
        flierprops=dict(marker='.', markersize=3, alpha=0.4,
                        markeredgecolor='gray'),
        patch_artist=False,
    )

    # --- x-axis: thin ticks to ~max_xticks evenly spaced ---
    n = len(cycles)
    stride = max(1, int(np.ceil(n / max_xticks)))
    tick_idx = np.arange(0, n, stride)
    # ensure the last cycle is shown so the axis range is bookended cleanly
    if tick_idx[-1] != n - 1:
        tick_idx = np.append(tick_idx, n - 1)

    ax.set_xticks(positions[tick_idx])
    ax.set_xticklabels(
        [cycles[i].strftime(date_fmt) for i in tick_idx],
        rotation=tick_rotation,
        ha='right' if tick_rotation else 'center',
    )
    # keep all box positions visible even though only some are labeled
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)

    ax.set_xlabel(xlbstr, fontsize=12)
    ax.set_ylabel(ylbstr, fontsize=12)
    if title is not None:
        ax.set_title(title, loc='right')

    if grid:
        ax.grid(alpha=0.5, axis='y')

    # --- optional: sample size above each box ---
    if show_n:
        ymax = max(np.nanmax(d) for d in data)
        ymin = min(np.nanmin(d) for d in data)
        offset = (ymax - ymin) * 0.02
        for pos, d in zip(positions, data):
            ax.annotate(f'n={len(d)}', (pos, ymax + offset),
                        ha='center', va='bottom', fontsize=9)

    if save and standalone:
        plt.savefig(savename, dpi=quality, bbox_inches='tight')
        plt.close(fig)

    return ax, artists