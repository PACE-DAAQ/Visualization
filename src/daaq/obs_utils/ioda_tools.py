# obs_utils/ioda_tools.py
from __future__ import annotations
 
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
 
import netCDF4 as nc
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
 
 
class IODAFile:
    """
    Reader for a single IODA v2 NetCDF4 observation file.
 
    Designed for interactive use in notebooks and for scripted pipelines.
    The configuration methods (`select_groups`, `select_variables`, …) return
    ``self`` so calls can be chained.
    """
 
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
 
    def __init__(
        self,
        filepath: Union[str, Path],
        *,
        metadata_group: str = "MetaData",
        metadata_prefix: bool = False,
        nlocs_dim: str = "Location",
        channel_dim: Optional[str] = "Channel",
        datetime_col: Optional[str] = "dateTime",
        parse_datetime: bool = True,
    ) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"IODA file not found: {self.filepath}")
 
        self.metadata_group = metadata_group
        self.metadata_prefix = metadata_prefix
        self.nlocs_dim = nlocs_dim
        self.channel_dim = channel_dim
        self.datetime_col = datetime_col
        self.parse_datetime = parse_datetime
 
        self._groups: Optional[List[str]] = None
        self._variables: Optional[Dict[str, List[str]]] = None
        self._string_fill: str = ""

        self._schema: Optional[Dict[str, List[str]]] = None
        self.channel_meta, self._nchans = self._load_channel_meta()
        self._channel_index: Optional[List[int]] = list(range(self._nchans)) if self._nchans is not None else None
 
    # ------------------------------------------------------------------
    # Chainable configuration (unchanged)
    # ------------------------------------------------------------------
 
    def select_groups(self, *groups: str) -> "IODAFile":
        self._groups = list(groups)
        self._variables = None
        return self
 
    def select_variables(self, group: str, variables: List[str]) -> "IODAFile":
        if self._variables is None:
            self._variables = {}
        self._variables[group] = list(variables)
        self._groups = None
        return self
 
    def select_channel(self, index: List[int]) -> "IODAFile":
        if self._nchans is not None:
            invalid = [i for i in index if not (0 <= i < self._nchans)]
            if invalid:
                raise ValueError(f"Channel indices {invalid} out of range [0, {self._nchans})")
        self._channel_index = index
        return self
 
    def set_string_fill(self, value: str) -> "IODAFile":
        self._string_fill = value
        return self
 
    def reset(self) -> "IODAFile":
        self._groups = None
        self._variables = None
        self._channel_index = list(range(self._nchans)) if self._nchans is not None else None
        return self
 
    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
 
    def schema(self) -> Dict[str, List[str]]:
        if self._schema is None:
            with nc.Dataset(self.filepath, "r") as ds:
                self._schema = {
                    g: list(grp.variables.keys())
                    for g, grp in ds.groups.items()
                }
        return self._schema
 
    def info(self, *, verbose: bool = True) -> str:
        lines = [f"IODA file : {self.filepath.name}"]
        for grp, vars_ in self.schema().items():
            lines.append(f"  {grp}")
            for v in vars_:
                lines.append(f"    - {v}")
        text = "\n".join(lines)
        if verbose:
            print(text)
        return text
 
    def groups(self) -> List[str]:
        return list(self.schema().keys())
 
    def variables(self, group: str) -> List[str]:
        return self.schema().get(group, [])
 
    # ------------------------------------------------------------------
    # Loading (unchanged from your version — omitted here for brevity
    # in the explanation, but kept intact in the file)
    # ------------------------------------------------------------------
 
    def load(self) -> pd.DataFrame:
        columns: Dict[str, pd.Series] = {}
 
        with nc.Dataset(self.filepath, "r") as ds:
            load_map = self._resolve_load_map(ds)
 
            nlocs: Optional[int] = None
            if self.nlocs_dim in ds.dimensions:
                nlocs = ds.dimensions[self.nlocs_dim].size

            for grp_name, var_names in load_map.items():
                if grp_name not in ds.groups:
                    continue
                grp = ds.groups[grp_name]
                is_meta = grp_name == self.metadata_group
 
                for var_name in var_names:
                    if var_name not in grp.variables:
                        continue
 
                    var = grp.variables[var_name]
                    raw = var[:]
 
                    if (
                        self.channel_dim is not None
                        and self.channel_dim in var.dimensions
                    ):
                        if not is_meta and self._nchans is not None:
                            chan_ax = var.dimensions.index(self.channel_dim)
                            ch_slices = [slice(None)] * var.ndim
 
                            for chidx in self._channel_index:
                                ch_slices[chan_ax] = chidx
                                series = _to_series(
                                    raw[tuple(ch_slices)], var, nlocs, self._string_fill
                                )
                                col_name = (
                                    var_name
                                    if (is_meta and not self.metadata_prefix)
                                    else f"{grp_name}_{var_name}_{chidx}"
                                )
                                columns[col_name] = series
                        else:
                            continue
                    else:
                        series = _to_series(raw, var, nlocs, self._string_fill)
                        col_name = (
                            var_name
                            if (is_meta and not self.metadata_prefix)
                            else f"{grp_name}_{var_name}"
                        )
                        columns[col_name] = series
 
        if not columns:
            return pd.DataFrame()
 
        df = pd.DataFrame(columns)
 
        if self.parse_datetime and self.datetime_col and self.datetime_col in df.columns:
            df = _parse_datetime(df, self.datetime_col)
 
        return df
 
    # ==================================================================
    # Single-file subset helpers  (crop / thin / subset)
    # ==================================================================
 
    def crop(
        self,
        output: Union[str, Path],
        poly_file: Union[str, Path],
        *,
        lat_col: str = "Lat",
        lon_col: str = "Lon",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
    ) -> int:
        """
        Write a spatial subset of this file using a polygon defined in a CSV.
 
        The CSV must contain columns named ``Lat`` and ``Lon`` (configurable),
        one row per polygon vertex, in order.  Observations whose
        ``MetaData/latitude`` and ``MetaData/longitude`` fall inside that
        polygon are kept.
 
        Parameters
        ----------
        output : str | Path
            Destination NetCDF4 path.
        poly_file : str | Path
            CSV file of polygon vertices.
        lat_col, lon_col : str
            Column names in the polygon CSV.  Defaults: ``"Lat"``, ``"Lon"``.
 
        Returns
        -------
        int
            Number of observations written.
        """
        with nc.Dataset(self.filepath, "r") as ds:
            meta = ds.groups[self.metadata_group]
            lat = np.asarray(meta.variables[lat_var][:]).ravel()
            lon = np.asarray(meta.variables[lon_var][:]).ravel()

        mask = self._polygon_mask(lat, lon, poly_file, lat_col=lat_col, lon_col=lon_col)
        return self._write_subset(output, mask)
 
    def thin(
        self,
        output: Union[str, Path],
        thinning_ratio: float,
        *,
        seed: Optional[int] = None,
    ) -> int:
        """
        Write a randomly thinned subset of this file.
 
        ``thinning_ratio`` is the fraction of observations to *discard*, so
        ``thinning_ratio=0.9`` keeps ~10% of obs, matching the JEDI / IODA
        convention used elsewhere in the stack.
 
        Parameters
        ----------
        output : str | Path
            Destination NetCDF4 path.
        thinning_ratio : float
            Value in ``[0, 1]``.  Fraction of obs to drop.
        seed : int | None
            Optional RNG seed for reproducibility.
 
        Returns
        -------
        int
            Number of observations written.
        """
        if not 0.0 <= thinning_ratio <= 1.0:
            raise ValueError(
                f"thinning_ratio must be in [0, 1], got {thinning_ratio}"
            )
 
        with nc.Dataset(self.filepath, "r") as ds:
            n = ds.dimensions[self.nlocs_dim].size
 
        rng = np.random.default_rng(seed)
        mask = rng.uniform(size=n) > thinning_ratio
        return self._write_subset(output, mask)
 
    def subset(self, output: Union[str, Path], mask: np.ndarray) -> int:
        """
        Write an arbitrary boolean-masked subset.  Bring your own mask.
 
        Parameters
        ----------
        output : str | Path
            Destination NetCDF4 path.
        mask : np.ndarray of bool
            Length must equal the source file's Location dimension.
 
        Returns
        -------
        int
            Number of observations written.
        """
        return self._write_subset(output, np.asarray(mask, dtype=bool))
 
    # ------------------------------------------------------------------
    # Subset internals
    # ------------------------------------------------------------------
 
    def _write_subset(self, output: Union[str, Path], mask: np.ndarray) -> int:
        """
        Copy this file to *output*, keeping only locations where ``mask`` is True.
 
        All groups, variables, attributes, and non-location dimensions are
        preserved.  Variables whose leading dimension is *nlocs_dim* are sliced;
        all others are copied verbatim.
        """
        output = Path(output)
 
        with nc.Dataset(self.filepath, "r") as src:
            if self.nlocs_dim not in src.dimensions:
                raise ValueError(
                    f"Source has no '{self.nlocs_dim}' dimension; nothing to subset."
                )
            n_src = src.dimensions[self.nlocs_dim].size
            if n_src == 0:
                raise ValueError("Source file has no observations.")
            if mask.shape != (n_src,):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match Location size ({n_src},)."
                )
 
            n_keep = int(np.count_nonzero(mask))
            if n_keep == 0:
                raise ValueError("Mask selects zero observations.")
 
            print(f"{n_keep} of {n_src} observations selected ({n_keep / n_src:.1%}).")
 
            output.parent.mkdir(parents=True, exist_ok=True)
            with nc.Dataset(output, "w") as dst:
                self._copy_dataset(src, dst, mask)
 
        return n_keep
 
    def _copy_dataset(
        self,
        src: nc.Dataset,
        dst: nc.Dataset,
        mask: np.ndarray,
    ) -> None:
        """Recursively copy src → dst, slicing on nlocs_dim by `mask`."""
        n_keep = int(np.count_nonzero(mask))
 
        # global attributes
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
 
        # dimensions
        for name, dim in src.dimensions.items():
            if name == self.nlocs_dim:
                size = n_keep
            elif dim.isunlimited():
                size = None
            else:
                size = dim.size
            dst.createDimension(name, size)
 
        # root variables (rare in IODA v2, but handle them)
        for name, var in src.variables.items():
            self._copy_variable(var, dst, name, mask)
 
        # groups
        for gname, grp in src.groups.items():
            dgrp = dst.createGroup(gname)
            dgrp.setncatts({k: grp.getncattr(k) for k in grp.ncattrs()})
            for name, var in grp.variables.items():
                self._copy_variable(var, dgrp, name, mask)
 
    def _copy_variable(
        self,
        src_var: nc.Variable,
        dst_parent: Union[nc.Dataset, nc.Group],
        name: str,
        mask: np.ndarray,
    ) -> None:
        """Create dst variable mirroring src, slicing along nlocs_dim if present."""
        # Preserve compression / chunking where possible
        filters = src_var.filters() or {}
        zlib = filters.get("zlib", False)
        complevel = filters.get("complevel", 4)
        shuffle = filters.get("shuffle", False)
 
        fill = None
        if "_FillValue" in src_var.ncattrs():
            fill = src_var.getncattr("_FillValue")
 
        new_var = dst_parent.createVariable(
            name,
            src_var.dtype,
            dimensions=src_var.dimensions,
            zlib=zlib,
            complevel=complevel,
            shuffle=shuffle,
            fill_value=fill,
        )
 
        # Copy attributes (skip _FillValue — handled above)
        new_var.setncatts(
            {k: src_var.getncattr(k) for k in src_var.ncattrs() if k != "_FillValue"}
        )
 
        data = src_var[:]
 
        if self.nlocs_dim in src_var.dimensions:
            axis = src_var.dimensions.index(self.nlocs_dim)
            data = np.take(data, np.where(mask)[0], axis=axis)
 
        new_var[:] = data
 
    # ------------------------------------------------------------------
    # Polygon mask
    # ------------------------------------------------------------------
 
    @staticmethod
    def _polygon_mask(
        lat_arr: np.ndarray,
        lon_arr: np.ndarray,
        poly_file: Union[str, Path],
        *,
        lat_col: str = "Lat",
        lon_col: str = "Lon",
    ) -> np.ndarray:
        df = pd.read_csv(poly_file)
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(
                f"Polygon CSV must contain '{lat_col}' and '{lon_col}' columns; "
                f"got {list(df.columns)}."
            )
 
        # Shapely uses (x, y) — we treat x=lon, y=lat so Point(lon, lat).
        polygon = Polygon(zip(df[lon_col].values, df[lat_col].values))
 
        minlat, maxlat = df[lat_col].min(), df[lat_col].max()
        minlon, maxlon = df[lon_col].min(), df[lon_col].max()
 
        # Fast bounding-box prefilter, then exact point-in-polygon on survivors.
        near = (
            (lat_arr >= minlat) & (lat_arr <= maxlat) &
            (lon_arr >= minlon) & (lon_arr <= maxlon)
        )
        mask = np.zeros(len(lat_arr), dtype=bool)
        for i in np.where(near)[0]:
            mask[i] = polygon.contains(Point(lon_arr[i], lat_arr[i]))
        return mask
 
    # ==================================================================
    # Multi-file helpers
    # ==================================================================
 
    @classmethod
    def concat(
        cls,
        filepaths: Sequence[Union[str, Path]],
        *,
        ignore_errors: bool = False,
        reset_index: bool = True,
        channel_index: Optional[List[int]] = None,
        **reader_kwargs,
    ) -> tuple[pd.DataFrame, dict]:
        frames: List[pd.DataFrame] = []
        channel_meta = {}
        for fp in filepaths:
            try:
                reader = cls(fp, **reader_kwargs)
                if channel_index is not None:
                    reader.select_channel(channel_index)
                frames.append(reader.load())
                if not channel_meta:
                    channel_meta = reader.channel_meta
            except Exception as exc:
                if ignore_errors:
                    warnings.warn(f"Skipping {fp}: {exc}", stacklevel=2)
                else:
                    raise
        if not frames:
            return pd.DataFrame(), {}
        df = pd.concat(frames, ignore_index=reset_index)
        return df, channel_meta
 
    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
 
    def __repr__(self) -> str:
        grp_filter = (
            f"groups={self._groups}"
            if self._groups
            else f"variables={list(self._variables.keys())}"
            if self._variables
            else "all groups"
        )
        chan_info = f", channels={self._channel_index}" if self._channel_index != list(range(self._nchans or 0)) else ""
        return f"IODAFile({self.filepath.name!r}, {grp_filter}{chan_info})"
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _resolve_load_map(self, ds: nc.Dataset) -> Dict[str, List[str]]:
        if self._variables is not None:
            return dict(self._variables)
        available = ds.groups
        wanted = self._groups if self._groups is not None else list(available.keys())
        return {
            g: list(available[g].variables.keys())
            for g in wanted
            if g in available
        }
    
    def _load_channel_meta(self) -> tuple[Dict[str, np.ndarray], Optional[int]]:
        result = {}
        nchans = None
        with nc.Dataset(self.filepath, "r") as ds:
            if self.metadata_group not in ds.groups:
                return result, nchans
            if self.channel_dim is None or self.channel_dim not in ds.dimensions:
                return result, nchans
            nchans = ds.dimensions[self.channel_dim].size
            grp = ds.groups[self.metadata_group]
            for var_name, var in grp.variables.items():
                if self.channel_dim in var.dimensions:
                    result[var_name] = var[:]
        return result, nchans
 
 
# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
 
def load_ioda(
    filepath: Union[str, Path],
    *,
    groups: Optional[List[str]] = None,
    variables: Optional[Dict[str, List[str]]] = None,
    **reader_kwargs,
) -> pd.DataFrame:
    reader = IODAFile(filepath, **reader_kwargs)
    if variables is not None:
        for grp, vs in variables.items():
            reader.select_variables(grp, vs)
    elif groups is not None:
        reader.select_groups(*groups)
    return reader.load()
 
def ioda_schema(filepath: Union[str, Path]) -> Dict[str, List[str]]:
    return IODAFile(filepath).schema()
 
 
# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------
 
def _to_series(
    raw: np.ma.MaskedArray,
    var: nc.Variable,
    nlocs: Optional[int],
    string_fill: str,
) -> pd.Series:
    if raw.dtype.kind in ("S", "U") or raw.dtype == object:
        if raw.ndim == 2:
            arr = nc.chartostring(raw.filled(b""))
        else:
            arr = raw.filled(string_fill) if hasattr(raw, "filled") else raw
        return pd.Series(arr.astype(str))
 
    data = raw.filled(np.nan) if hasattr(raw, "filled") else np.asarray(raw)
 
    if data.ndim == 1:
        return pd.Series(data)
 
    if data.ndim == 2:
        if data.shape[1] == 1:
            return pd.Series(data[:, 0])
        warnings.warn(
            f"Variable '{var.name}' has unexpected shape {data.shape} after "
            "processing; taking index 0 along axis 1.",
            UserWarning,
            stacklevel=4,
        )
        return pd.Series(data[:, 0])
 
    return pd.Series(data.ravel())
 
 
def _parse_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    sample = df[col].dropna()
    if sample.empty:
        return df
    s = sample.iloc[0]
    try:
        if isinstance(s, (int, float, np.integer, np.floating)):
            df[col] = pd.to_datetime(df[col], unit="s", utc=True)
        else:
            df[col] = pd.to_datetime(df[col], utc=True)
    except Exception:
        pass
    return df

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

