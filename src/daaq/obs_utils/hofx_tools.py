# obs_utils/hofx_tools.py
import os
from typing import List, Sequence, Union
from pathlib import Path
import pandas as pd
from daaq.obs_utils.ioda_tools import IODAFile


def _compute_omb_oma(
    df: pd.DataFrame,
    varname: str,
    channel_index: List[int],
    bkg_group: str,
    ana_group: str,
) -> pd.DataFrame:
    """
    Rename hofx group columns to hofbkg_/hofana_ and compute omb_/oma_.

    Operates in place on `df` and also returns it for chaining.
    """
    rename_map = {}
    for chidx in channel_index:
        rename_map[f'{bkg_group}_{varname}_{chidx}'] = f'hofbkg_{varname}_{chidx}'
        rename_map[f'{ana_group}_{varname}_{chidx}'] = f'hofana_{varname}_{chidx}'
    df.rename(columns=rename_map, inplace=True)

    for chidx in channel_index:
        obsvcol = f'ObsValue_{varname}_{chidx}'
        hofbcol = f'hofbkg_{varname}_{chidx}'
        hofacol = f'hofana_{varname}_{chidx}'
        df[f'ombg_{varname}_{chidx}'] = df[obsvcol] - df[hofbcol]
        df[f'oman_{varname}_{chidx}'] = df[obsvcol] - df[hofacol]
    return df


def load_hofx_cycles_single(
    dates: Sequence,
    hofx_path: Union[str, Path],
    file_template: str,
    product: str,
    varname: str,
    channel_index: List[int],
    *,
    bkg_group: str = 'hofx0',
    ana_group: str = 'hofx2',
    date_fmt: str = '%Y%m%d%H',
) -> tuple[pd.DataFrame, list]:
    """
    Load hofx across cycles where each cycle is a SINGLE IODA file containing
    both bkg and ana hofx as separate groups (e.g. hofx0 and hofx2).

    `file_template` takes two slots: cycle string and product.
    """
    files, cycles = [], []
    for date in dates:
        cdate_str = date.strftime(date_fmt)
        fpath = os.path.join(hofx_path, file_template.format(datestr=cdate_str, product=product))
        if os.path.exists(fpath):
            files.append(fpath)
            cycles.append(date)

    if not files:
        return pd.DataFrame(), []

    df, _ = IODAFile.concat(files, cycles=cycles, channel_index=channel_index)
    probe_chidx = channel_index[0]
    has_precomputed = (
        f'ombg_{varname}_{probe_chidx}' in df.columns
        and f'oman_{varname}_{probe_chidx}' in df.columns
    )
    if not has_precomputed:
        df = _compute_omb_oma(df, varname, channel_index, bkg_group, ana_group)
        
    return df, cycles


def load_hofx_cycles_paired(
    dates: Sequence,
    hofx_path: Union[str, Path],
    file_template: str,
    product: str,
    varname: str,
    channel_index: List[int],
    *,
    bkg_stage_tag: str,
    ana_stage_tag: str,
    hofx_group: str = 'hofx',
    date_fmt: str = '%Y%m%d%H',
) -> tuple[pd.DataFrame, list]:
    """
    Load hofx across cycles where each cycle has TWO separate IODA files —
    one for bkg, one for ana — distinguished by a stage tag in the filename.

    `file_template` takes three slots: cycle string, stage tag, product.
    Both files are assumed to expose the same hofx group name (default
    'hofx'), since the stage is encoded in the filename rather than the
    group structure.
    """
    bkg_files, ana_files, cycles = [], [], []
    for date in dates:
        cdate_str = date.strftime(date_fmt)
        bkg_fpath = os.path.join(
            hofx_path, file_template.format(datestr=cdate_str, stagetag=bkg_stage_tag, product=product)
        )
        ana_fpath = os.path.join(
            hofx_path, file_template.format(datestr=cdate_str, stagetag=ana_stage_tag, product=product)
        )
        if os.path.exists(bkg_fpath) and os.path.exists(ana_fpath):
            bkg_files.append(bkg_fpath)
            ana_files.append(ana_fpath)
            cycles.append(date)

    if not cycles:
        return pd.DataFrame(), []

    bkg_df, _ = IODAFile.concat(bkg_files, cycles=cycles, channel_index=channel_index)
    ana_df, _ = IODAFile.concat(ana_files, cycles=cycles, channel_index=channel_index)

    # Pull ana hofx into bkg_df under a temporary group-prefixed name so the
    # shared rename/compute helper sees a consistent layout.
    ana_tmp_group = f'{hofx_group}_ana'
    for chidx in channel_index:
        src = f'{hofx_group}_{varname}_{chidx}'
        bkg_df[f'{ana_tmp_group}_{varname}_{chidx}'] = ana_df[src].values

    df = _compute_omb_oma(
        bkg_df, varname, channel_index,
        bkg_group=hofx_group,
        ana_group=ana_tmp_group,
    )
    return df, cycles

