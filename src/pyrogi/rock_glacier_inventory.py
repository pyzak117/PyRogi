#TODO : just overload a dataframe with functions. To heavy otherwise.

from tqdm import tqdm
import numbers
import contextily as cx
from matplotlib import pyplot as plt
import shapely
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
import numpy as np
from telenvi import raster_tools as rt
from telenvi import vector_tools as vt
from pyrogi.rock_glacier_unit import RockGlacierUnit
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RockGlacierInventory:
    """
    Describe a Rock Glacier Inventory dataset.
    Methods are usable for collective analysis of Rock Glacier Unit features.
    """
    
    def __init__(self,
        pms_layer,
        ous_layer=None,
        pms_layername = 'rogi_ch_pms_cur',
        ous_layername = 'rogi_ch_ous_cur',
        epsg=2056,
        fig_dir='/home/duvanelt/PyRogi/tests/test_fig_dir',
        version_note=None
        ):

        # Open pms and ous layers
        self.epsg=epsg
        self.pms_layer = vt.Open(pms_layer, pms_layername, set_crs_epsg=self.epsg)
        if ous_layer is None:
            ous_layer = pms_layer
        self.ous_layer = vt.Open(ous_layer, ous_layername, set_crs_epsg=self.epsg)

        # Define version name for the figures
        now = datetime.now()
        date_hour_str = now.strftime("%Y-%m-%d")
        if version_note is not None:
            version_note = f"_{version_note}"
        else:
            version_note = ""
        self.version = f"rogi_ch_v{date_hour_str.replace('-','').replace('_','')}{version_note}"

        # Define directory to store the figures
        self.fig_dir=Path(fig_dir)
        if not self.fig_dir.exists():
            self.fig_dir.mkdir()

        # Filter features
        self.pms_layer = self.pms_layer[self.pms_layer.pm_type == 'rock_glacier']

    def crop(self, cropzone):
        self.pms_layer = vt.spatial_selection(self.pms_layer, cropzone, predicate='intersects')
        self.ous_layer = vt.spatial_selection(self.ous_layer, cropzone, predicate='intersects')
        return self

    def show_map(self, save_fig=False, figsize=(10, 4)):
        fig,ax=plt.subplots(figsize=figsize)
        self.pms_layer.plot(markersize=0.4, color='black', ax=ax)
        self.ous_layer.plot(ax=ax)
        fig_name = f'rogimap_{self.version}.png'
        fig_path = Path(self.fig_dir, fig_name)
        ax.set_title(fig_name[:-3])
        if save_fig:
            fig.savefig(fig_path)
        return ax

    def __repr__(self):
        return self.version
