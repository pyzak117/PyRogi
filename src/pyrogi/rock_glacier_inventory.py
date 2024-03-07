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
from pyrogi import rock_glacier_unit as rg

class RockGlacierInventory:
    """
    Describe a Rock Glacier Inventory dataset.
    Methods are usable for collective analysis of Rock Glacier Unit features.
    """
    
    def __init__(self,
        pms_layer,
        out_layer='',
        epsg=2056,
        ):

        # Define attributes
        self.pms_layer = pms_layer
        self.out_layer = out_layer
        self.epsg=epsg

    def show_map(self, pms_color='red', pms_size=0.6, outlines_color = ('blue', 'green'), outlines_linewidth=0.3, basemap=False):
        ax = self.pms_layer.plot(color=pms_color, markersize=pms_size)
        if type(self.out_layer) != str:
            self.out_layer[self.out_layer['Out.Type']=='Extended'].boundary.plot(color=outlines_color[0]   , ax=ax, linewidth=outlines_linewidth)
            self.out_layer[self.out_layer['Out.Type']=='Restricted '].boundary.plot(color=outlines_color[1], ax=ax, linewidth=outlines_linewidth)
        return ax

    def get_rgus(
        self,
        outlines_status = '',

    ):
        return [rg.read_rgik_feature(ft, rgu_epsg=self.epsg) for ft in self.pms_layer.iloc]
        
def Open(
    primary_markers_path,
    outlines_path = '',
    primary_markers_layername = '',
    outlines_layername = '',
    primary_markers_epsg=0,
    outlines_epsg=0):

    # Get Primary Markers layer (mandatory)
    pms_layer = vt.Open(primary_markers_path, primary_markers_layername, primary_markers_epsg)

    # Get Outlines layer (optionnal)
    out_layer = ''
    if outlines_path != '':
        out_layer = vt.Open(outlines_path, outlines_layername, outlines_epsg)

    if type(out_layer) == str:
        return RockGlacierInventory(pms_layer)
    else:
        return RockGlacierInventory(pms_layer, out_layer)