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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RockGlacierInventory:
    """
    Describe a Rock Glacier Inventory dataset.
    Methods are usable for collective analysis of Rock Glacier Unit features.
    """
    
    def __init__(self,
        pms_layer,
        out_layer=None,
        epsg=2056,
        ):

        # Define attributes
        self.pms_layer = pms_layer
        self.out_layer = out_layer
        self.epsg=epsg
        self.population = self.get_population()

    def get_population(self):

        # Create empty list
        population = []

        # For each Primary Marker in the layersource
        for pm in self.pms_layer.iloc:

            # Build a Rock Glacier Unit feature
            rgu = RockGlacierUnit(pm)

            # Load the outlines
            rgu.load_outlines(self.out_layer)

            # Add the rgu to the population list
            population.append(rgu)
        
        return population

    def init_dems(self,
        dem_source  : str | gpd.GeoDataFrame,  
        layername   : str ='', 
        nRes        : numbers.Real = None, 
        width       : numbers.Real = 500):

        new_pop = []
        
        # For each RGU of the ROGI
        for rgu in tqdm(self.get_population()):

            if not rgu.outlines_loaded():
                continue

            # Create empty serie
            row = pd.Series(dtype='object')

            # Init DEM
            rgu.initialize_dem(dem_source, layername, nRes, width)

            new_pop.append(rgu)
        
        return get_rogi_from_population(new_pop)

    def __repr__(self):
        return "rogi"

    def get_alti_ranges(self,
        dem_source  : str | gpd.GeoDataFrame,  
        layername   : str ='', 
        nRes        : numbers.Real = None, 
        width       : numbers.Real = 500) -> list:

        # Empty lists
        alti_ranges = []

        # For each RGU of the ROGI
        for rgu in tqdm(self.get_population()):

            if not rgu.outlines_loaded():
                continue

            # Create empty serie
            row = pd.Series(dtype='object')

            # Init DEM
            rgu.initialize_dem(dem_source, layername, nRes, width)

            # Put them into the lists
            zmin, zmax = rgu.get_altitudinal_range()
            row['zmin'] = zmin
            row['zmax'] = zmax
            row['acti_cl'] = rgu.rgik_pm_acti_cl
            alti_ranges.append(row)

            # Close dem
            rgu.close_dem()

        return pd.DataFrame(alti_ranges)

    def crop(self, extent):

        # 1 - take the population of the Rock Glaciers Units
        population = self.get_population()

        # Select within the layer
        pms_layer = vt.cropLayerFromExtent(self.pms_layer, extent)
        out_layer = vt.cropLayerFromExtent(self.out_layer, extent)

        # Build a new rogi from the layer
        new_rogi = get_rogi_from_layers(
            primary_markers_layer=pms_layer,
            outlines_layer=out_layer)

        # 2 - take only the markers inside the extent
        # sub_population = [rgu for rgu in population if rgu.rgu_pm_geom.within(extent)]

        # 3 - build a new rogi from this sub_population
        # new_rogi = get_rogi_from_population(sub_population, epsg=self.epsg)

        return new_rogi


    def show_map(self, pms_color='red', pms_size=0.6, outlines_color = ('white', 'green'), outlines_linewidth=0.5, basemap=False):

        # Plot a map with the Markers
        ax = self.pms_layer.plot(color=pms_color, markersize=pms_size)

        # Add outlines if known
        if self.out_layer is not None:
            self.out_layer[self.out_layer['Out.Type']=='Extended'].boundary.plot(color=outlines_color[0]   , ax=ax, linewidth=outlines_linewidth)
            self.out_layer[self.out_layer['Out.Type']=='Restricted '].boundary.plot(color=outlines_color[1], ax=ax, linewidth=outlines_linewidth)

        # Add background with satellite view
        if basemap :
            cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery, crs=self.epsg)
        return ax

    def show_alti_ranges_with_hbars(self,
        dem_source  : str | gpd.GeoDataFrame,  
        layername   : str ='', 
        nRes        : numbers.Real = None, 
        width       : numbers.Real = 500):

        # TODO : some rock glaciers have -9999 as min_value when we make a resampling
        # I think it's because the value is prelevated in the mask

        # Get the alti values
        zs = self.get_alti_ranges(dem_source, layername, nRes, width).sort_values('zmin')

        # Map 'acti_cl' to colors (adjust colors as needed)
        color_map = {
            'Active':           '#FE938C',
            'Active uncertain': '#FE938C',
            'Transitional':     '#F2AF29',
            'Relict':           '#B9F5D8',
            'Relict uncertain': '#B9F5D8'
        }

        default_color = 'blue'  # Default color for any unmapped or NaN 'acti_cl' values

        # Apply the color map to 'acti_cl', using .get() to handle unmapped values gracefully
        bar_colors = zs['acti_cl'].map(lambda x: color_map.get(x, default_color))

        # Calculate the altitudinal range
        altitudinal_ranges = zs['zmax'] - zs['zmin']

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(zs.index, altitudinal_ranges, left=zs['zmin'], color=bar_colors)

        # Customize the plot
        plt.xlabel('Altitude (m)')
        plt.title('Altitudinal Ranges of Rock Glaciers by Activity Class in Valaisan Alps')

        plt.grid(True, axis='x')
        plt.show()

    def show_alti_ranges_with_boxes(self,
        dem_source  : str | gpd.GeoDataFrame,  
        layername   : str ='', 
        nRes        : numbers.Real = None, 
        width       : numbers.Real = 500):

        # Get the alti values
        zs = self.get_alti_ranges(dem_source, layername, nRes, width).sort_values('zmin')

        pass

def get_rogi_from_layers(
    primary_markers_layer,
    outlines_layer = None,
    primary_markers_layername = '',
    outlines_layername = '',
    primary_markers_epsg=0,
    outlines_epsg=0,
    ):

    # Get Primary Markers layer (mandatory)
    pms_layer = vt.Open(primary_markers_layer, primary_markers_layername, primary_markers_epsg)

    # Get Outlines layer (optionnal)
    out_layer = ''
    if outlines_layer is not None:
        out_layer = vt.Open(outlines_layer, outlines_layername, outlines_epsg)

    if type(out_layer) == str:
        return RockGlacierInventory(pms_layer)
    else:
        return RockGlacierInventory(pms_layer, out_layer)
    
def get_rogi_from_population(population, epsg=2056):
    """
    Send a RockGlacierInventory feature from a list of Rock Glacier Units instances
    """

    # Create 3 empty lists
    pms = []
    out = []

    # First we have to build different layers from the features
    for rgu in population:
        
        # Get the pandas.Series representative of each things
        pm, oue, our = rgu.get_series()

        # Add Primary Marker serie to the PMS list
        pms.append(pm)

        # Add the outlines to the OUT list
        [out.append(t) for t in (oue, our) if t is not None]

    # Now we build GeoDataFrames layers from those lists
    pms_layer = gpd.GeoDataFrame(pms)
    out_layer = gpd.GeoDataFrame(out)

    # And now we can build a rogi from them
    return get_rogi_from_layers(
        primary_markers_layer = pms_layer,
        outlines_layer = out_layer,
        primary_markers_epsg=epsg,
        outlines_epsg=epsg,
    )

