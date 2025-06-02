import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import numpy as np
import geopandas as gpd

from telenvi import vector_tools as vt
from telenvi import raster_tools as rt

from pathlib import Path

# Utilities : paths useful to store raw data

# Ortho images
GEODATA_PHD = Path('/mnt/Data_Part/geodata_switzerland_insar_glaciers_dem-srtm')
ORTHOS_PHD_DIR = Path(GEODATA_PHD, 'admin-ch_swiss-images')
ORTHOSMD_PHD_DIR = Path(ORTHOS_PHD_DIR, 'metadata')
ORTHOS_PHD_MAP_FILEPATH = Path(ORTHOSMD_PHD_DIR, 'orthos-availables-local.gpkg')

# Dems
DEMS_PHD_DIR = Path(GEODATA_PHD, 'admin-ch_dems')

# Alti
SA3D_PHD_DIR = Path(DEMS_PHD_DIR, 'sa3d_dems')
SA3D_MD_PHD_DIR = Path(SA3D_PHD_DIR, 'metadata')

# Surface
SS3D_PHD_DIR = Path(DEMS_PHD_DIR, 'ss3d_dems')
SS3D_MD_PHD_DIR = Path(SS3D_PHD_DIR, 'metadata')

# The layer where User ask what he want
DL_ZONES_FILEPATH = Path(GEODATA_PHD, 'download_zones.gpkg')
DL_ZONES_LAYER = gpd.read_file(DL_ZONES_FILEPATH)

def update_available_orthos():
    """
    Update the gpkg file with extents and metadata of the orthoimages available in the phd directory of orthoimages
    """
    new_orthomap =rt.getRastermap(ORTHOS_PHD_DIR, epsg=2056)
    new_orthomap['year'] = new_orthomap.apply(lambda row:int(row.filepath.split('/')[-1].split('-')[0]), axis=1)
    new_orthomap.to_file(ORTHOS_PHD_MAP_FILEPATH)
    return new_orthomap

def show_available_orthos_map():
    fig, axes=plt.subplots(1, 3, figsize=(30,10))
    orthomap = gpd.read_file(ORTHOS_PHD_MAP_FILEPATH)
    for i, y in enumerate((2017, 2020, 2023)):
        for_year = orthomap[orthomap.year == y]
        vt.add_wmts_layer(for_year, epsg=2056, ax=axes[i], geo_target_linewidth=1, geo_target_color='red')
        axes[i].set_title(str(y))
    # return fig

def show_disp_field(target, title='Disp field', show_colorbar=True, figsize=(10, 5), ax=None, dx_col='dx', dy_col='dy', d_col='m_yr', arrowstyle='->', source_basemap=None, alpha=1, epsg=2056, column=None, cmap='mako_r', linewidth=0.8, vmin=None, vmax=None, vectors_color='black', colorbar_label=None):
    """
    Plot a point layer in a GeoDataFrame as a displacement field with arrows, where arrow color
    can be dynamically set based on a specified column and a colormap.

    Parameters:
    target : String or Path or GeoDataFrame
        The input GeoDataFrame with points and displacement columns.
    dx_col : str, optional
        The column name for displacement in the x direction. Default is 'dx_in_meters'.
    dy_col : str, optional
        The column name for displacement in the y direction. Default is 'dy_in_meters'.
    d_col : str, optional
        The column name for scaling the arrow size.
    column : str, optional
        The column name used to determine the color of each arrow.
    cmap : str, optional
        The name of the colormap to use. Default is 'viridis'.

    Returns:
    matplotlib.axes._subplots.AxesSubplot
        A plot displaying the displacement vectors.
    """
    # Open the target
    target = vt.Open(target) if isinstance(target, (str, bytes)) else target

    # Create an empty ax if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the base layer
    target.plot(ax=ax, markersize=0, alpha=0)
    
    # Add the basemap layer if provided
    if source_basemap is not None:
        vt.add_wmts_layer(target, ax=ax, epsg=epsg, geo_target_alpha=0, source=source_basemap)
    
    # Apply colormap if a target column for color is specified
    if column is not None:
        values = target[column]
        if vmax is None and vmin is None:
            norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        scalar_map = None

    # Iterate through each row in the GeoDataFrame to plot arrows
    for row in target.iloc:
        point = row.geometry
        dx = row[dx_col]
        dy = row[dy_col]
        
        # Set color from colormap if a target column for color is specified
        if scalar_map is not None:
            color = scalar_map.to_rgba(row[column])
        else:
            color = vectors_color
        
        # Add an arrow representing the displacement
        arrow = FancyArrowPatch((point.x, point.y), 
                                (point.x + dx, point.y + dy), 
                                color=color, 
                                alpha=alpha,
                                mutation_scale=10 * row[d_col], 
                                arrowstyle=arrowstyle,
                                linewidth=linewidth,
                                )
        ax.add_patch(arrow)
    
    # Optional colorbar
    if colorbar_label is None:
        colorbar_label = column
    if show_colorbar:
        plt.colorbar(scalar_map, ax=ax, label=colorbar_label)
    
    # Optional title and labels
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # ax.grid(True)
    
    return ax

