import seaborn as sns
from copy import deepcopy
import numpy as np
import json
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from telenvi import vector_tools as vt
from datetime import datetime
from pyrogi.rock_glacier_unit import RockGlacierUnit
import pyrogi.colors as rgc
from telenvi import raster_tools as rt
from telenvi import aida
import warnings
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from IPython.display import display
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)

TEMP_FIGS_DIR = '/home/duvanelt/Desktop/'
DATAMAP_PATH = '/home/duvanelt/datamap.json'

# Define dictionnary for full variable names
human_keys = {
    'rgu_activity_class_wu':'Rock Glacier Activity',
    'rgu_activity_class':'Rock Glacier Activity',
    'alti_oue_min':'Rock Glacier Front Elevation (m a.s.l.)',
    'alti_oue_max':'Rock Glacier Roots Elevation (m a.s.l.)',
    'alti_oue_mean':'Rock Glacier Mean Elevation (m a.s.l.)',
    'alti_oue_med': 'Rock Glacier Median Elevation (m a.s.l.)',
    'slope_our_med':'Rock Glacier Surface Median Steepness (°)',
    'rgu_kin_att': 'Rock Glacier Kinematic Attribute (annual semi order of magnitude)',
    'cardi_pt':'Rock Glacier Main Orientation',
    'dir_ins':'Mean Direct Insolation in Summer (kJ/m²/day)',
    'surf_oue_ha':'Rock Glacier Extended surface (hectares)',
    'surf_our_ha':'Rock Glacier Restricted surface (hectares)',
    'len_min_our':'Rock Glacier Restricted Width',
    'len_max_our':'Rock Glacier Restricted Length',
    'upslope_con':'Rock Glacier Upslope Connection'
}

# Open DATAMAP
with open(DATAMAP_PATH) as f:
    DATAMAP = json.load(f)[0]

# Open the study area
ROGI_REGIONS_LAYER = gpd.read_file(DATAMAP['auxiliaires_path'], layer=DATAMAP['rogi_regions_layername'])

def Open(aoi_name, meta_figs_dir=TEMP_FIGS_DIR, load_external_datasets=True, **kwargs):

    # Only arguments to change
    figs_dir = f'{meta_figs_dir}_{aoi_name}'

    # Open the study area
    aoi = ROGI_REGIONS_LAYER[ROGI_REGIONS_LAYER.rogi_reg_name == aoi_name]

    # Open the whole rogi enhanced with altitudinal data
    rgdf = gpd.read_file(DATAMAP['rogi_alti_data'], layer=f'rogi_alti_{aoi_name}')

    # Create the RockGlacierInventory instance
    rogi_instance = RockGlacierInventory(
        b_layer=rgdf,
        aoi=aoi,
        aoi_name=aoi_name,
        out_figs_dir=figs_dir,
        **kwargs,
    )

    # Load external data
    rogi_instance.pre_process_main_layer(load_external_datasets=load_external_datasets)

    return rogi_instance

class RockGlacierInventory:
    """
    Describe a Rock Glacier Inventory dataset.
    It can be instantiated from either:
    - the Primary Markers layer and the Outlines layer from ROGI_2.0 (raw mode)
    - the Boosted layer from ROGI_2.0 + alti metrics (advanced mode)
    The class contains methods to pre-process the main layer, compute topographic metrics, and create maps.
    
    Keywords:
    pms_layer : gpd.GeoDataFrame, the Primary Markers layer from ROGI_2.0
    ous_layer : gpd.GeoDataFrame, the Outlines layer from ROGI_2.0
    b_layer : gpd.GeoDataFrame, the Boosted layer from ROGI_2.0 + alti metrics
    aoi_name : str, name of the area of interest
    aoi : gpd.GeoDataFrame, the area of interest geometry
    aoi_buffer_size : int, buffer size in meters to apply to the convex hull of the rock glaciers
    epsg : int, EPSG code of the coordinate reference system
    out_figs_dir : str, path to the output directory for the figures
    out_figs_version_note : str, note to add to the output directory name for versioning
    self.aoi_dem_res : int, resolution of the DEM to use for the area of interest
    aoi_dem_reclass_step : int, step in meters for reclassifying the DEM
    aoi_hillshade_azimut : int, azimuth of the hillshade
    aoi_hillshade_altitude : int, altitude of the hillshade
    aoi_places : gpd.GeoDataFrame, places from OSM within the area of interest
    aoi_lakes : gpd.GeoDataFrame, lakes from OSM within the area of interest
    aoi_sgi16 : gpd.GeoDataFrame, glaciers from SGI 2016 within the area of interest
    dem : raster_tools object, the DEM for the area of interest
    dem_reclass : raster_tools object, the reclassified DEM for the area of interest
    hillshade : raster_tools object, the hillshade for the area of interest
    dir_ins : raster_tools object, the direct insolation raster for the area of interest
    gc : dict, graphical chart dictionary for the qualitative variables
    boosted : bool, whether the boosted layer is available or not

    Methods:
    load_external_datasets_from_DATAMAP_on_thib_pc : load external datasets from a DATAMAP json file
    get_median_values_in_geoim : compute the median value of a raster within each polygon of the main layer
    pre_process_main_layer : pre-process the main layer with some quick & dirty patches
    is_boosted : check if the boosted layer is available
    get_complete_name : get the complete name for the output figures directory
    create_figs_dir : create the output figures directory
    get_alti_metrics : compute topographic metrics for the features of the inventory and save the boosted layer
    make_b_layer : build the boosted layer from a list of series
    save_b_layer : save the boosted layer to a geopackage
    __repr__ : get the string representation of the inventory
    get_basic_map : create a basic map with hillshade, dem, glaciers, lakes and places
    get_rgs_map : create a map with rock glaciers colored by a given column

    Attributes:
    pms_layer : gpd.GeoDataFrame, the Primary Markers layer from ROGI_2.0
    ous_layer : gpd.GeoDataFrame, the Outlines layer from ROGI_2.0
    oue_layer : gpd.GeoDataFrame, the Outlines Extended layer from ROGI_2.0
    our_layer : gpd.GeoDataFrame, the Outlines Restricted layer from ROGI_2.0
    b_layer : gpd.GeoDataFrame, the Boosted layer from ROGI_2.0 + alti metrics

    DATAMAP : json file with all the paths
    """
    
    def __init__(self,

        # Rogi input layers
        pms_layer=None,
        ous_layer=None,
        b_layer=None,

        # Area Of Interest
        aoi_name='unnnamed-aoi',
        aoi = None,
        aoi_buffer_size=300,

        # DEM on the AOI settings
        aoi_dem_res = 20,
        aoi_dem_reclass_step = 500,
        aoi_hillshade_azimut = 315,
        aoi_hillshade_altitude = 45,

        # Other
        epsg=2056,
        out_figs_dir = None,
        out_figs_version_note='',

        ):

        # Check the mandatory inputs
        assert (pms_layer is not None and ous_layer is not None) or b_layer is not None, 'Either (pms and ous) or b_layer required for instancing RockGlacierInventory'

        # Define coordinates system
        self.epsg=epsg

        # ROGI_2.0 (raw mode)
        if (pms_layer is not None and ous_layer is not None):
    
            # Open Primary Markers layer
            pms_layer = pms_layer[pms_layer.pm_type == 'rock_glacier']
            self.pms_layer = pms_layer

            # Open outlines layer
            self.ous_layer = ous_layer
            self.oue_layer = self.ous_layer[self.ous_layer.ou_type == 'extended']
            self.our_layer = self.ous_layer[self.ous_layer.ou_type == 'restricted']

            # Outlines extended as main layer
            self.main_layer = self.oue_layer

            # None the b_layer
            self.b_layer = None

        # ROGI_2.0 + alti metrics (advanced mode)
        elif b_layer is not None:
            self.b_layer = b_layer
            self.main_layer = b_layer

        # Default figures outputs dir
        self.aoi_name = aoi_name
        if out_figs_dir is None:
            self.out_figs_version_note = out_figs_version_note
            self.out_figs_dir = f"{TEMP_FIGS_DIR}/{self.get_complete_name()}"
        else:
            self.out_figs_dir = out_figs_dir

        # Create it
        self.create_figs_dir(self.out_figs_dir)

        # Open or build AOI
        if aoi is None:
            convex_hull = self.main_layer.unary_union.convex_hull.buffer(aoi_buffer_size)
            aoi = gpd.GeoDataFrame([{'geometry':convex_hull}])
        self.aoi_gdf = aoi
        self.aoi_geom = self.aoi_gdf.geometry.iloc[0]

        # Create graphical chart dictionnary for each column
        self.gc={
            'cardi_pt_colors': rgc.cardi_pt_colors,
            'rgu_activity_class_colors': rgc.rgu_activity_class_colors,
            'rgu_activity_class_wu_colors': rgc.rgu_activity_class_wu_colors,
            'rgu_kin_att_colors': rgc.rgu_kin_att_colors,
            'upslope_con_colors':rgc.upslope_con_colors,

            'cardi_pt_order': rgc.cardi_pt_order,
            'rgu_activity_class_order': rgc.rgu_activity_class_order,
            'rgu_activity_class_wu_order': rgc.rgu_activity_class_wu_order,
            'rgu_kin_att_order': rgc.rgu_kin_att_order,
            'upslope_con_order':rgc.upslope_con_order,

            'cardi_pt_palette': rgc.cardi_pt_palette,
            'rgu_activity_class_palette': rgc.rgu_activity_class_palette,
            'rgu_activity_class_wu_palette': rgc.rgu_activity_class_wu_palette,
            'rgu_kin_att_palette': rgc.rgu_kin_att_palette,
            'upslope_con_palette':rgc.upslope_con_palette,

            'cardi_pt_cmap': rgc.cardi_pt_cmap,
            'rgu_activity_class_cmap': rgc.rgu_activity_class_cmap,
            'rgu_activity_class_wu_cmap': rgc.rgu_activity_class_wu_cmap,
            'rgu_kin_att_cmap': rgc.rgu_kin_att_cmap,
            'upslope_con_cmap':rgc.upslope_con_cmap
        }

        # DEM on the AOI
        self.aoi_dem_res = aoi_dem_res
        self.aoi_dem_reclass_step = aoi_dem_reclass_step
        self.aoi_hillshade_azimut = aoi_hillshade_azimut
        self.aoi_hillshade_altitude = aoi_hillshade_altitude

        # Basemap
        self.basic_map_ax = None
        self.basic_map_fig = None

    def basic_map_available(self):
        """
        Check if the basic map has been computed
        """
        return self.basic_map_fig is not None and self.basic_map_ax is not None

    def load_external_datasets_from_DATAMAP_on_thib_pc(self):
        """
        Load DEMS, glaciers, lakes... from a json file called DATAMAP. Specific for TD. 

        Keywords:
        DATAMAP : dict, dictionnary with all the paths to the datasets

        Returns:
        None, but set the following attributes:
        aoi_places : gpd.GeoDataFrame, places from OSM within the area of interest
        aoi_lakes : gpd.GeoDataFrame, lakes from OSM within the area of interest
        aoi_sgi16 : gpd.GeoDataFrame, glaciers from SGI 2016 within the area of interest
        dem : raster_tools object, the DEM for the area of interest
        dem_reclass : raster_tools object, the reclassified DEM for the area of interest
        hillshade : raster_tools object, the hillshade for the area of interest
        dir_ins : raster_tools object, the direct insolation raster for the area of interest
        Note: this method is specific to the DATAMAP file available on Thib's PC
        """

        # TODO: make it more generic
        # The DEMS, Hillshades and PISR are computed manually in Qgis. This should be automated in the future
        # Also, the arguments given here such as dem_res, hillshade_azimut, etc. are then specific to the files that are available on disk

        # DEM
        self.alti_3d_path = DATAMAP['sa3d_dir']
        self.derived_sa3d = Path(self.alti_3d_path, 'derived_products')
        self.dem = rt.Open(Path(self.derived_sa3d, f'dem_{self.aoi_dem_res}_merged.tif'), load_pixels=True, geoExtent=self.aoi_geom)

        # Reclassify the DEM each 500m
        self.dem_reclass = aida.get_manual_clusters(self.dem, np.arange(0, 4500, self.aoi_dem_reclass_step))

        # Hillshade
        self.hillshade = rt.Open(Path(self.derived_sa3d, f'hillshade_{self.aoi_dem_res}_{self.aoi_hillshade_azimut}_{self.aoi_hillshade_altitude}.tif'), load_pixels=True,  geoExtent=self.aoi_geom)

        # PISR
        self.dir_ins = rt.Open(Path(self.derived_sa3d, f'pisr_{self.aoi_dem_res}/mean_direct-insolation_46_Kjm_with-svf_0715-0815-0915-1015.tif'),  geoExtent=self.aoi_geom, load_pixels=True)

        # Glaciers
        sgi16 = gpd.read_file(DATAMAP['glaciers_inv'], layer='sgi_2016_glaciers')

        # Places and waterways from OSM
        osm_path = DATAMAP['osm_dir']
        places = gpd.read_file(Path(osm_path, 'gis_osm_places_free_1.shp')).to_crs(epsg=2056)
        lakes = gpd.read_file(Path(osm_path, 'gis_osm_water_a_free_1.shp')).to_crs(epsg=2056)
        lakes = lakes[lakes.fclass != 'glacier']

        # Spatial selection within the area
        self.aoi_places = places.clip(self.aoi_gdf)
        self.aoi_lakes =  lakes.clip(self.aoi_gdf)
        self.aoi_sgi16 =  sgi16.clip(self.aoi_gdf)

        # Attribute selection
        lakes = lakes[lakes.fclass != 'glacier']

    def get_median_values_in_geoim(self, target_geoim, colname='raster_median'):
        """
        Add a column to the self.main_layer by reading a raster and computing the median for each polygon
        """
        self.main_layer[colname] = self.main_layer.apply(lambda row: target_geoim.compute_med_value_in_georow(row), axis=1)
        n_before = len(self.main_layer)
        self.main_layer = self.main_layer.dropna(subset=[colname])
        n_after = len(self.main_layer)
        print(f"{round((n_before - n_after) / n_before * 100, 2)}% of rock glaciers have nodata for dir_ins")
        self.b_layer = self.main_layer

    def pre_process_main_layer(self, load_external_datasets=True):
        """
        Make simple and redundant pre-processings on the self.main_layer
        This method can be used to quick & dirty patches
        """

        # Change font style for the figures
        plt.rcParams.update({'font.family': 'monospace', 'font.size':9})

        # Fix the annoying issues with activity classes
        def _get_activity_wu(row):
            if row.rgu_activity_class is not None:
                return row.rgu_activity_class.split('_')[0]
            else:
                return None
        self.main_layer['rgu_activity_class_wu'] = self.main_layer.apply(lambda row: _get_activity_wu(row),axis=1) 
        self.main_layer=self.main_layer[~self.main_layer.rgu_activity_class_wu.isna()]
        self.main_layer.rgu_activity_class_wu.unique()

        # Load external datasets
        if load_external_datasets:
            self.load_external_datasets_from_DATAMAP_on_thib_pc()
            self.get_median_values_in_geoim(self.dir_ins, 'dir_ins')
        
        return None

    def is_boosted(self):
        """
        Dynamic method which check if the boosted layer is available or not
        """
        self.boosted = self.b_layer is not None
        return self.boosted
    
    def get_complete_name(self):
        """
        Define version name for the figures
        """
        now = datetime.now()
        date_hour_str = now.strftime("%Y-%m-%d")
        complete_name = f"rogi_ch_{self.aoi_name}_{date_hour_str.replace('-','').replace('_','')}{self.out_figs_version_note}"
        return complete_name

    def create_figs_dir(self, out_figs_dir):
        """
        Create directory to store the figures
        """
        self.out_figs_dir=Path(out_figs_dir)
        if not self.out_figs_dir.exists():
            self.out_figs_dir.mkdir()

    def get_alti_metrics(self, dem_map, output_path, output_layername, aoi_dem_res_order_1=5, aoi_dem_res_order_2=0.5, save_steps=10):
        """
        Compute topographic metrics for the features of a rogi
        aoi_dem_res_order_1 and 2 are for different applications (altitude, aspect or slope within the margins)
        Save the output layer on disk
        Return the boosted layer as a gpd.GeoDataFrame
        Keywords:
        dem_map: raster_tools object, the DEM to use for the calculations
        output_path: str, path to the output geopackage
        output_layername: str, name of the layer within the geopackage
        aoi_dem_res_order_1: float, resolution order for the main calculations (altitude, aspect)
        aoi_dem_res_order_2: float, resolution order for the secondary calculations (slope within the margins)
        save_steps: int, number of steps to wait before saving a temporary version of the boosted layer
        Returns:
        b_layer: gpd.GeoDataFrame, the boosted layer with the topographic metrics
        Note: this method can take several hours to run depending on the number of rock glaciers and the DEM resolution
        """

        # Initialisation des variables
        b_series=[]
        unbound_local_error=[]
        no_outlines_error = []

        # Pour chaque glacier rocheux
        i = 0
        for target_rg in self.pms_layer.iloc:
            try:
                # Fabrication d'une entité RGU du pyrogi
                rg = RockGlacierUnit(target_rg)

                # Chargement des polygones
                rg.load_outlines(self.ous_layer)

                # Initialisation du DEM sur l'emprise du glacier rocheux
                rg.initialize_dem(dem_map, verbose=False)

                # Lecture et calcul des variables topographiques
                rg_detailled_serie = rg.get_detailled_serie(dem_map, aoi_dem_res_order_1, aoi_dem_res_order_2)

                # Ajout à la liste des séries si le résultat n'est pas un échec
                if rg_detailled_serie is not None:
                    b_series.append(rg_detailled_serie)

                # Sinon, probablement erreur de polygones
                else:
                    no_outlines_error.append(rg.pm_pid)
                    continue

            # Erreur à la cause inconnue
            except UnboundLocalError:
                unbound_local_error.append(rg.pm_pid)
                continue

            # Sauvegardes régulières si jamais le code plante à un moment
            if i == save_steps:
                print(i)
                print('sauvegarde')
                self.make_b_layer(b_series)
                self.save_b_layer(output_path, output_layername)

                # Réinitialise i
                i = 0

            # Incrémente i
            i += 1

        # Sauvegarde finale - manque l'établissement d'un rapport à partir des listes d'erreurs rencontrées
        self.make_b_layer(b_series)
        self.save_b_layer(output_path, output_layername)
        return self.b_layer

    def make_b_layer(self, b_series):
        """
        Build a boosted rogi layer from a list of gpd series
        """

        # Production d'un GeoDataFrame de sortie
        b_layer = gpd.GeoDataFrame(b_series).set_geometry('oue_geom').set_crs(epsg=2056)

        # Suppression des géométries parasites - impossible de sauver un gpd en gpkg avec plusieurs géométries
        b_layer = b_layer.drop(['our_geom', 'pm_geom'], axis=1)

        # Sauvegarde sur l'instance
        self.b_layer = b_layer

        return b_layer

    def save_b_layer(self, output_path, output_layername):
        """
        Save the boosted layer instance on disk
        """
        if self.b_layer is not None:
            self.b_layer.to_file(output_path, layer=output_layername)
            return True
        else:
            print('no b_layer on this rogi instance')
            return False

    def __repr__(self):
        return self.get_complete_name()

    def reinit_basic_map(self):
        """
        Reinitialize the basic map (fig and ax) to None
        """
        self.basic_map_ax = None
        self.basic_map_fig = None

    def get_basic_map(
        self,
        ax=None,
        mask_alpha=0.8,
        map_figsize = (15, 15),
        expand_extent_x = 0,
        expand_extent_y = 0,
        glaciers_color='#caf0f8',
        water_color='#03045e',
        water_alpha=0.3,
        glaciers_alpha=0.4,
        places_labels_fontsize=7.5,
        places_label_alpha=0.5,
        dem_alpha=0.3,
        dem_cmap='bone',
        hillshade_alpha=0.8,
        aerial_view=True,
        show_places=True,
        inplace = True,
        metadata_x_loc = 0.2,
        metadata_y_loc = 0.15,
        gap_between_arrow_and_scale_bar = 0.05,
        scale_bar_length = 5000,
        scale_bar_units = 'm',
        scale_bar_fontsize = 10,
        north_arrow_size = 0.1,
        north_arrow_color = 'black',
        ):
        """
        Make a basic map with hillshade, dem, glaciers, lakes and places
        Additional keyword arguments:
        mask_alpha: float, alpha for the aerial image mask
        map_figsize: tuple, figsize for the map
        x_buffer, y_buffer: space in meters to add on the sides of the aoi
        glaciers_color: str, color for the glaciers
        water_color: str, color for the water bodies
        water_alpha: float, alpha for the water bodies
        glaciers_alpha: float, alpha for the glaciers
        places_labels_fontsize: float, fontsize for the places labels
        places_label_alpha: float, alpha for the places labels background
        dem_alpha: float, alpha for the dem
        dem_cmap: str, cmap for the dem
        hillshade_alpha: float, alpha for the hillshade
        aerial_view: bool, whether to add the aerial image or not
        show_places: bool, whether to show the places or not
        Returns:
        ax: matplotlib axis with the map
        """

        # If the basic map is available, and user didn't give an ax, return it as a fig and ax
        if self.basic_map_available() and ax is None:
            return self.basic_map_fig, self.basic_map_ax

        # If the basic map is not available and no template is given as axis, create it with the figure
        if ax is None:
            fig, ax = plt.subplots(figsize=map_figsize)

        # If the basic map is not available but a template is given as axis, use it, and extract it's figure        
        else:
            ax = ax
            fig = ax.figure            

        # Add aerial image
        if aerial_view:
            vt.add_wmts_layer(self.aoi_gdf, mask_outside_geo_target=True, epsg=2056, ax=ax, mask_color='white', mask_alpha=mask_alpha, expand_extent_x=expand_extent_x, expand_extent_y=expand_extent_y)

        # Add the hillshade
        self.hillshade.maskFromVector(self.aoi_gdf, 2056)
        self.hillshade.show_on_map(ax=ax, cmap='Greys_r', vmin=0, vmax=255, bar=False, alpha=hillshade_alpha)

        # Add the DEM reclassified
        self.dem_reclass.show_on_map(ax=ax, alpha=dem_alpha, bar=False, cmap=dem_cmap)

        # Add the main locations
        if show_places:
            self.aoi_places[self.aoi_places.fclass=='town'].plot(ax=ax, color='black', edgecolor='white', markersize=50)
            for row in self.aoi_places[self.aoi_places.fclass=='town'].iloc:
                ax.annotate(row['name'], xy=(row.geometry.x, row.geometry.y+800), color='white', bbox={"edgecolor":'white', 'facecolor':'black', 'alpha':places_label_alpha}, ha='center', fontsize=places_labels_fontsize).set_zorder(100)

        # Add the waterways
        self.aoi_lakes.plot(ax=ax, color=water_color, alpha=water_alpha)

        # Glaciers
        self.aoi_sgi16.plot(ax=ax, facecolor=glaciers_color, alpha=glaciers_alpha)
        self.aoi_sgi16.plot(ax=ax, facecolor='none', edgecolor=glaciers_color, linewidth=0.75, alpha=glaciers_alpha)

        # Add north arrow and scale bar
        if north_arrow_size > 0 :
            vt.add_scale_and_north(
                ax, 
                x_loc = metadata_x_loc, 
                y_loc = metadata_y_loc, 
                gap_between_arrow_and_scale_bar = gap_between_arrow_and_scale_bar,
                scale_bar_length = scale_bar_length,
                scale_bar_units = scale_bar_units,
                scale_bar_fontsize = scale_bar_fontsize,
                north_arrow_size = north_arrow_size,
                north_arrow_color = north_arrow_color)
        else :
            vt.add_scale_bar(
                ax, 
                length=scale_bar_length, 
                location=(metadata_x_loc, metadata_y_loc), 
                units=scale_bar_units, 
                fontsize=scale_bar_fontsize)

        # Title
        ax.set_title(f'Study Area: {self.aoi_name}', fontsize=12)

        # Save on the instance
        if inplace:
            self.basic_map_ax = ax
            self.basic_map_fig = fig        

        return fig, ax

    def get_rgs_map(
        self,
        hue='rgu_activity_class_wu',
        ax=None,
        rg_body_alpha=0.5,
        rg_edge_linewidth=1,
        legend_fontsize=8,
        show_num_rgs=True,
        legend_loc=None,
        **kwargs
    ):
        """
        Make a map with rock glaciers colored by a given column

        hue: str, name of the column to use for coloring the rock glaciers
        It should be one of the keys of self.gc, e.g. 'rgu_activity_class_wu', 'cardi_pt', 'rgu_kin_att'
        Additional keyword arguments are passed to get_basic_map.
        """

        # If no specific ax template is given, create a new basic map - or use the existing one
        # Deepcopy to avoid modifying the existing one
        # And return another ax and fig
        if ax is None:
            fig, ax = deepcopy(self.get_basic_map(ax=ax, **kwargs))
        else:
            fig = ax.figure

        # Pre-process the main layer to remove nans and affect colors
        sub = self.pre_process_colors(hue)

        # Add rock glaciers footprints
        sub.plot(ax=ax, facecolor=sub['temp_color'], alpha=rg_body_alpha)
        sub.plot(ax=ax, facecolor='none', edgecolor=sub['temp_color'], linewidth=rg_edge_linewidth)

        # Add a legend to the map
        legend_elements = [Patch(facecolor=color, label=f"{label} ({len(sub[sub[hue] == label])})") for label, color in self.gc[f'{hue}_colors'].items()]
        ax.legend(handles=legend_elements, title='', fontsize=legend_fontsize, loc=legend_loc).set_zorder(100)

        # Title
        ax.set_title(f'Rock Glaciers {self.aoi_name} and {hue}\n({len(sub)} units)', fontsize=12)
        return fig, ax

    def pre_process_colors(self, hue, sub=None):
        
        # Cumulative selections
        if sub is None:
            sub = self.main_layer.copy()

        # Explicitly affect a color to each feature from the column acti cl without uncertainty
        sub['temp_color'] = sub[hue].map(self.gc[f"{hue}_colors"])

        # Remove nan values for the color column 
        sub = sub.dropna(subset=['temp_color'])

        return sub
        
    def get_boxplots_hue_and_y(self, hue='rgu_activity_class', y='alti_oue_med', ax=None, figsize=(5, 8), show_lines=True, linecolor='grey', sub=None, **kwargs):
        """
        Plot boxplots of a given quantitative column vs a given qualitative column

        Args:
            self (RockGlacierInventory): The rock glacier inventory object
            hue (str): The qualitative column name
            y (str): The quantitative column name
            ax (matplotlib.axes.Axes): The axes to plot on
            figsize (tuple): The figure size if ax is None
            **kwargs: Additional arguments to pass to sns.boxplot
        Returns:
            ax (matplotlib.axes.Axes): The axes with the boxplots   
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Pre-process the main layer to remove nans and affect colors
        if sub is None:
            sub = sub.main_layer.copy()

        # Remove the nans values for hue
        sub = sub.dropna(subset=[hue, y])

        # Order the hue categories
        order = self.gc[f'{hue}_order']

        # Create the boxplot
        sns.boxplot(data=sub, x=hue, y=y, order=order, palette=self.gc[f'{hue}_palette'], ax=ax, **kwargs)

        # Add connecting lines
        if show_lines:
            sub = sub.set_index(hue).loc[order].reset_index() if set(order).issubset(sub[hue].unique()) else sub.sort_values(by=hue)
            sns.lineplot(data=sub, x=hue, y=y, estimator='median', ci=None, color=linecolor, marker='o', ax=ax)

        # Customize chart
        ax.grid(visible=True, axis='y', linewidth=0.5, color='grey', linestyle='dashed')
        ax.set_ylabel(get_labels(y))
        ax.set_xlabel(get_labels(hue))

        # Title
        ax.set_title(f'{y} vs {hue}', fontsize=10)

        return ax

    def get_medvalues_x1_x2_y(self, x1='rgu_activity_class_wu', x2='cardi_pt', y='alti_oue_med'):
        """
        Create a dataframe with the median values of y for each combination of x1 and x2
        Args:
            self (RockGlacierInventory): The rock glacier inventory object
            x1 (str): The first qualitative column name - rgu_activity_class or cardi_pt or rgu_kin_att
            x2 (str): The second qualitative column name - rgu_activity_class or cardi_pt or rgu_kin_att
            y (str): The quantitative column name - alti_oue_med, alti_oue_min, alti_oue_max, alti_oue_range, alti_oue_std, etc.
        Returns:
            df (pd.DataFrame): The dataframe with the median values of y for each combination of x1 and x2
        """

        # Pre-process the main layer to remove nans and affect colors
        sub = self.pre_process_colors(x1)
        sub = self.pre_process_colors(x2)

        # Create a dataframe with the median values of y for each combination of x1 and x2
        df = sub.groupby([x1, x2])[y].median().reset_index()

        # Pivot the dataframe to have x1 as a regular column (not index)
        df = df.pivot(index=x1, columns=x2, values=y).reset_index()

        return df

    def get_boxplots_x1_x2_y(self, ax=None, x1='rgu_activity_class_wu', x2='cardi_pt', y='alti_oue_med', figsize=(10, 5), show_boxes=True, show_lines=True, **kwargs):
        """
        Create a figure with n axes. n is the number of unique values in x1.
        Each axis contains a boxplot of y vs x2 for the subset of data where x1 is equal to the unique value.
        Args:
            self (RockGlacierInventory): The rock glacier inventory object
            x1 (str): The first qualitative column name - rgu_activity_class or cardi_pt or rgu_kin_att
            x2 (str): The second qualitative column name - rgu_activity_class or cardi_pt or rgu_kin_att
            y (str): The quantitative column name - alti_oue_med, alti_oue_min, alti_oue_max, alti_oue_range, alti_oue_std, etc.
            ax (matplotlib.axes.Axes): The axes to plot on
            figsize (tuple): The figure size if ax is None
            **kwargs: Additional arguments to pass to sns.boxplot
        Returns:
            fig (matplotlib.figure.Figure): The figure containing the axes
            axs (list of matplotlib.axes.Axes): The list of axes with the boxplots        
        """

        # Pre-process the main layer to remove nans and affect colors
        sub = self.pre_process_colors(x1)
        sub = self.pre_process_colors(x2)

        # Get the unique values of x1
        unique_x1 = self.gc[f'{x1}_order']
        n_unique = len(unique_x1)

        # Create the figure and axes if ax is None
        if ax is None:
            fig, axs = plt.subplots(1, n_unique, figsize=figsize, sharey=True)
            axs = axs.flatten()
        else:
            fig = ax.figure
            axs = [ax]

        # For each unique value of x1, create a boxplot of y vs x2 for the subset of data where x1 is equal to the unique value        
        for i, val in enumerate(unique_x1):
            sub_sub = sub[sub[x1] == val]
            order = self.gc[f'{x2}_order']
            if show_boxes:
                sns.boxplot(data=sub_sub, x=x2, y=y, order=order, color=self.gc[f'{x1}_colors'][val], ax=axs[i], **kwargs)
            if show_lines:
                sub_sub_sorted = sub_sub.set_index(x2).loc[order].reset_index() if set(order).issubset(sub_sub[x2].unique()) else sub_sub.sort_values(by=x2)
                sns.lineplot(data=sub_sub_sorted, x=x2, y=y, estimator='median', ci=None, color=self.gc[f'{x1}_colors'][val], marker='o', ax=axs[i])
    
            axs[i].set_title(f'{val}')
            if i == 0:
                axs[i].set_ylabel(y)
            else:
                axs[i].set_ylabel('')

            # Customize chart
            axs[i].grid(visible=True, axis='y', linewidth=0.5, color='grey', linestyle='dashed')

        # Adjust layout
        plt.tight_layout()

        return fig, axs

    def get_rgs_counts(self):
        return vt.count_coalescent_systems(self.main_layer)

    def get_rgs_layer(self):
        self.rgs_layer = self.main_layer.dissolve().explode(index_parts=False)
        return rgs_layer
    
    def get_scatter_plot_hue_y1_y2(
        self,
        hue,
        y1,
        y2,
        reg_line_width=1,
        s=5,
        sub=None,
        ax=None,
        **kwargs
    ):

        if sub is None:
            sub = self.main_layer.copy()
        sub = sub.dropna(subset=hue)

        aida.explore_linear_relation(data=sub, x=y1, y=y2, s=s, hue=hue, palette_dict=self.gc[f"{hue}_colors"], hue_order=self.gc[f"{hue}_order"], reg_line_width=reg_line_width, ax=ax, **kwargs)
        return ax
            
    def get_fcc_plot(self, y, hue, ax=None, figsize=(5,8), legend=False, val_to_avoid=[], sub=None):
        """
        Draw a chart with the cumulative frequencies for each hue class (qualtitative) on the y variable (quantitative)
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if type(val_to_avoid) == str:
            val_to_avoid = [val_to_avoid]

        if sub is None:
            sub = self.pre_process_colors(hue)

        for val in self.gc[f'{hue}_order']:

            if val in val_to_avoid:
                continue

            # Get the quantitative metric of the rock glaciers
            sorted_ys = sub[sub[hue] == val][y].values
            sorted_ys.sort()

            # normalise les positions x pour qu'elles soient comprises entre 0 et 100, et pas entre 1 et n (avec n=nombre d'individus dans chaque classe)
            x = np.linspace(0, 100, num=len(sorted_ys))

            # Scatterplot
            sns.scatterplot(x=x, y=sorted_ys, ax=ax, color=self.gc[f'{hue}_colors'][val], s=10, linewidth=0, alpha=1)

            if legend:
                sns.lineplot(x=x, y=sorted_ys, ax=ax, color=self.gc[f'{hue}_colors'][val], alpha=0.8, linewidth=1, label=val)
                ax.legend()

            else:
                sns.lineplot(x=x, y=sorted_ys, ax=ax, color=self.gc[f'{hue}_colors'][val], alpha=0.8, linewidth=1)

        ax.set_title(y)
        return ax

    def countplot_x1_x2(self, x1, x2, ax=None, figsize=(30, 10)):
        """
        Countplots of each class x1 for each class x2    
        """
        
        # Pre-process
        sub = self.pre_process_colors(x1)
        sub = self.pre_process_colors(x2, sub=sub)

        # Create ax
        unique_x1 = sub[x1].dropna().unique()
        n = len(unique_x1)

        # Ensure even number of axes (if n is odd, add 1)
        if n % 2 != 0:
            n += 1

        # Create a mosaic layout with n axes in a single row
        if ax is None:
            fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
        else:
            axes = [ax]

        for i, val in enumerate(self.gc[f"{x1}_order"]):
            ax_i = axes[i] if len(axes) > 1 else axes[0]
            data = sub[sub[x1] == val]
            sns.countplot(data=data, x=x2, palette=self.gc.get(f"{x2}_palette", None), order=self.gc.get(f"{x2}_order", None), ax=ax_i)
            ax_i.set_title(f"{x1}: {val}")
            ax_i.set_xlabel(x2)
            ax_i.set_ylabel("count")
            ax_i.tick_params(axis='x', rotation=45)


        # Compute the global 80% y-value across all axes
        y_max_global = max(ax_i.get_ylim()[1] for ax_i in axes)
        y_min_global = min(ax_i.get_ylim()[0] for ax_i in axes)
        y_80_global = y_min_global + 0.9 * (y_max_global - y_min_global)
        for i, val in enumerate(self.gc[f"{x1}_order"]):
            ax_i = axes[i] if len(axes) > 1 else axes[0]
            ax_i.axhline(y=y_80_global, color=self.gc[f"{x1}_colors"][val], linewidth=40)

        return ax

    def get_quali_request(self, criterias={}):
        sub = rogi.main_layer.copy()
        for col in list(criterias.keys()):
            sub = sub[sub[col].isin(criterias[col])]
        return sub

def get_labels(col):
    try:
        h_label = human_keys[col]
    except KeyError:
        h_label = col
    return h_label