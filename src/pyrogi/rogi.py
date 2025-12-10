# Standard Libs
import json
import warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime

# Stats & Number libs
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from random import randint, choice

# Viz libs
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Geo libs
import geopandas as gpd

# Silent some warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")
from telenvi import vector_tools as vt
from telenvi import raster_tools as rt
from telenvi import aida
rt.VERBOSE=False

# Pyrogi other components
from pyrogi.rgu import RockGlacierUnit
import pyrogi.colors as rgc
from pyrogi.__init__ import datamap, temp_figs_dir

# Open the study area
ROGI_REGIONS_LAYER = gpd.read_file(datamap['auxiliaires_path'], layer=datamap['rogi_regions_layername'])

def Open(aoi_name, meta_figs_dir=temp_figs_dir, load_external_datasets=True, aoi_rasters_res=20, **kwargs):

    # Only arguments to change
    figs_dir = f'{meta_figs_dir}_{aoi_name}'

    # Open the study area
    aoi = ROGI_REGIONS_LAYER[ROGI_REGIONS_LAYER.rogi_reg_name == aoi_name]

    # Open the whole rogi enhanced with altitudinal data
    rgdf = gpd.read_file(datamap['rogi_ch_3'], layer=f'rogi_ch_3_{aoi_name}')

    # Create the RockGlacierInventory instance
    rogi_instance = RockGlacierInventory(
        b_layer=rgdf,
        aoi=aoi,
        aoi_name=aoi_name,
        out_figs_dir=figs_dir,
        aoi_rasters_res=aoi_rasters_res,
        **kwargs,
    )

    # Load external data
    rogi_instance.pre_process_main_layer(load_external_datasets=load_external_datasets)

    return rogi_instance

class RockGlacierInventory:
    """
    Describe a Rock Glacier Inventory dataset.

    It can be instantiated from either:
    - the Primary Markers layer and the Outlines layer from ROGI_2 (raw mode)
    - the Boosted layer from ROGI_2 + alti metrics (advanced mode)
    The class contains methods to pre-process the main layer, compute topographic metrics, and create maps.
    
    Keywords:
    pms_layer : gpd.GeoDataFrame, the Primary Markers layer from ROGI_2
    ous_layer : gpd.GeoDataFrame, the Outlines layer from ROGI_2
    b_layer : gpd.GeoDataFrame, the Boosted layer from ROGI_2 + alti metrics
    aoi_name : str, name of the area of interest
    aoi : gpd.GeoDataFrame, the area of interest geometry
    aoi_buffer_size : int, buffer size in meters to apply to the convex hull of the rock glaciers
    epsg : int, EPSG code of the coordinate reference system
    out_figs_dir : str, path to the output directory for the figures
    out_figs_version_note : str, note to add to the output directory name for versioning
    self.aoi_rasters_res : int, resolution of the DEM to use for the area of interest
    aoi_dem_reclass_step : int, step in meters for reclassifying the DEM
    aoi_hillshade_azimut : int, azimuth of the hillshade
    aoi_hillshade_altitude : int, altitude of the hillshade
    aoi_places : gpd.GeoDataFrame, places from OSM within the area of interest
    aoi_lakes : gpd.GeoDataFrame, lakes from OSM within the area of interest
    aoi_sgi16 : gpd.GeoDataFrame, glaciers from SGI 2016 within the area of interest
    aoi_sgi1850: gpd.GeoDataFrame, glaciers from SGI 1850 within the area of interest
    dem : raster_tools object, the DEM for the area of interest
    dem_reclass : raster_tools object, the reclassified DEM for the area of interest
    hillshade : raster_tools object, the hillshade for the area of interest
    gc : dict, graphical chart dictionary for the qualitative variables
    boosted : bool, whether the boosted layer is available or not

    Methods:
    load_external_datasets_from_datamap : load external datasets from a datamap json file
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
    pms_layer : gpd.GeoDataFrame, the Primary Markers layer from ROGI_2
    ous_layer : gpd.GeoDataFrame, the Outlines layer from ROGI_2
    oue_layer : gpd.GeoDataFrame, the Outlines Extended layer from ROGI_2
    our_layer : gpd.GeoDataFrame, the Outlines Restricted layer from ROGI_2
    b_layer : gpd.GeoDataFrame, the Boosted layer from ROGI_2 + alti metrics

    datamap : json file with all the paths
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
        aoi_rasters_res = 20,
        aoi_dem_reclass_step = 500,
        aoi_hillshade_azimut = 315,
        aoi_hillshade_altitude = 45,

        # Other
        epsg=2056,
        out_figs_dir=None,
        ):

        # Define attributes
        self.pms_layer = pms_layer
        self.ous_layer = ous_layer
        self.epsg=epsg

        # ROGI_2 (raw mode)
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

        # ROGI_2 + alti metrics (advanced mode)
        elif b_layer is not None:
            self.b_layer = b_layer
            self.main_layer = b_layer

        # Default figures outputs dir
        self.aoi_name = aoi_name
        if out_figs_dir is None:
            self.out_figs_version_note = out_figs_version_note
            self.out_figs_dir = f"{temp_figs_dir}/{self.get_complete_name()}"
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
            'status_colors':rgc.status_colors,

            'cardi_pt_order': rgc.cardi_pt_order,
            'rgu_activity_class_order': rgc.rgu_activity_class_order,
            'rgu_activity_class_wu_order': rgc.rgu_activity_class_wu_order,
            'rgu_kin_att_order': rgc.rgu_kin_att_order,
            'upslope_con_order':rgc.upslope_con_order,
            'status_order':rgc.status_order,

            'cardi_pt_palette': rgc.cardi_pt_palette,
            'rgu_activity_class_palette': rgc.rgu_activity_class_palette,
            'rgu_activity_class_wu_palette': rgc.rgu_activity_class_wu_palette,
            'rgu_kin_att_palette': rgc.rgu_kin_att_palette,
            'upslope_con_palette':rgc.upslope_con_palette,
            'status_palette':rgc.status_palette,

            'cardi_pt_cmap': rgc.cardi_pt_cmap,
            'rgu_activity_class_cmap': rgc.rgu_activity_class_cmap,
            'rgu_activity_class_wu_cmap': rgc.rgu_activity_class_wu_cmap,
            'rgu_kin_att_cmap': rgc.rgu_kin_att_cmap,
            'upslope_con_cmap':rgc.upslope_con_cmap,
            'status_cmap':rgc.status_cmap,
        }

        # DEM on the AOI
        self.aoi_rasters_res = aoi_rasters_res
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

    def load_external_datasets_from_datamap(self):
        """
        Necessary to create nice map backgrounds. 
        Load DEMS, glaciers, lakes... from a json file called datamap. Specific for TD. 

        Keywords:
        datamap : dict, dictionnary with all the paths to the datasets

        Returns:
        None, but set the following attributes:
        aoi_places : gpd.GeoDataFrame, places from OSM within the area of interest
        aoi_lakes : gpd.GeoDataFrame, lakes from OSM within the area of interest
        aoi_sgi16 : gpd.GeoDataFrame, glaciers from SGI 2016 within the area of interest
        dem : raster_tools object, the DEM for the area of interest
        dem_reclass : raster_tools object, the reclassified DEM for the area of interest
        hillshade : raster_tools object, the hillshade for the area of interest
        Note: this method is specific to the datamap file available on Thib's PC
        """

        # TODO: make it more generic
        # The DEMS, Hillshades and PISR are computed manually in Qgis. This should be automated in the future
        # Also, the arguments given here such as dem_res, hillshade_azimut, etc. are then specific to the files that are available on disk

        # DEM
        self.alti_3d_path = datamap['sa3d_dir']
        self.derived_sa3d = Path(self.alti_3d_path, 'derived_products')
        self.dem = rt.Open(Path(datamap['alti_3D_20']), load_pixels=True, geoExtent=self.aoi_geom, nRes=self.aoi_rasters_res)

        # Reclassify the DEM each 500m
        self.dem_reclass = aida.get_manual_clusters(self.dem, np.arange(0, 4500, self.aoi_dem_reclass_step))

        # Hillshade
        self.hillshade = rt.Open(Path(datamap['hillshade_alti3D_20']), load_pixels=True,  geoExtent=self.aoi_geom, nRes=self.aoi_rasters_res)

        # Glaciers
        sgi16 = gpd.read_file(datamap['glaciers_inv'], layer='sgi_2016_glaciers')
        sgi1850 = gpd.read_file(datamap['glaciers_inv'], layer='sgi_1850_glaciers')

        # Places and waterways from OSM
        osm_path = datamap['osm_dir']
        places = gpd.read_file(Path(osm_path, 'gis_osm_places_free_1.shp')).to_crs(epsg=2056)
        lakes = gpd.read_file(Path(osm_path, 'gis_osm_water_a_free_1.shp')).to_crs(epsg=2056)
        lakes = lakes[lakes.fclass != 'glacier']

        # Spatial selection within the area
        self.aoi_places = places.clip(self.aoi_gdf)
        self.aoi_lakes =  lakes.clip(self.aoi_gdf)
        self.aoi_sgi16 =  sgi16.clip(self.aoi_gdf)
        self.aoi_sgi1850 = sgi1850.clip(self.aoi_gdf)

        # Attribute selection
        lakes = lakes[lakes.fclass != 'glacier']

    def get_median_values_in_geoim(self, target_geoim, colname='raster_median'):
        """
        Add a column to the self.main_layer by reading a raster and computing the median for each polygon
        """

        # Compute the median value in each polygon
        self.main_layer[colname] = self.main_layer.apply(lambda row: target_geoim.compute_med_value_in_georow(row), axis=1)
        n_before = len(self.main_layer)

        # Drop the rows with nodata
        self.main_layer = self.main_layer.dropna(subset=[colname])
        n_after = len(self.main_layer)

        # Make a brief report
        print(f"{round((n_before - n_after) / n_before * 100, 2)}% of rock glaciers have nodata for {colname}")

        # Update b_layer
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

        # Create a binary status column : intact or relict
        def _intact_or_relict(row):
            if row.rgu_activity_class_wu == 'relict': return 'relict'
            elif row.rgu_activity_class_wu == 'active' or row.rgu_activity_class_wu == 'transitional': return 'intact'
        self.main_layer['status'] = self.main_layer.apply(lambda row: _intact_or_relict(row) ,axis=1)

        # Apply a conversion from rock glacier elevation to the Mean Annual Air Temperature with a simple linear model of 0.55°C / 100m and a base value from the Tnorm product
        def _thermal_model(alti):
            return 13.404878856909363 + (-0.55/100) * alti
        self.main_layer['maat'] = self.main_layer.apply(lambda row: _thermal_model(row.alti_oue_med), axis=1)

        # Load external datasets
        if load_external_datasets:
            self.load_external_datasets_from_datamap()
        
        else:
            self.hillshade = None
            self.dem_reclass = None
            self.dem = None
            self.aoi_places = None
            self.aoi_lakes = None
            self.aoi_sgi16 = None
            
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

    def get_alti_metrics(self, dem_map, output_path, output_layername, aoi_rasters_res_order_1=5, aoi_rasters_res_order_2=0.5, save_steps=10, only_order_1=False, already_processed_layer=None):
        """
        Compute topographic metrics for the features of a rogi
        aoi_rasters_res_order_1 and 2 are for different applications (altitude, aspect or slope within the margins)
        Save the output layer on disk
        Return the boosted layer as a gpd.GeoDataFrame
        Keywords:
        dem_map: raster_tools object, the DEM to use for the calculations
        output_path: str, path to the output geopackage
        output_layername: str, name of the layer within the geopackage
        aoi_rasters_res_order_1: float, resolution order for the main calculations (altitude, aspect)
        aoi_rasters_res_order_2: float, resolution order for the secondary calculations (slope within the margins)
        save_steps: int, number of steps to wait before saving a temporary version of the boosted layer
        only_order_1 : Boolean, control if the algorithm will reload the DEM with the much more precise DEM to compute detailled metrics (at present, only the variables related to the margins, i.e. in between the extended and restricted outlines)
        already_processed_layer: a layer which contains the features where the alti_metrics step has already been performed - that we need to avoid in the calculation to increase the velocity
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
            
            # Check if already processed
            if already_processed_layer is not None:
                if target_rg.pm_pid_nov25 in already_processed_layer.pm_pid.unique():
                    print(f"{target_rg.pm_pid} already processed")

                    # Get the associated already processed polygon
                    rg_detailled_serie = already_processed_layer[already_processed_layer.pm_pid == target_rg.pm_pid].iloc[0]

                    # Change the geometry column name (what happened in the step "make_b_layer")
                    rg_detailled_serie['oue_geom'] = rg_detailled_serie.geometry
                    rg_detailled_serie = rg_detailled_serie.drop(['geometry'])
                    b_series.append(rg_detailled_serie)
                    continue

            try:
                # Fabrication d'une entité RGU du pyrogi
                rg = RockGlacierUnit(target_rg)

                # Chargement des polygones
                rg.load_outlines(self.ous_layer)

                # Initialisation du DEM sur l'emprise du glacier rocheux
                rg.initialize_dem(dem_map, verbose=False)

                # Lecture et calcul des variables topographiques
                rg_detailled_serie = rg.get_detailled_serie(dem_map, aoi_rasters_res_order_1, aoi_rasters_res_order_2, only_order_1=only_order_1)

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
        if 'geometry' in b_layer.columns:
            b_layer = b_layer.drop(['geometry'], axis=1)
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
        figsize = (15, 15),
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
        raster_res=None,
        show_dem_bar=False,
        dem_bar_fraction=0.9,
        dem_bar_pad=0.01,
        dem_bar_ticks_pos='left',
        dem_bar_orientation='vertical',
        dem_bar_ticks_fontsize=8,
        dem_bar_label=None,
        dem_bar_label_fontsize=10,
        dem_bar_label_position='bottom',
        dem_bar_ticks_fontcolor='black',
        dem_bar_label_fontcolor='black',
        ):
        """
        Make a basic map with hillshade, dem, glaciers, lakes and places
        Additional keyword arguments:
        mask_alpha: float, alpha for the aerial image mask
        figsize: tuple, figsize for the map
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
            fig, ax = plt.subplots(figsize=figsize)

        # If the basic map is not available but a template is given as axis, use it, and extract it's figure        
        else:
            ax = ax
            fig = ax.figure            

        # Add aerial image
        if aerial_view:
            vt.add_wmts_layer(self.aoi_gdf, mask_outside_geo_target=True, epsg=2056, ax=ax, mask_color='white', mask_alpha=mask_alpha, expand_extent_x=expand_extent_x, expand_extent_y=expand_extent_y)

        # Add the hillshade
        if self.hillshade is not None:
            
            # Change the pixel size if needed (faster with big res)
            if raster_res is not None:
                hs = self.hillshade.resize(raster_res)
            else:
                hs = self.hillshade.copy()

            hs.maskFromVector(self.aoi_gdf, 2056)
            hs.show_on_map(ax=ax, cmap='Greys_r', vmin=0, vmax=255, bar=False, alpha=hillshade_alpha)

        # Add the DEM reclassified
        if self.dem_reclass is not None:

            # Change the pixel size if needed (faster with big res)
            if raster_res is not None:
                dem = self.dem.resize(raster_res)
            else:
                dem = self.dem.copy()

            dem.maskFromVector(self.aoi_gdf, 2056)
            dem.show_on_map(ax=ax, alpha=dem_alpha, bar=show_dem_bar, cmap=dem_cmap, bar_fraction=dem_bar_fraction, bar_pad=dem_bar_pad, bar_ticks_pos=dem_bar_ticks_pos, bar_orientation=dem_bar_orientation, bar_ticks_fontsize=dem_bar_ticks_fontsize, bar_label=dem_bar_label, bar_label_fontsize=dem_bar_label_fontsize, bar_label_position=dem_bar_label_position, bar_ticks_fontcolor=dem_bar_ticks_fontcolor, bar_label_fontcolor=dem_bar_label_fontcolor)

        # Add the main locations
        if show_places and self.aoi_places is not None:
            self.aoi_places[self.aoi_places.fclass=='town'].plot(ax=ax, color='black', edgecolor='white', markersize=50)
            for row in self.aoi_places[self.aoi_places.fclass=='town'].iloc:
                ax.annotate(row['name'], xy=(row.geometry.x, row.geometry.y+800), color='white', bbox={"edgecolor":'white', 'facecolor':'black', 'alpha':places_label_alpha}, ha='center', fontsize=places_labels_fontsize).set_zorder(100)

        # Add the waterways
        if self.aoi_lakes is not None:
            self.aoi_lakes.plot(ax=ax, color=water_color, alpha=water_alpha)

        # Glaciers
        if self.aoi_sgi16 is not None:
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
        legend_loc=None,
        sub=None,
        color=None,
        add_counts=True,
        labels_to_add_to_legend=[],
        colors_to_add_to_legend=[],
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

        if sub is None:
            sub = self.main_layer.copy()

        # Pre-process the main layer to remove nans and affect colors
        sub = self.pre_process_colors(hue, sub=sub)

        # Add rock glaciers footprints
        if color is None:
            sub.plot(ax=ax, facecolor=sub['temp_color'], alpha=rg_body_alpha)
            sub.plot(ax=ax, facecolor='none', edgecolor=sub['temp_color'], linewidth=rg_edge_linewidth)
        else:
            sub.plot(ax=ax, facecolor=color, alpha=rg_body_alpha)
            sub.plot(ax=ax, facecolor='none', edgecolor=color, linewidth=rg_edge_linewidth)

        # Add a legend to the map with the basic elements
        if add_counts:
            legend_elements = [Patch(facecolor=color, label=f"{label} ({len(sub[sub[hue] == label])})") for label, color in self.gc[f'{hue}_colors'].items()]
        else:
            legend_elements = [Patch(facecolor=color, label=label) for label, color in self.gc[f'{hue}_colors'].items()]

        # Add custom labels/colors to the legend if needed
        for lbl, clr in zip(labels_to_add_to_legend, colors_to_add_to_legend):
            legend_elements.append(Patch(facecolor=clr, label=lbl))

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
        
    def get_boxplots_hue_and_y(self, hue='rgu_activity_class', y='alti_oue_med', ax=None, figsize=(10, 5), show_lines=True, linecolor='grey', sub=None, **kwargs):
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
            sub = self.main_layer.copy()

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
        ax.set_ylabel(get_nc(y))
        ax.set_xlabel(get_nc(hue))

        # Title
        ax.set_title(f'{y} vs {hue}', fontsize=10)

        return ax    

    def get_boxplots_hue_and_y_plotly(self, 
        hue='rgu_activity_class_wu', 
        y='alti_oue_med', 
        sub=None,
        ybounds=None):
        """
        Plotly version: boxplots of a given quantitative column vs a given qualitative column.
        Allows user to change hue interactively in the chart.

        Args:
            self (RockGlacierInventory): The rock glacier inventory object
            hue (str): The qualitative column name (default 'rgu_activity_class_wu')
            y (str): The quantitative column name
            sub (GeoDataFrame | DataFrame): optional subset
            ybounds (tuple): y-axis bounds
        Returns:
            fig (plotly.graph_objs.Figure)
        """

        # Allowed hue options
        hue_options = ['cardi_pt', 'rgu_activity_class_wu', 'rgu_activity_class', 'rgu_kin_att', 'upslope_con', 'status']

        # Ensure data validity
        if sub is None:
            sub = self.main_layer.copy()
        sub = sub.dropna(subset=[hue, y])

        # Create boxplots for all hue options and store their traces
        all_traces = []
        all_layouts = []
        for h in hue_options:
            filtered_sub = sub.dropna(subset=[h, y])
            ord_h = self.gc.get(f'{h}_order', sorted(filtered_sub[h].unique()))
            pal_h = self.gc.get(f'{h}_colors', None)

            # Convert RGB tuples in palette to hex strings if needed
            def rgb_to_hex(c):
                if isinstance(c, tuple) and len(c) == 3:
                    return '#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255))
                return c
            pal_h_hex = {k: rgb_to_hex(v) for k, v in pal_h.items()} if pal_h else None

            box = px.box(
                filtered_sub,
                x=h,
                y=y,
                category_orders={h: ord_h},
                color=h,
                color_discrete_map=pal_h_hex,
            )
            all_traces.append(box.data)
            all_layouts.append({
                'xaxis': {'title': get_nc(h)},
                'yaxis': {'title': get_nc(y)},
                'title': f"{get_nc(y)} vs {get_nc(h)}"
            })

        # Flatten all traces into a single list
        traces = []
        trace_visibility = []
        for i, trace_group in enumerate(all_traces):
            for trace in trace_group:
                # Only show traces for the first hue option initially
                trace.visible = (i == 0)
                traces.append(trace)
                trace_visibility.append(i)

        # Create figure with all traces
        fig = go.Figure(data=traces)
        fig.update_layout(
            yaxis_title=get_nc(y),
            xaxis_title=get_nc(hue_options[0]),
            title=f"{get_nc(y)} vs {get_nc(hue_options[0])}",
            boxmode="group",
            template="plotly_white"
        )

        if ybounds is not None:
            fig.update_yaxes(range=ybounds)

        # Add dropdown for hue selection
        dropdown_buttons = []
        trace_counts = [len(tg) for tg in all_traces]
        start = 0
        for i, h in enumerate(hue_options):
            # Build visibility mask for all traces
            vis = []
            for j, cnt in enumerate(trace_counts):
                vis.extend([j == i] * cnt)
            dropdown_buttons.append(dict(
                method="update",
                label=h,
                args=[
                    {"visible": vis},
                    all_layouts[i]
                ]
            ))

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )

        return fig

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

    def get_boxplots_x1_x2_y(self,x1='rgu_activity_class_wu', x2='cardi_pt', y='alti_oue_med', figsize=(10, 5), show_boxes=True, show_lines=True,  ax=None, sub=None, **kwargs):
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

        if sub is None:
            sub = self.main_layer.copy()
        
        # Pre-process the main layer to remove nans and affect colors
        sub = self.pre_process_colors(x1, sub)
        sub = self.pre_process_colors(x2, sub)

        # Get the unique values of x1
        unique_x1 = self.gc[f'{x1}_order']
        n_unique = len(unique_x1)

        # Create the figure and axes if ax is None
        if ax is None:
            fig, axs = plt.subplots(1, n_unique, figsize=figsize, sharey=True, sharex=True)
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
        return self.rgs_layer

    def get_cluster_kmeans(self, y1, y2, n_clusters=3, n_init=10, random_state=None, sub=None):
        """
        Send a geodataframe with a new column, "g{y1}-{y2}", which correspond to the cluster of which each rock glacier belongs to, according to a kmeans clustering on the two quantitative variables y1 and y2
        """

        # Pre-process the main layer to remove nans and affect colors
        if sub is None:
            sub = self.main_layer.copy()
            sub = sub.dropna(subset=[y1, y2])

        # Get the kmeans clustering from aida
        labels, centers, estimator = aida.get_clusters_kmeans_from_df(sub, y1, y2, n_clusters=n_clusters, n_init=n_init, random_state=random_state)

        # Get the cluster labels
        sub[f"clu_lab_{y1}_{y2}"] = labels
        sub[f"g{y1}-{y2}"] = labels.astype(str)

        return sub, centers, estimator

    def get_scatter_plot_hue_y1_y2(
        self,
        y1,
        y2,
        reg_line_width=1,
        s=5,
        hue=None,
        sub=None,
        ax=None,
        show_score=True,
        show_legend=True,
        **kwargs
    ):

        if sub is None:
            sub = self.main_layer.copy()

        if hue is not None:
            sub = sub.dropna(subset=[hue, y1, y2])
        
        else:
            sub = sub.dropna(subset=[y1, y2])

        if hue is None:
            ax, _ =aida.explore_linear_relation(data=sub, x=y1, y=y2, s=s, reg_line_width=reg_line_width, ax=ax, show_score=show_score, show_legend=show_legend, **kwargs)
            
        else:
            ax, _ =aida.explore_linear_relation(data=sub, x=y1, y=y2, s=s, hue=hue, palette_dict=self.gc[f"{hue}_colors"], hue_order=self.gc[f"{hue}_order"], reg_line_width=reg_line_width, ax=ax, show_score=show_score, show_legend=show_legend, **kwargs)

        ax.set_xlabel(get_nc(y1))
        ax.set_ylabel(get_nc(y2))
        ax.set_title(f'{y1} vs {y2}' + (f' colored by {hue}' if hue is not None else ''))
        
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

    def countplot_x1_x2(self, x1, x2, ax=None, figsize=(30, 10), n=None, sub=None):
        """
        Countplots of each class x1 for each class x2    
        """
        
        if sub is None:
            sub = self.main_layer.copy()

        # Pre-process
        sub = self.pre_process_colors(x1, sub=sub)
        sub = self.pre_process_colors(x2, sub=sub)

        # Create ax
        unique_x1 = sub[x1].dropna().unique()

        if n is None:            
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

    def countplot_x1(self, x1, ax=None, figsize=(8, 5), sub=None):
        """
        Countplot of a qualitative column x1
        """

        if sub is None:
            sub = self.main_layer.copy()

        # Pre-process
        sub = self.pre_process_colors(x1, sub=sub)

        # Create ax
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.countplot(data=sub, x=x1, palette=self.gc.get(f"{x1}_palette", None), order=self.gc.get(f"{x1}_order", None), ax=ax)
        ax.set_title(f"Distribution of rock glaciers from {x1} | {self.aoi_name}")
        ax.set_xlabel(get_nc(x1))
        ax.set_ylabel("count")
        ax.tick_params(axis='x', rotation=45)

        return ax

    def get_surfplot_x1(self, x1, ax=None, figsize=(8, 5)):
        """
        Surface plot of a qualitative column x1
        """

        # Pre-process
        sub = self.pre_process_colors(x1)

        # Create ax
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Compute the surface area of each rock glacier
        sub['surface_area'] = sub.geometry.area / 10000

        # Compute the total surface area for each class of x1
        surface_areas = sub.groupby(x1)['surface_area'].sum().reset_index()

        sns.barplot(data=surface_areas, x=x1, y='surface_area', palette=self.gc.get(f"{x1}_palette", None), order=self.gc.get(f"{x1}_order", None), ax=ax)
        ax.set_title(f"Surface area of rock glaciers from {x1} | {self.aoi_name}")
        ax.set_xlabel(get_nc(x1))
        ax.set_ylabel("Surface area (ha)")
        ax.tick_params(axis='x', rotation=45)

        return ax

    def get_quali_request(self, criterias={}, mode='in', sub=None):
        """
        Criterias needs to looks like:
        criterias = {
            'rgu_activity_class_wu': ['active', 'inactive'],
            'cardi_pt': ['A', 'B']
        }
        """
        if sub is None:
            sub = self.main_layer.copy()
    
        if mode == 'in':
            for col in list(criterias.keys()):
                sub = sub[sub[col].isin(criterias[col])]
        else:
            for col in list(criterias.keys()):
                sub = sub[~sub[col].isin(criterias[col])]
        return sub
    
    def get_quanti_request(self, criterias={}, sub=None):
        """
        Criterias needs to looks like:
        criterias = {
            'alti_oue_med': (2000, 3000),
            'slope_oue_med': (10, 30)
        }
        """
        if sub is None:
            sub = self.main_layer.copy()

        for col in list(criterias.keys()):
            sub = sub[(sub[col] >= criterias[col][0]) & (sub[col] <= criterias[col][1])]
        return sub
    
    def get_histplot_x_hue(self, x, hue=None, ax=None, figsize=(8, 5), sub=None, color='grey', **kwargs):
        """
        Histogram of a quantitative column x, colored by a qualitative column hue
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if sub is None:
            sub = self.pre_process_colors(hue)

        # Remove nan values for x and hue

        if hue is not None:
            sub = sub.dropna(subset=[x, hue])
            sns.histplot(data=sub, x=x, hue=hue, hue_order=self.gc.get(f"{hue}_order", None), palette=self.gc.get(f"{hue}_palette", None), ax=ax, **kwargs)
            ax.set_title(f"Distribution of {get_nc(x)} colored by {get_nc(hue)} | {self.aoi_name}")

        else:
            sub = sub.dropna(subset=[x])
            sns.histplot(data=sub, x=x, ax=ax, color=color, **kwargs)
            ax.set_title(f"Distribution of {get_nc(x)} | {self.aoi_name}")
    
        ax.set_xlabel(get_nc(x))
        ax.set_ylabel("count")

        return ax

    def get_mas_scores(self, mas_layer):
        
        def _compute_vel_scores(row, target_vel_class, mas_layer):

            # Clip the moving areas within the RGU
            mas_rgu = mas_layer.clip(row.geometry)

            # Dissolve them by velocity class
            mas_rgu = mas_rgu.dissolve(by='vel_class_int', as_index=False, aggfunc='sum')

            # Select the moving areas with the target velocity class
            target_mas_rgu = mas_rgu[mas_rgu.vel_class_int == target_vel_class]

            # Remove GeometryCollections, keep only Polygon and MultiPolygon
            target_mas_rgu = target_mas_rgu[
                target_mas_rgu.geometry.type.isin(['Polygon', 'MultiPolygon'])
            ]

            # If there is an existent ma within the RGU
            if len(target_mas_rgu) > 0:

                # Remove areas which will be take into account during the higher velocity classes
                faster_mas_rgu = mas_rgu[mas_rgu.vel_class_int > target_vel_class]

                if len(faster_mas_rgu) > 0:
                    target_mas_rgu_dissolved = gpd.overlay(target_mas_rgu, faster_mas_rgu, how='difference')
                else:
                    target_mas_rgu_dissolved = target_mas_rgu

                # Remove GeometryCollections, keep only Polygon and MultiPolygon
                target_mas_rgu_dissolved = target_mas_rgu_dissolved[
                    target_mas_rgu_dissolved.geometry.type.isin(['Polygon', 'MultiPolygon'])
                ]

                # Clip to the outline geometry
                target_mas_rgu_dissolved = gpd.overlay(target_mas_rgu_dissolved, gpd.GeoDataFrame(geometry=[row.geometry], crs=mas_layer.crs), how='intersection')

                # Remove GeometryCollections, keep only Polygon and MultiPolygon after intersection
                target_mas_rgu_dissolved = target_mas_rgu_dissolved[
                    target_mas_rgu_dissolved.geometry.type.isin(['Polygon', 'MultiPolygon'])
                ]

                # Compute area ratio
                area_ratio = float(target_mas_rgu_dissolved.area.sum() / row.geometry.area) if row.geometry.area > 0 else 0
                return area_ratio
            
            else:
                return 0

        # Pre-process mas layer
        mas_layer['vel_class_int'] = mas_layer['Vel.Class'].astype(int)
        mas_layer = mas_layer.dropna(subset=['vel_class_int'])
        mas_layer['geometry'] = mas_layer['geometry'].buffer(0)
        foo=len(mas_layer)
        mas_layer = mas_layer[mas_layer.is_valid]
        print(f"{len(mas_layer) - foo} invalid geometries have been removed from the moving areas layer")

        # For each possible velocity class
        for target_vel_class in tqdm(mas_layer.vel_class_int.unique()):
            self.main_layer[f"kin_score_{target_vel_class}"] = self.main_layer.apply(
                lambda row: _compute_vel_scores(row, target_vel_class, mas_layer),
                axis=1)

        return None

    def random_forest_classifier(self,
        hue='status',
        drivers=['rgu_dir_ins', 'alti_oue_min'],
        eval_size=0.3,
        test_size=0.3,
        random_state=None,
        n_estimators=100,
        sub=None
    ):
        """
        Classify the rock glaciers into the classes of the hue column using a random forest classifier and the drivers
        Return the model, and the resulting classified dataset with a new column, the result of the  model predictions
        """
        
        # Copy the main layer if no subset is given
        if sub is None:
            sub = self.main_layer.copy()

        # Delete NaN values
        rgdf = sub.dropna(subset=hue)

        # Shuffle it to ensure homogeneous repartition
        rgdf = rgdf.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Show the Statistical distribution of the dataset from the hue variable
        pd.DataFrame([dict(rgdf.groupby(hue).pm_pid.count()/len(rgdf)*100)])

        # Split the model between train, test and evaluation sets
        border_tt_eval = int(len(rgdf)*(1-eval_size))

        # Train and set : the head of the whole dataset until the threshold defined by eval_size
        tt_set = rgdf.iloc[:border_tt_eval]

        # Evaluation set : the tail of the whole dataset
        eval_set = rgdf.iloc[border_tt_eval:]

        # Define features and target variable
        X_tt = tt_set[drivers]
        Y_tt = tt_set[hue]

        # Split the dataset into training and testing sets
        print(len(X_tt))
        print(len(Y_tt))

        X_train, X_test, Y_train, Y_test = train_test_split(
            X_tt, 
            Y_tt, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=Y_tt)

        # Create a Random Forest Classifier
        classifier = RandomForestClassifier(n_estimators=n_estimators)

        # Train the classifier (train set)
        classifier.fit(X_train, Y_train)

        # Test (test_set)
        Y_test_pred = classifier.predict(X_test)

        # Evaluation on the eval dataset
        X_eval = eval_set[drivers]
        Y_eval = eval_set[hue]
        Y_eval_pred = classifier.predict(X_eval)

        # Visualisation of the results on the test set
        print(classification_report(Y_test, Y_test_pred))

        # Evaluation on the eval dataset
        X_eval = eval_set[drivers]
        Y_eval = eval_set[hue]
        Y_eval_pred = classifier.predict(X_eval)
        print(classification_report(Y_eval, Y_eval_pred))

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)

        class_names = self.gc[f"{hue}_order"]

        aida.show_confusion_matrix(Y_test, Y_test_pred, class_names=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix - Test set')

        aida.show_confusion_matrix(Y_eval, Y_eval_pred, class_names=class_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix - Eval set')

        eval_set[f'{hue}_random'] = eval_set.apply(lambda row: choice(self.gc[f"{hue}_order"]), axis=1)

        aida.show_confusion_matrix(eval_set[hue], eval_set[f'{hue}_random'], class_names=class_names, ax=axes[2])
        axes[2].set_title('Confusion Matrix - Random prediction')

        fig.suptitle(f"Random Forest Classifier - Drivers : {', '.join(drivers)}", fontsize=16)
        plt.tight_layout()

        return classifier

    def get_rgs_in_raster_range(self, raster, b_inf, b_sup, sub=None, predicate='intersects'):
        """
        Return the rock glaciers whose raster values are between b_inf and b_sup
        raster : vt.Raster object
        b_inf : float, lower bound
        b_sup : float, upper bound
        sub : geodataframe, optional, subset of the main layer to use
        """
        if sub is None:
            sub = self.main_layer.copy()

        # Process the raster
        _, area_in_raster_range = rt.get_surf_in_raster_range(raster, self.aoi_gdf, b_inf, b_sup, vectorize_result=True)

        # Spatial join between the rock glaciers and the area in raster range
        rgs_in_raster_range = vt.spatial_selection(sub, area_in_raster_range, predicate=predicate)
        return rgs_in_raster_range

    def get_surf_covered_by_rgs_in_aoi(self, aoi, sub=None):
        """
        Return the total surface in squared meters occupied by rock glaciers in the aoi.
        aoi : geodataframe
        """
        if sub is None:
            sub = self.main_layer.copy()
        return vt.get_surf_covered_by_contents_in_container(sub, aoi)

    def get_contours_y1_y2(self, y1, y2, ax=None, figsize=(8, 5), sub=None, levels=[0.5], color='red', cmap='Reds', linewidth=1):
        """
        Isolines of the levels (0.1, 0.5, 0.9) of y2 vs y1
        """
        if sub is None:
            sub = self.main_layer.copy()

        # Remove nan values for y1 and y2
        sub = sub.dropna(subset=[y1, y2])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        aida.show_density_contours(
            data=sub,
            x_col=y1,
            y_col=y2,
            fill=False,
            ax=ax,
            levels=levels,
            color=color,
            cmap=cmap,
            linewidth=linewidth,
        )

        ax.set_xlabel(get_nc(y1))
        ax.set_ylabel(get_nc(y2))
        ax.set_title(f'Density contours of {get_nc(y2)} vs {get_nc(y1)}')

        return ax

    def get_contours_and_scatter(self, y1, y2, hue, levels=[0.5], ax=None, figsize=(8, 5), sub=None, s=2, linewidth=1):
        """
        Scatter and contours of y2 vs y1, colored by hue
        """

        # Data selection
        if sub is None:
            sub = self.main_layer.copy()
        sub = sub.dropna(subset=[y1, y2])

        # Create figure
        if ax is None:
            fig, ax=plt.subplots(figsize=figsize)

        # For each hue value
        for i, v in enumerate(self.gc[f"{hue}_order"]):

            # Data selection    
            sub_v = self.get_quali_request({hue:[v]}, sub=sub)

            # Scatter
            self.get_scatter_plot_hue_y1_y2(y1, y2, s=s, ax=ax, reg_line_width=0, hue=hue, sub=sub_v, show_score=False)

            # Contours
            self.get_contours_y1_y2(y1, y2, levels=levels, cmap=None, color=self.gc[f"{hue}_colors"][v] , ax=ax, sub=sub_v, linewidth=linewidth)

            # Legend
            if i == len(self.gc[f"{hue}_order"])-1:
                ax.legend().remove()

        # Title
        ax.set_title(f'Density Contours {levels}')
        return ax

    def get_pairplot(self, drivers=None, hue=None, height=2, sub=None):
        """
        return a seaborn pairplot of the drivers, colored by hue
        """
        if sub is None:
            sub = self.main_layer.copy()
        sub = sub[(sub[hue] != 'uncertain') & (sub[hue] != 'undefined')]

        if hue is not None:
            sns.pairplot(sub[drivers + [hue]], hue=hue, palette=self.gc[f"{hue}_palette"], height=2, kind='scatter', diag_kind='kde', hue_order=self.gc[f"{hue}_order"])

        else:
            sns.pairplot(sub[drivers], height=2, kind='scatter', diag_kind='kde')

        return None

    def get_histplot_and_vlines_x_aoi(self, raster, ax=None, figsize=(3, 3), xlabel=None, vlines_list=None, **kwargs):
        """
        Draw an histogram of a raster within the rogi aoi 
        raster_no_data : ()
        vlines_list : [(xpos, color, linewidth, label)]
        **kwargs are transferred to sns.histplot()
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Get raster metadata
        pixel_side_length = raster.getPixelSize()[0]
        pixel_ha = pixel_side_length**2 / 10000

        # Get a copy of the target raster
        target_raster = raster.copy()

        # Ensure the already done pre-processings (like no data fill)
        target_raster.array = raster.array

        # Mask the raster on the aoi to get only the pixels of interest
        target_raster.unmask()
        target_raster.maskFromVector(self.aoi_gdf, epsg=2056)
        target_array = target_raster.array[~target_raster.array.mask]

        # Draw the histogram
        sns.histplot(target_array, ax=ax, **kwargs)

        # Change the ytickslabels as hectares
        ax.set_yticklabels(ax.get_yticks()*pixel_ha/1000)
        ax.set_ylabel('surface coverage (thousands of ha)')

        # Add vlines
        if vlines_list is not None:
            [ax.axvline(v[0], color=v[1], linewidth=v[2], label=v[3]) for v in vlines_list]

        # Custom the axis
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        ax.legend()
        return ax

    def get_fa_steepness(
        self,
        slope,
        slope_threshold=30,
        slope_opening_radius=4,
        save_rasters=True,
        out_path=None):
        """
        Identify areas favorable to rock glaciers based on slope raster
        Return a raster with 0 where the slope is greater than the slope_threshold and 1 where steepness is lower
        """
        if out_path is None:
            out_path = self.out_figs_dir

        # Create a copy of the slope geoim
        slope_masked = slope.copy()

        # First, we mask the slope based on the threshold
        slope_masked.maskFromThreshold(slope_threshold, greater=True)

        # Now, binarize that result
        bin_mask = slope_masked.array.mask
        bin_mask = bin_mask.astype(int)
        slope_masked.array = bin_mask

        # Denoise the mask with an morphological opening
        slope_masked_denoised = slope_masked.apply_blur(r=slope_opening_radius)

        if save_rasters:
            slope_masked.save(out_path+f'bin_slope_thresh-{np.round(slope_threshold, 2)}.tif')
            slope_masked_denoised.save(out_path+f'bin_slope_thresh-{np.round(slope_threshold, 2)}_k-{int(slope_opening_radius)}.tif')

        return slope_masked_denoised

    def get_fa_glaciers_pag(self, pixel_size=10):
        """
        Return a binary raster with areas in the SGI 1850 inventory as 0
        """

        # Rasterize the glaciers extent during PAG
        gl_pag_r = vt.rasterize(self.aoi_sgi1850, pixel_size=pixel_size, load_pixels=True, extent=self.aoi_gdf)
        gl_pag_r.array = 1 - gl_pag_r.array

        return gl_pag_r

    def sf(self, title, dpi=200, sub_folder_name=None, bbox_inches='tight', pad_inches=0.2):
        """
        Save the last edited figure in the output figure directory
        """
        figpath = self.out_figs_dir

        if sub_folder_name is not None:
            figpath = Path(figpath, sub_folder_name)

        plt.savefig(Path(figpath, f"{title}_{self.aoi_name}.png"), dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    # def copy(self):
    #     """
    #     Return a copy of the RockGlacierInventory object
    #     """
    #     # Create the RockGlacierInventory instance
    #     self_copy = RockGlacierInventory(
    #         b_layer=self.b_layer.copy(),
    #         aoi=self.aoi_gdf.copy(),
    #         aoi_name=self.aoi_name,
    #         aoi_rasters_res=self.aoi_rasters_res,
    #         **kwargs,
    #     )

    #     load_external_datasets = False
    #     if self.dem is not None:
    #         load_external_datasets = True

    #     # Load external data
    #     self_copy.pre_process_main_layer(load_external_datasets=load_external_datasets)

    #     return self_copy

    # def crop(self, extent_gdf, extent_name):
    #     """
    #     Crop the main layer to the given extent
    #     extent : (minx, miny, maxx, maxy)
    #     """
    #     foo = self.copy()
    #     foo.main_layer = vt.spatial_selection(self.main_layer, extent_gdf)
    #     foo.b_layer = foo.main_layer.copy()
    #     foo.aoi_gdf = extent_gdf
    #     foo.aoi_geom = extent_gdf.geometry.iloc[0]
    #     foo.aoi_name = extent_name

    #     if self.dem is not None:
    #         foo.dem = rt.Open(rt.cropFromVector(self.dem, extent_geom), load_pixels=True, epsg=2056)

    #     if self.hillshade is not None:
    #         foo.hillshade = rt.Open(rt.cropFromVector(self.hillshade, extent_geom), load_pixels=True, epsg=2056)
        
    #     return foo

def get_nc(col):
    try:
        h_label = nc[col]
    except KeyError:
        h_label = col
    return h_label