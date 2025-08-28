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

warnings.filterwarnings("ignore", category=UserWarning)

TEMP_FIGS_DIR = '/home/duvanelt/Desktop/'
DATAMAP_PATH = '/home/duvanelt/datamap.json'

class RockGlacierInventory:
    """
    Describe a Rock Glacier Inventory dataset.
    Les méthodes s'appliquent à des dataframes au standard rogi_ch_2.0
    pms_layer, ous_layer, our_layer, oue_layer

    Datamap : json file with all the paths
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
        datamap = None,

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

            'cardi_pt_order': rgc.cardi_pt_order,
            'rgu_activity_class_order': rgc.rgu_activity_class_order,
            'rgu_activity_class_wu_order': rgc.rgu_activity_class_wu_order,
            'rgu_kin_att_order': rgc.rgu_kin_att_order,

            'cardi_pt_palette': rgc.cardi_pt_palette,
            'rgu_activity_class_palette': rgc.rgu_activity_class_palette,
            'rgu_activity_class_wu_palette': rgc.rgu_activity_class_wu_palette,
            'rgu_kin_att_palette': rgc.rgu_kin_att_palette,

            'cardi_pt_cmap': rgc.cardi_pt_cmap,
            'rgu_activity_class_cmap': rgc.rgu_activity_class_cmap,
            'rgu_activity_class_wu_cmap': rgc.rgu_activity_class_wu_cmap,
            'rgu_kin_att_cmap': rgc.rgu_kin_att_cmap,
        }

    def load_external_datasets_from_datamap_on_thib_pc(
        self,
        datamap = None,
        aoi_dem_res = 20,
        aoi_dem_reclass_step = 500,
        aoi_hillshade_azimut = 315,
        aoi_hillshade_altitude = 45,
        ):
        """
        Load DEMS, glaciers, lakes... from a json file called datamap. Specific for TD. 
        """

        if datamap is None:
            with open(DATAMAP_PATH) as f:
                datamap = json.load(f)[0]

        # DEM
        self.alti_3d_path = datamap['sa3d_dir']
        self.derived_sa3d = Path(self.alti_3d_path, 'derived_products')
        self.dem = rt.Open(Path(self.derived_sa3d, f'dem_{aoi_dem_res}_merged.tif'), load_pixels=True, geoExtent=self.aoi_geom)

        # Reclassify the DEM each 500m
        self.dem_reclass = aida.get_manual_clusters(self.dem, np.arange(0, 4500, aoi_dem_reclass_step))

        # Hillshade
        self.hillshade = rt.Open(Path(self.derived_sa3d, f'hillshade_{aoi_dem_res}_{aoi_hillshade_azimut}_{aoi_hillshade_altitude}.tif'), load_pixels=True,  geoExtent=self.aoi_geom)

        # PISR
        self.dir_ins = rt.Open(Path(self.derived_sa3d, f'pisr_{aoi_dem_res}/mean_direct-insolation_46_Kjm_with-svf_0715-0815-0915-1015.tif'),  geoExtent=self.aoi_geom, load_pixels=True)

        # Glaciers
        sgi16 = gpd.read_file(datamap['glaciers_inv'], layer='sgi_2016_glaciers')

        # Places and waterways from OSM
        osm_path = datamap['osm_dir']
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
        self.main_layer = self.main_layer.dropna(subset=['dir_ins'])
        n_after = len(self.main_layer)
        print(f"{round((n_before - n_after) / n_before * 100, 2)}% of rock glaciers have nodata for dir_ins")

    def pre_process_main_layer(self):
        """
        Make simple and redundant pre-processings on the self.main_layer
        This method can be used to quick & dirty patches
        """

        # Load external datasets
        self.load_external_datasets_from_datamap_on_thib_pc()

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

        # Load dir_ins raster
        self.get_median_values_in_geoim(self.dir_ins, 'dir_ins')

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

    def get_alti_metrics(self, dem_map, output_path, output_layername, aoi_dem_res_order_1=5, aoi_dem_res_order_2=0.5, save_steps=50):
        """
        Compute topographic metrics for the features of a rogi
        aoi_dem_res_order_1 and 2 are for different applications (altitude, aspect or slope within the margins)
        """

        # Initialisation des variables
        b_series=[]
        unbound_local_error=[]
        no_outlines_error = []

        # Pour chaque glacier rocheux
        for i, target_rg in enumerate(self.pms_layer.iloc):
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
            if i%save_steps == 0:
                self.make_b_layer(b_series)
                self.save_b_layer(output_path, output_layername)

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

    def get_basic_map(
        self,
        ax=None,
        mask_alpha=0.8,
        map_figsize = (15, 15),
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
        show_places=True
        ):
        """
        Make a basic map which be used in this study
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=map_figsize)

        # Add aerial image
        if aerial_view:
            vt.add_wmts_layer(self.aoi_gdf, mask_outside_geo_target=True, epsg=2056, ax=ax, mask_color='white', mask_alpha=mask_alpha)

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
        self.aoi_sgi16.plot(ax=ax, facecolor='none', edgecolor=glaciers_color, linewidth=0.75)

        return ax

    def get_rgs_map(
        self,
        ax=None,
        mask_alpha=0.8,
        map_figsize = (15, 15),
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
        y_column='rgu_activity_class_wu'
    ):

        ax = self.get_basic_map(
            ax=ax,
            mask_alpha=mask_alpha,
            map_figsize = map_figsize,
            glaciers_color=glaciers_color,
            water_color=water_color,
            water_alpha=water_alpha,
            glaciers_alpha=glaciers_alpha,
            places_labels_fontsize=places_labels_fontsize,
            places_label_alpha=places_label_alpha,
            dem_alpha=dem_alpha,
            dem_cmap=dem_cmap,
            hillshade_alpha=hillshade_alpha,
            aerial_view=aerial_view,
            show_places=show_places
        )

        # Remove nan values for the target column
        sub = self.main_layer.dropna(subset=[y_column])

        # Explicitly affect a color to each feature from the column acti cl without uncertainty
        sub['temp_color'] = sub[y_column].map(self.gc[f"{y_column}_colors"])

        # Remove nan values for the color column 
        sub = sub.dropna(subset=['temp_color'])

        # Add rock glaciers footprints
        sub.plot(ax=ax, facecolor=sub['temp_color'], alpha=0.8)
        sub.plot(ax=ax, facecolor='none', edgecolor=sub['temp_color'], linewidth=0.75)

        # Add a legend to the map
        legend_elements = [
            Patch(facecolor=color, label=label.capitalize())
            for label, color in self.gc[f'{y_column}_colors'].items()]
        ax.legend(handles=legend_elements, title='').set_zorder(100)

        # Title
        ax.set_title(f'Rock Glaciers {self.aoi_name} - ({len(sub)} units)', fontsize=12)
        return ax
