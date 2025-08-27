from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from telenvi import vector_tools as vt
from datetime import datetime
from pyrogi.rock_glacier_unit import RockGlacierUnit

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RockGlacierInventory:
    """
    Describe a Rock Glacier Inventory dataset.
    Les méthodes s'appliquent à des dataframes au standard rogi_ch_2.0
    pms_layer, ous_layer, our_layer, oue_layer
    """
    
    def __init__(self,
        pms_layer,
        ous_layer,
        rogi_reg_name,
        dem_res = 20,
        dem_reclass_step = 500,
        h_azimuth = 315,
        h_altitude = 315,
        figs_dir = None,
        epsg=2056,
        figs_version_note=None,
        b_layer=None
        ):

        # Define coordinates system
        self.epsg=epsg

        # Default figures outputs dir
        if figs_dir is None:
            figs_dir = f'/home/duvanelt/Desktop/{rogi_reg_name}'
            self.figs_dir = figs_dir
            self.set_figs_dir(figs_dir)

        # Open Primary Markers layer
        pms_layer = pms_layer[pms_layer.pm_type == 'rock_glacier']
        self.pms_layer = pms_layer

        # Open outlines layer
        self.ous_layer = ous_layer
        self.oue_layer = self.ous_layer[self.ous_layer.ou_type == 'extended']
        self.our_layer = self.ous_layer[self.ous_layer.ou_type == 'restricted']

        # Open the layer with the enhanced dataset
        # because it's heavy to compute alti metrics each time, we save the file
        # to be able to re-open it quickly
        self.b_layer = b_layer

        # Create a fig dir
        if figs_dir is not None:
            self.set_figs_dir(figs_dir)
        else:
            self.figs_dir = None

        # Add a note to the version
        if figs_version_note is not None and not figs_version_note.startswith('_'):
            self.figs_version_note = f"_{figs_version_note}"
        elif figs_version_note is not None and figs_version_note.startswith('_'):
            self.figs_version_note = figs_version_note
        else:
            self.figs_version_note = ""

    def get_version(self):
        """
        Define version name for the figures
        """
        now = datetime.now()
        date_hour_str = now.strftime("%Y-%m-%d")
        version = f"rogi_ch_v{date_hour_str.replace('-','').replace('_','')}{self.figs_version_note}"
        return version

    def set_figs_dir(self, figs_dir):
        """
        Define directory to store the figures
        """
        self.figs_dir=Path(figs_dir)
        if not self.figs_dir.exists():
            self.figs_dir.mkdir()

    def copy(self):
        """
        Return a deep copy of the instance
        """
        pms_copy = self.pms_layer.copy(deep=True)
        ous_copy = self.ous_layer.copy(deep=True)
        copy_instance = RockGlacierInventory(pms_copy, ous_copy, figs_dir=self.figs_dir)
        return copy_instance

    def get_raw_map(self, save_fig=False, figsize=(10, 4)):
        """
        Show a map and create a figure with an overview of the rock glaciers
        """
        fig,ax=plt.subplots(figsize=figsize)
        self.pms_layer.plot(markersize=0.4, color='black', ax=ax)
        self.ous_layer.plot(ax=ax)

        fig_name = f'rogimap_{self.get_version()}.png'
        fig_path = Path(self.figs_dir, fig_name)
        ax.set_title(fig_name[:-3])

        if save_fig:
            fig.savefig(fig_path)
        return ax

    def get_alti_metrics(self, dem_map, output_path, output_layername, dem_res_order_1=5, dem_res_order_2=0.5, save_steps=50):
        """
        Compute topographic metrics for the features of a rogi
        dem_res_order_1 and 2 are for different applications (altitude, aspect or slope within the margins)
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
                rg_detailled_serie = rg.get_detailled_serie(dem_map, dem_res_order_1, dem_res_order_2)

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
        return self.get_version()
