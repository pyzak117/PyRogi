import pandas as pd
import numpy as np
from osgeo import gdal
import shapely
from telenvi import raster_tools as rt
from pyrogi.rock_glacier_inventory import get_rogi_from_layers, get_rogi_from_population, RockGlacierInventory
from pyrogi.rock_glacier_unit import RockGlacierUnit
import geopandas as gpd
import unittest
from pathlib import Path
import random
from matplotlib import pyplot as plt

# Define path to tests inputs datasets
test_rogi_operator = 'td'
test_study_root_path = '/home/duvanelt/OneDrive/rodynalps_these-thibaut/rodynalps_collaborative-rogi/rogi_valais'
test_study_rogi_gpkg_path = Path(test_study_root_path, 'data', f'public-data_{test_rogi_operator}', f'Rogi_{test_rogi_operator}.gpkg')
test_study_markers_layername = f'primary-markers_{test_rogi_operator}'
test_study_outlines_layername = f'outlines_{test_rogi_operator}'

# External datasets
test_dems_dir = Path('/media/duvanelt/LaCie/rock-glaciers_valais/data_dems')
test_dems_metamap = gpd.read_file('/home/duvanelt/Desktop/dem_maps.gpkg')
test_crop_area = gpd.read_file('/home/duvanelt/Desktop/crop_area.gpkg').iloc[0].geometry

# Open datatest
test_epsg=2056
test_rogi_pms = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_markers_layername).to_crs(epsg=test_epsg)
test_rogi_ous = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_outlines_layername).to_crs(epsg=test_epsg)


class TestRockGlacierInventory(unittest.TestCase):

    def setUp(self):
        self.test_rogi_from_layers = get_rogi_from_layers(
                primary_markers_layer     = test_study_rogi_gpkg_path,
                outlines_layer            = test_study_rogi_gpkg_path,
                primary_markers_layername = test_study_markers_layername,
                outlines_layername        = test_study_outlines_layername,
                primary_markers_epsg      = test_epsg,
                outlines_epsg             = test_epsg)

    def test_get_rogi_from_layers(self):
        self.assertIsInstance(self.test_rogi_from_layers, RockGlacierInventory)

    def test_get_population_from_layers(self):
        population = self.test_rogi_from_layers.get_population()
        self.assertIsInstance(random.choice(population), RockGlacierUnit)

    def test_show_map(self):
        self.test_rogi_from_layers.show_map()
        plt.show()

    def test_get_rogi_from_population(self):

        # Get a list of Rock Glacier Units from the rogi feature test
        population = self.test_rogi_from_layers.get_population()

        # Extract a sample of this population
        sub_population = population[10:20]

        # Try to create a new rogi from this population
        tested = get_rogi_from_population(sub_population, epsg=test_epsg)

        # Test : tested must be a RockGlacierInventory feature
        self.assertIsInstance(tested, RockGlacierInventory)

        # Test : show the map to see if it changed
        tested.show_map(basemap=True)
        plt.show()

    def test_crop(self):
        cropped_rogi = self.test_rogi_from_layers.crop(test_crop_area)
        cropped_rogi.show_map(basemap=True)
        plt.show()

    def test_get_altitudinal_values(self):
        cropped_rogi = self.test_rogi_from_layers.crop(test_crop_area)
        altis = cropped_rogi.get_altitudinal_values(
            dem_source=test_dems_metamap, nRes=5)
        self.assertIsInstance(altis, np.ndarray)
        
if __name__ == '__main__':
    unittest.main()