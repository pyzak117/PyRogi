import pandas as pd
import numpy as np
from osgeo import gdal
import shapely
from telenvi import raster_tools as rt
from pyrogi.rock_glacier_inventory import RockGlacierInventory
from pyrogi.rock_glacier_unit import RockGlacierUnit
import geopandas as gpd
import unittest
from pathlib import Path
import random
from matplotlib import pyplot as plt

# Define path to tests inputs datasets
test_rogi_operator = 'td'

# Directory with the rogi
test_study_root_path = '/media/duvanelt/Data_Part1/rogi_switzerland_unil_unifr_rodynalps_phd_thib/garage/rogi_ch_2.0/'
test_study_rogi_gpkg_path = Path(test_study_root_path, 'rogi_ch_2.0.gpkg')

# Layernames
test_study_markers_layername = f'rogi_ch_pms_cur'
test_study_outlines_layername = f'rogi_ch_ous_cur'

# External datasets
test_dem_srtm = Path('/media/duvanelt/Data_Part1/geodata_switzerland_insar_glaciers_dem-srtm/srtm_dems/SRTM')

# test_dems_dir = Path('/media/duvanelt/LaCie/geodata_raster_dems_insars_aerial/raster_tiles_dems_swiss-surface3d-ss3d')
# test_dems_metamap = gpd.read_file(Path(test_dems_dir, 'ss3d-extent-map.gpkg'))
test_crop_area = gpd.read_file('/home/duvanelt/PyRogi/tests/test_data/cropping.gpkg').iloc[0].geometry

# Open datatest
test_epsg=2056
test_rogi_pms = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_markers_layername).to_crs(epsg=test_epsg)
test_rogi_ous = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_outlines_layername).to_crs(epsg=test_epsg)
test_version_note = 'saucisse'

class TestRockGlacierInventory(unittest.TestCase):

    def setUp(self):
        self.test_rogi = RockGlacierInventory(
                pms_layer = test_study_rogi_gpkg_path,
                ous_layer = test_study_rogi_gpkg_path,
                epsg      = test_epsg,
                version_note=test_version_note)

    def test_get_rogi_from_layers(self):
        self.assertIsInstance(self.test_rogi, RockGlacierInventory)

    def test_make_figdir(self):
        self.assertTrue(self.test_rogi.fig_dir.exists())

    def test_show_map(self):
        self.test_rogi.show_map(save_fig=True)
        self.assertTrue(Path(self.test_rogi.fig_dir, f'rogimap_{self.test_rogi.version}.png').exists())

    def test_version(self):
        print(self.test_rogi.version)

if __name__ == '__main__':
    unittest.main()