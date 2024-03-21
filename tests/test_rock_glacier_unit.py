import pandas as pd
import numpy as np
from osgeo import gdal
import shapely
from telenvi import raster_tools as rt
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
test_working_id_with_outlines = 'mont-fort-3601'

# External datasets
test_dems_dir = Path('/media/duvanelt/LaCie/rock-glaciers_valais/data_dems')
test_dems_metamap = gpd.read_file('/home/duvanelt/Desktop/dem_maps.gpkg')

# Open datatest
test_rogi_pms = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_markers_layername).to_crs(epsg=2056)
test_rogi_ous = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_outlines_layername).to_crs(epsg=2056)

# Extract a specific feature with outlines available
test_rgu_feature = test_rogi_pms[test_rogi_pms['WorkingID']==test_working_id_with_outlines].iloc[0]

class TestRockGlacierUnit(unittest.TestCase):
    def setUp(self):
        self.rgu_feature = RockGlacierUnit(test_rgu_feature)

    def test_attributes_creation(self):
        self.assertTrue('RG' in self.rgu_feature.rgik_pm_primaryid)

    def test_representative_geom_is_polygon(self):
        self.assertTrue('polygon' in self.rgu_feature.get_most_representative_polygon().geom_type.lower())

    def test_dem_initialization(self):

        # Dem initialization
        self.rgu_feature.initialize_dem(dem_source = test_dems_metamap, nRes=10)

        # Test : rgu_dem attribute should now be a gdal Dataset 
        self.assertIsInstance(self.rgu_feature.rgu_dem, gdal.Dataset)

    def test_dem_loading(self):

        # Dem initialization
        self.rgu_feature.initialize_dem(test_dems_metamap, nRes=10)

        # Dem loading
        self.rgu_feature.get_dem()

        # Test : the rgu_dem attribute should now be a Geoim
        self.assertIsInstance(self.rgu_feature.rgu_dem, rt.geoim.Geoim)

        # Display it
        plt.title(f"{self.rgu_feature.rgik_pm_workingid} DEM")
        self.rgu_feature.rgu_dem.show()

    def test_slope_loading(self):

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dems_metamap, nRes=10)        
        self.rgu_feature.get_dem()

        # Slope loading
        self.rgu_feature.get_slope()

        # Test : the rgu_slope attribute should new be a Geoim
        self.assertIsInstance(self.rgu_feature.rgu_slope, rt.geoim.Geoim)

        # Display it
        plt.title(f"{self.rgu_feature.rgik_pm_workingid} Slope")
        self.rgu_feature.rgu_slope.show()

    def test_extended_outline(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Test : rgu_oue_geom attribute should now be a polygon
        self.assertTrue('polygon' in self.rgu_feature.rgu_oue_geom.geom_type.lower())

    def test_restricted_outline(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Test : rgu_oue_geom attribute should now be a polygon
        self.assertTrue('polygon' in self.rgu_feature.rgu_our_geom.geom_type.lower())

    def test_show_map(self):
        self.rgu_feature.load_outlines(test_rogi_ous)
        self.rgu_feature.show_map()
        plt.show()

    def test_show_profiles(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dems_metamap)        
        self.rgu_feature.get_dem()

        self.rgu_feature.show_topo_profile('major', window_size=2)
        plt.show()

    def test_show_pannel(self):
        plt.style.use('dark_background')
        plt.rc('font', family='monospace', size=8)
        
        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dems_metamap)        
        self.rgu_feature.get_dem()

        # Test : show the RGU pannel
        self.rgu_feature.show_pannel(window_size=50, basemap=True)
        plt.show()

    def test_get_points_cloud(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dems_metamap)
        self.rgu_feature.get_dem()

        # Extract pointscloud
        points = self.rgu_feature.get_points_cloud(z_factor=2)

        # Test : pointcloud is a numpy array
        self.assertIsInstance(points, np.ndarray)

    def test_show_points_cloud(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dems_metamap)
        self.rgu_feature.get_dem()

        # Test : display GUI console to show the rock glacier
        self.rgu_feature.show_points_cloud(z_factor=3)

    def test_get_altitudinal_range(self):
        
        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dems_metamap)
        self.rgu_feature.get_dem()

        # Get the lowest and higher point in the rock glacier
        lowest_point, higher_point = self.rgu_feature.get_altitudinal_range()
        
        # Test : the points must be greater than 1000 (just avoid nodata)
        self.assertGreater(lowest_point, 500)
        self.assertGreater(higher_point, 500)

    def test_get_series_without_outlines(self):

        # Test the creation of a pandas.serie from the instance attributes
        # We should get a list with [pd.Series, None, None]
        pm_row, oue_row, our_row = self.rgu_feature.get_series()
        self.assertIsInstance(pm_row, pd.Series)
        self.assertIsNone(oue_row)
        self.assertIsNone(our_row)
        print(pm_row)

    def test_get_series_with_outlines(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Get the relatives series
        pm_row, oue_row, our_row = self.rgu_feature.get_series()

        # Test type of each
        self.assertIsInstance(pm_row, pd.Series)
        self.assertIsInstance(oue_row, pd.Series)
        self.assertIsInstance(our_row, pd.Series)
        print(our_row)

    def test_get_topographic_values(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dems_metamap)        
        self.rgu_feature.get_dem()

        # Test : get topographic values
        topo_values = self.rgu_feature.get_topographic_values(test_dems_metamap)
        # Test if the dernier and avant dernier elements of the topo_values list
        # which corresponds to alt_max and alt_min are coherents
        self.assertGreater(topo_values[-1], topo_values[-2])

        # Test if the attribute rgu_dem is well deleted from memory
        with self.assertRaises(Exception):
            print(self.rgu_feature.rgu_dem.array[0])

if __name__ == '__main__':
    unittest.main()
