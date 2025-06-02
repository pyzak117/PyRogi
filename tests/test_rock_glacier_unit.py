#%%
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

# Directory with the rogi
test_study_root_path = '/media/duvanelt/Data_Part1/rogi_switzerland_unil_unifr_rodynalps_phd_thib/garage/rogi_ch_2.0/'
test_study_rogi_gpkg_path = Path(test_study_root_path, 'rogi_ch_2.0.gpkg')

# Layernames
test_study_markers_layername = f'rogi_ch_pms_cur'
test_study_outlines_layername = f'rogi_ch_ous_cur'

# Rock glacier target Working ID
test_working_id_with_outlines = 'RGU460592N073398E'

# Open datatest
test_rogi_pms = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_markers_layername).to_crs(epsg=2056)
test_rogi_ous = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_outlines_layername).to_crs(epsg=2056)

# External datasets
test_dem_srtm = Path('/media/duvanelt/Data_Part1/geodata_switzerland_insar_glaciers_dem-srtm/srtm_dems/SRTM')

# test_dems_dir = Path('/media/duvanelt/LaCie/geodata_raster_dems_insars_aerial/raster_tiles_dems_swiss-surface3d-ss3d')
# test_dem_metamap = gpd.read_file(Path(test_dems_dir, 'ss3d-extent-map.gpkg'))

test_insar_data_dir = '/media/duvanelt/Data_Part1/geodata_switzerland_insar_glaciers_dem-srtm/gamma-rs_insars/switzerland_insar_gamma_sentinel-1_heavy'
test_moving_areas_path = '/media/duvanelt/Data_Part1/rogi_switzerland_unil_unifr_rodynalps_phd_thib/rogi_ch/rogi_ch_standalone_starterpack/rogi_ch_data_current/mas.gpkg'
test_moving_areas_layer = gpd.read_file(test_moving_areas_path)

# Insar and Insar scanning algorithm settings
test_orbit = 'd'
test_interval = 24
test_r = 5
test_c = 100
test_n_classes = 2
test_class_to_avoid = 2

# Extract a specific feature with outlines available
test_rgu_feature = test_rogi_pms[test_rogi_pms['pm_pid']==test_working_id_with_outlines].iloc[0]

class TestRockGlacierUnit(unittest.TestCase):
    def setUp(self):
        self.rgu_feature = RockGlacierUnit(test_rgu_feature)

    def test_attributes_creation(self):
        self.assertTrue('RG' in self.rgu_feature.pm_pid)

    def test_representative_geom_is_polygon(self):
        self.assertTrue('polygon' in self.rgu_feature.get_most_representative_polygon().geom_type.lower())

    def test_dem_initialization(self):

        # Dem initialization
        self.rgu_feature.initialize_dem(dem_source = test_dem_srtm, nRes=10)

        # Test : rgu_dem attribute should now be a gdal Dataset 
        self.assertIsInstance(self.rgu_feature.rgu_dem, gdal.Dataset)

    def test_dem_loading(self):

        # Dem initialization
        self.rgu_feature.initialize_dem(test_dem_srtm, nRes=10)

        # Dem loading
        self.rgu_feature.get_dem()

        # Test : the rgu_dem attribute should now be a Geoim
        self.assertIsInstance(self.rgu_feature.rgu_dem, rt.geoim.Geoim)

        # Display it
        plt.title(f"{self.rgu_feature.pm_pid} DEM")
        self.rgu_feature.rgu_dem.show()

    def test_slope_loading(self):

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dem_srtm, nRes=10)        
        self.rgu_feature.get_dem()

        # Slope loading
        self.rgu_feature.get_slope()

        # Test : the rgu_slope attribute should new be a Geoim
        self.assertIsInstance(self.rgu_feature.rgu_slope, rt.geoim.Geoim)

        # Display it
        plt.title(f"{self.rgu_feature.pm_pid} Slope")
        self.rgu_feature.rgu_slope.show()

    def test_aspect_loading(self):

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dem_srtm, nRes=10)        
        self.rgu_feature.get_dem()

        # aspect loading
        self.rgu_feature.get_aspect()

        # Test : the rgu_aspect attribute should new be a Geoim
        self.assertIsInstance(self.rgu_feature.rgu_aspect, rt.geoim.Geoim)

        # Display it
        plt.title(f"{self.rgu_feature.pm_pid} aspect")
        self.rgu_feature.rgu_aspect.show()

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
        self.rgu_feature.initialize_dem(test_dem_srtm)        
        self.rgu_feature.get_dem()

        self.rgu_feature.show_topo_profile('major', window_size=2)
        plt.show()

    def test_show_pannel(self):
        plt.style.use('dark_background')
        plt.rc('font', family='monospace', size=8)
        
        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dem_srtm)        
        self.rgu_feature.get_dem()

        # Test : show the RGU pannel
        self.rgu_feature.show_pannel(window_size=50, basemap=False)
        plt.show()

    def test_get_points_cloud(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dem_srtm)
        self.rgu_feature.get_dem()

        # Extract pointscloud
        points = self.rgu_feature.get_points_cloud(z_factor=2)

        # Test : pointcloud is a numpy array
        self.assertIsInstance(points, np.ndarray)

    def test_show_points_cloud(self):

        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dem_srtm)
        self.rgu_feature.get_dem()

        # Test : display GUI console to show the rock glacier
        self.rgu_feature.show_points_cloud(z_factor=3)

    def test_get_alti_ranges(self):
        
        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dem_srtm)
        self.rgu_feature.get_dem()

        # Get the lowest and higher point in the rock glacier
        lowest_point, higher_point = self.rgu_feature.get_alti_ranges()
        
        # Test : the points must be greater than 500 (just avoid nodata)
        self.assertGreater(lowest_point, 500)
        self.assertGreater(higher_point, 500)

    def test_get_alti_med_mean(self):
        
        # Insert outlines into the RGU feature
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Initialize & load the dem
        self.rgu_feature.initialize_dem(test_dem_srtm)
        self.rgu_feature.get_dem()

        # Get the lowest and higher point in the rock glacier
        alti_med, alti_mean = self.rgu_feature.get_alti_med_mean()
        
        # Test : the points must be greater than 1000 (just avoid nodata)
        self.assertGreater(alti_med, 500)
        self.assertGreater(alti_mean, 500)

    def test_get_series_without_outlines(self):

        # Test the creation of a pandas.serie from the instance attributes
        # We should get a list with [pd.Series, None, None]
        pm_row, oue_row, our_row = self.rgu_feature.get_series()
        self.assertIsInstance(pm_row, pd.Series)
        self.assertIsNone(oue_row)
        self.assertIsNone(our_row)
        print(pm_row)

    def test_load_manual_moving_areas(self):

        # Load the outlines
        self.rgu_feature.load_outlines(test_rogi_ous)

        # Load the moving areas
        self.rgu_feature.load_manual_moving_areas(test_moving_areas_layer)

        # Check if it's a geodataframe
        self.assertIsInstance(self.rgu_feature.rgu_our_mas, gpd.GeoDataFrame)
        self.assertIsInstance(self.rgu_feature.rgu_oue_mas, gpd.GeoDataFrame)

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

#    def test_get_moving_areas_one_interval(self):
#
#        # Insert outlines into the RGU feature
#        self.rgu_feature.load_outlines(test_rogi_ous)
#
#        # Call the insar detection algorithm
#        kinematics_data = self.rgu_feature.get_moving_areas_one_interval(
#            test_insar_data_dir,
#            orbit = test_orbit,
#            interval = test_interval,
#            r = test_r,
#            c = test_c,
#            n_classes = test_n_classes, 
#            class_to_avoid = test_class_to_avoid,
#            show_steps_charts=True
#        )
#
#        # Test the dictionnary result contain a vector layer
#        self.assertIsInstance(kinematics_data, gpd.GeoDataFrame)
#        output_file_path = Path(f'/home/duvanelt/Desktop/{test_working_id_with_outlines}.gpkg')
#        output_layername = f"{test_working_id_with_outlines}_{test_orbit}_{test_interval}"
#        kinematics_data.to_file(output_file_path, layername=output_layername)

if __name__ == '__main__':
    unittest.main()

# %%
