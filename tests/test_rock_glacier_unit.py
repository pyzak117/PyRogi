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
test_working_id_with_outlines = 'mont-fort-152'

# External datasets
test_dems_dir = Path('/media/duvanelt/LaCie/rock-glaciers_valais/data_dems')
test_dems_metamap = gpd.read_file('/home/duvanelt/Desktop/dem_maps.gpkg')

# Open datatest
test_rogi_pms = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_markers_layername).to_crs(epsg=2056)
test_rogi_ous = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_outlines_layername).to_crs(epsg=2056)

class TestRockGlacierUnit(unittest.TestCase):
    def setUp(self):

        # Simulating the transformation of GeoSeries to valid attribute names in RockGlacierUnit initialization
        self.rgu_pm = RockGlacierUnit(test_rogi_pms.iloc[random.randint(0, len(test_rogi_pms))])

        known_feature_with_outlines = test_rogi_pms[test_rogi_pms['WorkingID']==test_working_id_with_outlines].iloc[0]
        self.rgu_pm_with_outlines = RockGlacierUnit(known_feature_with_outlines)

    def test_attributes_creation(self):
        self.assertTrue('RG' in self.rgu_pm.rgik_pm_primaryid)

    def test_representative_geom_is_polygon(self):
        self.assertTrue('polygon' in self.rgu_pm.get_most_representative_geometry().geom_type.lower())

    def test_dem_initialisation_from_geopackage(self):
        self.assertIsInstance(self.rgu_pm.initialize_dem(dem_source = test_dems_metamap, nRes=10), gdal.Dataset)

    def test_dem_loading(self):
        self.rgu_pm.initialize_dem(test_dems_metamap, nRes=10)        
        self.assertIsInstance(self.rgu_pm.load_dem(), rt.geoim.Geoim)
        plt.title(f"{self.rgu_pm.rgik_pm_workingid} DEM")
        self.rgu_pm.rgu_dem.show()

    def test_slope_loading(self):
        self.rgu_pm.initialize_dem(test_dems_metamap, nRes=10)        
        self.rgu_pm.load_dem()
        self.assertIsInstance(self.rgu_pm.load_slope(), rt.geoim.Geoim)
        plt.title(f"{self.rgu_pm.rgik_pm_workingid} Slope")
        self.rgu_pm.rgu_slope.show()

    def test_extended_outlines_loading(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.assertTrue('polygon' in self.rgu_pm_with_outlines.rgu_oue_geom.geom_type.lower())

    def test_restricted_outlines_loading(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.assertTrue('polygon' in self.rgu_pm_with_outlines.rgu_our_geom.geom_type.lower())

    def test_minor_axis(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.assertTrue('line' in self.rgu_pm_with_outlines.rgu_minor_axis.geom_type.lower())

    def test_major_axis(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.assertTrue('line' in self.rgu_pm_with_outlines.rgu_major_axis.geom_type.lower())

    def test_major_topo_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_major_topo_profile()
        self.assertIsInstance(self.rgu_pm_with_outlines.major_topo_profile, np.ndarray)

    def test_major_slope_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_slope()
        self.rgu_pm_with_outlines.load_major_slope_profile()
        self.assertIsInstance(self.rgu_pm_with_outlines.major_slope_profile, np.ndarray)

    def test_minor_topo_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_minor_topo_profile()
        self.assertIsInstance(self.rgu_pm_with_outlines.minor_topo_profile, np.ndarray)

    def test_minor_slope_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_slope()
        self.rgu_pm_with_outlines.load_minor_slope_profile()
        self.assertIsInstance(self.rgu_pm_with_outlines.minor_slope_profile, np.ndarray)

    def test_show_map(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.show_map(basemap=True)
        plt.show()

    def test_show_major_topo_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_minor_topo_profile()
        self.rgu_pm_with_outlines.load_major_topo_profile()
        self.rgu_pm_with_outlines.show_topo_profiles(mode='major')
        plt.title(f"{self.rgu_pm_with_outlines.rgik_pm_workingid} Major Axis Topo")
        plt.show()

    def test_show_minor_topo_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_minor_topo_profile()
        self.rgu_pm_with_outlines.load_minor_topo_profile()
        self.rgu_pm_with_outlines.show_topo_profiles(mode='minor')
        plt.title(f"{self.rgu_pm_with_outlines.rgik_pm_workingid} minor Axis Topo")
        plt.show()

    def test_show_major_slope_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_slope()
        self.rgu_pm_with_outlines.load_minor_topo_profile()
        self.rgu_pm_with_outlines.load_major_topo_profile()
        self.rgu_pm_with_outlines.load_minor_slope_profile()
        self.rgu_pm_with_outlines.load_major_slope_profile()
        self.rgu_pm_with_outlines.show_slope_profiles(mode='major')
        plt.title(f"{self.rgu_pm_with_outlines.rgik_pm_workingid} Major Axis Slope")
        plt.show()

    def test_show_minor_slope_profile(self):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_slope()
        self.rgu_pm_with_outlines.load_minor_topo_profile()
        self.rgu_pm_with_outlines.load_major_topo_profile()
        self.rgu_pm_with_outlines.load_minor_slope_profile()
        self.rgu_pm_with_outlines.load_minor_slope_profile()
        self.rgu_pm_with_outlines.show_slope_profiles(mode='minor')
        plt.title(f"{self.rgu_pm_with_outlines.rgik_pm_workingid} minor Axis Slope")
        plt.show()


    def test_show_pannel(self, basemap=True):
        self.rgu_pm_with_outlines.load_outlines(test_rogi_ous)
        self.rgu_pm_with_outlines.load_geo_axes()
        self.rgu_pm_with_outlines.initialize_dem(test_dems_metamap, nRes=2) 
        self.rgu_pm_with_outlines.load_dem()
        self.rgu_pm_with_outlines.load_slope()
        self.rgu_pm_with_outlines.load_minor_topo_profile()
        self.rgu_pm_with_outlines.load_major_topo_profile()
        self.rgu_pm_with_outlines.show_pannel(basemap=True)
        plt.show()

if __name__ == '__main__':
    unittest.main()
