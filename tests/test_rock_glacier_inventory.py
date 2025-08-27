from load import *

# Directory with the rogi
test_study_rogi_gpkg_path = Path(datamap['rogi_ch_data_current_2.0'])

# Layernames
test_study_markers_layername = datamap['rogi_ch_data_current_2.0_pms_layername']
test_study_outlines_layername = datamap['rogi_ch_data_current_2.0_ous_layername']

# External datasets
test_dem_srtm = Path(datamap['srtm_dem_path'])
test_sa3d_dir = Path(datamap['sa3d_dir'])
test_dems_metamap_path = datamap['sa3d_map']
test_crop_zone_path = Path(Path(__file__).parent, '_test_data', 'cropping.gpkg')

# test_dems_metamap = gpd.read_file(test_dems_metamap)

# Load datasets
test_epsg=2056
test_rogi_pms = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_markers_layername)
test_rogi_ous = gpd.read_file(test_study_rogi_gpkg_path, layer=test_study_outlines_layername)
test_crop_zone = gpd.read_file(test_crop_zone_path)

# Selections
test_rogi_pms = test_rogi_pms[test_rogi_pms.pm_type=='rock_glacier']
test_rogi_pms = vt.spatial_selection(test_rogi_pms, test_crop_zone)
test_rogi_ous = vt.spatial_selection(test_rogi_ous, test_crop_zone)

test_version_note = 'saucisse'

class TestRockGlacierInventory(unittest.TestCase):

    def setUp(self):
        self.test_rogi = RockGlacierInventory(
                pms_layer = test_rogi_pms,
                ous_layer = test_rogi_ous,
                epsg      = test_epsg,
                version_note=test_version_note,
                fig_dir = f"{Path(__file__).with_name('_test_fig_dir')}")

    def test_version(self):
        print(self.test_rogi.get_version())
    
    def test_copy(self):
        old = self.test_rogi
        new = old.copy()
        new.pms_layer['pm_pid'] = 'new_' + new.pms_layer.pm_pid
        self.assertTrue(new.pms_layer.iloc[0].pm_pid != old.pms_layer.iloc[0].pm_pid)

    def test_show_map(self):
        # Draw a map
        self.test_rogi.show_map(save_fig=True)

        # Check if it has been correctly generated
        self.assertTrue(Path(self.test_rogi.fig_dir, f"rogimap_{self.test_rogi.get_version()}.png").exists())

    def test_get_alti_metrics(self):
        """
        Test the method to merge a DEM folder and the rogi
        """
        pass

        # Read the DEMS
        # Compute the metrics
        # Write the enhanced dataset
        # Check if the file exists
        # And if the columns regarding elevation are there

        self.test_rogi.get_alti_metrics(
            dem_map = test_dems_metamap_path,
            dem_res_order_1=5, 
            dem_res_order_2=0.5,
        )

        self.assertIsInstance(self.test_rogi.enhanced_layer, gpd.GeoDataFrame)

if __name__ == '__main__':
    unittest.main()