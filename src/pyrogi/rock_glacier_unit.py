from osgeo import gdal
import numbers
from typing import List
from matplotlib import gridspec
import contextily as cx
from matplotlib import pyplot as plt
import shapely
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
import numpy as np
import numpy.ma as ma
from telenvi import raster_tools as rt
from telenvi import vector_tools as vt
from telenvi import swiss_geo_downloader as sgd
import pyvista as pv
from tqdm import tqdm

# import pyinsar
import warnings

# Supprimer tous les avertissements
warnings.filterwarnings("ignore")
   
def identify_feature_type(rogi_ch_feature):
    """
    Check if a rogi_ch_feature is a Primary Marker, an extended outline or a restricted outline
    """

    if 'polygon' in rogi_ch_feature.geometry.geom_type.lower():
        if 'extended' in rogi_ch_feature.ou_type:
            ft_type = 'oue'
        if 'restricted' in rogi_ch_feature.ou_type:
            ft_type = 'our'
    elif 'point' in rogi_ch_feature.geometry.geom_type.lower():
        ft_type = 'pm'
    return ft_type

class RockGlacierUnit:

    """
    Describe a Rock Glacier. 
    This conceptual class link all the differents
    objects described in the ROGI CH Model.
    """

    def __init__(self, rogi_ch_feature): 

        """
        Initializes a RockGlacierUnit object, dynamically creating an attribute
        for each column in the rogi_ch geopackage
        """

        # Transform geopackage columns into attributes
        self._set_attributes_from_rogi_ch_feature(rogi_ch_feature)
        
        # Rename some of the attributes
        self.pm_geom = self.pm_geom
        self.rgu_oue_geom  = None
        self.rgu_our_geom  = None
        self.rgu_margins_geom = None
        self.pm_color  = '#CC2936'
        self.rgu_oue_color = '#0b1d51'
        self.rgu_our_color = '#10df7e'
        self.rgu_dem       = None
        self.rgu_slope     = None
        self.rgu_aspect    = None
        self.rgu_cardinals = None
        self.rgu_geo_major = None
        self.rgu_geo_minor = None
        self.rgu_major_axis_color = 'red'
        self.rgu_minor_axis_color = 'orange' 
        self.rgu_epsg      = 2056

    def _set_attributes_from_rogi_ch_feature(self, rogi_ch_feature):
        """
        Dynamically set attributes based on the GeoSeries feature's columns.
        Works with a geodataframe structured as the rogi_ch data model
        """
        # Identify the type of the feature
        ft_type = identify_feature_type(rogi_ch_feature)
        if ft_type is None:
            return None

        # Set one attribute per column in the feature
        for column_name, value in tqdm(rogi_ch_feature.items()):

            # For the primary markers            
            if ft_type == 'pm':

                # Geometry column
                if column_name == 'geometry':
                    if not value.is_valid:
                        value = value.buffer(0)
                    clean_att_name = 'pm_geom'
                
                # Others columns : we keep as it is
                else:
                    clean_att_name = column_name
            
            # For the outlines
            if 'ou' in ft_type:
                if column_name == 'geometry':
                    clean_att_name = f"rgu_{ft_type}_geom"
                    if not value.is_valid:
                        print('geom patch')
                        value = value.buffer(0)
                else:
                    clean_att_name = f"rgu_{ft_type}_{column_name[3:]}"

            # Set the attributes
            setattr(self, clean_att_name, value)

        return None

    def get_pm_relatives_attributes_names(self):
        return [rgu_attr for rgu_attr in dir(self) if 'pm_' in rgu_attr and not 'get' in rgu_attr]

    def get_oue_rel_atives_attributes_names(self):
        return [rgu_attr for rgu_attr in dir(self) if 'oue_' in rgu_attr and not 'get' in rgu_attr]

    def get_our_rel_atives_attributes_names(self):
        return [rgu_attr for rgu_attr in dir(self) if 'our_' in rgu_attr and not 'get' in rgu_attr]

    def smashed_in_serie(self):
        """
        Send a pandas.Series object. with all the attributes. 
        The column 'geometry' is set from the extended outlines
        """

        # Check availibility of the outlines
        assert self.extended_outlines_loaded(), 'nothing to smash because no outlines'

        # Create empty serie
        rgu_row = pd.Series(dtype='object')

        # Add Primary Markers attributes
        for rgu_attr_name in self.get_pm_relatives_attributes_names():
            rgu_row[rgu_attr_name] = getattr(self, rgu_attr_name)
        
        # Add restricted outlines attributes
        for rgu_attr_name in self.get_our_rel_atives_attributes_names():
            rgu_row[rgu_attr_name] = getattr(self, rgu_attr_name) 
        
        # Add extended outlines attributes
        for rgu_attr_name in self.get_oue_rel_atives_attributes_names():
            rgu_row[rgu_attr_name] = getattr(self, rgu_attr_name)

            # Add explicit geometry column
            rgu_row["geometry"] = self.rgu_oue_geom

        return rgu_row

    def get_series(self):
        """
        Send a list of pandas.Series objects. 
        At least you have one row for the pm_relatives attributes. 
        And if the outlines are loaded, you have 3 rows, for pm, oue and our.
        """

        # Create empty pandas rows or series
        pm_row, oue_row, our_row = (pd.Series(dtype='object'), None, None)

        # Initialize columns and values for each columns from the instance's attributes
        for rgu_attr_name in self.get_pm_relatives_attributes_names():
            pm_row[rgu_attr_name] = getattr(self, rgu_attr_name)

        # Change the name of the geometry to easily build geodataframes with this serie
        pm_row["geometry"] = pm_row["pm_geom"]

        # Now same job for the outlines
        if self.extended_outlines_loaded():

            # Extended
            oue_row = pd.Series(dtype='object')
            for rgu_attr_name in self.get_oue_rel_atives_attributes_names():
                oue_row[rgu_attr_name] = getattr(self, rgu_attr_name)
        
            # Add explicit geometry column
            oue_row["geometry"] = oue_row["rgu_oue_geom"]

            # Add a column to have the name
            oue_row["Out.Type"] = 'Extended'

            # Restricted
            our_row = pd.Series(dtype='object')
            for rgu_attr_name in self.get_our_rel_atives_attributes_names():
                our_row[rgu_attr_name] = getattr(self, rgu_attr_name) 
            our_row["geometry"] = our_row["rgu_our_geom"]
            our_row["Out.Type"] = 'Restricted'
        
        return pm_row, oue_row, our_row

    def get_buffer_around_marker(self, buffer_width=200):
        return self.pm_geom.buffer(buffer_width).envelope
        
    def get_most_representative_polygon(self, buffer_width=200):
        """
        Return either extended outline if available or a squared buffer around the primary marker 
        """
        if self.rgu_oue_geom is not None : 
            aoi = self.rgu_oue_geom
        else:
            aoi = self.get_buffer_around_marker(buffer_width)
        return aoi

    def load_outlines(self, outlines_layer):
        """
        return Rock Glacier Unit outlines if availables
        """

        # Search the oulines containing the primary marker
        outlines_features = outlines_layer[outlines_layer.contains(self.pm_geom)]
        outlines_features = outlines_features.dropna(subset='ou_type')

        # Get the attributes of each outlines into the RGU feature
        [self._set_attributes_from_rogi_ch_feature(outline) for outline in outlines_features.iloc]
        
        try:
            if self.rgu_our_geom is None:
                self.rgu_our_geom = self.rgu_oue_geom
            elif self.rgu_oue_geom is None:
                self.rgu_oue_geom = self.rgu_our_geom

            return self.rgu_oue_geom, self.rgu_our_geom

        except AttributeError:
            return None, None

    def extended_outlines_loaded(self):
        return self.rgu_oue_geom is not None
    
    def restricted_outlines_loaded(self):
        return self.rgu_our_geom is not None

    def make_ous_geoms_valids(self):
        if self.extended_outlines_loaded():
            self.rgu_oue_geom = shapely.make_valid(self.rgu_oue_geom)
        if self.restricted_outlines_loaded():
            self.rgu_our_geom = shapely.make_valid(self.rgu_our_geom)
                           
    def read_raster_on_rgu(self, 
            target_source   : str | gpd.GeoDataFrame | rt.geoim.Geoim,  
            layername       : str ='',  
            nRes            : numbers.Real = None, 
            buffer_width    : numbers.Real = 500, 
            load_pixels     : bool = False):
        """
        Open any raster or set of rasters on the extent of the rock glacier
        Caution : the epsg have to be the same for the raster and for the rgu

        target_source : a string to a raster or to a raster dir or a geodataframe
        layername     : to access the good layer if target_source is a path to a geopackage
        """

        # Get the extent of the Rock Glacier Unit
        aoi = self.get_most_representative_polygon(buffer_width)

        # Load the raster or rasters on this extent, with resampling if needed
        if type(target_source) is not rt.geoim.Geoim:
            target = rt.OpenFromMultipleTargets(
                target_source = target_source, 
                layername = layername, 
                area_of_interest = aoi, 
                nRes = nRes, 
                load_pixels=load_pixels)
        else:
            target = rt.Open(
                target_source, 
                geoExtent = aoi, 
                nRes = nRes, 
                load_pixels=load_pixels) 
        
        return target

    def initialize_dem(self,
        dem_source  : str | gpd.GeoDataFrame | rt.geoim.Geoim,  
        layername   : str ='', 
        nRes        : numbers.Real = None, 
        buffer_width: numbers.Real = 200,
        verbose=True) -> gdal.Dataset:
        """
        Open any DEM or set of DEMS on the extent of the rock glacier
        target_source : a string to a DEM or to a DEM dir or a geodataframe with metadata of the DEMS
        layername     : to access the good layer if target_source is a path to a geopackage  
        buffer_width  : open the DEM outside of the rg extent, but on a larger squared area of width corresponding to buffer_width argument
        """
        if not self.extended_outlines_loaded():
            if verbose:
                print(f'outlines not loaded, dem only around primary marker with {buffer_width}m buffer around')
        self.rgu_dem = self.read_raster_on_rgu(dem_source, layername, nRes, buffer_width, load_pixels=False)
        return self.rgu_dem

    def get_dem(self):

        # Make sure the DEM is initialized on the RGU
        assert self.rgu_dem is not None, 'dem not initialized'

        # If it's already loaded we just return it
        if type(self.rgu_dem) == rt.geoim.Geoim:
            return self.rgu_dem

        # Else we load it
        else:
            self.rgu_dem = rt.Open(self.rgu_dem, load_pixels=True)
        return self.rgu_dem

    def get_slope(self):
        """
        Return a slope raster with degrees values. The DEM have to be initialized on the RGU feature.
        """
        if self.rgu_slope is None:
            self.rgu_slope = rt.getSlope(self.get_dem())
        return self.rgu_slope

    def get_aspect(self):
        """
        Return an aspect raster with degrees values. The DEM have to be initialized on the RGU feature.
        """
        if self.rgu_aspect is None:
            self.rgu_aspect = rt.getAspect(self.get_dem())
        return self.rgu_aspect

    def get_cardinals(self):
        """
        Return a raster with values between 1 and 8.
        1 : North, 3 : East, 5 : South, 7 : West
        DEM have to be initialized on the RGU feature.
        """
        if self.rgu_cardinals is None:
            self.rgu_cardinals = rt.Open(self.get_aspect(), load_pixels=True).getCardiPointsFromAspect()
        return self.rgu_cardinals
        
    def get_geo_segments(self):
        """
        Send major segment, which is supposed to be the one perpendicular to the front, and the minor, which is ideally parallel to the front. 
        """
        assert self.extended_outlines_loaded(), 'outlines unloaded'
        if self.rgu_geo_major is not None and self.rgu_geo_minor is not None:
            return self.rgu_geo_major, self.rgu_geo_minor
        else:  self.rgu_geo_major, self.rgu_geo_minor = vt.getMainAxes(self.rgu_oue_geom)        
        return self.rgu_geo_major, self.rgu_geo_minor

    def get_raster_values_on_geo_segments(self, target, window_size=1):
        """
        Extract the pixels values of the raster along the 2 geoaxes of the RGU feature
        """

        # Get geo axes
        geo_major, geo_minor = self.get_geo_segments()
        
        # Extract values on the raster
        major_profile = np.array(target.inspectGeoLine(geo_major))
        minor_profile = np.array(target.inspectGeoLine(geo_minor))

        # Apply a convolutionnal filter if required
        conv_kernel = np.ones(window_size) / window_size
        major_profile = np.convolve(major_profile, conv_kernel)
        minor_profile = np.convolve(minor_profile, conv_kernel)

        return major_profile, minor_profile

    def get_topo_profiles(self, window_size=1):
        """
        Extract topographic profiles along the 2 geoaxes of the RGU feature
        """
        dem = self.get_dem()
        topo_major_profile, topo_minor_profile = self.get_raster_values_on_geo_segments(dem, window_size)
        return topo_major_profile, topo_minor_profile

    def get_slope_profiles(self, window_size=1):
        """
        Extract topographic profiles along the 2 geoaxes of the RGU feature
        """
        slope = self.get_slope()
        slope_major_profile, slope_minor_profile = self.get_raster_values_on_geo_segments(slope, window_size)
        return slope_major_profile, slope_minor_profile

    def load_manual_moving_areas(self, moving_areas_layer):
        """
        create new instance attributes :
            rgu_oue_ma and rgu_our_ma, de type pd df
            (eventuellement d'autres, pour lire directement l'attribut cinématique depuis celui de la moving-area)            
        """

        # Read the moving areas full layer
        moving_areas_layer = vt.Open(moving_areas_layer)

        # Clip du layer de moving areas sur le glacier rocheux
        self.rgu_oue_mas = moving_areas_layer.clip(self.rgu_oue_geom)
        self.rgu_our_mas = moving_areas_layer.clip(self.rgu_our_geom)
        
        return (self.rgu_oue_mas, self.rgu_our_mas)
        
    def close_dem(self):
        """
        Delete from memory the dem and the slope, which are 2 heavy objects
        """
        del self.rgu_dem
        return None

    def close_slope(self):
        del self.rgu_slope
        return None
        
    def close_aspect(self):
        del self.rgu_aspect
        return None
        
    def show_map(self, ax='', basemap=False, markersize=20, show_coordinates=False, buffersize=250, draw_vertexes=True):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # If outlines are loaded:
        if self.extended_outlines_loaded():
        
            # Draw outlines
            gpd.GeoSeries([self.rgu_oue_geom]).boundary.plot(ax=ax, color=self.rgu_oue_color, linewidth=1.5)
            gpd.GeoSeries([self.rgu_our_geom]).boundary.plot(ax=ax, color=self.rgu_our_color, linewidth=1.5)

            if draw_vertexes:

                # Draw axes vertexes
                major, minor = self.get_geo_segments()
                gpd.GeoSeries([major]).boundary.plot(ax=ax, color=self.rgu_major_axis_color , markersize=10)
                gpd.GeoSeries([minor]).boundary.plot(ax=ax, color=self.rgu_minor_axis_color,  markersize=10)

                # Draw axes
                gpd.GeoSeries([major]).plot(ax=ax, color=self.rgu_major_axis_color , linewidth=1.5)
                gpd.GeoSeries([minor]).plot(ax=ax, color=self.rgu_minor_axis_color,  linewidth=1.5)

        # Else : we simply draw a buffer around the primary marker and we show it
        else:
            print('Outlines not loaded - display only the Primary Marker')
            # Create a buffer around the marker but do not show it
            basic_extent = gpd.GeoSeries([self.get_buffer_around_marker(buffersize)]).plot(ax=ax, alpha=0)

        # Draw the Primary Marker - atfer the outlines or the buffer, 
        # just to define the map extent on them and not just on the marker
        gpd.GeoSeries([self.pm_geom]).plot(ax=ax, color=self.pm_color, markersize=markersize)
                    
        # Add background with satellite view
        if basemap :
            cx.add_basemap(ax=ax, source=cx.providers.SwissFederalGeoportal.SWISSIMAGE, crs=self.rgu_epsg)
        
        # Deactivate geographic coordinates on the borders of the graph
        if not show_coordinates: 
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        return ax

    def show_topo_profile(self, mode='', ax='', window_size=1, linewidth=0.2):

        if ax == '':
            ax = plt.subplot()

        # Extract the color to give to the line from the instances settings
        color = {'major':self.rgu_major_axis_color, 'minor':self.rgu_minor_axis_color}[mode]

        # Get topo values in degrees along the axis
        topo_values = self.get_topo_profiles()
        topo_profiles = {'major':topo_values[0], 'minor':topo_values[1]}
        raw_profile = topo_profiles[mode.lower()]

        # Draw a first line
        ax.plot(raw_profile, linewidth=linewidth*15, color=color)
        ax.plot(raw_profile, linewidth=linewidth*2, color='white')

        # Draw a larger line in the background after applying a convolutionnal filtering on the values
        if window_size != 1:
            conv_topo_values = self.get_topo_profiles(window_size=window_size)
            conv_topo_profiles = {'major':conv_topo_values[0], 'minor':conv_topo_values[1]}
            conv_profile = conv_topo_profiles[mode.lower()]
            # ax.plot(conv_profile[window_size:-window_size], linewidth=linewidth*15, color=color, alpha=0.5)
            # ax.plot(conv_profile[window_size:-window_size], linewidth=linewidth*2, color='white', alpha=1)

    def show_slope_profile(self, mode='', ax='', window_size=1, linewidth=0.2):

        if ax == '':
            ax = plt.subplot()

        # Extract the color to give to the line from the instances settings
        color = {'major':self.rgu_major_axis_color, 'minor':self.rgu_minor_axis_color}[mode]

        # Get slope values in degrees along the axis
        slope_values = self.get_slope_profiles()
        slope_profiles = {'major':slope_values[0], 'minor':slope_values[1]}
        raw_profile = slope_profiles[mode.lower()]

        # Draw a first line
        # ax.plot(raw_profile, linewidth=linewidth, color=color)

        # Draw a larger line in the background after applying a convolutionnal filtering on the values
        if window_size != 1:
            conv_slope_values = self.get_slope_profiles(window_size=window_size)
            conv_slope_profiles = {'major':conv_slope_values[0], 'minor':conv_slope_values[1]}
            conv_profile = conv_slope_profiles[mode.lower()]
            ax.plot(conv_profile[window_size:-window_size], linewidth=linewidth*15, color=color, alpha=0.5)
            ax.plot(conv_profile[window_size:-window_size], linewidth=linewidth*2, color='white', alpha=1)
        return ax
        
    def show_pannel(self, basemap=False, window_size=1, z_factor=1):
        
        # Create an empty figure
        fig = plt.figure(figsize=(10, 6))

        # Create a 2x2 grid
        gs = gridspec.GridSpec(2, 3)

        # Add a first figure 
        # Note : [:,0] in numpy means 'the first column'
        ax1 = fig.add_subplot(gs[0,0])
        ax1.set_title(f'{self.pm_pid} rock glacier')

        # Add an second figure, first row second column
        ax2 = fig.add_subplot(gs[0,1])
        ax2.set_title(f'{self.pm_pid} topographic profile 1')
        ax2.set_ylabel('elevation (meters)')

        # Third figure, second row second column
        ax3 = fig.add_subplot(gs[1,1])
        ax3.set_title(f'{self.pm_pid} topographic profile 2')
        ax3.set_ylabel('elevation (meters)')

        # Forth
        ax4 = fig.add_subplot(gs[0,2])
        ax4.set_title(f'{self.pm_pid} slope profile 1')
        ax4.set_ylabel('slope (degrees)')

        # Fiveth
        ax5 = fig.add_subplot(gs[1,2])
        ax5.set_title(f'{self.pm_pid} slope profile 2')
        ax5.set_ylabel('slope (degrees)')

        # Sixth
        ax6 = fig.add_subplot(gs[1,0])
        ax6.set_title(f'{self.pm_pid} Slope Map')
        ax6.set_ylabel('Slope Map')

        # Send the result of self.show_map to ax1
        self.show_map(ax=ax1, basemap=basemap)

        # Same for ax2 & ax3, with the major & minor profiles
        self.show_topo_profile(ax=ax2, mode='major', window_size=window_size)
        self.show_topo_profile(ax=ax3, mode='minor', window_size=window_size)

        self.show_slope_profile(ax=ax4, mode='major', window_size=window_size)
        self.show_slope_profile(ax=ax5, mode='minor', window_size=window_size)

        ax6.imshow(self.get_slope().array, cmap='viridis')
        plt.colorbar(ax6.imshow(self.get_slope().array, cmap='viridis'), fraction=0.1, pad=0.1)

        # What's for ?
        plt.tight_layout()

        return fig

    def get_points_cloud(self, z_factor=1):

        # Get the RGU DEM
        dem = self.get_dem()
        # dem.maskFromVector(self.rgu_oue_geom, epsg=2056)

        # Apply a exageration on it
        dem *= z_factor
        dem_array = dem.array

        # Generate corresponding x and y coordinates
        # Assuming the spatial resolution (cell size) of the DEM is known
        cell_size = dem.getPixelSize()[0]
        height, width = dem_array.shape
        y, x = np.mgrid[0:height, 0:width] * cell_size

        # Optionally, adjust x and y if you have specific origin coordinates
        x_origin, y_origin = dem.getOrigin()
        x += x_origin
        y += y_origin

        # Flatten the x, y, and dem_array to create a list of points
        points = np.vstack((x.flatten(), y.flatten(), dem_array.flatten())).T
        return points

    def show_points_cloud(self, z_factor=1, cmap='plasma', point_size=1.5):

        points = self.get_points_cloud(z_factor=z_factor)

        # Convert the points array into a PyVista PolyData object
        cloud = pv.PolyData(points)
        
        # Add elevation as a scalar value for coloring
        cloud["elevation"] = points[:, 2]
        
        # Plot the point cloud with elevation colormap
        p = pv.Plotter()
        p.add_mesh(
            cloud, scalars="elevation", cmap=cmap, point_size=point_size)
        p.view_isometric()

        p.set_background('black')
        p.add_axes()
        p.show()

    def get_moving_areas_one_interval(
        self,
        insar_data_dir,
        orbit = 'd',
        interval = 24,
        r = 5,
        c = 100,
        n_classes = 2,
        interpolate_max_search_distance=1,
        class_to_avoid=None,
        clip_on_rock_glacier=False,
        show_steps_charts = False,
        map_background=None,
        polygons_rounded_factor=30,
        min_overlap=2,
        ):

        if orbit.lower()[0] == 'd':
            sentinel_1_track = 'T138D'
        elif orbit.lower()[0] == 'a':
            sentinel_1_track = 'T160A'

        """
        Create GeoDataFrame with potential moving areas detected on a set of insar sharing the same interval and orbit
        """

        assert self.extended_outlines_loaded(), 'The outlines are not loaded for this rock glacier'

        # Load the insars on the rock glacier bounding box
        target_insars = pyinsar.get_geoinsars(
            insar_data_dir,
            geoExtent=self.rgu_oue_geom,
            intervals=[interval],
            criterias=[sentinel_1_track],
            load_pixels=True,
            )
        
        # Detect the moving-areas on it
        stacked_moving_areas = target_insars.detect_redundant_moving_areas(
            r = r,
            c = c,
            n_classes = n_classes,
            class_to_avoid = class_to_avoid,
            interpolate_max_search_distance=interpolate_max_search_distance,
            show_steps_charts=show_steps_charts,
            map_background=map_background,
            polygons_rounded_factor=polygons_rounded_factor,
            min_overlap=min_overlap,
        )

        return stacked_moving_areas

        # Clip on the rock glacier extent
        if clip_on_rock_glacier:
            stacked_moving_areas = gpd.clip(stacked_moving_areas, mask=self.rgu_oue_geom)

        # Count the redundance score of each part of the detected signals
        redundancy_scored_moving_areas_parts = vt.count_overlap(stacked_moving_areas)

        # Make a selection to get only the parts overlapping another part
        redundant_parts = redundancy_scored_moving_areas_parts[redundancy_scored_moving_areas_parts.overlap_score >= 2]

        # Dissolve and explode to merge contiguous features
        # But we want indenpendant polygons if the are discontinuous
        # We don't want multipolygons, so we explode it just after
        redundant_moving_areas = redundant_parts.dissolve().explode().set_crs(epsg=self.rgu_epsg)
        
        return redundant_moving_areas

    def get_activity(self):
        return self.rgu_activity_class

    def get_alti_pm(self):
        dem = self.get_dem()
        return dem.inspectGeoPoint(self.pm_geom)

    def get_alti_ranges(self, out_type = 'extended'):
        """
        Return alti min and max of extended outline (default) or restricted
        """
        dem = self.get_dem()

        # Process for Extended outline
        if out_type.startswith('ext'):
            dem.maskFromVector(self.rgu_oue_geom, epsg=2056)

        # Same for Restricted
        elif out_type.startswith('res'):
            dem.maskFromVector(self.rgu_our_geom, epsg=2056)

        res = np.array((ma.min(dem.array[dem.array > 0]), ma.max(dem.array[dem.array > 0])))
        dem.unmask()
        return res

    def get_alti_med_mean(self, out_type = 'extended'):
        """
        Return alti min and max of extended outline (default) or restricted
        """
        dem = self.get_dem()

        # Process for Extended outline
        if out_type.lower()[0] == 'e':
            dem.maskFromVector(self.rgu_oue_geom, epsg=2056)

        # Same for Restricted
        else:
            dem.maskFromVector(self.rgu_our_geom, epsg=2056)

        res = ma.median(dem.array[dem.array > 0]), ma.mean(dem.array[dem.array > 0])
        dem.unmask()
        return res

    def get_slope_pm(self):
        slope = self.get_slope()
        return slope.inspectGeoPoint(self.pm_geom)

    def get_slope_ranges(self, out_type = 'extended'):
        """
        Return slope min and max of extended outline (default) or restricted
        """
        slope = self.get_slope()

        # Process for Extended outline
        if out_type.lower()[0] == 'e':
            if self.rgu_oue_geom is not None:
                slope.maskFromVector(self.rgu_oue_geom, epsg=2056)
            else:
                return None

        # Same for Restricted
        else:
            if self.rgu_our_geom is not None:
                slope.maskFromVector(self.rgu_our_geom, epsg=2056)

        res = np.array((ma.min(slope.array[slope.array > 0]), ma.max(slope.array[slope.array > 0])))
        slope.unmask()
        return res
        
    def get_slope_med_mean(self, out_type = 'extended'):
        """
        Return slope min and max of extended outline (default) or restricted
        """
        slope = self.get_slope()

        # Process for Extended outline
        if out_type.lower()[0] == 'e':
            if self.rgu_oue_geom is not None:
                slope.maskFromVector(self.rgu_oue_geom, epsg=2056)
            else:
                return None

        # Same for Restricted
        else:
            if self.rgu_our_geom is not None:
                slope.maskFromVector(self.rgu_our_geom, epsg=2056)
            else:
                return None
        res = ma.median(slope.array), ma.mean(slope.array)
        slope.unmask()
        return res

    def load_margins(self):
        """
        Return geometry only between the restricted and extended outlines
        """
        if self.extended_outlines_loaded() and self.restricted_outlines_loaded():
            self.make_ous_geoms_valids()
            self.rgu_margins_geom = self.rgu_oue_geom - self.rgu_our_geom
        
    def get_slope_stats_margins(self):
        """
        Return min, max, med, mean slope values only between the restricted and extended outlines
        """
        if self.extended_outlines_loaded() and self.restricted_outlines_loaded():
            slope = self.get_slope()
            slope.maskFromVector(self.rgu_margins_geom, epsg=self.rgu_epsg)
            res = (ma.min(slope.array), ma.max(slope.array), slope.median(), slope.mean())
            slope.unmask()
            return res
        else:
            return (None, None, None, None)
        
    def get_cardinal_point(self, out_type='extended'):
        """
        Get one cardinal point per dem pixel inside the outline
        and return the most redundant value 
        """
        cardis = self.get_cardinals().copy()

        # Process for Extended outline
        if out_type.lower()[0] == 'e':
            if self.rgu_oue_geom is not None:
                cardis.maskFromVector(self.rgu_oue_geom, epsg=2056)
            else:
                return None

        # Same for Restricted
        else:
            if self.rgu_our_geom is not None:
                cardis.maskFromVector(self.rgu_our_geom, epsg=2056)

        cardinals_letters = {-1:'unknown', 1:'N', 2:'NE', 3:'E', 4:'SE', 5:'S', 6:'SW', 7:'W', 8:'NW'}
        return cardinals_letters[round(cardis.median())]

    def get_lengths(self, out_type='extended'):
        """
        Return the length of minor and major segments
        """

        # Process for Extended outline
        if out_type.lower()[0] == 'e':
            if self.rgu_oue_geom is not None:
                target_geom = self.rgu_oue_geom
            else:
                return None

        # Same for Restricted
        else:
            if self.rgu_our_geom is not None:
                target_geom = self.rgu_our_geom

        # Make a bounding box
        ax_a, ax_b = vt.getMainAxes(target_geom)
        len_a, len_b = shapely.length(ax_a), shapely.length(ax_b)
        len_min = np.min((len_a, len_b))
        len_max = np.max((len_a, len_b))
        return len_min, len_max
        
    def get_elongated_ratio(self, out_type='extended'):
        """
        Compute the ratio between the 2 main axes of the outline bounding box
        """
        len_min, len_max = self.get_lengths(out_type)
        return np.round(len_min / len_max * 100)

    def get_swimage(self, local_repository, acq_year=None, nRes=None, load_pixels=False):
        """
        Get list of geoims representing aerial images of the rock glacier. 
        If the image is not existing in local_repository, download the image.
        """

        # Download and / or identify the images intersecting the rock glacier
        target_tiles = sgd.get_swimages(
            dest_repo=local_repository,
            study_area_source=self.get_most_representative_polygon(buffer_width=1500),
            acq_year=acq_year,
            verbose=True)

        # Rename the column just to be coherent with rt.OpenMultipleTargets
        # target_tiles = target_tiles.rename({'dest_filepath':'filepath'}, axis=1)        

        # Assume there is only one aerial campaign per year on the rock glacier
        # ds_per_year = []
        # for year in target_tiles.acq_year.unique():
        #     target_year_tiles = target_tiles[target_tiles.acq_year == year]
        #     ds = rt.OpenFromMultipleTargets(
        #         target_year_tiles,
        #         area_of_interest=self.get_most_representative_polygon(buffer_width=1500).buffer(500),
        #         load_pixels=load_pixels,
        #         nRes=nRes)
        #     ds_per_year.append(ds)

        return target_tiles

    def get_detailled_serie(self, source_dem, dem_res_order_1=5, dem_res_order_2=0.5, terrain_attributes=False):
        """
        Make a very detailled pandas.series (a 'line' of a dataset)
        The dem_res_order_1 refers to the size of dem pixels which will be used to compute variables such as the cardinal point of the rock glacier
        order_2 will be used for variables such as the mean slope in the margins, asking for a more fine-grain dem.
        """

        if not self.extended_outlines_loaded() and not self.extended_outlines_loaded():
            return None

        self.load_margins()

        rgu_row = pd.Series(dtype='object')
        rgu_row['pm_pid'] = self.pm_pid

        # Activity and kinematics
        rgu_row['rgu_activity_class'] = self.get_activity()
        rgu_row['rgu_kin_att'] = self.rgu_kin_att
        rgu_row['rgu_destabilized'] = self.rgu_destabilized

        if terrain_attributes :

            # Load DEM with resolution order 1
            self.initialize_dem(source_dem, nRes=dem_res_order_1)

            # Compute rough topo indexes
            rgu_row['cardi_pt'] = self.get_cardinal_point()
            rgu_row['alti_oue_min'], rgu_row['alti_oue_max'] = self.get_alti_ranges('e')
            rgu_row['alti_our_min'], rgu_row['alti_our_max'] = self.get_alti_ranges('r')
            rgu_row['alti_oue_med'], rgu_row['alti_oue_mean'] = self.get_alti_med_mean('e')
            rgu_row['alti_our_med'], rgu_row['alti_our_mean'] = self.get_alti_med_mean('r')
            rgu_row['alti_pm'] = self.get_alti_pm()
            rgu_row['slope_oue_min'], rgu_row['slope_oue_max'] = self.get_slope_ranges('e')
            rgu_row['slope_oue_med'], rgu_row['slope_oue_mean'] = self.get_slope_med_mean('e')    
            rgu_row['slope_our_med'], rgu_row['slope_our_mean'] = self.get_slope_med_mean('r')
            rgu_row['slope_our_min'], rgu_row['slope_our_max'] = self.get_slope_ranges('r')
    
            # Reload the dem with the 2nd order
            del self.rgu_dem
            del self.rgu_slope
            self.rgu_dem = None
            self.rgu_slope = None
            self.initialize_dem(source_dem, nRes=dem_res_order_2)

            # Compute detailled topo indexes
            rgu_row['slope_marg_min'], rgu_row['slope_marg_max'], rgu_row['slope_marg_med'], rgu_row['slope_marg_mean'] = self.get_slope_stats_margins()

        # 2D geometry
        rgu_row['len_min_our'], rgu_row['len_max_our'] = self.get_lengths('r')
        rgu_row['surf_our_ha'] = self.rgu_our_geom.area / 10000
        rgu_row['elongated_our_ratio'] = self.get_elongated_ratio('r')

        rgu_row['len_min_oue'], rgu_row['len_max_oue'] = self.get_lengths('e')
        rgu_row['elongated_oue_ratio'] = self.get_elongated_ratio('e')
        rgu_row['surf_oue_ha'] = self.rgu_oue_geom.area / 10000

        if self.rgu_oue_geom != self.rgu_our_geom:
            rgu_row['surf_margins'] = self.rgu_margins_geom.area / 10000
        else:
            rgu_row['surf_margins'] = 0

        # Moving Areas
        # if len(self.rgu_our_mas) > 0:
        #     rgu_row['surf_oue_covered_by_ma_ratio'] = (self.rgu_oue_mas.dissolve().area.iloc[0] / 10000) / rgu_row['surf_oue_ha'] * 100
        #     rgu_row['surf_our_covered_by_ma_ratio'] = (self.rgu_our_mas.dissolve().area.iloc[0] / 10000) / rgu_row['surf_our_ha'] * 100
        # else:
        #     rgu_row['surf_oue_covered_by_ma_ratio'] = 0
        #     rgu_row['surf_our_covered_by_ma_ratio'] = 0

        # Geometry
        # TODO : on peut mettre plusieurs géométries différentes dans un gdf pour peu que les colonnes ne s'appellent pas "geometry"
        # On peut ensuite modifier la géométrie active de l'entité avec set_geometry('column_name').
        # En revanche, pour l'écrire dans un fichier gpkg ou shapefile, ce n'est pas possible : il faut une et unique colonne contenant
        # des objets de type geometry. 
        rgu_row['oue_geom'] = self.rgu_oue_geom
        rgu_row['our_geom'] = self.rgu_our_geom
        rgu_row['pm_geom'] = self.pm_geom

        # Upslope connection
        rgu_row['upslope_con'] = self.rgu_upsl_con

        return rgu_row