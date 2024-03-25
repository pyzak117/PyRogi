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
from telenvi import raster_tools as rt
from telenvi import vector_tools as vt
import pyvista as pv

import warnings

# Supprimer tous les avertissements
warnings.filterwarnings("ignore")
    
class RockGlacierUnit:

    """
    Describe a Rock Glacier. 
    This conceptual class link all the differents
    objects described in the RGIK Guidelines.
    """

    def __init__(self, rgik_feature): 

        """
        Initializes a RockGlacierUnit object, dynamically creating an attribute
        for each column in the rgik geopackage or shapefile, after transforming the column names
        to valid Python attribute names.
        """

        # Transform geopackage columns into attributes
        self._set_attributes_from_rgik_feature(rgik_feature)
        
        # Rename some of the attributes
        self.rgu_pm_geom = self.rgik_pm_geometry
        self.rgu_oue_geom  = None
        self.rgu_our_geom  = None
        self.rgu_oue_color = 'blue'
        self.rgu_our_color = 'green'
        self.rgu_dem       = None
        self.rgu_slope     = None
        self.rgu_aspect    = None
        self.rgu_geo_major = None
        self.rgu_geo_minor = None
        self.rgu_major_axis_color = 'purple'
        self.rgu_minor_axis_color = 'orange' 
        self.rgu_epsg     = 2056

    def _set_attributes_from_rgik_feature(self, rgik_feature):
        """
        Dynamically set attributes based on the GeoSeries feature's columns.
        Transforms column names to create valid Python attribute names.
        Used for "drink" the PM attributes and the Outlines attributes from a pandas.Series object
        """

        # Build a prefix to know where the attributes are coming from
        if 'polygon' in rgik_feature.geometry.geom_type.lower():
            if 'extended' in rgik_feature['Out.Type'].lower():
                attribute_prefix = 'rgik_oue'
            if 'restricted' in rgik_feature['Out.Type'].lower():
                attribute_prefix = 'rgik_our'
        elif 'point' in rgik_feature.geometry.geom_type.lower():
            attribute_prefix = 'rgik_pm'

        # Set one attribute per column in the feature
        for column_name, value in rgik_feature.items():

            # Lowercase
            valid_attr_name = column_name.lower()

            # Remove the last point
            if valid_attr_name.endswith('.'):
                valid_attr_name = valid_attr_name[:-1]

            # Change points in underscores
            valid_attr_name = valid_attr_name.replace('.', '_')

            # Add rgik as as signature if not already existing
            if not attribute_prefix in valid_attr_name:
                valid_attr_name = f'{attribute_prefix}_{valid_attr_name}'

            # Add attribute to the RGU instance
            setattr(self, valid_attr_name, value)

    def get_pm_relatives_attributes_names(self):
        return [rgu_attr for rgu_attr in dir(self) if 'pm_' in rgu_attr and not 'get' in rgu_attr]

    def get_oue_relatives_attributes_names(self):
        return [rgu_attr for rgu_attr in dir(self) if 'oue_' in rgu_attr and not 'get' in rgu_attr]

    def get_our_relatives_attributes_names(self):
        return [rgu_attr for rgu_attr in dir(self) if 'our_' in rgu_attr and not 'get' in rgu_attr]

    def smashed_in_serie(self):
        """
        Send a pandas.Series object. with all the attributes. 
        The column 'geometry' is set from the extended outlines
        """

        # Check availibility of the outlines
        assert self.outlines_loaded(), 'nothing to smash because no outlines'

        # Create empty serie
        rgu_row = pd.Series(dtype='object')

        # Add Primary Markers attributes
        for rgu_attr_name in self.get_pm_relatives_attributes_names():
            rgu_row[rgu_attr_name] = getattr(self, rgu_attr_name)
        
        # Add restricted outlines attributes
        for rgu_attr_name in self.get_our_relatives_attributes_names():
            rgu_row[rgu_attr_name] = getattr(self, rgu_attr_name) 
        
        # Add extended outlines attributes
        for rgu_attr_name in self.get_oue_relatives_attributes_names():
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
        pm_row["geometry"] = pm_row["rgu_pm_geom"]

        # Now same job for the outlines
        if self.outlines_loaded():

            # Extended
            oue_row = pd.Series(dtype='object')
            for rgu_attr_name in self.get_oue_relatives_attributes_names():
                oue_row[rgu_attr_name] = getattr(self, rgu_attr_name)
        
            # Add explicit geometry column
            oue_row["geometry"] = oue_row["rgu_oue_geom"]

            # Add a column to have the name
            oue_row["Out.Type"] = 'Extended'

            # Restricted
            our_row = pd.Series(dtype='object')
            for rgu_attr_name in self.get_our_relatives_attributes_names():
                our_row[rgu_attr_name] = getattr(self, rgu_attr_name) 
            our_row["geometry"] = our_row["rgu_our_geom"]
            our_row["Out.Type"] = 'Restricted'
        
        return pm_row, oue_row, our_row

    def get_buffer_around_marker(self, buffer_width=500):
        return self.rgu_pm_geom.buffer(buffer_width).envelope

    def get_most_representative_polygon(self, buffer_width=500):
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
        outlines_features = outlines_layer[outlines_layer.contains(self.rgu_pm_geom)]

        # Get the attributes of each outlines into the RGU feature
        [self._set_attributes_from_rgik_feature(outline) for outline in outlines_features.iloc]

        try:
            # Rename the geometrics attributes
            self.rgu_oue_geom = self.rgik_oue_geometry
            self.rgu_our_geom = self.rgik_our_geometry

            # Like "rename" the attributes
            del self.rgik_oue_geometry
            del self.rgik_our_geometry

            return self.rgu_oue_geom, self.rgu_our_geom 

        except AttributeError:
            return None, None

    def outlines_loaded(self):
        return self.rgu_oue_geom is not None

    def read_raster_on_rgu(self, 
            target_source   : str | gpd.GeoDataFrame,  
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
        target = rt.OpenFromMultipleTargets(
            target_source = target_source, 
            layername = layername, 
            area_of_interest = aoi, 
            nRes = nRes, 
            load_pixels=load_pixels)
        
        return target

    def initialize_dem(self,
        dem_source  : str | gpd.GeoDataFrame,  
        layername   : str ='', 
        nRes        : numbers.Real = None, 
        width       : numbers.Real = 500) -> gdal.Dataset:
        """
        Open any DEM or set of DEMS on the extent of the rock glacier
        target_source : a string to a DEM or to a DEM dir or a geodataframe with metadata of the DEMS
        layername     : to access the good layer if target_source is a path to a geopackage        
        """
        self.rgu_dem = self.read_raster_on_rgu(dem_source, layername, nRes, width, load_pixels=False)
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

    def get_alti_pm(self):
        dem = self.get_dem()
        return dem.inspectGeoPoint(self.rgu_pm_geom)
        
    def get_altitudinal_range(self):
        dem = self.get_dem()
        dem.maskFromVector(self.rgu_oue_geom, epsg=2056)
        return np.array((np.min(dem.array), np.max(dem.array)))

    def get_geo_segments(self):
        assert self.outlines_loaded(), 'outlines unloaded'
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
        
    def show_map(self, ax='', basemap=False):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # Draw outlines
        gpd.GeoSeries([self.rgu_oue_geom]).boundary.plot(ax=ax, color=self.rgu_oue_color, linewidth=1.5)
        gpd.GeoSeries([self.rgu_our_geom]).boundary.plot(ax=ax, color=self.rgu_our_color, linewidth=1.5)

        # Draw axes vertexes
        major, minor = self.get_geo_segments()
        gpd.GeoSeries([major]).boundary.plot(ax=ax, color=self.rgu_major_axis_color , markersize=10)
        gpd.GeoSeries([minor]).boundary.plot(ax=ax, color=self.rgu_minor_axis_color,  markersize=10)

        # Draw axes
        gpd.GeoSeries([major]).plot(ax=ax, color=self.rgu_major_axis_color , linewidth=1.5)
        gpd.GeoSeries([minor]).plot(ax=ax, color=self.rgu_minor_axis_color,  linewidth=1.5)

        # Add background with satellite view
        if basemap :
            cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery, crs=self.rgu_epsg)
        
        # Deactivate geographic coordinates on the borders of the graph
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax

    def show_topo_profile(self, mode='', ax='', window_size=1, linewidth=0.1):

        if ax == '':
            ax = plt.subplot()

        # Extract the color to give to the line from the instances settings
        color = {'major':self.rgu_major_axis_color, 'minor':self.rgu_minor_axis_color}[mode]

        # Get topo values in degrees along the axis
        topo_values = self.get_topo_profiles()
        topo_profiles = {'major':topo_values[0], 'minor':topo_values[1]}
        raw_profile = topo_profiles[mode.lower()]

        # Draw a first line
        ax.plot(raw_profile, linewidth=linewidth, color=color)

        # Draw a larger line in the background after applying a convolutionnal filtering on the values
        if window_size != 1:
            conv_topo_values = self.get_topo_profiles(window_size=window_size)
            conv_topo_profiles = {'major':conv_topo_values[0], 'minor':conv_topo_values[1]}
            conv_profile = conv_topo_profiles[mode.lower()]
            ax.plot(conv_profile[window_size:-window_size], linewidth=linewidth*15, color=color, alpha=0.5)
            ax.plot(conv_profile[window_size:-window_size], linewidth=linewidth*2, color='white', alpha=1)

    def show_slope_profile(self, mode='', ax='', window_size=1, linewidth=0.1):

        if ax == '':
            ax = plt.subplot()

        # Extract the color to give to the line from the instances settings
        color = {'major':self.rgu_major_axis_color, 'minor':self.rgu_minor_axis_color}[mode]

        # Get slope values in degrees along the axis
        slope_values = self.get_slope_profiles()
        slope_profiles = {'major':slope_values[0], 'minor':slope_values[1]}
        raw_profile = slope_profiles[mode.lower()]

        # Draw a first line
        ax.plot(raw_profile, linewidth=linewidth, color=color)

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
        ax1.set_title(f'{self.rgik_pm_workingid} rock glacier')

        # Add an second figure, first row second column
        ax2 = fig.add_subplot(gs[0,1])
        ax2.set_title(f'{self.rgik_pm_workingid} topographic profile 1')
        ax2.set_ylabel('elevation (meters)')

        # Third figure, second row second column
        ax3 = fig.add_subplot(gs[1,1])
        ax3.set_title(f'{self.rgik_pm_workingid} topographic profile 2')
        ax3.set_ylabel('elevation (meters)')

        # Forth
        ax4 = fig.add_subplot(gs[0,2])
        ax4.set_title(f'{self.rgik_pm_workingid} slope profile 1')
        ax4.set_ylabel('slope (degrees)')

        # Fiveth
        ax5 = fig.add_subplot(gs[1,2])
        ax5.set_title(f'{self.rgik_pm_workingid} slope profile 2')
        ax5.set_ylabel('slope (degrees)')

        # Sixth
        ax6 = fig.add_subplot(gs[1,0])
        ax6.set_title(f'{self.rgik_pm_workingid} Slope Map')
        ax6.set_ylabel('Slope Map')

        # Send the result of self.show_map to ax1
        self.show_map(ax=ax1, basemap=basemap)

        # Same for ax2 & ax3, with the major & minor profiles
        self.show_topo_profile(ax=ax2, mode='major', window_size=window_size)
        self.show_topo_profile(ax=ax3, mode='minor', window_size=window_size)

        self.show_slope_profile(ax=ax4, mode='major', window_size=window_size)
        self.show_slope_profile(ax=ax5, mode='minor', window_size=window_size)

        ax6.imshow(self.get_slope().array, cmap='Oranges')

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