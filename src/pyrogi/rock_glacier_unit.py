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
        self.rgu_pm_geom  = self.rgik_pm_geometry
        self.rgu_oue_geom = None
        self.rgu_our_geom = None
        self.rgu_dem      = None
        self.rgu_slope    = None
        self.rgu_major_axis_color = 'red'
        self.rgu_minor_axis_color = 'orange'
        self.rgu_epsg     = 2056

    def _set_attributes_from_rgik_feature(self, rgik_feature):
        """
        Dynamically set attributes based on the GeoSeries feature's columns.
        Transforms column names to create valid Python attribute names.
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

            # Add rgik as as signature
            valid_attr_name = f'{attribute_prefix}_{valid_attr_name}'

            # Add attribute to the RGU instance
            setattr(self, valid_attr_name, value)

    def get_most_representative_geometry(self, buffer_width=500, buffer_height=500):
        """
        Return either extended outline if available or a squared buffer around the primary marker 
        """
        if self.outlines_is_known():
            aoi = self.rgu_oue_geom
        else:
            aoi = self.rgu_pm_geom.buffer(buffer_width).envelope
        return aoi

    def get_raster_on_rgu(
            self,
            target_source  : str | gpd.GeoDataFrame, 
            layername       : str ='', 
            nRes            : numbers.Real = None,
            width           : numbers.Real = 500, 
            height          : numbers.Real = 500,
            load            : bool = False):

        # Here we draw a geographic area where to crop and merge the dems
        aoi = self.get_most_representative_geometry(width, height)

        target = rt.OpenFromMultipleTargets(
            target_source = target_source,
            layername = layername,
            area_of_interest = aoi,
            nRes = nRes,
            load_pixels=False)

        return target

    def load_outlines(self, outlines_layer):
        """
        return Rock Glacier Unit outlines if availables
        """
        outlines_features = outlines_layer[outlines_layer.contains(self.rgu_pm_geom)]
        [self._set_attributes_from_rgik_feature(outline) for outline in outlines_features.iloc]
        self.rgu_oue_geom = self.rgik_oue_geometry
        self.rgu_our_geom = self.rgik_our_geometry
        return self.rgu_oue_geom, self.rgu_our_geom

    def initialize_dem(self, dem_source  : str | gpd.GeoDataFrame,  layername   : str ='',  nRes        : numbers.Real = None, width       : numbers.Real = 500,  height      : numbers.Real = 500) -> rt.geoim.Geoim:
        self.rgu_dem = self.get_raster_on_rgu(dem_source, layername, nRes, width, height)
        return self.rgu_dem

    def load_dem(self):
        assert self.dem_is_known(), 'dem not loaded'
        self.rgu_dem = rt.Open(self.rgu_dem, load_pixels=True)
        return self.rgu_dem

    def load_slope(self):
        assert self.dem_is_known(), 'dem not loaded'
        self.rgu_slope = rt.getSlope(self.rgu_dem)
        return self.rgu_slope

    def load_geo_axes(self):
        assert self.outlines_is_known(), 'outlines unknown'
        self.rgu_major_axis, self.rgu_minor_axis = vt.getMainAxes(self.rgu_oue_geom)
        return self.rgu_major_axis, self.rgu_minor_axis

    def load_minor_topo_profile(self, window_size=5):

        # Build a convolutionnal kernel
        kernel = np.ones(window_size) / window_size
        self.load_geo_axes()
        assert self.dem_is_known(), 'dem not loaded'
        self.minor_topo_profile = np.convolve(np.array(self.rgu_dem.inspectGeoLine(self.rgu_minor_axis)), kernel)

        return self.minor_topo_profile

    def load_major_topo_profile(self, window_size=5):

        # Build a convolutionnal kernel
        kernel = np.ones(window_size) / window_size
        self.load_geo_axes()
        assert self.dem_is_known(), 'dem not loaded'
        self.major_topo_profile = np.convolve(np.array(self.rgu_dem.inspectGeoLine(self.rgu_major_axis)), kernel)

        return self.major_topo_profile

    def load_minor_slope_profile(self, window_size=5):

        # Build a convolutionnal kernel
        kernel = np.ones(window_size) / window_size
        assert self.slope_is_known(), 'slope not loaded'
        self.load_geo_axes()
        self.minor_slope_profile = np.convolve(np.array(self.rgu_slope.inspectGeoLine(self.rgu_minor_axis)), kernel)

        return self.minor_slope_profile

    def load_major_slope_profile(self, window_size=5):

        # Build a convolutionnal kernel
        kernel = np.ones(window_size) / window_size
        assert self.slope_is_known(), 'slope not loaded'
        self.load_geo_axes()
        self.major_slope_profile = np.convolve(np.array(self.rgu_slope.inspectGeoLine(self.rgu_major_axis)), kernel)

        return self.major_slope_profile

    def slope_is_known(self):
        return self.rgu_slope is not None

    def outlines_is_known(self):
        return self.rgu_oue_geom is not None

    def dem_is_known(self):
        return self.rgu_dem is not None
    
    def show_map(self, ax='', basemap=False):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # Draw outlines
        gpd.GeoSeries([self.rgu_oue_geom]).boundary.plot(ax=ax, color='blue' , linewidth=1.5)
        gpd.GeoSeries([self.rgu_our_geom]).boundary.plot(ax=ax, color='green', linewidth=1.5)

        # Draw axes vertexes
        major, minor = self.load_geo_axes()
        gpd.GeoSeries([major]).boundary.plot(ax=ax, color=self.rgu_major_axis_color , markersize=10)
        gpd.GeoSeries([minor]).boundary.plot(ax=ax, color=self.rgu_minor_axis_color,  markersize=10)

        # Draw axes
        gpd.GeoSeries([major]).plot(ax=ax, color=self.rgu_major_axis_color , linewidth=1.5)
        gpd.GeoSeries([minor]).plot(ax=ax, color=self.rgu_minor_axis_color,     linewidth=1.5)

        # Add background with satellite view
        if basemap :
            cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery, crs=self.rgu_epsg)
        
        # Deactivate geographic coordinates on the borders of the graph
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax
    
    def show_topo_profiles(self, mode='', ax='', window_size=1):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # Draw profile
        rel_major = self.load_major_topo_profile()
        rel_minor = self.load_minor_topo_profile()

        if mode == 'major':
            ax.plot(rel_major[5:-5], linewidth=1.2, color=self.rgu_major_axis_color)

        elif mode == 'minor':
            ax.plot(rel_minor[5:-5], linewidth=1.2, color=self.rgu_minor_axis_color)

        return ax

    def show_slope_profiles(self, mode='', ax='', window_size=1):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # Draw profile
        rel_major = self.load_major_slope_profile()
        rel_minor = self.load_minor_slope_profile()

        if mode == 'major':
            ax.plot(rel_major[5:-5], linewidth=1.2, color=self.rgu_major_axis_color)

        elif mode == 'minor':
            ax.plot(rel_minor[5:-5], linewidth=1.2, color=self.rgu_minor_axis_color)

        return ax

    def show_pannel(self, basemap=False):

        # Create an empty figure
        fig = plt.figure(figsize=(10, 6))

        # Create a 2x2 grid
        gs = gridspec.GridSpec(2, 3)

        # Add a first figure 
        # Note : [:,0] in numpy means 'the first column'
        ax1 = fig.add_subplot(gs[:,0])
        ax1.set_title(f'{self.rgik_pm_workingid} rock glacier')

        # Add an second figure, first row second column
        ax2 = fig.add_subplot(gs[0,1])
        ax2.set_title(f'{self.rgik_pm_workingid} topographic profile 1')
        ax2.set_ylabel('relative elevation (meters)')

        # Third figure, second row second column
        ax3 = fig.add_subplot(gs[1,1])
        ax3.set_title(f'{self.rgik_pm_workingid} topographic profile 2')
        ax3.set_ylabel('relative elevation (meters)')

        ax4 = fig.add_subplot(gs[0,2])
        ax4.set_title(f'{self.rgik_pm_workingid} slope profile 1')
        ax4.set_ylabel('slope (degrees)')

        ax5 = fig.add_subplot(gs[1,2])
        ax5.set_title(f'{self.rgik_pm_workingid} slope profile 2')
        ax5.set_ylabel('slope (degrees)')

        # Send the result of self.show_map to ax1
        self.show_map(ax=ax1, basemap=basemap)

        # Same for ax2 & ax3, with the major & minor profiles
        self.show_topo_profiles(ax=ax2, mode='major', window_size=5)
        self.show_topo_profiles(ax=ax3, mode='minor', window_size=5)

        self.show_slope_profiles(ax=ax4, mode='major')
        self.show_slope_profiles(ax=ax5, mode='minor')

        # What's for ?
        plt.tight_layout()
