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

class RockGlacierUnit:

    """
    Describe a Rock Glacier. 
    This conceptual class link all the differents
    objects described in the RGIK Guidelines.

    ------------
    # RGIK System Attributes
    ------------

    # Primary Markers
    rgik_id        : PrimaryID
    rgik_morpho    : Morpho.
    rgik_upslcon   : Upsl.Con.
    rgik_upslcur   : Upsl.Cur.
    rgik_complet   : Complet.
    rgik_acticl    : Acti.Cl.
    rgik_destab    : Destabili.
    rgik_comment   : Comment
    rgik_workingID : WorkingID
    rgik_assoc_rgs : Assoc.RGS
    rgik_acti_ass  : Acti.Ass.
    rgik_kin_att   : Kin.Att.
    rgik_kin_rel   : Rel.Kin.
    rgik_kin_period: Kin.Period

    # Outlines (Extended = oue, Restricted = our)
    rgik_oue_relFr      = 0      : RelFr
    rgik_oue_relLeftLM      = 0  : RelLeftLM
    rgik_oue_relRightLM         = 0 : RelRightLM
    rgik_oue_relUpsCon      = 0  : RelUpsCon
    rgik_oue_RelIndex       = 0   : RelIndex
    rgik_oue_Comment        = 0    : Comment

    ------------
    # RoDynAlps System Attributes
    ------------
    
    rgu_id        : numerical identifier of the rock glacier unit
    rgs_id        : numerical identifier of the rock glacier system embedding the rgu
    rgu_operator  : initials of the operator who created the point
    rgu_reviewer  : initials of the operator who review the point
    rgu_colorcode : hexadecimal code for display the marker in qgis    
    kin_velocity_value (float) : mean velocity (cm/y) 
    """

    def __init__(
        self,

        rgik_id,
        rgik_morpho,
        rgik_upslcon,
        rgik_upslcur,
        rgik_complet,
        rgik_acticl,
        rgik_destab,
        rgik_comment,
        rgik_workingID,
        rgik_assoc_rgs,
        rgik_acti_ass,
        rgik_kin_att,
        rgik_kin_rel,
        rgik_kin_period,
        rgu_pm_geom,
        
        rgik_oue_relFr      = 0,
        rgik_oue_relLeftLM  = 0,
        rgik_oue_relRightLM = 0,
        rgik_oue_relUpsCon  = 0,
        rgik_oue_RelIndex   = 0,
        rgik_oue_Comment    = '',
        
        rgik_our_relFr      = 0,
        rgik_our_relLeftLM  = 0,
        rgik_our_relRightLM = 0,
        rgik_our_relUpsCon  = 0,
        rgik_our_RelIndex   = 0,
        rgik_our_Comment    = '',

        rgu_id              = 0,
        rgu_operator        = '',
        rgu_reviewer        = '',
        rgu_colorcode       = '',
        rgu_oue_geom        = '',
        rgu_our_geom        = '',
        rgu_epsg            = 2056,
        rgu_major_axis_color = 'orange',
        rgu_minor_axis_color = 'red' ):

        # RGIK Conceptual Model Attributes
        self.rgik_id             = rgik_id
        self.rgik_morpho         = rgik_morpho
        self.rgik_upslcon        = rgik_upslcon
        self.rgik_upslcur        = rgik_upslcur
        self.rgik_complet        = rgik_complet
        self.rgik_acticl         = rgik_acticl
        self.rgik_destab         = rgik_destab
        self.rgik_comment        = rgik_comment
        self.rgik_workingID      = rgik_workingID
        self.rgik_assoc_rgs      = rgik_assoc_rgs
        self.rgik_acti_ass       = rgik_acti_ass
        self.rgik_kin_att        = rgik_kin_att
        self.rgik_kin_rel        = rgik_kin_rel
        self.rgik_kin_period     = rgik_kin_period

        self.rgik_oue_relFr      = rgik_oue_relFr
        self.rgik_oue_relLeftLM  = rgik_oue_relLeftLM
        self.rgik_oue_relRightLM = rgik_oue_relRightLM
        self.rgik_oue_relUpsCon  = rgik_oue_relUpsCon
        self.rgik_oue_RelIndex   = rgik_oue_RelIndex
        self.rgik_oue_Comment    = rgik_oue_Comment
        
        self.rgik_our_relFr      = rgik_our_relFr
        self.rgik_our_relLeftLM  = rgik_our_relLeftLM
        self.rgik_our_relRightLM = rgik_our_relRightLM
        self.rgik_our_relUpsCon  = rgik_our_relUpsCon
        self.rgik_our_RelIndex   = rgik_our_RelIndex
        self.rgik_our_Comment    = rgik_our_Comment

        # RoDynAlps Conceptual Model Attributes
        self.rgu_id = rgu_id
        self.rgu_operator = rgu_operator
        self.rgu_reviewer = rgu_reviewer
        self.rgu_colorcode = rgu_colorcode
        self.rgu_pm_geom = rgu_pm_geom
        self.rgu_oue_geom = rgu_oue_geom
        self.rgu_our_geom = rgu_our_geom
        self.rgu_major_axis_color = rgu_major_axis_color
        self.rgu_minor_axis_color = rgu_minor_axis_color
        self.rgu_epsg = rgu_epsg
        self.rgu_dem = ''

    def __repr__(self):
        return f"""---
Rock Glacier Unit Feature        
RGIK Primary ID : {self.rgik_id}
RoDynAlps ID    : {self.rgu_id}
Outlines        : {self.get_outlines_status()}
---
        """

    def read_outline(self, outline_feature):
        """
        add attributes to the Rock Glacier Unit relatives to the outline feature
        outline_feature : a pd.Series or gpd.GeoSeries
        """

        # Extract the type of the outline_feature
        outline_type = outline_feature['Out.Type']

        # Extended feature : rgik_oue attributes
        if outline_type.lower().strip() == 'extended':
            self.rgik_oue_relFr      = outline_feature['RelFr']
            self.rgik_oue_relLeftLM  = outline_feature['RelLeftLM']
            self.rgik_oue_relRightLM = outline_feature['RelRightLM']
            self.rgik_oue_relUpsCon  = outline_feature['RelUpsCon']
            self.rgik_oue_RelIndex   = outline_feature['RelIndex']
            self.rgik_oue_Comment    = outline_feature['Comment']
            self.rgu_oue_geom        = outline_feature['geometry']

        # Restricted feature : rgik_our attributes
        elif outline_type.lower().strip() == 'restricted':
            self.rgik_our_relFr      = outline_feature['RelFr']
            self.rgik_our_relLeftLM  = outline_feature['RelLeftLM']
            self.rgik_our_relRightLM = outline_feature['RelRightLM']
            self.rgik_our_relUpsCon  = outline_feature['RelUpsCon']
            self.rgik_our_RelIndex   = outline_feature['RelIndex']
            self.rgik_our_Comment    = outline_feature['Comment']
            self.rgu_our_geom        = outline_feature['geometry']
    
        return self.rgu_oue_geom, self.rgu_our_geom

    def search_outline(self, outlines_layer):
        """
        return the outlines containing the Rock Glacier Unit marker
        outlines_layer : a geodataframe of outlines
        """

        # Track the outlines containing the marker feature
        outlines_containers = outlines_layer[outlines_layer.contains(self.rgu_pm_geom)]

        # Read each of the outlines (1 or 2, extended and / or restricted)
        return [self.read_outline(outline) for outline in outlines_containers.iloc]

    def get_outlines_status(self):
        return type(self.rgu_oue_geom) != str

    def get_axes(self):
        """
        Return 2 shapely.LineString objects, describing the major axes of the RGU extended outlines bounding box
        """

        # Get the rotated rectangle of the Extended outline
        row_box = self.rgu_oue_geom.minimum_rotated_rectangle

        # Box coords
        corners = np.array(row_box.boundary.coords)[:-1]
        
        # Split X and Y corners coordinates
        xa, xb, xc, xd = corners[:,0]
        ya, yb, yc, yd = corners[:,1]
        
        # Middle Points
        e = shapely.Point([(xa+xb)/2, (ya+yb)/2])
        f = shapely.Point([(xc+xd)/2, (yc+yd)/2])
        g = shapely.Point([(xa+xd)/2, (ya+yd)/2])
        h = shapely.Point([(xb+xc)/2, (yb+yc)/2])

        # Axis
        self.rgu_major_axis = shapely.LineString([e,f])
        self.rgu_minor_axis = shapely.LineString([g,h])
    
        return self.rgu_major_axis, self.rgu_minor_axis

    def get_dem(self, dem_metamap='', layername='', nRes=0):
        """
        dem_metamap : path to a directory containing dems,
                      path to a geopackage describing the dems maps,
                      path to a shapefile describing the dems maps,
                      a geodataframe describing the dems maps
        """

        # If the Rock Glacier Unit have already a rgu_dem feature
        if self.rgu_dem != '':
            if nRes == 0 or nRes == abs(self.rgu_dem.getPixelSize()[0]):
                return self.rgu_dem
            else:
                return self.rgu_dem.resize(nRes)

        # Here we check the existence of the RGU outlines - at least the extended one
        assert type(self.rgu_oue_geom) != '', 'the rock glacier outlines are not known'

        # First case : the dem_metamap is a string
        if type(dem_metamap) == str:

            # Make sure it's not empty
            assert dem_metamap != '', 'we need metadata about the dems to open them'
                
            # Here we have to open as a geopackage with the layername
            if dem_metamap.endswith('.gpkg'):
                if layername != '':
                    dem_metamap = gpd.read_file(dem_metamap, layer=layername)
                else:
                    dem_metamap = gpd.read_file(dem_metamap)

            # Here we have to open as a shapefile
            if dem_metamap.endswith('.shp'):
                dem_metamap = gpd.read_file(dem_metamap)

            # Here dem_metamap is relative to a path containing hundreds or thousands of dems
            else:
                dem_metamap = rt.getRastermap(dem_metamap)

        # Now we should have a metamap and we can start to work
        assert type(dem_metamap) == gpd.GeoDataFrame, 'invalid dem_metamap'

        # Here we find the tracks intersecting the rock glacier outlines
        rgu_demtracks = dem_metamap[dem_metamap.intersects(self.rgu_oue_geom)==True]

        # Here we crop each raster on the extent of the extended outline
        if nRes == 0:
            rgu_dems = [rt.Open(demtrack.filepath, load_pixels=False, geoExtent=self.rgu_oue_geom) for demtrack in rgu_demtracks.iloc]
        else:
            rgu_dems = [rt.Open(demtrack.filepath, load_pixels=False, geoExtent=self.rgu_oue_geom, nRes=nRes) for demtrack in rgu_demtracks.iloc]

        # Here we merge the tracks
        rgu_dems_merged = rt.merge(rgu_dems)

        # Here we load the data
        rgu_dem = rt.Open(rgu_dems_merged, load_pixels=True)

        # Here we attribute the dem to the RGU feature rgu_dem attribute
        self.rgu_dem = rgu_dem
        return rgu_dem

    def get_topo_profiles(self):
        """
        Return 2 arrays containing the elevation values extracted from the DEM along the Rock Glacier Unit axes
        """

        # Make sur the Rock Glacier Unit is linked to a DEM
        dem = self.get_dem()

        # Extract profiles in each axes
        major_profile = np.array(dem.inspectGeoLine(self.rgu_major_axis))
        minor_profile = np.array(dem.inspectGeoLine(self.rgu_minor_axis))
        return major_profile, minor_profile
        
    def get_relative_topo_profiles(self, window_size=1):

        # Main axes
        major, minor = self.get_topo_profiles()

        # Normalize the elevation from the lowest point
        rel_major = np.array(major) - major.min()
        rel_minor = np.array(minor) - minor.min()

        # Build a convolutionnal kernel
        kernel = np.ones(window_size) / window_size

        # 
        rel_major_new = np.convolve(rel_major, kernel)

        rel_minor_new = np.convolve(rel_minor, kernel)

        return rel_major_new, rel_minor_new

    def show_map(self, ax='', basemap=False):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # Draw outlines
        gpd.GeoSeries([self.rgu_oue_geom]).boundary.plot(ax=ax, color='blue' , linewidth=1.5)
        gpd.GeoSeries([self.rgu_our_geom]).boundary.plot(ax=ax, color='green', linewidth=1.5)

        # Draw axes vertexes
        rel_major, rel_minor = self.get_axes()
        gpd.GeoSeries([rel_major]).boundary.plot(ax=ax, color=self.rgu_major_axis_color , markersize=10)
        gpd.GeoSeries([rel_minor]).boundary.plot(ax=ax, color=self.rgu_minor_axis_color,     markersize=10)

        # Draw axes
        gpd.GeoSeries([rel_major]).plot(ax=ax, color=self.rgu_major_axis_color , linewidth=1.5)
        gpd.GeoSeries([rel_minor]).plot(ax=ax, color=self.rgu_minor_axis_color,     linewidth=1.5)

        # Add background with satellite view
        if basemap :
            cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery, crs=self.rgu_epsg)
        
        # Deactivate geographic coordinates on the borders of the graph
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax

    def show_profiles(self, mode='', ax='', window_size=1):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # Draw profile
        rel_major, rel_minor = self.get_relative_topo_profiles(window_size=window_size)
        if mode == 'major':
            ax.plot(rel_major[5:-5], linewidth=1.2, color=self.rgu_major_axis_color)

        elif mode == 'minor':
            ax.plot(rel_minor[5:-5], linewidth=1.2, color=self.rgu_minor_axis_color)

        # ax.set_aspect(aspect=5)

    def show_pannel(self):

        # Create an empty figure
        fig = plt.figure(figsize=(10, 6))

        # Create a 2x2 grid
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        # Add a first figure 
        # Note : [:,0] in numpy means 'the first column'
        ax1 = fig.add_subplot(gs[:,0])
        ax1.set_title(f'{self.rgu_id} rock glacier')

        # Add an second figure, first row second column
        ax2 = fig.add_subplot(gs[0,1])
        ax2.set_title(f'{self.rgu_id} topographic profile 1')
        ax2.set_ylabel('relative elevation (meters)')

        # Third figure, second row second column
        ax3 = fig.add_subplot(gs[1,1])
        ax3.set_title(f'{self.rgu_id} topographic profile 2')
        ax3.set_ylabel('relative elevation (meters)')

        # Send the result of self.show_map to ax1
        self.show_map(ax=ax1)

        # Same for ax2 & ax3, with the major & minor profiles
        self.show_profiles(ax=ax2, mode='major', window_size=5)
        self.show_profiles(ax=ax3, mode='minor', window_size=5)

        # What's for ?
        plt.tight_layout()
        
    def write_rgik_pm(self):
        """
        Return a pd.Series ready to be saved in a geopackage or a shapefile, 
        containing the infos of the Rock Glacier Unit Primary Markers
        """            

        pm_row = pd.Series(dtype='object')
        pm_row['PrimaryID'] = self.rgik_id
        pm_row['Morpho.'] = self.rgik_morpho
        pm_row['Upsl.Con.'] = self.rgik_upslcon
        pm_row['Upsl.Cur.'] = self.rgik_upslcur
        pm_row['Complet.'] = self.rgik_complet
        pm_row['Acti.Cl.'] = self.rgik_acticl
        pm_row['Destabili.'] = self.rgik_destab
        pm_row['Comment'] = self.rgik_comment
        pm_row['WorkingID'] = self.rgik_workingID
        pm_row['Assoc.RGS'] = self.rgik_assoc_rgs
        pm_row['Acti.Ass.'] = self.rgik_acti_ass
        pm_row['Kin.Att.'] = self.rgik_kin_att
        pm_row['Rel.Kin.'] = self.rgik_kin_rel
        pm_row['Kin.Period'] = self.rgik_kin_period
        pm_row['geometry'] = self.rgu_pm_geom
        return pm_row

    def write_outlines(self, model='rgik'):
        """
        Return 2 pd.Series ready to be saved in a geopackage or a shapefile, 
        containing the infos of the Rock Glacier Unit outlines 
        """

        match model.lower():

            # Write the outlines with the RGIK conceptual model attributes
            case 'rgik':
                oue_row = pd.Series(dtype='object')
                oue_row['Out.Type']   = 'Extended'
                oue_row['PrimaryID'] = self.rgik_id
                oue_row['WorkingID'] = self.rgik_workingID
                oue_row['RelFr']      = self.rgik_oue_relFr
                oue_row['RelLeftLM']  = self.rgik_oue_relLeftLM
                oue_row['RelRightLM'] = self.rgik_oue_relRightLM
                oue_row['RelUpsCon']  = self.rgik_oue_relUpsCon
                oue_row['RelIndex']   = self.rgik_oue_RelIndex
                oue_row['Comment']    = self.rgik_oue_Comment
                oue_row['geometry']   = self.rgu_oue_geom

                our_row = pd.Series(dtype='object')
                our_row['Out.Type']   = 'Restricted'
                our_row['PrimaryID'] = self.rgik_id
                our_row['WorkingID'] = self.rgik_workingID
                our_row['RelFr']      = self.rgik_our_relFr
                our_row['RelLeftLM']  = self.rgik_our_relLeftLM
                our_row['RelRightLM'] = self.rgik_our_relRightLM
                our_row['RelUpsCon']  = self.rgik_our_relUpsCon
                our_row['RelIndex']   = self.rgik_our_RelIndex
                our_row['Comment']    = self.rgik_our_Comment
                our_row['geometry']   = self.rgu_our_geom

            # TODO : Create a RoDynAlps outlines conceptual model attributes - 
            # In clear, it is what we want for the project in our shapefiles
            case 'rodynalps':
                pass

        return oue_row, our_row

def read_rgik_feature(pm, oux='', oue='', rgu_epsg=0):

    # Primary Markers Data
    rgu = RockGlacierUnit(
            rgik_id         = pm['PrimaryID'],
            rgik_morpho     = pm['Morpho.'],
            rgik_upslcon    = pm['Upsl.Con.'],
            rgik_upslcur    = pm['Upsl.Cur.'],
            rgik_complet    = pm['Complet.'],
            rgik_acticl     = pm['Acti.Cl.'],
            rgik_destab     = pm['Destabili.'],
            rgik_comment    = pm['Comment'],
            rgik_workingID  = pm['WorkingID'],
            rgik_assoc_rgs  = pm['Assoc.RGS'],
            rgik_acti_ass   = pm['Acti.Ass.'],
            rgik_kin_att    = pm['Kin.Att.'],
            rgik_kin_rel    = pm['Rel.Kin.'],
            rgik_kin_period = pm['Kin.Period'],
            rgu_pm_geom     = pm['geometry'],
            rgu_id          = pm['WorkingID'],
            rgu_epsg        = rgu_epsg)

    for out_ft in [oux, oue]:
        if type(out_ft) != str:
            rgu.read_outline(out_ft)

    return rgu