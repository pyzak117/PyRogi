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
        ):

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
        self.rgu_major_axis_color = 'orange'
        self.rgu_minor_axis_color = 'red'
        self.rgu_epsg = rgu_epsg

    def write_rgik_pm(self):
        """
        Convert RockGlacierUnit into a RGIK Primary Marker feature
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

    def __repr__(self):
        return self.rgik_id

    def write_rgik_outline(self, mode):
        """
        Convert RockGlacierUnit into a RGIK Outline feature
        mode (str) : ['restricted', 'r', 'extended', 'e']
        """

        ou_row = pd.Series(dtype='object')
        ou_row['PrimaryID'] = self.rgik_id
        ou_row['WorkingID'] = self.rgik_workingID

        match mode.lower()[0]:
            case 'e':
                ou_row['Out.Type']   = 'Extended'
                ou_row['RelFr']      = self.rgik_oue_relFr
                ou_row['RelLeftLM']  = self.rgik_oue_relLeftLM
                ou_row['RelRightLM'] = self.rgik_oue_relRightLM
                ou_row['RelUpsCon']  = self.rgik_oue_relUpsCon
                ou_row['RelIndex']   = self.rgik_oue_RelIndex
                ou_row['Comment']    = self.rgik_oue_Comment
                ou_row['geometry']   = self.rgu_oue_geom

            case 'r':
                ou_row['Out.Type']   = 'Restricted'
                ou_row['RelFr']      = self.rgik_our_relFr
                ou_row['RelLeftLM']  = self.rgik_our_relLeftLM
                ou_row['RelRightLM'] = self.rgik_our_relRightLM
                ou_row['RelUpsCon']  = self.rgik_our_relUpsCon
                ou_row['RelIndex']   = self.rgik_our_RelIndex
                ou_row['Comment']    = self.rgik_our_Comment
                ou_row['geometry']   = self.rgu_our_geom

        return ou_row

    def read_outline(self, outline_feature):

        mode = outline_feature['Out.Type']

        if mode.lower()[0] == 'e':
            self.rgik_oue_relFr      = outline_feature['RelFr']
            self.rgik_oue_relLeftLM  = outline_feature['RelLeftLM']
            self.rgik_oue_relRightLM = outline_feature['RelRightLM']
            self.rgik_oue_relUpsCon  = outline_feature['RelUpsCon']
            self.rgik_oue_RelIndex   = outline_feature['RelIndex']
            self.rgik_oue_Comment    = outline_feature['Comment']
            self.rgu_oue_geom        = outline_feature['geometry']

        elif mode.lower()[0] == 'r':
            self.rgik_our_relFr      = outline_feature['RelFr']
            self.rgik_our_relLeftLM  = outline_feature['RelLeftLM']
            self.rgik_our_relRightLM = outline_feature['RelRightLM']
            self.rgik_our_relUpsCon  = outline_feature['RelUpsCon']
            self.rgik_our_RelIndex   = outline_feature['RelIndex']
            self.rgik_our_Comment    = outline_feature['Comment']
            self.rgu_our_geom        = outline_feature['geometry']
    
        return None

    def search_outline(self, outlines_layer):
        
        # Track the outlines containing the marker feature
        outlines_containers = outlines_layer[outlines_layer.contains(self.rgu_pm_geom)]

        # Check if it's not empty
        if len(outlines_containers) == 0:
            return False

        else:
            try:
                self.read_outline(outlines_containers.iloc[0])
                self.read_outline(outlines_containers.iloc[1])
                return True
            except IndexError:
                print(f'one of the outlines is missing for {self.rgik_id}')
                return False

    def get_topo_profiles(self):
        dem = self.get_dem()
        major_profile = np.array(dem.inspectGeoLine(self.rgu_major_axis))
        minor_profile = np.array(dem.inspectGeoLine(self.rgu_minor_axis))
        return major_profile, minor_profile
        
    def get_axis(self):

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

    def get_dem(self, dem_metamap='', epsg=4326, layername='', nRes=0):
        """
        dem_metamap : path to a directory containing dems,
                      path to a geopackage describing the dems maps,
                      path to a shapefile describing the dems maps,
                      a geodataframe describing the dems maps

        epsg : the epsg of the rasters - you have to know it before
        """

        # If the job is already done
        try:
            type(self.rgu_dem) == rt.geoim.Geoim
            return self.rgu_dem
        except AttributeError:
            pass
        
        # Here we check the existence of the RGU outlines - at least the extended one
        assert type(self.rgu_oue_geom) != '', 'the rock glacier outlines are not loaded'

        # First cases
        if type(dem_metamap) == str:
    
            assert dem_metamap != '', 'we need metadata about the dems to open them'
                
            # Here we have to open the geopackage with the layername
            if dem_metamap.endswith('.gpkg'):
                if layername != '':
                    dem_metamap = gpd.read_file(dem_metamap, layer=layername)
                else:
                    dem_metamap = gpd.read_file(dem_metamap)

            # Here we have to open this shapefile
            if dem_metamap.endswith('.shp'):
                dem_metamap = gpd.read_file(dem_metamap)

            # Here we have to build the map of the DEM rasters directory
            else:
                dem_metamap = rt.getRastermap(dem_metamap)

        # Here we should have a metamap so we can start to work
        assert type(dem_metamap) == gpd.GeoDataFrame, 'invalid dem_metamap'

        # Here we find the tracks touching the rock glacier outlines
        rgu_demtracks = dem_metamap[dem_metamap.intersects(self.rgu_oue_geom)==True]

        # Here we crop each raster on the extent of the outlines
        if nRes == 0:
            rgu_dems = [rt.Open(demtrack.filepath, load_pixels=False, geoExtent=self.rgu_oue_geom) for demtrack in rgu_demtracks.iloc]
        else:
            rgu_dems = [rt.Open(demtrack.filepath, load_pixels=False, geoExtent=self.rgu_oue_geom, nRes=nRes) for demtrack in rgu_demtracks.iloc]

        # Here we merge the tracks
        rgu_dems_merged = rt.merge(rgu_dems)

        # Here we load the data
        rgu_dem = rt.Open(rgu_dems_merged, load_pixels=True)

        # Here we mask it on the rock glaciers outlines
        # rgu_dem.maskFromVector(self.rgu_oue_geom, epsg=epsg)

        self.rgu_dem = rgu_dem
        return rgu_dem

    def get_relative_topo_profiles(self, window_size=1):
        major, minor = self.get_topo_profiles()
        rel_major = np.array(major) - major.min()
        kernel = np.ones(window_size) / window_size
        rel_major_new = np.convolve(rel_major, kernel)
        rel_minor = np.array(minor) - minor.min()
        rel_minor_new = np.convolve(rel_minor, kernel)
        return rel_major_new, rel_minor_new

    def show_map(self, ax='', epsg=self.epsg):

        # Define an empty ax if not provided
        if ax == '':
            ax = plt.subplot()

        # Draw outlines
        gpd.GeoSeries([self.rgu_oue_geom]).boundary.plot(ax=ax, color='blue' , linewidth=1.5)
        gpd.GeoSeries([self.rgu_our_geom]).boundary.plot(ax=ax, color='green', linewidth=1.5)

        # Draw axes points
        rel_major, rel_minor = self.get_axis()
        gpd.GeoSeries([rel_major]).boundary.plot(ax=ax, color=self.rgu_major_axis_color , markersize=10)
        gpd.GeoSeries([rel_minor]).boundary.plot(ax=ax, color=self.rgu_minor_axis_color,     markersize=10)

        # Draw axes
        gpd.GeoSeries([rel_major]).plot(ax=ax, color=self.rgu_major_axis_color , linewidth=1.5)
        gpd.GeoSeries([rel_minor]).plot(ax=ax, color=self.rgu_minor_axis_color,     linewidth=1.5)

        cx.add_basemap(
            ax=ax, source=cx.providers.Esri.WorldImagery, crs=self.epsg
        )

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
            ax.set_ybound((0,150))
            ax.set_xbound((0,500))

        elif mode == 'minor':
            ax.plot(rel_minor[5:-5], linewidth=1.2, color=self.rgu_minor_axis_color)
            ax.set_ybound((0,150))
            ax.set_xbound((0,500))

        # ax.set_aspect(aspect=5)

    def show_pannel(self):
        pass

def read_rgik_feature(pm, oux='', oue=''):

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
            rgu_id          = pm['WorkingID'])

    for out_ft in [oux, oue]:
        if type(out_ft) != str:
            rgu.read_outline(out_ft)

    return rgu