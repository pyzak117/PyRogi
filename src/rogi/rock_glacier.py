import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
import numpy as np

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
            rgu_pm_geom     = pm['geometry'])

    for out_ft in [oux, oue]:
        if type(out_ft) != str:
            rgu.read_outline(out_ft)

    return rgu