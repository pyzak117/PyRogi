import unittest
from pathlib import Path

import json

import random
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import shapely
from osgeo import gdal
import geopandas as gpd
from telenvi import raster_tools as rt
from telenvi import vector_tools as vt

from pyrogi.rock_glacier_inventory import RockGlacierInventory
from pyrogi.rock_glacier_unit import RockGlacierUnit

print("""
----------
PyRogi here
----------      
""")

# Open and load JSON file into a dictionary
with open('/home/duvanelt/datamap.json', 'r', encoding='utf-8') as file:
    datamap = json.load(file)[0]

# print(list(datamap.keys()))