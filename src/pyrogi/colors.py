from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from seaborn import color_palette
from pathlib import Path
import json

# Open the graphical chart
with open(Path(__file__).with_name('graphical_chart.json'), 'r') as f:
    gc = json.load(f)

# Define qualitative variables orders of appearance in legends and charts
cardi_pt_order = list(gc['cardi_pt'].keys())
rgu_activity_class_order = list(gc['rgu_activity_class'].keys())
rgu_activity_class_wu_order = list(filter(lambda value:  'uncertain' not in value, rgu_activity_class_order))

# Define the colors associated to each qualtitative variables
cardi_pt_colors = gc['cardi_pt']
rgu_activity_class_colors = gc['rgu_activity_class']
rgu_activity_class_wu_colors = {target_value: rgu_activity_class_colors[target_value] for target_value in rgu_activity_class_wu_order}

# Creates palette object from that
cardi_pt_palette = color_palette(cardi_pt_colors.values())
rgu_activity_class_palette = color_palette(rgu_activity_class_colors.values())
rgu_activity_class_wu_palette = color_palette(rgu_activity_class_wu_colors.values())

# Create cmap objects
cardi_pt_cmap = ListedColormap(cardi_pt_palette)
rgu_activity_class_cmap = ListedColormap(rgu_activity_class_palette)
rgu_activity_class_wu_cmap = ListedColormap(rgu_activity_class_wu_palette)

# ---

# Process the semi-quantitative values like rgu_kin_att
rgu_kin_att_order = gc['rgu_kin_att_order']
rgu_kin_att_cmap_label = gc['rgu_kin_att']

# Create a cmap and sequencialize it with the number of values + convert it in list
foo = list(color_palette(rgu_kin_att_cmap_label, n_colors=len(rgu_kin_att_order)))

# Modify the last one (undefined) in black
foo[-1] = (0,0,0) # RGB norm

# Make a dict colors from it
rgu_kin_att_colors = dict(zip(rgu_kin_att_order, tuple(foo)))

# Palette object
rgu_kin_att_palette = color_palette(tuple(foo))

# Cmap object
rgu_kin_att_cmap = ListedColormap(rgu_kin_att_palette)