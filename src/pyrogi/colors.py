from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
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
rgu_activity_class_wu_order = list(filter(lambda value:  'uncert' not in value, rgu_activity_class_order))
upslope_con_order = list(gc['upslope_con'].keys())
status_order = list(gc['status'].keys())

# Define the colors associated to each qualtitative variables
cardi_pt_colors = gc['cardi_pt']
rgu_activity_class_colors = gc['rgu_activity_class']
rgu_activity_class_wu_colors = {target_value: rgu_activity_class_colors[target_value] for target_value in rgu_activity_class_wu_order}
upslope_con_colors = gc['upslope_con']
status_colors = gc['status']

# Creates palette object from that
cardi_pt_palette = color_palette(cardi_pt_colors.values())
rgu_activity_class_palette = color_palette(rgu_activity_class_colors.values())
rgu_activity_class_wu_palette = color_palette(rgu_activity_class_wu_colors.values())
upslope_con_palette = color_palette(upslope_con_colors.values())
status_palette = color_palette(status_colors.values())

# Create cmap objects
cardi_pt_cmap = ListedColormap(cardi_pt_palette)
rgu_activity_class_cmap = ListedColormap(rgu_activity_class_palette)
rgu_activity_class_wu_cmap = ListedColormap(rgu_activity_class_wu_palette)
upslope_con_cmap = ListedColormap(upslope_con_palette)
status_cmap = ListedColormap(status_palette)

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


def get_intensity_cmap(hex_color: str, name: str | None = None, n: int = 256):
    """
    Create a white→hex_color intensity colormap.

    Parameters
    ----------
    hex_color : str
        Target color in any Matplotlib-parseable format (e.g., "#e41a1c", "tab:blue").
    name : str | None
        Optional colormap name.
    n : int
        Number of discrete steps in the colormap.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    tgt = to_rgba(hex_color)               # (r,g,b,a) in 0–1
    start = (1.0, 1.0, 1.0, 0.0 if tgt[3] < 1 else 1.0)  # white; keep transparent start if target is translucent
    cmap_name = name or f"intensity_{hex_color.strip('#')}"
    return LinearSegmentedColormap.from_list(cmap_name, [start, tgt], N=n)

def get_intensity_cmap_transparent(hex_color: str, name: str | None = None, n: int = 256):
    r, g, b, a = to_rgba(hex_color)
    start = (1.0, 1.0, 1.0, 0.0)   # blanc transparent
    end   = (r, g, b, 1.0)         # couleur opaque
    cmap_name = name or f"intensity_{hex_color.strip('#')}"
    return LinearSegmentedColormap.from_list(cmap_name, [start, end], N=n)