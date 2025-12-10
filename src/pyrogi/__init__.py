from pathlib import Path
import json

version = 'V1 / 2025-12-10'

print(f'''-------------
pyrogi {version}
-------------''')

# Open Configuration file
with open(Path(__file__).with_name('config.json')) as f:
    config = json.load(f)

# Open datamap
with open(config['datamap_file']) as f:
    datamap = json.load(f)[0]

temp_figs_dir = config['temp_figs_dir']

from pyrogi import rgu, rogi