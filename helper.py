import os
from config import Config 
import geopandas as gpd
from typing import List

def get_violations_data(tile: str):
    violations_path = os.path.join(Config.BASE_DATSET_DIR, f"{tile}/{tile}_validations.geojson")
    return gpd.read_file(violations_path)
def get_tiles() -> List[str]:
    path = Config.BASE_DATSET_DIR
    return [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
