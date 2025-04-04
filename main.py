import geopandas as gpd
import os
import warnings
from dotenv import load_dotenv
from coord import Coord
from road_classifier import RoadClassifier
from road_attribution_correction import RoadAttributionCorrection
from road_segment_corrector import RoadSegmentCorrector

# Load environment variables
load_dotenv()


result = RoadAttributionCorrection('23599610').process()
print("\n"*20)
print("===========\nROAD ATTRIBUTION CORRECTION\n==========")
print(result)
print("\n"*20)

result = RoadSegmentCorrector("23599610").process()
print("\n"*20)
print("===========\nROAD ATTRIBUTION CORRECTION\n==========")
print(result)
print("\n"*20)
