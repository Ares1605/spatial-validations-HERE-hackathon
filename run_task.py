from dotenv import load_dotenv
from road_attribution_correction import RoadAttributionCorrection
from road_segment_corrector import RoadSegmentCorrector
import time
import helper

# Load environment variables
load_dotenv()

tile = '23599610'

violations_data = helper.get_violations_data(tile)

# result1, violations_data = RoadAttributionCorrection(tile, violations_data).process()
result2, violations_data = RoadSegmentCorrector(tile, violations_data).process()

print("\n"*20)
print(f'looking through tile: {tile}')
# print(result1)
print(result2)
