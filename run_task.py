from dotenv import load_dotenv
from road_attribution_correction import RoadAttributionCorrection
from road_segment_corrector import RoadSegmentCorrector
import time

# Load environment variables
load_dotenv()


time.sleep(1)
tile = '23599610'

result1 = RoadAttributionCorrection(tile).process()
result2 = RoadSegmentCorrector(tile).process()

print("\n"*20)
print(f'looking through tile: {tile}')
print(result1)
print(result2)
