import geopandas as gpd
import os
import warnings
import tensorflow as tf
from dotenv import load_dotenv
from coord import Coord
from road_classifier import RoadClassifier
from road_attribution_correction import RoadAttributionCorrection

# Load environment variables
load_dotenv()


result = RoadAttributionCorrection('23599610').process()
print("===========\nROAD ATTRIBUTION CORRECTION\n==========")
print(result)
# Get satellite image from coordinates
print("Fetching satellite image...")
content = Coord(49.192900000000002, 8.1283499999999993).get_satellite_image()

# Initialize classifier and get predictions
print("Classifying image...")
classifier = RoadClassifier()
result = classifier.classify(content)

# Print results
print("\nClassification Results:")
print(f"Predicted Class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
