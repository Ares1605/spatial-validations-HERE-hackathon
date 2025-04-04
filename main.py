import geopandas as gpd
import os
import warnings
import tensorflow as tf
from dotenv import load_dotenv
from coord import Coord
from road_classifier import RoadClassifier
from road_attribution_correction import RoadAttributionCorrection

# More aggressive TensorFlow logging suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
warnings.filterwarnings('ignore')

# Disable CUDA/GPU warnings and disable XLA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Silence absl logging (used by TensorFlow internally)
import logging
from absl import logging as absl_logging
logging.root.removeHandler(absl_logging._absl_handler)
absl_logging._warn_preinit_stderr = False
logging.basicConfig(level=logging.ERROR)

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
