import geopandas as gpd
import os
import pandas as pd
from coord import Coord
from road_classifier import RoadClassifier
from typing import Dict, Tuple, List, Any
from config import Config

class RoadAttributionCorrection:
    def __init__(self, tile: str, startAtViolationI: int = 0):
        """
        Initialize the road attribution correction tool.
        
        Args:
            tile: Name of the tile you're targetting
            startAtViolationI: Index of the violation to start processing from
        """
        self.tile = tile
        self.startAtViolationI = startAtViolationI
        self.topology_data = None
        self.violations_data = None
        self.classifier = RoadClassifier()

    def load_data(self):
        """Load all required datasets"""
        # Load topology data
        topology_path = os.path.join(Config.BASE_DATSET_DIR, f"{self.tile}/{self.tile}_full_topology_data.geojson")
        self.topology_data = gpd.read_file(topology_path)
        
        # Load violations data
        violations_path = os.path.join(Config.BASE_DATSET_DIR, f"{self.tile}/{self.tile}_validations.geojson")
        self.violations_data = gpd.read_file(violations_path)
        
    def extract_topology_id_from_error(self, error_message: str) -> str:
        """
        Extract topology ID from violation error message.
        
        Args:
            error_message: The error message from the violation
            
        Returns:
            Topology ID as a string
        """
        # Extract the topology ID from error message
        if "associated to a Topology" in error_message:
            start_idx = error_message.find("associated to a Topology") + len("associated to a Topology ")
            end_idx = error_message.find(" that has a range")
            return error_message[start_idx:end_idx]
        return None

    def extract_coordinates_from_error(self, error_message: str) -> Tuple[float, float]:
        """
        Extract coordinates from violation error message.
        
        Args:
            error_message: The error message from the violation
            
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            # Parse for coordinates with format "at Lat X.XXX Lon Y.YYY"
            if "at Lat" in error_message and "Lon" in error_message:
                lat_start = error_message.find("at Lat") + len("at Lat ")
                lat_end = error_message.find(" Lon")
                lon_start = lat_end + len(" Lon")
                lon_end = error_message.find(" is associated")
                
                if lon_end == -1:  # If "is associated" not found, try looking for the next part
                    lon_end = error_message.find(" that has")
                
                if lon_end == -1:  # If still not found, take the rest of the string
                    lon_end = len(error_message)
                
                lat = float(error_message[lat_start:lat_end].strip())
                lon = float(error_message[lon_start:lon_end].strip())
                return (lat, lon)
            else:
                return None
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            print(f"Error message: {error_message}")
            return None

    def get_topology_pedestrian_access(self, topology_id: str) -> bool:
        """
        Determine if a topology allows pedestrian access.
        
        Args:
            topology_id: ID of the topology to check
            
        Returns:
            Boolean indicating if pedestrian access is allowed
        """
        # Find matching topology
        matching_topologies = None
        
        if 'id' in self.topology_data.columns:
            matching_topologies = self.topology_data[self.topology_data['id'] == topology_id]
        elif 'properties' in self.topology_data.columns:
            matching_topologies = self.topology_data[
                self.topology_data['properties'].apply(lambda x: x.get('id') == topology_id)
            ]
        else:
            # Try string matching in any column
            matching_topologies = self.topology_data[self.topology_data.apply(
                lambda row: any(topology_id in str(val) for val in row.values), axis=1
            )]
            
        if matching_topologies is None or len(matching_topologies) == 0:
            return None
        
        # Extract the first matching topology
        topology = matching_topologies.iloc[0]
        
        # Check pedestrian access in different possible data structures
        try:
            # Check if we have a nested properties field or if columns are already unpacked
            if 'properties' in topology:
                props = topology['properties']
                if isinstance(props, str):
                    # If properties is a string (serialized JSON), try to parse it
                    import json
                    try:
                        props = json.loads(props)
                    except json.JSONDecodeError:
                        print("Warning: Could not parse properties JSON string")
                        props = {}
            else:
                # Assume properties are already unpacked into columns
                props = topology
            
            # Try to find pedestrian access information
            if 'accessCharacteristics' in props:
                access_chars = props['accessCharacteristics']
                if isinstance(access_chars, list) and len(access_chars) > 0:
                    return access_chars[0].get('pedestrian', False)
            elif 'pedestrian' in props:
                return props['pedestrian']
            
            # If we reach here, we couldn't find pedestrian access info
            return False
            
        except Exception as e:
            print(f"Error getting pedestrian access: {e}")
            return None

    def compare_classification_with_topology(self, classification_result: Dict, pedestrian_access: bool) -> bool:
        """
        Compare the classification result with the topology's pedestrian access.
        
        Args:
            classification_result: Dictionary containing classification results
            pedestrian_access: Boolean indicating if pedestrian access is allowed
            
        Returns:
            Boolean indicating if a correction is needed
        """
        # Get the predicted class
        predicted_class = classification_result['class_name']
        confidence = classification_result['confidence']
        
        # Check if the classifier is confident enough (threshold can be adjusted)
        if confidence < 0.7:
            return False
        
        # Determine if correction is needed based on the predicted class
        if "Pedestrian" in predicted_class and not pedestrian_access:
            # Classifier says pedestrian, but topology says no pedestrian
            return True
        elif "Non-Pedestrian" in predicted_class and pedestrian_access:
            # Classifier says non-pedestrian, but topology says pedestrian
            return True
        
        # No correction needed
        return False

    def process(self) -> Dict[str, Dict[str, Any]]:
        """
        Process violations and identify topology corrections based on AI classification.
        
        Returns:
            Dictionary with violation IDs as keys and correction details as values
        """
        self.load_data()
        
        # Use a dictionary to store corrections keyed by violation ID
        corrections_dict = {}
        
        # Get violations data as a list to use the startAtViolationI parameter
        violations_list = list(self.violations_data.iterrows())
        
        # Process violations
        for idx, violation in violations_list[self.startAtViolationI:]:
            # Access violation information
            violation_id = violation.get('Violation ID', f"violation_{idx}")
            error_message = violation.get('Error Message', '')
            
            # Extract coordinates and topology ID
            coordinates = self.extract_coordinates_from_error(error_message)
            topology_id = self.extract_topology_id_from_error(error_message)
            
            if coordinates is None or topology_id is None:
                continue
            
            lat, lon = coordinates
            
            # Get satellite image for the coordinates
            try:
                content = Coord(lat, lon).get_satellite_image()
            except Exception:
                continue
            
            # Classify the image
            try:
                classification_result = self.classifier.classify(content)
            except Exception:
                continue
            
            # Get current pedestrian access status from topology
            pedestrian_access = self.get_topology_pedestrian_access(topology_id)
            
            if pedestrian_access is None:
                continue
            
            # Compare classification with topology
            correction_needed = self.compare_classification_with_topology(classification_result, pedestrian_access)
            
            # If correction is needed, add to results dictionary
            if correction_needed:
                corrections_dict[violation_id] = {
                    'topology_id': topology_id,
                    'coordinates': (lat, lon),
                    'current_pedestrian_access': pedestrian_access,
                    'recommended_pedestrian_access': not pedestrian_access,
                    'classification_result': classification_result['class_name'],
                    'confidence': classification_result['confidence']
                }
        
        return corrections_dict


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Road Attribution Correction Tool')
    parser.add_argument('--start', type=int, default=0, help='Index of the violation to start processing from')
    parser.add_argument('--tile', type=str, default="23599610", 
                        help='Directory containing the dataset files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize and run the processor with the specified parameters
    processor = RoadAttributionCorrection(args.tile, startAtViolationI=args.start)
    results = processor.process()
    
    # Output summary to console
    if len(results) == 0:
        print("No corrections recommended.")
    else:
        print("\n--- FINAL CORRECTION RESULTS ---")
        for violation_id, correction in results.items():
            print(f"CORRECT: Violation={violation_id}, Topology={correction['topology_id']}, " 
                  f"Pedestrian={correction['current_pedestrian_access']} → {correction['recommended_pedestrian_access']}")
