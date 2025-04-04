import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os
from typing import Dict, Any, Tuple, List, Union
from config import Config

class RoadSegmentCorrector:
    def __init__(self, tile: str, violations_data):
        """
        Initialize the road sign reassignment solution.
        
        Args:
            base_dir: Base directory containing the dataset files
            startAtViolationI: Index of the violation to start processing from (default: 0)
        """
        self.tile = tile
        self.topology_data = None
        self.signs_data = None
        self.violations_data = None
        self.startAtViolationI = 0
        self.violations_data = violations_data
        
    def load_data(self):
        """Load all required datasets"""
        # Load topology data
        topology_path = os.path.join(Config.BASE_DATSET_DIR, f"{self.tile}/{self.tile}_full_topology_data.geojson")
        self.topology_data = gpd.read_file(topology_path)
        
        # Load signs data
        signs_path = os.path.join(Config.BASE_DATSET_DIR, f"{self.tile}/{self.tile}_signs.geojson")
        self.signs_data = gpd.read_file(signs_path)
        
        # Handle different types of violations_data
        if isinstance(self.violations_data, list):
            # Convert list to DataFrame if needed
            if len(self.violations_data) > 0 and hasattr(self.violations_data[0], '_asdict'):
                # If it's a list of named tuples
                self.violations_data = pd.DataFrame([v._asdict() for v in self.violations_data])
            else:
                # Convert simple list to DataFrame
                self.violations_data = pd.DataFrame(self.violations_data)
        
        # Print columns to help with debugging
        print(f"Topology data columns: {self.topology_data.columns.tolist()}")
        print(f"Signs data columns: {self.signs_data.columns.tolist()}")
        
        # Safely print violations data columns
        if hasattr(self.violations_data, 'columns'):
            print(f"Violations data columns: {self.violations_data.columns.tolist()}")
        else:
            print(f"Violations data is not a DataFrame. Type: {type(self.violations_data)}")
        
        print(f"Loaded {len(self.topology_data)} topology features")
        print(f"Loaded {len(self.signs_data)} sign features")
        
        # Safely print violations count
        if hasattr(self.violations_data, '__len__'):
            print(f"Loaded {len(self.violations_data)} violation features")
        else:
            print("Violations data count unknown")
        
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
    
    def find_nearby_topologies(self, sign_point: Point, max_distance_m: float = 20) -> gpd.GeoDataFrame:
        """
        Find all topology segments within a specified distance of a sign.
        
        Args:
            sign_point: Point geometry of the sign
            max_distance_m: Maximum distance in meters to search
            
        Returns:
            GeoDataFrame of nearby topology segments
        """
        # Create a buffer around the sign point
        # Note: We need to convert the distance to degrees for the buffer
        # This is a rough approximation - for more precision, we should use a projected CRS
        # 1 degree is approximately 111,000 meters at the equator
        distance_degrees = max_distance_m / 111000
        buffer = sign_point.buffer(distance_degrees)
        
        # Find all topologies that intersect with the buffer
        nearby_topologies = self.topology_data[self.topology_data.geometry.intersects(buffer)]
        
        # Calculate actual distances
        nearby_topologies['distance_to_sign'] = nearby_topologies.geometry.apply(
            lambda geom: sign_point.distance(geom) * 111000  # Convert to approximate meters
        )
        
        # Filter by actual distance
        nearby_topologies = nearby_topologies[nearby_topologies['distance_to_sign'] <= max_distance_m]
        
        return nearby_topologies
    
    def extract_topology_attributes(self, topology: pd.Series) -> Dict:
        """
        Extract key attributes from a topology feature for comparison.
        
        Args:
            topology: Pandas Series containing a topology feature
            
        Returns:
            Dictionary of key attributes
        """
        attributes = {}
        
        try:
            # Check if we have a nested properties field or if columns are already unpacked
            if 'properties' in topology:
                # Extract properties we want to compare
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
            
            # Basic attributes
            # Try both direct access and get() method depending on what props is
            if isinstance(props, dict):
                attributes['id'] = props.get('id', '')
            else:
                attributes['id'] = props['id'] if 'id' in props else ''
            
            # Access characteristics - handle different possible structures
            if isinstance(props, dict) and 'accessCharacteristics' in props:
                access_chars = props.get('accessCharacteristics', [{}])[0] if props.get('accessCharacteristics') else {}
            elif 'accessCharacteristics' in topology:
                access_chars = topology['accessCharacteristics']
                if isinstance(access_chars, str):
                    import json
                    try:
                        access_chars = json.loads(access_chars)[0] if json.loads(access_chars) else {}
                    except:
                        access_chars = {}
            elif 'pedestrian' in topology:
                # Directly use unpacked columns
                attributes['pedestrian'] = topology['pedestrian'] if 'pedestrian' in topology else False
                attributes['auto'] = topology['auto'] if 'auto' in topology else False
                attributes['truck'] = topology['truck'] if 'truck' in topology else False
                
                # Skip the rest of the access characteristics processing
                access_chars = None
            else:
                access_chars = {}
                
            # Process access characteristics if we have them
            if access_chars is not None:
                if isinstance(access_chars, dict):
                    attributes['pedestrian'] = access_chars.get('pedestrian', False)
                    attributes['auto'] = access_chars.get('auto', False) 
                    attributes['truck'] = access_chars.get('truck', False)
                else:
                    # Default values if we can't extract them
                    attributes['pedestrian'] = False
                    attributes['auto'] = True  # Assume auto access by default
                    attributes['truck'] = True  # Assume truck access by default
            
            # Functional class - handle different possible structures
            if isinstance(props, dict) and 'functionalClass' in props:
                func_class = props.get('functionalClass', [{}])[0] if props.get('functionalClass') else {}
            elif 'functionalClass' in topology:
                func_class = topology['functionalClass']
                if isinstance(func_class, str):
                    import json
                    try:
                        func_class = json.loads(func_class)[0] if json.loads(func_class) else {}
                    except:
                        func_class = {}
            else:
                func_class = {}
                
            if isinstance(func_class, dict):
                attributes['functionalClass'] = func_class.get('value', 0) if isinstance(func_class, dict) else 0
            else:
                attributes['functionalClass'] = 0
            
            # Road characteristics - handle different possible structures
            if isinstance(props, dict) and 'topologyCharacteristics' in props:
                topo_chars = props.get('topologyCharacteristics', {})
            elif 'topologyCharacteristics' in topology:
                topo_chars = topology['topologyCharacteristics']
                if isinstance(topo_chars, str):
                    import json
                    try:
                        topo_chars = json.loads(topo_chars)
                    except:
                        topo_chars = {}
            elif 'isMotorway' in topology:
                # Direct access to motorway flag
                motorway_value = topology['isMotorway']
                if isinstance(motorway_value, list) and len(motorway_value) > 0:
                    attributes['isMotorway'] = motorway_value[0].get('value', False) if isinstance(motorway_value[0], dict) else motorway_value[0]
                else:
                    attributes['isMotorway'] = motorway_value if isinstance(motorway_value, bool) else False
                    
                # Skip the rest of the motorway processing
                topo_chars = None
            else:
                topo_chars = {}
            
            # Process topology characteristics if we have them
            if topo_chars is not None:
                # Check if isMotorway exists and extract the first value if it's a list
                is_motorway_list = topo_chars.get('isMotorway', [])
                if is_motorway_list and len(is_motorway_list) > 0:
                    attributes['isMotorway'] = is_motorway_list[0].get('value', False) if isinstance(is_motorway_list[0], dict) else False
                else:
                    attributes['isMotorway'] = False
            
            # Speed limit - handle different possible structures
            if isinstance(props, dict) and 'speedLimit' in props:
                speed_limits = props.get('speedLimit', [])
            elif 'speedLimit' in topology:
                speed_limits = topology['speedLimit']
                if isinstance(speed_limits, str):
                    import json
                    try:
                        speed_limits = json.loads(speed_limits)
                    except:
                        speed_limits = []
            else:
                speed_limits = []
                
            if speed_limits and len(speed_limits) > 0:
                attributes['speedLimit'] = speed_limits[0].get('valueKph', 0) if isinstance(speed_limits[0], dict) else 0
            else:
                attributes['speedLimit'] = 0
                
        except Exception as e:
            print(f"Error in extract_topology_attributes: {e}")
            # Set default values
            attributes = {
                'id': str(topology.name) if hasattr(topology, 'name') else '',
                'pedestrian': False,
                'auto': True,
                'truck': True,
                'functionalClass': 0,
                'isMotorway': False,
                'speedLimit': 0
            }
            
        return attributes
    
    def compare_topologies(self, sign_type: str, current_topology: Dict, candidate_topology: Dict) -> float:
        """
        Compare two topologies and return a similarity score based on relevant attributes.
        
        Args:
            sign_type: Type of the sign
            current_topology: Attribute dictionary for the current topology
            candidate_topology: Attribute dictionary for the candidate topology
            
        Returns:
            Similarity score between 0 and 1
        """
        # Define weights for different attributes based on sign type
        weights = {
            'isMotorway': 0.4,
            'pedestrian': 0.3,
            'functionalClass': 0.2,
            'speedLimit': 0.1
        }
        
        # Adjust weights for motorway signs
        if sign_type == "MOTORWAY":
            weights['isMotorway'] = 0.5
            weights['pedestrian'] = 0.3  # Pedestrian access should be False for motorway
        
        # Calculate scores for each attribute
        scores = {}
        
        # isMotorway attribute
        scores['isMotorway'] = 1.0 if candidate_topology['isMotorway'] == True else 0.0
        
        # pedestrian attribute (for motorway signs, pedestrian should be False)
        if sign_type == "MOTORWAY":
            scores['pedestrian'] = 1.0 if candidate_topology['pedestrian'] == False else 0.0
        else:
            scores['pedestrian'] = 1.0 if candidate_topology['pedestrian'] == current_topology['pedestrian'] else 0.0
        
        # functionalClass attribute (lower is more important)
        # Score based on how close the functional class is (1 is highest road class)
        fc_diff = abs(candidate_topology['functionalClass'] - current_topology['functionalClass'])
        scores['functionalClass'] = max(0, 1 - (fc_diff / 5))  # Normalize by max difference of 5
        
        # speedLimit attribute
        # Score based on similarity in speed limit
        if current_topology['speedLimit'] > 0 and candidate_topology['speedLimit'] > 0:
            speed_ratio = min(current_topology['speedLimit'], candidate_topology['speedLimit']) / max(current_topology['speedLimit'], candidate_topology['speedLimit'])
            scores['speedLimit'] = speed_ratio
        else:
            scores['speedLimit'] = 0.5  # Neutral score if speed limits not available
        
        # Calculate weighted score
        weighted_score = sum(weights[key] * scores[key] for key in weights)
        
        return weighted_score
    
    def extract_sign_type_from_violation(self, violation: pd.Series) -> str:
        """
        Extract the sign type from a violation record.
        
        Args:
            violation: Pandas Series or dict containing a violation record
            
        Returns:
            Sign type string
        """
        # Handle both Series objects and dictionary-like objects
        if isinstance(violation, pd.Series) and 'properties' in violation:
            error_message = violation['properties']['Error Message']
        elif isinstance(violation, pd.Series) and 'Error Message' in violation:
            error_message = violation['Error Message']
        elif isinstance(violation, dict) and 'properties' in violation:
            error_message = violation['properties']['Error Message']
        else:
            print(f"Warning: Unexpected violation format: {type(violation)}")
            return "UNKNOWN"
            
        if "Motorway Sign" in error_message:
            return "MOTORWAY"
        return "UNKNOWN"
    
    def find_sign_by_id(self, sign_id: str) -> pd.Series:
        """
        Find a sign by its ID.
        
        Args:
            sign_id: ID of the sign to find
            
        Returns:
            Pandas Series containing the sign data
        """
        # Check if 'id' exists directly in the columns
        if 'id' in self.signs_data.columns:
            matching_signs = self.signs_data[self.signs_data['id'] == sign_id]
        # Check if we need to access properties.id
        elif 'properties' in self.signs_data.columns:
            matching_signs = self.signs_data[self.signs_data['properties'].apply(lambda x: x.get('id') == sign_id)]
        # If we have a full nested properties structure
        else:
            print(f"Searching for sign ID {sign_id} in nested properties...")
            # Print a sample row to debug
            if len(self.signs_data) > 0:
                sample_row = self.signs_data.iloc[0]
                print(f"Sample row columns: {sample_row.index.tolist()}")
                
            # Try direct string matching on the sign ID in any field
            matching_signs = self.signs_data[self.signs_data.apply(
                lambda row: any(sign_id in str(val) for val in row.values), axis=1
            )]
            
        if len(matching_signs) > 0:
            return matching_signs.iloc[0]
            
        print(f"Warning: Sign {sign_id} not found in signs data")
        return None
    
    def analyze_violations(self):
        """
        Analyze violations and determine optimal topology assignments.
        
        Returns:
            DataFrame with simplified violation resolution results
        """
        results = []
        
        # Get violations data as a list to use the startAtViolationI parameter
        violations_list = list(self.violations_data.iterrows())
        
        # Check if startAtViolationI is valid
        if self.startAtViolationI < 0 or self.startAtViolationI >= len(violations_list):
            print(f"Warning: startAtViolationI {self.startAtViolationI} is out of range. Using 0 instead.")
            self.startAtViolationI = 0
        
        print(f"Starting analysis from violation index {self.startAtViolationI}")
        
        # Process only violations starting from the specified index
        for idx, violation in violations_list[self.startAtViolationI:]:
            # Access fields directly from GeoDataFrame columns
            sign_id = violation['Feature ID']
            error_message = violation['Error Message']
            violation_id = violation.get('Violation ID', f"violation_{idx}")
            
            # Create a violation-like object to maintain compatibility with other methods
            violation_obj = {
                'properties': {
                    'Feature ID': sign_id,
                    'Error Message': error_message
                }
            }
            sign_type = self.extract_sign_type_from_violation(pd.Series(violation_obj))
            
            print(f"\nAnalyzing violation for sign {sign_id} (Type: {sign_type})")
            
            # Get the current topology ID from the error message
            current_topology_id = self.extract_topology_id_from_error(error_message)
            
            if not current_topology_id:
                print(f"Could not extract topology ID from error message for violation {sign_id}")
                continue
            
            # Find the sign
            sign = self.find_sign_by_id(sign_id)
            if sign is None:
                print(f"Sign {sign_id} not found in signs data")
                continue
            
            # Find nearby topologies
            nearby_topologies = self.find_nearby_topologies(sign.geometry)
            print(f"Found {len(nearby_topologies)} topologies within 20m of sign {sign_id}")
            
            if len(nearby_topologies) == 0:
                print(f"No nearby topologies found for sign {sign_id}")
                continue
            
            # Find current topology in data - adapt to column structure
            if 'id' in self.topology_data.columns:
                current_topology_data = self.topology_data[self.topology_data['id'] == current_topology_id]
            elif 'properties' in self.topology_data.columns:
                current_topology_data = self.topology_data[
                    self.topology_data['properties'].apply(lambda x: x.get('id') == current_topology_id)
                ]
            else:
                # Try to find it by string matching in any column
                current_topology_data = self.topology_data[self.topology_data.apply(
                    lambda row: any(current_topology_id in str(val) for val in row.values), axis=1
                )]
            
            if len(current_topology_data) == 0:
                print(f"Current topology {current_topology_id} not found in topology data")
                # Print sample column names to help debug
                print(f"Topology data columns: {self.topology_data.columns.tolist()[:10]}...")
                continue
                
            current_topology_data = current_topology_data.iloc[0]
            
            # Extract attributes for comparison - adapt to possibly different structure
            try:
                current_topology_attrs = self.extract_topology_attributes(current_topology_data)
            except Exception as e:
                print(f"Error extracting attributes from current topology: {e}")
                # Print some debug info
                print(f"Current topology data types: {current_topology_data.dtypes}")
                continue
            
            # Compare with all nearby topologies
            topology_scores = []
            
            for _, candidate_topology in nearby_topologies.iterrows():
                candidate_attrs = self.extract_topology_attributes(candidate_topology)
                score = self.compare_topologies(sign_type, current_topology_attrs, candidate_attrs)
                
                topology_scores.append({
                    'topology_id': candidate_attrs['id'],
                    'score': score,
                    'distance': candidate_topology['distance_to_sign'],
                    'is_current': candidate_attrs['id'] == current_topology_id
                })
            
            # Sort by score (descending)
            topology_scores.sort(key=lambda x: x['score'], reverse=True)
            
            best_topology = topology_scores[0]
            current_topology_score = next((x for x in topology_scores if x['is_current']), None)
            
            # Decision: should we reassign?
            should_reassign = (not best_topology['is_current'] and 
                              best_topology['score'] > (current_topology_score['score'] if current_topology_score else 0))
            
            # Only add to results if we recommend reassignment
            if should_reassign:
                results.append({
                    'violation_id': violation_id,
                    'sign_id': sign_id,
                    'current_topology_id': current_topology_id,
                    'new_topology_id': best_topology['topology_id'],
                    'confidence_score': best_topology['score']
                })
                
                # Print results for this violation
                print(f"REASSIGNMENT RECOMMENDED:")
                print(f"  Violation ID: {violation_id}")
                print(f"  Sign ID: {sign_id}")
                print(f"  Current topology: {current_topology_id}")
                print(f"  Recommended new topology: {best_topology['topology_id']}")
                print(f"  Confidence score: {best_topology['score']:.4f}")
            else:
                print(f"No reassignment needed for violation {violation_id}")
        
        # Create simple DataFrame with only the essential information
        return pd.DataFrame(results)
    def process(self) -> Tuple[Dict[str, Union[str, Dict[str, Any]]], List[Any]]:
        """
        Main processing function
        
        Returns:
            Tuple containing:
                - Dictionary with violation IDs as keys and reassignment details as values
                - List of remaining violations that weren't corrected
        """
        self.load_data()
        results_df = self.analyze_violations()
        
        # Create a list to track uncorrected violations
        remaining_violations = []
        
        # Get violations data as a list to use the startAtViolationI parameter
        violations_list = list(self.violations_data.iterrows())
        
        # Keep track of violations that have been corrected
        corrected_violation_ids = set(results_df['violation_id'].tolist() if not results_df.empty else [])
        
        # Add all violations that didn't get corrections to the remaining list
        for idx, violation in violations_list[self.startAtViolationI:]:
            violation_id = violation.get('Violation ID', f"violation_{idx}")
            
            if violation_id not in corrected_violation_ids:
                remaining_violations.append(violation)
        
        # Convert DataFrame to dictionary keyed by violation_id
        results_dict = {}
        for _, row in results_df.iterrows():
            violation_id = row['violation_id']
            results_dict[violation_id] = {
                'sign_id': row['sign_id'],
                'current_topology_id': row['current_topology_id'],
                'new_topology_id': row['new_topology_id'],
                'confidence_score': row['confidence_score']
            }
        
        return {'type': 'road-segment-corrector', 'data': results_dict}, remaining_violations   
if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Road Sign Reassignment Tool')
    parser.add_argument('--start', type=int, default=0, help='Index of the violation to start processing from')
    parser.add_argument('--tile', type=str, default="23599610", 
                        help='Directory containing the dataset files')
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize and run the processor with the specified parameters
    processor = RoadSegmentCorrector(args.tile, startAtViolationI=args.start)
    results = processor.process()
    
    # Simple final output for easy parsing
    print("\n--- FINAL REASSIGNMENT RESULTS ---")
    if len(results) == 0:
        print("No reassignments recommended.")
    else:
        for _, row in results.iterrows():
            print(f"REASSIGN: Violation={row['violation_id']}, Sign={row['sign_id']}, From={row['current_topology_id']}, To={row['new_topology_id']}")
    
    print("\nAnalysis complete.")
    
    print("\n--- Final Results ---")
    print(results)
