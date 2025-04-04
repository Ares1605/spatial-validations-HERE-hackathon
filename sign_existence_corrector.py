import pandas as pd
import random
import os
from typing import Dict, Any, Tuple, List, Union
from config import Config

class SignExistenceCorrector:
    def __init__(self, tile: str, violations_data):
        """
        Initialize the sign existence corrector.
        
        Args:
            tile: Tile ID for the data
            violations_data: DataFrame or list of violations to process
        """
        self.tile = tile
        self.violations_data = violations_data
        self.startAtViolationI = 0
        
    def load_data(self):
        """Load and prepare violations data"""
        # Handle different types of violations_data
        if isinstance(self.violations_data, list):
            # Convert list to DataFrame if needed
            if len(self.violations_data) > 0 and hasattr(self.violations_data[0], '_asdict'):
                # If it's a list of named tuples
                self.violations_data = pd.DataFrame([v._asdict() for v in self.violations_data])
            else:
                # Convert simple list to DataFrame
                self.violations_data = pd.DataFrame(self.violations_data)
        
        # Print info about the loaded data
        print(f"Processing violations data for tile: {self.tile}")
        
        # Safely print violations data columns
        if hasattr(self.violations_data, 'columns'):
            print(f"Violations data columns: {self.violations_data.columns.tolist()}")
        else:
            print(f"Violations data is not a DataFrame. Type: {type(self.violations_data)}")
        
        # Safely print violations count
        if hasattr(self.violations_data, '__len__'):
            print(f"Loaded {len(self.violations_data)} violation features")
        else:
            print("Violations data count unknown")
    
    def check_sign_existence(self, violation_id: str, sign_id: str) -> bool:
        """
        Check if a sign exists with a 5% probability.
        
        Args:
            violation_id: ID of the violation
            sign_id: ID of the sign to check
            
        Returns:
            Boolean indicating if the sign exists (5% chance True, 95% chance False)
        """
        # Use random to determine if sign exists (5% chance)
        return random.random() < 0.05
    
    def analyze_violations(self):
        """
        Analyze violations and determine sign existence.
        
        Returns:
            DataFrame with sign existence results
        """
        results = []
        
        # Get violations data as a list
        violations_list = list(self.violations_data.iterrows())
        
        # Check if startAtViolationI is valid
        if self.startAtViolationI < 0 or self.startAtViolationI >= len(violations_list):
            print(f"Warning: startAtViolationI {self.startAtViolationI} is out of range. Using 0 instead.")
            self.startAtViolationI = 0
        
        print(f"Starting analysis from violation index {self.startAtViolationI}")
        
        # Process violations
        for idx, violation in violations_list[self.startAtViolationI:]:
            try:
                # Extract violation details
                sign_id = violation['Feature ID']
                violation_id = violation.get('Violation ID', f"violation_{idx}")
                
                print(f"\nAnalyzing violation for sign {sign_id}")
                
                # Check if sign exists (random 5% chance)
                sign_exists = self.check_sign_existence(violation_id, sign_id)
                
                # If sign exists, add to results
                if sign_exists:
                    # For the 5% of signs that exist, add to results
                    results.append({
                        'violation_id': violation_id,
                        'sign_id': sign_id,
                        'sign_exists': True,
                        'confidence_score': random.uniform(0.8, 0.99)  # Random confidence score
                    })
                    
                    # Print results for this violation
                    print(f"SIGN EXISTS:")
                    print(f"  Violation ID: {violation_id}")
                    print(f"  Sign ID: {sign_id}")
                    print(f"  Confidence score: {results[-1]['confidence_score']:.4f}")
                else:
                    print(f"Sign does not exist for violation {violation_id}")
            except Exception as e:
                print(f"Error processing violation {idx}: {e}")
                continue
        
        # Create DataFrame with results
        return pd.DataFrame(results)
    
    def process(self) -> Tuple[Dict[str, Union[str, Dict[str, Any]]], List[Any]]:
        """
        Main processing function
        
        Returns:
            Tuple containing:
                - Dictionary with formatted results (sign existence data)
                - List of remaining violations that weren't corrected
        """
        self.load_data()
        results_df = self.analyze_violations()
        
        # Create a list to track uncorrected violations (signs that don't exist)
        remaining_violations = []
        
        # Get violations data as a list
        violations_list = list(self.violations_data.iterrows())
        
        # Keep track of violations that have been corrected (signs that exist)
        corrected_violation_ids = set(results_df['violation_id'].tolist() if not results_df.empty else [])
        
        # Add all violations with non-existent signs to the remaining list
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
                'sign_exists': row['sign_exists'],
                'confidence_score': row['confidence_score']
            }
        
        return {'type': 'sign-existence-correction', 'data': results_dict}, remaining_violations

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Sign Existence Verification Tool')
    parser.add_argument('--start', type=int, default=0, help='Index of the violation to start processing from')
    parser.add_argument('--tile', type=str, default="23599610", 
                        help='Tile ID for processing')
    # Parse arguments
    args = parser.parse_args()
    
    # For testing, create dummy violations data
    dummy_violations = [
        {'Violation ID': f'urn:here::here:Violation:signs-{i}', 'Feature ID': f'urn:here::here:signs:{i}', 'Error Message': 'Test error message'} 
        for i in range(1, 21)
    ]
    
    # Initialize and run the processor with the specified parameters
    processor = SignExistenceCorrector(args.tile, dummy_violations)
    results, remaining = processor.process()
    
    # Simple final output
    print("\n--- FINAL SIGN EXISTENCE RESULTS ---")
    if len(results['data']) == 0:
        print("No existing signs found.")
    else:
        for violation_id, details in results['data'].items():
            print(f"SIGN EXISTS: Violation={violation_id}, Sign={details['sign_id']}, Confidence={details['confidence_score']:.4f}")
    
    print(f"\nProcessed {len(dummy_violations)} violations")
    print(f"Found {len(results['data'])} existing signs")
    print(f"Identified {len(remaining)} non-existent signs")
    
    print("\nAnalysis complete.")
