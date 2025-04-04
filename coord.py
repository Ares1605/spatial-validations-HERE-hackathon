import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlencode


class Coord:
    """
    A class representing geographical coordinates (latitude and longitude)
    with functionality to retrieve satellite imagery from Google Maps API.
    """
    
    def __init__(self, lat: float, lng: float):
        """
        Initialize a coordinate with latitude and longitude.
        
        Args:
            lat (float): Latitude value
            lng (float): Longitude value
        """
        self.lat = lat
        self.lng = lng
        
        # Load API key from .env file
        load_dotenv()
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    
    def __str__(self):
        """String representation of coordinates."""
        return f"({self.lat}, {self.lng})"
    
    def get_satellite_image(self, zoom=18, size=(640, 640), maptype='satellite'):
        """
        Retrieve a satellite image for these coordinates from Google Maps Static API.
        
        Args:
            zoom (int): Zoom level (0-21, with 21 being the most detailed)
            size (tuple): Image dimensions as (width, height) in pixels
            maptype (str): Map type ('satellite', 'hybrid', 'terrain', 'roadmap')
            
        Returns:
            bytes: Binary image data suitable for direct use with image classifiers
        """
        # Construct the URL for the Google Maps Static API
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        
        params = {
            'center': f"{self.lat},{self.lng}",
            'zoom': zoom,
            'size': f"{size[0]}x{size[1]}",
            'maptype': maptype,
            'key': self.api_key,
            'format': 'png'  # Using PNG for better image quality
        }
        
        url = f"{base_url}?{urlencode(params)}"
        
        # Send the request to Google Maps API
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code != 200:
            error_msg = f"Failed to fetch satellite image: {response.status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details.get('error_message', 'Unknown error')}"
            except:
                pass
            raise Exception(error_msg)
        
        # Return the binary image data
        return response.content
    
    def get_satellite_image_with_marker(self, zoom=18, size=(640, 640), maptype='satellite', marker_color='red'):
        """
        Retrieve a satellite image with a marker at these coordinates.
        
        Args:
            zoom (int): Zoom level (0-21, with 21 being the most detailed)
            size (tuple): Image dimensions as (width, height) in pixels
            maptype (str): Map type ('satellite', 'hybrid', 'terrain', 'roadmap')
            marker_color (str): Color for the marker ('red', 'green', 'blue', etc.)
            
        Returns:
            bytes: Binary image data suitable for direct use with image classifiers
        """
        # Construct the URL for the Google Maps Static API
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        
        params = {
            'center': f"{self.lat},{self.lng}",
            'zoom': zoom,
            'size': f"{size[0]}x{size[1]}",
            'maptype': maptype,
            'markers': f"color:{marker_color}|{self.lat},{self.lng}",
            'key': self.api_key,
            'format': 'png'  # Using PNG for better image quality
        }
        
        url = f"{base_url}?{urlencode(params)}"
        
        # Send the request to Google Maps API
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code != 200:
            error_msg = f"Failed to fetch satellite image: {response.status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details.get('error_message', 'Unknown error')}"
            except:
                pass
            raise Exception(error_msg)
        
        # Return the binary image data
        return response.content


# Example usage
if __name__ == "__main__":
    import argparse
    from PIL import Image
    from io import BytesIO
    
    parser = argparse.ArgumentParser(description='Retrieve a satellite image for given coordinates')
    parser.add_argument('--lat', required=True, type=float, help='Latitude')
    parser.add_argument('--lng', required=True, type=float, help='Longitude')
    parser.add_argument('--zoom', type=int, default=18, help='Zoom level (0-21)')
    parser.add_argument('--width', type=int, default=640, help='Image width in pixels')
    parser.add_argument('--height', type=int, default=640, help='Image height in pixels')
    parser.add_argument('--maptype', choices=['satellite', 'hybrid', 'terrain', 'roadmap'], default='satellite', help='Map type')
    parser.add_argument('--marker', action='store_true', help='Add a marker at the coordinates')
    parser.add_argument('--output', help='Save image to this file (optional)')
    parser.add_argument('--show', action='store_true', help='Show the image (requires display)')
    
    args = parser.parse_args()
    
    try:
        # Create coordinate object
        coord = Coord(args.lat, args.lng)
        
        print(f"Retrieving satellite image for coordinates: {coord}")
        
        # Get image based on whether a marker is requested
        if args.marker:
            image_data = coord.get_satellite_image_with_marker(
                zoom=args.zoom, 
                size=(args.width, args.height), 
                maptype=args.maptype
            )
        else:
            image_data = coord.get_satellite_image(
                zoom=args.zoom, 
                size=(args.width, args.height), 
                maptype=args.maptype
            )
        
        # Show image if requested
        if args.show:
            img = Image.open(BytesIO(image_data))
            img.show()
        
        # Save image if output path provided
        if args.output:
            with open(args.output, 'wb') as f:
                f.write(image_data)
            print(f"Image saved to {args.output}")
            
        print(f"Successfully retrieved image ({len(image_data)} bytes)")
        print("This image data can be passed directly to an image classifier.")
        
    except Exception as e:
        print(f"Error: {e}")
