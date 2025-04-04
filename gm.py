import requests
import io
from PIL import Image

def get_satellite_image(api_key, lat, lng, zoom=18, size="600x600"):
    """
    Fetch a satellite image for the specified latitude and longitude using Google Maps API.
    
    Parameters:
        api_key (str): Your Google Maps API key
        lat (float): Latitude
        lng (float): Longitude
        zoom (int): Zoom level (1-21, where 21 is the closest)
        size (str): Image dimensions in pixels (e.g., "600x600")
        
    Returns:
        PIL Image object
    """
    # Construct the URL for the Google Maps Static API
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": size,
        "maptype": "satellite",
        "key": api_key
    }
    
    # Make the request
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        # Convert the response content to an image
        image = Image.open(io.BytesIO(response.content))
        return image
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def save_image(image, filename="satellite_image.jpg"):
    """Save the image to a file"""
    if image:
        image.save(filename)
        print(f"Image saved as {filename}")
