import numpy as np
import keras
from functools import lru_cache
from io import BytesIO
from PIL import Image
import os
import warnings
import logging
# More aggressive TensorFlow logging suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
# Silence Python warnings
warnings.filterwarnings('ignore')

# Disable CUDA/GPU warnings and disable XLA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

logging.disable(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Disable computation placer warnings
os.environ['TF_COMPUTATION_PLACER_DISABLE_WARNING'] = '1'

class RoadClassifier:
    """
    A class to classify road images using a pre-trained Keras model
    with optimizations for faster inference.
    Now supports image streams in addition to file paths.
    """
    
    def __init__(self, model_path='keras_model.h5', labels_path='labels.txt', input_size=224):
        """
        Initialize the road classifier with model and labels
        
        Args:
            model_path (str): Path to the Keras .h5 model file
            labels_path (str): Path to the labels.txt file
            input_size (int): Input image size (both width and height)
        """
        # Set environment variables to suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
        
        # Try to load with custom objects to handle compatibility issues
        try:
            # Load the model and optimize it for inference
            self.model = keras.models.load_model(model_path)
        except (ValueError, TypeError) as e:
            try:
                # Try loading with TensorFlow's approach
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR') # or 'CRITICAL', etc.
                
                # Define custom objects that handle compatibility issues
                custom_objects = {
                    # This handles the 'groups' parameter issue in DepthwiseConv2D
                    'DepthwiseConv2D': lambda **kwargs: tf.keras.layers.DepthwiseConv2D(
                        **{k: v for k, v in kwargs.items() if k != 'groups'}
                    )
                }
                
                self.model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects,
                    compile=False  # Skip compilation to avoid optimizer issues
                )
            except Exception as e2:
                # If both methods fail, try one last approach with a compatibility wrapper
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(
                        model_path,
                        compile=False
                    )
                except Exception as e3:
                    raise RuntimeError(f"Failed to load model after multiple attempts. Last error: {e3}") from e
        
        # Enable model optimizations through TensorFlow if available
        try:
            import tensorflow as tf
            if hasattr(tf, 'config'):
                # Enable mixed precision for faster inference on compatible hardware
                tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
                
                # Enable XLA acceleration if available
                tf.config.optimizer.set_jit(True)
        except (ImportError, AttributeError):
            pass
            
        # Load labels if available
        try:
            self.labels = self._load_labels(labels_path)
        except FileNotFoundError:
            # Default to binary classification if labels file not found
            self.labels = ["no_road", "road"]
            
        # Try to convert model to a more optimized format if tf.lite is available
        try:
            import tensorflow as tf
            if hasattr(tf, 'lite') and hasattr(tf.lite, 'TFLiteConverter'):
                self._optimize_with_tflite()
            else:
                self.using_tflite = False
        except (ImportError, AttributeError) as e:
            self.using_tflite = False
        
        self.input_size = (input_size, input_size)
        
        # Warm up the model by running inference on a dummy input
        self._warmup_model()
    
    def _optimize_with_tflite(self):
        """Convert the model to TFLite format for faster inference if possible"""
        try:
            import tensorflow as tf
            
            # Create a converter to generate a TFLite model
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.using_tflite = True
        except Exception as e:
            self.using_tflite = False
        
    def _warmup_model(self):
        """Warm up the model with a dummy input to compile operations"""
        try:
            # Create a dummy RGB input (3 channels)
            dummy_input = np.zeros((1, self.input_size[0], self.input_size[1], 3), dtype=np.float32)
            
            # Run the model once to warm it up
            if hasattr(self, 'using_tflite') and self.using_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
                self.interpreter.invoke()
            else:
                _ = self.model.predict(dummy_input, verbose=0)
            
        except Exception as e:
            warnings.warn(f"Failed to warm up model: {e}. This may affect initial prediction speed.")
    
    @staticmethod
    @lru_cache(maxsize=10)  # Cache the last 10 label files
    def _load_labels(labels_file):
        """Load labels from a text file with caching"""
        with open(labels_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    @lru_cache(maxsize=32)  # Cache the preprocessed images for file paths
    def preprocess_image_from_path(self, img_path):
        """
        Preprocess an image from file path for model prediction with caching
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Load image with PIL to better handle different image formats
        img = Image.open(img_path)
        img = img.resize(self.input_size)
        return self._process_pil_image(img)
    
    def preprocess_image_from_stream(self, image_data):
        """
        Preprocess an image from binary stream data for model prediction
        
        Args:
            image_data (bytes): Binary image data
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Open image from binary data
        img = Image.open(BytesIO(image_data))
        img = img.resize(self.input_size)
        return self._process_pil_image(img)
    
    def _process_pil_image(self, img):
        """
        Process a PIL Image object to prepare it for the model
        
        Args:
            img (PIL.Image): PIL Image object
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Convert image to RGB mode if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Ensure shape is correct (H, W, 3)
        if len(img_array.shape) == 2:
            # Grayscale image (H, W) -> convert to RGB (H, W, 3)
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 1:
            # Single channel image (H, W, 1) -> convert to RGB (H, W, 3)
            img_array = np.concatenate([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            # RGBA image (H, W, 4) -> convert to RGB (H, W, 3)
            img_array = img_array[:, :, :3]
        
        # Pre-allocate the expanded array for better memory efficiency
        batch_img = np.zeros((1, *img_array.shape), dtype=np.float32)
        batch_img[0] = img_array
        
        # Normalize using vectorized operation (faster)
        batch_img /= 255.0
        
        return batch_img
    
    def classify(self, image_input):
        """
        Classify an image from either a file path or a binary stream
        
        Args:
            image_input (str or bytes): Path to image file or binary image data
            
        Returns:
            dict: Classification results including class name, confidence, and all predictions
        """
        try:
            # Determine if input is a file path (str) or image data (bytes)
            if isinstance(image_input, str):
                # Process as a file path
                preprocessed_img = self.preprocess_image_from_path(image_input)
            elif isinstance(image_input, bytes):
                # Process as binary image data
                preprocessed_img = self.preprocess_image_from_stream(image_input)
            else:
                raise TypeError("image_input must be a file path (str) or binary image data (bytes)")
            
            # Ensure image has correct shape with 3 channels (RGB)
            if preprocessed_img.shape[3] != 3:
                raise ValueError(f"Expected 3 channels (RGB) but got {preprocessed_img.shape[3]} channels")
            
            # Make prediction - use TFLite if available
            if hasattr(self, 'using_tflite') and self.using_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_img)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            else:
                # Use the standard Keras model
                predictions = self.model.predict(preprocessed_img, verbose=0)  # Disable progress bar
            
            # Get top prediction
            if predictions.shape[1] == 1:
                # Binary classification with sigmoid
                confidence = float(predictions[0][0])
                class_idx = 1 if confidence >= 0.5 else 0
                confidence = confidence if class_idx == 1 else 1 - confidence
            else:
                # Multi-class with softmax - use np.argmax which is faster
                class_idx = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][class_idx])
            
            # Get class name
            class_name = self.labels[class_idx]
            
            # Return a minimal result first, with a method to get all predictions if needed
            result = {
                'class_index': class_idx,
                'class_name': class_name,
                'confidence': confidence,
            }
            
            # Add all predictions in a dictionary format
            result['all_predictions'] = {
                self.labels[i]: float(predictions[0][i]) for i in range(min(len(self.labels), predictions.shape[1]))
            }
            
            return result
            
        except Exception as e:
            # Catch and re-raise exceptions with more context
            raise RuntimeError(f"Classification failed: {e}")
    
    def batch_classify(self, image_inputs):
        """
        Classify multiple images in a batch for better throughput.
        Each input can be either a file path or binary image data.
        
        Args:
            image_inputs (list): List of image file paths or binary image data
            
        Returns:
            list: List of classification results dictionaries
        """
        # Preprocess all images
        batch_size = min(len(image_inputs), 32)  # Process in batches of 32 or less
        results = []
        
        for i in range(0, len(image_inputs), batch_size):
            batch_inputs = image_inputs[i:i+batch_size]
            
            # Preprocess each input in the batch
            batch_imgs = []
            for input_data in batch_inputs:
                if isinstance(input_data, str):
                    batch_imgs.append(self.preprocess_image_from_path(input_data))
                elif isinstance(input_data, bytes):
                    batch_imgs.append(self.preprocess_image_from_stream(input_data))
                else:
                    raise TypeError("Each input must be a file path (str) or binary image data (bytes)")
            
            # Stack all preprocessed images into a single batch
            batch_imgs = np.vstack(batch_imgs)
            
            # Make batch prediction
            if hasattr(self, 'using_tflite') and self.using_tflite:
                # TFLite doesn't support batching as easily, so we process one by one
                batch_predictions = []
                for j in range(batch_imgs.shape[0]):
                    self.interpreter.set_tensor(self.input_details[0]['index'], batch_imgs[j:j+1])
                    self.interpreter.invoke()
                    batch_predictions.append(self.interpreter.get_tensor(self.output_details[0]['index']))
                batch_predictions = np.vstack(batch_predictions)
            else:
                # Use the standard Keras model with batching
                batch_predictions = self.model.predict(batch_imgs, verbose=0)
            
            # Process each prediction
            for j, predictions in enumerate(batch_predictions):
                if predictions.shape[0] == 1:
                    # Binary classification
                    confidence = float(predictions[0])
                    class_idx = 1 if confidence >= 0.5 else 0
                    confidence = confidence if class_idx == 1 else 1 - confidence
                else:
                    # Multi-class
                    class_idx = int(np.argmax(predictions))
                    confidence = float(predictions[class_idx])
                
                # Store the result
                results.append({
                    'class_index': class_idx,
                    'class_name': self.labels[class_idx],
                    'confidence': confidence,
                    'input': batch_inputs[j],
                    'all_predictions': {self.labels[i]: float(predictions[i]) for i in range(min(len(self.labels), len(predictions)))}
                })
        
        return results

if __name__ == "__main__":
    import argparse
    import requests
    import sys
    
    parser = argparse.ArgumentParser(description='Classify a road image from file or URL')
    parser.add_argument('--image', help='Path to image file to classify')
    parser.add_argument('--url', help='URL of image to classify')
    parser.add_argument('--model', default='keras_model.h5', help='Path to Keras .h5 model file')
    parser.add_argument('--labels', default='labels.txt', help='Path to labels.txt file')
    parser.add_argument('--size', type=int, default=224, help='Input image size (default: 224)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if not args.image and not args.url:
        parser.error("Either --image or --url must be provided")
    
    # Configure logging based on debug flag
    if not args.debug:
        # Suppress warnings and TF noise
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
        warnings.filterwarnings('ignore')
    
    # Create classifier
    classifier = RoadClassifier(
        model_path=args.model,
        labels_path=args.labels,
        input_size=args.size
    )
    
    # Classify image based on input type
    if args.url:
        # Download image from URL
        response = requests.get(args.url)
        image_data = response.content
        result = classifier.classify(image_data)
    else:
        # Classify local image file
        result = classifier.classify(args.image)
    
    # Print results
    print("\nClassification Results:")
    print(f"Predicted Class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    
    print("\nAll Class Probabilities:")
    for class_name, prob in result['all_predictions'].items():
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
