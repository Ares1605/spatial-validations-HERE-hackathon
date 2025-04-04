import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse
import os
import cv2

# Custom DepthwiseConv2D that ignores the 'groups' parameter
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' if present in kwargs
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # Remove 'groups' if present in the config dictionary
        config.pop('groups', None)
        return super(CustomDepthwiseConv2D, cls).from_config(config)

def load_model(model_path):
    """
    Loads the Keras model using the custom DepthwiseConv2D layer.
    """
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def load_labels(labels_path):
    """
    Loads class labels from a text file (one label per line).
    """
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def preprocess_image(self, image_data):
    # Assuming image_data is a PIL Image or similar
    img = np.array(image_data)
    
    # Check if image is grayscale (1 channel) and convert to RGB (3 channels) if needed
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        # Convert grayscale to RGB by duplicating the channel
        img = np.stack((img,) * 3, axis=-1) if len(img.shape) == 2 else np.concatenate([img] * 3, axis=-1)
    
    # Resize to expected input size
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
def classify_image(model, img_path, target_size):
    """
    Preprocesses an image, runs the model prediction, and returns the output.
    """
    processed_img = preprocess_image(img_path, target_size)
    predictions = model.predict(processed_img)
    return predictions[0]

def main():
    parser = argparse.ArgumentParser(
        description='Classify images using a Teachable Machine model with workaround for DepthwiseConv2D groups parameter'
    )
    parser.add_argument('images', type=str, nargs='+', help='Path(s) to image file(s) to classify')
    parser.add_argument('--model', type=str, default='keras_model.h5', help='Path to the exported model (Keras .h5 or saved_model directory)')
    parser.add_argument('--labels', type=str, default=None, help='Optional path to a text file containing class labels (one per line)')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224], help='Target image size (width height), default is 224x224')
    args = parser.parse_args()

    # Load the classifier model using the custom DepthwiseConv2D workaround
    model = load_model(args.model)
    
    # Load labels if provided
    labels = load_labels(args.labels) if args.labels and os.path.exists(args.labels) else None

    # Process each image file provided in the arguments
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        predictions = classify_image(model, img_path, tuple(args.size))
        print(f"\nPredictions for {img_path}:")

        # If you have labels, print them with scores
        if labels:
            for label, score in zip(labels, predictions):
                print(f"  {label}: {score:.4f}")
        else:
            # Otherwise, just print the raw prediction vector
            print(predictions)

if __name__ == '__main__':
    main()
