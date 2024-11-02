import rasterio
import numpy as np
import os
from PIL import Image

# Load data
def AvgGradientNorm(params):
    total_norm = 0.0
    counter = 0 
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
            counter += 1
    avg = total_norm/counter
    return avg

def process_images(image_dir, subsetted_image_paths):
    pil_images = []
    
    def process_image(image_path):
        with rasterio.open(image_path) as src:
            image_array = src.read()  # Read image as an array (bands, height, width)
            
            # Convert the image to uint8 by normalizing to the range [0, 255]
            if image_array.dtype == np.float64 or image_array.dtype == np.float32:
                image_min = np.min(image_array)
                image_max = np.max(image_array)
                if image_max != image_min:  # Avoid division by zero
                    # Normalize to [0, 255] for uint8
                    image_array = 255 * (image_array - image_min) / (image_max - image_min)
                image_array = image_array.astype(np.uint8)  # Convert to uint8
            
            # If the image has more than one band, reshape accordingly
            if image_array.shape[0] == 1:  # Grayscale image with 1 band
                image_array = image_array.squeeze(0)  # Remove the first dimension (band)
            elif image_array.shape[0] == 3:  # RGB image with 3 bands
                image_array = np.transpose(image_array, (1, 2, 0))  # (bands, height, width) -> (height, width, bands)
            else:
                raise ValueError(f"Unsupported number of bands: {image_array.shape[0]}")
            
            return Image.fromarray(image_array)

    for file in subsetted_image_paths:
        image_path = os.path.join(image_dir, file)
        pil_images.append(process_image(image_path))

    print(f"Processed {len(pil_images)} images from {image_dir}")

    return pil_images