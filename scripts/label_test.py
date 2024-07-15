import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_and_save_image(image_path, output_path):
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return
    
    # Load image
    image = Image.open(image_path)
    # Display image
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    
    # Save the image to the specified output path
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Specify the path to your image
    #image_path = "/media/data2/imagenet21k_masks/outputs_imagenet21k/n03865949/n03865949_4345_mask.png"
    image_path = "/media/data/Datasets/imagenet21k_resized/imagenet21k_train/n03865949/n03865949_4345.JPEG"
    output_path = "output_image.png"
    
    # Load and save the image
    load_and_save_image(image_path, output_path)