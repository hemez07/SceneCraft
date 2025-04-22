import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Define dataset path
dataset_path = "./data/nerf_images"

# Load images from different viewpoints
image_filenames = sorted(os.listdir(dataset_path))
images = [imageio.imread(os.path.join(dataset_path, fname)) for fname in image_filenames]

# Convert images to grayscale and normalize
processed_images = [img / 255.0 for img in images]

# Display a few sample images
fig, axes = plt.subplots(1, len(processed_images[:4]), figsize=(12, 4))
for i, img in enumerate(processed_images[:4]):
    axes[i].imshow(img)
    axes[i].axis("off")
plt.show()