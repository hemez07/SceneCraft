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



import torch
import torch.nn as nn
import torch.optim as optim

# Define NeRF model
class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)  # RGB + density
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Normalize output
        return x

# Initialize model
model = NeRF()



# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate random training data (replace with actual dataset later)
num_samples = 1000
train_data = torch.rand(num_samples, 3) * 2 - 1  # 3D coordinates in range [-1, 1]
train_labels = torch.rand(num_samples, 4)  # Random RGB + density values

# Training loop
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    
    import open3d as o3d

# Generate novel views
test_points = torch.rand(1000, 3) * 2 - 1
predicted_values = model(test_points).detach().numpy()

# Extract point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(test_points.numpy())

# Visualize 3D point cloud
o3d.visualization.draw_geometries([point_cloud])


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize  # To handle size mismatches

# Compute PSNR
def compute_psnr(image1, image2):
    return psnr(image1, image2, data_range=1.0)

# Compute SSIM with custom window size
def compute_ssim(image1, image2):
    return ssim(image1, image2, data_range=1.0, multichannel=True, win_size=3)  # Use a smaller window size

# Example ground truth and predicted images (already loaded)
ground_truth_image = processed_images[0]
predicted_image = processed_images[1]  # Simulated new view

# Ensure both images have the same size (resize if necessary)
predicted_image_resized = resize(predicted_image, ground_truth_image.shape, preserve_range=True)

# Compute PSNR and SSIM values
psnr_value = compute_psnr(ground_truth_image, predicted_image_resized)
ssim_value = compute_ssim(ground_truth_image, predicted_image_resized)

# Print the results
print(f"PSNR: {psnr_value:.2f}")
print(f"SSIM: {ssim_value:.4f}")
