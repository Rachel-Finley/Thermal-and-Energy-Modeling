import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import random

from tqdm import tqdm as tqdm

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cycle_range = random.randrange(150000, 450000, 1500000)

for i in tqdm(range(cycle_range)):
    # Create a random image tensor
    image = torch.rand(3, 224, 224).to(device)  # move tensor to GPU if available

    # Define sepia transformation matrix
    sepia_weights = torch.tensor([[0.393, 0.769, 0.189],
                                  [0.349, 0.686, 0.168],
                                  [0.272, 0.534, 0.131]]).to(device)  # move tensor to GPU if available

    # Reshape the image tensor for matrix multiplication
    original_shape = image.shape  # save original shape
    image = image.view(3, -1)  # reshape for matrix multiplication

    # Apply the sepia filter
    sepia_image = sepia_weights.matmul(image)
    sepia_image = sepia_image.view(original_shape)  # reshape back to original shape
    sepia_image = sepia_image.clamp(0, 1)  # clamp pixel values between 0 and 1
