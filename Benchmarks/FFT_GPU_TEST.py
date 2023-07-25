import torch
from tqdm import tqdm as tqdm
import random

# Define array size
N = 10000
cycle_range = random.randrange(5000, 15000, 5000)

for i in tqdm(range(cycle_range)):
    # Create the array on GPU
    x = torch.rand(N, N, device='cuda').float()

    # Perform the FFT
    torch.fft.fft(x)