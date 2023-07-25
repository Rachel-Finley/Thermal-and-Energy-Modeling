import torch
from tqdm import tqdm as tqdm
import random

# Define matrix size
N = 10000
cycle_range = random.randrange(3000, 9000, 3000)

for i in tqdm(range(cycle_range)):
    # Create two matrices on the GPU
    A = torch.rand(N, N, device='cuda').float()
    B = torch.rand(N, N, device='cuda').float()

    # Calculate the Euclidean distance between each pair of corresponding rows
    torch.sqrt(((A - B)**2).sum(dim=1))