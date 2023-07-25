import torch
import random
from tqdm import tqdm as tqdm

cycle_range = random.randrange(20, 80, 20)


def run_random_matmuls(n):
    for i in tqdm(range(n)):
        tensor_1 = torch.randn(10000, 10000).to('cuda')
        tensor_2 = torch.randn(10000, 10000).to('cuda')
        torch.matmul(tensor_1, tensor_2)  # perform matrix multiplication
        torch.cuda.synchronize()


def main():
    run_random_matmuls(cycle_range)


main()