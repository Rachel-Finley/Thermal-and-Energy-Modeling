import torch
from tqdm import tqdm as tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def blackScholes_pyTorch(S_0, strike, time_to_expiry, implied_vol, riskfree_rate):
    S = S_0
    K = strike
    dt = time_to_expiry
    sigma = implied_vol
    r = riskfree_rate
    Phi = torch.distributions.Normal(0,1).cdf
    d_1 = (torch.log(S_0 / K) + (r+sigma**2/2)*dt) / (sigma*torch.sqrt(dt))
    d_2 = d_1 - sigma*torch.sqrt(dt)
    return S*Phi(d_1) - K*torch.exp(-r*dt)*Phi(d_2)


cycle_range = random.randrange(50000, 150000, 50000)

for i in tqdm(range(cycle_range)):
    # Define your tensors and move them to the appropriate device right away
    S_0 = torch.tensor([100.], requires_grad = True, device=device)
    K = torch.tensor([101.], requires_grad = True, device=device)
    T = torch.tensor([1.], requires_grad = True, device=device)
    sigma = torch.tensor([0.3], requires_grad = True, device=device)
    r = torch.tensor([0.01], requires_grad = True, device=device)
    npv_pytorch = blackScholes_pyTorch(S_0, K, T, sigma, r)
