import torch
from tqdm import tqdm as tqdm
import random

def monte_carlo_down_out_torch_cuda(S_0, strike, time_to_expiry, implied_vol, riskfree_rate, barrier, steps, samples):
    stdnorm_random_variates = torch.cuda.FloatTensor(steps, samples).normal_()
    S = S_0
    K = strike
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    sigma = implied_vol
    r = riskfree_rate
    B = barrier
    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet
    B_shift = B*torch.exp(0.5826*sigma*torch.sqrt(dt))
    S_T = S * torch.cumprod(torch.exp((r-sigma**2/2)*dt+sigma*torch.sqrt(dt)*stdnorm_random_variates), dim=1)
    non_touch = torch.min(S_T, dim=1)[0] > B_shift
    non_touch = non_touch.type(torch.cuda.FloatTensor)
    call_payout = S_T[:,-1] - K
    call_payout[call_payout<0]=0
    npv = torch.mean(non_touch * call_payout)
    return torch.exp(-time_to_expiry*r)*npv


cycle_range = random.randrange(300, 900, 300)

for i in tqdm(range(cycle_range)):
    S = torch.tensor([100.], requires_grad = True, device = 'cuda')
    K = torch.tensor([110.], requires_grad = True, device = 'cuda')
    T = torch.tensor([2.], requires_grad = True, device = 'cuda')
    sigma = torch.tensor([0.2], requires_grad = True, device = 'cuda')
    r = torch.tensor([0.03], requires_grad = True, device = 'cuda')
    B = torch.tensor([90.], requires_grad = True, device = 'cuda')

    npv_torch_mc = monte_carlo_down_out_torch_cuda(S, K, T, sigma, r, B, 1000, 100000)
    npv_torch_mc.backward()