import torch
import torchaudio.transforms as T
from tqdm import tqdm as tqdm
import warnings
import random

warnings.filterwarnings("ignore")
cycle_range = random.randrange(50000, 150000, 50000)

for i in tqdm(range(cycle_range)):
    # Create synthetic audio waveform
    waveform = torch.randn(30, 16000).cuda()  # 1 second of audio at 16kHz

    # Define a transformation: Spectrogram
    spectrogram_transform = T.Spectrogram().cuda()

    # Apply the transformation to the waveform
    spectrogram_transform(waveform)