from kmeans_gpu import KMeans
from tqdm import tqdm as tqdm
import torch
import random

# Config
batch_size = 128
feature_dim = 1024
pts_dim = 3
num_pts = 256
num_cluster = 15

cycle_range = random.randrange(20, 80, 20)

for i in tqdm(range(cycle_range)):
    # Create data
    features = torch.randn(batch_size, feature_dim, num_pts).to("cuda")
    # Pay attention to the different dimension order between features and points.
    points = torch.randn(batch_size, num_pts, pts_dim).to("cuda")

    # Create KMeans Module
    kmeans = KMeans(
        n_clusters=num_cluster,
        max_iter=100,
        tolerance=1e-4,
        distance='euclidean',
        sub_sampling=None,
        max_neighbors=15,
    )

    # Forward
    centroids, features = kmeans(points, features)

    # output:
    # >>> torch.Size([128, 15, 3]) torch.Size([128, 1024, 15])