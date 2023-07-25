import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from numba import cuda
import warnings
from tqdm import tqdm as tqdm

warnings.filterwarnings("ignore")


def draw_image(mat, cmap='inferno', powern=0.5, dpi=72):
    ## Value normalization
    # Apply power normalization, because number of iteration is
    # distributed according to a power law (fewer pixels have
    # higher iteration number)
    mat = np.power(mat, powern)

    # Colormap: set the color the black for values under vmin (inner points of
    # the set), vmin will be set in the imshow function
    new_cmap = copy.copy(cm.get_cmap(cmap))
    new_cmap.set_under('black')

    ## Plotting image

    # Figure size
    plt.figure(figsize=(mat.shape[0] / dpi, mat.shape[1] / dpi))

    # Plotting mat with cmap
    # vmin=1 because smooth iteration count is always > 1
    # We need to transpose mat because images use row-major
    # ordering (C convention)
    # origin='lower' because mat[0,0] is the lower left pixel
    plt.imshow(mat.T, cmap=new_cmap, vmin=1, origin='lower')

    # Remove axis and margins
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')


@cuda.jit
def mandelbrot_gpu(mat, maxiter=100, xmin=-2.6, xmax=1.85, ymin=-1.25, ymax=1.25):
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x

    # Mapping pixel to C
    creal = xmin + x / mat.shape[0] * (xmax - xmin)
    cim = ymin + y / mat.shape[1] * (ymax - ymin)

    # Initialisation of C and Z
    c = complex(creal, cim)
    z = complex(0, 0)

    # Mandelbrot iteration
    for n in range(maxiter):
        z = z * z + c
        # If unbounded: save iteration count and break
        if z.real * z.real + z.imag * z.imag > 4.0:
            # Smooth iteration count
            mat[x, y] = n + 1 - math.log(math.log(abs(z * z + c))) / math.log(2)
            break
        # Otherwise: leave it to 0


# Running and plotting result
for i in tqdm(range(50)):
    xmin, xmax = -2.6, 1.85
    ymin, ymax = -1.25, 1.25
    xpixels = 600
    ypixels = round(xpixels / (xmax - xmin) * (ymax - ymin))
    maxiter = 100
    mat_1 = np.zeros((xpixels, ypixels))
    mandelbrot_gpu[xpixels, ypixels](mat_1, maxiter, xmin, xmax, ymin, ymax)

    xmin, xmax = -1.7687782, -1.7687794
    ymin, ymax = -0.0017384, -0.0017394
    xpixels = 1200
    ypixels = round(xpixels / (xmax - xmin) * (ymax - ymin))
    maxiter = 1e4
    mat_2 = np.zeros((xpixels, ypixels))
    mandelbrot_gpu[xpixels, ypixels](mat_2, maxiter, xmin, xmax, ymin, ymax)

    xmin, xmax = -0.776606, -0.776581
    ymin, ymax = -0.136651, -0.136630
    xpixels = 1200
    ypixels = round(xpixels / (xmax - xmin) * (ymax - ymin))
    mat_3 = np.zeros((xpixels, ypixels))
    mandelbrot_gpu[xpixels, ypixels](mat_3, maxiter, xmin, xmax, ymin, ymax)


draw_image(mat_1)
draw_image(mat_2, cmap='hot', powern=1)
draw_image(mat_3, cmap='gist_stern', powern=.8)