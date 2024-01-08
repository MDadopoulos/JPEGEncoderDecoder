import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspc
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import sys
imag = imageio.imread("baboon.png")
red = imag[:, :, 0]
grn = imag[:, :, 1]
blu = imag[:, :, 2]
COEF_R = 0.2126
COEF_G = 0.7152
COEF_B = 0.0722
    
luma = imag[:, :, 0] * COEF_R + imag[:, :, 1] * COEF_G + imag[:, :, 2] * COEF_B

pb = luma - blu
pr = luma - red


##this way not that good with the values,had negative values

#####
import numpy as np

A = np.random.randint(5, size = [8,8]) # 4:4:4
print(A)

B = A.copy() # 4:2:0
B[1::2, :] = B[::2, :] 
# Vertically, every second element equals to element above itself.
B[:, 1::2] = B[:, ::2] 
# Horizontally, every second element equals to the element on its left side.
print(B)

C = A.copy()
C[:, 1::2] = C[:, ::2] # 4:2:2
print(C)

D = A.copy()
D[1::2, :] = D[::2, :] # 4:1:0
print(D)

E = A.copy()
orig_size = E.shape
E = np.reshape(E, [-1, 1, 4]) # Get a 3D tensor
E[:,:,:] = E[:,:,0].reshape([-1,1,1]) 
# Get first element of each vector. This creates a matrix. 
# So reshape it into a tensor again.
# First values are then broadcasted to array itself
E = np.reshape(E, orig_size) # Return to its original size
print(E)

from scipy.fft import dct
from scipy.signal import convolve2d
    class Downsampling():
    def __init__(self, ratio='4:2:0'):
        assert ratio in ('4:4:4', '4:2:2', '4:2:0'), "Please choose one of the following {'4:4:4', '4:2:2', '4:2:0'}"
        self.ratio = ratio
        
    def __call__(self, x):
        # No subsampling
        if self.ratio == '4:4:4':
            return x
        else:
            # Downsample with a window of 2 in the horizontal direction
            if self.ratio == '4:2:2':
                kernel = np.array([[0.5], [0.5]])
                out = np.repeat(convolve2d(x, kernel, mode='valid')[::2,:], 2, axis=0)
            # Downsample with a window of 2 in both directions
            else:
                kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
                out = np.repeat(np.repeat(convolve2d(x, kernel, mode='valid')[::2,::2], 2, axis=0), 2, axis=1)
            return np.round(out).astype('int')






def dct(block):
    # Apply 2D DCT on the block
    dct_block = np.zeros_like(block, dtype=float)
    M, N = block.shape
    for u in range(M):
        for v in range(N):
            alpha_u = np.sqrt(1/M) if u == 0 else np.sqrt(2/M)
            alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
            sum_val = 0.0
            for x in range(M):
                for y in range(N):
                    cos_val = np.cos(((2*x + 1) * u * np.pi) / (2 * M)) * np.cos(((2*y + 1) * v * np.pi) / (2 * N))
                    sum_val += block[x, y] * cos_val
            dct_block[u, v] = alpha_u * alpha_v * sum_val
    return dct_block

def idct(dct_block):
    # Apply 2D inverse DCT on the block
    block = np.zeros_like(dct_block, dtype=float)
    M, N = dct_block.shape
    for x in range(M):
        for y in range(N):
            sum_val = 0.0
            for u in range(M):
                for v in range(N):
                    alpha_u = np.sqrt(1/M) if u == 0 else np.sqrt(2/M)
                    alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
                    cos_val = np.cos(((2*x + 1) * u * np.pi) / (2 * M)) * np.cos(((2*y + 1) * v * np.pi) / (2 * N))
                    sum_val += alpha_u * alpha_v * dct_block[u, v] * cos_val
            block[x, y] = sum_val
    return block

