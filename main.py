from preprocessing import convert2ycrcb, convert2rgb, ensure_dimensions
from DCT_transform import blockDCT, iBlockDCT
import cv2
import numpy as np

# Read image
img = cv2.imread('baboon.png')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


# Example usage DCT transform
block = np.random.rand(8, 8)  # Example 8x8 block
dctBlock = blockDCT(block)
# reconstructed_block = iBlockDCT(dctBlock)