import numpy as np

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




import cv2 as cv2
import numpy as np

def doDct(inputMatrix):
    fltMatrix = np.float32(inputMatrix)/255.0
    fltDct = cv2.dct(fltMatrix)
    return np.uint8(fltMatrix * 255)


import cv2
import numpy as np

def blockDCT(block):
    """
    Applies Discrete Cosine Transform (DCT) to a block.

    Parameters:
    block (numpy.ndarray): The input block (2D array).

    Returns:
    numpy.ndarray: DCT coefficients of the block.
    """
    fltblock = np.float32(block) / 255.0  # Normalize the block
    fltDct = cv2.dct(fltblock)            # Apply DCT
    dctBlock = np.uint8(fltDct * 255)     # Convert back to 8-bit format (rescale)

    return dctBlock

def iBlockDCT(dctBlock):
    """
    Applies inverse Discrete Cosine Transform (iDCT) to a block.

    Parameters:
    dctBlock (numpy.ndarray): DCT coefficients of the block (in 8-bit format).

    Returns:
    numpy.ndarray: The reconstructed block after applying iDCT.
    """
    # Convert from 8-bit to float and normalize
    fltDctBlock = np.float32(dctBlock) / 255.0

    # Apply inverse DCT
    fltBlock = cv2.idct(fltDctBlock)

    # Rescale back to original range and convert to 8-bit
    reconstructedBlock = np.uint8(fltBlock * 255)

    return reconstructedBlock


