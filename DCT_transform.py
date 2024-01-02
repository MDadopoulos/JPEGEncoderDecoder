
import cv2 as cv2
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


