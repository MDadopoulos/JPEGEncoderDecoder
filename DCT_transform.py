
import cv2 as cv2
import numpy as np

# Set the precision for the DCT coefficients
P=8

def blockDCT(block):
    """
    Applies Discrete Cosine Transform (DCT) to a block.

    Parameters:
    block (numpy.ndarray): The input block (2D array).

    Returns:
    numpy.ndarray: DCT coefficients of the block.
    """

    #Apply level shift for DCT compression  
    shift_value = 2 ** (P - 1)
    block=block - shift_value 
    # Convert from int to float and normalize
    fltblock = np.float32(block) / 255.0 
    # Apply DCT
    fltDct = cv2.dct(fltblock)    
    # Convert back to int format (rescale)       
    dctBlock = np.int32(fltDct * 255)     
    return dctBlock

def iBlockDCT(dctBlock):
    """
    Applies inverse Discrete Cosine Transform (iDCT) to a block.

    Parameters:
    dctBlock (numpy.ndarray): DCT coefficients of the block (in 8-bit format).

    Returns:
    numpy.ndarray: The reconstructed block after applying iDCT.
    """

    # Convert from int to float and normalize
    dctBlock = np.float32(dctBlock) / 255.0

    # Apply inverse DCT
    fltBlock = cv2.idct(dctBlock)

    # Rescale back to original range and convert to int
    reconstructedBlock = np.int32(fltBlock * 255)

    #Apply inverse level shift
    shift_value = 2 ** (P - 1)
    DctBlock=reconstructedBlock + shift_value

    return DctBlock#reconstructedBlock

if __name__ == "__main__":
    # Example usage DCT transform
    block = np.random.randint(0, 256, size=(8, 8))
    dctBlock = blockDCT(block)
    print("Original block:")
    print(block)
    print("\nDCT block:")
    print(dctBlock)
    reconstructed_block = iBlockDCT(dctBlock)
    print("\nReconstructed block:")
    print(reconstructed_block)