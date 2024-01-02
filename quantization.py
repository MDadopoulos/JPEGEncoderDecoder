import numpy as np

def quantizeJPEG(dctBlock, qTable, qScale):
    """
    Quantizes DCT coefficients of a block using the given quantization table and scale.

    Parameters:
    dctBlock (numpy.ndarray): The DCT coefficients of the block.
    qTable (numpy.ndarray): The quantization table.
    qScale (float): The quantization scale.

    Returns:
    numpy.ndarray: The quantized DCT coefficients of the block.
    """
    # Perform quantization
    qBlock = np.round(dctBlock / (qTable * qScale))
    return qBlock


def dequantizeJPEG(qBlock, qTable, qScale):
    """
    Dequantizes the quantized DCT coefficients of a block using the given quantization table and scale.

    Parameters:
    qBlock (numpy.ndarray): The quantized DCT coefficients of the block.
    qTable (numpy.ndarray): The quantization table.
    qScale (float): The quantization scale.

    Returns:
    numpy.ndarray: The dequantized DCT coefficients of the block.
    """
    # Perform dequantization
    dctBlock = qBlock * (qTable * qScale)
    return dctBlock


