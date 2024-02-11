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

if __name__ == "__main__" : 

    from DCT_transform import blockDCT
    from DCT_transform import iBlockDCT
    block = np.random.randint(0, 256, size=(8, 8))
    dctBlock = blockDCT(block)
    print("Original block:")
    print(block)
    print("\nDCT block:")
    print(dctBlock)
    # Example usage quantization
    # qTable is one of the quantization tables provided,# Example quantization tables for Luminance and Chrominance, as provided in the JPEG standard
    luminance_qTable = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    chrominance_qTable = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])

    # Quantization scale (a positive number) 
    qScale = 1  # Example scale

    # Quantization and Dequantization
    qBlock = quantizeJPEG(dctBlock, luminance_qTable, qScale)
    dequantizedDctBlock = dequantizeJPEG(qBlock, luminance_qTable, qScale)
    reconstructed =iBlockDCT(dctBlock)
    print("Quantized Block:")
    print(qBlock)
    #here in quantization not perfectly quantized..error in dewuantization
    print("\nDequantized DCT Block:")
    print(dequantizedDctBlock)
    print("reconstructed block")
    print(reconstructed)

