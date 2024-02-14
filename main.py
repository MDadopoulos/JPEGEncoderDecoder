from preprocessing import convert2ycrcb, convert2rgb, ensure_dimensions
from DCT_transform import blockDCT, iBlockDCT
from quantization import quantizeJPEG, dequantizeJPEG
from zigzag_RLE import runLength, irunLength
from tables import huffman_table_DC_luminance,huffman_table_DC_chrominance ,huffman_table_AC_luminance,huffman_table_AC_chrominance
from huffman import huffEnc,huffDec
import cv2
import numpy as np

# Read image
original_imageRGB = cv2.imread('baboon.png')

# Ensure dimensions are multiples of 8
resized_imageRGB = ensure_dimensions(original_imageRGB)

subimg = [4, 2, 0]  # Example subsampling matrix
imageY, imageCr, imageCb = convert2ycrcb(resized_imageRGB, subimg)
reconstructed_imageRGB = convert2rgb(imageY, imageCr, imageCb, subimg)


# Example usage DCT transform
block = np.random.rand(8, 8)*255  # Example 8x8 block
dctBlock = blockDCT(block)
print("Original block:")
print(block)
print("\nDCT block:")
print(dctBlock)
# reconstructed_block = iBlockDCT(dctBlock)

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

print("Quantized Block:")
print(qBlock)
#here in quantization not perfectly quantized..error in dewuantization
print("\nDequantized DCT Block:")
print(dequantizedDctBlock)


# Example usage of run-length encoding
# Assume qBlock is an 8x8 quantized block of DCT coefficients,
# and DCpred is the predicted DC coefficient from the previous block.

DCpred = 0  # Example DC coefficient prediction
runSymbols = runLength(qBlock, DCpred)
print(runSymbols)
decoded_qBlock = irunLength(runSymbols, DCpred)
print(decoded_qBlock)

# Example usage of Huffman encoding
# Assume runSymbols is a list of run-length symbols,    
# and huffman_table_DC and huffman_table_AC are the Huffman tables for DC and AC coefficients, respectively.
huffStream=huffEnc(runSymbols, huffman_table_DC_luminance,huffman_table_AC_luminance)
print(huffStream)

# Example usage of Huffman decoding
# Assume huffStream is a list of bits representing the Huffman stream, 
#and huffman_table_DC and huffman_table_AC are the Huffman tables for DC and AC coefficients, respectively.

runSymbol=huffDec(huffStream, huffman_table_DC_luminance,huffman_table_AC_luminance)
print(runSymbol)








if __name__ == "__main__" :

    
    from DCT_transform import blockDCT
    from DCT_transform import iBlockDCT
    from quantization import quantizeJPEG
    from quantization import dequantizeJPEG

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
    print("Quantized Block:")
    print(qBlock)
    # Example usage of run-length encoding
    # Assume qBlock is an 8x8 quantized block of DCT coefficients,
    # and DCpred is the predicted DC coefficient from the previous block.

    DCpred = 0  # Example DC coefficient prediction
    runSymbols = runLength(qBlock, DCpred)
    print(runSymbols)
    decoded_qBlock = irunLength(runSymbols, DCpred)
    print(decoded_qBlock)
    # Example usage of Huffman encoding
    # Assume runSymbols is a list of run-length symbols,    
    # and huffman_table_DC and huffman_table_AC are the Huffman tables for DC and AC coefficients, respectively.
    # huffman_encoded = huffmanEncoding(runSymbols, huffman_table_DC, huffman_table_AC)
    dequantizedDctBlock = dequantizeJPEG(decoded_qBlock, luminance_qTable, qScale)
    reconstructed =iBlockDCT(dctBlock)
    #here in quantization not perfectly quantized..error in dewuantization
    print("\nDequantized DCT Block:")
    print(dequantizedDctBlock)
    print("reconstructed block")
    print(reconstructed)