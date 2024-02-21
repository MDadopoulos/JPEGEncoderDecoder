from preprocessing import convert2ycrcb, convert2rgb, ensure_dimensions
from DCT_transform import blockDCT, iBlockDCT
from quantization import quantizeJPEG, dequantizeJPEG
from zigzag_RLE import runLength, irunLength
from huffman_tables import huffman_table_DC_luminance,huffman_table_DC_chrominance ,huffman_table_AC_luminance,huffman_table_AC_chrominance
from huffman import huffEnc,huffDec
import matplotlib.pyplot as plt
import cv2
import numpy as np



def DCT_quantize_channel(image_channel,qTable,qScale):
    """
    Apply DCT and quantization to an image channel.
    Parameters:
    image_channel (numpy.ndarray): The image channel to be processed.
    qScale (float): The quantization scale.
    qTable (numpy.ndarray): The quantization table.
    Returns:
    numpy.ndarray: The quantized channel.
    """ 

    #create 2 arrays of zeros with the same shape as the image channel to store 
    #the DCT and quantized values
    DCT_channel = np.zeros(image_channel.shape)
    quantized_channel = np.zeros(image_channel.shape)
    for i in range(0, len(image_channel), 8):
        for j in range(0, len(image_channel[0]), 8):
            #apply DCT to each 8x8 block
            DCT_channel[i:i+8, j:j+8] = blockDCT(image_channel[i:i+8, j:j+8])
            #apply quantization to each 8x8 block
            quantized_channel[i:i+8, j:j+8] = quantizeJPEG(DCT_channel[i:i+8, j:j+8], qTable, qScale)
    
    return quantized_channel
    
def dequantize_channel(quantized_channel,qTable,qScale):
    """
    Dequantize and apply inverse DCT to an image channel.
    Parameters:
    quantized_channel (numpy.ndarray): The quantized channel to be processed.
    qScale (float): The quantization scale.
    qTable (numpy.ndarray): The quantization table.
    Returns:
    numpy.ndarray: The dequantized channel.
    """ 
    #create 2 arrays of zeros with the same shape as the image channel to store 
    #the dequantized and inverse DCT values
    dequantized_channel = np.zeros(quantized_channel.shape)
    iDCT_channel = np.zeros(quantized_channel.shape)
    for i in range(0, len(quantized_channel), 8):
        for j in range(0, len(quantized_channel[0]), 8):
            #apply dequantization to each 8x8 block
            dequantized_channel[i:i+8, j:j+8] = dequantizeJPEG(quantized_channel[i:i+8, j:j+8], qTable, qScale)
            #apply inverse DCT to each 8x8 block
            iDCT_channel[i:i+8, j:j+8] = iBlockDCT(dequantized_channel[i:i+8, j:j+8])
    
    return iDCT_channel


if __name__ == "__main__" :

    # parameters
    subimg1 = [4, 2, 2]  
    subimg2 = [4, 4, 4]
    qscale1=0.6
    qscale2=5


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


    # Read images and display
    image1 = cv2.imread('baboon.png')
    image1_RGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('lena_color_512.png')
    image2_RGB = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    plt.imshow(image1_RGB)
    plt.title("Original Image 1")
    plt.show()
    plt.imshow(image2_RGB)
    plt.title("Original Image 2")
    plt.show()

    # Ensure dimensions are multiples of 8
    resized_image1RGB = ensure_dimensions(image1_RGB)
    resized_image2RGB = ensure_dimensions(image2_RGB)



    # Convert to YCrCb
    image1Y, image1Cr, image1Cb = convert2ycrcb(resized_image1RGB, subimg1)
    image2Y, image2Cr, image2Cb = convert2ycrcb(resized_image2RGB, subimg2)

    # Convert back to RGB and display
    image1RGB_from_YCrCb = convert2rgb(image1Y, image1Cr, image1Cb, subimg1)
    image2RGB_from_YCrCb = convert2rgb(image2Y, image2Cr, image2Cb, subimg2)
    plt.imshow(image1RGB_from_YCrCb)
    plt.title("Reconstructed Image 1 from YCrCb")
    plt.show()
    plt.imshow(image2RGB_from_YCrCb)
    plt.title("Reconstructed Image 2 from YCrCb")
    plt.show()

    # Apply DCT and quantization to the luminance channel
    quantized_image1Y = DCT_quantize_channel(image1Y,luminance_qTable,qscale1)
    quantized_image2Y = DCT_quantize_channel(image2Y,luminance_qTable,qscale2)

    # Apply DCT and quantization to the chrominance channels
    quantized_image1Cr = DCT_quantize_channel(image1Cr,chrominance_qTable,qscale1)
    quantized_image1Cb = DCT_quantize_channel(image1Cb,chrominance_qTable,qscale1)
    quantized_image2Cr = DCT_quantize_channel(image2Cr,chrominance_qTable,qscale2)
    quantized_image2Cb = DCT_quantize_channel(image2Cb,chrominance_qTable,qscale2)

    # Dequantize and apply inverse DCT to the luminance channel
    dequantized_image1Y = dequantize_channel(quantized_image1Y,luminance_qTable,qscale1)
    dequantized_image2Y = dequantize_channel(quantized_image2Y,luminance_qTable,qscale2)

    # Dequantize and apply inverse DCT to the chrominance channels
    dequantized_image1Cr = dequantize_channel(quantized_image1Cr,chrominance_qTable,qscale1)
    dequantized_image1Cb = dequantize_channel(quantized_image1Cb,chrominance_qTable,qscale1)
    dequantized_image2Cr = dequantize_channel(quantized_image2Cr,chrominance_qTable,qscale2)
    dequantized_image2Cb = dequantize_channel(quantized_image2Cb,chrominance_qTable,qscale2)

    # Convert back to RGB and display
    dequantized_image1RGB = convert2rgb(dequantized_image1Y, dequantized_image1Cr, dequantized_image1Cb, subimg1)
    dequantized_image2RGB = convert2rgb(dequantized_image2Y, dequantized_image2Cr, dequantized_image2Cb, subimg2)

    plt.imshow(dequantized_image1RGB)
    plt.title("Reconstructed Image 1 from DCT and quantization")
    plt.show()
    plt.imshow(dequantized_image2RGB)
    plt.title("Reconstructed Image 2 from DCT and quantization")
    plt.show()
    