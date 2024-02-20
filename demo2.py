from preprocessing import convert2ycrcb, convert2rgb, ensure_dimensions
from DCT_transform import blockDCT, iBlockDCT
from quantization import quantizeJPEG, dequantizeJPEG
from zigzag_RLE import runLength, irunLength
from huffman_tables import huffman_table_DC_luminance,huffman_table_DC_chrominance ,huffman_table_AC_luminance,huffman_table_AC_chrominance
from huffman import huffEnc,huffDec
from demo1 import DCT_quantize_channel
from JPEG_encoder_decoder import JPEGtables
import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage


def calculate_run_length_entropy(run_length_pairs):
    """
    Calculate the entropy of run-length encoded data.

    Parameters:
    run_length_pairs (list of tuples): Each tuple represents (symbol, run_length).

    Returns:
    float: The entropy of the run-length encoded data.
    """
    # Flatten the run_length_pairs to treat each (symbol, run_length) as unique
    flat_pairs = [str(symbol) + '_' + str(run_length) for symbol, run_length in run_length_pairs]
    
    # Calculate the counts of unique values in the array
    values, counts = np.unique(flat_pairs, return_counts=True)

     # Calculate the probabilities of each unique value
    probabilities =  counts / len(flat_pairs)

    # Calculate the entropy,adding a small value to avoid log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9 ))
    
    return entropy


def calculate_entropy(image):
    """
    Calculate the entropy of an image or an image channel.

    Parameters:
    image (numpy.ndarray): The input image or image channel.

    Returns:
    float: The entropy of the image or channel.
    """
    # Flatten the input to a 1D array
    pixel_values = image.flatten()
    
    # Calculate the counts of unique values in the array
    values, counts = np.unique(pixel_values, return_counts=True)

    # Calculate the probabilities of each unique value
    probabilities =  counts / len(pixel_values)



    # Calculate the entropy,adding a small value to avoid log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9 ))
    
    return entropy

def run_length_channel_encode(quantized_img_channel):
    """
    Run-length encode a quantized image channel.

    Parameters:
    quantized_img_channel (numpy.ndarray): The quantized image channel.

    Returns:
    list of tuples: The run-length encoded data.
    """

    # Create an empty list to store the run-length pairs
    run_length_pairs = []
    # Initialize the DC coefficient prediction
    DCpred = 0
    for i in range(0, len(quantized_img_channel), 8):
            for j in range(0, len(quantized_img_channel[0]), 8):
                #Run length encode each 8x8 block
                runSymbols = runLength(quantized_img_channel[i:i+8, j:j+8], DCpred)
                #Set the DC coefficient prediction for the next block
                DCpred = runSymbols[0][1] + DCpred
                #Append the run length pairs to the list
                run_length_pairs.extend(runSymbols)


    return run_length_pairs
    

if __name__ == "__main__" :

    # parameters
    subimg1 = [4, 2, 2]  
    subimg2 = [4, 4, 4]
    qscale1=0.6
    qscale2=5

    # Create a JPEGtables object
    tables = JPEGtables()
    luminance_qTable = tables.qTableL
    chrominance_qTable = tables.qTableC

    # Read images and display
    image1 = cv2.imread('baboon.png')
    image1_RGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('lena_color_512.png')
    image2_RGB = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    plt.imshow(image1_RGB)
    plt.show()
    plt.imshow(image2_RGB)
    plt.show()


    entropy1_R = calculate_entropy(image1_RGB[:,:,0])
    entropy1_G = calculate_entropy(image1_RGB[:,:,1])
    entropy1_B = calculate_entropy(image1_RGB[:,:,2])
    print("Entropy of channels R G B of image 1 :", entropy1_R, entropy1_G, entropy1_B)
    print("Total entropy and mean entropy of image 1:", (entropy1_R+entropy1_G+entropy1_B), (entropy1_R+entropy1_G+entropy1_B)/3)
    entropy2_R = calculate_entropy(image2_RGB[:,:,0])
    entropy2_G = calculate_entropy(image2_RGB[:,:,1])
    entropy2_B = calculate_entropy(image2_RGB[:,:,2])
    print("Entropy of channels R G B of image 2 :", entropy2_R, entropy2_G, entropy2_B)
    print("Total entropy and mean entropy of image 2:", (entropy2_R+entropy2_G+entropy2_B), (entropy2_R+entropy2_G+entropy2_B)/3)
    #entropy_r = calculate_entropy(image1_RGB[:,:,0])
    #print("Entropy of the red channel:", entropy_r)
    #print(skimage.measure.shannon_entropy(image1_RGB))

    # Ensure dimensions are multiples of 8
    resized_image1RGB = ensure_dimensions(image1_RGB)
    resized_image2RGB = ensure_dimensions(image2_RGB)

    # Convert to YCrCb
    image1Y, image1Cr, image1Cb = convert2ycrcb(resized_image1RGB, subimg1)
    image2Y, image2Cr, image2Cb = convert2ycrcb(resized_image2RGB, subimg2)

    # Apply DCT and quantization to the luminance channel
    quantized_image1Y = DCT_quantize_channel(image1Y,luminance_qTable,qscale1)
    quantized_image2Y = DCT_quantize_channel(image2Y,luminance_qTable,qscale2)

    # Apply DCT and quantization to the chrominance channels
    quantized_image1Cr = DCT_quantize_channel(image1Cr,chrominance_qTable,qscale1)
    quantized_image1Cb = DCT_quantize_channel(image1Cb,chrominance_qTable,qscale1)
    quantized_image2Cr = DCT_quantize_channel(image2Cr,chrominance_qTable,qscale2)
    quantized_image2Cb = DCT_quantize_channel(image2Cb,chrominance_qTable,qscale2)

    # Calculate entropy of quantized channels and print
    entropy_quantized_image1Y = calculate_entropy(quantized_image1Y)
    entropy_quantized_image1Cr = calculate_entropy(quantized_image1Cr)
    entropy_quantized_image1Cb = calculate_entropy(quantized_image1Cb)
    print("Entropy of quantized channels Y Cr Cb of image 1 :", entropy_quantized_image1Y, entropy_quantized_image1Cr, entropy_quantized_image1Cb)
    print("Total entropy and mean entropy of quantized image 1:", (entropy_quantized_image1Y+entropy_quantized_image1Cr+entropy_quantized_image1Cb), (entropy_quantized_image1Y+entropy_quantized_image1Cr+entropy_quantized_image1Cb)/3)
    entropy_quantized_image2Y = calculate_entropy(quantized_image2Y)
    entropy_quantized_image2Cr = calculate_entropy(quantized_image2Cr)
    entropy_quantized_image2Cb = calculate_entropy(quantized_image2Cb)
    print("Entropy of quantized channels Y Cr Cb of image 2 :", entropy_quantized_image2Y, entropy_quantized_image2Cr, entropy_quantized_image2Cb)
    print("Total entropy and mean entropy of quantized image 2:", (entropy_quantized_image2Y+entropy_quantized_image2Cr+entropy_quantized_image2Cb), (entropy_quantized_image2Y+entropy_quantized_image2Cr+entropy_quantized_image2Cb)/3)

    RLE_image1Y = run_length_channel_encode(quantized_image1Y)
    RLE_image1Cr = run_length_channel_encode(quantized_image1Cr)
    RLE_image1Cb = run_length_channel_encode(quantized_image1Cb)
    entropy_RLE_image1Y = calculate_run_length_entropy(RLE_image1Y)
    entropy_RLE_image1Cr = calculate_run_length_entropy(RLE_image1Cr)
    entropy_RLE_image1Cb = calculate_run_length_entropy(RLE_image1Cb)
    print("Entropy of run length encoded channels Y Cr Cb of image 1 :", entropy_RLE_image1Y, entropy_RLE_image1Cr, entropy_RLE_image1Cb)
    print("Total entropy and mean entropy of run length encoded image 1:", (entropy_RLE_image1Y+entropy_RLE_image1Cr+entropy_RLE_image1Cb), (entropy_RLE_image1Y+entropy_RLE_image1Cr+entropy_RLE_image1Cb)/3)
    RLE_image2Y = run_length_channel_encode(quantized_image2Y)
    RLE_image2Cr = run_length_channel_encode(quantized_image2Cr)
    RLE_image2Cb = run_length_channel_encode(quantized_image2Cb)
    entropy_RLE_image2Y = calculate_run_length_entropy(RLE_image2Y)
    entropy_RLE_image2Cr = calculate_run_length_entropy(RLE_image2Cr)
    entropy_RLE_image2Cb = calculate_run_length_entropy(RLE_image2Cb)
    print("Entropy of run length encoded channels Y Cr Cb of image 2 :", entropy_RLE_image2Y, entropy_RLE_image2Cr, entropy_RLE_image2Cb)
    print("Total entropy and mean entropy of run length encoded image 2:", (entropy_RLE_image2Y+entropy_RLE_image2Cr+entropy_RLE_image2Cb), (entropy_RLE_image2Y+entropy_RLE_image2Cr+entropy_RLE_image2Cb)/3)