from preprocessing import convert2ycrcb, convert2rgb, ensure_dimensions
from DCT_transform import blockDCT, iBlockDCT
from quantization import quantizeJPEG, dequantizeJPEG
from zigzag_RLE import runLength, irunLength
from huffman_tables import huffman_table_DC_luminance,huffman_table_DC_chrominance ,huffman_table_AC_luminance,huffman_table_AC_chrominance
from huffman import huffEnc,huffDec
import matplotlib.pyplot as plt
import cv2
import numpy as np


def calculate_entropy(image):
    """
    Calculate the entropy of an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    float: The entropy of the image.
    """
    # Flatten the image to a 1D array
    pixel_values = image.flatten()
    
    # Calculate the histogram of pixel values
    histogram = np.bincount(pixel_values, minlength=256)
    
    # Normalize the histogram to get probabilities
    probabilities = histogram / np.sum(histogram)
    
    # Filter out zero probabilities (log(0) is undefined)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate the entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def calculate_entropy_image(image):
    # Flatten the image to a 1D array per channel
    pixel_values_r = image[:, :, 0].flatten()
    histogram_r = np.bincount(pixel_values_r, minlength=256)
    probabilities_r = histogram_r / np.sum(histogram_r)
    pixel_values_g = image[:, :, 1].flatten()
    histogram_g = np.bincount(pixel_values_g, minlength=256)
    probabilities_g = histogram_g / np.sum(histogram_g)
    pixel_values_b = image[:, :, 2].flatten()
    histogram_b = np.bincount(pixel_values_b, minlength=256)
    probabilities_b = histogram_b / np.sum(histogram_b)
    
    # Calculate entropy for each channel
    entropy_r = -np.sum(probabilities_r * np.log2(probabilities_r + 1e-9))
    entropy_g = -np.sum(probabilities_g * np.log2(probabilities_g + 1e-9))
    entropy_b = -np.sum(probabilities_b * np.log2(probabilities_b + 1e-9))
    
    # Sum the entropies of each channel to get total entropy
    total_entropy = entropy_r + entropy_g + entropy_b
    return total_entropy


if __name__ == "__main__" :

    # parameters
    subimg1 = [4, 2, 2]  
    subimg2 = [4, 4, 4]
    qscale1=0.6
    qscale2=5



    # Read images and display
    image1 = cv2.imread('baboon.png')
    image1_RGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('lena_color_512.png')
    image2_RGB = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    plt.imshow(image1_RGB)
    plt.show()
    plt.imshow(image2_RGB)
    plt.show()

    
    # Example usage:
    # Assuming `image` is a 2D numpy array representing a grayscale image
    entropy = calculate_entropy(image1_RGB)
    print("Entropy of the image:", entropy)
    entropy_image = calculate_entropy_image(image1_RGB)
    print("Entropy of the image:", entropy_image)