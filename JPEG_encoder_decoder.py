from preprocessing import convert2ycrcb, convert2rgb, ensure_dimensions
from DCT_transform import blockDCT, iBlockDCT
from quantization import quantizeJPEG, dequantizeJPEG
from zigzag_RLE import runLength, irunLength
from huffman_tables import huffman_table_DC_luminance,huffman_table_DC_chrominance ,huffman_table_AC_luminance,huffman_table_AC_chrominance
from huffman import huffEnc,huffDec
from demo1 import DCT_quantize_channel,dequantize_channel
import matplotlib.pyplot as plt
import cv2
import numpy as np


class JPEGtables:
    
    qTableL = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    qTableC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])
    DCL = huffman_table_DC_luminance
    DCC = huffman_table_DC_chrominance
    ACL = huffman_table_AC_luminance
    ACC = huffman_table_AC_chrominance


    def reset_highfreq(self):
        high_freq_positions = [(i // 8, i % 8) for i in [20 , 40 , 50 ,60 , 63]]
        for position in high_freq_positions:
            self.qTableL[position] = 9999
            self.qTableC[position] = 9999



class JPEGencoded:
    def __init__(self,blkType,  indVer, indHor, huffStream):
        self.blkType = blkType
        self.indHor = indHor
        self.indVer = indVer
        self.huffStream = huffStream

def channel_encode(img, qScale, JPEGenc, tables,blkType):
    """
    Encodes a single channel of an image using the JPEG standard.
    Parameters:
    img (numpy.ndarray): The input image channel.
    qScale (float): The quantization scale.
    JPEGenc (list): The list to store the encoded blocks of the channel.
    tables (JPEGtables): The JPEGtables object.
    blkType (str): The type of the image channel.
    """
    Dct_Y = np.zeros(img.shape).astype(int)
    quantized_Y = np.zeros(img.shape).astype(int)
    DCpred = 0  # Example DC coefficient prediction
    if blkType == "Y":
        for i in range(0, len(img), 8):
            for j in range(0, len(img[0]), 8):
                #apply DCT to each 8x8 block
                Dct_Y[i:i+8, j:j+8] = blockDCT(img[i:i+8, j:j+8])
                #apply quantization to each 8x8 block
                quantized_Y[i:i+8, j:j+8] = quantizeJPEG(Dct_Y[i:i+8, j:j+8], tables.qTableL, qScale)
                runSymbols = runLength(quantized_Y[i:i+8, j:j+8], DCpred)
                #Set the DC coefficient prediction for the next block
                DCpred = runSymbols[0][1] + DCpred
                #encode the runSymbols using huffman encoding
                huffStream=huffEnc(runSymbols, tables.DCL,tables.ACL)
                #append the encoded block to the JPEGenc list
                JPEGenc.append(JPEGencoded(blkType,i,j,huffStream))
    elif blkType == "Cr":
        for i in range(0, len(img), 8):
            for j in range(0, len(img[0]), 8):
                #apply DCT to each 8x8 block
                Dct_Y[i:i+8, j:j+8] = blockDCT(img[i:i+8, j:j+8])
                #apply quantization to each 8x8 block
                quantized_Y[i:i+8, j:j+8] = quantizeJPEG(Dct_Y[i:i+8, j:j+8], tables.qTableC, qScale)
                runSymbols = runLength(quantized_Y[i:i+8, j:j+8], DCpred)
                #Set the DC coefficient prediction for the next block
                DCpred = runSymbols[0][1] + DCpred
                #encode the runSymbols using huffman encoding
                huffStream=huffEnc(runSymbols, tables.DCC,tables.ACC)
                #append the encoded block to the JPEGenc list
                JPEGenc.append(JPEGencoded(blkType,i,j,huffStream))
    elif blkType == "Cb":
        for i in range(0, len(img), 8):
            for j in range(0, len(img[0]), 8):
                #apply DCT to each 8x8 block
                Dct_Y[i:i+8, j:j+8] = blockDCT(img[i:i+8, j:j+8])
                #apply quantization to each 8x8 block
                quantized_Y[i:i+8, j:j+8] = quantizeJPEG(Dct_Y[i:i+8, j:j+8], tables.qTableC, qScale)
                runSymbols = runLength(quantized_Y[i:i+8, j:j+8], DCpred)
                #Set the DC coefficient prediction for the next block
                DCpred = runSymbols[0][1] + DCpred
                #encode the runSymbols using huffman encoding
                huffStream=huffEnc(runSymbols, tables.DCC,tables.ACC)
                #append the encoded block to the JPEGenc list
                JPEGenc.append(JPEGencoded(blkType,i,j,huffStream))
    else:
        print("Invalid block type")

    return 

def JPEGencode(img,  qScale, subImg):
    """
    Encodes an image using the JPEG standard.
    Parameters:
    img (numpy.ndarray): The input image.
    qScale (float): The quantization scale.
    subImg (list): 1x3 matrix for subsampling [4, 2, 0], [4, 2, 2], [4, 4, 4].
    Returns:
    JPEGenc: A touple with N+1 elements, where N is the number of blocks in the image.
    """

    #Define the touple to store the encoded image
    JPEGenc = []

    # Create a JPEGtables object
    tables = JPEGtables()
    tables.reset_highfreq()

    # Append the  tables to the JPEGenc list
    JPEGenc.append(tables)

    # Ensure the image has the correct dimensions
    resized_img = ensure_dimensions(img)

    # Convert to YCrCb
    img1Y, img1Cr, img1Cb = convert2ycrcb(resized_img, subImg)


    # Encode each channel
    channel_encode(img1Y, qScale, JPEGenc, tables,"Y")
    channel_encode(img1Cr, qScale, JPEGenc, tables,"Cr")
    channel_encode(img1Cb, qScale, JPEGenc, tables,"Cb")

    return tuple(JPEGenc)

def retrieve_subImg(JPEGenc):
    """
    Retrieves the subsampling factor from the JPEGenc object.
    Parameters:
    JPEGenc: A touple with N+1 elements, where N is the number of blocks in the image.
    Returns:
    subImg (list): 1x3 matrix for subsampling [4, 2, 0], [4, 2, 2], [4, 4, 4].
    """

    # Count the number of blocks for each channel
    blocksY = len(list(filter(lambda obj:  obj.blkType == "Y" , JPEGenc[1:])))
    blocksCr = len(list(filter(lambda obj:  obj.blkType == "Cr" , JPEGenc[1:])))
    blocksCb = len(list(filter(lambda obj:  obj.blkType == "Cb" , JPEGenc[1:])))
    # Determine the subsampling format
    if blocksY == blocksCr and blocksY == blocksCb:
        subImg = [4, 4, 4]
    elif blocksY == 2 * blocksCr and blocksY == 2 * blocksCb:
        subImg = [4, 2, 2]
    elif blocksY == 4 * blocksCr and blocksY == 4 * blocksCb:
        subImg = [4, 2, 0]
    else:
        raise ValueError("Unsupported subsampling format")
    return subImg


def JPEGdecode(JPEGenc,qScale):
    """
    Decodes an image using the JPEG standard.
    Parameters:
    JPEGenc: A touple with N+1 elements, where N is the number of blocks in the image.
    qScale (float): The quantization scale.
    Returns:
    numpy.ndarray: The reconstructed image.
    """

    # Retrieve quantization tables
    qTableL = JPEGenc[0].qTableL
    qTableC = JPEGenc[0].qTableC
    
    # Retrieve DC and AC coefficients
    DCL = JPEGenc[0].DCL
    DCC = JPEGenc[0].DCC
    ACL = JPEGenc[0].ACL
    ACC = JPEGenc[0].ACC
    
    # Retrieve the subsampling factor
    subImg = retrieve_subImg(JPEGenc)

    # Retrieve the dimensions of the image
    width = max(JPEGenc[1:], key=lambda obj: obj.indHor).indHor + 8 
    height = max(JPEGenc[1:], key=lambda obj: obj.indVer).indVer + 8

    # Initialize list for YCrCb components
    imageY = np.zeros((height, width))

    if subImg == [4, 4, 4]:
        imageCr = np.zeros((height, width))
        imageCb = np.zeros((height, width))
    elif subImg == [4, 2, 2]:
        imageCr = np.zeros((height, width//2))
        imageCb = np.zeros((height, width//2))
    elif subImg == [4, 2, 0]:
        imageCr = np.zeros((height//2, width//2))
        imageCb = np.zeros((height//2, width//2))
    else:
        raise ValueError("Unsupported subsampling format")
    
    # Initialize DC prediction values
    DCpredY = 0
    DCpredCr = 0
    DCpredCb = 0
    
    # Decode each channel
    for i in range(1, len(JPEGenc)):
        # Retrieve block type and indices
        blkType = JPEGenc[i].blkType
        indHor = JPEGenc[i].indHor
        indVer = JPEGenc[i].indVer
        huffStream = JPEGenc[i].huffStream
        
        if blkType == "Y":
            # Decode Huffman stream
            runSymbols = huffDec(huffStream, DCL, ACL)
            # Decode run-length symbols
            quantized_Y = irunLength(runSymbols, DCpredY)
            # Update DC prediction
            DCpredY = runSymbols[0][1] + DCpredY #or quantized_Y[0, 0]
            # Dequantize the coefficients
            dequantized_Y = dequantizeJPEG(quantized_Y, qTableL, qScale)
            #apply inverse DCT to the 8x8 block
            imageY[indVer:indVer+8, indHor:indHor+8] = iBlockDCT(dequantized_Y)

        elif blkType == "Cr":
            # Decode Huffman stream
            runSymbols = huffDec(huffStream, DCC, ACC)
            # Decode run-length symbols
            quantized_Cr = irunLength(runSymbols, DCpredCr)
            # Update DC prediction
            DCpredCr = runSymbols[0][1] + DCpredCr #or quantized_Cr[0, 0]
            # Dequantize the coefficients
            dequantized_Cr = dequantizeJPEG(quantized_Cr, qTableC, qScale)
            #apply inverse DCT to the 8x8 block
            imageCr[indVer:indVer+8, indHor:indHor+8] = iBlockDCT(dequantized_Cr)

        elif blkType == "Cb":
            # Decode Huffman stream
            runSymbols = huffDec(huffStream, DCC, ACC)
            # Decode run-length symbols
            quantized_Cb = irunLength(runSymbols, DCpredCb)
            # Update DC prediction
            DCpredCb = runSymbols[0][1] + DCpredCb #or quantized_Cb[0, 0]
            # Dequantize the coefficients
            dequantized_Cb = dequantizeJPEG(quantized_Cb, qTableC, qScale)
            #apply inverse DCT to the 8x8 block
            imageCb[indVer:indVer+8, indHor:indHor+8] = iBlockDCT(dequantized_Cb)

        else:
            print("Invalid block type")
    
    # Convert YCrCb components back to an RGB image
    imgRec = convert2rgb(imageY, imageCr, imageCb, subImg)

    # Return the reconstructed image
    return imgRec


if __name__ == "__main__" :

    # Load the images and display
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

    # Define the quantization scales
    qScales = [0.1 , 0.3 , 0.6 , 1.0 , 2 , 5 , 10]

    # Define the subsampling factors
    subimg1 = [4, 2, 2]  
    subimg2 = [4, 4, 4]

    # Initialize the list to store the MSE values and the number of bits of the encoded images
    mse_values = [ [] , [] ]
    num_bits = [ [] , [] ]

    # Loop through the quantization scales
    for qScale in qScales:
        # Encode the image 1
        JPEGenc1=JPEGencode(image1_RGB,  qScale, subimg1)
        imgRec1=JPEGdecode(JPEGenc1,qScale)
        mse = np.mean((image1_RGB - imgRec1) ** 2)
        mse_values[0].append(mse)
        bits = 0
        for block in JPEGenc1[1:]:
            bits += len(block.huffStream)
        num_bits[0].append(bits)
        plt.imshow(imgRec1)
        plt.title("Reconstructed Image 1 with qscale = "+str(qScale))
        plt.show()
        print("Number of bits in the encoded image 1 :", bits)

        # Encode the image 2
        JPEGenc2=JPEGencode(image2_RGB,  qScale, subimg2)
        imgRec2=JPEGdecode(JPEGenc2,qScale)
        mse = np.mean((image2_RGB - imgRec2) ** 2)
        mse_values[1].append(mse)
        bits = 0
        for block in JPEGenc2[1:]:
            bits += len(block.huffStream)
        num_bits[1].append(bits)
        plt.imshow(imgRec2)
        plt.title("Reconstructed Image 2 with qscale = "+str(qScale))
        plt.show()
        print("Number of bits in the encoded image 2 :", bits)


    # Define the quantization scale
    qScale = 1
        
    # Define the quantization scale
    qScale = 1

    # Plot the MSE values for image 1 and image 2 with respect to the quantization scales
    plt.plot(qScales, mse_values[0], label='Image 1')
    plt.xlabel('Quantization Scale')
    plt.ylabel('Mean Square Error')
    plt.title('Mean Square Error vs. Quantization Scale')
    plt.show()
    plt.plot(qScales, mse_values[1], label='Image 2')
    plt.xlabel('Quantization Scale')
    plt.ylabel('Mean Square Error')
    plt.title('Mean Square Error vs. Quantization Scale')
    plt.show()

    # Plot the number of bits for image 1 and image 2 with respect to the quantization scales
    plt.plot(qScales, num_bits[0], label='Image 1')
    plt.xlabel('Quantization Scale')
    plt.ylabel('Number of Bits')
    plt.title('Number of Bits vs. Quantization Scale')
    plt.show()
    plt.plot(qScales, num_bits[1], label='Image 2')
    plt.xlabel('Quantization Scale')
    plt.ylabel('Number of Bits')
    plt.title('Number of Bits vs. Quantization Scale')
    plt.show()