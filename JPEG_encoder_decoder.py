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



class JPEGencoded:
    def __init__(self,blkType, indHor, indVer, huffStream):
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

    DCT_Y = np.zeros(img.shape)
    quantized_Y = np.zeros(img.shape)
    DCpred = 0  # Example DC coefficient prediction
    if blkType == "Y":
        for i in range(0, len(img), 8):
            for j in range(0, len(img[0]), 8):
                #apply DCT to each 8x8 block
                DCT_Y[i:i+8, j:j+8] = blockDCT(img[i:i+8, j:j+8])
                #apply quantization to each 8x8 block
                quantized_Y[i:i+8, j:j+8] = quantizeJPEG(DCT_Y[i:i+8, j:j+8], tables.qTableL, qScale)
                runSymbols = runLength(quantized_Y[i:i+8, j:j+8], DCpred)
                DCpred = runSymbols[0][1]
                huffStream=huffEnc(runSymbols, tables.DCL,tables.ACL)
                JPEGenc.append(JPEGencoded(blkType,i,j,huffStream))
    elif blkType == "Cr":
        for i in range(0, len(img), 8):
            for j in range(0, len(img[0]), 8):
                #apply DCT to each 8x8 block
                DCT_Y[i:i+8, j:j+8] = blockDCT(img[i:i+8, j:j+8])
                #apply quantization to each 8x8 block
                quantized_Y[i:i+8, j:j+8] = quantizeJPEG(DCT_Y[i:i+8, j:j+8], tables.qTableC, qScale)
                runSymbols = runLength(quantized_Y[i:i+8, j:j+8], DCpred)
                DCpred = runSymbols[0][1]
                huffStream=huffEnc(runSymbols, tables.DCC,tables.ACC)
                JPEGenc.append(JPEGencoded(blkType,i,j,huffStream))
    elif blkType == "Cb":
        for i in range(0, len(img), 8):
            for j in range(0, len(img[0]), 8):
                #apply DCT to each 8x8 block
                DCT_Y[i:i+8, j:j+8] = blockDCT(img[i:i+8, j:j+8])
                #apply quantization to each 8x8 block
                quantized_Y[i:i+8, j:j+8] = quantizeJPEG(DCT_Y[i:i+8, j:j+8], tables.qTableC, qScale)
                runSymbols = runLength(quantized_Y[i:i+8, j:j+8], DCpred)
                DCpred = runSymbols[0][1]
                huffStream=huffEnc(runSymbols, tables.DCC,tables.ACC)
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
    subImg (int): The subsampling factor.
    Returns:
    JPEGenc: A touple with N+1 elements, where N is the number of blocks in the image.
    """

    #Define the touple to store the encoded image
    JPEGenc = []

    # Create a JPEGtables object
    tables = JPEGtables()

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

def JPEGdecode(JPEGenc):
    """
    Decodes an image using the JPEG standard.
    Parameters:
    JPEGenc: A touple with N+1 elements, where N is the number of blocks in the image.
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
    
    # Initialize list for reconstructed image
    imgRec = []
    
    # Decode each channel
    for i in range(1, len(JPEGenc)):
        # Retrieve block type and indices
        blkType = JPEGenc[i].blkType
        indHor = JPEGenc[i].indHor
        indVer = JPEGenc[i].indVer
        huffStream = JPEGenc[i].huffStream
        
        # Decode Huffman stream
        if blkType == "Y":
            runSymbols = huffDec(huffStream, DCL, ACL)
            DCpred = 0
            quantized_Y = irunLength(runSymbols, DCpred)
            iDCT_Y = dequantize_channel(quantized_Y, qTableL, 1)
            imgRec.append(iBlockDCT(iDCT_Y))
        elif blkType == "Cr":
            runSymbols = huffDec(huffStream, DCC, ACC)
            DCpred = 0
            quantized_Cr = irunLength(runSymbols, DCpred)
            iDCT_Cr = dequantize_channel(quantized_Cr, qTableC, 1)
            imgRec.append(iBlockDCT(iDCT_Cr))
        elif blkType == "Cb":
            runSymbols = huffDec(huffStream, DCC, ACC)
            DCpred = 0
            quantized_Cb = irunLength(runSymbols, DCpred)
            iDCT_Cb = dequantize_channel(quantized_Cb, qTableC, 1)
            imgRec.append(iBlockDCT(iDCT_Cb))
        else:
            print("Invalid block type")
    
    # Return the reconstructed image
    return imgRec
