import numpy as np
import cv2

def convert2ycrcb(imageRGB, subimg):
    """
    Converts an RGB image to YCrCb format with subsampling.
    
    Parameters:
    imageRGB (numpy.ndarray): Input RGB image.
    subimg (list): 1x3 matrix for subsampling [4, 2, 0], [4, 2, 2], [4, 4, 4].
    Returns:
    tuple: Y, Cr, and Cb components of the image.
    """

    # Convert to YCrCb
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb    )
    imgY, imgCb, imgCr = imgYCrCb[:,:,0], imgYCrCb[:,:,1], imgYCrCb[:,:,2]
    # Determine subsampling and upsample Cr and Cb channels
    if subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 0:
        imageCb=np.zeros((imgCb.shape[0]//2, imgCb.shape[1] // 2), dtype=np.uint8)
        imageCr=np.zeros((imgCr.shape[0]//2, imgCr.shape[1] // 2), dtype=np.uint8)

        imageCb[:, :] = imgCb[::2, ::2] 
        imageCr[:, :] = imgCr[::2, ::2] 
        
    elif subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 2:
        imageCb=np.zeros((imgCb.shape[0], imgCb.shape[1]//2), dtype=np.uint8)#it is first height then weight
        imageCr=np.zeros((imgCr.shape[0], imgCr.shape[1]//2), dtype=np.uint8)

        imageCb[:, :] = imgCb[:, ::2] 
        imageCr[:, :] = imgCr[:, ::2]
    elif subimg[0] == 4 and subimg[1] == 4 and subimg[2] == 4:
        imageCr = imgCr
        imageCb = imgCb
    else:
        raise ValueError("Unsupported subsampling format. Choose from '4:4:4', '4:2:2', '4:2:0'.")


   
    return imageY, imageCr, imageCb


def convert2rgb(imageY, imageCr, imageCb, subimg):
    """
    Converts YCrCb components back to an RGB image with subsampling.
    
    Parameters:
    imageY, imageCr, imageCb (numpy.ndarray): Y, Cr, and Cb components of the image.
    subimg (list): 1x3 matrix for subsampling [4, 2, 0], [4, 2, 2], [4, 4, 4].
    
    Returns:
    numpy.ndarray: Combined RGB image.
    """
    imgY=imageY
    # Determine subsampling and upsample Cr and Cb channels
    if subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 0:
        imgCr=np.zeros((imgY.shape[0], imgY.shape[1] ), dtype=np.uint8)
        imgCb=np.zeros((imgY.shape[0], imgY.shape[1]), dtype=np.uint8)
        imgCr[::2, ::2] = imageCr[:, :]
        imgCr[1::2, 1::2] = imageCr[:, :]
        imgCb[::2, ::2] = imageCb[:, :]
        imgCb[1::2, 1::2] = imageCb[:, :]


    elif subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 2:
        imgCr=np.zeros((imgY.shape[0], imgY.shape[1]), dtype=np.uint8)
        imgCb=np.zeros((imgY.shape[0], imgY.shape[1]), dtype=np.uint8)
        imgCr[:, ::2] = imageCr[:, :]
        imgCr[:, 1::2] = imageCr[:, :]
        imgCb[:, ::2] = imageCb[:, :]
        imgCb[:, 1::2] = imageCb[:, :]
    elif subimg[0] == 4 and subimg[1] == 4 and subimg[2] == 4:
        imgCr, imgCb=imageCr, imageCb
    else:
        raise ValueError("Unsupported subsampling format. Choose from '4:4:4', '4:2:2', '4:2:0'.")
    
    # Merge and convert back to RGB
    imageYCrCb = cv2.merge((imgY, imgCr, imgCb))
    
    imageRGB = cv2.cvtColor(imageYCrCb, cv2.COLOR_YCrCb2RBG)


    return imageRGB

def ensure_dimensions(image, multiple_of=8):
    """
    Ensure that the dimensions of the image are multiples of a given number.

    Parameters:
    image (numpy.ndarray): The input image.
    multiple_of (int): The number that the dimensions must be a multiple of.

    Returns:
    numpy.ndarray: The resized image with dimensions as multiples of the given number.
    """
    width, height = image.shape[:2]
    new_width = width - (width % multiple_of)
    new_height = height - (height % multiple_of)
    diff_width = width - new_width
    diff_height = height - new_height
    left = diff_width // 2
    right = diff_width - left
    top = diff_height // 2
    bottom = diff_height - top
    return image[left:width-right, top:height-bottom, :]

img = cv2.imread('baboon.png')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

