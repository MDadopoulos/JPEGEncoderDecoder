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
    T = np.array([[0.299, 0.587, 0.114],
                  [0.5, -0.4187, -0.0813],
                  [-0.1687, -0.3313, 0.5]])
   
    # Apply the transformation matrix to each pixel
    imgYCrCb = np.dot(imageRGB, T.T)

    # Add offsets for Cr and Cb channels
    imgYCrCb[:,:, 1] += 128
    imgYCrCb[:,:, 2] += 128
    
    # Convert to YCrCb with OpenCV
    #imgYCrCb = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2YCrCb)
 
    # Split the channels and round them to integers
    imageY, imgCr, imgCb = np.round(imgYCrCb[:,:,0]).astype(int),np.round(imgYCrCb[:,:,1]).astype(int),np.round(imgYCrCb[:,:,2]).astype(int)
    # Determine subsampling and upsample Cr and Cb channels
    if subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 0:
        imageCb=np.zeros((imgCb.shape[0]//2, imgCb.shape[1] // 2), dtype=np.int32)
        imageCr=np.zeros((imgCr.shape[0]//2, imgCr.shape[1] // 2), dtype=np.int32)

        imageCb[:, :] = imgCb[::2, ::2] 
        imageCr[:, :] = imgCr[::2, ::2] 
        
    elif subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 2:
        imageCb=np.zeros((imgCb.shape[0], imgCb.shape[1]//2), dtype=np.int32)#it is first height then weight
        imageCr=np.zeros((imgCr.shape[0], imgCr.shape[1]//2), dtype=np.int32)

        imageCb[:, :] = imgCb[:, ::2] 
        imageCr[:, :] = imgCr[:, ::2]
    elif subimg[0] == 4 and subimg[1] == 4 and subimg[2] == 4:
        imageCr = imgCr.copy()
        imageCb = imgCb.copy()
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
    imgY=imageY.copy()
    # Determine subsampling and upsample Cr and Cb channels
    if subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 0:
        # # Upsample imgCr and imgCb to the same size as imgY
        # imgCr = cv2.resize(imageCr, (imgY.shape[0], imgY.shape[1]), interpolation=cv2.INTER_NEAREST)
        # imgCb = cv2.resize(imageCb, (imgY.shape[0], imgY.shape[1]), interpolation=cv2.INTER_NEAREST)
        imgCr=np.zeros((imgY.shape[0], imgY.shape[1] ), dtype=np.int32)
        imgCb=np.zeros((imgY.shape[0], imgY.shape[1]), dtype=np.int32)
        imgCr[::2, ::2] = imageCr[:, :]
        imgCr[1::2, 1::2] = imageCr[:, :]
        imgCb[::2, ::2] = imageCb[:, :]
        imgCb[1::2, 1::2] = imageCb[:, :]


    elif subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 2:
        imgCr=np.zeros((imgY.shape[0], imgY.shape[1]),dtype=np.int32)
        imgCb=np.zeros((imgY.shape[0], imgY.shape[1]), dtype=np.int32)
        imgCr[:, ::2] = imageCr[:, :]
        imgCr[:, 1::2] = imageCr[:, :]
        imgCb[:, ::2] = imageCb[:, :]
        imgCb[:, 1::2] = imageCb[:, :]
    elif subimg[0] == 4 and subimg[1] == 4 and subimg[2] == 4:
        imgCr, imgCb=imageCr.copy(), imageCb.copy()
    else:
        raise ValueError("Unsupported subsampling format. Choose from '4:4:4', '4:2:2', '4:2:0'.")
    
    # Ensure all images have the same depth
    imgY = imgY.astype('int32')
    imgCr = imgCr.astype('int32')
    imgCb = imgCb.astype('int32')

    # Merge and convert back to RGB
    imageYCrCb = cv2.merge((imgY, imgCr, imgCb))
    
    # Define the inverse transformation matrix from YCrCb to RGB
    # T_inv = np.array([[1.0, 1.4019, -3.6819],
    #               [1.0, -7.1410, -3.4411],
    #               [1.0, -1.3458, 1.7719]])
    
    T = np.array([[0.299, 0.587, 0.114],
                  [0.5, -0.4187, -0.0813],
                  [-0.1687, -0.3313, 0.5]])
    T_inv = np.linalg.inv(T)
    
    # # Offset the Cr and Cb channels before the transformation
    offset_imageYCrCb = imageYCrCb.copy()
    offset_imageYCrCb[:, :, 1] -= 128
    offset_imageYCrCb[:, :, 2] -= 128
    
    # Apply the inverse transformation matrix to each pixel
    imageRGB = np.dot(offset_imageYCrCb, T_inv.T)
    imageRGB = np.round(imageRGB).astype(int)
    # Convert to RGB with OpenCV
    #imageRGB = cv2.cvtColor(imageYCrCb, cv2.COLOR_YCrCb2RGB)
    #cv2.COLOR_YCR_CB2BGR
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




if __name__ == "__main__":
    
    # Read image
    original_imageRGB = cv2.imread('baboon.png')
    #cv2.imshow('original',original_imageRGB)
    # Ensure dimensions are multiples of 8
    resized_imageRGB = ensure_dimensions(original_imageRGB)

    subimg = [4, 2, 2]  # Example subsampling matrix
    print(resized_imageRGB[:,:,0])
    imageY, imageCr, imageCb = convert2ycrcb(resized_imageRGB, subimg)

    #print(min(imageCr),max(imageCr))
    reconstructed_imageRGB = convert2rgb(imageY, imageCr, imageCb, subimg)
    #cv2.imshow('reconstructed',reconstructed_imageRGB)
    print(reconstructed_imageRGB[:,:,0])
