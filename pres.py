


from PIL import Image
import numpy as np
import cv2

def convert2ycrcb(imageRGB, subimg):
    """
    Converts an RGB image to YCrCb format with subsampling.
    
    Parameters:
    imageRGB (PIL.Image): Input RGB image.
    subimg (list): 1x3 matrix for subsampling [4, 2, 0], [4, 2, 2], [4, 4, 4].
    Returns:
    tuple: Y, Cr, and Cb components of the image.
    """

    # Convert to YCrCb
    imageYCrCb = imageRGB.convert('YCbCr')
    imageY, imageCb, imageCr = imageYCrCb.split()

    # Subsample the Cr and Cb channels
    sub_x, sub_y = subimg[1], subimg[2]  # Assuming [4, 2, 0] format for subimg

    # Determine subsampling and upsample Cr and Cb channels
    if subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 0:
        imageCr = imageCr.resize((imageCr.width // sub_x, imageCr.height // sub_y), Image.BICUBIC)
        imageCb = imageCb.resize((imageCb.width // sub_x, imageCb.height // sub_y), Image.BICUBIC)
    elif subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 2:
        imageCr = imageCr.resize((imageCr.width // sub_x, imageCr.height), Image.BICUBIC)
        imageCb = imageCb.resize((imageCb.width // sub_x, imageCb.height), Image.BICUBIC)
    elif subimg[0] == 4 and subimg[1] == 4 and subimg[2] == 4:
        imageCr = imageCr.resize((imageCr.width, imageCr.height), Image.BICUBIC)
        imageCb = imageCb.resize((imageCb.width, imageCb.height), Image.BICUBIC)
    else:
        raise ValueError("Unsupported subsampling format. Choose from '4:4:4', '4:2:2', '4:2:0'.")


   
    return imageY, imageCr, imageCb

def convert2rgb(imageY, imageCr, imageCb, subimg):
    """
    Converts YCrCb components back to an RGB image with subsampling.
    
    Parameters:
    imageY, imageCr, imageCb (PIL.Image): Y, Cr, and Cb components of the image.
    subimg (list): 1x3 matrix for subsampling [4, 2, 0], [4, 2, 2], [4, 4, 4].
    
    Returns:
    PIL.Image: Combined RGB image.
    """
    
    # Determine subsampling and upsample Cr and Cb channels
    if subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 0:
        imageCr = imageCr.resize((imageY.width, imageY.height), Image.BICUBIC)
        imageCb = imageCb.resize((imageY.width, imageY.height), Image.BICUBIC)
    elif subimg[0] == 4 and subimg[1] == 2 and subimg[2] == 2:
        imageCr = imageCr.resize((imageY.width, imageY.height), Image.BICUBIC)
        imageCb = imageCb.resize((imageY.width, imageY.height), Image.BICUBIC)
    elif subimg[0] == 4 and subimg[1] == 4 and subimg[2] == 4:
        imageCr = imageCr.resize((imageY.width, imageY.height), Image.BICUBIC)
        imageCb = imageCb.resize((imageY.width, imageY.height), Image.BICUBIC)
    else:
        raise ValueError("Unsupported subsampling format. Choose from '4:4:4', '4:2:2', '4:2:0'.")
    
    # Merge and convert back to RGB
    imageYCrCb = Image.merge('YCbCr', (imageY, imageCb, imageCr))
    imageRGB = imageYCrCb.convert('RGB')

    return imageRGB

def ensure_dimensions(image, multiple_of=8):
    """
    Ensure that the dimensions of the image are multiples of a given number.

    Parameters:
    image (PIL.Image): The input image.
    multiple_of (int): The number that the dimensions must be a multiple of.

    Returns:
    PIL.Image: The resized image with dimensions as multiples of the given number.
    """
    width, height = image.size
    new_width = width - (width % multiple_of)
    new_height = height - (height % multiple_of)
    return image.crop((0, 0, new_width, new_height))

# Example usage
# Load an RGB image
# image_path = "path_to_your_image.jpg"
# original_imageRGB = Image.open(image_path)

# Ensure dimensions are multiples of 8
# resized_imageRGB = ensure_dimensions(original_imageRGB)

# Now we can use the resized image with the conversion functions
# subimg = [4, 2, 0]  # Example subsampling matrix
# imageY, imageCr, imageCb = convert2ycrcb(resized_imageRGB, subimg)
# reconstructed_imageRGB = convert2rgb(imageY, imageCr, imageCb, subimg)

# Note: The above example code is commented out. To run it, you need to provide the path to your image file.










# Huffman Coding in python

string = 'BCAADDDCCACACAC'


# Creating tree nodes
class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)


# Main function implementing huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d


# Calculating frequency
freq = {}
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

nodes = freq

while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))

    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

huffmanCode = huffman_code_tree(nodes[0][0])

print(' Char | Huffman code ')
print('----------------------')
for (char, frequency) in freq:
    print(' %-4r |%12s' % (char, huffmanCode[char]))




#########
    
from collections import Counter


class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def huffman_code_tree(node, binString=''):
    '''
    Function to find Huffman Code
    '''
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    return d


def make_tree(nodes):
    '''
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    '''
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]


if __name__ == '__main__':
    string = 'BCAADDDCCACACAC'
    freq = dict(Counter(string))
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    node = make_tree(freq)
    encoding = huffman_code_tree(node)
    for i in encoding:
        print(f'{i} : {encoding[i]}')