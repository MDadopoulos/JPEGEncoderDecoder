import numpy as np

def zigzag_index():
    """
    Generate a list of indexes for traversing an 8x8 matrix in zig-zag order.
    """
    zigzag = [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]
    return zigzag

def index_to_coordinate(index_list):
    """
    Convert a list of indexes to a list of (row, column) coordinates
    for an 8x8 matrix.
    """
    return [(index // 8, index % 8) for index in index_list]



def runLength(qBlock, DCpred):
    """
    Encodes a quantized block of DCT coefficients using run-length encoding.
    """
    # Get the zigzag order and convert it to (row, column) coordinates
    zz_order = index_to_coordinate(zigzag_index())

    # Calculate DC term as the difference from the previous block's DC
    DCdiff = qBlock[0, 0] - DCpred

    # Start with the DC coefficient
    runSymbols = [(0, DCdiff)] 
   
    # Process AC coefficients
    zeros = 0
    for i, j in zz_order[1:]:  # Skip the DC coefficient
        if qBlock[i, j] == 0:
            zeros += 1
        else:
            while zeros > 15:
                runSymbols.append((15, 0))
                zeros -= 16
            runSymbols.append((zeros, qBlock[i, j]))
            zeros = 0
    if zeros > 0:
        runSymbols.append((0, 0))  # End-of-block (EOB)

    return runSymbols

def irunLength(runSymbols, DCpred):
    """
    Decodes run-length encoded symbols into a quantized block of DCT coefficients.
    """
    # Get the zigzag order and convert it to (row, column) coordinates
    zz_order = index_to_coordinate(zigzag_index())

    # Create a zeroed 8x8 block
    qBlock = np.zeros((8, 8), dtype=int)

    # Set the DC coefficient
    qBlock[0, 0] = runSymbols[0][1] + DCpred  # Add DCpred to the DC coefficient

    # Process AC coefficients
    ac_index = 1  # Start with the first AC coefficient
    for zeros, coeff in runSymbols[1:]:
        if zeros == 0 and coeff == 0:  # Check for end-of-block (EOB)
            break
        ac_index += zeros
        if ac_index >= len(zz_order):
            break
        i, j = zz_order[ac_index]
        qBlock[i, j] = coeff
        ac_index += 1

    return qBlock


if __name__ == "__main__" :

    qBlock = np.random.randint(-128, 129, size=(8, 8))
    print("Original block:")
    print(qBlock)
    
    # Example usage of run-length encoding
    # Assume qBlock is an 8x8 quantized block of DCT coefficients,
    # and DCpred is the predicted DC coefficient from the previous block.

    DCpred = 0  # Example DC coefficient prediction
    runSymbols = runLength(qBlock, DCpred)
    print("Run-length symbols:")
    print(runSymbols)
    decoded_qBlock = irunLength(runSymbols, DCpred)
    print("Decoded qBlock:")
    print(decoded_qBlock)
    
   