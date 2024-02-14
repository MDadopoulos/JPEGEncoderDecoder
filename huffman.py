
import numpy as np


def calculate_category(value):
    """
    Calculate the  magnitudes category SSSS for Huffman encoding,
    12 categories for DC coefficients and 11 for AC coefficients
    """
    if value == 0:
        return 0
    magnitude = abs(value)
    category = 0
    while magnitude:
        magnitude >>= 1
        category += 1
    return category

def get_low_order_bits(DIFF, SSSS):
    # Adjust for negative DIFF
    if DIFF < 0:
        # Subtract 1 from DIFF for negative numbers
        DIFF = DIFF - 1

    # Creating a bitmask with SSSS bits set to 1 in its binary representation.
    #This is typically a number like 2**SSSS - 1 or (1 << SSSS) - 1.
    bitmask = (1 << SSSS) - 1

    # Performing a bitwise AND to get the low order bits
    low_order_bits = DIFF & bitmask

    # Convert to binary string and remove the prefix '0b' and pad the binary to have SSSS bits if needed
    low_order_bits = bin(low_order_bits)[2:].zfill(SSSS)
    return low_order_bits


# This is typically used in the Huffman decoding process where `value` is the additional bits
# that were used along with the Huffman code to represent the quantized coefficient.
# The `category` (SSSS=T) is the number of additional bits used, derived from the Huffman code.

def extend(value, T):
    """
    Converts the partially decoded DIFF value of precision T to the full precision difference
    Parameters:
    value (int): The decoded value V.
    T (int): The category SSSS for Huffman encoding.
    
    Returns:
    int: The extended value.
    """
    Vt = 2 ** (T - 1)
    if value < Vt and value>0 :
        return value - (2 ** T) + 1
    else:
        return value






def huffEnc(runSymbols, huffman_table_DC,huffman_table_AC):
    """
    Encodes run-length symbols using Huffman coding based on the provided Huffman table.
    first should categorize the amplitude to get the category of it and then encode it using the huffman table ,
    it should be at the form of 0,0 or 0/0 to find it in the dictionary to binary
    the first pair is the DC coefficient and the others are the AC coefficient
    """
    huffStream = ''
    for index,values in enumerate(runSymbols):
        run=values[0]
        symbol=values[1]
        category = calculate_category(symbol)
        lowOrderBits=get_low_order_bits(symbol, category)
        if index==0:
            huffCode = huffman_table_DC[category]
        else:
            huffCode = huffman_table_AC[(run,category)]
        huffStream += huffCode + lowOrderBits
    return huffStream




def huffDec(huffStream, huffman_table_DC,huffman_table_AC):
    """
    Decodes a stream of bits into run-length symbols using the provided Huffman table.
    """
    runSymbols = []
    for category, huffCode in huffman_table_DC.items():
        if huffStream.startswith(huffCode):
            #the DECODE procedure
            huffStream = huffStream[len(huffCode):]
            if category == 0:
                symbol = 0
            else:
                #the RECEIVE procedure
                additionalBits = huffStream[:category]
                symbol = int(additionalBits, 2) 
                if additionalBits[0] == '0':
                    # If the number is negative, convert to 2's complement
                    symbol = symbol - (1 << category) + 1
                    #the EXTEND procedure
                    symbol = extend(symbol, category)
                
                huffStream = huffStream[category:]
            runSymbols.append((0, symbol))
            break
    while huffStream:
        for category, huffCode in huffman_table_AC.items():
            if huffStream.startswith(huffCode):
                #the DECODE procedure
                huffStream = huffStream[len(huffCode):]
                R = category[0]
                SSSS=category[1]
                #the RECEIVE procedure
                additionalBits = huffStream[:SSSS]
                symbol = int(additionalBits, 2) 
                if additionalBits[0] == '0':
                    # If the number is negative, convert to 2's complement
                    symbol = symbol - (1 << SSSS) + 1
                    #the EXTEND procedure
                symbol = extend(symbol, SSSS)
                
                huffStream = huffStream[SSSS:]
                runSymbols.append((R, symbol))
                break
    return runSymbols


if __name__ == "__main__" :

    from zigzag_RLE import runLength
    from zigzag_RLE import irunLength
    from huffman_tables import huffman_table_DC_luminance
    from huffman_tables import huffman_table_AC_luminance

    qBlock = np.random.randint(-128, 129, size=(8, 8))
    print("Original block:")
    print(qBlock)
    # Assume qBlock is an 8x8 quantized block of DCT coefficients,
    # and DCpred is the predicted DC coefficient from the previous block.

    DCpred = 0  # Example DC coefficient prediction
    runSymbols = runLength(qBlock, DCpred)
    print("runSymbols")
    print(runSymbols)
    huffStream=huffEnc(runSymbols, huffman_table_DC_luminance,huffman_table_AC_luminance)
    print("huffStream")
    print(huffStream)
    runSymbol=huffDec(huffStream, huffman_table_DC_luminance,huffman_table_AC_luminance)
    print("runSymbol")
    print(runSymbol)
    decoded_qBlock = irunLength(runSymbol, DCpred)
    print("decoded_qBlock")
    print(decoded_qBlock)
    
    