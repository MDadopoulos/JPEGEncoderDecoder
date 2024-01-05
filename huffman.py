
import numpy as np


huffman_tables = {
    'luminance_dc': {
        0: '00', 1: '010', 2: '011', 3: '100', 4: '101', 5: '110',
        6: '1110', 7: '11110', 8: '111110', 9: '1111110', 10: '11111110', 11: '111111110'
    },
    'chrominance_dc': {
        0: '00', 1: '01', 2: '10', 3: '110', 4: '1110', 5: '11110',
        6: '111110', 7: '1111110', 8: '11111110', 9: '111111110', 10: '1111111110', 11: '11111111110'
    }
}

def huffEnc(runSymbols, huffman_table):
    """
    Encodes run-length symbols using Huffman coding based on the provided Huffman table.
    """
    huffStream = ''
    for run, symbol in runSymbols:
        category = int(np.log2(abs(symbol))) if symbol != 0 else 0
        huffCode = huffman_table[category]
        additionalBits = '{0:b}'.format(symbol) if symbol > 0 else '{0:b}'.format(-symbol)
        huffStream += huffCode + additionalBits
    return huffStream

def huffDec(huffStream, huffman_table):
    """
    Decodes a stream of bits into run-length symbols using the provided Huffman table.
    """
    runSymbols = []
    while huffStream:
        for category, huffCode in huffman_table.items():
            if huffStream.startswith(huffCode):
                huffStream = huffStream[len(huffCode):]
                if category == 0:
                    symbol = 0
                else:
                    additionalBits = huffStream[:category]
                    symbol = int(additionalBits, 2) if additionalBits[0] == '1' else -int(additionalBits, 2)
                    huffStream = huffStream[category:]
                runSymbols.append((0, symbol))  # Assuming only DC components for simplicity
                break
    return runSymbols

# Example usage:
# Assuming runSymbols is the output from the runLength function
# huffman_table = huffman_tables['luminance_dc']  # or 'chrominance_dc' based on the component
# huffStream = huffEnc(runSymbols, huffman_table)
# decodedRunSymbols = huffDec(huffStream, huffman_table)






######
# Huffman tables for luminance and chrominance DC coefficient differences
huffman_table_luminance = {
    0: '00',
    1: '010',
    2: '011',
    3: '100',
    4: '101',
    5: '110',
    6: '1110',
    7: '11110',
    8: '111110',
    9: '1111110',
    10: '11111110',
    11: '111111110'
}

huffman_table_chrominance = {
    0: '00',
    1: '01',
    2: '10',
    3: '110',
    4: '1110',
    5: '11110',
    6: '111110',
    7: '1111110',
    8: '11111110',
    9: '111111110',
    10: '1111111110',
    11: '11111111110'
}

def huffEnc(runSymbols):
    """
    Encodes run length symbols using Huffman coding.
    """
    huffStream = ''
    for run, symbol in runSymbols:
        # Assuming the runSymbols are for luminance DC coefficients
        category = calculate_category(symbol)
        huffStream += huffman_table_luminance[category]
        # Encode the additional bits for the symbol if necessary
        if category > 0:
            huffStream += format(symbol & ((1 << category) - 1), '0' + str(category) + 'b')
    return huffStream

def calculate_category(value):
    """
    Calculate the category of a value for Huffman encoding.
    """
    if value == 0:
        return 0
    magnitude = abs(value)
    category = 0
    while magnitude:
        magnitude >>= 1
        category += 1
    return category

def huffDec(huffStream):
    """
    Decodes a Huffman encoded bit stream into run length symbols.
    """
    runSymbols = []
    i = 0
    while i < len(huffStream):
        # Find the category by matching the prefix in the Huffman table
        for category, code in huffman_table_luminance.items():
            if huffStream.startswith(code, i):
                i += len(code)
                if category == 0:
                    runSymbols.append((0, 0))
                else:
                    additional_bits = huffStream[i:i+category]
                    i += category
                    symbol = int(additional_bits, 2)
                    if symbol < (1 << (category - 1)):
                        symbol -= (1 << category) - 1
                    runSymbols.append((0, symbol))
                break
    return runSymbols

# Example usage:
# Assume we have a list of run length symbols
# runSymbols = [(0, 0), (0, -3), (2, 2), ...]

# Encoding
# huffStream = huffEnc(runSymbols)

# Decoding
# decodedRunSymbols = huffDec(huffStream)





##
###
def huffEnc(runSymbols, huffmanTable):
    huffStream = ""
    for symbol in runSymbols:
        huffStream += huffmanTable[symbol]
    return huffStream

def huffDec(huffStream, huffmanTable):
    runSymbols = []
    temp = ""
    for bit in huffStream:
        temp += bit
        if temp in huffmanTable.values():
            runSymbols.append(get_key(temp, huffmanTable))
            temp = ""
    return runSymbols

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

# Example Huffman Tables (These would be filled with actual values from the provided tables)
huffmanTableDC = {"0/0": "101", "0/1": "110", ...}  # DC coefficients
huffmanTableAC = {"1/1": "010", "1/2": "011", ...}  # AC coefficients

# Example usage
runSymbols = ["0/0", "1/1", ...]
encodedStream = huffEnc(runSymbols, huffmanTableDC)  # Encoding
decodedSymbols = huffDec(encodedStream, huffmanTableDC)  # Decoding
