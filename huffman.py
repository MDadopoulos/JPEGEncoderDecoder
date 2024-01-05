
import numpy as np
from tables import huffman_table_DC_luminance,huffman_table_DC_chrominance ,huffman_table_AC_luminance,huffman_table_AC_chrominance



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


##sshould i do something for the negative one??diff-1? the value or the binary -1?
def huffEnc(runSymbols, huffman_table_AC,huffman_table_DC):
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
        if index==0:
            huffCode = huffman_table_DC[category]
            huffStream += huffCode
            continue
        huffCode = huffman_table_AC[(run,category)]
        huffStream += huffCode 
    return huffStream


# decoded_message = ''
# current_code = ''

# for bit in encoded_data:
#     current_code += bit
#     if current_code in huffman_dict:
#         decoded_message += huffman_dict[current_code]
#         current_code = ''





def huffDec(huffStream, huffman_table_AC,huffman_table_DC):
    """
    Decodes a stream of bits into run-length symbols using the provided Huffman table.
    """
    runSymbols = []
    while huffStream:
        for category, huffCode in huffman_table_DC.items():
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






##


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




# ##
# ###
# def huffEnc(runSymbols, huffmanTable):
#     huffStream = ""
#     for symbol in runSymbols:
#         huffStream += huffmanTable[symbol]
#     return huffStream

# def huffDec(huffStream, huffmanTable):
#     runSymbols = []
#     temp = ""
#     for bit in huffStream:
#         temp += bit
#         if temp in huffmanTable.values():
#             runSymbols.append(get_key(temp, huffmanTable))
#             temp = ""
#     return runSymbols

# def get_key(val, my_dict):
#     for key, value in my_dict.items():
#         if val == value:
#             return key





# def huffEnc(runSymbols, huffman_table):
#     """
#     Encodes run-length symbols using Huffman coding based on the provided Huffman table.
#     first should categorize the amplitude to get the category of it and then encode it using the huffman table ,
#     it should be at the form of 0,0 or 0/0 to find it in the dictionary to binary
#     the first pair is the DC coefficient and the others are the AC coefficient
#     """
#     huffStream = ''
#     for run, symbol in runSymbols:
#         category = calculate_category(symbol)
#         huffCode = huffman_table[category]
#         additionalBits = '{0:b}'.format(symbol) if symbol > 0 else '{0:b}'.format(-symbol)
#         huffStream += huffCode + additionalBits
#     return huffStream