# def receive(ssss, bitstream):
#     """
#     Function to receive the next 'ssss' bits from 'bitstream' and interpret them as a value.

#     Parameters:
#     ssss (int): The number of bits to read (category of the coefficient).
#     bitstream (str): The bitstream, a string of '0's and '1's.

#     Returns:
#     int: The value obtained by interpreting the next 'ssss' bits of the bitstream.
#     """
#     I = 0
#     V = 0
#     while I < ssss:
#         V = (V << 1) | int(bitstream[I])  # SLL V by 1 and add the next bit
#         I += 1
#     return V

#OTHER implementation for receive
# def calculate_value(category, additional_bits):
#     """
#     Calculate the value based on the category and additional bits for decoding.
#     """
#     if category == 0:
#         return 0
#     magnitude = int(additional_bits, 2)
#     if magnitude < (1 << (category - 1)):
#         magnitude -= (1 << category) - 1
#     return magnitude


#OTHER implementation for function decode

    # for i in range(len(huffStream)):
    #     # Find the category by matching the prefix in the Huffman table
    #     for category, code in huffman_table_AC.items():
    #         if huffStream.startswith(code, i):
    #             i += len(code)
    #             if category == 0:
    #                 runSymbols.append((0, 0))
    #             else:
    #                 additional_bits = huffStream[i:i+category]
    #                 i += category
    #                 symbol = int(additional_bits, 2)
    #                 if symbol < (1 << (category - 1)):
    #                     symbol -= (1 << category) - 1
    #                 runSymbols.append((0, symbol))
    #             break

#OTHER implementation for function decode
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




# decoded_message = ''
# current_code = ''

# for bit in encoded_data:
#     current_code += bit
#     if current_code in huffman_dict:
#         decoded_message += huffman_dict[current_code]
#         current_code = ''


#other implementations of huffman encoding
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