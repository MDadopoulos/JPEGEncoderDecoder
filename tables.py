#can you create a dictionary with tuples as keys and string as values based on the data i will give you.The data is a pair per 2 rows,
#the first row is the tuple and the second the string :
# huffman_tables = {
#     'luminance_dc': {
#         0: '00', 1: '010', 2: '011', 3: '100', 4: '101', 5: '110',
#         6: '1110', 7: '11110', 8: '111110', 9: '1111110', 10: '11111110', 11: '111111110'
#     },
#     'chrominance_dc': {
#         0: '00', 1: '01', 2: '10', 3: '110', 4: '1110', 5: '11110',
#         6: '111110', 7: '1111110', 8: '11111110', 9: '111111110', 10: '1111111110', 11: '11111111110'
#     }
# }

huffman_table_DC_luminance = {
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

huffman_table_DC_chrominance = {
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


huffman_table_AC_luminance = dict([
    ((0, 0), "1010"),
    ((0, 1), "00"),
    ((0, 2), "01"),
    ((0, 3), "100"),
    ((0, 4), "1011"),
    ((0, 5), "11010"),
    ((0, 6), "1111000"),
    ((0, 7), "11111000"),
    ((0, 8), "1111110110"),
    ((0, 9), "1111111110000010"),
    ((0, 10), "1111111110000011"),
    ((1, 1), "1100"),
    ((1, 2), "11011"),
    ((1, 3), "1111001"),
    ((1, 4), "111110110"),
    ((1, 5), "11111110110"),
    ((1, 6), "1111111110000100"),
    ((1, 7), "1111111110000101"),
    ((1, 8), "1111111110000110"),
    ((1, 9), "1111111110000111"),
    ((1, 10), "1111111110001000"),
    ((2, 1), "11100"),
    ((2, 2), "11111001"),
    ((2, 3), "1111110111"),
    ((2, 4), "111111110100"),
    ((2, 5), "1111111110001001"),
    ((2, 6), "1111111110001010"),
    ((2, 7), "1111111110001011"),
    ((2, 8), "1111111110001100"),
    ((2, 9), "1111111110001101"),
    ((2, 10), "1111111110001110"),
    ((3, 1), "111010"),
    ((3, 2), "111110111"),
    ((3, 3), "111111110101"),
    ((3, 4), "1111111110001111"),
    ((3, 5), "1111111110010000"),
    ((3, 6), "1111111110010001"),
    ((3, 7), "1111111110010010"),
    ((3, 8), "1111111110010011"),
    ((3, 9), "1111111110010100"),
    ((3, 10), "1111111110010101"),
    ((4, 1), "111011"),
    ((4, 2), "1111111000"),
    ((4, 3), "1111111110010110"),
    ((4, 4), "1111111110010111"),
    ((4, 5), "1111111110011000"),
    ((4, 6), "1111111110011001"),
    ((4, 7), "1111111110011010"),
    ((4, 8), "1111111110011011"),
    ((4, 9), "1111111110011100"),
    ((4, 10), "1111111110011101"),
    ((5, 1), "1111010"),
    ((5, 2), "11111110111"),
    ((5, 3), "1111111110011110"),
    ((5, 4), "1111111110011111"),
    ((5, 5), "1111111110100000"),
    ((5, 6), "1111111110100001"),
    ((5, 7), "1111111110100010"),
    ((5, 8), "1111111110100011"),
    ((5, 9), "1111111110100100"),
    ((5, 10), "1111111110100101"),
    ((6, 1), "1111011"),
    ((6, 2), "11111111000"),
    ((6, 3), "1111111110100110"),
    ((6, 4), "1111111110100111"),
    ((6, 5), "1111111110101000"),
    ((6, 6), "1111111110101001"),
    ((6, 7), "1111111110101010"),
    ((6, 8), "1111111110101011"),
    ((6, 9), "1111111110101100"),
    ((6, 10), "1111111110101101"),
    ((7, 1), "11111010"),
    ((7, 2), "11111111001"),
    ((7, 3), "1111111110101110"),
    ((7, 4), "1111111110101111"),
    ((7, 5), "1111111110110000"),
    ((7, 6), "1111111110110001"),
    ((7, 7), "1111111110110010"),
    ((7, 8), "1111111110110011"),
    ((7, 9), "1111111110110100"),
    ((7, 10), "1111111110110101"),
    ((8, 1), "111111000"),
    ((8, 2), "111111111000000"),
    ((8, 3), "1111111110110110"),
    ((8, 4), "1111111110110111"),
    ((8, 5), "1111111110111000"),
    ((8, 6), "1111111110111001"),
    ((8, 7), "1111111110111010"),
    ((8, 8), "1111111110111011"),
    ((8, 9), "1111111110111100"),
    ((8, 10), "1111111110111101"),
    ((9, 1), "111111001"),
    ((9, 2), "1111111110111110"),
    ((9, 3), "1111111110111111"),
    ((9, 4), "1111111111000000"),
    ((9, 5), "1111111111000001"),
    ((9, 6), "1111111111000010"),
    ((9, 7), "1111111111000011"),
    ((9, 8), "1111111111000100"),
    ((9, 9), "1111111111000101"),
    ((9, 10), "1111111111000110"),
    ((10, 1), "111111010"),
    ((10, 2), "1111111111000111"),
    ((10, 3), "1111111111001000"),
    ((10, 4), "1111111111001001"),
    ((10, 5), "1111111111001010"),
    ((10, 6), "1111111111001011"),
    ((10, 7), "1111111111001100"),
    ((10, 8), "1111111111001101"),
    ((10, 9), "1111111111001110"),
    ((10, 10), "1111111111001111"),
    ((11, 1), "1111111001"),
    ((11, 2), "1111111111010000"),
    ((11, 3), "1111111111010001"),
    ((11, 4), "1111111111010010"),
    ((11, 5), "1111111111010011"),
    ((11, 6), "1111111111010100"),
    ((11, 7), "1111111111010101"),
    ((11, 8), "1111111111010110"),
    ((11, 9), "1111111111010111"),
    ((11, 10), "1111111111011000"),
    ((12, 1), "1111111010"),
    ((12, 2), "1111111111011001"),
    ((12, 3), "1111111111011010"),
    ((12, 4), "1111111111011011"),
    ((12, 5), "1111111111011100"),
    ((12, 6), "1111111111011101"),
    ((12, 7), "1111111111011110"),
    ((12, 8), "1111111111011111"),
    ((12, 9), "1111111111100000"),
    ((12, 10), "1111111111100001"),
    ((13, 1), "11111111000"),
    ((13, 2), "1111111111100010"),
    ((13, 3), "1111111111100011"),
    ((13, 4), "1111111111100100"),
    ((13, 5), "1111111111100101"),
    ((13, 6), "1111111111100110"),
    ((13, 7), "1111111111100111"),
    ((13, 8), "1111111111101000"),
    ((13, 9), "1111111111101001"),
    ((13, 10), "1111111111101010"),
    ((14, 1), "1111111111101011"),
    ((14, 2), "1111111111101100"),
    ((14, 3), "1111111111101101"),
    ((14, 4), "1111111111101110"),
    ((14, 5), "1111111111101111"),
    ((14, 6), "1111111111110000"),
    ((14, 7), "1111111111110001"),
    ((14, 8), "1111111111110010"),
    ((14, 9), "1111111111110011"),
    ((14, 10), "1111111111110100"),
    ((15, 1), "1111111111110101"),
    ((15, 2), "1111111111110110"),
    ((15, 3), "1111111111110111"),
    ((15, 4), "1111111111111000"),
    ((15, 5), "1111111111111001"),
    ((15, 6), "1111111111111010"),
    ((15, 7), "1111111111111011"),
    ((15, 8), "1111111111111100"),
    ((15, 9), "1111111111111101"),
    ((15, 10), "1111111111111110"),
    ((15, 0), "11111111001")
])



#From the following data which consists of 3 columns i want to create a dictionary where the first column will be the key and the third column the value.The key should be a touple of integers(replace A with 10,B with 11,C with 12,D with 13,E with 14 and F with 15) and the value a string.

data_chrominance = [
    ("0/0", 12, "00"),
    ("0/1", 12, "01"),
    ("0/2", 13, "100"),
    ("0/3", 14, "1010"),
    ("0/4", 15, "11000"),
    ("0/5", 15, "11001"),
    ("0/6", 16, "111000"),
    ("0/7", 17, "1111000"),
    ("0/8", 19, "111110100"),
    ("0/9", 10, "1111110110"),
    ("0/A", 12, "111111110100"),
    ("1/1", 14, "1011"),
    ("1/2", 16, "111001"),
    ("1/3", 18, "11110110"),
    ("1/4", 19, "111110101"),
    ("1/5", 11, "11111110110"),
    ("1/6", 12, "111111110101"),
    ("1/7", 16, "1111111110001000"),
    ("1/8", 16, "1111111110001001"),
    ("1/9", 16, "1111111110001010"),
    ("1/A", 16, "1111111110001011"),
    ("2/1", 15, "11010"),
    ("2/2", 18, "11110111"),
    ("2/3", 10, "1111110111"),
    ("2/4", 12, "111111110110"),
    ("2/5", 15, "111111111000010"),
    ("2/6", 16, "1111111110001100"),
    ("2/7", 16, "1111111110001101"),
    ("2/8", 16, "1111111110001110"),
    ("2/9", 16, "1111111110001111"),
    ("2/A", 16, "1111111110010000"),
    ("3/1", 15, "11011"),
    ("3/2", 18, "11111000"),
    ("3/3", 10, "1111111000"),
    ("3/4", 12, "111111110111"),
    ("3/5", 16, "1111111110010001"),
    ("3/6", 16, "1111111110010010"),
    ("3/7", 16, "1111111110010011"),
    ("3/8", 16, "1111111110010100"),
    ("3/9", 16, "1111111110010101"),
    ("3/A", 16, "1111111110010110"),
    ("4/1", 16, "111010"),
    ("4/2", 19, "111110110"),
    ("4/3", 16, "1111111110010111"),
    ("4/4", 16, "1111111110011000"),
    ("4/5", 16, "1111111110011001"),
    ("4/6", 16, "1111111110011010"),
    ("4/7", 16, "1111111110011011"),
    ("4/8", 16, "1111111110011100"),
    ("4/9", 16, "1111111110011101"),
    ("4/A", 16, "1111111110011110"),
    ("5/1", 16, "111011"),
    ("5/2", 10, "1111111001"),
    ("5/3", 16, "1111111110011111"),
    ("5/4", 16, "1111111110100000"),
    ("5/5", 16, "1111111110100001"),
    ("5/6", 16, "1111111110100010"),
    ("5/7", 16, "1111111110100011"),
    ("5/8", 16, "1111111110100100"),
    ("5/9", 16, "1111111110100101"),
    ("5/A", 16, "1111111110100110"),
    ("6/1", 17, "1111001"),
    ("6/2", 11, "11111110111"),
    ("6/3", 16, "1111111110100111"),
    ("6/4", 16, "1111111110101000"),
    ("6/5", 16, "1111111110101001"),
    ("6/6", 16, "1111111110101010"),
    ("6/7", 16, "1111111110101011"),
    ("6/8", 16, "1111111110101100"),
    ("6/9", 16, "1111111110101101"),
    ("6/A", 16, "1111111110101110"),
    ("7/1", 17, "1111010"),
    ("7/2", 11, "11111111000"),
    ("7/3", 16, "1111111110101111"),
    ("7/4", 16, "1111111110110000"),
    ("7/5", 16, "1111111110110001"),
    ("7/6", 16, "1111111110110010"),
    ("7/7", 16, "1111111110110011"),
    ("7/8", 16, "1111111110110100"),
    ("7/9", 16, "1111111110110101"),
    ("7/A", 16, "1111111110110110"),
    ("8/1", 18, "11111001"),
    ("8/2", 16, "1111111110110111"),
    ("8/3", 16, "1111111110111000"),
    ("8/4", 16, "1111111110111001"),
    ("8/5", 16, "1111111110111010"),
    ("8/6", 16, "1111111110111011"),
    ("8/7", 16, "1111111110111100"),
    ("8/8", 16, "1111111110111101"),
    ("8/9", 16, "1111111110111110"),
    ("8/A", 16, "1111111110111111"),
    ("9/1", 19, "111110111"),
    ("9/2", 16, "1111111111000000"),
    ("9/3", 16, "1111111111000001"),
    ("9/4", 16, "1111111111000010"),
    ("9/5", 16, "1111111111000011"),
    ("9/6", 16, "1111111111000100"),
    ("9/7", 16, "1111111111000101"),
    ("9/8", 16, "1111111111000110"),
    ("9/9", 16, "1111111111000111"),
    ("9/A", 16, "1111111111001000"),
    ("A/1", 19, "111111000"),
    ("A/2", 16, "1111111111001001"),
    ("A/3", 16, "1111111111001010"),
    ("A/4", 16, "1111111111001011"),
    ("A/5", 16, "1111111111001100"),
    ("A/6", 16, "1111111111001101"),
    ("A/7", 16, "1111111111001110"),
    ("A/8", 16, "1111111111001111"),
    ("A/9", 16, "1111111111010000"),
    ("A/A", 16, "1111111111010001"),
    ("B/1", 19, "111111001"),
    ("B/2", 16, "1111111111010010"),
    ("B/3", 16, "1111111111010011"),
    ("B/4", 16, "1111111111010100"),
    ("B/5", 16, "1111111111010101"),
    ("B/6", 16, "1111111111010110"),
    ("B/7", 16, "1111111111010111"),
    ("B/8", 16, "1111111111011000"),
    ("B/9", 16, "1111111111011001"),
    ("B/A", 16, "1111111111011010"),
    ("C/1", 19, "111111010"),
    ("C/2", 16, "1111111111011011"),
    ("C/3", 16, "1111111111011100"),
    ("C/4", 16, "1111111111011101"),
    ("C/5", 16, "1111111111011110"),
    ("C/6", 16, "1111111111011111"),
    ("C/7", 16, "1111111111100000"),
    ("C/8", 16, "1111111111100001"),
    ("C/9", 16, "1111111111100010"),
    ("C/A", 16, "1111111111100011"),
    ("D/1", 11, "11111111001"),
    ("D/2", 16, "1111111111100100"),
    ("D/3", 16, "1111111111100101"),
    ("D/4", 16, "1111111111100110"),
    ("D/5", 16, "1111111111100111"),
    ("D/6", 16, "1111111111101000"),
    ("D/7", 16, "1111111111101001"),
    ("D/8", 16, "1111111111101010"),
    ("D/9", 16, "1111111111101011"),
    ("D/A", 16, "1111111111101100"),
    ("E/1", 14, "11111111100000"),
    ("E/2", 16, "1111111111101101"),
    ("E/3", 16, "1111111111101110"),
    ("E/4", 16, "1111111111101111"),
    ("E/5", 16, "1111111111110000"),
    ("E/6", 16, "1111111111110001"),
    ("E/7", 16, "1111111111110010"),
    ("E/8", 16, "1111111111110011"),
    ("E/9", 16, "1111111111110100"),
    ("E/A", 16, "1111111111110101"),
    ("F/0", 10, "1111111010"),
    ("F/1", 15, "111111111000011"),
    ("F/2", 16, "1111111111110110"),
    ("F/3", 16, "1111111111110111"),
    ("F/4", 16, "1111111111111000"),
    ("F/5", 16, "1111111111111001"),
    ("F/6", 16, "1111111111111010"),
    ("F/7", 16, "1111111111111011"),
    ("F/8", 16, "1111111111111100"),
    ("F/9", 16, "1111111111111101"),
    ("F/A", 16, "1111111111111110")
]

huffman_table_AC_chrominance = {}
for item in data_chrominance:
    key = (int(item[0][0], 16), int(item[0][2], 16))
    value = item[2]
    huffman_table_AC_chrominance[key] = value
