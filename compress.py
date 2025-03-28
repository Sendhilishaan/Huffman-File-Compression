"""
Assignment 2 starter code
CSC148, Winter 2025
Instructors: Bogdan Simion, Rutwa Engineer, Marc De Benedetti, Romina Piunno

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for x in text:
        if x in freq_dict:
            freq_dict[x] += 1
        else:
            freq_dict[x] = 1
    return freq_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 1:
        all_keys = list(freq_dict.keys())
        symbol = all_keys[0]

        # here we are using a dummy symbol that is not equal to the actual
        # symbol by adding one and taking mod 256 since it must still be a
        # byte in the range.
        dummy_symbol = (symbol + 1) % 256
        return HuffmanTree(None, HuffmanTree(symbol), HuffmanTree(dummy_symbol))

    # build a list of items
    items = []
    count = 0
    for symbol, freq in freq_dict.items():
        items.append((freq, 0, count, HuffmanTree(symbol)))
        count += 1

    while len(items) > 1:
        first_item = _pop_min(items)
        second_item = _pop_min(items)
        freq1, tree1 = first_item[0], first_item[3]
        freq2, tree2 = second_item[0], second_item[3]

        new_freq = freq1 + freq2

        # since freq1 came first, it is less than freq2
        new_tree = HuffmanTree(None, tree1, tree2)

        items.append((new_freq, 1, count, new_tree))
        count += 1

    # Now there is only one item left, our completed huffman tree:
    return items[0][3]


def _pop_min(lst: list) -> tuple[int, int, int, HuffmanTree]:
    """
    Helper function to remove the minimum item from items
    Compare items by (freq, is_subtree, insertion_order

    Smaller frequency first, if there's a tie pick the one that is
    not merged subtree, if still a tie pick whichever appeared first.
    """
    min_index = 0

    for i in range(1, len(lst)):
        if lst[i][:3] < lst[min_index][:3]:
            min_index = i

    return lst.pop(min_index)


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    # Base case:
    if tree.is_leaf():
        return {tree.symbol: ''}

    # Recursive cases: get codes from children, then add 0 or 1
    codes = {}
    if tree.left is not None:
        left_codes = get_codes(tree.left)
        for symbol, code in left_codes.items():
            codes[symbol] = '0' + code

    if tree.right is not None:
        right_codes = get_codes(tree.right)
        for symbol, code in right_codes.items():
            codes[symbol] = '1' + code

    return codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    _number_postorder(tree, 0)


def _number_postorder(subtree: HuffmanTree, count: int) -> int:
    """
    Helper function to number_nodes just to use a counter and different return

    Assign numbers to each internal node in subtree and return the
    next available number after labelling subtree.
    """
    if subtree is None or subtree.is_leaf():
        return count

    # Labelling left first then right:
    count = _number_postorder(subtree.left, count)
    count = _number_postorder(subtree.right, count)

    # Labelling the internal node:
    subtree.number = count
    return count + 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    codes = get_codes(tree)

    # Calculating total frequencies and weighted sum
    total_freq = sum(freq_dict.values())
    total_bits = 0
    for symbol, frequencey in freq_dict.items():
        total_bits += frequencey * len(codes[symbol])

    # check division piazza
    return total_bits / total_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    At this stage of the assignment, I decided to read the python bytes()
    documentation and read from this website:

    https://www.programiz.com/python-programming/methods/built-in/bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    string = ''
    # Make one strig with all the codes
    for symbol in text:
        string += codes[symbol]

    compressed_bytes = []
    for i in range(0, len(string), 8):
        bite = string[i:i + 8]

        compressed_bytes.append(bits_to_byte(bite))

    return bytes(compressed_bytes)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    For more clarification on bytes and strings I read this website and watched
    the video that is embedded in the website:

    https://www.pythonmorsels.com/representing-binary-data-with-bytes/

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    # base case
    if tree is None or tree.is_leaf():
        return b''

    # recursively get postorder bytes from left / right trees
    left_bytes = tree_to_bytes(tree.left)
    right_bytes = tree_to_bytes(tree.right)

    if tree.left is not None and not tree.left.is_leaf():
        left_bin = 1
        left_val = tree.left.number
    else:
        left_bin = 0
        if tree.left is not None:
            left_val = tree.left.symbol
        else:
            left_val = 0

    if tree.right is not None and not tree.right.is_leaf():
        right_bin = 1
        right_val = tree.right.number
    else:
        right_bin = 0
        if tree.right is not None:
            right_val = tree.right.symbol
        else:
            right_val = 0

    node_bite = bytes([left_bin, left_val, right_bin, right_val])

    return left_bytes + right_bytes + node_bite


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    node = node_lst[root_index]

    # Base case
    if node.l_type == 0:
        left_sub = HuffmanTree(node.l_data)

    # Recursive case
    else:
        left_sub = generate_tree_general(node_lst, node.l_data)

    # Doing the same for the right subtree
    if node.r_type == 0:
        right_sub = HuffmanTree(node.r_data)

    else:
        right_sub = generate_tree_general(node_lst, node.r_data)

    return HuffmanTree(None, left_sub, right_sub)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    # Using a list like a stack:
    stack = []
    for i in range(root_index + 1):
        curr_rn = node_lst[i]

        # If leaf
        if curr_rn.r_type == 0:
            right_sub = HuffmanTree(curr_rn.r_data)
        else:
            right_sub = stack.pop()

        # same for the left
        if curr_rn.l_type == 0:
            left_sub = HuffmanTree(curr_rn.l_data)
        else:
            left_sub = stack.pop()

        # Create new internal node
        new_tree = HuffmanTree(None, left_sub, right_sub)
        stack.append(new_tree)

        # Last tree is our root
    return stack.pop()


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    code_dict = get_codes(tree)
    bits_dict = {}
    # Reverse the dictionary to get codes to map symbols
    for sym, bits in code_dict.items():
        bits_dict[bits] = sym

    all_bits = _bits_from_bytes(text)

    # Decompressed symbols
    decompressed_sym = []
    # Acummulator for bits
    curr_bits = ''

    for bit in all_bits:
        curr_bits += bit

        # Check if valid code has been formed
        if curr_bits in bits_dict:
            decompressed_sym.append(bits_dict[curr_bits])
            curr_bits = ''  # Reset the string
            if len(decompressed_sym) == size:
                break

    # If loop ends before hitting size, return whatever we have, incomplete data
    return bytes(decompressed_sym)


def _bits_from_bytes(text: bytes) -> str:
    """
    Helper function to decompress_bytes()
    Return a string of bits for the entire sequence of bytes in text
    """
    bit_str = ''
    for bit in text:
        for i in range(7, -1, -1):
            bit_str += str(get_bit(bit, i))

    return bit_str


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    leaves = _get_leaves(tree, 0)

    num_leaves = len(leaves)
    for i in range(num_leaves):
        min_index = i
        for j in range(i + 1, num_leaves):
            if leaves[j][0] < leaves[min_index][0]:
                min_index = j

        # Swap
        if min_index != i:
            leaves[i], leaves[min_index] = leaves[min_index], leaves[i]

    sym = []
    for i in range(num_leaves):
        sym.append(leaves[i][1].symbol)

    num_sym = len(sym)
    for i in range(num_sym):
        max_index = i
        for j in range(i + 1, num_sym):
            if freq_dict.get(sym[j], 0) > freq_dict.get(sym[max_index], 0):
                max_index = j

        if max_index != i:
            sym[i], sym[max_index] = sym[max_index], sym[i]

    # Reassign symbols to leaves so that highest freq -> shallowest leaf
    for i in range(num_sym):
        leaves[i][1].symbol = sym[i]


def _get_leaves(node: HuffmanTree, depth: int) -> list:
    """
    Helper function to get all the leaves in the input tree
    """

    if node is None:
        return []
    elif node.is_leaf():
        return [(depth, node)]

    else:
        left_leave = _get_leaves(node.left, depth + 1)
        right_leave = _get_leaves(node.right, depth + 1)

    return left_leave + right_leave


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
