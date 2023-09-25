"""
compress.py

Main file for file compression/decompression
Compressed files will have an extension .huf
Decompressing the file add an extension .orig (on top of .huf)
Simply removing .huf.orig will yield the original file

Author: Tymofiy Sompura
"""
from __future__ import annotations

import time
from typing import Optional

from huffman import HuffmanTree
from utils import *


class Queue:
    """
    A Queue class for the build_huffman_tree implementation
    Each item will store a tuple of a HuffmanTree and its frequency

    Representation Invariant
        - self._first is None iff Queue is empty
    """
    _first: Optional[Node]
    _last: Optional[Node]

    class Node:
        """
        A Node class for the Queue implementation
        Each node stores a tuple of a HuffmanTree and integer
        """
        value: tuple[HuffmanTree, int]
        prev: Optional[Queue.Node]

        def __init__(self, v: tuple[HuffmanTree, int]) -> None:
            """
            Initialize the node with the given value
            """
            self.value = v
            self.prev = None

    def __init__(self) -> None:
        """
        Initialize the Queue
        """
        self._first = None
        self._last = None

    def is_empty(self) -> bool:
        """
        Return True iff the queue is empty
        """
        return self._first is None

    def is_size_one(self) -> bool:
        """
        Return True iff the queue has only one item
        """
        return self._first is self._last and self._first is not None

    def enqueue(self, value: tuple[HuffmanTree, int]) -> None:
        """
        Add <value> to the queue
        """
        node = self.Node(value)

        if self.is_empty():
            self._first = node
            self._last = node
            return

        self._last.prev = node
        self._last = node

    def dequeue(self) -> tuple[HuffmanTree, int]:
        """
        Remove and return a value from the queue
        """
        if self.is_empty():
            raise Exception("Cannot dequeue from emtpy queue")

        return_node = self._first
        self._first = return_node.prev

        if self._first is None:
            self._last = None

        return return_node.value

    def peek_freq(self) -> int:
        """
        Get the frequency of the first value in the queue.
        Frequency is stored at index 1 of the tuple.

        Precondition: Queue is not empty
        """
        return self._first.value[1]

    def __repr__(self) -> str:
        str_ = ""
        curr = self._first

        while curr is not None:
            str_ += str(curr.value) + " <- "
            curr = curr.prev

        return str_[:-4]


# ====================
# Functions for compression
def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dict_ = {}

    for c in text:
        if c not in dict_:
            dict_[c] = 0
        dict_[c] += 1

    return dict_


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
        s = list(freq_dict.keys())[0]
        return HuffmanTree(left=HuffmanTree(s),
                           right=HuffmanTree((s + 1) % 256))

    leaves = Queue()

    for item in sorted(list(freq_dict.items()), key=lambda pair: pair[1]):
        leaves.enqueue((HuffmanTree(symbol=item[0]), item[1]))

    merged = Queue()

    while not leaves.is_empty() or not merged.is_size_one():

        # Check whether to pick a symbol or a non-symbol for left
        comp = merged.is_empty()
        comp = comp or merged.peek_freq() >= (merged.peek_freq() + 1
                                              if leaves.is_empty()
                                              else leaves.peek_freq())

        if comp:
            left, left_freq = leaves.dequeue()
        else:
            left, left_freq = merged.dequeue()

        # Check whether to pick a symbol or a non-symbol for right
        comp = merged.is_empty()
        comp = comp or merged.peek_freq() >= (merged.peek_freq() + 1
                                              if leaves.is_empty()
                                              else leaves.peek_freq())

        if comp:
            right, right_freq = leaves.dequeue()
        else:
            right, right_freq = merged.dequeue()

        # The merged tree must have higher or equal frequency than previously
        # merged trees, because we always consider minimums
        merged.enqueue((HuffmanTree(left=left, right=right),
                        left_freq + right_freq))

    return merged.dequeue()[0]


def _get_codes_helper(tree: Optional[HuffmanTree],
                      prefix: str) -> dict[int, str]:
    if tree is None:
        return {}
    if tree.symbol is not None:
        return {tree.symbol: prefix}

    dict_ = {}

    dict_.update(_get_codes_helper(tree.left, prefix + '0'))
    dict_.update(_get_codes_helper(tree.right, prefix + '1'))

    return dict_


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    return _get_codes_helper(tree, "")


def _number_nodes_helper(tree: Optional[HuffmanTree], count: list[int]) -> None:
    if tree.symbol is not None:
        return

    _number_nodes_helper(tree.left, count)
    _number_nodes_helper(tree.right, count)

    tree.number = count[0]
    count[0] += 1


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
    _number_nodes_helper(tree, [0])


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
    weighted_sum = sum([freq_dict[key] * len(code)
                        for key, code in get_codes(tree).items()])
    total_freq = sum([freq_dict[key] for key in freq_dict])

    return weighted_sum / total_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

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
    compressed = bytes()

    str_comp = "".join([codes[byte] for byte in text])

    str_comp += (((len(str_comp) + 7) // 8 - len(str_comp) // 8) * 8) * '0'

    compressed += bytes([int(str_comp[8 * i: 8 * (i + 1)], 2)
                         for i in range(len(str_comp) // 8)])

    return compressed


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

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
    if tree.symbol is not None:
        return bytes()

    compressed_tree = bytes()

    compressed_tree += tree_to_bytes(tree.left)
    compressed_tree += tree_to_bytes(tree.right)

    bytes_for_root = [0, 0, 0, 0]

    if tree.left.symbol is None:
        bytes_for_root[0] = 1
        bytes_for_root[1] = tree.left.number
    else:
        bytes_for_root[1] = tree.left.symbol

    if tree.right.symbol is None:
        bytes_for_root[2] = 1
        bytes_for_root[3] = tree.right.number
    else:
        bytes_for_root[3] = tree.right.symbol

    compressed_tree += bytes(bytes_for_root)

    return compressed_tree


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
def _generate_tree_general_helper(node_lst: list[ReadNode],
                                  root_index: list[int]) -> HuffmanTree:
    tree = HuffmanTree()

    root_node = node_lst[root_index[0]]

    if root_node.l_type == 0:
        tree.left = HuffmanTree(root_node.l_data)
    else:
        root_index[0] = root_node.l_data
        tree.left = _generate_tree_general_helper(node_lst, root_index)

    if root_node.r_type == 0:
        tree.right = HuffmanTree(root_node.r_data)
    else:
        root_index[0] = root_node.r_data
        tree.right = _generate_tree_general_helper(node_lst, root_index)

    return tree


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
    return _generate_tree_general_helper(node_lst, [root_index])


def _generate_tree_postorder_helper(node_lst: list[ReadNode],
                                    root_index: list[int]) -> HuffmanTree:
    tree = HuffmanTree()

    root_node = node_lst[root_index[0]]

    if root_node.r_type == 0:
        tree.right = HuffmanTree(root_node.r_data)
    else:
        root_index[0] -= 1
        tree.right = _generate_tree_postorder_helper(node_lst, root_index)

    if root_node.l_type == 0:
        tree.left = HuffmanTree(root_node.l_data)
    else:
        root_index[0] -= 1
        tree.left = _generate_tree_postorder_helper(node_lst, root_index)

    return tree


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
    return _generate_tree_postorder_helper(node_lst, [root_index])


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.
    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    decompressed_str = ""
    current_tree_node = tree

    for byte in text:
        current_bits = f"{byte:08b}"

        for bit in current_bits:
            if bit == '0':
                current_tree_node = current_tree_node.left
            else:
                current_tree_node = current_tree_node.right

            if current_tree_node.symbol is not None:
                decompressed_str += bytes([current_tree_node.symbol]).hex()
                current_tree_node = tree

    decompressed_str = decompressed_str[:2 * size]
    return bytes.fromhex(decompressed_str)


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


if __name__ == "__main__":

    import doctest

    doctest.testmod()

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
