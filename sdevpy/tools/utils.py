""" Various utilities for software versions and so on """
import struct


def print_python_bit_version():
    """ Prints number of bits of the installed python version (32 or 64) """
    print(struct.calcsize("P") * 8)
