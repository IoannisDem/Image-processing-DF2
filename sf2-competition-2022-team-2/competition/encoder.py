""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple
from .cued_sf2_lab.jpeg import jpegenclbt
from scipy import optimize
from .common import HeaderType, jpeg_quant_size

def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header.
    
    If you have no header, return `0`. """
    hufftab, step_opt = header
    hufftab_bits = (len(hufftab.bits) + len(hufftab.huffval)) * 8
    step_opt_bits = 10  # 6 bits for the 64-float integer + 4 bits to represent the decimal part (up to 0.0625)

    return hufftab_bits + step_opt_bits


def encode(X: np.ndarray) -> Tuple[np.ndarray, HeaderType]:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """

    def lbt_opt_step(X):

        # Upper bound on size
        ref_size = 40960 - 20

        def error_difference(step):
            # Quantise layer
            print(f"Step: {step}")
            vlc, hufftab = jpegenclbt(X, step, opthuff=True, log=False)
            header = (hufftab, step)
            size_header = header_bits(header)

            print(f"size_header: {size_header}")
            size_vlc = sum(vlc[:, 1])

            size_total = size_vlc + size_header
            size_error = np.abs(ref_size - size_total)
            print(size_total)

            return size_error

        res = optimize.minimize_scalar(error_difference, bounds = (1, 100), method = "bounded")

        step_opt = res.x
        print(f"Optimal Step: {step_opt}")

        return step_opt

    step_opt = lbt_opt_step(X)
    vlc, hufftab = jpegenclbt(X, step_opt, opthuff=True, log=False)

    return vlc, (hufftab, step_opt)
