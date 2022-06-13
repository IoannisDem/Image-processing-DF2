""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.jpeg import jpegdec
from .common import my_function, HeaderType, jpeg_quant_size
from .cued_sf2_lab.jpeg import jpegdeclbt

def decode(vlc: np.ndarray, header: HeaderType) -> np.ndarray:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen decoding scheme
    hufftab, step_opt = header
    return jpegdeclbt(vlc, step_opt, hufftab=hufftab, log=False)
