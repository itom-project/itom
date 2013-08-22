import numpy as np
from numpy import fft

def evaluate(image1, image2):
    '''determines the offset between image1 and image2
    using cross-correlation and returns a tuple containing
    the shift in x and y-direction'''
    
    npImg1 = np.array(image1)
    npImg2 = np.array(image2)
    
    npImg1FFT = fft.fft2(npImg1)
    npImg2FFT = fft.fft2(npImg2)
    ccr = fft.ifft2( npImg1FFT * npImg2FFT.conj() )
    ccr_abs = np.ascontiguousarray(np.abs(ccr))
    
    [m,n] = ccr_abs.shape
    max_pos = np.argmax(ccr_abs)
    offset_x = max_pos % n
    offset_y = (max_pos - offset_x) / n
    
    if (offset_x > n/2):
        offset_x = offset_x - n
    if (offset_y > m/2):
        offset_y = offset_y - m
    
    return (offset_x, offset_y)