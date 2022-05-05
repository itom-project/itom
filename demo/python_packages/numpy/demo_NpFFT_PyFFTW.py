"""Numpy FFT, PyFFTW
====================

This example shows how to use the FFT or IFFT from ``Numpy``.
If possible the package PyFFTW is searched. If it is avaible,
the fast implementation of FFT and IFFT is used from this package (GPL license!!!).

In order to have the fast version of the fourier transform
in ``pyfftw``, align your input data using ``getAlignNdArray``. This is
an idle operation if PyFFTW is not available. Either ``np.fft.fft2``
or ``pyfftw.interfaces.numpy_fft.fft2`` are mapped to ``myfft2``, such the
overall call can be done by using ``myfft2(...)``. The same holds for ``myifft2(...)``."""
import numpy as np

myfft2 = np.fft.fft2  # default: fft2 from numpy
myifft2 = np.fft.ifft2  # default: ifft2 from numpy
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoNumpy.png'

###############################################################################
# Define function and overwrite it when ``pyfftw`` is available. 

def getAlignNdArray(image):
    return np.array(image)


try:
    import pyfftw

    myfft2 = pyfftw.interfaces.numpy_fft.fft2  # if PyFFTW: use fft2 from this package
    myifft2 = pyfftw.interfaces.numpy_fft.ifft2
    alignSize = pyfftw.simd_alignment

    def getAlignNdArray(image):
        return pyfftw.n_byte_align(np.array(image), alignSize)

except ModuleNotFoundError:
    print("pyfftw could not be found. Numpy fft is used instead")


image = np.random.randn(1024, 512)
I = getAlignNdArray(image)
I1 = myfft2(I)
I2 = myifft2(I)
I2