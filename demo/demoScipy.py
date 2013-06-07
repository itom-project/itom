#this demo requires the python packages scipy and matplotlib

'''
opens the lena image from scipy.misc and shifts the image.
Finally the shift offsets are determined using cross-correlation
'''

import matplotlib
matplotlib.use('module://mpl_itom.backend_itomagg',False)
import scipy.misc
import numpy
from numpy.fft import fft2
from numpy.fft import fftshift
import pylab
from matplotlib import cm, colors

lena = scipy.misc.lena()
print("The lena-image has a size of",lena.shape)
print("The maximum value of this image is ", lena.max())
print("Its data type is ", lena.dtype)

F = fft2(lena)
F2 = fftshift(F)



pylab.figure()
pylab.gray()
pylab.subplot(221)
pylab.imshow(lena)
pylab.title("Lena (Original)")

pylab.subplot(222)
img = pylab.imshow(numpy.real(F))
img.set_clim(0,100)
pylab.title("Lena (FFT)")

pylab.subplot(223)
img = pylab.imshow(numpy.real(F2))
img.set_clim(0,100)
pylab.title("Lena (FFT), fftshift")


#cross-correlation
pylab.figure()
pylab.subplot(231)
pylab.imshow(lena)

lena_roll = numpy.roll(lena, 50, 1)
lena_roll = numpy.roll(lena_roll, -150, 0)

pylab.subplot(232)
pylab.imshow(lena_roll)

F = fftshift(fft2(lena))
F2 = fftshift(fft2(lena_roll))

F3 = numpy.multiply(F,F2.conj())

F4 = fftshift(numpy.fft.ifft2(F3))

pylab.subplot(233)
img = pylab.imshow(numpy.real(F))
img.set_clim(0,100)

pylab.subplot(234)
img = pylab.imshow(numpy.real(F2))
img.set_clim(0,100)

pylab.subplot(235)
img = pylab.imshow(numpy.real(F3))
img.set_clim(0,100)

pylab.subplot(236)
F5 = numpy.real(F4)
img = pylab.imshow(F5, vmin=0, vmax=0.001)

max_pos = numpy.argmax(F5)

offset_x = max_pos % 512
offset_y = (max_pos - offset_x) / 512

print("offset_x: ", offset_x - 256)
print("offset_y: ", offset_y - 256)

pylab.show()



