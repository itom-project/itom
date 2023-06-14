"""MultiChannelDummyGrabber
================

This demo shows with the example of the ``MultiChannelDummyGrabber``
how grabber and cameras are used in ``itom``."""

from itom import dataIO
from itom import dataObject
from itom import plot
from itom import liveImage
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoDummyGrabber.png'

###############################################################################
# Start camera (e.g.: ``DummyMultiChannelGrabber``) with a ``noise image`` (default).
camera = dataIO("DummyMultiChannelGrabber")

###############################################################################
# Start camera (e.g.: ``DummyMultiChannelGrabber``) with moving ``Gaussian spot``.
cameraGaussian = dataIO("DummyMultiChannelGrabber", imageType="gaussianSpot")

###############################################################################
# Start camera (e.g.: ``DummyMultiChannelGrabber``) with moving ``4 Gaussian spots``.
cameraGaussianArray = dataIO("DummyMultiChannelGrabber", imageType="gaussianSpotArray")

print("width:", camera.getParam("sizex"))
print("height:", camera.getParam("sizey"))

###############################################################################
# Set pixel format to ``mono8``
camera.setParam("pixelFormat", "mono8")

# print available parameters of that device
print("DummyMultiChannelGrabber has the following parameters:")
print(camera.getParamList())

# print detailed information about parameters:
print(camera.getParamListInfo())

###############################################################################
# Read parameters from device.
roi = camera.getParam("roi")
###############################################################################
# Start camera.
camera.startDevice()

###############################################################################
# Acquire single image.
camera.acquire()

# Create empty dataObject for getting the image
data = dataObject()

# get a reference to the acquired image
# the reference is then available by the recently created dataObject
camera.getVal(data)

###############################################################################
# .. warning::
#
#     The method **getVal** returns only a shallow copy of the plugin internal memory.
#     Therefore, the content of data will change when the next image is acquired.
#     In order to create a deep copy of data, type:
#
#     .. code-block:: python
#
#         camera.copyVal(data)
#

# You can also convert the data afterwards to a deep copy by typing:
dataCopy = data.copy()

# plot the acquired image
plot(data)

###############################################################################
# Stop camera.
camera.stopDevice()

###############################################################################
# Start a live image.
liveImage(camera)

###############################################################################
#
# .. image:: ../../_static/demoMultiChannelDummyGrabber_1.png
#    :width: 100%

liveImage(cameraGaussian)
liveImage(cameraGaussianArray)

###############################################################################
# Acquire an image stack of 10 measurements.
num = 100
camera.startDevice()
image = dataObject()
imageStack = dataObject([num, roi[2], roi[3]], "uint8")

# stop the auto grabbing of the live image
camera.disableAutoGrabbing()

for idx in range(num):
    camera.acquire()
    camera.getVal(image)
    imageStack[idx, :, :] = image
    print(idx)

camera.stopDevice()
# acquire stack finished

# plot stack (use arrows in widget to switch between planes)
plot(imageStack)

# enable the auto grabbing of the live image
camera.enableAutoGrabbing()
