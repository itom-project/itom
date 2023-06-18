"""DummyGrabber
================

This demo shows with the example of the ``DummyGrabber``
how grabber and cameras are used in ``itom``.
3 ``DummyGrabber`` instances are created.
The ``camera`` shows how to set parameter like ``region of interest``.
Using the ``cameraGaussian`` it is show how to define meta information
of the image showing axis descriptions, units, ..."""

from itom import dataIO
from itom import dataObject
from itom import plot
from itom import liveImage
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoDummyGrabber.png'


###############################################################################
# Start camera (e.g.: ``DummyGrabber``) with a ``noise image`` (default).
camera = dataIO("DummyGrabber")

###############################################################################
# Start camera (e.g.: ``DummyGrabber``) with moving ``Gaussian spot``.
cameraGaussian = dataIO("DummyGrabber", imageType="gaussianSpot")

###############################################################################
# Start camera (e.g.: ``DummyGrabber``) with moving ``4 Gaussian spots``.
cameraGaussianArray = dataIO("DummyGrabber", imageType="gaussianSpotArray")

###############################################################################
# Set region of interest (ROI) by a tuple.
#
# x: [100,499] -> width: 400 (borders are included!)
#
# y: [40, 349] -> height: 310
camera.setParam("roi", [100, 40, 400, 300])

###############################################################################
# or by explicite roi values.
camera.setParam("roi[0]", 100)
camera.setParam("roi[2]", 400)

print("width:", camera.getParam("sizex"))
print("height:", camera.getParam("sizey"))

###############################################################################
# Set bits per pixel (bpp).
camera.setParam("bpp", 8)

# print available parameters of that device
print("DummyGrabber has the following parameters:")
print(camera.getParamList())

# print detailed information about parameters:
print(camera.getParamListInfo())

###############################################################################
# Read parameters from device.
sizex = camera.getParam("sizex")
sizey = camera.getParam("sizey")
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
# Set meta information for the gaussianSpot of image for liveImage and return image.
# The axis properties are set assuming a pixel pitch of 10µm.
# The offset is set such that origin is in the center.
cameraGaussian.setParam("axisScale", (10e-6, 10e-6))  # scale 10µm
cameraGaussian.setParam("axisOffset", ((480-1)/2, (640-1)/2))  # origin on center
cameraGaussian.setParam("axisDescription", ("y axis", "x axis"))
cameraGaussian.setParam("axisUnit", ("µm", "µm"))
cameraGaussian.setParam("valueDescription", "counts")
cameraGaussian.setParam("valueUnit", "a.u.")

###############################################################################
# Start a live image with aspect ratio of 1 and visible colorBar.
#
# .. image:: ../../_static/demoDummyGrabber_2.png
#    :width: 100%
liveImage(cameraGaussian, properties={"colorMap": "falseColorIR", "keepAspectRatio": True, "colorBarVisible": True})

###############################################################################
# .. image:: ../../_static/demoDummyGrabber_3.png
#    :width: 100%
liveImage(cameraGaussianArray)

###############################################################################
# Acquire an image stack of 10 measurements.
num = 100
camera.startDevice()
image = dataObject()
imageStack = dataObject([num, sizey, sizex], "uint8")

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
