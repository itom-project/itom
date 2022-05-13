"""Cameras and images
===================

This demo shows with the example of the ``DummyGrabber``
how you acquire an image and apply some filters."""

from itom import dataIO
from itom import dataObject
from itom import algorithms
from itom import liveImage
from itom import saveIDC
from itom import loadIDC
from itom import plot
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoCameraAndImages.png'

###############################################################################
# Initialize a ``DummyGrabber`` camera

cam = dataIO("DummyGrabber")
cam.setParam("bpp", 8)
# start camera (only once)
cam.startDevice()

# show live image of camera
liveImage(cam)

###############################################################################
# .. hint:: The live images tries to acquire and get up to 50 images per second
#     from the camera. If you want to acquire images by yourself in a script, you need
#     to stop the timer of the live images for a certain amount of time. After you are
#     done with your manual acquisition, you can restart the timer again.

currentStatus = cam.getAutoGrabbing()
print("Current value of auto grabbing property of the camera:", currentStatus)

cam.setAutoGrabbing(False)

###############################################################################
# Acquire 10 images in a list of dataObjects

result = []
d = dataObject()  # empty data object where the image should be put in

for i in range(0, 10):
    cam.acquire()
    cam.getVal(d)  # d is a shallow copy of the camera image
    result.append(d.copy())

###############################################################################
# Save the list of images to the **image1.idc** file (idc is a file format for the python pickle module)
saveIDC("image1.idc", {"result": result, "description": "sample 1"})

###############################################################################
# Load the list of images
loaded_objects = loadIDC("image1.idc")
result2 = loaded_objects["result"]

# plot the 3rd image from the list
plot(result2[2])

###############################################################################
# Acquire 10 images in an image stack
num = 10
sizex = cam.getParam("sizex")
sizey = cam.getParam("sizey")
bpp = cam.getParam("bpp")

if bpp == 8:
    d = dataObject([num, sizey, sizex], "uint8")
else:
    d = dataObject([num, sizey, sizex], "uint16")

for idx in range(num):
    cam.acquire()
    cam.copyVal(d[idx, :, :])  # partial deep copy into one part of the 3d object d

plot(d)

###############################################################################
# Calculate mean value of image stack in z-direction.
result_mean = dataObject()

algorithms.calcMeanZ(d, result_mean, ignoreInf=0, calcStd=0)
# result_mean is a 3d Object with [1 x sizey x sizex] dimensions.
# We squeeze it to get a 2D Object
result_mean = result_mean.squeeze()

result_mean.setTag("title", "mean value of {} acquisitions".format(num))
result_mean.axisUnits = ("px", "px")
result_mean.axisDescriptions = ("y", "x")
plot(result_mean)
###############################################################################
# .. image:: ../../_static/demoCameraAndImages_1.png
#    :width: 100%

###############################################################################
# Apply Gaussian filter onto the mean image.
result_filter = dataObject()
kernelVal = 9
algorithms.gaussianFilter(result_mean, result_filter, kernelx=kernelVal, kernely=kernelVal)

###############################################################################
# Copy meta information from source ``dataObject``.
result_filter.copyMetaInfo(result_mean)
result_filter.setTag("title", "Gaussian filter with kernel {}".format(kernelVal))
plot(result_filter)
###############################################################################
# .. image:: ../../_static/demoCameraAndImages_2.png
#    :width: 100%

# reset the auto grabbing functionality of possibly connected live images
cam.setAutoGrabbing(currentStatus)

# end camera
cam.stopDevice()
