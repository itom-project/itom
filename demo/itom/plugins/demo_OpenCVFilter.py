"""OpenCV filter
================

This demo shows how OpenCV filters are applied to the ``dataObject``."""

from itom import dataObject
from itom import algorithms
from itom import plot
from itom import filterHelp
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoOpenCV.png'

# create a randomly filled  150x150px dataObject
dObj = dataObject.rand([150, 150], "float32")

###############################################################################
# Get information about the filter and its parameters.
# First, no idea about the filters name, therefore get all names containing the keyword ``cv``:
filterInfoDict = filterHelp("cv", dictionary=1)

for key, value in filterInfoDict.items():
    print("{}: {}".format(value["name"], value["description"]))

# the desired filter is called "cvMedianBlur", now obtain detailed information
medianBlurDict = filterHelp("cvMedianBlur", dictionary=1)

print("{}: {}".format(medianBlurDict["cvMedianBlur"]["name"], medianBlurDict["cvMedianBlur"]["description"]))

# create an empty output image (its content will be filled within the filter-call)
outputImage = dataObject()
###############################################################################
# Call the filter. The parameters after the filter name are the
# ``mandatory`` parameters followed by the ``optional`` ones (if desired):
#
# 1. parameter (mand): input image
#
# 2. parameter (mand): output image
#
# 3. kernellength (opt): size of squared filter kernel (default: 3)
algorithms.cvMedianBlur(dObj, outputImage, 5)

plot(dObj)  # plot original image
plot(outputImage)  # plot filtered image

###############################################################################
# .. image:: ../../_static/demoOpenCVFilter_1.png
#    :width: 100%