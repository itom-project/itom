"""Gaussian DummyGrabber
=====================

This demo shows with the example of the ``DummyGrabber``
how grabber and cameras are used in ``itom``.
The ``DummyGrabber`` can optionally show a Gaussian spot or
an array of Gaussian spots. In addition, it is shown here how
fruther meta information (*axis description, axis unit, axis scale, axis offset,
value description, value unit*) of the frames are set, which are adopted
for the ``liveImage`` and the returned ``dataObject`` frames.."""

from itom import dataIO
from itom import liveImage

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoGaussianDummyGrabber.png'

###############################################################################
# Start camera (e.g.: ``DummyGrabber``) with moving ``Gaussian spot``.
cameraGaussian = dataIO("DummyGrabber", imageType="gaussianSpot")

###############################################################################
# Start camera (e.g.: ``DummyGrabber``) with moving ``4 Gaussian spots``.
cameraGaussianArray = dataIO("DummyGrabber", imageType="gaussianSpotArray")


###############################################################################
# Set meta information for the gaussianSpot of image for liveImage and return image.
# The axis properties are set assuming a pixel pitch of 10µm.
# The offset is set such that origin is in the center.
cameraGaussian.setParam("axisScale", (10e-6, 10e-6))  # scale 10µm
cameraGaussian.setParam(
    "axisOffset", ((480 - 1) / 2, (640 - 1) / 2)
)  # origin on center
cameraGaussian.setParam("axisDescription", ("y axis", "x axis"))
cameraGaussian.setParam("axisUnit", ("µm", "µm"))
cameraGaussian.setParam("valueDescription", "counts")
cameraGaussian.setParam("valueUnit", "a.u.")

###############################################################################
# Start a live image with aspect ratio of 1 and visible colorBar.
#
# .. image:: ../../_static/demoGaussianDummyGrabber_1.png
#    :width: 100%
liveImage(
    cameraGaussian,
    properties={
        "colorMap": "falseColorIR",
        "keepAspectRatio": True,
        "colorBarVisible": True,
    },
)

###############################################################################
# .. image:: ../../_static/demoGaussianDummyGrabber_2.png
#    :width: 100%
liveImage(cameraGaussianArray)
