# coding=utf8

"""Gaussian spot centroid detection
=================================

This demo shows how the ``itom.algorithms`` can be used.
In this example a camera ``DummyGrabber`` is used to acquire a moving gaussian spot.
For the centroid detection the ``itom.algorithms.centroidxy`` function is used. 
Finally, this examples shows how a centroid position marker is added to the gaussian spot image.

.. note::

    For older ``itom`` versions the ``itom.filter`` function is used for the centroidXY algorithm.
    By using the itom.algorithms you will get the algorithms information by a pop-up during typing."""

from itom import dataIO
from itom import dataObject
from itom import plot2
from itom import timer

try:
    from itom import algorithms

    hasItomAlgo = True
except ImportError:
    from itom import filter

    hasItomAlgo = False
    print(
        "The itom.algorithms module cannot be used because" "of an older itom version. Use the itom.filter functions."
    )
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoGaussianSpotCentroidDetection.png'

###############################################################################
# Constructor of the live gaussianSpot centroid detection.


class GaussianSpotCentroidDetection:
    def __init__(self):
        """Constructor function of the class."""
        # first initialize the DummyGrabber with a gaussianSpot as private member
        self.__cam = dataIO("DummyGrabber", imageType="gaussianSpot")
        self.__cam.startDevice()

        # define a dataobject for the gaussienSpot image as a public member
        self.dObj = dataObject()
        self.__cam.acquire()
        self.__cam.getVal(self.dObj)
        self.dObj = self.__defineDObj(self.dObj)

        # plot index and handle as private member
        self.__plotHandle = None
        self.__plotIndex = None

        # define timer to update live image
        self.__liveTimer = timer(50, self.__updateTimerCallBack, name="liveTimer")

        return

    ###############################################################################
    # Callback function to update the live image.

    def __updateTimerCallBack(self):
        """Callback function to update the live image."""
        # stop live time if plot window is closed
        if self.__plotHandle:
            if not self.__plotHandle.exists():
                self.__liveTimer.stop()
                return

        # acquire and get image
        self.__cam.acquire()
        self.__cam.getVal(self.dObj)

        # calculate the centroid of the gaussianSpot
        if hasItomAlgo:  # new itom.algorithms with auto completion and docstring in pop-up
            [intensityX, intensityY, centroidX, centroidY] = algorithms.centroidXY(self.dObj)
        else:  # old itom.filter
            [intensityX, intensityY, centroidX, centroidY] = filter("centroidXY", self.dObj)

        print(
            "Centroid (x, y) position: {:.3f} {}, {:.3f} {}".format(
                centroidX, self.dObj.axisUnits[1], centroidY, self.dObj.axisUnits[0]
            )
        )

        # create the centroid xy marker
        centroidMarker = dataObject([2, 1], "float32")
        centroidMarker[0, :] = centroidY
        centroidMarker[1, :] = centroidX

        # plot the camera image with marked centroid position
        if not self.__plotHandle:  # open figure window by first time
            [self.__plotIndex, self.__plotHandle] = plot2(
                self.dObj, properties={"keepAspectRatio": True, "colorBarVisible": True, "colorMap": "viridis"}
            )

        else:  # replot the figure
            self.__plotHandle.call("replot")

        # delete markers and plot the new one
        if self.__plotHandle:
            self.__plotHandle.call("deleteMarkers")
            self.__plotHandle.call("plotMarkers", centroidMarker, "r+25;2", "centroid", 0)

        return

    ###############################################################################
    # Function to defines the dataObject by meta information.

    def __defineDObj(self, dObj: dataObject) -> dataObject:
        """Private function to define the meta information of the image dataObject.

        Args:
            image (dataObject): 2D image

        Returns:
            dataObject: 2D image with define meta information
        """
        # define dataobject
        dObj.setAxisDescription(0, "y axis")
        dObj.setAxisDescription(1, "x axis")
        dObj.setAxisUnit(0, "\xb5m")
        dObj.setAxisUnit(1, "\xb5m")
        dObj.setAxisScale(0, 10e-3)  # pixel pitch of 10 \xb5m
        dObj.setAxisScale(1, 10e-3)  # pixel pitch of 10 \xb5m
        dObj.valueDescription = "intensity"
        dObj.valueUnit = "counts"

        return dObj


if __name__ == "__main__":
    gaussianSpotCentroid = GaussianSpotCentroidDetection()

###############################################################################
# .. image:: ../../_static/demoGaussianSpotCentroidDetection_1.png
#    :width: 100%
