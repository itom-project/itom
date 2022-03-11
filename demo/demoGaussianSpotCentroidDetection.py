# coding=iso-8859-15

try:
    from itom import algorithms
    hasItomAlgo = True
except ImportError:
    from itom import filter
    hasItomAlgo = False
    print("The itom.algorithms module cannot be used because"
          "of an older itom version. Use the itom.filter functions.")

from itom import dataIO
from itom import dataObject
from itom import plot2
from itom import timer

"""This demo shows how the itom.algorithms can be used.
In this example a camera *DummyGrabber* is used to acquire a gaussian spot.
For the centroid detection the itom.algorithms.centroidxy function is used. 
Finally, this examples shows how such centroid detection can be applied continuously.

For older itom versions the itom.filter function is used for the centroidXY algorithm.
By using the itom.algorithms you will get the algorithms information by a pop-up."""


class GaussianSpotCentroidDetection:
    def __init__(self) -> None:
        """Constructor of the live gaussianSpot centroid detection.

        Returns:
            None:
        """
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

    def __updateTimerCallBack(self) -> None:
        """Callback function to update the live image.

        Returns:
            None:
        """
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

        print("Centroid (x, y) position: {:.3f} {}, {:.3f} {}".format(centroidX, self.dObj.axisUnits[1], centroidY, self.dObj.axisUnits[0]))
        
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
        self.__plotHandle.call("deleteMarkers")
        self.__plotHandle.call("plotMarkers", centroidMarker, "r+25;2", "centroid", 0)

        return

    def __defineDObj(self, dObj: dataObject) -> dataObject:
        """Defines the meta information of the image

        Args:
            image (dataObject): 2D image

        Returns:
            dataObject: 2D image with define meta information
        """
        # define dataobject
        dObj.setAxisDescription(0, "y axis")
        dObj.setAxisDescription(1, "x axis")
        dObj.setAxisUnit(0, "µm")
        dObj.setAxisUnit(1, "µm")
        dObj.setAxisScale(0, 10)  # pixel pitch of 10 µm
        dObj.setAxisScale(1, 10)  # pixel pitch of 10 µm
        dObj.valueDescription = "intensity"
        dObj.valueUnit = "counts"

        return dObj


if __name__ == "__main__":
    gaussianSpotCentroid = GaussianSpotCentroidDetection()
