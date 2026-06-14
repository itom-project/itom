"""Timer
===========

This script creates an instance of a ``DummyGrabber``
and implements a timer with an interval of ``1000ms``.

Everytime, the interval is expired the method ``imageAcquisition``
is called. It acquires a new image an appends it to the global
list ``myImages``.

To stop the timer, call the method ``cancel``. Additionally, the timer
is automatically interrupted after ``10`` iterations.
"""
from itom import timer
from itom import dataIO
from itom import dataObject
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoTimer.png'

###############################################################################
# Function for the image acquisition.


def imageAcquisition():
    global iters
    global t
    global cam
    global myImages

    print("acquire next image")
    cam.acquire()
    d = dataObject()
    cam.getVal(d)
    globals()["myImages"].append(d.copy())
    iters += 1
    if iters >= 10:
        t.stop()

###############################################################################
# Call this method (e.g. by your gui) to stop the timer.


def cancel():
    global t
    global cam
    t.stop()
    cam.stopDevice()

###############################################################################
# Define a ``DummyGrabber`` camera and timer for a timed acquisition.


cam = dataIO("DummyGrabber")
myImages = []
iters = 0
cam.startDevice()
t = timer(1000, imageAcquisition)
