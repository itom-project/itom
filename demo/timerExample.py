"""
This script creates an instance of a dummy grabber
and implements a timer with an interval of 1000ms.

Everytime, the interval is expired the method imageAcquisition
is called. It acquires a new image an appends it to the global
list myImages

To stop the timer, call the method cancel. Additionally, the timer
is automatically interrupted after 10 iterations.
"""
from itom import *


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


def cancel():
    """call this method (e.g. by your gui) to stop the timer"""
    global t
    global cam
    t.stop()
    cam.stopDevice()


def demo_timedAcquisition():
    global t
    global cam
    global myImages
    global iters

    cam = dataIO("DummyGrabber")
    myImages = []
    iters = 0
    cam.startDevice()
    t = timer(1000, imageAcquisition)


if __name__ == "__main__":
    demo_timedAcquisition()
