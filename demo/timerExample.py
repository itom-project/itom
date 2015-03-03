'''
This script creates an instance of a dummy grabber
and implements a timer with an interval of 1000ms.

Everytime, the interval is expired the method imageAcquisition
is called. It acquires a new image an appends it to the global
list myImages

To stop the timer, call the method cancel. Additionally, the timer
is automatically interrupted after 10 iterations.
'''

cam = dataIO("DummyGrabber")
myImages = []
iters = 0

def imageAcquisition():
    global iters
    
    cam.acquire()
    d = dataObject()
    cam.getVal(d)
    globals()["myImages"].append(d.copy())
    iters += 1
    if iters >= 10:
        t.stop()
    
cam.startDevice()

t = timer(1000, imageAcquisition)

def cancel():
    '''call this method (e.g. by your gui) to stop the timer'''
    t.stop()
    cam.stopDevice()