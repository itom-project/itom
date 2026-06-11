"""Cross correlation of images
============================

In this demo the cross-correlation is calculated between two images
acquired via the integrated webcam of your PC.
"""

from itom import dataIO
from itom import dataObject
from itom import ui
import numpy as np
from numpy import fft
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoCrossCorrelation.png'


# some methods
def acquireImage1():
    g = cam.getAutoGrabbing()
    cam.setAutoGrabbing(False)
    cam.acquire()
    cam.copyVal(image1)
    gui.plot1["source"] = image1
    cam.setAutoGrabbing(g)


def acquireImage2():
    g = cam.getAutoGrabbing()
    cam.setAutoGrabbing(False)
    cam.acquire()
    cam.copyVal(image2)
    gui.plot2["source"] = image2
    cam.setAutoGrabbing(g)


def evaluate():
    """determines the offset between image1 and image2
    using cross-correlation and returns a tuple containing
    the shift in x and y-direction"""

    npImg1 = np.array(image1)
    npImg2 = np.array(image2)

    npImg1FFT = fft.fft2(npImg1)
    npImg2FFT = fft.fft2(npImg2)
    ccr = fft.ifft2(npImg1FFT * npImg2FFT.conj())
    ccr_abs = np.abs(ccr)  # np.ascontiguousarray(np.abs(ccr))

    [m, n] = ccr_abs.shape
    max_pos = np.argmax(ccr_abs)
    offset_x = max_pos % n
    offset_y = (max_pos - offset_x) / n

    if offset_x > n / 2:
        offset_x = offset_x - n
    if offset_y > m / 2:
        offset_y = offset_y - m

    gui.lbl_dx["text"] = "dx: " + str(offset_x)
    gui.lbl_dy["text"] = "dy: " + str(offset_y)


def saveImages():
    filename = ui.getSaveFileName("Filename", filters="IDC (*.idc)", parent=gui)
    if filename:
        saveIDC(filename, {"image1": image1, "image2": image2})


def loadImages():
    global image1
    global image2
    filename = ui.getOpenFileName("Filename", filters="IDC (*idc)", parent=gui)
    if filename:
        d = loadIDC(filename)
        image1 = d["image1"]
        image2 = d["image2"]
        gui.plot1["source"] = image1
        gui.plot2["source"] = image2


# open camera (make it before you start this script)
try:
    cam = dataIO("OpenCVGrabber", colorMode="gray")
except:
    print("Can not open camera with OpenCVGrabber. Probabel root cause: WebCam is not available.")
    print("Used Itom Dummy Grabber instead.")
    cam = dataIO("DummyGrabber")

# start camera
cam.startDevice()

# create data objects
image1 = dataObject()
image2 = dataObject()

# create gui
gui = ui("dialog.ui", ui.TYPEWINDOW)
gui.btnAcquire1.connect("clicked()", acquireImage1)
gui.btnAcquire2.connect("clicked()", acquireImage2)
gui.btnLoad.connect("clicked()", loadImages)
gui.btnSave.connect("clicked()", saveImages)
gui.btnEval.connect("clicked()", evaluate)

# show live image in upper plot
if cam.name() != "FileGrabber":
    cam.setAutoGrabbing(True)
else:
    cam.setAutoGrabbing(False)

gui.plotLive["camera"] = cam
gui.plotLive["keepAspectRatio"] = True
gui.plotLive["yAxisFlipped"] = True

gui.show()
