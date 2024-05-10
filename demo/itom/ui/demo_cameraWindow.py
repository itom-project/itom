"""Camera widget
================

"""

from itom import dataIO
from itom import ui
from itom import dataObject
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoDummyGrabber.png'

cam = dataIO("DummyGrabber")

win = ui("cameraWindow.ui", ui.TYPEWINDOW, childOfMainWindow=True)


def integrationTime_changed():
    if win.radioInt1["checked"]:
        cam.setParam("integration_time", 0.005)
    elif win.radioInt2["checked"]:
        cam.setParam("integration_time", 0.010)
    else:
        cam.setParam("integration_time", 0.060)


def autoGrabbing_changed(checked):
    if checked:
        cam.enableAutoGrabbing()
    else:
        cam.disableAutoGrabbing()


def snap():
    d = dataObject()
    cam.startDevice()
    autoGrabbingStatus = cam.getAutoGrabbing()
    cam.disableAutoGrabbing()
    cam.acquire()
    cam.getVal(d)
    win.plot["source"] = d
    if autoGrabbingStatus:
        cam.enableAutoGrabbing()
    cam.stopDevice()


def live():
    win.plot["camera"] = cam


# initialize all signal/slots
win.radioInt1.connect("clicked()", integrationTime_changed)
win.radioInt2.connect("clicked()", integrationTime_changed)
win.radioInt3.connect("clicked()", integrationTime_changed)

win.btnSnap.connect("clicked()", snap)
win.btnLive.connect("clicked()", live)

win.checkAutoGrabbing.connect("clicked(bool)", autoGrabbing_changed)

# initialize gui elements
win.checkAutoGrabbing["checked"] = cam.getAutoGrabbing()
win.radioInt1["checked"] = True

win.show(0)
