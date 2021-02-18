import calc_correlation

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
    [dx, dy] = calc_correlation.evaluate(image1, image2)
    gui.lbl_dx["text"] = "dx: " + str(dx)
    gui.lbl_dy["text"] = "dy: " + str(dy)


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
cam = dataIO("MSMediaFoundation", colorMode="gray")
# cam = dataIO("FileGrabber","*.tif","samples",8,2)

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

gui.show()
