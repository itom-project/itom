#open camera
cam = dataIO("OpenCVGrabber")

#start camera
cam.startDevice()

#acquire first image
image1 = dataObject()
cam.acquire()
cam.copyVal(image1)

ui.msgInformation("move camera", "move camera to next position and confirm")

#acquire second image
image2 = dataObject()
cam.acquire()
cam.copyVal(image2)

#close camera
cam.stopDevice()
del cam