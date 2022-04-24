# PointGray Firefly mit 15FPS
camera = dataIO("CMU1394", 0, 5, 3)
# PointGray Firefly mit 30FPS
# camera = dataIO("CMU1394", 0, 5, 4)
# PointGray Firefly mit 60FPS
# camera = dataIO("CMU1394", 0, 5, 5)

# Sony SX 900 mit 3.75 FPS
# camera = dataIO("CMU1394", 2, 2, 1)
# Sony SX 900 mit 7.5 FPS
# camera = dataIO("CMU1394", 2, 2, 2)

# Sony XCD-X700 mit 7.5 FPS
# camera = dataIO("CMU1394", 1, 5, 2)
# Sony XCD-X700 mit 15 FPS
# camera = dataIO("CMU1394", 1, 5, 3)

# for more info about starting CMU1394, type
# pluginHelp("CMU1394")
# and read the documentation of CMU1394

# print possible parameter list
print("CMU-FirewireGrabber has the following parameters:")
print(camera.getParamList())

# read parameters
sizex = camera.getParam("sizex")
print("FirewireGrabber width: " + str(sizex))

# start camera
camera.startDevice()

# acquire image
camera.acquire()

# create empty data object container
data = dataObject()

# obtain image
camera.getVal(data)

# plot the obtained image
plot(data)

# stop the camera
camera.stopDevice()

# start a live image
liveImage(camera)

# delete and close the camera
del camera
