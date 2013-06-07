#start camera (here: DummyGrabber)
camera = dataIO("DummyGrabber")

#set ROI
# x: [100,500] -> width: 401 (borders are included!)
# y: [50, 350] -> height: 301
camera.setParam("x0", 100)
camera.setParam("x1", 500)
camera.setParam("y0", 50)
camera.setParam("y1", 350)
print("width:", camera.getParam("sizex"))
print("height:", camera.getParam("sizey"))

#set bitdepth
camera.setParam("bpp",8)

#print available parameters of that device
print("DummyGrabber has the following parameters:")
print(camera.getParamList())

#print detailed information about parameters:
print(camera.getParamListInfo())

#read parameters
sizex = camera.getParam("sizex")
sizey = camera.getParam("sizey")

#start camera
camera.startDevice()

#acquire single image
camera.acquire()

#create empty data object for getting the image
data = dataObject()

#get a reference to the acquired image
#the reference is then available by the recently created dataObject
camera.getVal(data)

#Warning: data is only a shallow copy of the plugin internal memory.
#Therefore, the content of data will change when the next image is acquired.
#In order to create a deep copy of data, type:
dataCopy = data.copy()

#You can also combine both commands by typing:
#dataCopy = camera.copyVal(data)

#plot the acquired image
plot(data)

#close camera
camera.stopDevice()

#start a live image
liveImage(camera)

# Acquire an image stack of 10 measurements
camera.startDevice()
image = dataObject()
imageStack = dataObject([10,sizey,sizex],'uint8')

#stop the auto grabbing of the live image
camera.disableAutoGrabbing()

for i in range(0, 10, 1):
    camera.acquire()
    camera.getVal(image)
    print(i)
    imageStack[i, :, :] = image
    
camera.stopDevice()
# acquire stack finished

#plot stack (use arrows in widget to switch between planes)
plot(imageStack)

#enable the auto grabbing of the live image
camera.enableAutoGrabbing()

#finally delete the camera handle in order to close the device
del camera