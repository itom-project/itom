from itom import *


def demo_CameraAndImages_Samples(bpp=8):

    # initialize a dummy grabber camera
    cam = dataIO("DummyGrabber")
    cam.setParam("bpp", bpp)
    # start camera (only once)
    cam.startDevice()

    # show live image of camera
    liveImage(cam)

    # be careful: the live images tries to acquire and get up to 50 images per second
    # from the camera. If you want to acquire images by yourself in a script, you need
    # to stop the timer of the live images for a certain amount of time. After you are
    # done with your manual acquisition, you can restart the timer again.
    currentStatus = cam.getAutoGrabbing()
    print(
        "Current value of auto grabbing property of the camera:", currentStatus
    )

    cam.setAutoGrabbing(False)

    """
    Sample 1: acquire 10 images in a list of data objects
    """
    result = []
    d = dataObject()  # empty data object where the image should be put in

    for i in range(0, 10):
        cam.acquire()
        cam.getVal(d)  # d is a shallow copy of the camera image
        result.append(d.copy())

    # save the list of images to the image1.idc file (idc is a file format for the python pickle module)
    saveIDC("image1.idc", {"result": result, "description": "sample 1"})

    # load the list of images
    loaded_objects = loadIDC("image1.idc")
    result2 = loaded_objects["result"]

    # plot the 3rd image from the list
    plot(result2[2])

    """
    Sample 2: acquire 10 images in an image stack
    """
    sizex = cam.getParam("sizex")
    sizey = cam.getParam("sizey")
    bpp = cam.getParam("bpp")

    if bpp == 8:
        d = dataObject([10, sizey, sizex], "uint8")
    else:
        d = dataObject([10, sizey, sizex], "uint16")

    for i in range(0, 10):
        cam.acquire()
        cam.copyVal(
            d[i, :, :]
        )  # partial deep copy into one part of the 3d object d

    plot(d)

    # calculate mean value of image stack in z-direction
    result_mean = dataObject()

    filter("calcMeanZ", d, result_mean, ignoreInf=0, calcStd=0)
    # result_mean is a 3d Object with [1 x sizey x sizex] dimensions.
    # We squeeze it to get a 2D Object
    result_mean = result_mean.squeeze()

    result_mean.setTag("title", "mean value of 10 acquisitions")
    result_mean.axisUnits = ("px", "px")
    result_mean.axisDescriptions = ("y", "x")
    plot(result_mean)

    # reset the auto grabbing functionality of possibly connected live images
    cam.setAutoGrabbing(currentStatus)

    # end camera
    cam.stopDevice()


if __name__ == "__main__":
    demo_CameraAndImages_Samples()
