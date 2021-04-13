from itom import *

# Demo about using filters provided by plugins
def demo_opencv_filter():
    # create a randomly filled  150x150px data object
    dObj = dataObject.rand([150, 150], "float32")

    # get information about the filter and its parameters
    # first, no idea about the filters name, therefore get all names containing the keyword "cv":
    filterHelp("cv")

    # the desired filter is called "cvMedianBlur", now obtain detailed information
    filterHelp("cvMedianBlur")

    # create an empty output image (its content will be filled within the filter-call)
    outputImage = dataObject()

    # call the filter
    # the parameters after the filter name are the mandatory parameters followed by the optional ones (if desired):
    # 1. parameter (mand): input image
    # 2. parameter (mand): output image
    # 3. kernellength (opt): size of squared filter kernel (default: 3)
    filter("cvMedianBlur", dObj, outputImage, 5)

    plot(dObj)  # plot original image
    plot(outputImage)  # plot filtered image


if __name__ == "__main__":
    demo_opencv_filter()
