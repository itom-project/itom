.. include:: ../include/global.inc

determine lateral image shift and show images by using itom figure plots
*************************************************************************

A lateral image shift can be determined by cross correlation. To minimize the calculation time, the inverse fourier transformed product of the fourier transformed lateral shifted images, wheras one image is complex conjugated, is calculated

.. code-block:: python
    
    #-----------------------------------------------------------
    #
    # determine the amount of a lateral image shift
    #
    #-----------------------------------------------------------


    # import necessary modules
    #
    import numpy as np
    import scipy.misc


    #--- create test data (lateral shift an image of Lena) -----
    #

    # load an image of Lena
    #
    imageLena = scipy.misc.lena()

    # plot lena
    #
    plot(np.flipud(imageLena),'itom2DQwtFigure')


    # amount of pixel shift in x- and y-direction
    #
    xPixelShift = 16
    yPixelShift = -7


    # determine the ROI size: relative (centered) size of original image (relativeSize=1: original size)
    #
    row, col     = imageLena.shape
    relativeSize = np.floor( min( 1-abs(xPixelShift)/col, 1-abs(yPixelShift)/row ) * 10 ) /10

    x0 = int( (col - col*relativeSize)/2 )
    x1 = col-x0 + 1
    y0 = int( (row - row*relativeSize)/2 )
    y1 = row-y0 + 1

    # not shifted ROI
    image1 = imageLena[y0:y1,x0:x1].copy()
    plot(np.flipud(image1),'itom2DQwtFigure')

    # shifted ROI
    image2 = imageLena[y0+yPixelShift:y1+yPixelShift,x0+xPixelShift:x1+xPixelShift].copy()
    plot(np.flipud(image2),'itom2DQwtFigure')


    #
    #-----------------------------------------------------------


    #--- determine the pixel shift -----------------------------
    #

    # discrete fast fourier transformation and complex conjugation of image 2
    #
    image1FFT = np.fft.fft2(image1)
    image2FFT = np.conjugate( np.fft.fft2(image2) )


    # inverse fourier transformation of product -> equal to cross correlation
    #
    imageCCor = np.real( np.fft.ifft2( (image1FFT*image2FFT) ) )


    # Shift the zero-frequency component to the center of the spectrum
    #
    imageCCorShift = np.fft.fftshift(imageCCor)
    plot(imageCCorShift,'itom2DQwtFigure')


    # determine the distance of the maximum from the center
    #
    row, col = image1.shape

    yShift, xShift = np.unravel_index( np.argmax(imageCCorShift), (row,col) )

    yShift -= int(row/2)
    xShift -= int(col/2)

    print("shift of image1 in x-direction [pixel]: " + str(xShift))
    print("shift of image1 in y-direction [pixel]: " + str(yShift))

    #
    #-----------------------------------------------------------

First the necessary modules have to be imported:

.. code-block:: python
    
    # import necessary modules
    #
    import numpy as np
    import scipy.misc

Test data is created by lateral shifting a region of interest (ROI) of the Lena image. First the image of Lena is loaded:

.. code-block:: python
    
    # load an image of Lena
    #
    imageLena = scipy.misc.lena()

The image of Lena is plotted by using the itom figure plot 'itom2DQwtFigure', which is optimized for 2D static images. Since the row index for images starts at the top of an image and not at the bottom like for matrixes, the image as to be flipped up side down before plotting.

.. code-block:: python
    
    # plot lena
    #
    plot(np.flipud(imageLena),'itom2DQwtFigure')

The ROI size is determined by the amount of lateral shift in x- and y-direction. One ROI is selected from the center of the original image. Another ROI with the same size is shifted from the center about the defined amount.

.. code-block:: python
    
    # amount of pixel shift in x- and y-direction
    #
    xPixelShift = 16
    yPixelShift = -7


    # determine the ROI size: relative (centered) size of original image (relativeSize=1: original size)
    #
    row, col     = imageLena.shape
    relativeSize = np.floor( min( 1-abs(xPixelShift)/col, 1-abs(yPixelShift)/row ) * 10 ) /10

    x0 = int( (col - col*relativeSize)/2 )
    x1 = col-x0 + 1
    y0 = int( (row - row*relativeSize)/2 )
    y1 = row-y0 + 1

    # not shifted ROI
    image1 = imageLena[y0:y1,x0:x1].copy()
    plot(np.flipud(image1),'itom2DQwtFigure')

    # shifted ROI
    image2 = imageLena[y0+yPixelShift:y1+yPixelShift,x0+xPixelShift:x1+xPixelShift].copy()
    plot(np.flipud(image2),'itom2DQwtFigure')

Now the lateral shift is determined by calculating the inverse fourier transformed of the product of the fourier transformed ROIs and evaluating the distance from the center of the position of its maximum.

.. code-block:: python
    
    # discrete fast fourier transformation and complex conjugation of image 2
    #
    image1FFT = np.fft.fft2(image1)
    image2FFT = np.conjugate( np.fft.fft2(image2) )


    # inverse fourier transformation of product -> equal to cross correlation
    #
    imageCCor = np.real( np.fft.ifft2( (image1FFT*image2FFT) ) )


    # Shift the zero-frequency component to the center of the spectrum
    #
    imageCCorShift = np.fft.fftshift(imageCCor)
    plot(imageCCorShift,'itom2DQwtFigure')


    # determine the distance of the maximum from the center
    #
    row, col = image1.shape

    yShift, xShift = np.unravel_index( np.argmax(imageCCorShift), (row,col) )

    yShift -= int(row/2)
    xShift -= int(col/2)

    print("shift of image1 in x-direction [pixel]: " + str(xShift))
    print("shift of image1 in y-direction [pixel]: " + str(yShift))