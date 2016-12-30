.. include:: ../include/global.inc


Popular Python packages
**********************************

One main advantage of |python| are thousands of packages, that enhance the base functionality in many application fields. Most of those packages are
also freely available and listed in the official python package index (https://pypi.python.org/pypi). |itom| comes with a GUI based :ref:`package manager <gui-pipmanager>`
that can be used to browse the python package index, automatically download and install or upgrade these packages.

Many 3rd party packages are completely implemented in Python code. These packages can usually be used within |itom| without limitations. Nevertheless, you have to take
care that the selected package is compatible to the right version of |python| (see About dialog of |itom|) and to your operation system (also 32bit or 64bit, depending on |itom|).
Other packages, like some numerical intense packages or packages that wrap further libraries, can internally also be written in C. Usually, the packages that are provided by
the python package index, only provide the pure sources. Hence, you have to compile the sources upon installation, which is sometimes done automatically if an appropriate compiler
is found on the destination computer. Nevertheless, this might often fail especially for Windows operation systems. In this case, check the website http://www.lfd.uci.edu/~gohlke/pythonlibs/
for ready-to-use binaries for Windows. The offered **whl** files can simply be downloaded and installed by the itom internal python package manager.

Here is a list of popular |python| packages, that can be very useful when working with |itom|:

**Numpy**

Numpy is the main |python| package for numeric calculations and work with arrays and matrices. Numpy is directly integrated in |itom| and **required** to run |itom|. For Windows users,
Numpy should always be obtained as binary package from http://www.lfd.uci.edu/~gohlke/pythonlibs/. Please consider, that a major jump of the Numpy version might require a new compilation of |itom|!
There is a strong compatibility between the main array class :py:class:`itom.dataObject` of |itom| and :py:class:`numpy.array`. For more information about this, see the section :ref:`itomDataObjectVsNumpyArray`

**Scipy**

Scipy is a sister-project to **Numpy** and provides more functions from the field of scientific calculations, statistics, linear algebra, interpolation, signal processing... Windows users
should also directly download the binaries from http://www.lfd.uci.edu/~gohlke/pythonlibs/. The version of Scipy must be compatible with Numpy, therefore try to update both at the same time.


**Matplotlib**

Matplotlib is a huge plotting package that allows the plot of many different types of schemes, graphics or charts. Matplotlib should also be downloaded as binary package for Windows users.
While |itom| comes with many built-in designer plugins that can be used to visualize camera images and matrices in a fast way, Matplotlib can be used for more specific plots and charts.
|itom| provides a specific backend for Matplotlib plots, such that the rendered output images are displayed in |itom| windows and can also be embedded in user defined GUIs. For more
information about **Matplotlib** and |itom|, see the section :ref:`pymod-matplotlib`.

**sphinx**

Sphinx is a popular package for rendering user documentations to various output formats (e.g. html, latex, QtHelp, Windows Help Format, ...). Sphinx is used to parse this documentation and for the
homepage of |itom| (itom.bitbucket.io). See the section :ref:`buildDocumentation` for more information how to create the user documentation (e.g. if you have a self-compiled version of |itom|).

**pip**

Pip is the main package that is used to install other packages. From Python 3.4 on, pip comes with the |python| installer and is implicitely called by the :ref:`package manager <gui-pipmanager>`.

**scikit-image**

Scikit-image is the image processing sub-package of **Scipy**. See the website http://scikit-image.org/ for an overview about the features of this great library.

**OpenCV**

This package is a wrapper to the popular open-source image processing library **OpenCV**. Although OpenCV is also the basic structure of :py:class:`~itom.dataObject`, the python package can be used
to call specific functions from OpenCV directly from Python. Since, the |python| wrapper was not available for Python 3 until november 2015, many functions from OpenCV are also wrapped within
the itom plugin **OpenCVFilters**.

**frosted**

This package is a syntax check for python scripts. Once the package is installed, |itom| is able to continously check your script files and show hints and bugs in the first column of the script.
This syntax check can be enabled and configured in the :ref:`property dialog of itom <gui-prop-py-general>`.

**pywin32** (Windows only)

This is a popular package, that is also required by many other packages and is used to directly call functions from DLLs under Windows. It has also many more features to interact with other applications under Windows.
Get it also as binary file from http://www.lfd.uci.edu/~gohlke/pythonlibs/.