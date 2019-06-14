.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Version 3.2.1
#########################

The version 3.2.1 is a bugfix (and minor improvements) release with respect to version 3.2.0 only.

itom
********

**Version 3.2.1 (2019-06-12)**

* install / upgrade dialog of python package manager can now store the current settings during one itom session. This simplifies the installation of several packages.
* due to upcoming support for OpenCV 4, a check for the deprecated CV_USRTYPE1 type of cv::DataType has been removed. This check was responsible to reject the unsupported datatype uint32. Now this check has been implemented again in the create methods for ito::DataObject.
* improvements of std::cout, std::cerr (as well as python print commands) to avoid deadlocks or crashes when printing thousands of lines within a very short time: inserted small delays to avoid buffer overflows.
* menu action "no available figures" set to disabled (like other similar actions)
* documentation added about how to build *itom* under CentOS / linux added


Plugins
******************

**Version 3.2.1 (2019-06-12)**

* MSMediaFoundation: major improvements concerning necessary CPU consumption (tiny sleeps inserted in while(1) loops)
* GenICam: there exists devices which cannot report the real access state. Instead they report changed the accessStatus DEVICE_ACCESS_STATUS_UNKNOWN. If this is the case, the plugin assumes a read/write access state and tries to open this device though. 
* GenICam: Start to support color cameras with the exemplary YCbCr422_8 encoding (tested with Basler puA1280-54uc)
* FittingFilters: small bug-fix in method **fillInvalidAreas**

Designer Plugins
******************

**Version 3.2.1 (2019-06-12)**

* itom2dqwtplot: improved positioning of child figures if their preferred position exceeds the geometry of the screen where the parent plot is located.
