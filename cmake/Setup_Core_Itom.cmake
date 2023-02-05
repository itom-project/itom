###############################################################################
# SETUP Configuration to define CMAKE Variables ITOM Core condiguration

if(NOT EXISTS ${OpenCV_DIR})
    if(EXISTS $ENV{OPENCV_ROOT})
        set(OpenCV_DIR $ENV{OPENCV_ROOT} CACHE PATH "Path to the OpenCV Directory")
    else(EXISTS $ENV{OPENCV_ROOT})
        set(OpenCV_DIR "OpenCV_DIR-NOTFOUND" CACHE PATH "Path to the OpenCV Directory")
    endif(EXISTS $ENV{OPENCV_ROOT})
endif(NOT EXISTS ${OpenCV_DIR})

if(WIN32)
    if(NOT EXISTS ${OpenCV_DIR})
        message(FATAL_ERROR "Depencencies Missing for OpenCV Library. Please make sure that OpenCV_DIR are well defined")
    endif(NOT EXISTS ${OpenCV_DIR})
endif(WIN32)


###############################################################################
# QT Definitions

if(NOT EXISTS BUILD_QTVERSION)
    SET(BUILD_QTVERSION "Qt6" CACHE STRING "Qt Version to be used, currently only Qt6 and Qt5 is supported.Qt6 by default.")
endif(NOT EXISTS BUILD_QTVERSION)
set_property(CACHE BUILD_QTVERSION PROPERTY STRINGS Qt6 Qt5)

if(NOT EXISTS ${Qt_DIR})
    if(EXISTS $ENV{QT_ROOT})
        set(Qt_DIR $ENV{QT_ROOT} CACHE PATH "Path to the Qt Directory")
    else(EXISTS $ENV{QT_ROOT})
        set(Qt_DIR "QT_DIR-NOTFOUND" CACHE PATH "Path to the Qt Directory")
    endif(EXISTS $ENV{QT_ROOT})
endif(NOT EXISTS ${Qt_DIR})

if(WIN32)
    if(NOT EXISTS ${Qt_DIR})
        message(FATAL_ERROR "Depencencies Missing for Qt Library. Please make sure that the Cmake Variable Qt_DIR or the Environment Variable QT_ROOT are well defined")
    endif(NOT EXISTS ${Qt_DIR})
endif(WIN32)