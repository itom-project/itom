###############################################################################
# CMAKE Variables ITOM Core configuration

###############################################################################
# ITOM Diszribution Definitions

option(ITOM_BUILD_PACKAGE "Choose to build ITOM as a System specific Package for further distribution-" OFF)

###############################################################################
# Python Definitions

if(NOT EXISTS ${Python_ROOT_DIR})
    if(EXISTS $ENV{PYTHON_ROOT})
        set(Python_ROOT_DIR $ENV{PYTHON_ROOT} CACHE PATH "Path to the OpenCV Directory")
    else(EXISTS $ENV{PYTHON_ROOT})
        set(Python_ROOT_DIR "Python_ROOT_DIR-NOTFOUND" CACHE PATH "Path to the OpenCV Directory")
    endif(EXISTS $ENV{PYTHON_ROOT})
endif(NOT EXISTS ${Python_ROOT_DIR})

if(WIN32)
    if(NOT EXISTS ${Python_ROOT_DIR})
        message(FATAL_ERROR "Depencencies Missing for Python Library. Please make sure that the Cmake Variable Python_ROOT_DIR or the Environment Variable PYTHON_ROOT are well defined")
    endif(NOT EXISTS ${Python_ROOT_DIR})
endif(WIN32)


###############################################################################
# OpenCV Definitions

if(NOT EXISTS ${OpenCV_DIR})
    if(EXISTS $ENV{OPENCV_ROOT})
        set(OpenCV_DIR $ENV{OPENCV_ROOT} CACHE PATH "Path to the OpenCV Directory")
    else(EXISTS $ENV{OPENCV_ROOT})
        set(OpenCV_DIR "OpenCV_DIR-NOTFOUND" CACHE PATH "Path to the OpenCV Directory")
    endif(EXISTS $ENV{OPENCV_ROOT})
endif(NOT EXISTS ${OpenCV_DIR})

if(WIN32)
    if(NOT EXISTS ${OpenCV_DIR})
        message(FATAL_ERROR "Depencencies Missing for OpenCV Library. Please make sure that the Cmake Variable OpenCV_DIR or the Environment Variable OPENCV_ROOT are well defined")
    endif(NOT EXISTS ${OpenCV_DIR})
endif(WIN32)


###############################################################################
# QT Definitions

if(NOT EXISTS BUILD_QTVERSION)
    SET(BUILD_QTVERSION "Qt5" CACHE STRING "Qt Version to be used, currently only Qt6 and Qt5 is supported.Qt5 by default.")
endif(NOT EXISTS BUILD_QTVERSION)
set_property(CACHE BUILD_QTVERSION PROPERTY STRINGS Qt6 Qt5)

if(NOT EXISTS ${Qt_Prefix_DIR})
    if(EXISTS $ENV{QT_ROOT})
        set(Qt_Prefix_DIR $ENV{QT_ROOT} CACHE PATH "Path to the Qt Directory")
    else(EXISTS $ENV{QT_ROOT})
        set(Qt_Prefix_DIR "Qt_Prefix_DIR-NOTFOUND" CACHE PATH "Path to the Qt Directory")
    endif(EXISTS $ENV{QT_ROOT})
endif(NOT EXISTS ${Qt_Prefix_DIR})

if(WIN32)
    if(NOT EXISTS ${Qt_Prefix_DIR})
        message(FATAL_ERROR "Depencencies Missing for Qt Library. Please make sure that the Cmake Variable Qt_Prefix_DIR or the Environment Variable QT_ROOT are well defined")
    endif(NOT EXISTS ${Qt_Prefix_DIR})
endif(WIN32)