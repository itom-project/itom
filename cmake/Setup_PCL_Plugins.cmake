###############################################################################
# SETUP Configuration to define CMAKE Variables
# to be used for the PCL Detection in ITOM

if(NOT EXISTS ${FLANN_ROOT})
    if(EXISTS $ENV{FLANN_ROOT})
        set(FLANN_ROOT $ENV{FLANN_ROOT} CACHE PATH "Path to the FLANN Directory")
    else(EXISTS $ENV{FLANN_ROOT})
        set(FLANN_ROOT "FLANN_ROOT-NOTFOUND" CACHE PATH "Path to the FLANN Directory")
    endif(EXISTS $ENV{FLANN_ROOT})
endif(NOT EXISTS ${FLANN_ROOT})

if(NOT EXISTS ${QHULL_ROOT})
    if(EXISTS $ENV{QHULL_ROOT})
        set(QHULL_ROOT $ENV{QHULL_ROOT} CACHE PATH "Path to the QHULL Directory")
    else(EXISTS $ENV{QHULL_ROOT})
        set(QHULL_ROOT "QHULL_ROOT-NOTFOUND" CACHE PATH "Path to the QHULL Directory")
    endif(EXISTS $ENV{QHULL_ROOT})
endif(NOT EXISTS ${QHULL_ROOT})

if(NOT EXISTS ${VTK_DIR})
    if(EXISTS $ENV{VTK_ROOT})
        set(VTK_DIR $ENV{VTK_ROOT} CACHE PATH "Path to the VTK Directory")
    else(EXISTS $ENV{VTK_ROOT})
        set(VTK_DIR "VTK_DIR-NOTFOUND" CACHE PATH "Path to the VTK Directory")
    endif(EXISTS $ENV{VTK_ROOT})
endif(NOT EXISTS ${VTK_DIR})

if(WIN32)
    if(NOT EXISTS ${FLANN_ROOT} OR NOT EXISTS ${QHULL_ROOT} OR NOT EXISTS ${VTK_DIR})
        message(FATAL_ERROR "Dependencies Missing for Point-Cloud Library. Please make sure that FLANN_ROOT, QHULL_ROOT and VTK_ROOT are well defined")
    endif(NOT EXISTS ${FLANN_ROOT} OR NOT EXISTS ${QHULL_ROOT} OR NOT EXISTS ${VTK_DIR})
endif(WIN32)

include(Setup_PCL_Itom)
