###########################################################
#                  Find OpenCV Library
# See http://sourceforge.net/projects/opencvlibrary/
#----------------------------------------------------------
#
## 1: Setup:
# The following variables are optionally searched for defaults
#  OpenCV_DIR:            Base directory of OpenCv tree to use.
#
## 2: Variable
# The following are set after configuration is done: 
#  
#  OpenCV_FOUND
#  OpenCV_LIBS
#  OpenCV_INCLUDE_DIR
#  OpenCV_VERSION (OpenCV_VERSION_MAJOR, OpenCV_VERSION_MINOR, OpenCV_VERSION_PATCH)
#
#
# Deprecated variable are used to maintain backward compatibility with
# the script of Jan Woetzel (2006/09): www.mip.informatik.uni-kiel.de/~jw
#  OpenCV_INCLUDE_DIRS
#  OpenCV_LIBRARIES
#  OpenCV_LINK_DIRECTORIES
# 
## 3: Version
#
# 2010/04/07 Benoit Rat, Correct a bug when OpenCVConfig.cmake is not found.
# 2010/03/24 Benoit Rat, Add compatibility for when OpenCVConfig.cmake is not found.
# 2010/03/22 Benoit Rat, Creation of the script.
#
#
# tested with:
# - OpenCV 2.1:  MinGW, MSVC2008
# - OpenCV 2.0:  MinGW, MSVC2008, GCC4
#
#
## 4: Licence:
#
# LGPL 2.1 : GNU Lesser General Public License Usage
# Alternatively, this file may be used under the terms of the GNU Lesser

# General Public License version 2.1 as published by the Free Software
# Foundation and appearing in the file LICENSE.LGPL included in the
# packaging of this file.  Please review the following information to
# ensure the GNU Lesser General Public License version 2.1 requirements
# will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
# 
#----------------------------------------------------------

#on linux the hint helps to find opencv obtained by the package manager
find_path(
    OpenCV_DIR 
    "OpenCVConfig.cmake" 
    HINTS 
    "/usr/share/OpenCV" 
    "/usr/local/share/OpenCV" 
    "/usr/lib64/OpenCV" 
    "/usr/lib64/cmake/OpenCV" 
    "/usr/lib/OpenCV" 
    "/usr/lib/x86_64-linux-gnu/cmake/opencv4"
    DOC "Root directory of OpenCV")

set(CVLIB_LIBSUFFIX "/lib")
#set(OpenCV_LIB_VERSION "0815" CACHE PATH "version of OpenCV")

if(OpenCV_FIND_COMPONENTS)
    message(STATUS "OpenCV components: ${OpenCV_FIND_COMPONENTS}")
endif()

##====================================================
## Find OpenCV libraries
##----------------------------------------------------
if(EXISTS "${OpenCV_DIR}")

    #When its possible to use the Config script use it.
    if(EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
        
        ## include the standard CMake script
        include("${OpenCV_DIR}/OpenCVConfig.cmake")

        ## Search for a specific version
        set(CVLIB_SUFFIX "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
        
        set(OpenCV_BIN_DIR "${OpenCV_LIB_PATH}/../bin" CACHE PATH "OpenCV bin dir")
        set(OpenCV_LIB_VERSION ${CVLIB_SUFFIX} CACHE PATH "version of OpenCV")
    
    elseif(EXISTS "${OpenCV_DIR}/include" AND EXISTS "${OpenCV_DIR}/x86" AND EXISTS "${OpenCV_DIR}/x64")
        #the OpenCV_DIR seems to point to a build version of OpenCV 2.3 on a windows pc
        if(NOT OpenCV_FIND_COMPONENTS)
            set(OPENCV_LIB_COMPONENTS opencv_core opencv_ml opencv_highgui opencv_imgproc)
        else()
            foreach(__CVLIB ${OpenCV_FIND_COMPONENTS})
                set(OPENCV_LIB_COMPONENTS ${OPENCV_LIB_COMPONENTS} "opencv_${__CVLIB}")
            endforeach(__CVLIB)
        endif()
        
        set(OpenCV_INCLUDE_DIR ) #reset it
        find_path(OpenCV_INCLUDE_DIR "cv.h" PATHS "${OpenCV_DIR}" PATH_SUFFIXES "include" "include/opencv" DOC "")
        get_filename_component(OpenCV_INCLUDE_DIR "${OpenCV_INCLUDE_DIR}/.." REALPATH)
        if(EXISTS  ${OpenCV_INCLUDE_DIRS})
                include_directories(${OpenCV_INCLUDE_DIR})
        endif(EXISTS  ${OpenCV_INCLUDE_DIRS})
        
        #message(SEND_ERROR "include dir: ${OpenCV_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR2}")

        #Find OpenCV version by looking at cvver.h
        file(STRINGS ${OpenCV_INCLUDE_DIR}/opencv2/core/version.hpp OpenCV_VERSIONS_TMP REGEX "^#define CV_[A-Z]+_VERSION[ \t]+[0-9]+$")
        string(REGEX REPLACE ".*#define CV_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" OpenCV_VERSION_MAJOR ${OpenCV_VERSIONS_TMP})
        string(REGEX REPLACE ".*#define CV_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" OpenCV_VERSION_MINOR ${OpenCV_VERSIONS_TMP})
        string(REGEX REPLACE ".*#define CV_SUBMINOR_VERSION[ \t]+([0-9]+).*" "\\1" OpenCV_VERSION_PATCH ${OpenCV_VERSIONS_TMP})
        set(OpenCV_VERSION ${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH} CACHE STRING "" FORCE)
        set(CVLIB_SUFFIX "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
        set(CVLIB_LIBSUFFIX "/x86/vc10/lib")
        set(OpenCV_LIB_VERSION ${CVLIB_SUFFIX} CACHE PATH "version of OpenCV")
        
        if(MSVC10)
            if(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x64/vc10/lib")
            else(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x86/vc10/lib")
            endif(CMAKE_CL_64)
		elseif(MSVC11)
            if(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x64/vc11/lib")
            else(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x86/vc11/lib")
            endif(CMAKE_CL_64)
		elseif(MSVC12)
            if(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x64/vc12/lib")
            else(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x86/vc12/lib")
            endif(CMAKE_CL_64)
        elseif(MSVC14)
            if(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x64/vc14/lib")
            else(CMAKE_CL_64)   
                set(CVLIB_LIBSUFFIX "/x86/vc14/lib")
            endif(CMAKE_CL_64)
        elseif(MSVC90)
            if(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x64/vc9/lib")
            else(CMAKE_CL_64)
                set(CVLIB_LIBSUFFIX "/x86/vc9/lib")
            endif(CMAKE_CL_64)
        endif(MSVC10)
        
        set(OpenCV_BIN_DIR "${OpenCV_DIR}${CVLIB_LIBSUFFIX}/../bin" CACHE PATH "OpenCV bin dir")
        
    #Otherwise it try to guess it.
    else(EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    
        if(NOT OpenCV_FIND_COMPONENTS)
            set(OPENCV_LIB_COMPONENTS cxcore cv ml highgui cvaux imgproc)
        else()
            foreach(__CVLIB ${OpenCV_FIND_COMPONENTS})
                set(OPENCV_LIB_COMPONENTS ${OPENCV_LIB_COMPONENTS} "${__CVLIB}")
            endforeach(__CVLIB)
        endif()
        set(OPENCV_LIB_COMPONENTS cxcore cv ml highgui cvaux imgproc)
        find_path(OpenCV_INCLUDE_DIR "cv.h" PATHS "${OpenCV_DIR}" PATH_SUFFIXES "include" "include/opencv" DOC "")
        if(EXISTS  ${OpenCV_INCLUDE_DIRS})
                include_directories(${OpenCV_INCLUDE_DIR})
        endif(EXISTS  ${OpenCV_INCLUDE_DIRS})

        #Find OpenCV version by looking at cvver.h
        file(STRINGS ${OpenCV_INCLUDE_DIR}/cvver.h OpenCV_VERSIONS_TMP REGEX "^#define CV_[A-Z]+_VERSION[ \t]+[0-9]+$")
        string(REGEX REPLACE ".*#define CV_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" OpenCV_VERSION_MAJOR ${OpenCV_VERSIONS_TMP})
        string(REGEX REPLACE ".*#define CV_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" OpenCV_VERSION_MINOR ${OpenCV_VERSIONS_TMP})
        string(REGEX REPLACE ".*#define CV_SUBMINOR_VERSION[ \t]+([0-9]+).*" "\\1" OpenCV_VERSION_PATCH ${OpenCV_VERSIONS_TMP})
        set(OpenCV_VERSION ${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH} CACHE STRING "" FORCE)
        set(CVLIB_SUFFIX "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
        set(OpenCV_LIB_VERSION ${CVLIB_SUFFIX} CACHE PATH "version of OpenCV")

    endif(EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")

    #message(STATUS "GENERATOR: ::: ${CMAKE_GENERATOR} - ${BUILD_TARGET64} -- ${CVLIB_LIBSUFFIX}")

    ## Initiate the variable before the loop
    set(GLOBAL OpenCV_LIBS "")
    set(OpenCV_FOUND_TMP true)
    
    #if(EXISTS "${OpenCV_BIN_DIR}/release/" AND "${OpenCV_BIN_DIR}/debug/"}
    #    set(OpenCV_BIN_RELEASE_DIR "${OpenCV_DIR}${CVLIB_LIBSUFFIX}/../bin" CACHE PATH "OpenCV bin dir")
    #    set(OpenCV_BIN_DEBUG_DIR "${OpenCV_DIR}${CVLIB_LIBSUFFIX}/../bin" CACHE PATH "OpenCV bin dir")
    #endif
#    message(STATUS "OpenCV_BIN_DIR: ${OpenCV_BIN_DIR}; OpenCV_DIR: ${OpenCV_DIR}")
    
    ## Loop over each components
    foreach(__CVLIB ${OPENCV_LIB_COMPONENTS})
            
            find_library(OpenCV_${__CVLIB}_LIBRARY_DEBUG NAMES "${__CVLIB}${CVLIB_SUFFIX}d" "lib${__CVLIB}${CVLIB_SUFFIX}d"  PATHS "${OpenCV_DIR}${CVLIB_LIBSUFFIX}" NO_DEFAULT_PATH)
            find_library(OpenCV_${__CVLIB}_LIBRARY_RELEASE NAMES "${__CVLIB}${CVLIB_SUFFIX}" "lib${__CVLIB}${CVLIB_SUFFIX}" PATHS "${OpenCV_DIR}${CVLIB_LIBSUFFIX}" NO_DEFAULT_PATH)
            
            #Remove the cache value
            set(OpenCV_${__CVLIB}_LIBRARY "" CACHE STRING "" FORCE)
            
            #both debug/release
            if(OpenCV_${__CVLIB}_LIBRARY_DEBUG AND OpenCV_${__CVLIB}_LIBRARY_RELEASE)
                    set(OpenCV_${__CVLIB}_LIBRARY debug ${OpenCV_${__CVLIB}_LIBRARY_DEBUG} optimized ${OpenCV_${__CVLIB}_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
            #only debug
            elseif(OpenCV_${__CVLIB}_LIBRARY_DEBUG)
                    set(OpenCV_${__CVLIB}_LIBRARY ${OpenCV_${__CVLIB}_LIBRARY_DEBUG}  CACHE STRING "" FORCE)
            #only release
            elseif(OpenCV_${__CVLIB}_LIBRARY_RELEASE)
                    set(OpenCV_${__CVLIB}_LIBRARY ${OpenCV_${__CVLIB}_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
            #no library found
            else()
                    set(OpenCV_FOUND_TMP false)
                    message(STATUS "$[lib]{__CVLIB}${CVLIB_SUFFIX}[d] or not found")
            endif()
            
            #Add to the general list
            if(OpenCV_${__CVLIB}_LIBRARY)
                    set(OpenCV_LIBS ${OpenCV_LIBS} ${OpenCV_${__CVLIB}_LIBRARY})
            endif(OpenCV_${__CVLIB}_LIBRARY)
            
    endforeach(__CVLIB)

    set(OpenCV_FOUND ${OpenCV_FOUND_TMP} CACHE BOOL "" FORCE)

else(EXISTS "${OpenCV_DIR}")
        set(ERR_MSG "Please specify OpenCV directory using OpenCV_DIR env. variable")
endif(EXISTS "${OpenCV_DIR}")
##====================================================


##====================================================
## Print message
##----------------------------------------------------
if(NOT OpenCV_FOUND)
        # make FIND_PACKAGE friendly
        if(NOT OpenCV_FIND_QUIETLY)
                if(OpenCV_FIND_REQUIRED)
                        message(SEND_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
                else(OpenCV_FIND_REQUIRED)
                        message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
                endif(OpenCV_FIND_REQUIRED)
        endif(NOT OpenCV_FIND_QUIETLY)
endif(NOT OpenCV_FOUND)
##====================================================


##====================================================
## Backward compatibility
##----------------------------------------------------
if(OpenCV_FOUND)
        option(OpenCV_BACKWARD_COMPA "Add some variable to make this script compatible with the other version of FindOpenCV.cmake" false)
        if(OpenCV_BACKWARD_COMPA)
                find_path(OpenCV_INCLUDE_DIRS "cv.h" PATHS "${OpenCV_DIR}" PATH_SUFFIXES "include" "include/opencv" DOC "include directory") 
                find_path(OpenCV_INCLUDE_DIR "cv.h" PATHS "${OpenCV_DIR}" PATH_SUFFIXES "include" "include/opencv" DOC "include directory")
                set(OpenCV_LIBRARIES "${OpenCV_LIBS}" CACHE STRING "" FORCE)
        endif(OpenCV_BACKWARD_COMPA)
endif(OpenCV_FOUND)
##====================================================
