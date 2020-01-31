###########################################################
#                  Find ITOM SDK
#----------------------------------------------------------
#
## 1: Setup:
# The following variables are optionally searched for defaults
#  ITOM_SDK_DIR:            Base directory of itom SDK tree to use.
#  ITOM_DIR:         
#
## 2: Variable
# The following are set after configuration is done: 
#  
#  ITOM_SDK_FOUND
#  ITOM_SDK_LIBRARIES -> list of all necessary libraries to link against. This list also contains
#                        necessary libraries from dependent packages like OpenCV (dataobject) or PCL (pointcloud)
#  ITOM_SDK_INCLUDE_DIR -> include directory of itom SDK
#  ITOM_SDK_INCLUDE_DIRS -> list of necessary include directories including ITOM_SDK_INCLUDE_DIR plus
#                           additional include directories e.g. from dependent packages like OpenCV.
#
#
#----------------------------------------------------------

cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

if(CMAKE_SIZEOF_VOID_P GREATER 4)
    option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
else()
    option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
endif()

if (NOT EXISTS ${ITOM_SDK_CONFIG_FILE})
    unset(ITOM_SDK_CONFIG_FILE CACHE)
endif()

if(EXISTS ${ITOM_SDK_DIR})
    #find itom_sdk.cmake configuration file
    find_file(ITOM_SDK_CONFIG_FILE "itom_sdk.cmake" PATHS ${ITOM_SDK_DIR} PATH_SUFFIXES cmake DOC "")
else()
    set(ITOM_SDK_CONFIG_FILE "")
    set(ERR_MSG "The directory indicated by ITOM_SDK_DIR could not be found.")
endif()

if(EXISTS ${ITOM_SDK_CONFIG_FILE})
    
    include(${ITOM_SDK_CONFIG_FILE})
    
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    
    if(ITOM_SDK_PCL_SUPPORT)
        add_definitions(-DUSEPCL -D_USEPCL)
    endif(ITOM_SDK_PCL_SUPPORT)

    string( TOLOWER "${ITOM_SDK_BUILD_TARGET64}" ITOM_SDK_BUILD_TARGET64_LOWER )
    if(BUILD_TARGET64)
        if(NOT ((${ITOM_SDK_BUILD_TARGET64_LOWER} STREQUAL "true") OR (${ITOM_SDK_BUILD_TARGET64_LOWER} STREQUAL "on")))
            message(WARNING "BUILD_TARGET64 does not correspond to configuration of itom SDK. SDK was build with option ${ITOM_SDK_BUILD_TARGET64}. BUILD_TARGET64 is set to OFF.")
            set(BUILD_TARGET64 OFF CACHE BOOL "Build for 64 bit target if set to ON or 32 bit if set to OFF." FORCE)
        endif()
    else (BUILD_TARGET64)
        if(NOT ((${ITOM_SDK_BUILD_TARGET64_LOWER} STREQUAL "false") OR (${ITOM_SDK_BUILD_TARGET64_LOWER} STREQUAL "off")))
            message(WARNING "BUILD_TARGET64 does not correspond to configuration of itom SDK. SDK was build with option ${ITOM_SDK_BUILD_TARGET64}. BUILD_TARGET64 is set to ON.")
            set(BUILD_TARGET64 ON CACHE BOOL "Build for 64 bit target if set to ON or 32 bit if set to OFF." FORCE)
        endif()
    endif(BUILD_TARGET64)
    
    if(BUILD_TARGET64)
        if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
            message(FATAL_ERROR "BUILD_TARGET64 is ON, but CMAKE_SIZEOF_VOID_P is unequal to 8 bytes. Maybe change the compiler.")
        endif()
    else()
        if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
            message(FATAL_ERROR "BUILD_TARGET64 is OFF, but CMAKE_SIZEOF_VOID_P is unequal to 4 bytes. Maybe change the compiler.")
        endif()
    endif()

    #find include directory
    find_path(ITOM_SDK_INCLUDE_DIR "itom_sdk.h" PATHS "${ITOM_SDK_DIR}" PATH_SUFFIXES "include" DOC "")
    
    find_path(ITOM_APP_DIR "itoDebugger.py" PATHS "${ITOM_SDK_DIR}" "${ITOM_DIR}" PATH_SUFFIXES ".." "." DOC "")
    get_filename_component(ITOM_APP_DIR ${ITOM_APP_DIR} ABSOLUTE)
    
    if(EXISTS "${ITOM_APP_DIR}")
        #try to load the CMakeCache file from itom and extract some useful variables. 
        #The variables must be filepathes or pathes. They are only copied to this project
        #... if they exist in itom's CMakeCache, 
        #... if they are valid, 
        #... if they exist in the file system and 
        #... if they do not exist or are not valid in this project, yet.
        set(CACHE_VARIABLES VTK_DIR VISUALLEAKDETECTOR_DIR Qt5_DIR PCL_DIR OpenCV_DIR EIGEN_INCLUDE_DIRS Boost_LIBRARY_DIR Boost_INCLUDE_DIR BLUBBER GIT_EXECUTABLE)
        
        if(EXISTS "${ITOM_APP_DIR}/CMakeCache.txt")
            load_cache("${ITOM_APP_DIR}" READ_WITH_PREFIX "ITOMCACHE_" ${CACHE_VARIABLES})
            
            message(STATUS "Try to load selected CMake variables from CMakeCache of itom project and copy them to this set of variables.")
            
            foreach(CACHE_VAR ${CACHE_VARIABLES})
                
                if(DEFINED "ITOMCACHE_${CACHE_VAR}")
                    message(STATUS "  - Variable ${CACHE_VAR} exists in itom cache: ${ITOMCACHE_${CACHE_VAR}}")
                    
                    if(DEFINED ${CACHE_VAR} AND ${CACHE_VAR})
                        message(STATUS "    -> This variable is not copied since it already is defined and valid in this project: ${${CACHE_VAR}}")
                    else()
                        if(EXISTS "${ITOMCACHE_${CACHE_VAR}}")
                            message(STATUS "    -> This variable is copied to this project.")
                            set(${CACHE_VAR} "${ITOMCACHE_${CACHE_VAR}}" CACHE PATH "Variable obtained from CMakeCache of itom project." FORCE)
                        else()
                            message(STATUS "    -> This variable is not copied to this project since it is invalid.")
                        endif()
                    endif()
                else()
                    message(STATUS "  - Variable ${CACHE_VAR} did not exist in itom cache. Ignore it.")
                endif()
            endforeach()
        endif()
    endif()
    
    if(BUILD_TARGET64)
      set(SDK_PLATFORM "x64")
    else()
      set(SDK_PLATFORM "x86")
    endif()
    
    # The following list has to be consistent with the
    # macro itom_add_library_to_appdir_and_sdk of ItomBuildMacroInternal.cmake.
    # From VS higher than 1900, the default case vc${MSVC_VERSION} is used.
    if(MSVC_VERSION EQUAL 1900)
        set(SDK_COMPILER "vc14")
    elseif(MSVC_VERSION EQUAL 1800)
        set(SDK_COMPILER "vc12")
    elseif(MSVC_VERSION EQUAL 1700)
        set(SDK_COMPILER "vc11")
    elseif(MSVC_VERSION EQUAL 1600)
        set(SDK_COMPILER "vc10")
    elseif(MSVC_VERSION EQUAL 1500)
        set(SDK_COMPILER "vc9")
    elseif(MSVC_VERSION EQUAL 1400)
        set(SDK_COMPILER "vc8")
    elseif(MSVC)
        set(SDK_COMPILER "vc${MSVC_VERSION}")
    elseif(CMAKE_COMPILER_IS_GNUCXX)
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SDK_COMPILER "clang")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(SDK_COMPILER "intel")
    elseif(APPLE)
        set(SDK_COMPILER "osx_default")
    else()
        set(SDK_COMPILER "unknown")
    endif()
    
    set(ITOM_SDK_LIBSUFFIX "/lib/${SDK_COMPILER}_${SDK_PLATFORM}")
    message(STATUS "ITOM LIB SUFFIX: ${ITOM_SDK_LIBSUFFIX}")
    
    #Initiate the variable before the loop
    set(GLOBAL ITOM_SDK_LIBS "")
    set(ITOM_SDK_FOUND_TMP true)
    
    if(NOT ITOM_SDK_FIND_COMPONENTS)
        set(ITOM_SDK_LIB_COMPONENTS ${ITOM_SDK_LIB_COMPONENTS}) #ITOM_SDK_LIB_COMPONENTS is described in itom_sdk.cmake
    else()
        foreach(__ITOMLIB ${ITOM_SDK_LIB_COMPONENTS})
            set(ITOM_SDK_LIB_COMPONENTS ${ITOM_SDK_FIND_COMPONENTS} "${__ITOMLIB}")
        endforeach(__ITOMLIB)
    endif()

    # Loop over each components
    foreach(__ITOMLIB ${ITOM_SDK_LIB_COMPONENTS})
            find_library(ITOM_SDK_${__ITOMLIB}_LIBRARY_DEBUG NAMES "${__ITOMLIB}d"  PATHS "${ITOM_SDK_DIR}${ITOM_SDK_LIBSUFFIX}" NO_DEFAULT_PATH)
            find_library(ITOM_SDK_${__ITOMLIB}_LIBRARY_RELEASE NAMES "${__ITOMLIB}" PATHS "${ITOM_SDK_DIR}${ITOM_SDK_LIBSUFFIX}" NO_DEFAULT_PATH)
            
            #Remove the cache value
            set(ITOM_SDK_${__ITOMLIB}_LIBRARY "" CACHE STRING "" FORCE)
            
            #both debug/release
            if(ITOM_SDK_${__ITOMLIB}_LIBRARY_DEBUG AND ITOM_SDK_${__ITOMLIB}_LIBRARY_RELEASE)
                    set(ITOM_SDK_${__ITOMLIB}_LIBRARY debug ${ITOM_SDK_${__ITOMLIB}_LIBRARY_DEBUG} optimized ${ITOM_SDK_${__ITOMLIB}_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
            #only debug
            elseif(ITOM_SDK_${__ITOMLIB}_LIBRARY_DEBUG)
                    set(ITOM_SDK_${__ITOMLIB}_LIBRARY ${ITOM_SDK_${__ITOMLIB}_LIBRARY_DEBUG}  CACHE STRING "" FORCE)
            #only release
            elseif(ITOM_SDK_${__ITOMLIB}_LIBRARY_RELEASE)
                    set(ITOM_SDK_${__ITOMLIB}_LIBRARY ${ITOM_SDK_${__ITOMLIB}_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
            #no library found
            else()
              if(${__ITOMLIB} STREQUAL "dataobject")
                    set(ITOM_SDK_FOUND_TMP false)
                    #message(STATUS "${OpenCV_DIR} -- ${OPENCV_LIB_COMPONENTS} --  ${__ITOMLIB}${CVLIB_SUFFIX}d not found")
              endif()
            endif()
            
    endforeach(__ITOMLIB)
    
    
    set(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIR})
    
    set(ITOM_SDK_LIBRARIES)
    foreach(__ITOMLIB ${ITOM_SDK_LIB_COMPONENTS})
        
        if(ITOM_SDK_${__ITOMLIB}_LIBRARY)
            set(ITOM_SDK_LIBRARIES ${ITOM_SDK_LIBRARIES} ${ITOM_SDK_${__ITOMLIB}_LIBRARY})
        else()
            message(SEND_ERROR "Required component ${__ITOMLIB} could not be found in itom SDK")
        endif()
        
        #dataobject has a dependency to OpenCV, therefore adapt ITOM_SDK_INCLUDE_DIRS
        #and add the core library of OpenCV to the ITOM_SDK_LIBRARIES
        if(${__ITOMLIB} STREQUAL "dataobject")
            
            if(OpenCV_FOUND) 
                #store the current value of OpenCV_LIBS and reset it afterwards
                set(__OpenCV_LIBS "${OpenCV_LIBS}")
            else(OpenCV_FOUND)
                set(__OpenCV_LIBS "")
            endif(OpenCV_FOUND)
            
            if(ITOM_SDK_FIND_QUIETLY)
                find_package(OpenCV QUIET COMPONENTS core)
            else(ITOM_SDK_FIND_QUIETLY)
                find_package(OpenCV COMPONENTS core)
            endif(ITOM_SDK_FIND_QUIETLY)
            
            if(OpenCV_FOUND)
                set(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIRS} ${OpenCV_DIR}/include)
                set(ITOM_SDK_LIBRARIES ${ITOM_SDK_LIBRARIES} ${OpenCV_LIBS})
            else(OpenCV_FOUND)
                set(ITOM_SDK_FOUND_TMP false)
                set(ERR_MSG "OpenCV not found. Use OpenCV_DIR to indicate the (build-)folder of OpenCV.")
            endif(OpenCV_FOUND)
            
            if(__OpenCV_LIBS)
                #reset OpenCV_LIBS
                set(OpenCV_LIBS "${__OpenCV_LIBS}")
            endif()
        endif()
        
        #pointcloud has a dependency to the core component of the point cloud library, 
        #therefore adapt ITOM_SDK_INCLUDE_DIRS and add the core library of PCL to the ITOM_SDK_LIBRARIES
        if(${__ITOMLIB} STREQUAL "pointcloud")
        
            if(PCL_FOUND)
                #store the current value of PCL_INCLUDE_DIRS and PCL_LIBRARY_DIRS and reset it afterwards
                set(__PCL_INCLUDE_DIRS "${PCL_INCLUDE_DIRS}")
                set(__PCL_LIBRARY_DIRS "${PCL_LIBRARY_DIRS}")
            else(PCL_FOUND)
                set(__PCL_INCLUDE_DIRS "")
                set(__PCL_LIBRARY_DIRS "")
            endif(PCL_FOUND)
            
            if(ITOM_SDK_FIND_QUIETLY)
                find_package(PCL 1.5.1 QUIET COMPONENTS common)
            else(ITOM_SDK_FIND_QUIETLY)
                find_package(PCL 1.5.1 COMPONENTS common)
            endif(ITOM_SDK_FIND_QUIETLY)
                
            if(PCL_FOUND)
                set(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
                set(ITOM_SDK_LIBRARIES ${ITOM_SDK_LIBRARIES} ${PCL_LIBRARIES})
            else(PCL_FOUND)
                set(ITOM_SDK_FOUND_TMP false)
                set(ERR_MSG "PCL not found. Use PCL_DIR to indicate the (install-)folder of PCL.")
            endif(PCL_FOUND)
            
            if(__PCL_INCLUDE_DIRS)
                #reset PCL_INCLUDE_DIRS and PCL_LIBRARY_DIRS
                set(PCL_INCLUDE_DIRS "${__PCL_INCLUDE_DIRS}")
                set(PCL_LIBRARY_DIRS "${__PCL_LIBRARY_DIRS}")
            endif(__PCL_INCLUDE_DIRS)
            
        endif()
        
        #itomWidgets often requires the SDK_INCLUDE_DIR/itomWidgets directory as further include directory
        if(${__ITOMLIB} STREQUAL "itomWidgets")
            set(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIRS} ${ITOM_SDK_INCLUDE_DIR}/itomWidgets)
        endif()
        
    endforeach(__ITOMLIB)


    set(ITOM_SDK_FOUND ${ITOM_SDK_FOUND_TMP} CACHE BOOL "" FORCE)
    
    
else(EXISTS ${ITOM_SDK_CONFIG_FILE})
    set(ERR_MSG "File itom_sdk.cmake could not be found in subdirectory 'cmake' of ITOM_SDK_DIR")
endif(EXISTS ${ITOM_SDK_CONFIG_FILE})
#====================================================


#====================================================
# Print message
#----------------------------------------------------
if(NOT ITOM_SDK_FOUND)
        #make FIND_PACKAGE friendly
         if(NOT ITOM_SDK_FIND_QUIETLY)
                 if(ITOM_SDK_FIND_REQUIRED)
                         message(SEND_ERROR "itom SDK required but some headers or libs not found. ${ERR_MSG}")
                 else()
                         message(STATUS "WARNING: itom SDK was not found. ${ERR_MSG}")
                 endif()
         endif()
 endif()





#====================================================

