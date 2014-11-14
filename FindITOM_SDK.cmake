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

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 

if (BUILD_TARGET64)
   set(CMAKE_SIZEOF_VOID_P 8)
else (BUILD_TARGET64)
   set(CMAKE_SIZEOF_VOID_P 4)
endif (BUILD_TARGET64)

IF(EXISTS ${ITOM_SDK_DIR})
    #find itom_sdk.cmake configuration file
    FIND_FILE(ITOM_SDK_CONFIG_FILE "itom_sdk.cmake" ${ITOM_SDK_DIR} DOC "")
ELSE(EXISTS ${ITOM_SDK_DIR})
    SET(ITOM_SDK_CONFIG_FILE "")
    SET(ERR_MSG "The directory indicated by ITOM_SDK_DIR could not be found.")
ENDIF(EXISTS ${ITOM_SDK_DIR})

message(STATUS ${ITOM_SDK_FIND_QUIETLY})

IF(EXISTS ${ITOM_SDK_CONFIG_FILE})
    
    INCLUDE(${ITOM_SDK_CONFIG_FILE})

    IF (${BUILD_TARGET64})
        IF (NOT ((${ITOM_SDK_BUILD_TARGET64} STREQUAL "TRUE") OR (${ITOM_SDK_BUILD_TARGET64} STREQUAL "ON")))
            MESSAGE(FATAL_ERROR "BUILD_TARGET64 (ON) option does not correspond to configuration of itom SDK")
        ENDIF()
    ELSE (${BUILD_TARGET64})
        IF (NOT ((${ITOM_SDK_BUILD_TARGET64} STREQUAL "FALSE") OR (${ITOM_SDK_BUILD_TARGET64} STREQUAL "OFF")))
            MESSAGE(FATAL_ERROR "BUILD_TARGET64 (OFF) option does not correspond to configuration of itom SDK")
        ENDIF()
    ENDIF (${BUILD_TARGET64})

    #find include directory
    FIND_PATH(ITOM_SDK_INCLUDE_DIR "itom_sdk.h" PATHS "${ITOM_SDK_DIR}" PATH_SUFFIXES "include" DOC "")
    
    FIND_PATH(ITOM_APP_DIR "itoDebugger.py" PATHS "${ITOM_SDK_DIR}" "${ITOM_DIR}" PATH_SUFFIXES ".." "." DOC "")
    get_filename_component(ITOM_APP_DIR ${ITOM_APP_DIR} ABSOLUTE)
    
    
    if ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
      SET(SDK_PLATFORM "x86")
    else ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
      SET(SDK_PLATFORM "x64")
    endif ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
    
    IF(MSVC10)
        SET(SDK_COMPILER "vc10")
    ELSEIF(MSVC9)
        SET(SDK_COMPILER "vc9")
    ELSEIF(MSVC8)
        SET(SDK_COMPILER "vc8")
    ELSEIF(MSVC)
        SET(SDK_COMPILER "vc${MSVC_VERSION}$")
    ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    SET(SDK_COMPILER "gnucxx")
    ELSE(MSVC10)
        SET(SDK_COMPILER "unknown")
    ENDIF(MSVC10)
    
    SET(ITOM_SDK_LIBSUFFIX "/lib/${SDK_COMPILER}_${SDK_PLATFORM}")

    
    #Initiate the variable before the loop
    set(GLOBAL ITOM_SDK_LIBS "")
    set(ITOM_SDK_FOUND_TMP true)

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
    
    if(NOT ITOM_SDK_FIND_COMPONENTS)
        set(ITOM_SDK_FIND_COMPONENTS ${ITOM_SDK_LIB_COMPONENTS})
    endif(NOT ITOM_SDK_FIND_COMPONENTS)
    
    SET(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIR})
    
    SET (ITOM_SDK_LIBRARIES)
    foreach(__ITOMLIB ${ITOM_SDK_FIND_COMPONENTS})
        
        if (ITOM_SDK_${__ITOMLIB}_LIBRARY)
            set(ITOM_SDK_LIBRARIES ${ITOM_SDK_LIBRARIES} ${ITOM_SDK_${__ITOMLIB}_LIBRARY})
        else()
            message(SEND_ERROR "Required component ${__ITOMLIB} could not be found in itom SDK")
        endif()
        
        #dataobject has a dependency to OpenCV, therefore adapt ITOM_SDK_INCLUDE_DIRS
        #and add the core library of OpenCV to the ITOM_SDK_LIBRARIES
        if (${__ITOMLIB} STREQUAL "dataobject")
            
            if (OpenCV_FOUND) 
                #store the current value of OpenCV_LIBS and reset it afterwards
                SET(__OpenCV_LIBS "${OpenCV_LIBS}")
            else(OpenCV_FOUND)
                SET(__OpenCV_LIBS "")
            endif (OpenCV_FOUND)
            
            if(ITOM_SDK_FIND_QUIETLY)
                find_package(OpenCV QUIET COMPONENTS core)
            else(ITOM_SDK_FIND_QUIETLY)
                find_package(OpenCV COMPONENTS core)
            endif(ITOM_SDK_FIND_QUIETLY)
            
            if(OpenCV_FOUND)
                SET(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIRS} ${OpenCV_DIR}/include)
                set(ITOM_SDK_LIBRARIES ${ITOM_SDK_LIBRARIES} ${OpenCV_LIBS})
            else(OpenCV_FOUND)
                set(ITOM_SDK_FOUND_TMP false)
                SET(ERR_MSG "OpenCV not found. Use OpenCV_DIR to indicate the (build-)folder of OpenCV.")
            endif(OpenCV_FOUND)
            
            IF(__OpenCV_LIBS)
                #reset OpenCV_LIBS
                SET(OpenCV_LIBS "${__OpenCV_LIBS}")
            ENDIF()
        endif()
        
        #pointcloud has a dependency to the core component of the point cloud library, 
        #therefore adapt ITOM_SDK_INCLUDE_DIRS and add the core library of PCL to the ITOM_SDK_LIBRARIES
        if (${__ITOMLIB} STREQUAL "pointcloud")
        
            if (PCL_FOUND)
                #store the current value of PCL_INCLUDE_DIRS and PCL_LIBRARY_DIRS and reset it afterwards
                SET(__PCL_INCLUDE_DIRS "${PCL_INCLUDE_DIRS}")
                SET(__PCL_LIBRARY_DIRS "${PCL_LIBRARY_DIRS}")
            else(PCL_FOUND)
                SET(__PCL_INCLUDE_DIRS "")
                SET(__PCL_LIBRARY_DIRS "")
            endif (PCL_FOUND)
            
            if(ITOM_SDK_FIND_QUIETLY)
                find_package(PCL 1.5.1 QUIET COMPONENTS common)
            else(ITOM_SDK_FIND_QUIETLY)
                find_package(PCL 1.5.1 COMPONENTS common)
            endif(ITOM_SDK_FIND_QUIETLY)
                
            if(PCL_FOUND)
                SET(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
                set(ITOM_SDK_LIBRARIES ${ITOM_SDK_LIBRARIES} ${PCL_LIBRARIES})
            else(PCL_FOUND)
                set(ITOM_SDK_FOUND_TMP false)
                SET(ERR_MSG "PCL not found. Use PCL_DIR to indicate the (install-)folder of PCL.")
            endif(PCL_FOUND)
            
            IF(__PCL_INCLUDE_DIRS)
                #reset PCL_INCLUDE_DIRS and PCL_LIBRARY_DIRS
                SET(PCL_INCLUDE_DIRS "${__PCL_INCLUDE_DIRS}")
                SET(PCL_LIBRARY_DIRS "${__PCL_LIBRARY_DIRS}")
            ENDIF(__PCL_INCLUDE_DIRS)
            
        endif()
        
        #itomWidgets often requires the SDK_INCLUDE_DIR/itomWidgets directory as further include directory
        if (${__ITOMLIB} STREQUAL "itomWidgets")
            SET(ITOM_SDK_INCLUDE_DIRS ${ITOM_SDK_INCLUDE_DIRS} ${ITOM_SDK_INCLUDE_DIR}/itomWidgets)
        endif()
        
    endforeach(__ITOMLIB)


    set(ITOM_SDK_FOUND ${ITOM_SDK_FOUND_TMP} CACHE BOOL "" FORCE)
    
    
ELSE(EXISTS ${ITOM_SDK_CONFIG_FILE})
    SET(ERR_MSG "File itom_sdk.cmake could not be found in ITOM_SDK_DIR")
ENDIF(EXISTS ${ITOM_SDK_CONFIG_FILE})
#====================================================


#====================================================
# Print message
#----------------------------------------------------
 if(NOT ITOM_SDK_FOUND)
        #make FIND_PACKAGE friendly
         if(NOT ITOM_SDK_FIND_QUIETLY)
                 if(ITOM_SDK_FIND_REQUIRED)
                         message(SEND_ERROR "itom SDK required but some headers or libs not found. ${ERR_MSG}")
                 else(ITOM_SDK_FIND_REQUIRED)
                         message(STATUS "WARNING: itom SDK was not found. ${ERR_MSG}")
                 endif(ITOM_SDK_FIND_REQUIRED)
         endif(NOT ITOM_SDK_FIND_QUIETLY)
 endif(NOT ITOM_SDK_FOUND)





#====================================================


