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
#  ITOM_SDK_LIBS
#  ITOM_SDK_INCLUDE_DIR
#
#
#----------------------------------------------------------

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 

if (BUILD_TARGET64)
   set(CMAKE_SIZEOF_VOID_P 8)
else (BUILD_TARGET64)
   set(CMAKE_SIZEOF_VOID_P 4)
endif (BUILD_TARGET64)

IF(EXISTS "${ITOM_SDK_DIR}")
    #message(FATAL_ERROR "blub - ${ITOM_SDK_INCLUDE_DIRS} - ${ITOM_SDK_DIR}")
    #find include directory
    FIND_PATH(ITOM_SDK_INCLUDE_DIR "itom_sdk.h" PATHS "${ITOM_SDK_DIR}" PATH_SUFFIXES "include" DOC "")

    IF(EXISTS  ${ITOM_SDK_INCLUDE_DIRS})
        include_directories(${ITOM_SDK_INCLUDE_DIR})
    ENDIF(EXISTS  ${ITOM_SDK_INCLUDE_DIRS})
    
    FIND_PATH(ITOM_APP_DIR "itoDebugger.py" PATHS "${ITOM_SDK_DIR}" "${ITOM_DIR}" PATH_SUFFIXES ".." "." DOC "")
    get_filename_component(ITOM_APP_DIR ${ITOM_APP_DIR} ABSOLUTE)
    
    set(ITOM_SDK_LIB_COMPONENTS dataobject pointcloud qpropertyeditor)
        
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
    ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
	SET(SDK_COMPILER "gnucxx")
    ELSE(MSVC10)
        SET(SDK_COMPILER "unknown")
    ENDIF(MSVC10)
    
    SET(ITOM_SDK_LIBSUFFIX "/lib/${SDK_COMPILER}_${SDK_PLATFORM}")
	#message(STATUS "ITOM_SDK_LIBSUFFIX ${ITOM_SDK_LIBSUFFIX} ${CMAKE_SIZEOF_VOID_P}")
    
    
    
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
            
            #Add to the general list
            if(ITOM_SDK_${__ITOMLIB}_LIBRARY)
                    set(ITOM_SDK_LIBS ${ITOM_SDK_LIBS} ${ITOM_SDK_${__ITOMLIB}_LIBRARY})
            endif(ITOM_SDK_${__ITOMLIB}_LIBRARY)
            
    endforeach(__ITOMLIB)


    set(ITOM_SDK_FOUND ${ITOM_SDK_FOUND_TMP} CACHE BOOL "" FORCE)
    
    
ELSE(EXISTS "${ITOM_SDK_DIR}")
    SET(ERR_MSG "Please specify itom SDK directory using ITOM_SDK_DIR env. variable")
ENDIF(EXISTS "${ITOM_SDK_DIR}")
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


