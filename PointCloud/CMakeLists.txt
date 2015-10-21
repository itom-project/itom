SET (target_name pointcloud)
project(${target_name}) 
 
cmake_minimum_required(VERSION 2.8)

IF(APPLE AND CMAKE_VERSION VERSION_GREATER 2.8.7)
    if(POLICY CMP0042)
        cmake_policy(SET CMP0042 OLD)
    ENDIF(POLICY CMP0042)
ENDIF()

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.") 
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
 

IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DPOINTCLOUD_DLL -D_POINTCLOUD_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if pointcloud is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)
 
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

find_package(OpenCV COMPONENTS core REQUIRED)
find_package(PCL 1.5.1 REQUIRED COMPONENTS common)

IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

IF (BUILD_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
ENDIF (BUILD_ITOMLIBS_SHARED)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)


INCLUDE_DIRECTORIES( 
    ${PCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${OpenCV_DIR}/include
)

set(libSrcs 
	${CMAKE_CURRENT_SOURCE_DIR}/pclVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pclStructures.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pclStructures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pclFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pclFunctions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/pclFunctionsImpl.h
)

if(MSVC)
	list(APPEND libSrcs ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif(MSVC)

add_library(pointcloud ${LIBRARY_TYPE} ${libSrcs}) # ../dataobject)

IF(BUILD_ITOMLIBS_SHARED)

    TARGET_LINK_LIBRARIES(${target_name} ${OpenCV_LIBS} ${PCL_LIBRARIES} itomCommonLib dataobject)
ENDIF(BUILD_ITOMLIBS_SHARED)

#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclDefines.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclStructures.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclFunctions.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/impl/pclFunctionsImpl.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud/impl)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(pointcloud COPY_SOURCES COPY_DESTINATIONS)
