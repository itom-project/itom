project(PointCloud)

cmake_minimum_required(VERSION 2.8)

#OPTION(BUILD_SHARED "Build shared library." OFF)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.") 

IF (WIN32)
  SET (BUILD_SHARED OFF)
ENDIF (WIN32)

IF(BUILD_SHARED)
  SET(LIBRARY_TYPE SHARED)
ELSE(BUILD_SHARED)
  SET(LIBRARY_TYPE STATIC)
ENDIF(BUILD_SHARED)

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

find_package(OpenCV REQUIRED)
find_package(PCL 1.5.1 REQUIRED)

IF (BUILD_UNICODE)
	ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

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
    ${OpenCV_DIR}/include
)

set(libSrcs 
    ${CMAKE_CURRENT_SOURCE_DIR}/pclStructures.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pclStructures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pclFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pclFunctions.cpp
)

add_library(pointcloud ${LIBRARY_TYPE} ${libSrcs}) # ../dataobject)

#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclStructures.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclFunctions.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_OUTPUTLIBRARY_TO_SDK_LIB(pointcloud COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(pointcloud COPY_SOURCES COPY_DESTINATIONS)
