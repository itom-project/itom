set(target_name pointcloud)
project(${target_name}) 
    
cmake_minimum_required(VERSION 3.0.2)
 
option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)
option(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.") 
option(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
 
include("../ItomBuildMacros.cmake")
INIT_ITOM_LIBRARY() #apply general project settings, CMake configurations, policies... for itom and its libraries

if(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE SHARED)
    
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    add_definitions(-DPOINTCLOUD_DLL -D_POINTCLOUD_DLL)
else(BUILD_ITOMLIBS_SHARED)
    set(LIBRARY_TYPE STATIC)
    
    #if pointcloud is static, add -fPIC as compiler flag for linux
    if(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif(UNIX)
endif(BUILD_ITOMLIBS_SHARED)
 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

find_package(OpenCV COMPONENTS core REQUIRED)
find_package(PCL 1.5.1 REQUIRED COMPONENTS common)

if(BUILD_UNICODE)
    add_definitions(-DUNICODE -D_UNICODE)
endif(BUILD_UNICODE)

if(BUILD_ITOMLIBS_SHARED)
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
endif(BUILD_ITOMLIBS_SHARED)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
endif()


include_directories( 
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
endif()

add_library(pointcloud ${LIBRARY_TYPE} ${libSrcs}) # ../dataobject)

if(BUILD_ITOMLIBS_SHARED)

    target_link_libraries(${target_name} ${OpenCV_LIBS} ${PCL_LIBRARIES} itomCommonLib dataobject)
endif(BUILD_ITOMLIBS_SHARED)

#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclVersion.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclDefines.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclStructures.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclFunctions.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/impl/pclFunctionsImpl.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud/impl)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(pointcloud COPY_SOURCES COPY_DESTINATIONS)
