# - itom software
# URL: http://www.uni-stuttgart.de/ito
# Copyright (C) 2020, Institut fuer Technische Optik (ITO),
# Universitaet Stuttgart, Germany
#
# This file is part of itom and its software development toolkit (SDK).
#
# itom is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public Licence as published by
# the Free Software Foundation; either version 2 of the Licence, or (at
# your option) any later version.
#
# In addition, as a special exception, the Institut fuer Technische
# Optik (ITO) gives you certain additional rights.
# These rights are described in the ITO LGPL Exception version 1.0,
# which can be found in the file LGPL_EXCEPTION.txt in this package.
#
# itom is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
# General Public Licence for more details.
#
# You should have received a copy of the GNU Library General Public License
# along with itom. If not, see <http://www.gnu.org/licenses/>.

set(target_name pointcloud)
project(${target_name}) 

cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)
set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.") 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

# add 3rd party packages
find_package(OpenCV COMPONENTS core REQUIRED)
find_package(PCL 1.5.1 REQUIRED COMPONENTS common)

# set some preprocessor directives
add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
add_definitions(-DPOINTCLOUD_DLL -D_POINTCLOUD_DLL)
add_definitions(-DUNICODE -D_UNICODE)

include_directories( 
    ${PCL_INCLUDE_DIRS}
    #${Boost_INCLUDE_DIR}
    #${EIGEN_INCLUDE_DIRS}
    #${FLANN_INCLUDE_DIRS}
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

add_library(${target_name} SHARED ${libSrcs})

target_link_libraries(${target_name} ${OpenCV_LIBS} ${PCL_LIBRARIES} itomCommonLib dataobject)

#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclVersion.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclDefines.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclStructures.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/pclFunctions.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../PointCloud/impl/pclFunctionsImpl.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/PointCloud/impl)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)
