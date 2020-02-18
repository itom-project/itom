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

cmake_minimum_required(VERSION 3.1)

set(target_name pointcloud)
project(${target_name}) 

message(STATUS "--------------- LIBRARY ${target_name} -----------------")

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_init_cmake_policy(3.12)
itom_init_core_common_vars()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#pre-create the necessary CMake variables for PCL to simplify the creation in the CMake GUI
set(Boost_INCLUDE_DIR "" CACHE PATH "include directory of Boost (Windows: usually the root directory of Boost)")
set(Boost_LIBRARY_DIR "" CACHE PATH "library directory of Boost (Windows: usually a lib subdirectory of Boost)")
set(EIGEN_INCLUDE_DIR "" CACHE PATH "base directory of header-only library Eigen")

# add 3rd party packages
find_package(OpenCV COMPONENTS core REQUIRED)
find_package(PCL 1.5.1 REQUIRED COMPONENTS common)

# set some preprocessor directives
add_definitions(-DPOINTCLOUD_DLL -D_POINTCLOUD_DLL)

include_directories( 
    ${PCL_INCLUDE_DIRS}
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
