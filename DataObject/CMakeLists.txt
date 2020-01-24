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

set(target_name dataobject)
project(${target_name}) 

cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")
message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

# add 3rd party packages
find_package(OpenCV COMPONENTS core REQUIRED)

# set some preprocessor directives
add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
add_definitions(-DDATAOBJECT_DLL -D_DATAOBJECT_DLL)
add_definitions(-DUNICODE -D_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif()

include_directories( 
    ${OpenCV_DIR}/include
)

link_directories(
    ${OpenCV_DIR}/lib
)

set(libSrcs
    ${CMAKE_CURRENT_SOURCE_DIR}/dataobjVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataobj.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataobj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectFuncs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectFuncs.cpp
)

if(MSVC)
    list(APPEND libSrcs ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

add_library(${target_name} SHARED ${libSrcs})

target_link_libraries(${target_name} ${OpenCV_LIBS} itomCommonLib)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/defines.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/dataobj.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/dataObjectFuncs.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/dataobjVersion.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)

#readWriteLock.h is not part of the SDK anymore
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject/readWriteLock.h)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject/readWriteLock.h)
endif()

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_post_build_copy_files(dataobject COPY_SOURCES COPY_DESTINATIONS)
