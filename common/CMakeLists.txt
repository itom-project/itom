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

set(target_name itomCommonLib)  
project(${target_name}) 

cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
set(ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

# set some preprocessor directives
add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
add_definitions(-DITOMCOMMON_DLL -D_ITOMCOMMON_DLL)
add_definitions(-DUNICODE -D_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif()

set(ITOMCOMMON_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/commonVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/retVal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/typeDefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/numeric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/byteArray.h
    ${CMAKE_CURRENT_SOURCE_DIR}/param.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramMeta.h
    ${CMAKE_CURRENT_SOURCE_DIR}/color.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helperColor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interval.h
    ${CMAKE_CURRENT_SOURCE_DIR}/itomPlotHandle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sharedStructures.h
)

set(ITOMCOMMON_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/retVal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/byteArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/param.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/paramMeta.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/interval.cpp
)

if(MSVC)
    list(APPEND ITOMCOMMON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

add_library(${target_name} SHARED ${ITOMCOMMON_SOURCES} ${ITOMCOMMON_HEADERS})

target_link_libraries(${target_name})

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)
