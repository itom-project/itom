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

set(target_name dataobject)
project(${target_name}) 

message(STATUS "--------------- LIBRARY ${target_name} -----------------")

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_init_cmake_policy(3.12)
itom_init_core_common_vars()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add 3rd party packages
find_package(OpenCV COMPONENTS core REQUIRED)

# set some preprocessor directives
add_definitions(-DDATAOBJECT_DLL -D_DATAOBJECT_DLL)

include_directories( 
)

link_directories(
    
)

set(libSrcs
    ${CMAKE_CURRENT_SOURCE_DIR}/dataobjVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataobj.h
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
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
