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

set(target_name itomShapeLib)
project(${target_name})

message(STATUS "--------------- LIBRARY ${target_name} -----------------")

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_init_cmake_policy(3.12)
itom_init_core_common_vars()

option(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add 3rd party packages
itom_find_package_qt(ON Core Designer LinguistTools)
find_package(OpenCV COMPONENTS core imgproc REQUIRED)

# set some preprocessor directives
add_definitions(-DITOMSHAPE_DLL -D_ITOMSHAPE_DLL)
add_definitions(-DITOMSHAPE_MOC)

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

link_directories(
    
)

set(itomShape_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shapeCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/shapeVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/shape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/shapeDObject.h
)

set(itomShape_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/sources/shapeDObject.cpp
)

if(MSVC)
    list(APPEND itomShape_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

add_library(${target_name} SHARED ${itomShape_SOURCES} ${itomShape_HEADERS} ${itomShape_HEADERS_MOC} ${EXISTING_TRANSLATION_FILES})

target_link_libraries(${target_name} ${OpenCV_LIBS} ${QT5_LIBRARIES} dataobject itomCommonLib)

#translation
file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
set(FILES_TO_TRANSLATE ${itomShape_SOURCES} ${itomShape_HEADERS})
itom_library_translation(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/translation)

# COPY SECTION
set(COPY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/translation/itomShapeLib_de.qm")
set(COPY_DESTINATIONS "${CMAKE_CURRENT_BINARY_DIR}/../translation")
itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)

foreach(headerFile ${itomShape_HEADERS}) 
    file(COPY ${headerFile}    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/shape FILES_MATCHING PATTERN "*.h")
endforeach(headerFile)