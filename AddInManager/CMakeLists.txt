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

set(target_name addinmanager)
project(${target_name}) 

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_init_cmake_policy(3.12)
itom_init_core_common_vars()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)

# add 3rd party packages
itom_find_package_qt(ON Core Widgets Xml Designer Svg PrintSupport LinguistTools)
find_package(OpenCV COMPONENTS core REQUIRED)

# set some preprocessor directives
add_definitions(-DADDINMGR_DLL -D_ADDINMGR_DLL)

if(BUILD_WITH_PCL)
    find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
    add_definitions(-DUSEPCL)
endif()

include_directories( 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${VISUALLEAKDETECTOR_INCLUDE_DIR}
    ../common
    ../DataObject
    ../PointCloud
)

set(lib_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/addInMgrDefines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManagerVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramHelper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/algoInterfaceValidator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/apiFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManagerPrivate.h
)

set(lib_SOURCES

    ${CMAKE_CURRENT_SOURCE_DIR}/pluginModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/algoInterfaceValidator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/apiFunctions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/addInManagerPrivate.cpp
)

if(MSVC)
    list(APPEND lib_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

add_library(${target_name} SHARED ${lib_SOURCES} ${lib_HEADERS})

target_link_libraries(${target_name} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${QT5_LIBRARIES} ${VISUALLEAKDETECTOR_LIBRARIES} dataobject itomCommonLib itomCommonQtLib)

# Qt: enable all automoc, autouic and autorcc. Autouic file will be disabled for all files that are processed manually by itom_qt_wrap_ui.
set_target_properties(${target_name} PROPERTIES AUTOMOC ON AUTORCC ON AUTOUIC ON)

#translation
itom_library_translation(QM_FILES TARGET ${target_name} FILES_TO_TRANSLATE ${lib_SOURCES} ${lib_HEADERS})

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")

#copy all qm files to the translations subfolder of the itom root directory
foreach(_qm ${QM_FILES})
    set(COPY_SOURCES ${COPY_SOURCES} ${_qm})
    set(COPY_DESTINATIONS ${COPY_DESTINATIONS} "${CMAKE_CURRENT_BINARY_DIR}/../translation")
endforeach()

itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_post_build_copy_files(addinmanager COPY_SOURCES COPY_DESTINATIONS)
