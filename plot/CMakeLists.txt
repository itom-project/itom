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

set(target_name itomCommonPlotLib) 
project(${target_name}) 

message(STATUS "--------------- LIBRARY ${target_name} -----------------")

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_init_cmake_policy(3.12)
itom_init_core_common_vars()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add 3rd party packages
itom_find_package_qt(ON Core PrintSupport Designer Xml Svg Widgets UiTools LinguistTools)
find_package(OpenCV COMPONENTS core REQUIRED)

# set some preprocessor directives
add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
add_definitions(-DITOMCOMMONPLOT_DLL -D_ITOMCOMMONPLOT_DLL)
add_definitions(-DUNICODE -D_UNICODE)
add_definitions(-DITOMCOMMONPLOT_MOC)

if(BUILD_WITH_PCL)
    find_package(PCL 1.5.1 REQUIRED COMPONENTS common)
    add_definitions(-DUSEPCL -D_USEPCL)
endif(BUILD_WITH_PCL)

if(BUILD_WITH_PCL)
    set(POINTCLOUD_LIBRARIES pointcloud)
else()
    set(POINTCLOUD_LIBRARIES "")
endif()

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/../shapes
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PCL_INCLUDE_DIRS}
)

link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    ${CMAKE_CURRENT_BINARY_DIR}/../PointCloud
    
)

set(itomCommonPlot_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/plotVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plotCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AbstractDObjFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AbstractDObjPCLFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AbstractFigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AbstractItomDesignerPlugin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AbstractNode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/designerPluginInterfaceVersion.h
)

set(itomCommonPlot_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/AbstractDObjFigure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/AbstractDObjPCLFigure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/AbstractFigure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/AbstractNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/Channel.cpp
)

if(MSVC)
    list(APPEND itomCommonQt_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

add_library(${target_name} SHARED ${itomCommonPlot_SOURCES} ${itomCommonPlot_HEADERS})

target_link_libraries(${target_name} ${OpenCV_LIBS} ${POINTCLOUD_LIBRARIES} ${QT5_LIBRARIES} dataobject itomCommonLib itomShapeLib qpropertyeditor itomCommonQtLib)

# Qt: enable all automoc, autouic and autorcc. Autouic file will be disabled for all files that are processed manually by itom_qt_wrap_ui.
set_target_properties(${target_name} PROPERTIES AUTOMOC ON AUTORCC ON AUTOUIC ON)

#translation
itom_library_translation(QM_FILES TARGET ${target_name} FILES_TO_TRANSLATE ${itomCommonPlot_SOURCES} ${itomCommonPlot_HEADERS})

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")

#copy all qm files to the translations subfolder of the itom root directory
foreach(_qm ${QM_FILES})
    set(COPY_SOURCES ${COPY_SOURCES} ${_qm})
    set(COPY_DESTINATIONS ${COPY_DESTINATIONS} "${CMAKE_CURRENT_BINARY_DIR}/../translation")
endforeach()

itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)