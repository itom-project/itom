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

set(target_name qpropertyeditor)

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
itom_find_package_qt(ON Core PrintSupport Widgets Designer Xml Svg UiTools)
find_package(OpenCV COMPONENTS core REQUIRED)

# set some preprocessor directives
add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
add_definitions(-DQPROPERTYEDITOR_DLL -D_QPROPERTYEDITOR_DLL)
add_definitions(-DUNICODE -D_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif()

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ITOM_SDK_INCLUDE_DIRS}
    )


set(qpropertyeditor_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ColorCombo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/BooleanCombo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FontEditor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/EnumProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FlagsProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Property.h
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditorWidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/QVariantDelegate.h
    ${CMAKE_CURRENT_SOURCE_DIR}/itomCustomTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qVector2DProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qVector3DProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qVector4DProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/autoIntervalProperty.h
    ${CMAKE_CURRENT_SOURCE_DIR}/stringListEditor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/stringListDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/propertyEditorVersion.h
)

set(qpropertyeditor_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/ColorCombo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BooleanCombo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FontEditor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EnumProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FlagsProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Property.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditorWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QVariantDelegate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itomCustomTypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qVector2DProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qVector3DProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qVector4DProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/autoIntervalProperty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stringListEditor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stringListDialog.cpp
)

set(qpropertyeditor_UI
    ${CMAKE_CURRENT_SOURCE_DIR}/stringListDialog.ui
)

set(qpropertyeditor_RCC
    #add absolute pathes to any *.qrc resource files here
)

if(QT5_FOUND)
    QT5_WRAP_UI(qpropertyeditor_UI_MOC ${qpropertyeditor_UI})
    QT5_ADD_RESOURCES(qpropertyeditor_RCC_MOC ${qpropertyeditor_RCC})
else()
    message(SEND_ERROR "Currently only Qt5 is supported")
endif()

if(MSVC)
    list(APPEND qpropertyeditor_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif()

add_library(${target_name} SHARED ${qpropertyeditor_SOURCES} ${qpropertyeditor_HEADERS} ${qproperteditor_HEADERS_MOC} ${qpropertyeditor_UI_MOC} ${qpropertyeditor_RCC_MOC})

target_link_libraries(${target_name} ${QT5_LIBRARIES} itomCommonLib itomWidgets)


#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
set(SOURCEFILE ${CMAKE_CURRENT_SOURCE_DIR}/../QPropertyEditor)
file(COPY ${SOURCEFILE}/defines.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
file(COPY ${SOURCEFILE}/Property.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
file(COPY ${SOURCEFILE}/QPropertyEditorWidget.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)
file(COPY ${SOURCEFILE}/QVariantDelegate.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/QPropertyEditor)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)