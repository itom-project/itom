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
# itom is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
# General Public Licence for more details.
#
# You should have received a copy of the GNU Library General Public License
# along with itom. If not, see <http://www.gnu.org/licenses/>.

project(itom)
 
cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64")
endif()

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)
option(BUILD_WITH_PCL "Build itom with PointCloudLibrary support (pointCloud, polygonMesh, point...)" ON)
set(ITOM_APP_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "base path to itom")
set(ITOM_SDK_DIR ${CMAKE_CURRENT_BINARY_DIR}/SDK CACHE PATH "base path to itom_sdk")
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
option(BUILD_UNITTEST "Build unittest for itom (including gtest)." OFF)
option(SETUP_ISSCONFIG "Set up batch for inno setup compilation." OFF)

# Determined by try-compile from cmake 3.0.2 onwards. Not sure if it's a good idea to set this manually...
if(BUILD_TARGET64)
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "BUILD_TARGET64 is ON, but CMAKE_SIZEOF_VOID_P is unequal to 8 bytes. Maybe change the compiler.")
    endif()
else()
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
        message(FATAL_ERROR "BUILD_TARGET64 is OFF, but CMAKE_SIZEOF_VOID_P is unequal to 4 bytes. Maybe change the compiler.")
    endif()
endif(BUILD_TARGET64)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/cmake)

include("cmake/ItomBuildMacros.cmake")
include("cmake/ItomBuildMacrosInternal.cmake")

# Set a default build type if none was specified
itom_set_default_build_type()

if(WIN32)
    # add /LTCG flag to remove MSVC linker warning in release build
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
endif()

add_subdirectory(DataObject)

if(BUILD_WITH_PCL)
    add_subdirectory(PointCloud)
    set(ITOM_SDK_LIB_COMPONENTS_STR "dataobject pointcloud addinmanager qpropertyeditor itomCommonLib itomCommonQtLib itomCommonPlotLib itomWidgets itomShapeLib")
else()
    set(ITOM_SDK_LIB_COMPONENTS_STR "dataobject addinmanager qpropertyeditor itomCommonLib itomCommonQtLib itomCommonPlotLib itomWidgets itomShapeLib")
endif()

add_subdirectory(common)
add_subdirectory(AddInManager)
add_subdirectory(shape)
add_subdirectory(commonQt)
add_subdirectory(QPropertyEditor)
add_subdirectory(plot)
add_subdirectory(itomWidgets)
add_subdirectory(Qitom)
add_subdirectory(iconThemes)

if(BUILD_UNITTEST)
    #put this as last, since gtest is globally changing many compiler flags
    add_subdirectory(gtest-1.6.0)
    add_subdirectory(itom_unittests)
endif()

if(BUILD_ISSCONFIG)
    set(InnoSetupPath CACHE PATH "base path to inno setup")
endif()

#find current itom version from qitom/global.h
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/Qitom/global.h ITOM_versionStringFull LIMIT_COUNT 1 REGEX "^#define ITOM_VERSION_STR.*$")
string(REGEX MATCH "([0-9]+).([0-9]+).([0-9]+)" ITOM_versionString ${ITOM_versionStringFull}) #searches for Number.Number.Number[.Number] in ITOM_versionStringFull

message(STATUS "itom version is: ${ITOM_versionString}. If wrong change it in global.h and reconfigure CMake")
message(STATUS "")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "")

################################################################################################################
#CONFIGURE AND PREPARE THE INNO SETUP FILES (require location of Qt, therefore find_package_qt must be called)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/itom_doxygen.dox.in ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen/itom_doxygen.dox )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/userDoc/create_doc.py.in ${CMAKE_CURRENT_BINARY_DIR}/docs/userDoc/create_doc.py )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/userDoc/modify_doc.py.in ${CMAKE_CURRENT_BINARY_DIR}/docs/userDoc/modify_doc.py )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/pythonSqlDb/PythonAPI_SqlCreator.py ${CMAKE_CURRENT_BINARY_DIR}/docs/pythonSqlDb/PythonAPI_SqlCreator.py )
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docs/pluginDoc DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/docs)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/styles DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/licenses DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

################################################################################################################
itom_find_package_qt(ON Core)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win32/itom_setup_win32.iss.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/itom_setup_win32.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/itom_setup_code.iss.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/itom_setup_win32_code.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/start_qt_deployment.bat.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/start_qt_deployment.bat )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win64/itom_setup_win64.iss.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/itom_setup_win64.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/itom_setup_code.iss.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/itom_setup_win64_code.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/start_qt_deployment.bat.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/start_qt_deployment.bat )

if(SETUP_ISSCONFIG)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win32/compile_iss_win32.bat.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/compile_iss_win32.bat )
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win64/compile_iss_win64.bat.in ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/compile_iss_win64.bat )
endif()
################################################################################################################

if(WIN32)
    set(CHMSource "${CMAKE_CURRENT_SOURCE_DIR}/docs/itom_doc.chm")
    set(CHMDest "${CMAKE_CURRENT_BINARY_DIR}/docs")

    if(EXISTS ${CHMSource})
        if(EXISTS "${CHMDest}/itom_doc.chm")
            if(${CHMSource} IS_NEWER_THAN "${CHMDest}/itom_doc.chm")
                message(STATUS "copy ${CHMSource} to ${CHMDest} since it is newer")
                file(COPY ${CHMSource} DESTINATION ${CHMDest})
            endif()
        else()
            message(STATUS "copy ${CHMSource} to ${CHMDest}")
            file(COPY ${CHMSource} DESTINATION ${CHMDest})
        endif()
    else()
        message(STATUS "${CHMSource} does not exist")
    endif()

    if(MSVC)
        
        if(CMAKE_CL_64)
            set(MSVC_BITNESS1 x64)
            set(MSVC_BITNESS2 x64)
        else()
            set(MSVC_BITNESS1 x86)
            set(MSVC_BITNESS2 win32)
        endif()
    endif()
    if(MSVC_VERSION EQUAL 1600)
        set(MSVC_VSCOMNTOOLS "%VS100COMNTOOLS%")
        set(MSVC_COMPILE_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/msvc/compile_100to140.bat.in")
    elseif(MSVC_VERSION EQUAL 1700)
        set(MSVC_VSCOMNTOOLS "%VS110COMNTOOLS%")
        set(MSVC_COMPILE_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/msvc/compile_100to140.bat.in")
    elseif(MSVC_VERSION EQUAL 1800)
        set(MSVC_VSCOMNTOOLS "%VS120COMNTOOLS%")
        set(MSVC_COMPILE_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/msvc/compile_100to140.bat.in")
    elseif(MSVC_VERSION EQUAL 1900)
        set(MSVC_VSCOMNTOOLS "%VS140COMNTOOLS%")
        set(MSVC_COMPILE_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/msvc/compile_100to140.bat.in")
    elseif((MSVC_VERSION GREATER 1909) AND (MSVC_VERSION LESS 1920))
        set(MSVC_VSCOMNTOOLS "")
        set(MSVC_COMPILE_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/msvc/compile_150toXXX.bat.in")
    else()
        set(MSVC_COMPILE_TMPL "")
    endif()
    
    if(MSVC_COMPILE_TMPL)
        set(MSVC_REBUILD_TAG "")
        set(MSVC_CONFIGURATION "debug")
        configure_file("${MSVC_COMPILE_TMPL}" "${CMAKE_CURRENT_BINARY_DIR}/compile_debug.bat" @ONLY)
        set(MSVC_CONFIGURATION "release")
        configure_file("${MSVC_COMPILE_TMPL}" "${CMAKE_CURRENT_BINARY_DIR}/compile_release.bat" @ONLY)
        
        set(MSVC_REBUILD_TAG "/t:rebuild")
        set(MSVC_CONFIGURATION "debug")
        configure_file("${MSVC_COMPILE_TMPL}" "${CMAKE_CURRENT_BINARY_DIR}/compile_debug_rebuild.bat" @ONLY)
        set(MSVC_CONFIGURATION "release")
        configure_file("${MSVC_COMPILE_TMPL}" "${CMAKE_CURRENT_BINARY_DIR}/compile_release_rebuild.bat" @ONLY)
    endif()
endif()

#ADD a batch file for itom-release
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/COPYING.txt ${CMAKE_CURRENT_BINARY_DIR}/COPYING.txt )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/itom_sdk.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/SDK/cmake/itom_sdk.cmake )
