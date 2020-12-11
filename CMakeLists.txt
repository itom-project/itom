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

cmake_minimum_required(VERSION 3.1)

project(itom)

message(STATUS "--------------- PROJECT itom -----------------")
 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/ItomBuildMacros.cmake)
include(cmake/ItomBuildMacrosInternal.cmake)

itom_init_cmake_policy(3.15)
itom_init_core_common_vars()

option(BUILD_UNITTEST "Build unittest for itom (including gtest)." OFF)
option(SETUP_ISSCONFIG "Set up batch for inno setup compilation." OFF)

option(ITOM_UPDATE_TRANSLATIONS "Update source translation translation/*.ts files\
    (WARNING: make clean will delete the source .ts files! Danger!)" OFF)
set(ITOM_LANGUAGES de CACHE STRING "semicolon separated list of languages that should be created\
    (en must not be given since it is the default)")
option(ITOM_UPDATE_TRANSLATIONS_REMOVE_UNUSED_STRINGS
    "If ITOM_UPDATE_TRANSLATIONS is ON, this option defines if strings,\
    which are in current *.ts files, but not in the source code, will be\
    removed (ON) from *.ts or not (OFF)." OFF)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.11.0")
    # add Debug flag if applicable
    # flag will be available in all subsequent targets and folders 
    # from this file
    # https://cmake.org/cmake/help/latest/command/add_compile_definitions.html
    add_compile_definitions($<$<CONFIG:DEBUG>:_DEBUG>)
endif()
    
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/cmake)

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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/itom_doxygen.dox.in
    ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen/itom_doxygen.dox )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/userDoc/create_doc.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/docs/userDoc/create_doc.py )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/userDoc/modify_doc.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/docs/userDoc/modify_doc.py )
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docs/pluginDoc
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/docs)

# copy the entire styles folder including subfolders to the binary dir
# However ignore files with endings svg, png, qrc, py (mainly the source
# files for Qt style, since these sources should be compiled to a rcc and 
# qss file).
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/styles
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    PATTERN "*.svg" EXCLUDE
    PATTERN "*.png" EXCLUDE
    PATTERN "*.qrc" EXCLUDE
    PATTERN "*.py" EXCLUDE
    PATTERN "*.odg" EXCLUDE)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/licenses
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

################################################################################################################
itom_find_package_qt(ON Core)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win32/itom_setup_win32.iss.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/itom_setup_win32.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/itom_setup_code.iss.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/itom_setup_win32_code.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/start_qt_deployment.bat.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/start_qt_deployment.bat )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win64/itom_setup_win64.iss.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/itom_setup_win64.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/itom_setup_code.iss.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/itom_setup_win64_code.iss )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/common/start_qt_deployment.bat.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/start_qt_deployment.bat )

if(SETUP_ISSCONFIG)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win32/compile_iss_win32.bat.in
        ${CMAKE_CURRENT_BINARY_DIR}/setup/win32/compile_iss_win32.bat )
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup/win64/compile_iss_win64.bat.in
        ${CMAKE_CURRENT_BINARY_DIR}/setup/win64/compile_iss_win64.bat )
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

# - copy files to itom SDK folder in binary directory (during CMake generate)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/common                        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/include FILES_MATCHING PATTERN "*.h")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/plot                          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/include FILES_MATCHING PATTERN "*.h")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/pluginTemplates               DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/pluginTemplates PATTERN ".svn" EXCLUDE)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake                         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK FILES_MATCHING PATTERN "Find*.cmake")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ItomBuildMacros.cmake   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/cmake)

if(MSVC)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/designerPluginLibraryVersion.rc DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/pluginLibraryVersion.rc   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docs/docsTmpl/_themes         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/docs)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docs/docsTmpl/pluginDoc       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/docs)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docs/docsTmpl/sphinxext       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/docs)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docs/userDoc/source/ext       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/docs)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/gitVersion.h.in               DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SDK/cmake)

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/itom_sdk.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/SDK/cmake/itom_sdk.cmake )
