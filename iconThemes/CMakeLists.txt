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

cmake_minimum_required(VERSION 3.1...3.15)
 
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
  
include(CMakeParseArguments)

# _itom_qt5_parse_qrc_file(infile _out_depends _rc_depends)
# internal
function(_ITOM_QT5_PARSE_QRC_FILE infile _out_depends _rc_depends)
    get_filename_component(rc_path ${infile} PATH)

    if(EXISTS "${infile}")
        #  parse file for dependencies
        #  all files are absolute paths or relative to the location of the qrc file
        file(READ "${infile}" RC_FILE_CONTENTS)
        string(REGEX MATCHALL "<file[^<]+" RC_FILES "${RC_FILE_CONTENTS}")
        foreach(RC_FILE ${RC_FILES})
            string(REGEX REPLACE "^<file[^>]*>" "" RC_FILE "${RC_FILE}")
            if(NOT IS_ABSOLUTE "${RC_FILE}")
                set(RC_FILE "${rc_path}/${RC_FILE}")
            endif()
            set(RC_DEPENDS ${RC_DEPENDS} "${RC_FILE}")
        endforeach()
        # Since this cmake macro is doing the dependency scanning for these files,
        # let's make a configured file and add it as a dependency so cmake is run
        # again when dependencies need to be recomputed.
        qt5_make_output_file("${infile}" "" "qrc.depends" out_depends)
        configure_file("${infile}" "${out_depends}" COPYONLY)
    else()
        # The .qrc file does not exist (yet). Let's add a dependency and hope
        # that it will be generated later
        set(out_depends)
    endif()

    set(${_out_depends} ${out_depends} PARENT_SCOPE)
    set(${_rc_depends} ${RC_DEPENDS} PARENT_SCOPE)
endfunction()


# itom_qt5_add_binary_resources(target inputfiles ... )
# Since Qt 5.5.0 this macro is part of Qt5 Core. It is copied here for backward compatibility.
function(itom_qt5_add_binary_resources target )

    set(options)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs OPTIONS)

    cmake_parse_arguments(_RCC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(rcc_files ${_RCC_UNPARSED_ARGUMENTS})
    set(rcc_options ${_RCC_OPTIONS})
    set(rcc_destination ${_RCC_DESTINATION})

    if(NOT rcc_destination)
        set(rcc_destination ${CMAKE_CURRENT_BINARY_DIR}/${target}.rcc)
    endif()

    foreach(it ${rcc_files})
        get_filename_component(infile ${it} ABSOLUTE)

        _ITOM_QT5_PARSE_QRC_file(${infile} _out_depends _rc_depends)
        set(infiles ${infiles} ${infile})
        set(out_depends ${out_depends} ${_out_depends})
        set(rc_depends ${rc_depends} ${_rc_depends})
    endforeach()

    add_custom_command(OUTPUT ${rcc_destination}
                       COMMAND ${Qt5Core_RCC_EXECUTABLE}
                       ARGS ${rcc_options} --binary --name ${target} --output ${rcc_destination} ${infiles}
                       DEPENDS ${rc_depends} ${out_depends} VERBATIM)

    add_custom_target(${target} ALL DEPENDS ${rcc_destination})
endfunction()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# include commonly used itom CMake macros (internal can only be used within the core projects) 
include("../cmake/ItomBuildMacros.cmake")
include("../cmake/ItomBuildMacrosInternal.cmake")

itom_find_package_qt(ON Core Widgets)

add_definitions(-DUNICODE -D_UNICODE)

if(QT5_FOUND)
    if(${Qt5_VERSION} VERSION_LESS 5.5.0)
      itom_qt5_add_binary_resources(iconThemeBright ${CMAKE_CURRENT_SOURCE_DIR}/../Qitom/iconThemeBright.qrc DESTINATION ${ITOM_APP_DIR}/iconThemeBright.rcc)
      itom_qt5_add_binary_resources(iconThemeDark ${CMAKE_CURRENT_SOURCE_DIR}/../Qitom/iconThemeDark.qrc DESTINATION ${ITOM_APP_DIR}/iconThemeDark.rcc)
    else()
      qt5_add_binary_resources(iconThemeBright ${CMAKE_CURRENT_SOURCE_DIR}/../Qitom/iconThemeBright.qrc DESTINATION ${ITOM_APP_DIR}/iconThemeBright.rcc)
      qt5_add_binary_resources(iconThemeDark ${CMAKE_CURRENT_SOURCE_DIR}/../Qitom/iconThemeDark.qrc DESTINATION ${ITOM_APP_DIR}/iconThemeDark.rcc)
    endif()
endif() #for Qt4, the bright theme is directly included in the itom application (no choice available) 
   