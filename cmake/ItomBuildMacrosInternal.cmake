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

# - appends two entries to sources and destinations in order to copy
# the linker file of the given target both to the root directory of itom (ITOM_APP_DIR)
# as well as to the SDK directory of itom (ITOM_SDK_DIR).
# 
# The copy operation itself is only created if the modified lists 'sources'
# and 'destinations' are passed to the macro 'itom_post_build_copy_files'.
#
# example:
# set(COPY_SOURCES "")
# set(COPY_DESTINATIONS "")
# itom_add_library_to_appdir_and_sdk(${target_name} COPY_SOURCES COPY_DESTINATIONS)
# itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)
# .
macro(itom_add_library_to_appdir_and_sdk target sources destinations)
    
    if(${ITOM_SDK_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_SDK_DIR is not indicated")
    endif()
    
    if(${ITOM_APP_DIR} STREQUAL "")
        message(SEND_ERROR "ITOM_APP_DIR is not indicated")
    endif()
    
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(SDK_PLATFORM "x86")
    else()
        set(SDK_PLATFORM "x64")
    endif()
    
    # The following list has to be consistent with FindITOM_SDK.cmake! 
    # From VS higher than 1900, the default case vc${MSVC_VERSION} is used.
    if(MSVC_VERSION EQUAL 1900)
        set(SDK_COMPILER "vc14")
    elseif(MSVC_VERSION EQUAL 1800)
        set(SDK_COMPILER "vc12")
    elseif(MSVC_VERSION EQUAL 1700)
        set(SDK_COMPILER "vc11")
    elseif(MSVC_VERSION EQUAL 1600)
        set(SDK_COMPILER "vc10")
    elseif(MSVC_VERSION EQUAL 1500)
        set(SDK_COMPILER "vc9")
    elseif(MSVC_VERSION EQUAL 1400)
        set(SDK_COMPILER "vc8")
    elseif(MSVC)
        set(SDK_COMPILER "vc${MSVC_VERSION}")
    elseif(CMAKE_COMPILER_IS_GNUCXX)
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SDK_COMPILER "clang")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SDK_COMPILER "gnucxx")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(SDK_COMPILER "intel")
    elseif(APPLE)
        set(SDK_COMPILER "osx_default")
    else()
        set(SDK_COMPILER "unknown")
    endif()
    
    set(SDK_DESTINATION "${ITOM_SDK_DIR}/lib/${SDK_COMPILER}_${SDK_PLATFORM}" )
    
    #copy library (dll) to app-directory and linker library (lib) to sdk_destination
    list(APPEND ${sources} "$<TARGET_LINKER_FILE:${target}>")
    list(APPEND ${destinations} ${SDK_DESTINATION})

    list(APPEND ${sources} "$<TARGET_FILE:${target}>")
    list(APPEND ${destinations} ${ITOM_APP_DIR})
endmacro()

# The following macros are for OSX (Mac OS) only.
if(APPLE)
    # Copy files from source directory to destination directory in app bundle, substituting any
    # variables (RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS.
    macro(itom_copy_to_bundle target srcDir destDir)
        file(GLOB_RECURSE templateFiles RELATIVE ${srcDir} ${srcDir}/*)
        foreach(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            if(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            endif(NOT IS_DIRECTORY ${srcTemplatePath})
        endforeach(templateFile)
    endmacro()
    
    # OSX ONLY: Copy files of certain type from source directory to destination directory in app bundle, substituting any
    # variables (NON-RECURSIVE). Create destination directory if it does not exist. destDir append ../abc.app/MacOS
    macro(itom_copy_type_to_bundle_nonrec target srcDir destDir type)
        file(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*${type})
        foreach(templateFile ${templateFiles})
            set(srcTemplatePath ${srcDir}/${templateFile})
            if(NOT IS_DIRECTORY ${srcTemplatePath})
                add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${srcTemplatePath}" "$<TARGET_FILE_DIR:${target_name}>/${destDir}/${templateFile}")
            endif(NOT IS_DIRECTORY ${srcTemplatePath})
        endforeach(templateFile)
    endmacro()
endif(APPLE)