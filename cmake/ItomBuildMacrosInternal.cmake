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

# - initializes settings, that are common to all core libraries and executables.
# These vars are widely used, shold be available whenever cmake is issued on individual
# itom project parts.
#
# This macro is automatically called from itom_init_plugin_library and 
# itom_init_designerplugin_library.
macro(itom_init_core_common_vars)
    set(BUILD_QTVERSION "auto" CACHE STRING "currently only Qt5 is supported. Set this value to 'auto' in order to auto-detect the correct Qt version or set it to 'Qt5' to hardly select Qt5.")
    option(BUILD_OPENMP_ENABLE "Use OpenMP parallelization if available. If TRUE, the definition USEOPENMP is set. This is only the case if OpenMP is generally available and if the build is release." ON)
    
    if(CMAKE_SIZEOF_VOID_P GREATER 4)
        option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
    else()
        option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
    endif()
    
    option(BUILD_WITH_PCL "Build itom with PointCloudLibrary support (pointCloud, polygonMesh, point...)" ON)
    set(ITOM_APP_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "base path to itom")
    set(ITOM_SDK_DIR ${CMAKE_CURRENT_BINARY_DIR}/SDK CACHE PATH "base path to itom_sdk")
    set(CMAKE_DEBUG_POSTFIX d CACHE STRING "Adds a postfix for debug-built libraries.")
    
    # Determined by try-compile from cmake 3.0.2 onwards. Not sure if it's a good idea to set this manually...
    if(BUILD_TARGET64)
        if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
            message(FATAL_ERROR "BUILD_TARGET64 is ON, but CMAKE_SIZEOF_VOID_P is unequal to 8 bytes. Maybe change the compiler.")
        endif()
    else()
        if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
            message(FATAL_ERROR "BUILD_TARGET64 is OFF, but CMAKE_SIZEOF_VOID_P is unequal to 4 bytes. Maybe change the compiler.")
        endif()
    endif()
    
    # Set a default build type if none was specified
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build.")
    # Set the possible values of build type for cmake-gui (will also influence the proposed values in the combo box of Visual Studio)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
    
    add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED) #build all core libraries as shared libraries
    
    #try to enable OpenMP (e.g. not available with VS Express)
    find_package(OpenMP QUIET)

    if(OPENMP_FOUND)
        if(BUILD_OPENMP_ENABLE)
            message(STATUS "OpenMP found and enabled for release compilation")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}" )
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_C_FLAGS}" )
            add_definitions(-DUSEOPENMP)
        else()
            message(STATUS "OpenMP found but not enabled for release compilation")
        endif()
    else()
        message(STATUS "OpenMP not found.")
    endif()
    
    #These are the overall pre-compiler directives for itom and its plugins:
    #
    #Windows:
    #  x86: WIN32, _WIN32
    #  x64: WIN32, _WIN32, WIN64, _WIN64
    #Linux:
    #       linux, Linux
    #Apple:
    #       __APPLE__
    if(CMAKE_HOST_WIN32)
        # check for EHa flag, as CMake does not set it by default we use this to determine if want
        # to add the itom compiler flags
        string(FIND ${CMAKE_CXX_FLAGS} "/EHa" _index)
        if(NOT ${_index} GREATER -1)
            # enable catching of machine exceptions, e.g. access violations
            string(REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

            # check some if some other flags are set, and if not set them
            string(FIND ${CMAKE_CXX_FLAGS} "/DWIN32" _index)
            if(NOT ${_index} GREATER -1)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32")
            endif()
            
            string(FIND ${CMAKE_CXX_FLAGS} "/D_WIN32" _index)
            if(NOT ${_index} GREATER -1)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN32")
            endif()

            # some more additional flags for win64
            if(BUILD_TARGET64)
                string(FIND ${CMAKE_CXX_FLAGS} "/DWIN64" _index)
                if(NOT ${_index} GREATER -1)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN64")
                endif()
                string(FIND ${CMAKE_CXX_FLAGS} "/D_WIN64" _index)
                if(NOT ${_index} GREATER -1)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN64")
                endif()
            endif()       
            
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "common C++ build flags" FORCE)
        endif()    
    elseif(CMAKE_HOST_APPLE)
        add_definitions(-D__APPLE__)
    elseif(CMAKE_HOST_UNIX) #this also includes apple, which is however already handled above
        add_definitions(-DLinux -Dlinux)
    endif()
    
    if(APPLE)
        set(CMAKE_OSX_ARCHITECTURES "x86_64")
    endif()
    
    if(MSVC)
        # ck 15/11/2017 changed, as adding /MP to definitions breaks cuda builds with e.g. enable_language(CUDA), i.e. better
        # msvs integration of cuda build
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "/MP")

        # set some optimization compiler flags
        # i.e.:
        #   - Ox full optimization (replaces standard O2 set by cmake)
        #   - Oi enable intrinsic functions
        #   - Ot favor fast code
        #   - Oy omit frame pointers
        #   - GL whole program optimization
        #   - GT fibre safe optimization
        #   - openmp enable openmp support, isn't enabled globally here as it breaks opencv
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL" )
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL" )
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
        
        # add /LTCG flag to remove MSVC linker warning in release build
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
        set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
        
        if(NOT BUILD_TARGET64)
            #Disable safe SEH for Visual Studio, 32bit (this is necessary since some 3rd party libraries are compiled without /SAFESEH)
            set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
            set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
            set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
            set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
            set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
            set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
            set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
            set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
            set(CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL "${CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL} /SAFESEH:NO")
            set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /SAFESEH:NO")
            set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} /SAFESEH:NO")
        endif()
    endif()
    
    # Begin: Remove duplicates compilation flags
    separate_arguments(CMAKE_CXX_FLAGS)
    list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS)
    string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    separate_arguments(CMAKE_CXX_FLAGS_RELEASE)
    list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS_RELEASE)
    string(REPLACE ";" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    separate_arguments(CMAKE_C_FLAGS_RELEASE)
    list(REMOVE_DUPLICATES CMAKE_C_FLAGS_RELEASE)
    string(REPLACE ";" " " CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    # End: Remove duplicates compilation flags
endmacro()

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
