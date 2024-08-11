# - Find libusb for portable USB support
# This module will find libusb as published by
#  http://libusb.sf.net and
#  http://libusb-win32.sf.net
#  http://libusb-win32.sf.net
#
# It will use PkgConfig if present and supported, else search
# it on its own. If the **LIBUSB_ROOT** environment variable
# is defined, it will be used as base path.
# The following standard variables get defined:
#  LibUSB_FOUND:        true if LibUSB was found
#  LibUSB_INCLUDE_DIRS: the directory that contains the include file
#  LibUSB_LIBRARIES:    the library

include ( CheckLibraryExists )
include ( CheckIncludeFile )

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)

find_package ( PkgConfig QUIET)
if( PKG_CONFIG_FOUND )
  pkg_check_modules ( PKGCONFIG_LIBUSB libusb-1.0>=1.0 )
endif( PKG_CONFIG_FOUND )

message(STATUS "PkgConfig: " ${PkgConfig})

if( PKGCONFIG_LIBUSB_FOUND )
    set( LibUSB_FOUND ${PKGCONFIG_LIBUSB_FOUND} )
    set( LibUSB_INCLUDE_DIRS ${PKGCONFIG_LIBUSB_INCLUDEDIR}/libusb-1.0 )
    foreach ( i ${PKGCONFIG_LIBUSB_LIBRARIES} )
    find_library ( ${i}_LIBRARY
      NAMES ${i}
      PATHS ${PKGCONFIG_LIBUSB_LIBDIR}
    )
    if( ${i}_LIBRARY )
        list ( APPEND LibUSB_LIBRARIES ${${i}_LIBRARY} )
    endif( ${i}_LIBRARY )
    mark_as_advanced ( ${i}_LIBRARY )
    endforeach ( i )

else()

    find_path(LibUSB_INCLUDE_DIRS
    NAMES
      libusb.h
    PATHS
      $ENV{ProgramFiles}/LibUSB-Win32
      $ENV{LIBUSB_ROOT}/libusb-cygwin-x64
      ${LibUSB_DIR}
    PATH_SUFFIXES
      libusb
      libusb-1.0
      include
      include/libusb-1.0
    DOC "root directory of LibUSB"
    )

    mark_as_advanced ( LibUSB_INCLUDE_DIRS )
    #  message( STATUS "LibUSB include dir: ${LIBUSB_ROOT}" )

    if(MSVC)

        if((MSVC_VERSION GREATER 1929) AND (MSVC_VERSION LESS_EQUAL 1940))
            set(LIBUSB_DETECTED_TOOLSET "vc143")
            set(LIBUSB_MSVC_FOLDER "VS2022")

        elseif((MSVC_VERSION GREATER 1919) AND (MSVC_VERSION LESS_EQUAL 1930))
            set(LIBUSB_DETECTED_TOOLSET "vc142")
            set(LIBUSB_MSVC_FOLDER "VS2019")

        elseif((MSVC_VERSION GREATER 1909) AND (MSVC_VERSION LESS_EQUAL 1920))
            set(LIBUSB_DETECTED_TOOLSET "vc141")
            set(LIBUSB_MSVC_FOLDER "VS2017")

        elseif(MSVC_VERSION EQUAL 1900)
            set(LIBUSB_DETECTED_TOOLSET "vc140")
            set(LIBUSB_MSVC_FOLDER "VS2015")

        elseif(MSVC_VERSION EQUAL 1800)
            set(LIBUSB_DETECTED_TOOLSET "vc120")
            set(LIBUSB_MSVC_FOLDER "VS2015")

        endif()

        message(STATUS "LIBUSB_MSVC_FOLDER: ${LIBUSB_MSVC_FOLDER}")

    endif(MSVC)

    if( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
        # LibUSB-Win32 binary distribution contains several libs.
        # Use the lib that got compiled with the same compiler.
        if( MSVC )
            if(BUILD_TARGET64)
                set( LibUSB_LIBRARY_PATH_SUFFIX
                    MS64/static
                    MS64/dll
                    ${LIBUSB_MSVC_FOLDER}/MS64/dll
                    ${LIBUSB_MSVC_FOLDER}/MS64/dll
                    MS64/static
                    MS64/dll
                    x64
                    x64/Release
                    x64/Debug
                    x64/Release/lib
                    x64/Debug/lib
                    x64/Release/dll
                    x64/Debug/dll
					VS2015-x64/lib
					VS2015-x64/dll
                    build/v141/x64/Debug/lib
                    build/v141/x64/Debug/dll
                    build/v141/x64/Release/lib
                    build/v141/x64/Release/dll
                    build/v142/x64/Debug/lib
                    build/v142/x64/Debug/dll
                    build/v142/x64/Release/lib
                    build/v142/x64/Release/dll
                    build/v143/x64/Debug/lib
                    build/v143/x64/Debug/dll
                    build/v143/x64/Release/lib
                    build/v143/x64/Release/dll
					)
            else (BUILD_TARGET64)
                set( LibUSB_LIBRARY_PATH_SUFFIX
                    MS32/static
                    MS32/dll
                    ${LIBUSB_MSVC_FOLDER}/MS32/dll
                    ${LIBUSB_MSVC_FOLDER}/MS32/dll
                    Win32
                    Win32/Release
                    Win32/Debug
                    Win32/Release/lib
                    Win32/Debug/lib
                    Win32/Release/dll
                    Win32/Debug/dll
					VS2015-Win32/lib
					VS2015-Win32/dll
                    build/v141/Win32/Debug/lib
                    build/v141/Win32/Debug/dll
                    build/v141/Win32/Release/lib
                    build/v141/Win32/Release/dll
                    build/v142/Win32/Debug/lib
                    build/v142/Win32/Debug/dll
                    build/v142/Win32/Release/lib
                    build/v142/Win32/Release/dll
                    build/v143/Win32/Debug/lib
                    build/v143/Win32/Debug/dll
                    build/v143/Win32/Release/lib
                    build/v143/Win32/Release/dll
					)
            endif(BUILD_TARGET64)
        elseif( BORLAND )
            set( LibUSB_LIBRARY_PATH_SUFFIX lib/bcc )
        elseif( CMAKE_COMPILER_IS_GNUCC )
            set( LibUSB_LIBRARY_PATH_SUFFIX lib/gcc )
        endif( MSVC )
    endif( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )

    find_library ( LibUSB_LIBRARY
    NAMES
        libusb usb libusb-1.0
    PATHS
        $ENV{ProgramFiles}/LibUSB-Win32
        $ENV{LIBUSB_ROOT}
        ${LibUSB_DIR}
    PATH_SUFFIXES
        ${LibUSB_LIBRARY_PATH_SUFFIX}
    )

    mark_as_advanced ( LibUSB_LIBRARY )
    if( LibUSB_LIBRARY )
        set( LibUSB_LIBRARIES ${LibUSB_LIBRARY} )
        get_filename_component(LibUSB_LIBRARY_DIR ${LibUSB_LIBRARY} DIRECTORY  CACHE)
        get_filename_component(DLLNAME ${LibUSB_LIBRARY} NAME_WE)
        find_file ( LibUSB_LIBRARY_DLL
        NAMES
            libusb.dll libusb-1.0.dll *.dll
        PATHS
            ${LibUSB_DIR}
        PATH_SUFFIXES
            ${LibUSB_LIBRARY_PATH_SUFFIX}
        )
    endif( LibUSB_LIBRARY )

    if( LibUSB_INCLUDE_DIRS AND LibUSB_LIBRARIES )
        set( LibUSB_FOUND true )
    endif( LibUSB_INCLUDE_DIRS AND LibUSB_LIBRARIES )
endif( PKGCONFIG_LIBUSB_FOUND )

if( LibUSB_FOUND )
    set( CMAKE_REQUIRED_INCLUDES "${LibUSB_INCLUDE_DIRS}" )
    check_include_file ( usb.h LibUSB_FOUND )
#    message( STATUS "LibUSB: usb.h is usable: ${LibUSB_FOUND}" )
endif( LibUSB_FOUND )
if( LibUSB_FOUND )
    check_library_exists ( "${LibUSB_LIBRARIES}" usb_open "" LibUSB_FOUND )
#    message( STATUS "LibUSB: library is usable: ${LibUSB_FOUND}" )
endif( LibUSB_FOUND )

if( NOT LibUSB_FOUND )
    if( NOT LibUSB_FIND_QUIETLY )
      message( STATUS "LibUSB not found, try setting LibUSB_ROOT environment variable." )
    endif( NOT LibUSB_FIND_QUIETLY )
    if( LibUSB_FIND_REQUIRED )
      message( FATAL_ERROR "" )
    endif( LibUSB_FIND_REQUIRED )
endif( NOT LibUSB_FOUND )
#  message( STATUS "LibUSB: ${LibUSB_FOUND}" )
