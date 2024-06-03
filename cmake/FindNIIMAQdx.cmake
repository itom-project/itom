# #########################################################################
# #########################################################################
# CMake FIND file for the proprietary NIIMAQdx Windows library (NIIMAQdx).
#
# Try to find NIIMAQdx
# Once done this will define
# NIIMAQDX_FOUND - System has NIIMAQDX
# NIIMAQDX_LIBRARY - The NIIMAQDX library
# NIIMAQDX_INCLUDE_DIR - The NIIMAQDX include file
# #########################################################################
# #########################################################################
# #########################################################################
# Useful variables

if(WIN32)
    list(APPEND NIIMAQDX_DIR
	    "C:/Program Files (x86)/National Instruments/Shared/ExternalCompilerSupport/C"
		$ENV{NIIMAQDX_ROOT}/Shared/ExternalCompilerSupport/C
		)

    if( CMAKE_SIZEOF_VOID_P EQUAL 4 )
      set(SUFFIXES "lib32/msvc")
    else()
      set(SUFFIXES "lib64/msvc")
    endif()
else()
    set(NIIMAQDX_DIR "/usr/include" "/usr/lib/x86_64-linux-gnu")
    set(SUFFIXES "")
endif()

# Find installed library using CMake functions
find_library(NIIMAQDX_LIBRARY
	NAMES "NIIMAQdx" "niimaqdx"
	PATHS ${NIIMAQDX_DIR}
	PATH_SUFFIXES ${SUFFIXES})

find_path(NIIMAQDX_INCLUDE_DIR
	NAMES "NIIMAQdx.h"
	PATHS ${NIIMAQDX_DIR}
	PATH_SUFFIXES "include")

message(STATUS "NIIMAQDX_DIR: " ${NIIMAQDX_DIR})
message(STATUS "NIIMAQDX_LIBRARY: " ${NIIMAQDX_LIBRARY})
message(STATUS "NIIMAQDX_INCLUDE_DIR: " ${NIIMAQDX_INCLUDE_DIR})

# Handle the QUIETLY and REQUIRED arguments and set NIIMAQDX_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NIIMAQDX DEFAULT_MSG NIIMAQDX_LIBRARY NIIMAQDX_INCLUDE_DIR)
# #########################################################################

# #########################################################################
mark_as_advanced(NIIMAQDX_LIBRARY NIIMAQDX_INCLUDE_DIR)
# #########################################################################
