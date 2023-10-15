###########################################################
#                  Find XIMEA SDK
#----------------------------------------------------------
# Applicable for SDK Versions > v4.15.05
#
# The following variables are optionally searched for defaults
#  XIMEA_SDK ROOT:            Base directory of Ximea SDK tree to use.
#
## 2: Variable
# The following are set after configuration is done:
#
#  XIMEA_SDK_FOUND		 -> Indicates if SDK has been found
#  XIMAE_SDK_LIBRARY	 -> list of all necessary libraries to link against. This list also contains
#  XIMEA_SDK_INCLUDE_DIR -> include directory of Ximea SDK
#  XIMEA_SDK_BINARY_DIR  -> include directory of Ximea SDK Dynamic Binary Library Files
#  XIMEA_SDK_VERSION     -> XIMEA Version ID Number
#
#----------------------------------------------------------

if(NOT XIMEA_APIDIR)
	set(XIMEA_APIDIR $ENV{XIMEA_API_ROOT})
endif(NOT XIMEA_APIDIR)

SET( XIMEA_INCLUDE_SEARCH_PATHS
    /opt/XIMEA/include
	/usr/local/include/ximea
    /usr/include/ximea
    /Library/Frameworks/m3api.framework/Headers
	${XIMEA_APIDIR}
)

SET( XIMEA_LIBRARY_SEARCH_PATH
    /opt/XIMEA/lib
    /usr/lib
    /usr/local/lib
    /Library/Frameworks/m3api.framework/Libraries
	${XIMEA_APIDIR}
)

SET( XIMEA_BINARY_SEARCH_PATH
    /opt/XIMEA/lib
    /usr/lib
    /usr/local/lib
    /Library/Frameworks/m3api.framework/Libraries
	${XIMEA_APIDIR}
)

if(WIN32)
	if(CMAKE_CL_64)
		set(XIMEA_LIB_NAME "xiapi64")
		set(XIMEA_BIN_NAME "xiapi64.dll")
	elseif(CMAKE_CL_64)
		set(XIMEA_LIB_NAME "xiapi32")
		set(XIMEA_BIN_NAME "xiapi32.dll")
	endif(CMAKE_CL_64)

	if(NOT XIMEA_SDK_VERSION)
		file(READ "${XIMEA_APIDIR}/Python/v3/ximea/__init__.py" XIMEA_VER_FILE)
		string(REGEX MATCH "__version__ = \'([0-9]+\.[0-9]+\.[0-9]+)" _ ${XIMEA_VER_FILE})
		set(XIMEA_SDK_VERSION ${CMAKE_MATCH_1})
	endif(NOT XIMEA_SDK_VERSION)

	# Please note: The Exact Version when XIMEA decided to change the library and binary
	# path definition is unknown, please feel free to adjust it accordingly.
	if(${XIMEA_SDK_VERSION} VERSION_LESS_EQUAL "4.16.00")
		if(CMAKE_CL_64)
			set(XIMEA_LIBRARY_SEARCH_PATH_SUFFIX "x64")
			set(XIMEA_BINARY_SEARCH_PATH_SUFFIX "x64")
		elseif(CMAKE_CL_64)
			set(XIMEA_LIBRARY_SEARCH_PATH_SUFFIX "x86")
			set(XIMEA_BINARY_SEARCH_PATH_SUFFIX "x86")
		endif(CMAKE_CL_64)
	elseif(${XIMEA_SDK_VERSION} VERSION_LESS_EQUAL "4.16.00")
			set(XIMEA_LIBRARY_SEARCH_PATH_SUFFIX "")
			set(XIMEA_BINARY_SEARCH_PATH_SUFFIX "")
	endif(${XIMEA_SDK_VERSION} VERSION_LESS_EQUAL "4.16.00")
else(WIN32)
	set(XIMEA_LIB_NAME "m3api")
	set(XIMEA_BIN_NAME "m3api")

	if(NOT XIMEA_SDK_VERSION)
		file(READ "/opt/XIMEA/version_LINUX_SP.txt" XIMEA_VER_FILE)
		string(REGEX MATCH "LINUX_SP_V([0-9]+\_[0-9]+\_[0-9]+)" _ ${XIMEA_VER_FILE})
		set(XIMEA_SDK_VERSION ${CMAKE_MATCH_1})
	endif(NOT XIMEA_SDK_VERSION)
	STRING(REGEX REPLACE "\_" "." XIMEA_SDK_VERSION ${XIMEA_SDK_VERSION})

	set(XIMEA_LIBRARY_SEARCH_PATH_SUFFIX "")
	set(XIMEA_BINARY_SEARCH_PATH_SUFFIX "")
endif(WIN32)


find_path(XIMEA_SDK_INCLUDE_DIR
    NAMES xiApi.h
    PATHS ${XIMEA_INCLUDE_SEARCH_PATHS}
)

find_library(XIMAE_SDK_LIBRARY
	NAMES ${XIMEA_LIB_NAME}
	PATHS ${XIMEA_LIBRARY_SEARCH_PATH}
	PATH_SUFFIXES ${XIMEA_BINARY_SEARCH_PATH_SUFFIX}
)

find_path(XIMEA_SDK_BINARY_DIR
	NAMES ${XIMEA_BIN_NAME}
	PATHS ${XIMEA_BINARY_SEARCH_PATH}
	PATH_SUFFIXES ${XIMEA_BINARY_SEARCH_PATH_SUFFIX}
)

message(STATUS "XIMEA_SDK_VERSION: " ${XIMEA_SDK_VERSION})
message(STATUS "XIMEA_SDK_INCLUDE_DIR: " ${XIMEA_SDK_INCLUDE_DIR})
message(STATUS "XIMEA_SDK_BINARY_DIR: " ${XIMEA_SDK_BINARY_DIR})
message(STATUS "XIMAE_SDK_LIBRARY: " ${XIMAE_SDK_LIBRARY})


if( XIMEA_SDK_INCLUDE_DIR AND XIMAE_SDK_LIBRARY )
    set( XIMEA_SDK_FOUND TRUE )
	message(STATUS "XIMEA_SDK_FOUND: " ${XIMEA_SDK_FOUND})
endif()
