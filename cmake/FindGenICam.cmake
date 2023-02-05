# - Try to find Genicam
# Once done this will define
#  GENICAM_FOUND - System has genicam
#  GENICAM_INCLUDE_DIRS - The genicam include directories
#  GENICAM_LIBRARIES - The libraries needed to use genicam
#C:\genicam\library\cpp\include
#C:\genicam\library\cpp\lib\win64_x64

if(NOT GENICAM_ROOT)
	find_path(GENICAM_INCLUDE_DIR GenICam.h PATHS $ENV{GENICAM_ROOT} ${GENICAM_ROOT} PATH_SUFFIXES library/cpp/include DOC "root directory of genicam, containing sub-directories library and include")
	set(GENICAM_ROOT $ENV{GENICAM_ROOT})
endif(NOT GENICAM_ROOT)

if(WIN32)
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set( GENICAM_LIBRARY "${GENICAM_ROOT}/library/cpp/lib/win64_x64" )
        set(GENICAM_BINARY_DIR "${GENICAM_ROOT}/bin/Win64_x64" CACHE PATH "path to the DLL files of Genicam that are copied as post built process to the itom lib folder" FORCE)
    else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set( GENICAM_LIBRARY "${GENICAM_ROOT}/library/cpp/lib/win32_i86" )
        set(GENICAM_BINARY_DIR "${GENICAM_ROOT}/bin/Win32_i86" CACHE PATH "path to the DLL files of Genicam that are copied as post built process to the itom lib folder" FORCE)
    endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
else(WIN32)
    if(UNIX)
        if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
            set( GENICAM_LIBRARY "/opt/genicam/bin/Linux64_x64" )
            set(GENICAM_BINARY_DIR "/opt/genicam/bin/Linux64_x64" CACHE PATH "path to the DLL files of Genicam that are copied as post built process to the itom lib folder" FORCE)
        else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
            set( GENICAM_LIBRARY "/opt/genicam/bin/Linux86_x86" )
            set(GENICAM_BINARY_DIR "/opt/genicam/bin/Linux86_x86" CACHE PATH "path to the DLL files of Genicam that are copied as post built process to the itom lib folder" FORCE)
        endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    endif(UNIX)
endif(WIN32)

find_library(GENICAM_GENAPI_LIBRARY 
	NAMES 
	GenApi_gcc40_v2_3 GenApi_MD_VC120_v3_0 GenApi_MD_VC120_v3_1 GenApi_MD_VC141_v3_2 GenApi_MD_VC141_v3_3 GenApi_MD_VC141_v3_4 GenApi_MD_VC100_v2_3
	PATHS
	${GENICAM_LIBRARY}
)

find_library(GENICAM_GCBASE_LIBRARY 
	NAMES 
	GCBase_gcc40_v2_3 GCBase_MD_VC120_v3_0 GCBase_MD_VC120_v3_1 GCBase_MD_VC141_v3_2 GCBase_MD_VC141_v3_3 GCBase_MD_VC141_v3_4 GCBase_MD_VC100_v2_3
	PATHS
	${GENICAM_LIBRARY}
)

find_library(GENICAM_LOG4CPP_LIBRARY 
	NAMES 
	log4cpp_gcc40_v2_3 log4cpp_MD_VC120_v3_0 log4cpp_MD_VC120_v3_1 log4cpp_MD_VC141_v3_2 log4cpp_MD_VC141_v3_3 log4cpp_MD_VC141_v3_4 log4cpp_MD_VC100_v2_3
	PATHS
	${GENICAM_LIBRARY}
)

find_library(GENICAM_LOG_GCC_LIBRARY 
	NAMES 
	Log_gcc40_v2_3 Log_MD_VC120_v3_0 Log_MD_VC120_v3_1 Log_MD_VC141_v3_2 Log_MD_VC141_v3_3 Log_MD_VC141_v3_4 Log_MD_VC100_v2_3
	PATHS
	${GENICAM_LIBRARY}
)

set(GENICAM_LIBRARIES ${GENICAM_GENAPI_LIBRARY} ${GENICAM_GCBASE_LIBRARY} ${GENICAM_LOG4CPP_LIBRARY} ${GENICAM_LOG_GCC_LIBRARY})
set(GENICAM_INCLUDE_DIRS ${GENICAM_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(GenICam DEFAULT_MSG
  GENICAM_INCLUDE_DIR
  GENICAM_LIBRARY
  GENICAM_BINARY_DIR)

mark_as_advanced(GENICAM_INCLUDE_DIR GENICAM_LIBRARIES GENICAM_BINARY_DIR)