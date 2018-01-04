# - Try to find Xerces-C
# Once done this will define
#
#  XERCESC_FOUND - system has Xerces-C
#  XERCESC_INCLUDE - the Xerces-C include directory
#  XERCESC_LIBRARY - Link these to use Xerces-C
#  XERCESC_VERSION - Xerces-C found version
#  XERCESC_BINARY - The binary file of Xerces-C

IF ((CMAKE_MAJOR_VERSION GREATER 2) AND (CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION GREATER 1))
    MESSAGE(STATUS "policy")
    cmake_policy(SET CMP0053 OLD)
ENDIF ((CMAKE_MAJOR_VERSION GREATER 2) AND (CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION GREATER 1))
 
IF (XERCESC_INCLUDE AND XERCESC_LIBRARY)
  # in cache already
  SET(XERCESC_FIND_QUIETLY TRUE)
ENDIF (XERCESC_INCLUDE AND XERCESC_LIBRARY)

IF (NOT  ${XERCESC_WAS_STATIC} STREQUAL ${XERCESC_STATIC})
  UNSET(XERCESC_LIBRARY CACHE)
  UNSET(XERCESC_BINARY CACHE)
  UNSET(XERCESC_LIBRARY_DEBUG CACHE)
ENDIF (NOT  ${XERCESC_WAS_STATIC} STREQUAL ${XERCESC_STATIC})

SET(XERCESC_WAS_STATIC ${XERCESC_STATIC} CACHE INTERNAL "" )

IF (DEFINED MSVC_VERSION)
  # Library postfix/ prefix for different vs version
  #   1300 = VS  7.0
  #   1400 = VS  8.0
  #   1500 = VS  9.0
  #   1600 = VS 10.0
  IF (MSVC_VERSION EQUAL 1300)
    SET(XERCES_LIB_POSTFIX "_vc70")
    SET(XERCES_LIBPATH_VERS_POSTFIX "vc-7.1/")
  ELSEIF (MSVC_VERSION EQUAL 1400)
    SET(XERCES_LIB_POSTFIX "_vc80")
    SET(XERCES_LIBPATH_VERS_POSTFIX "vc-8.0/")
  ELSEIF (MSVC_VERSION EQUAL 1500)
    SET(XERCES_LIB_POSTFIX "_vc90")
    SET(XERCES_LIBPATH_VERS_POSTFIX "vc-9.0/")
  ELSEIF (MSVC_VERSION EQUAL 1600)
    SET(XERCES_LIB_POSTFIX "_vc100")
    SET(XERCES_LIBPATH_VERS_POSTFIX "vc-10.0/")
  ELSEIF (MSVC_VERSION EQUAL 1800)
    SET(XERCES_LIB_POSTFIX "_vc120")
    SET(XERCES_LIBPATH_VERS_POSTFIX "vc-12.0/")
  ELSEIF (MSVC)
    #for all newer versions than VS 2013, use the libraries for VS 2013
    SET(XERCES_LIB_POSTFIX "_vc120")
    SET(XERCES_LIBPATH_VERS_POSTFIX "vc-12.0/")
  ELSE (MSVC_VERSION EQUAL 1300)
    # since we don't knwo wether we are on windows or not, we just undefined and see what happens
    UNSET(XERCES_LIB_PATH_POSTFIX)
  ENDIF (MSVC_VERSION EQUAL 1300)

  # Wiora: Set 64 bit target dir (currently this is windows only. How does this work on linux/mac?)
  IF (BUILD_SHARED_LIBS)  
     IF (CMAKE_CL_64)
        SET (XERCES_LIBPATH_POSTFIX lib64/)
        SET (XERCES_BINPATH_POSTFIX bin64/)
      ELSE (CMAKE_CL_64)
        SET (XERCES_LIBPATH_POSTFIX lib/)
        SET (XERCES_BINPATH_POSTFIX bin/)
      ENDIF (CMAKE_CL_64)
      SET(XERCES_LIBPATH_POSTFIX ${XERCES_LIBPATH_POSTFIX}${XERCES_LIBPATH_VERS_POSTFIX})
  ELSE (BUILD_SHARED_LIBS)
      if(CMAKE_CL_64)
        SET (XERCES_LIBPATH_POSTFIX lib64/)
        SET (XERCES_BINPATH_POSTFIX bin64/)
      ELSE (CMAKE_CL_64)
        SET (XERCES_LIBPATH_POSTFIX lib/)
        SET (XERCES_BINPATH_POSTFIX bin/)
      ENDIF (CMAKE_CL_64)
      SET(XERCES_LIBPATH_POSTFIX ${XERCES_LIBPATH_POSTFIX}${XERCES_LIBPATH_VERS_POSTFIX})
  ENDIF (BUILD_SHARED_LIBS)

ELSE(DEFINED MSVC_VERSION)
  SET(XERCES_LIB_PATH_POSTFIX "")
  SET(XERCES_LIB_POSTFIX "")
ENDIF (DEFINED MSVC_VERSION)


SET (XERCESC_POSSIBLE_ROOT_DIRS
  "$ENV{XERCESC_INCLUDE_DIR}/.."
  "${XERCESC_INCLUDE_DIR}/.."
  "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3"
  "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3"
  "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3"
  "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0"
  "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0"
  "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0"
  /usr/local
  /usr
 "$ENV{PATH}"
  )

  FIND_PATH(XERCESC_ROOT_DIR 
  NAMES 
  include/xercesc/util/XercesVersion.hpp  
  PATHS ${XERCESC_POSSIBLE_ROOT_DIRS}
  )

FIND_PATH(XERCESC_INCLUDE NAMES xercesc/util/XercesVersion.hpp
  PATHS
  "$ENV{XERCESC_INCLUDE_DIR}"
  "${XERCESC_INCLUDE_DIR}"
  "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/include"
  "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/include"
  "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/include"
  "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0/include"
  "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0/include"
  "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0/include"  
  /usr/local/include
  /usr/include
  "${XERCESC_ROOT_DIR}/include"
)

IF (BUILD_SHARED_LIBS)

    # Use DYNAMIC version of Xerces library
    # Find release dynamic link libraries
    # BUG (Wiora): This works only on windows if dlls have .lib files asside. This is not the case and not necessary. No idea how to fix this.
    FIND_LIBRARY(XERCESC_LIBRARY NAMES xerces-c_3 xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c-3.1${XERCES_LIB_POSTFIX} libxerces-c-3.1.dylib libxerces-c.dylib
        PATHS
        $ENV{XERCESC_LIBRARY_DIR}
        "${XERCESC_LIBRARY_DIR}"
        "${XERCESC_INCLUDE_DIR}/../${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"  
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        "${XERCESC_ROOT_DIR}"
        "${XERCESC_ROOT_DIR}/${XERCES_LIBPATH_POSTFIX}"
        "${XERCESC_ROOT_DIR}/lib"
        DOC "Xerces library dynamic linking"
    )

    # Find debug dynamic link libraries
    FIND_LIBRARY(XERCESC_LIBRARY_DEBUG NAMES xerces-c_3D xerces-c_3_1D${XERCES_LIB_POSTFIX} xerces-c-3.1${XERCES_LIB_POSTFIX} libxerces-c-3.1.dylib libxerces-c.dylib
        PATHS
        $ENV{XERCESC_LIBRARY_DIR}
        "${XERCESC_LIBRARY_DIR}"
        "${XERCESC_INCLUDE_DIR}/../${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"  
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        "${XERCESC_ROOT_DIR}"
        "${XERCESC_ROOT_DIR}/${XERCES_LIBPATH_POSTFIX}"
        "${XERCESC_ROOT_DIR}/lib"
        PATH_SUFFIXES ${XERCES_LIBPATH_POSTFIX} ""
        DOC "Xerces library dynamic linking debug"
    )

    FIND_FILE(XERCESC_BINARY NAMES xerces-c_3_1 xerces-c_3_1.dll xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c_3_1${XERCES_LIB_POSTFIX}.dll
        PATHS
        $ENV{XERCESC_LIBRARY_DIR}
        "${XERCESC_LIBRARY_DIR}"
        "${XERCESC_INCLUDE_DIR}/../${XERCES_BINPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/${XERCES_BINPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/${XERCES_BINPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/${XERCES_BINPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"  
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        "${XERCESC_ROOT_DIR}"
        "${XERCESC_ROOT_DIR}/${XERCES_BINPATH_POSTFIX}"
        "${XERCESC_ROOT_DIR}/lib"
        PATH_SUFFIXES ${XERCES_BINPATH_POSTFIX} ""
        NO_DEFAULT_PATH
        DOC "Xerces binary"
    )



ELSE (BUILD_SHARED_LIBS)
    FIND_LIBRARY(XERCESC_LIBRARY NAMES xerces-c_static_3 xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c-3.1 xerces-c_3 xerces-c libxerces-c.a
        PATHS
        $ENV{XERCESC_LIBRARY_DIR}
        "${XERCESC_INCLUDE_DIR}/../${XERCES_LIBPATH_POSTFIX}"
        "${XERCESC_LIBRARY_DIR}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        "${XERCESC_ROOT_DIR}"
        "${XERCESC_ROOT_DIR}/${XERCES_LIBPATH_POSTFIX}"
        "${XERCESC_ROOT_DIR}/lib"
        PATH_SUFFIXES ${XERCES_LIBPATH_POSTFIX} ""
        DOC "Xerces library static linking"
    )
  
    FIND_LIBRARY(XERCESC_LIBRARY_DEBUG NAMES xerces-c_static_3D xerces-c_3_1D${XERCES_LIB_POSTFIX} xerces-c-3.1D xerces-c_3D libxerces-c.la 
        PATHS
        $ENV{XERCESC_LIBRARY_DIR}
        "${XERCESC_LIBRARY_DIR}"
        "${XERCESC_INCLUDE_DIR}/../${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"   
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        "${XERCESC_ROOT_DIR}"
        "${XERCESC_ROOT_DIR}/${XERCES_LIBPATH_POSTFIX}"
        "${XERCESC_ROOT_DIR}/lib"
        PATH_SUFFIXES ${XERCES_LIBPATH_POSTFIX} ""
        DOC "Xerces library static linking debug"
    )
    
    FIND_FILE(XERCESC_BINARY NAMES xerces-c_3_1 xerces-c_3_1.dll xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c_3_1${XERCES_LIB_POSTFIX}.dll
        PATHS
        $ENV{XERCESC_LIBRARY_DIR}
        "${XERCESC_LIBRARY_DIR}"
        "${XERCESC_INCLUDE_DIR}/../${XERCES_BINPATH_POSTFIX}"
        "${XERCESC_INCLUDE_DIR}/../${XERCES_BINPATH_POSTFIX}"
        "$ENV{ProgramFiles}/CodeSynthesis XSD 3.3/${XERCES_BINPATH_POSTFIX}"
        "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 3.3/${XERCES_BINPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 3.3/${XERCES_BINPATH_POSTFIX}"
           "$ENV{ProgramFiles}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
            "$ENV{ProgramFiles(x86)}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        "$ENV{ProgramW6432}/CodeSynthesis XSD 4.0/${XERCES_LIBPATH_POSTFIX}"
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        "${XERCESC_ROOT_DIR}"
        "${XERCESC_ROOT_DIR}/${XERCES_BINPATH_POSTFIX}"
        "${XERCESC_ROOT_DIR}/lib"
        NO_DEFAULT_PATH
        DOC "Xerces binary"
    )

  #ADD_DEFINITIONS( -DXERCES_STATIC_LIBRARY ) #REMOVED
ENDIF (BUILD_SHARED_LIBS)

IF (XERCESC_INCLUDE AND XERCESC_LIBRARY)
    SET(XERCESC_FOUND TRUE)
ELSE (XERCESC_INCLUDE AND XERCESC_LIBRARY)
    SET(XERCESC_FOUND FALSE)
ENDIF (XERCESC_INCLUDE AND XERCESC_LIBRARY)

IF(XERCESC_FOUND)
    FIND_PATH(XERCESC_XVERHPPPATH NAMES XercesVersion.hpp PATHS
     ${XERCESC_INCLUDE}
     PATH_SUFFIXES xercesc/util)

    IF ( ${XERCESC_XVERHPPPATH} STREQUAL XERCESC_XVERHPPPATH-NOTFOUND )
     SET(XERCES_VERSION "0")
    ELSE( ${XERCESC_XVERHPPPATH} STREQUAL XERCESC_XVERHPPPATH-NOTFOUND )
     FILE(READ ${XERCESC_XVERHPPPATH}/XercesVersion.hpp XVERHPP)

     STRING(REGEX MATCHALL "\n *#define XERCES_VERSION_MAJOR +[0-9]+" XVERMAJ
       ${XVERHPP})
     STRING(REGEX MATCH "\n *#define XERCES_VERSION_MINOR +[0-9]+" XVERMIN
       ${XVERHPP})
     STRING(REGEX MATCH "\n *#define XERCES_VERSION_REVISION +[0-9]+" XVERREV
       ${XVERHPP})

     STRING(REGEX REPLACE "\n *#define XERCES_VERSION_MAJOR +" ""
       XVERMAJ ${XVERMAJ})
     STRING(REGEX REPLACE "\n *#define XERCES_VERSION_MINOR +" ""
       XVERMIN ${XVERMIN})
     STRING(REGEX REPLACE "\n *#define XERCES_VERSION_REVISION +" ""
       XVERREV ${XVERREV})

     SET(XERCESC_VERSION ${XVERMAJ}.${XVERMIN}.${XVERREV})

    ENDIF ( ${XERCESC_XVERHPPPATH} STREQUAL XERCESC_XVERHPPPATH-NOTFOUND )

    IF(NOT XERCESC_FIND_QUIETLY)
     MESSAGE(STATUS "Found Xerces-C: ${XERCESC_LIBRARY}")
     MESSAGE(STATUS "              : ${XERCESC_INCLUDE}")
     MESSAGE(STATUS "       Version: ${XERCESC_VERSION}")
    ENDIF(NOT XERCESC_FIND_QUIETLY)
    
ELSEIF(XERCESC_FIND_REQUIRED)
   MESSAGE(FATAL_ERROR "Could not find Xerces-C !")
ENDIF(XERCESC_FOUND)

