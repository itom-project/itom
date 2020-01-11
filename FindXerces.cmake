# - Try to find Xerces-C
# Once done this will define
#
#  XERCESC_FOUND - system has Xerces-C
#  XERCESC_INCLUDE - the Xerces-C include directory
#  XERCESC_LIBRARY - Link these to use Xerces-C
#  XERCESC_VERSION - Xerces-C found version
#  XERCESC_BINARY - The binary file of Xerces-C

if((CMAKE_MAJOR_VERSION GREATER 2) AND (CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION GREATER 1))
    message(STATUS "policy")
    cmake_policy(SET CMP0053 OLD)
endif((CMAKE_MAJOR_VERSION GREATER 2) AND (CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION GREATER 1))
 
if(XERCESC_INCLUDE AND XERCESC_LIBRARY)
  # in cache already
  set(XERCESC_FIND_QUIETLY TRUE)
endif(XERCESC_INCLUDE AND XERCESC_LIBRARY)

if(NOT  ${XERCESC_WAS_STATIC} STREQUAL ${XERCESC_STATIC})
  unset(XERCESC_LIBRARY CACHE)
  unset(XERCESC_BINARY CACHE)
  unset(XERCESC_LIBRARY_DEBUG CACHE)
endif(NOT  ${XERCESC_WAS_STATIC} STREQUAL ${XERCESC_STATIC})

set(XERCESC_WAS_STATIC ${XERCESC_STATIC} CACHE INTERNAL "" )

if(DEFINED MSVC_VERSION)
  # Library postfix/ prefix for different vs version
  #   1300 = VS  7.0
  #   1400 = VS  8.0
  #   1500 = VS  9.0
  #   1600 = VS 10.0
  if(MSVC_VERSION EQUAL 1300)
    set(XERCES_LIB_POSTFIX "_vc70")
    set(XERCES_LIBPATH_VERS_POSTFIX "vc-7.1/")
  elseif(MSVC_VERSION EQUAL 1400)
    set(XERCES_LIB_POSTFIX "_vc80")
    set(XERCES_LIBPATH_VERS_POSTFIX "vc-8.0/")
  elseif(MSVC_VERSION EQUAL 1500)
    set(XERCES_LIB_POSTFIX "_vc90")
    set(XERCES_LIBPATH_VERS_POSTFIX "vc-9.0/")
  elseif(MSVC_VERSION EQUAL 1600)
    set(XERCES_LIB_POSTFIX "_vc100")
    set(XERCES_LIBPATH_VERS_POSTFIX "vc-10.0/")
  elseif(MSVC_VERSION EQUAL 1800)
    set(XERCES_LIB_POSTFIX "_vc120")
    set(XERCES_LIBPATH_VERS_POSTFIX "vc-12.0/")
  elseif(MSVC)
    #for all newer versions than VS 2013, use the libraries for VS 2013
    set(XERCES_LIB_POSTFIX "_vc120")
    set(XERCES_LIBPATH_VERS_POSTFIX "vc-12.0/")
  else (MSVC_VERSION EQUAL 1300)
    # since we don't knwo wether we are on windows or not, we just undefined and see what happens
    unset(XERCES_LIB_PATH_POSTFIX)
  endif(MSVC_VERSION EQUAL 1300)

  # Wiora: Set 64 bit target dir (currently this is windows only. How does this work on linux/mac?)
  if(BUILD_SHARED_LIBS)  
     if(CMAKE_CL_64)
        set(XERCES_LIBPATH_POSTFIX lib64/)
        set(XERCES_BINPATH_POSTFIX bin64/)
      else (CMAKE_CL_64)
        set(XERCES_LIBPATH_POSTFIX lib/)
        set(XERCES_BINPATH_POSTFIX bin/)
      endif(CMAKE_CL_64)
      set(XERCES_LIBPATH_POSTFIX ${XERCES_LIBPATH_POSTFIX}${XERCES_LIBPATH_VERS_POSTFIX})
  else (BUILD_SHARED_LIBS)
      if(CMAKE_CL_64)
        set(XERCES_LIBPATH_POSTFIX lib64/)
        set(XERCES_BINPATH_POSTFIX bin64/)
      else (CMAKE_CL_64)
        set(XERCES_LIBPATH_POSTFIX lib/)
        set(XERCES_BINPATH_POSTFIX bin/)
      endif(CMAKE_CL_64)
      set(XERCES_LIBPATH_POSTFIX ${XERCES_LIBPATH_POSTFIX}${XERCES_LIBPATH_VERS_POSTFIX})
  endif(BUILD_SHARED_LIBS)

else(DEFINED MSVC_VERSION)
  set(XERCES_LIB_PATH_POSTFIX "")
  set(XERCES_LIB_POSTFIX "")
endif(DEFINED MSVC_VERSION)


set(XERCESC_POSSIBLE_ROOT_DIRS
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

  find_path(XERCESC_ROOT_DIR 
  NAMES 
  include/xercesc/util/XercesVersion.hpp  
  PATHS ${XERCESC_POSSIBLE_ROOT_DIRS}
  )

find_path(XERCESC_INCLUDE NAMES xercesc/util/XercesVersion.hpp
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

if(BUILD_SHARED_LIBS)

    # Use DYNAMIC version of Xerces library
    # Find release dynamic link libraries
    # BUG (Wiora): This works only on windows if dlls have .lib files asside. This is not the case and not necessary. No idea how to fix this.
    find_library(XERCESC_LIBRARY NAMES xerces-c_3 xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c-3.1${XERCES_LIB_POSTFIX} libxerces-c-3.1.dylib libxerces-c.dylib
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
    find_library(XERCESC_LIBRARY_DEBUG NAMES xerces-c_3D xerces-c_3_1D${XERCES_LIB_POSTFIX} xerces-c-3.1${XERCES_LIB_POSTFIX} libxerces-c-3.1.dylib libxerces-c.dylib
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

    find_file(XERCESC_BINARY NAMES xerces-c_3_1 xerces-c_3_1.dll xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c_3_1${XERCES_LIB_POSTFIX}.dll
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



else (BUILD_SHARED_LIBS)
    find_library(XERCESC_LIBRARY NAMES xerces-c_static_3 xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c-3.1 xerces-c_3 xerces-c libxerces-c.a
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
  
    find_library(XERCESC_LIBRARY_DEBUG NAMES xerces-c_static_3D xerces-c_3_1D${XERCES_LIB_POSTFIX} xerces-c-3.1D xerces-c_3D libxerces-c.la 
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
    
    find_file(XERCESC_BINARY NAMES xerces-c_3_1 xerces-c_3_1.dll xerces-c_3_1${XERCES_LIB_POSTFIX} xerces-c_3_1${XERCES_LIB_POSTFIX}.dll
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
endif(BUILD_SHARED_LIBS)

if(XERCESC_INCLUDE AND XERCESC_LIBRARY)
    set(XERCESC_FOUND TRUE)
else (XERCESC_INCLUDE AND XERCESC_LIBRARY)
    set(XERCESC_FOUND FALSE)
endif(XERCESC_INCLUDE AND XERCESC_LIBRARY)

if(XERCESC_FOUND)
    find_path(XERCESC_XVERHPPPATH NAMES XercesVersion.hpp PATHS
     ${XERCESC_INCLUDE}
     PATH_SUFFIXES xercesc/util)

    if( ${XERCESC_XVERHPPPATH} STREQUAL XERCESC_XVERHPPPATH-NOTFOUND )
     set(XERCES_VERSION "0")
    else( ${XERCESC_XVERHPPPATH} STREQUAL XERCESC_XVERHPPPATH-NOTFOUND )
     file(READ ${XERCESC_XVERHPPPATH}/XercesVersion.hpp XVERHPP)

     string(REGEX MATCHALL "\n *#define XERCES_VERSION_MAJOR +[0-9]+" XVERMAJ
       ${XVERHPP})
     string(REGEX MATCH "\n *#define XERCES_VERSION_MINOR +[0-9]+" XVERMIN
       ${XVERHPP})
     string(REGEX MATCH "\n *#define XERCES_VERSION_REVISION +[0-9]+" XVERREV
       ${XVERHPP})

     string(REGEX REPLACE "\n *#define XERCES_VERSION_MAJOR +" ""
       XVERMAJ ${XVERMAJ})
     string(REGEX REPLACE "\n *#define XERCES_VERSION_MINOR +" ""
       XVERMIN ${XVERMIN})
     string(REGEX REPLACE "\n *#define XERCES_VERSION_REVISION +" ""
       XVERREV ${XVERREV})

     set(XERCESC_VERSION ${XVERMAJ}.${XVERMIN}.${XVERREV})

    endif( ${XERCESC_XVERHPPPATH} STREQUAL XERCESC_XVERHPPPATH-NOTFOUND )

    if(NOT XERCESC_FIND_QUIETLY)
     message(STATUS "Found Xerces-C: ${XERCESC_LIBRARY}")
     message(STATUS "              : ${XERCESC_INCLUDE}")
     message(STATUS "       Version: ${XERCESC_VERSION}")
    endif(NOT XERCESC_FIND_QUIETLY)
    
elseif(XERCESC_FIND_REQUIRED)
   message(FATAL_ERROR "Could not find Xerces-C !")
endif(XERCESC_FOUND)

