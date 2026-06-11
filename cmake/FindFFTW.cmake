# - Find FFTW
# Find the native FFTW includes and library
# This module defines
# FFTW_INCLUDE_DIR, where to find fftw3.h, etc.
# FFTW_LIBRARIES, the libraries needed to use FFTW.
# FFTW_FOUND, If false, do not try to use FFTW.
# also defined, but not for general use are
# FFTW_LIBRARY, where to find the FFTW library.

set(FFTW_FOUND false)

if(NOT EXISTS ${FFTW_DIR})
    if(EXISTS $ENV{FFTW_ROOT})
        set(FFTW_DIR $ENV{FFTW_ROOT} CACHE PATH "Root directory of fftw3, containing sub-directories library and include")
    else(EXISTS $ENV{FFTW_ROOT})
        set(FFTW_DIR "FFTW_DIR-NOTFOUND" CACHE PATH "Root directory of fftw3, containing sub-directories library and include")
    endif(EXISTS $ENV{FFTW_ROOT})
endif(NOT EXISTS ${FFTW_DIR})

find_path(FFTW_DIR fftw3.h PATHS $ENV{FFTW_ROOT} /usr/local/include /usr/include /opt/local/lib DOC "Root directory of fftw")
find_path(FFTW_INCLUDE_DIR fftw3.h PATHS $ENV{FFTW_ROOT} /usr/local/include /usr/include /opt/local/lib ${FFTW_DIR})

find_library(FFTW_LIBRARY_F NAMES fftw3f-3 fftw3f libfftw3f-3 PATHS /usr/lib /usr/local/lib /opt/local/lib ${FFTW_DIR})
find_library(FFTW_LIBRARY_D NAMES fftw3-3 fftw3 libfftw3-3  PATHS /usr/lib /usr/local/lib /opt/local/lib ${FFTW_DIR})
find_library(FFTW_LIBRARY_L NAMES fftw3l-3 fftw3l libfftw3l-3 PATHS /usr/lib /usr/local/lib /opt/local/lib ${FFTW_DIR})

if(WIN32)
  find_file(FFTW_RUNTIME_LIBRARY_F NAMES libfftw3f-3.dll PATHS /usr/lib /usr/local/lib /opt/local/lib ${FFTW_DIR})
  find_file(FFTW_RUNTIME_LIBRARY_D NAMES libfftw3-3.dll PATHS /usr/lib /usr/local/lib /opt/local/lib ${FFTW_DIR})
  find_file(FFTW_RUNTIME_LIBRARY_L NAMES libfftw3l-3.dll PATHS /usr/lib /usr/local/lib /opt/local/lib ${FFTW_DIR})
endif(WIN32)

if(WIN32)
  if(FFTW_RUNTIME_LIBRARY_D AND FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
    set(FFTW_FOUND TRUE)
    set(FFTW_RUNTIME_LIBRARIES ${FFTW_RUNTIME_LIBRARY_F} ${FFTW_RUNTIME_LIBRARY_D} ${FFTW_RUNTIME_LIBRARY_L})
    set(FFTW_LIBRARIES ${FFTW_LIBRARY_F} ${FFTW_LIBRARY_D} ${FFTW_LIBRARY_L})
  else(FFTW_RUNTIME_LIBRARY_D AND FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
    set(FFTW_FOUND false)
    set(FFTW_LIBRARIES "")
  endif(FFTW_RUNTIME_LIBRARY_D AND FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
else(WIN32)
  if(FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
    set(FFTW_FOUND true)
    set(FFTW_LIBRARIES ${FFTW_LIBRARY_F} ${FFTW_LIBRARY_D} ${FFTW_LIBRARY_L})
  else(FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
    set(FFTW_FOUND false)
    set(FFTW_LIBRARIES "")
  endif(FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
endif(WIN32)

if(FFTW_FOUND)
   if(NOT FFTW_FIND_QUIETLY)
      message(STATUS "Found FFTW: ${FFTW_LIBRARIES}")
   endif(NOT FFTW_FIND_QUIETLY)
else (FFTW_FOUND)
   if(FFTW_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find FFTW library")
   endif(FFTW_FIND_REQUIRED)
endif(FFTW_FOUND)

if(WIN32)
  mark_as_advanced(FFTW_LIBRARY_F FFTW_LIBRARY_D FFTW_LIBRARY_L FFTW_RUNTIME_LIBRARY_F FFTW_RUNTIME_LIBRARY_D FFTW_RUNTIME_LIBRARY_L)
else(WIN32)
  mark_as_advanced(FFTW_LIBRARY_F FFTW_LIBRARY_D FFTW_LIBRARY_L)
endif(WIN32)
