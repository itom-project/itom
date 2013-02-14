# - Find FFTW
# Find the native FFTW includes and library
# This module defines
# FFTW_INCLUDE_DIR, where to find fftw3.h, etc.
# FFTW_LIBRARIES, the libraries needed to use FFTW.
# FFTW_FOUND, If false, do not try to use FFTW.
# also defined, but not for general use are
# FFTW_LIBRARY, where to find the FFTW library.

SET(FFTW_FOUND false)

find_path(FFTW_DIR fftw3.h PATHS /usr/local/include /usr/include /opt/local/lib DOC "Root directory of fftw")
FIND_PATH(FFTW_INCLUDE_DIR fftw3.h PATHS /usr/local/include /usr/include /opt/local/lib ${FFTW_DIR})

FIND_LIBRARY(FFTW_LIBRARY_F NAMES fftw3f-3 fftw3f libfftw3f-3 PATHS /usr/lib /usr/local/lib /opt/locala/lib ${FFTW_DIR})
FIND_LIBRARY(FFTW_LIBRARY_D NAMES fftw3-3 fftw3 libfftw3-3  PATHS /usr/lib /usr/local/lib /opt/locala/lib ${FFTW_DIR})
FIND_LIBRARY(FFTW_LIBRARY_L NAMES fftw3l-3 fftw3l libfftw3l-3 PATHS /usr/lib /usr/local/lib /opt/locala/lib ${FFTW_DIR})

IF(WIN32)
  FIND_FILE(FFTW_RUNTIME_LIBRARY_F NAMES libfftw3f-3.dll PATHS /usr/lib /usr/local/lib /opt/locala/lib ${FFTW_DIR})
  FIND_FILE(FFTW_RUNTIME_LIBRARY_D NAMES libfftw3-3.dll PATHS /usr/lib /usr/local/lib /opt/locala/lib ${FFTW_DIR})
  FIND_FILE(FFTW_RUNTIME_LIBRARY_L NAMES libfftw3l-3.dll PATHS /usr/lib /usr/local/lib /opt/locala/lib ${FFTW_DIR})
ENDIF(WIN32)


IF (FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
    SET(FFTW_LIBRARIES ${FFTW_LIBRARY_F} ${FFTW_LIBRARY_D} ${FFTW_LIBRARY_L})
    IF(WIN32)
      SET(FFTW_RUNTIME_LIBRARIES ${FFTW_RUNTIME_LIBRARY_F} ${FFTW_RUNTIME_LIBRARY_D} ${FFTW_RUNTIME_LIBRARY_L})
    ENDIF(WIN32)
    SET(FFTW_FOUND true)
ELSE (FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)
    SET(FFTW_FOUND false)
    SET(FFTW_LIBRARIES "")
ENDIF (FFTW_LIBRARY_D AND FFTW_INCLUDE_DIR)


IF (FFTW_FOUND)
   IF (NOT FFTW_FIND_QUIETLY)
      MESSAGE(STATUS "Found FFTW: ${FFTW_LIBRARIES}")
   ENDIF (NOT FFTW_FIND_QUIETLY)
ELSE (FFTW_FOUND)
   IF (FFTW_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find FFTW library")
   ENDIF (FFTW_FIND_REQUIRED)
ENDIF (FFTW_FOUND)

if(WIN32)
  mark_as_advanced(FFTW_LIBRARY_F FFTW_LIBRARY_D FFTW_LIBRARY_L FFTW_RUNTIME_LIBRARY_F FFTW_RUNTIME_LIBRARY_D FFTW_RUNTIME_LIBRARY_L)
ELSE(WIN32)
  mark_as_advanced(FFTW_LIBRARY_F FFTW_LIBRARY_D FFTW_LIBRARY_L)
ENDIF(WIN32)
