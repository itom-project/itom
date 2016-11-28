# - Try to find the QScintilla2 includes and library
# which defines
#
# QSCINTILLA_FOUND - system has QScintilla2
# QSCINTILLA_INCLUDE_DIR - where to find qextscintilla.h
# QSCINTILLA_LIBRARIES - the libraries to link against to use QScintilla
# QSCINTILLA_LIBRARY - where to find the QScintilla library (not for general use)

# copyright (c) 2007 Thomas Moenicke thomas.moenicke@kdemail.net
#
# Redistribution and use is allowed according to the terms of the BSD license.

SET(QSCINTILLA_FOUND "NO")
UNSET(QSCINTILLA_LIBRARY CACHE)
UNSET(QSCINTILLA_NAMES CACHE)

IF(QT5_FOUND)
	FIND_PATH(QSCINTILLA_INCLUDE_DIR qsciglobal.h PATHS ${Qt5Core_INCLUDE_DIRS} PATH_SUFFIXES Qsci)

	find_library(QSCINTILLA_LIBRARY_DEBUG NAMES "qscintilla2d" "libqscintilla2d" "libqscintilla2-8d" "libqt5scintilla2d" "libqscintilla2d.so" "libqscintilla2-8d.so" "libqt5scintilla2d.so" "libqscintilla2d-qt5" PATHS ${Qt5Core_INCLUDE_DIRS} "/usr/lib" PATH_SUFFIXES "../lib" "/usr/lib64" NO_DEFAULT_PATH)
	find_library(QSCINTILLA_LIBRARY_RELEASE NAMES "qscintilla2" "libqscintilla2" "libqscintilla2-8" "libqt5scintilla2" "libqscintilla2.so" "libqscintilla2-8.so" "libqt5scintilla2.so" "libqscintilla2-qt5"  PATHS ${Qt5Core_INCLUDE_DIRS} "/usr/lib" PATH_SUFFIXES "../lib" "/usr/lib64" NO_DEFAULT_PATH)
	
	#Remove the cache value
	set(QSCINTILLA_LIBRARY "" CACHE STRING "" FORCE)
	
	#both debug/release
	if(QSCINTILLA_LIBRARY_DEBUG AND QSCINTILLA_LIBRARY_RELEASE)
			set(QSCINTILLA_LIBRARY debug ${QSCINTILLA_LIBRARY_DEBUG} optimized ${QSCINTILLA_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
	#only debug
	elseif(QSCINTILLA_LIBRARY_DEBUG)
			set(QSCINTILLA_LIBRARY ${QSCINTILLA_LIBRARY_DEBUG}  CACHE STRING "" FORCE)
	#only release
	elseif(QSCINTILLA_LIBRARY_RELEASE)
			set(QSCINTILLA_LIBRARY ${QSCINTILLA_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
	#no library found
	endif()
	
	IF (QSCINTILLA_LIBRARY)
        SET(QSCINTILLA_LIBRARIES ${QSCINTILLA_LIBRARY})
        SET(QSCINTILLA_FOUND "YES")

        IF (CYGWIN)
            IF(BUILD_SHARED_LIBS)
            # No need to define QSCINTILLA_USE_DLL here, because it's default for Cygwin.
            ELSE(BUILD_SHARED_LIBS)
            SET (QSCINTILLA_DEFINITIONS -DQSCINTILLA_STATIC)
            ENDIF(BUILD_SHARED_LIBS)
        ENDIF (CYGWIN)
    ENDIF (QSCINTILLA_LIBRARY)
ELSE(QT5_FOUND)

    FIND_PATH(QSCINTILLA_INCLUDE_DIR qsciglobal.h
    "${QT_INCLUDE_DIR}/Qsci"
    )
	
	find_library(QSCINTILLA_LIBRARY_DEBUG NAMES "qscintilla2d" "libqscintilla2d" "libqscintilla2-8d"  PATHS ${QT_LIBRARY_DIR} "/usr/lib" NO_DEFAULT_PATH)
	find_library(QSCINTILLA_LIBRARY_RELEASE NAMES "qscintilla2" "libqscintilla2" "libqscintilla2-8" PATHS ${QT_LIBRARY_DIR} "/usr/lib" NO_DEFAULT_PATH)
	
	#Remove the cache value
	set(QSCINTILLA_LIBRARY "" CACHE STRING "" FORCE)
	
	#both debug/release
	if(QSCINTILLA_LIBRARY_DEBUG AND QSCINTILLA_LIBRARY_RELEASE)
			set(QSCINTILLA_LIBRARY debug ${QSCINTILLA_LIBRARY_DEBUG} optimized ${QSCINTILLA_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
	#only debug
	elseif(QSCINTILLA_LIBRARY_DEBUG)
			set(QSCINTILLA_LIBRARY ${QSCINTILLA_LIBRARY_DEBUG}  CACHE STRING "" FORCE)
	#only release
	elseif(QSCINTILLA_LIBRARY_RELEASE)
			set(QSCINTILLA_LIBRARY ${QSCINTILLA_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
	#no library found
	endif()

    IF (QSCINTILLA_LIBRARY)
        SET(QSCINTILLA_LIBRARIES ${QSCINTILLA_LIBRARY})
        SET(QSCINTILLA_FOUND "YES")

        IF (CYGWIN)
            IF(BUILD_SHARED_LIBS)
            # No need to define QSCINTILLA_USE_DLL here, because it's default for Cygwin.
            ELSE(BUILD_SHARED_LIBS)
            SET (QSCINTILLA_DEFINITIONS -DQSCINTILLA_STATIC)
            ENDIF(BUILD_SHARED_LIBS)
        ENDIF (CYGWIN)
    ENDIF (QSCINTILLA_LIBRARY)
ENDIF(QT5_FOUND)

IF (QSCINTILLA_FOUND)
  IF (NOT QSCINTILLA_FIND_QUIETLY)
    MESSAGE(STATUS "Found QScintilla2: ${QSCINTILLA_LIBRARY}")
  ENDIF (NOT QSCINTILLA_FIND_QUIETLY)
ELSE (QSCINTILLA_FOUND)
  IF (QSCINTILLA_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find QScintilla library")
  ENDIF (QSCINTILLA_FIND_REQUIRED)
ENDIF (QSCINTILLA_FOUND)

MARK_AS_ADVANCED(QSCINTILLA_INCLUDE_DIR QSCINTILLA_LIBRARY) 
