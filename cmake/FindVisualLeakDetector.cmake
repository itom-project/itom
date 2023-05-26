# - Try to find the Visual Leak Detector includes and library
# which defines
#
# VISUALLEAKDETECTOR_FOUND - system has Visual Leak Detector
# VISUALLEAKDETECTOR_INCLUDE_DIR - where to find qextscintilla.h
# VISUALLEAKDETECTOR_LIBRARIES - the libraries to link against to use QScintilla
# VISUALLEAKDETECTOR_LIBRARY - where to find the QScintilla library (not for general use)
# VISUALLEAKDETECTOR_RUNTIME_LIBRARY - the DLL-file
#
# If the Visual Leak Detector library could be found and if the option VISUALLEAKDETECTOR_ENABLED
# is set to true, the preprocessor VISUAL_LEAK_DETECTOR_CMAKE will be set.


set(VISUALLEAKDETECTOR_FOUND "NO")
unset(VISUALLEAKDETECTOR_LIBRARY CACHE)
unset(VISUALLEAKDETECTOR_INCLUDE_DIR CACHE)

option(VISUALLEAKDETECTOR_ENABLED "Indicate whether the visual leak detector should be switched on in debug" OFF)

find_path(VISUALLEAKDETECTOR_DIR "vld.ini" DOC "Root directory of Visual Leak Detector")

set(GLOBAL VISUALLEAKDETECTOR_LIBRARIES "")

if(EXISTS "${VISUALLEAKDETECTOR_DIR}")

    find_path(VISUALLEAKDETECTOR_INCLUDE_DIR vld.h PATHS "${VISUALLEAKDETECTOR_DIR}" PATH_SUFFIXES "include")

    if(MSVC)
        if(CMAKE_CL_64)
            find_library(VISUALLEAKDETECTOR_LIBRARY NAMES "vld" PATHS "${VISUALLEAKDETECTOR_INCLUDE_DIR}/../lib/Win64")
            #find_file(VISUALLEAKDETECTOR_RUNTIME_LIBRARY NAMES "vld_x64.dll" "dbghelp.dll" PATHS "${VISUALLEAKDETECTOR_INCLUDE_DIR}/../bin/Win64")
        else(CMAKE_CL_64)
            find_library(VISUALLEAKDETECTOR_LIBRARY NAMES "vld" PATHS "${VISUALLEAKDETECTOR_INCLUDE_DIR}/../lib/Win32")
            #find_file(VISUALLEAKDETECTOR_RUNTIME_LIBRARY NAMES "vld_x86.dll" "dbghelp.dll" PATHS "${VISUALLEAKDETECTOR_INCLUDE_DIR}/../bin/Win32")
        endif(CMAKE_CL_64)
    endif(MSVC)
endif(EXISTS "${VISUALLEAKDETECTOR_DIR}")



if(VISUALLEAKDETECTOR_LIBRARY)
    if(VISUALLEAKDETECTOR_ENABLED)
        set(VISUALLEAKDETECTOR_LIBRARIES debug ${VISUALLEAKDETECTOR_LIBRARY}) # optimized "")
        add_definitions(-DVISUAL_LEAK_DETECTOR_CMAKE)
    endif()

    set(VISUALLEAKDETECTOR_FOUND true CACHE BOOL "" FORCE)
endif(VISUALLEAKDETECTOR_LIBRARY)


if(VISUALLEAKDETECTOR_FOUND)
  if(NOT VISUALLEAKDETECTOR_FIND_QUIETLY)
    message(STATUS "Found VisualLeakDetector: ${VISUALLEAKDETECTOR_LIBRARIES}")
  endif(NOT VISUALLEAKDETECTOR_FIND_QUIETLY)
else (VISUALLEAKDETECTOR_FOUND)
  if(VISUALLEAKDETECTOR_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Visual Leak Detector library")
  endif(VISUALLEAKDETECTOR_FIND_REQUIRED)
endif(VISUALLEAKDETECTOR_FOUND)

mark_as_advanced(VISUALLEAKDETECTOR_INCLUDE_DIR VISUALLEAKDETECTOR_LIBRARY VISUALLEAKDETECTOR_FOUND)
